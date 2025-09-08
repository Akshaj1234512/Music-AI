#!/usr/bin/env python3
"""
Fine-tune Fretting Transformer on GuitarSet

This script fine-tunes a pre-trained Fretting Transformer model on GuitarSet
using the prepared data splits. Implements two-stage fine-tuning:
1. Frozen encoder + fine-tune decoder only
2. Fine-tune entire model with lower learning rate
"""

import sys
import os
import pickle
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
import time
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.guitarset_loader import GuitarSetExcerpt
from data.guitarset_dataset import create_guitarset_dataloaders
from data.unified_tokenizer import UnifiedFrettingTokenizer
from model.unified_fretting_t5 import create_model_from_tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune Fretting Transformer on GuitarSet')
    
    # Data paths
    parser.add_argument('--splits_file', type=str, 
                       default='/data/andreaguz/guitarset_data/guitarset_splits.pkl',
                       help='Path to saved GuitarSet splits')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/andreaguz/fretting_experiments/guitarset_finetune',
                       help='Output directory for checkpoints and logs')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--num_epochs_stage1', type=int, default=5,
                       help='Epochs for stage 1 (frozen encoder)')
    parser.add_argument('--num_epochs_stage2', type=int, default=10,
                       help='Epochs for stage 2 (full model)')
    parser.add_argument('--learning_rate_stage1', type=float, default=1e-4,
                       help='Learning rate for stage 1')
    parser.add_argument('--learning_rate_stage2', type=float, default=5e-5,
                       help='Learning rate for stage 2')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # System configuration
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='Specific GPU ID to use (e.g., 0, 1, 2...)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Evaluation
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation frequency (steps)')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Checkpoint save frequency (steps)')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Logging frequency (steps)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Run with small subset for testing')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(device_arg, gpu_id=None, gpu_ids=None):
    """Setup compute device with GPU selection."""
    
    # Handle GPU selection
    if gpu_ids:
        # Multiple GPUs specified
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"🔧 Set CUDA_VISIBLE_DEVICES to: {gpu_ids}")
    elif gpu_id is not None:
        # Single GPU specified
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"🔧 Set CUDA_VISIBLE_DEVICES to: {gpu_id}")
    
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda' and torch.cuda.is_available():
        print(f"🚀 CUDA available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {name} ({memory:.1f} GB)")
        
        # Set current device
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)  # Use first available GPU
            print(f"   Active GPU: {torch.cuda.current_device()}")
    else:
        print("🖥️  Using CPU")
    
    return device


def load_splits(splits_file):
    """Load the prepared GuitarSet splits."""
    print(f"Loading splits from: {splits_file}")
    
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    
    train_excerpts = splits['train']
    val_excerpts = splits['val']
    test_excerpts = splits['test']
    
    print(f"Loaded splits:")
    print(f"  Train: {len(train_excerpts)} excerpts")
    print(f"  Val: {len(val_excerpts)} excerpts")
    print(f"  Test: {len(test_excerpts)} excerpts")
    
    return train_excerpts, val_excerpts, test_excerpts


def create_model_and_tokenizer(pretrained_model_path=None, device='cuda'):
    """Create model and tokenizer."""
    
    print("Creating tokenizer...")
    tokenizer = UnifiedFrettingTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    print("Creating model...")
    model = create_model_from_tokenizer(tokenizer, config_type='paper')
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained weights from: {pretrained_model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        print("Pre-trained weights loaded successfully")
        
    else:
        print("No pre-trained model specified, training from scratch")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model, tokenizer


def freeze_encoder(model):
    """Freeze encoder parameters for stage 1 training."""
    print("Freezing encoder parameters...")
    
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if 'encoder' in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
    
    trainable_params = total_params - frozen_params
    print(f"Frozen: {frozen_params:,} parameters")
    print(f"Trainable: {trainable_params:,} parameters")
    
    return model


def unfreeze_all(model):
    """Unfreeze all parameters for stage 2 training."""
    print("Unfreezing all parameters...")
    
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_fp16=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with optional mixed precision
        if use_fp16:
            with autocast(device_type=device.type):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        # Backward pass
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate(model, dataloader, device, tokenizer):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    results = {'loss': avg_loss}
    return results


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, step, loss, output_dir):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{step}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)
    
    # Save model config
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_type': 'UnifiedFrettingT5Model',
            'vocab_size': tokenizer.vocab_size,
            'd_model': model.config.d_model,
            'd_ff': model.config.d_ff,
            'num_layers': model.config.num_layers,
            'num_heads': model.config.num_heads,
        }, f, indent=2)
    
    # Save tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'vocab': tokenizer.vocab,
            'config': tokenizer.config.__dict__
        }, f, indent=2)
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, os.path.join(checkpoint_dir, 'training_state.pt'))
    
    print(f"Checkpoint saved to: {checkpoint_dir}")
    return checkpoint_dir


def main():
    """Main fine-tuning function."""
    args = parse_args()
    
    print("="*60)
    print("FRETTING TRANSFORMER GUITARSET FINE-TUNING")
    print("="*60)
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device, args.gpu_id, args.gpu_ids)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    import logging
    log_file = os.path.join(args.output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting fine-tuning with output dir: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Save configuration
    config_file = os.path.join(args.output_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Training config saved to: {config_file}")
    
    # Setup mixed precision
    scaler = GradScaler() if args.use_fp16 else None
    
    # Load data
    train_excerpts, val_excerpts, test_excerpts = load_splits(args.splits_file)
    
    # Limit data for quick test
    if args.quick_test:
        print("Quick test mode: using small subset")
        train_excerpts = train_excerpts[:10]
        val_excerpts = val_excerpts[:5]
        test_excerpts = test_excerpts[:5]
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args.pretrained_model, device)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_guitarset_dataloaders(
        train_excerpts, val_excerpts, tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Training stages
    stages = [
        {
            'name': 'Stage 1: Frozen Encoder',
            'epochs': args.num_epochs_stage1,
            'lr': args.learning_rate_stage1,
            'freeze_encoder': True
        },
        {
            'name': 'Stage 2: Full Model',
            'epochs': args.num_epochs_stage2,
            'lr': args.learning_rate_stage2,
            'freeze_encoder': False
        }
    ]
    
    global_step = 0
    best_val_loss = float('inf')
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"{stage['name']}")
        print(f"{'='*60}")
        
        # Configure model freezing
        if stage['freeze_encoder']:
            model = freeze_encoder(model)
        else:
            model = unfreeze_all(model)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=stage['lr'],
            weight_decay=args.weight_decay
        )
        
        total_steps = len(train_loader) * stage['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(stage['epochs']):
            print(f"\nEpoch {epoch+1}/{stage['epochs']}")
            
            # Train
            start_time = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args.use_fp16
            )
            epoch_time = time.time() - start_time
            
            print(f"Train Loss: {train_loss:.4f}, Time: {epoch_time:.1f}s")
            
            # Evaluate
            if (epoch + 1) % 2 == 0 or epoch == stage['epochs'] - 1:
                print("Evaluating...")
                val_results = evaluate(model, val_loader, device, tokenizer)
                
                val_loss = val_results['loss']
                print(f"Val Loss: {val_loss:.4f}")
                
                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_dir = save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        epoch, global_step, val_loss, args.output_dir
                    )
                    print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            global_step += len(train_loader)
        
        print(f"Completed {stage['name']}")
    
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output directory: {args.output_dir}")
    
    # Final evaluation on test set
    if test_excerpts and not args.quick_test:
        print("\nRunning final evaluation on test set...")
        
        # Create test dataloader
        from torch.utils.data import DataLoader
        from data.guitarset_dataset import GuitarSetDataset
        
        test_dataset = GuitarSetDataset(
            excerpts=test_excerpts,
            tokenizer=tokenizer,
            augment_data=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        test_results = evaluate(model, test_loader, device, tokenizer)
        
        print(f"Final Test Results:")
        for metric, value in test_results.items():
            print(f"  {metric}: {value:.4f}")
    
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()