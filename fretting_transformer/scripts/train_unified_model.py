#!/usr/bin/env python3
"""
Unified Training Script for Fretting Transformer

This script uses the unified vocabulary approach that fixes the fundamental
T5 architecture issue where encoder and decoder had different vocabularies.
"""

import os
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from data.unified_tokenizer import UnifiedFrettingTokenizer
from data.unified_dataset import UnifiedFrettingDataProcessor, create_unified_data_loaders
from model.unified_fretting_t5 import create_model_from_tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Unified Fretting Transformer')
    
    # Data arguments
    parser.add_argument('--synthtab_path', type=str, 
                       default='/data/andreaguz/SynthTab_Dev',
                       help='Path to SynthTab dataset')
    parser.add_argument('--data_category', type=str, default='jams',
                       choices=['jams', 'acoustic'],
                       help='Which data category to use')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--cache_path', type=str, 
                       default='data/processed/unified_cache.pkl',
                       help='Path to cache processed data')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='paper',
                       choices=['paper', 'debug'],
                       help='Model configuration type')
    
    # Training arguments  
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size (reduced for unified model)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps for learning rate scheduler')
    parser.add_argument('--validation_steps', type=int, default=500,
                       help='Steps between validation runs')
    parser.add_argument('--logging_steps', type=int, default=50,
                       help='Steps between logging')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Steps between checkpoints')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default='experiments/unified_checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging_dir', type=str,
                       default='experiments/unified_logs', 
                       help='Logging directory')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use mixed precision training')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(args, tokenizer, model_info, save_path):
    """Save training configuration."""
    config = {
        'args': vars(args),
        'tokenizer_info': tokenizer.get_vocab_info(),
        'model_info': model_info,
        'vocab_size': tokenizer.vocab_size,
        'architecture': 'unified_t5'
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to: {save_path}")


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for step, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast('cuda', enabled=args.use_fp16):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch.get('decoder_input_ids'),
                labels=batch['labels']
            )
            loss = outputs.loss / args.gradient_accumulation_steps
        
        if args.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        if (step + 1) % args.logging_steps == 0:
            avg_loss = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step + 1}/{num_batches}: Loss = {avg_loss:.4f}, LR = {lr:.6f}")
    
    return total_loss / num_batches


def validate(model, val_loader, device):
    """Run validation."""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    if num_batches == 0:
        return float('inf')
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch.get('decoder_input_ids'),
                labels=batch['labels']
            )
            total_loss += outputs.loss.item()
    
    return total_loss / num_batches


def main():
    """Main training function."""
    args = parse_args()
    
    print("=== Unified Fretting Transformer Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Setup
    set_seed(args.seed)
    setup_directories(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create unified tokenizer
    print("\n=== Tokenizer Creation ===")
    tokenizer = UnifiedFrettingTokenizer()
    print(f"Unified vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")
    
    # 2. Load and process data
    print("\n=== Data Loading and Processing ===")
    processor = UnifiedFrettingDataProcessor(
        synthtab_path=args.synthtab_path
    )
    
    processor.load_and_process_data(
        category=args.data_category,
        max_files=args.max_files,
        cache_path=args.cache_path
    )
    
    if not processor.processed_sequences:
        print("❌ No sequences processed - check data path and configuration")
        return 1
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = processor.create_data_splits()
    train_loader, val_loader, test_loader = create_unified_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Data splits: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # 3. Create unified model
    print("\n=== Model Creation ===")
    model = create_model_from_tokenizer(tokenizer, args.model_type)
    model_info = model.get_model_info()
    print(f"Model size: {model_info['parameters_millions']:.2f}M parameters")
    print(f"Model vocabulary: {model_info['vocab_size']}")
    
    model.to(device)
    
    # 4. Setup training
    print("\n=== Training Setup ===")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler() if args.use_fp16 else None
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    save_config(args, tokenizer, model_info, config_path)
    
    # 5. Training loop
    print(f"\n=== Training ({args.num_epochs} epochs) ===")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, args)
        print(f"  Training loss: {train_loss:.4f}")
        
        # Validate
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader, device)
            print(f"  Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.output_dir, 'best_model')
                model.model.save_pretrained(best_model_path)
                tokenizer.save(os.path.join(best_model_path, 'tokenizer.json'))
                print(f"  💾 Saved new best model (val_loss: {val_loss:.4f})")
        else:
            print("  No validation data available")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch + 1}')
            model.model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, 'tokenizer.json'))
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
    
    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)