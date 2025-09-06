#!/usr/bin/env python3
"""
Main Training Script for Fretting Transformer

This script orchestrates the complete training pipeline:
1. Data loading and preprocessing from SynthTab
2. Model initialization with paper specifications
3. Training with Adafactor optimizer
4. Evaluation and checkpointing
"""

import os
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch

from data.synthtab_loader import SynthTabLoader
from data.tokenizer import FrettingTokenizer
from data.dataset import FrettingDataProcessor, create_data_loaders
from model.fretting_t5 import create_model_from_tokenizer
from training.train import FrettingTrainer, TrainingConfig
from training.utils import set_seed, plot_training_curves
from evaluation.metrics import FrettingEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Fretting Transformer model')
    
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
                       default='data/processed/cache.pkl',
                       help='Path to cache processed data')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='paper',
                       choices=['paper', 'debug'],
                       help='Model configuration type')
    
    # Training arguments  
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--validation_steps', type=int, default=500,
                       help='Steps between validation runs')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Steps between logging')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Steps between checkpoints')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default='experiments/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging_dir', type=str,
                       default='experiments/logs', 
                       help='Logging directory')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Checkpoint to resume from')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/splits').mkdir(parents=True, exist_ok=True)


def save_config(args, save_path: str):
    """Save training configuration."""
    config_dict = vars(args)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_and_process_data(args):
    """Load and process SynthTab data."""
    print("=== Data Loading and Processing ===")
    
    # Create data processor
    processor = FrettingDataProcessor(
        synthtab_path=args.synthtab_path
    )
    
    # Load and process data
    processor.load_and_process_data(
        category=args.data_category,
        max_files=args.max_files,
        cache_path=args.cache_path
    )
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = processor.create_data_splits()
    
    print(f"Dataset splits: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    return processor.tokenizer, train_loader, val_loader, test_loader


def create_model(tokenizer, args):
    """Create and initialize model."""
    print("=== Model Creation ===")
    
    model = create_model_from_tokenizer(tokenizer, args.model_type)
    
    model_info = model.get_model_info()
    print(f"Model created: {model_info['parameters_millions']:.2f}M parameters")
    print(f"Input vocab size: {model_info['input_vocab_size']}")
    print(f"Output vocab size: {model_info['output_vocab_size']}")
    
    return model


def create_training_config(args):
    """Create training configuration."""
    return TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_steps=args.validation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        use_fp16=args.use_fp16,
        seed=args.seed
    )


def run_evaluation(model, tokenizer, test_loader, output_dir):
    """Run final evaluation on test set."""
    print("=== Final Evaluation ===")
    
    evaluator = FrettingEvaluator(tokenizer)
    model.eval()
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate predictions with proper constraints
            input_length = batch['input_ids'].size(1)
            max_new_tokens = min(input_length * 2, 256)  # At most 2x input length
            
            generated = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.output_token_to_id[tokenizer.config.eos_token],
                pad_token_id=tokenizer.output_token_to_id[tokenizer.config.pad_token]
            )
            
            # Convert to tokens
            for i in range(len(batch['input_ids'])):
                input_tokens = tokenizer.ids_to_tokens(
                    batch['input_ids'][i].cpu().tolist(), 'input'
                )
                pred_tokens = tokenizer.ids_to_tokens(
                    generated[i].cpu().tolist(), 'output'
                )
                gt_tokens = tokenizer.ids_to_tokens(
                    batch['labels'][i].cpu().tolist(), 'output'
                )
                
                predictions.append({
                    'input_tokens': input_tokens,
                    'predicted_tokens': pred_tokens
                })
                
                ground_truth.append({
                    'input_tokens': input_tokens,
                    'ground_truth_tokens': gt_tokens
                })
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(predictions, ground_truth)
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create and save report
    report = evaluator.create_evaluation_report(metrics)
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("Evaluation Results:")
    print(f"  Pitch Accuracy: {metrics['pitch_accuracy_mean']:.2%}")
    print(f"  Tab Accuracy: {metrics['tab_accuracy_mean']:.2%}")
    print(f"  Avg Difficulty: {metrics['predicted_difficulty_mean']:.4f}")
    print(f"  Report saved to: {report_path}")
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    
    print("=== Fretting Transformer Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup directories
    setup_directories(args)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    save_config(args, config_path)
    print(f"Configuration saved to: {config_path}")
    
    try:
        # Load and process data
        tokenizer, train_loader, val_loader, test_loader = load_and_process_data(args)
        
        # Create model
        model = create_model(tokenizer, args)
        
        # Create trainer
        training_config = create_training_config(args)
        trainer = FrettingTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=training_config
        )
        
        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            trainer.resume_from_checkpoint(args.resume_from_checkpoint)
            print(f"Resumed training from: {args.resume_from_checkpoint}")
        
        # Train model
        print("=== Starting Training ===")
        training_history = trainer.train()
        
        # Plot training curves
        plot_path = os.path.join(args.logging_dir, 'training_curves.png')
        plot_training_curves(
            training_history['train_losses'],
            training_history['val_losses'],
            save_path=plot_path
        )
        print(f"Training curves saved to: {plot_path}")
        
        # Save training history
        history_path = os.path.join(args.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Run final evaluation
        eval_metrics = run_evaluation(model, tokenizer, test_loader, args.output_dir)
        
        print("=== Training Completed Successfully ===")
        print(f"Final metrics:")
        print(f"  Best validation loss: {training_history['best_val_loss']:.4f}")
        print(f"  Final pitch accuracy: {eval_metrics['pitch_accuracy_mean']:.2%}")
        print(f"  Final tab accuracy: {eval_metrics['tab_accuracy_mean']:.2%}")
        print(f"  Model saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)