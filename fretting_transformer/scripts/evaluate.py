#!/usr/bin/env python3
"""
Evaluation Script for Fretting Transformer

Evaluates trained models on test data and generates comprehensive reports.
Supports post-processing evaluation and comparison with baselines.
"""

import os
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from tqdm import tqdm

from data.synthtab_loader import SynthTabLoader
from data.tokenizer import FrettingTokenizer
from data.dataset import FrettingDataProcessor, create_data_loaders
from model.fretting_t5 import FrettingT5Model
from inference.generate import ChunkedInference
from inference.postprocess import apply_postprocessing
from evaluation.metrics import FrettingEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Fretting Transformer model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                       help='Path to saved tokenizer (optional)')
    
    # Data arguments
    parser.add_argument('--synthtab_path', type=str, 
                       default='/data/andreaguz/SynthTab_Dev',
                       help='Path to SynthTab dataset')
    parser.add_argument('--data_category', type=str, default='jams',
                       choices=['jams', 'acoustic'],
                       help='Which data category to evaluate on')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to evaluate (for testing)')
    parser.add_argument('--cache_path', type=str, 
                       default='data/processed/eval_cache.pkl',
                       help='Path to cache processed data')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Evaluation batch size')
    parser.add_argument('--apply_postprocessing', action='store_true',
                       help='Apply post-processing for pitch correction')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Number of beams for generation')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum generation length')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, 
                       default='experiments/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions')
    
    # Baseline comparison
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Compare with simple baseline (lowest fret)')
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, tokenizer_path: str = None):
    """Load trained model and tokenizer."""
    print("=== Loading Model ===")
    
    # Load model
    model = FrettingT5Model.load_model(model_path)
    
    # Load or create tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = FrettingTokenizer.load_vocab(tokenizer_path)
        print(f"Loaded tokenizer from: {tokenizer_path}")
    else:
        tokenizer = FrettingTokenizer()
        print("Created new tokenizer (may cause vocab mismatch)")
    
    model_info = model.get_model_info()
    print(f"Model loaded: {model_info['parameters_millions']:.2f}M parameters")
    
    return model, tokenizer


def create_baseline_predictions(input_notes: List[int], 
                              tuning: List[int] = None) -> List[tuple]:
    """
    Create simple baseline predictions (lowest fret for each pitch).
    
    Args:
        input_notes: List of MIDI pitches
        tuning: Guitar tuning
        
    Returns:
        List of (string, fret) predictions
    """
    if tuning is None:
        tuning = [64, 59, 55, 50, 45, 40]  # Standard tuning
    
    baseline_tabs = []
    
    for pitch in input_notes:
        # Find lowest fret position
        best_tab = None
        min_fret = float('inf')
        
        for string_idx, open_pitch in enumerate(tuning):
            fret = pitch - open_pitch
            if 0 <= fret <= 24 and fret < min_fret:
                min_fret = fret
                best_tab = (string_idx + 1, fret)  # Convert to 1-based string
        
        if best_tab is not None:
            baseline_tabs.append(best_tab)
        else:
            # Fallback to first string if no valid position found
            baseline_tabs.append((1, 0))
    
    return baseline_tabs


def evaluate_model(model, tokenizer, test_loader, args):
    """Evaluate model on test set."""
    print("=== Model Evaluation ===")
    
    evaluator = FrettingEvaluator(tokenizer)
    inference_engine = ChunkedInference(model, tokenizer)
    
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_baseline = [] if args.compare_baseline else None
    
    total_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate predictions
            generated = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            # Process each sample in batch
            for i in range(len(batch['input_ids'])):
                # Convert to tokens
                input_tokens = tokenizer.ids_to_tokens(
                    batch['input_ids'][i].cpu().tolist(), 'input'
                )
                pred_tokens = tokenizer.ids_to_tokens(
                    generated[i].cpu().tolist(), 'output'
                )
                gt_tokens = tokenizer.ids_to_tokens(
                    batch['labels'][i].cpu().tolist(), 'output'
                )
                
                # Apply post-processing if requested
                if args.apply_postprocessing:
                    pred_tokens, pp_metrics = apply_postprocessing(
                        input_tokens, pred_tokens, tokenizer
                    )
                
                # Store predictions
                pred_data = {
                    'input_tokens': input_tokens,
                    'predicted_tokens': pred_tokens,
                    'sample_id': f"batch_{batch_idx}_sample_{i}"
                }
                all_predictions.append(pred_data)
                
                # Store ground truth
                gt_data = {
                    'input_tokens': input_tokens,
                    'ground_truth_tokens': gt_tokens,
                    'sample_id': f"batch_{batch_idx}_sample_{i}"
                }
                all_ground_truth.append(gt_data)
                
                # Create baseline if requested
                if args.compare_baseline:
                    input_pitches = evaluator._extract_midi_pitches(input_tokens)
                    baseline_tabs = create_baseline_predictions(input_pitches)
                    
                    # Convert to tokens
                    baseline_tokens = ['<BOS>']
                    for string, fret in baseline_tabs:
                        baseline_tokens.append(f'TAB<{string},{fret}>')
                    baseline_tokens.append('<EOS>')
                    
                    baseline_data = {
                        'input_tokens': input_tokens,
                        'predicted_tokens': baseline_tokens,
                        'sample_id': f"baseline_batch_{batch_idx}_sample_{i}"
                    }
                    all_baseline.append(baseline_data)
    
    print(f"Collected {len(all_predictions)} predictions")
    
    return all_predictions, all_ground_truth, all_baseline


def compute_metrics(predictions, ground_truth, evaluator, name="Model"):
    """Compute evaluation metrics."""
    print(f"=== Computing {name} Metrics ===")
    
    try:
        metrics = evaluator.evaluate_dataset(predictions, ground_truth)
        
        print(f"{name} Results:")
        print(f"  Pitch Accuracy: {metrics['pitch_accuracy_mean']:.2%} ± {metrics['pitch_accuracy_std']:.2%}")
        print(f"  Tab Accuracy: {metrics['tab_accuracy_mean']:.2%} ± {metrics['tab_accuracy_std']:.2%}")
        print(f"  Difficulty Score: {metrics['predicted_difficulty_mean']:.4f} ± {metrics['predicted_difficulty_std']:.4f}")
        
        return metrics
    
    except Exception as e:
        print(f"Failed to compute {name} metrics: {e}")
        return None


def save_results(metrics, predictions, ground_truth, output_dir, args):
    """Save evaluation results."""
    print("=== Saving Results ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    if metrics:
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        # Create evaluation report
        evaluator = FrettingEvaluator(FrettingTokenizer())  # Dummy for report
        report = evaluator.create_evaluation_report(metrics)
        
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        pred_path = os.path.join(output_dir, 'predictions.json')
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to: {pred_path}")
        
        gt_path = os.path.join(output_dir, 'ground_truth.json')
        with open(gt_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        print(f"Ground truth saved to: {gt_path}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'eval_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=== Fretting Transformer Evaluation ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
        
        # Load test data
        print("=== Loading Test Data ===")
        processor = FrettingDataProcessor(synthtab_path=args.synthtab_path)
        
        processor.load_and_process_data(
            category=args.data_category,
            max_files=args.max_files,
            cache_path=args.cache_path
        )
        
        train_dataset, val_dataset, test_dataset = processor.create_data_splits()
        _, _, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=args.batch_size,
            num_workers=4
        )
        
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Evaluate model
        predictions, ground_truth, baseline_predictions = evaluate_model(
            model, tokenizer, test_loader, args
        )
        
        # Compute metrics
        evaluator = FrettingEvaluator(tokenizer)
        
        # Model metrics
        model_metrics = compute_metrics(predictions, ground_truth, evaluator, "Model")
        
        # Baseline metrics (if requested)
        baseline_metrics = None
        if args.compare_baseline:
            baseline_metrics = compute_metrics(baseline_predictions, ground_truth, evaluator, "Baseline")
        
        # Create comprehensive results
        results = {
            'model_metrics': model_metrics,
            'baseline_metrics': baseline_metrics,
            'evaluation_config': vars(args),
            'test_set_size': len(predictions)
        }
        
        # Save results
        save_results(model_metrics, predictions, ground_truth, args.output_dir, args)
        
        # Print comparison
        if baseline_metrics and model_metrics:
            print("\n=== Model vs Baseline Comparison ===")
            print(f"Pitch Accuracy: {model_metrics['pitch_accuracy_mean']:.2%} vs {baseline_metrics['pitch_accuracy_mean']:.2%}")
            print(f"Tab Accuracy: {model_metrics['tab_accuracy_mean']:.2%} vs {baseline_metrics['tab_accuracy_mean']:.2%}")
            print(f"Difficulty: {model_metrics['predicted_difficulty_mean']:.4f} vs {baseline_metrics['predicted_difficulty_mean']:.4f}")
        
        print("=== Evaluation Completed Successfully ===")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)