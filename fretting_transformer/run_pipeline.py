#!/usr/bin/env python3
"""
Unified Pipeline Runner for Fretting Transformer

Handles the complete pipeline: data preparation → training → evaluation
Can run individual stages or the full pipeline with a single command.
"""

import os
import argparse
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Fretting Transformer Pipeline')
    
    # Pipeline control
    parser.add_argument('--stage', type=str, 
                       choices=['data', 'train', 'eval', 'all'], 
                       default='all',
                       help='Which stage to run (data/train/eval/all)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='Base output directory (auto-generated if not specified)')
    parser.add_argument('--experiment_name', type=str,
                       default=None,
                       help='Experiment name (auto-generated if not specified)')
    
    # Data arguments
    parser.add_argument('--synthtab_path', type=str,
                       default='/data/andreaguz/SynthTab_Dev',
                       help='Path to SynthTab dataset')
    parser.add_argument('--data_category', type=str, default='jams',
                       choices=['jams', 'acoustic'],
                       help='Data category to use')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Max files to process (for testing)')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='paper',
                       choices=['paper', 'debug'],
                       help='Model size (paper=full, debug=small)')
    
    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use mixed precision training')
    
    # GPU configuration
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='GPU IDs to use (e.g., "0" or "0,1")')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage')
    
    # Evaluation configuration
    parser.add_argument('--apply_postprocessing', action='store_true',
                       help='Apply post-processing during evaluation')
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Compare with baseline during evaluation')
    
    # System configuration
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--clean_start', action='store_true',
                       help='Remove existing output directory if it exists')
    
    return parser.parse_args()


def setup_gpu(gpu_ids=None, force_cpu=False):
    """Setup GPU configuration."""
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        print("🖥️  Forcing CPU usage")
        return
    
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"🔧 Set CUDA_VISIBLE_DEVICES to: {gpu_ids}")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA available with {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {name} ({memory:.1f} GB)")
        else:
            print("🖥️  CUDA not available, using CPU")
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU status")


def create_experiment_dir(base_dir=None, experiment_name=None, clean_start=False):
    """Create experiment directory with timestamp."""
    if base_dir is None:
        base_dir = "experiments"
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    if clean_start and os.path.exists(experiment_dir):
        print(f"🗑️  Removing existing directory: {experiment_dir}")
        shutil.rmtree(experiment_dir)
    
    # Create directory structure
    subdirs = ['checkpoints', 'logs', 'data', 'evaluation']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"📁 Created experiment directory: {experiment_dir}")
    return experiment_dir


def run_data_preparation(args, experiment_dir):
    """Run data preparation stage."""
    print("\n" + "="*50)
    print("📊 STAGE 1: DATA PREPARATION")
    print("="*50)
    
    from scripts.prepare_data import main as prepare_main
    
    # Override sys.argv for the prepare_data script
    data_args = [
        'prepare_data.py',
        '--synthtab_path', args.synthtab_path,
        '--data_category', args.data_category,
        '--output_dir', os.path.join(experiment_dir, 'data'),
        '--save_tokenizer',
        '--analyze_data'
    ]
    
    if args.max_files:
        data_args.extend(['--max_files', str(args.max_files)])
    
    original_argv = sys.argv
    sys.argv = data_args
    
    try:
        result = prepare_main()
        if result != 0:
            raise RuntimeError("Data preparation failed")
        print("✅ Data preparation completed successfully")
    finally:
        sys.argv = original_argv
    
    return os.path.join(experiment_dir, 'data', 'synthtab_cache.pkl')


def run_training(args, experiment_dir, cache_path):
    """Run training stage."""
    print("\n" + "="*50)
    print("🏋️  STAGE 2: TRAINING")
    print("="*50)
    
    from scripts.train_model import main as train_main
    
    # Override sys.argv for training script
    train_args = [
        'train_model.py',
        '--synthtab_path', args.synthtab_path,
        '--data_category', args.data_category,
        '--cache_path', cache_path,
        '--model_type', args.model_type,
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--output_dir', os.path.join(experiment_dir, 'checkpoints'),
        '--logging_dir', os.path.join(experiment_dir, 'logs'),
        '--seed', str(args.seed)
    ]
    
    if args.max_files:
        train_args.extend(['--max_files', str(args.max_files)])
    
    if args.use_fp16:
        train_args.append('--use_fp16')
    
    original_argv = sys.argv
    sys.argv = train_args
    
    try:
        result = train_main()
        if result != 0:
            raise RuntimeError("Training failed")
        print("✅ Training completed successfully")
    finally:
        sys.argv = original_argv
    
    # Find latest checkpoint
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        return os.path.join(checkpoint_dir, latest_checkpoint)
    return None


def run_evaluation(args, experiment_dir, model_path, cache_path):
    """Run evaluation stage."""
    print("\n" + "="*50)
    print("📊 STAGE 3: EVALUATION")
    print("="*50)
    
    from scripts.evaluate import main as eval_main
    
    # Override sys.argv for evaluation script
    eval_args = [
        'evaluate.py',
        '--model_path', model_path,
        '--synthtab_path', args.synthtab_path,
        '--data_category', args.data_category,
        '--cache_path', cache_path,
        '--batch_size', str(args.batch_size),
        '--output_dir', os.path.join(experiment_dir, 'evaluation')
    ]
    
    if args.max_files:
        eval_args.extend(['--max_files', str(args.max_files)])
    
    if args.apply_postprocessing:
        eval_args.append('--apply_postprocessing')
    
    if args.compare_baseline:
        eval_args.append('--compare_baseline')
    
    original_argv = sys.argv
    sys.argv = eval_args
    
    try:
        result = eval_main()
        if result != 0:
            raise RuntimeError("Evaluation failed")
        print("✅ Evaluation completed successfully")
    finally:
        sys.argv = original_argv


def save_pipeline_config(args, experiment_dir):
    """Save pipeline configuration for reproducibility."""
    config = {
        'pipeline_args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'command_line': ' '.join(sys.argv),
        'environment': {
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')
        }
    }
    
    config_path = os.path.join(experiment_dir, 'pipeline_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Saved pipeline configuration to: {config_path}")


def print_final_summary(args, experiment_dir, success_stages):
    """Print final summary of the pipeline run."""
    print("\n" + "="*60)
    print("🎯 PIPELINE SUMMARY")
    print("="*60)
    
    print(f"📁 Experiment Directory: {experiment_dir}")
    print(f"⚙️  Configuration: {args.model_type} model, {args.data_category} data")
    print(f"📊 Data: {args.max_files or 'all'} files from {args.synthtab_path}")
    
    stage_names = {'data': 'Data Preparation', 'train': 'Training', 'eval': 'Evaluation'}
    print(f"\n📈 Completed Stages:")
    for stage in success_stages:
        print(f"   ✅ {stage_names.get(stage, stage)}")
    
    # Show key output files
    outputs = []
    if 'data' in success_stages:
        outputs.append(f"   📊 Processed Data: {experiment_dir}/data/")
    if 'train' in success_stages:
        outputs.append(f"   🏋️  Model Checkpoints: {experiment_dir}/checkpoints/")
        outputs.append(f"   📝 Training Logs: {experiment_dir}/logs/")
    if 'eval' in success_stages:
        outputs.append(f"   📊 Evaluation Results: {experiment_dir}/evaluation/")
    
    if outputs:
        print(f"\n📂 Key Outputs:")
        for output in outputs:
            print(output)
    
    print(f"\n🔧 To reproduce this run:")
    print(f"   python run_pipeline.py --output_dir {experiment_dir} --experiment_name {os.path.basename(experiment_dir)}")


def main():
    """Main pipeline runner."""
    args = parse_args()
    
    print("🎸 Fretting Transformer Pipeline")
    print("="*60)
    
    # Setup GPU
    setup_gpu(args.gpu_ids, args.force_cpu)
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(
        args.output_dir, 
        args.experiment_name, 
        args.clean_start
    )
    
    # Save configuration
    save_pipeline_config(args, experiment_dir)
    
    success_stages = []
    cache_path = None
    model_path = None
    
    try:
        # Stage 1: Data Preparation
        if args.stage in ['data', 'all']:
            cache_path = run_data_preparation(args, experiment_dir)
            success_stages.append('data')
        
        # Stage 2: Training
        if args.stage in ['train', 'all']:
            if cache_path is None:
                cache_path = os.path.join(experiment_dir, 'data', 'synthtab_cache.pkl')
            model_path = run_training(args, experiment_dir, cache_path)
            success_stages.append('train')
        
        # Stage 3: Evaluation
        if args.stage in ['eval', 'all']:
            if model_path is None:
                # Find existing checkpoint
                checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            if model_path:
                run_evaluation(args, experiment_dir, model_path, cache_path)
                success_stages.append('eval')
            else:
                print("⚠️  No trained model found for evaluation")
        
        print_final_summary(args, experiment_dir, success_stages)
        print("\n🎉 Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)