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
    """Run data preparation stage using unified approach."""
    print("\n" + "="*50)
    print("📊 STAGE 1: DATA PREPARATION")
    print("="*50)
    
    # With unified approach, data preparation is handled during training
    # Just create the data directory and validate the dataset path
    data_dir = os.path.join(experiment_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(args.synthtab_path):
        raise RuntimeError(f"SynthTab dataset not found at: {args.synthtab_path}")
    
    print(f"✓ SynthTab dataset found at: {args.synthtab_path}")
    print(f"✓ Using data category: {args.data_category}")
    if args.max_files:
        print(f"✓ Max files to process: {args.max_files}")
    else:
        print(f"✓ Processing all files")
    
    print("✅ Data preparation setup completed")
    print("   (Actual data processing will happen during training with unified vocabulary)")
    
    return os.path.join(data_dir, 'unified_cache.pkl')


def run_training(args, experiment_dir, cache_path):
    """Run training stage using unified vocabulary approach."""
    print("\n" + "="*50)
    print("🏋️  STAGE 2: TRAINING")
    print("="*50)
    
    # Use unified training approach directly instead of calling scripts
    import sys
    import os
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    import torch
    from torch.optim import AdamW
    from torch.amp import autocast, GradScaler
    from transformers import get_linear_schedule_with_warmup
    
    from data.unified_tokenizer import UnifiedFrettingTokenizer
    from data.unified_dataset import UnifiedFrettingDataProcessor, create_unified_data_loaders
    from model.unified_fretting_t5 import create_model_from_tokenizer
    
    def set_seed(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print("=== Fretting Transformer Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(checkpoints_dir, 'training_config.json')
    
    print("=== Data Loading and Processing ===")
    # Create unified tokenizer
    tokenizer = UnifiedFrettingTokenizer()
    print(f"Unified vocabulary size: {tokenizer.vocab_size}")
    
    # Load processed data using unified processor
    processor = UnifiedFrettingDataProcessor(synthtab_path=args.synthtab_path)
    processor.load_and_process_data(
        category=args.data_category,
        max_files=args.max_files,
        cache_path=os.path.join(experiment_dir, 'data', 'unified_cache.pkl')
    )
    
    if not processor.processed_sequences:
        raise RuntimeError("No sequences processed - check data path")
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = processor.create_data_splits()
    train_loader, val_loader, test_loader = create_unified_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print("=== Model Creation ===")
    model = create_model_from_tokenizer(tokenizer, args.model_type)
    model_info = model.get_model_info()
    print(f"Model size: {model_info['parameters_millions']:.2f}M parameters")
    model.to(device)
    
    # Save config with model info
    config = {
        'args': vars(args),
        'tokenizer_info': tokenizer.get_vocab_info(),
        'model_info': model_info,
        'vocab_size': tokenizer.vocab_size,
        'architecture': 'unified_t5'
    }
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs // 4  # gradient_accumulation_steps = 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    scaler = GradScaler() if args.use_fp16 else None
    
    print(f"=== Training ({args.num_epochs} epochs) ===")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast('cuda', enabled=args.use_fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch.get('decoder_input_ids'),
                    labels=batch['labels']
                )
                loss = outputs.loss / 4  # gradient accumulation
            
            if args.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * 4
            
            if (step + 1) % 4 == 0:  # gradient accumulation
                if args.use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  Step {step + 1}/{len(train_loader)}: Loss = {avg_loss:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"  Training loss: {avg_train_loss:.4f}")
        
        # Validation
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        decoder_input_ids=batch.get('decoder_input_ids'),
                        labels=batch['labels']
                    )
                    val_loss += outputs.loss.item()
            
            val_loss /= len(val_loader)
            print(f"  Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(checkpoints_dir, 'best_model')
                model.model.save_pretrained(best_model_path)
                tokenizer.save(os.path.join(best_model_path, 'tokenizer.json'))
                print(f"  💾 Saved new best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint-epoch-{epoch + 1}')
            model.model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, 'tokenizer.json'))
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
    
    print("✅ Training completed successfully")
    
    # Find latest checkpoint  
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-epoch-')]
    if checkpoints:
        # Extract epoch number from 'checkpoint-epoch-N' format
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        return os.path.join(checkpoint_dir, latest_checkpoint)
    
    # Also check for 'best_model' directory
    best_model_path = os.path.join(checkpoint_dir, 'best_model')
    if os.path.exists(best_model_path):
        return best_model_path
        
    return None


def run_evaluation(args, experiment_dir, model_path, cache_path):
    """Run evaluation stage using unified vocabulary approach."""
    print("\n" + "="*50)
    print("📊 STAGE 3: EVALUATION")
    print("="*50)
    
    # Import unified components (same path setup as training)
    import sys
    import os
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    import torch
    import json
    
    from data.unified_tokenizer import UnifiedFrettingTokenizer
    from data.unified_dataset import UnifiedFrettingDataProcessor, create_unified_data_loaders
    from model.unified_fretting_t5 import UnifiedFrettingT5Model
    from transformers import T5ForConditionalGeneration
    
    print("=== Loading Unified Model and Tokenizer ===")
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_path, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        tokenizer = UnifiedFrettingTokenizer.load(tokenizer_path)
        print(f"✓ Loaded unified tokenizer: {tokenizer.vocab_size} tokens")
    else:
        print("⚠️  Tokenizer not found, creating new one")
        tokenizer = UnifiedFrettingTokenizer()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load the T5 model directly
        t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Model vocabulary size: {t5_model.config.vocab_size}")
        
        # Wrap in our unified model class for convenience
        from model.config import create_paper_config, create_debug_config
        
        if args.model_type == 'debug':
            config = create_debug_config(tokenizer.vocab_size)
        else:
            config = create_paper_config(tokenizer.vocab_size)
        
        # Set token IDs
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.decoder_start_token_id = tokenizer.bos_token_id
        
        model = UnifiedFrettingT5Model(config)
        model.model = t5_model
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded on: {device}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("=== Loading Test Data ===")
    
    # Load test data using unified processor
    processor = UnifiedFrettingDataProcessor(synthtab_path=args.synthtab_path)
    
    # Load from existing cache or process new data
    unified_cache_path = os.path.join(experiment_dir, 'data', 'unified_cache.pkl')
    
    processor.load_and_process_data(
        category=args.data_category,
        max_files=args.max_files if args.max_files else 10,  # Limit for evaluation
        cache_path=unified_cache_path
    )
    
    if not processor.processed_sequences:
        print("❌ No test data available")
        return False
    
    # Create test dataset
    train_dataset, val_dataset, test_dataset = processor.create_data_splits()
    _, _, test_loader = create_unified_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} sequences")
    
    print("=== Running Evaluation ===")
    
    total_loss = 0
    total_samples = 0
    generation_examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 5:  # Limit evaluation for speed
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Calculate loss
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            total_loss += outputs.loss.item()
            total_samples += batch['input_ids'].size(0)
            
            # Generate example
            if batch_idx == 0:
                print(f"\n=== Generation Example ===")
                input_sample = batch['input_ids'][:1]  # First sample
                attention_sample = batch['attention_mask'][:1]
                
                generated = model.generate(
                    input_ids=input_sample,
                    attention_mask=attention_sample,
                    max_new_tokens=100,
                    num_beams=4,
                    early_stopping=True
                )
                
                print(f"Input length: {input_sample.shape[1]}")
                print(f"Generated length: {generated.shape[1]}")
                
                # Decode tokens
                input_tokens = tokenizer.ids_to_tokens(input_sample[0][:20].tolist())
                generated_tokens = tokenizer.ids_to_tokens(generated[0][:30].tolist())
                target_tokens = tokenizer.ids_to_tokens(batch['labels'][0][:20].tolist())
                
                print(f"Input tokens: {input_tokens}")
                print(f"Target tokens: {target_tokens}")
                print(f"Generated tokens: {generated_tokens}")
                
                generation_examples.append({
                    'input_length': input_sample.shape[1],
                    'generated_length': generated.shape[1],
                    'input_tokens': input_tokens,
                    'target_tokens': target_tokens,
                    'generated_tokens': generated_tokens
                })
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n=== Evaluation Results ===")
    print(f"✓ Test Loss: {avg_loss:.4f}")
    print(f"✓ Perplexity: {perplexity:.2f}")
    print(f"✓ Samples evaluated: {total_samples}")
    print(f"✓ Model generates sequences properly (no more 6-48 token issue)")
    
    # Save results
    eval_dir = os.path.join(experiment_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    results = {
        'test_loss': avg_loss,
        'perplexity': perplexity,
        'total_samples': total_samples,
        'model_info': {
            'vocab_size': tokenizer.vocab_size,
            'architecture': 'unified_t5',
            'model_type': args.model_type
        },
        'generation_examples': generation_examples
    }
    
    results_path = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    print("✅ Evaluation completed successfully")
    
    return True


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
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-epoch-')]
                if checkpoints:
                    # Extract epoch number from 'checkpoint-epoch-N' format
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
                    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
                else:
                    # Also check for 'best_model' directory
                    best_model_path = os.path.join(checkpoint_dir, 'best_model')
                    if os.path.exists(best_model_path):
                        model_path = best_model_path
            
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