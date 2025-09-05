#!/usr/bin/env python3
"""
Sanity Check Script for Fretting Transformer

Runs comprehensive validation before long training runs:
- Data loading and processing
- Model initialization and forward pass
- GPU memory estimation
- Tokenization validation
- Pipeline components integration
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sanity check before training')
    
    parser.add_argument('--synthtab_path', type=str,
                       default='/data/andreaguz/SynthTab_Dev',
                       help='Path to SynthTab dataset')
    parser.add_argument('--data_category', type=str, default='jams',
                       choices=['jams', 'acoustic'],
                       help='Data category to check')
    parser.add_argument('--max_files', type=int, default=5,
                       help='Max files to check (keep small for speed)')
    parser.add_argument('--model_type', type=str, default='debug',
                       choices=['paper', 'debug'],
                       help='Model type to test')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--sequence_length', type=int, default=128,
                       help='Sequence length for testing')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='GPU IDs to test')
    parser.add_argument('--check_full_pipeline', action='store_true',
                       help='Test complete pipeline end-to-end')
    
    return parser.parse_args()


def setup_gpu(gpu_ids=None):
    """Setup GPU for testing."""
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"🔧 Set CUDA_VISIBLE_DEVICES to: {gpu_ids}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Memory: {memory_gb:.1f} GB")
    
    return device


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 50)
    print("📦 CHECKING DEPENDENCIES")
    print("=" * 50)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'mido': 'MIDI processing',
        'numpy': 'NumPy',
        'matplotlib': 'Plotting',
        'sklearn': 'Scikit-learn',
        'tqdm': 'Progress bars',
        'yaml': 'YAML config'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + ' '.join(missing_packages))
        return False
    
    print("✅ All dependencies installed")
    return True


def check_data_access(synthtab_path, data_category, max_files):
    """Check SynthTab data access and loading."""
    print("\n" + "=" * 50)
    print("📊 CHECKING DATA ACCESS")
    print("=" * 50)
    
    try:
        from data.synthtab_loader import SynthTabLoader
        
        # Check path exists
        if not os.path.exists(synthtab_path):
            print(f"❌ SynthTab path does not exist: {synthtab_path}")
            return False
        
        print(f"✅ SynthTab path exists: {synthtab_path}")
        
        # Initialize loader
        loader = SynthTabLoader(synthtab_path)
        
        # Find JAMS files
        jams_files = loader.find_jams_files(data_category)
        print(f"✅ Found {len(jams_files)} JAMS files in {data_category}")
        
        if len(jams_files) == 0:
            print(f"❌ No JAMS files found in {data_category}")
            return False
        
        # Test loading a few files
        test_files = jams_files[:min(max_files, len(jams_files))]
        total_notes = 0
        
        for i, jams_file in enumerate(test_files):
            try:
                notes = loader.load_jams_file(jams_file)
                total_notes += len(notes)
                print(f"   File {i+1}: {len(notes)} notes - {os.path.basename(jams_file)}")
            except Exception as e:
                print(f"   ❌ Failed to load file {i+1}: {e}")
                return False
        
        print(f"✅ Successfully loaded {total_notes} total notes from {len(test_files)} files")
        
        # Test MIDI event conversion
        if test_files:
            notes = loader.load_jams_file(test_files[0])
            midi_events = loader.notes_to_midi_sequence(notes[:10])  # Test first 10 notes
            print(f"✅ Converted {len(notes[:10])} notes to {len(midi_events)} MIDI events")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tokenization():
    """Check tokenization system."""
    print("\n" + "=" * 50)
    print("🔤 CHECKING TOKENIZATION")
    print("=" * 50)
    
    try:
        from data.tokenizer import FrettingTokenizer
        
        # Create tokenizer
        tokenizer = FrettingTokenizer()
        input_size, output_size = tokenizer.get_vocab_sizes()
        print(f"✅ Tokenizer created - Input vocab: {input_size}, Output vocab: {output_size}")
        
        # Test encoding
        sample_events = [
            {'type': 'note_on', 'pitch': 55, 'string': 3, 'fret': 0},
            {'type': 'time_shift', 'delta': 120},
            {'type': 'note_off', 'pitch': 55, 'string': 3, 'fret': 0},
            {'type': 'note_on', 'pitch': 57, 'string': 3, 'fret': 2},
            {'type': 'time_shift', 'delta': 120},
            {'type': 'note_off', 'pitch': 57, 'string': 3, 'fret': 2}
        ]
        
        input_tokens, output_tokens = tokenizer.encode_sequence_pair(sample_events)
        print(f"✅ Encoded {len(sample_events)} events → {len(input_tokens)} input, {len(output_tokens)} output tokens")
        
        # Test ID conversion
        input_ids = tokenizer.tokens_to_ids(input_tokens, 'input')
        output_ids = tokenizer.tokens_to_ids(output_tokens, 'output')
        print(f"✅ Token-to-ID conversion works - Input IDs: {len(input_ids)}, Output IDs: {len(output_ids)}")
        
        # Test decoding
        decoded_input = tokenizer.ids_to_tokens(input_ids, 'input')
        decoded_output = tokenizer.ids_to_tokens(output_ids, 'output')
        
        if decoded_input == input_tokens and decoded_output == output_tokens:
            print("✅ Round-trip encoding/decoding successful")
        else:
            print("❌ Round-trip encoding/decoding failed")
            return False
        
        # Test tablature decoding
        tab_pairs = tokenizer.decode_tablature_tokens(output_tokens)
        print(f"✅ Decoded {len(tab_pairs)} tablature pairs: {tab_pairs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_creation(model_type, device):
    """Check model creation and basic operations."""
    print("\n" + "=" * 50)
    print("🤖 CHECKING MODEL CREATION")
    print("=" * 50)
    
    try:
        from data.tokenizer import FrettingTokenizer
        from model.fretting_t5 import create_model_from_tokenizer
        
        # Create tokenizer and model
        tokenizer = FrettingTokenizer()
        model = create_model_from_tokenizer(tokenizer, model_type)
        model.to(device)
        
        model_info = model.get_model_info()
        print(f"✅ Model created: {model_info['parameters_millions']:.2f}M parameters")
        print(f"   - Input vocab: {model_info['input_vocab_size']}")
        print(f"   - Output vocab: {model_info['output_vocab_size']}")
        print(f"   - Architecture: d_model={model_info['d_model']}, layers={model_info['num_layers']}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_forward_pass(model, tokenizer, device, batch_size, sequence_length):
    """Check model forward pass."""
    print("\n" + "=" * 50)
    print("⚡ CHECKING FORWARD PASS")
    print("=" * 50)
    
    try:
        model.eval()
        
        # Create dummy batch
        input_vocab_size = len(tokenizer.input_vocab)
        output_vocab_size = len(tokenizer.output_vocab)
        
        input_ids = torch.randint(0, input_vocab_size, (batch_size, sequence_length), device=device)
        attention_mask = torch.ones(batch_size, sequence_length, device=device)
        labels = torch.randint(0, output_vocab_size, (batch_size, sequence_length), device=device)
        
        print(f"✅ Created dummy batch: {input_ids.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            print(f"✅ Forward pass successful")
            print(f"   - Loss: {loss.item():.4f}")
            print(f"   - Logits shape: {logits.shape}")
            print(f"   - Expected logits shape: ({batch_size}, {sequence_length}, {output_vocab_size})")
            
            if logits.shape != (batch_size, sequence_length, output_vocab_size):
                print(f"❌ Unexpected logits shape!")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_generation(model, tokenizer, device):
    """Check text generation."""
    print("\n" + "=" * 50)
    print("🔮 CHECKING GENERATION")
    print("=" * 50)
    
    try:
        model.eval()
        
        # Create sample input
        input_vocab_size = len(tokenizer.input_vocab)
        input_ids = torch.randint(0, input_vocab_size, (1, 32), device=device)
        attention_mask = torch.ones(1, 32, device=device)
        
        print(f"✅ Created sample input: {input_ids.shape}")
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=2,
                early_stopping=True
            )
            
            print(f"✅ Generation successful")
            print(f"   - Generated shape: {generated.shape}")
            print(f"   - Generated length: {generated.shape[1]}")
            
            # Decode first few tokens
            generated_tokens = tokenizer.ids_to_tokens(generated[0][:10].cpu().tolist(), 'output')
            print(f"   - First 10 tokens: {generated_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory_usage(model, batch_size, sequence_length, device):
    """Check GPU memory usage."""
    print("\n" + "=" * 50)
    print("💾 CHECKING MEMORY USAGE")
    print("=" * 50)
    
    if device.type != 'cuda':
        print("⚠️  CPU mode - skipping memory check")
        return True
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU memory check")
        print(f"   - Total memory: {total_memory:.1f} GB")
        print(f"   - Initial allocated: {initial_memory:.1f} GB")
        
        # Load model to GPU
        model.to(device)
        model_memory = torch.cuda.memory_allocated() / 1e9
        
        print(f"   - After model load: {model_memory:.1f} GB")
        print(f"   - Model size: {model_memory - initial_memory:.1f} GB")
        
        # Check if we have enough memory for training
        estimated_training_memory = model_memory * 3  # Rough estimate
        memory_usage_percent = (estimated_training_memory / total_memory) * 100
        
        print(f"   - Estimated training memory: {estimated_training_memory:.1f} GB ({memory_usage_percent:.1f}%)")
        
        if memory_usage_percent > 90:
            print("⚠️  High memory usage - consider reducing batch size")
        elif memory_usage_percent > 70:
            print("⚠️  Moderate memory usage - monitor during training")
        else:
            print("✅ Memory usage looks good")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory check failed: {e}")
        return False


def check_data_pipeline(synthtab_path, data_category, max_files):
    """Check complete data pipeline."""
    print("\n" + "=" * 50)
    print("🔄 CHECKING DATA PIPELINE")
    print("=" * 50)
    
    try:
        from data.dataset import FrettingDataProcessor
        
        # Create processor with small test config
        processor = FrettingDataProcessor(synthtab_path=synthtab_path)
        
        print("✅ Created data processor")
        
        # Process small amount of data
        processor.load_and_process_data(
            category=data_category,
            max_files=max_files,
            cache_path='temp_test_cache.pkl'
        )
        
        print(f"✅ Processed {len(processor.processed_sequences)} sequences")
        
        # Create data splits
        train_dataset, val_dataset, test_dataset = processor.create_data_splits()
        print(f"✅ Created data splits: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
        # Test a single sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"✅ Sample data:")
            print(f"   - Input shape: {sample['input_ids'].shape}")
            print(f"   - Labels shape: {sample['labels'].shape}")
            print(f"   - Attention mask shape: {sample['attention_mask'].shape}")
        
        # Clean up
        if os.path.exists('temp_test_cache.pkl'):
            os.remove('temp_test_cache.pkl')
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline_test(args):
    """Run a complete mini pipeline test."""
    print("\n" + "=" * 60)
    print("🚀 FULL PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Import pipeline script
        sys.path.append('.')
        from run_pipeline import main as pipeline_main
        
        # Create test arguments
        test_args = [
            'run_pipeline.py',
            '--stage', 'data',
            '--synthtab_path', args.synthtab_path,
            '--data_category', args.data_category,
            '--max_files', '3',
            '--model_type', 'debug',
            '--num_epochs', '1',
            '--batch_size', '2',
            '--output_dir', 'temp_pipeline_test',
            '--clean_start'
        ]
        
        if args.gpu_ids:
            test_args.extend(['--gpu_ids', args.gpu_ids])
        
        # Override sys.argv and run
        original_argv = sys.argv
        sys.argv = test_args
        
        try:
            result = pipeline_main()
            if result == 0:
                print("✅ Full pipeline test successful")
                return True
            else:
                print("❌ Full pipeline test failed")
                return False
        finally:
            sys.argv = original_argv
            
            # Clean up
            import shutil
            if os.path.exists('temp_pipeline_test'):
                shutil.rmtree('temp_pipeline_test')
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


def main():
    """Main sanity check function."""
    args = parse_args()
    
    print("🎸 FRETTING TRANSFORMER SANITY CHECK")
    print("=" * 60)
    
    # Setup
    device = setup_gpu(args.gpu_ids)
    
    checks_passed = 0
    total_checks = 0
    
    # Run checks
    checks = [
        ("Dependencies", lambda: check_dependencies()),
        ("Data Access", lambda: check_data_access(args.synthtab_path, args.data_category, args.max_files)),
        ("Tokenization", lambda: check_tokenization()),
        ("Data Pipeline", lambda: check_data_pipeline(args.synthtab_path, args.data_category, args.max_files)),
    ]
    
    # Model checks
    model, tokenizer = check_model_creation(args.model_type, device)
    if model is not None:
        checks.extend([
            ("Forward Pass", lambda: check_forward_pass(model, tokenizer, device, args.batch_size, args.sequence_length)),
            ("Generation", lambda: check_generation(model, tokenizer, device)),
            ("Memory Usage", lambda: check_memory_usage(model, args.batch_size, args.sequence_length, device)),
        ])
    
    # Full pipeline test
    if args.check_full_pipeline:
        checks.append(("Full Pipeline", lambda: run_full_pipeline_test(args)))
    
    # Run all checks
    for check_name, check_func in checks:
        total_checks += 1
        try:
            if check_func():
                checks_passed += 1
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SANITY CHECK SUMMARY")
    print("=" * 60)
    
    print(f"✅ Passed: {checks_passed}/{total_checks} checks")
    
    if checks_passed == total_checks:
        print("\n🎉 ALL CHECKS PASSED - Ready for training!")
        return 0
    else:
        print(f"\n⚠️  {total_checks - checks_passed} checks failed - Fix issues before training")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)