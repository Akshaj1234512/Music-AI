"""
Test script to verify the Basic Pitch evaluation pipeline works correctly.
Run this before processing your full dataset.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError:
        print("  ✗ yaml - install with: pip install pyyaml")
        return False
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - install with: pip install numpy")
        return False
    
    try:
        import librosa
        print("  ✓ librosa")
    except ImportError:
        print("  ✗ librosa - install with: pip install librosa")
        return False
    
    try:
        import pretty_midi
        print("  ✓ pretty_midi")
    except ImportError:
        print("  ✗ pretty_midi - install with: pip install pretty_midi")
        return False
    
    try:
        import mir_eval
        print("  ✓ mir_eval")
    except ImportError:
        print("  ✗ mir_eval - install with: pip install mir_eval")
        return False
    
    try:
        import sklearn
        print("  ✓ sklearn")
    except ImportError:
        print("  ✗ sklearn - install with: pip install scikit-learn")
        return False
    
    try:
        import tensorflow as tf
        print("  ✓ tensorflow")
    except ImportError:
        print("  ✗ tensorflow - install with: pip install tensorflow")
        return False
    
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        print("  ✓ basic_pitch")
    except ImportError:
        print("  ✗ basic_pitch - install with: pip install basic-pitch[tf]")
        return False
    
    return True


def test_config():
    """Test that config.yaml exists and is valid."""
    print("\nTesting configuration...")
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("  ✗ config.yaml not found")
        print("    Please create config.yaml with your dataset paths")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['paths', 'basic_pitch', 'evaluation', 'gpu']
        for section in required_sections:
            if section not in config:
                print(f"  ✗ Missing config section: {section}")
                return False
        
        print("  ✓ config.yaml is valid")
        
        # Check if paths are filled in
        audio_dir = config['paths'].get('guitarset_audio_dir', '')
        midi_dir = config['paths'].get('guitarset_midi_dir', '')
        
        if not audio_dir or not midi_dir:
            print("  ⚠ Warning: Audio/MIDI directory paths not set in config.yaml")
            print("    You'll need to set these before running the evaluation")
        else:
            print(f"  ✓ Audio directory: {audio_dir}")
            print(f"  ✓ MIDI directory: {midi_dir}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading config.yaml: {e}")
        return False


def test_basic_pitch():
    """Test that Basic Pitch can be loaded."""
    print("\nTesting Basic Pitch model...")
    
    try:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        
        # Try to load the model
        model = Model(ICASSP_2022_MODEL_PATH)
        print("  ✓ Basic Pitch model loaded successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load Basic Pitch model: {e}")
        return False


def test_components():
    """Test that our custom components can be imported."""
    print("\nTesting pipeline components...")
    
    try:
        from basic_pitch_wrapper import BasicPitchGuitarWrapper
        print("  ✓ BasicPitchGuitarWrapper")
    except ImportError as e:
        print(f"  ✗ BasicPitchGuitarWrapper: {e}")
        return False
    
    try:
        from basic_pitch_loader import BasicPitchLoader
        print("  ✓ BasicPitchLoader")
    except ImportError as e:
        print(f"  ✗ BasicPitchLoader: {e}")
        return False
    
    try:
        from evaluation_metrics import BasicPitchEvaluator
        print("  ✓ BasicPitchEvaluator")
    except ImportError as e:
        print(f"  ✗ BasicPitchEvaluator: {e}")
        return False
    
    return True


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"    {i}: {gpu.name}")
        else:
            print("  ⚠ No GPUs found - will use CPU")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Basic Pitch Evaluation Pipeline Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config, 
        test_basic_pitch,
        test_components,
        test_gpu
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n✅ All tests passed! Your pipeline is ready to run.")
        print("\nNext steps:")
        print("1. Set your audio/MIDI directory paths in config.yaml")
        print("2. Run: python main.py --config config.yaml --max-files 1")
        print("3. Check the results, then run on your full dataset")
    else:
        print(f"\n❌ {len(tests) - passed} tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()