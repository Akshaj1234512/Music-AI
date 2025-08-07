#!/usr/bin/env python3
"""
Offline Basic Pitch Feature Extraction

This script uses the official Spotify Basic Pitch (TensorFlow) to preprocess 
audio files and extract pitch features offline. The extracted features are 
saved as PyTorch tensors for use in the training pipeline.

Usage:
    python extract_basic_pitch_features.py --input_dir /path/to/audio --output_dir /path/to/features
    
Features Extracted:
    - onset: Note onset probabilities [time, 88]
    - contour: Pitch contour information [time, 264]  
    - note: Note activation probabilities [time, 88]
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List
import soundfile as sf
from tqdm import tqdm
import json

# Official Spotify Basic Pitch
import basic_pitch.inference as bp
from basic_pitch import ICASSP_2022_MODEL_PATH


def extract_basic_pitch_features(audio_path: str) -> Dict[str, np.ndarray]:
    """
    Extract Basic Pitch features from audio file using official Spotify model.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with onset, contour, note features
    """
    try:
        # Use official Basic Pitch inference
        model_output, midi_data, note_events = bp.predict(audio_path)
        
        # Extract the three main feature types
        features = {
            'onset': model_output['onset'],      # [time, 88] - Note onsets
            'contour': model_output['contour'],  # [time, 264] - Pitch contour
            'note': model_output['note']         # [time, 88] - Note activations
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def save_features(features: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save features as PyTorch tensors."""
    # Convert numpy arrays to PyTorch tensors
    torch_features = {
        key: torch.from_numpy(value).float() 
        for key, value in features.items()
    }
    
    # Save as .pt file
    torch.save(torch_features, output_path)


def create_feature_manifest(input_dir: Path, output_dir: Path, processed_files: List[str]) -> None:
    """Create a manifest file listing all processed features."""
    manifest = {
        'source_audio_dir': str(input_dir),
        'feature_output_dir': str(output_dir),
        'total_files': len(processed_files),
        'processed_files': processed_files,
        'feature_format': {
            'onset': '[time, 88] - Note onset probabilities',
            'contour': '[time, 264] - Pitch contour information',
            'note': '[time, 88] - Note activation probabilities'
        },
        'basic_pitch_version': 'Official Spotify Basic Pitch (TensorFlow)',
        'model_path': ICASSP_2022_MODEL_PATH
    }
    
    manifest_path = output_dir / 'basic_pitch_features_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Feature manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract Basic Pitch features offline')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save extracted features')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.wav', '.mp3', '.flac', '.m4a'],
                       help='Audio file extensions to process')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing feature files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all audio files
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    processed_files = []
    failed_files = []
    
    for audio_path in tqdm(audio_files, desc="Extracting Basic Pitch features"):
        # Generate output filename
        relative_path = audio_path.relative_to(input_dir)
        feature_filename = relative_path.with_suffix('.pt').name
        feature_path = output_dir / feature_filename
        
        # Skip if already exists (unless overwrite)
        if feature_path.exists() and not args.overwrite:
            print(f"Skipping {audio_path.name} (feature file exists)")
            processed_files.append(str(relative_path))
            continue
        
        # Extract features
        features = extract_basic_pitch_features(str(audio_path))
        
        if features is not None:
            # Save features
            save_features(features, feature_path)
            processed_files.append(str(relative_path))
            
            # Print feature info
            onset_shape = features['onset'].shape
            contour_shape = features['contour'].shape
            note_shape = features['note'].shape
            
            print(f"✓ {audio_path.name}: onset{onset_shape}, contour{contour_shape}, note{note_shape}")
        else:
            failed_files.append(str(relative_path))
    
    # Create manifest
    create_feature_manifest(input_dir, output_dir, processed_files)
    
    # Summary
    print(f"\n=== Basic Pitch Feature Extraction Complete ===")
    print(f"Total files processed: {len(processed_files)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Features saved to: {output_dir}")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")


if __name__ == "__main__":
    main()