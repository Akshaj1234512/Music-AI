#!/usr/bin/env python3
"""
Data Preparation Script for Fretting Transformer

Preprocesses SynthTab dataset and creates cached tokenized sequences for training.
This script can be run independently to prepare data before training.
"""

import os
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.synthtab_loader import SynthTabLoader
from data.tokenizer import FrettingTokenizer, TokenConfig
from data.dataset import FrettingDataProcessor, DataConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare SynthTab data for training')
    
    # Data arguments
    parser.add_argument('--synthtab_path', type=str, 
                       default='/data/andreaguz/SynthTab_Dev',
                       help='Path to SynthTab dataset')
    parser.add_argument('--data_category', type=str, default='jams',
                       choices=['jams', 'acoustic'],
                       help='Which data category to process')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--cache_name', type=str, default='synthtab_cache.pkl',
                       help='Name of cache file')
    
    # Processing arguments
    parser.add_argument('--max_sequence_length', type=int, default=512,
                       help='Maximum sequence length for training')
    parser.add_argument('--min_notes_per_song', type=int, default=5,
                       help='Minimum notes per song to include')
    parser.add_argument('--max_notes_per_song', type=int, default=1000,
                       help='Maximum notes per song to include')
    
    # Vocabulary arguments
    parser.add_argument('--save_tokenizer', action='store_true',
                       help='Save tokenizer vocabulary')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer.json',
                       help='Tokenizer file name')
    
    # Analysis arguments
    parser.add_argument('--analyze_data', action='store_true',
                       help='Run data analysis and create statistics')
    
    return parser.parse_args()


def analyze_dataset(loader: SynthTabLoader, category: str, max_files: int = None):
    """Analyze dataset and print statistics."""
    print("=== Dataset Analysis ===")
    
    # Find all JAMS files
    jams_files = loader.find_jams_files(category)
    if max_files:
        jams_files = jams_files[:max_files]
    
    print(f"Found {len(jams_files)} JAMS files in {category}")
    
    # Analyze a sample of files
    sample_size = min(100, len(jams_files))
    sample_files = jams_files[:sample_size]
    
    total_notes = 0
    note_counts = []
    pitch_distribution = {}
    string_distribution = {i: 0 for i in range(1, 7)}
    fret_distribution = {}
    
    print(f"Analyzing {sample_size} files for statistics...")
    
    for jams_file in sample_files:
        try:
            notes = loader.load_jams_file(jams_file)
            note_count = len(notes)
            note_counts.append(note_count)
            total_notes += note_count
            
            for note in notes:
                # Pitch distribution
                pitch_distribution[note.pitch] = pitch_distribution.get(note.pitch, 0) + 1
                
                # String distribution
                string_distribution[note.string] += 1
                
                # Fret distribution
                fret_distribution[note.fret] = fret_distribution.get(note.fret, 0) + 1
                
        except Exception as e:
            print(f"Failed to analyze {jams_file}: {e}")
    
    # Print statistics
    print(f"\nDataset Statistics (from {sample_size} files):")
    print(f"  Total notes analyzed: {total_notes:,}")
    print(f"  Average notes per song: {total_notes / sample_size:.1f}")
    print(f"  Min notes per song: {min(note_counts) if note_counts else 0}")
    print(f"  Max notes per song: {max(note_counts) if note_counts else 0}")
    
    print(f"\nPitch Range:")
    if pitch_distribution:
        min_pitch = min(pitch_distribution.keys())
        max_pitch = max(pitch_distribution.keys())
        print(f"  MIDI pitch range: {min_pitch} - {max_pitch}")
        
        # Most common pitches
        sorted_pitches = sorted(pitch_distribution.items(), key=lambda x: x[1], reverse=True)
        print(f"  Most common pitches:")
        for pitch, count in sorted_pitches[:10]:
            print(f"    MIDI {pitch}: {count} occurrences")
    
    print(f"\nString Usage:")
    for string, count in string_distribution.items():
        percentage = (count / total_notes) * 100 if total_notes > 0 else 0
        print(f"  String {string}: {count} notes ({percentage:.1f}%)")
    
    print(f"\nFret Usage:")
    if fret_distribution:
        sorted_frets = sorted(fret_distribution.items())
        print(f"  Fret range: {min(fret_distribution.keys())} - {max(fret_distribution.keys())}")
        print(f"  Most common frets:")
        sorted_frets_by_count = sorted(fret_distribution.items(), key=lambda x: x[1], reverse=True)
        for fret, count in sorted_frets_by_count[:15]:
            percentage = (count / total_notes) * 100 if total_notes > 0 else 0
            print(f"    Fret {fret}: {count} notes ({percentage:.1f}%)")


def prepare_data(args):
    """Prepare and process data."""
    print("=== Data Preparation ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data processor
    data_config = DataConfig(
        max_sequence_length=args.max_sequence_length,
        min_notes_per_song=args.min_notes_per_song,
        max_notes_per_song=args.max_notes_per_song
    )
    
    processor = FrettingDataProcessor(
        synthtab_path=args.synthtab_path,
        data_config=data_config
    )
    
    # Process data
    cache_path = os.path.join(args.output_dir, args.cache_name)
    
    processor.load_and_process_data(
        category=args.data_category,
        max_files=args.max_files,
        cache_path=cache_path
    )
    
    print(f"Processed {len(processor.processed_sequences)} sequences")
    print(f"Data cached to: {cache_path}")
    
    # Save tokenizer if requested
    if args.save_tokenizer:
        tokenizer_path = os.path.join(args.output_dir, args.tokenizer_name)
        processor.tokenizer.save_vocab(tokenizer_path)
        print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Create data splits and save metadata
    train_dataset, val_dataset, test_dataset = processor.create_data_splits()
    
    # Save split information
    split_info = {
        'total_sequences': len(processor.processed_sequences),
        'train_sequences': len(train_dataset),
        'val_sequences': len(val_dataset),
        'test_sequences': len(test_dataset),
        'input_vocab_size': len(processor.tokenizer.input_vocab),
        'output_vocab_size': len(processor.tokenizer.output_vocab),
        'data_config': {
            'max_sequence_length': data_config.max_sequence_length,
            'min_notes_per_song': data_config.min_notes_per_song,
            'max_notes_per_song': data_config.max_notes_per_song,
            'train_split': data_config.train_split,
            'val_split': data_config.val_split,
            'test_split': data_config.test_split
        },
        'processing_args': vars(args)
    }
    
    split_info_path = os.path.join(args.output_dir, 'dataset_info.json')
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Dataset info saved to: {split_info_path}")
    
    return processor


def main():
    """Main data preparation function."""
    args = parse_args()
    
    print("=== SynthTab Data Preparation ===")
    print(f"SynthTab path: {args.synthtab_path}")
    print(f"Data category: {args.data_category}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Create loader for analysis
        loader = SynthTabLoader(args.synthtab_path)
        
        # Run analysis if requested
        if args.analyze_data:
            analyze_dataset(loader, args.data_category, args.max_files)
        
        # Prepare data
        processor = prepare_data(args)
        
        # Print summary
        print("\n=== Preparation Summary ===")
        print(f"✓ Processed {len(processor.processed_sequences)} training sequences")
        print(f"✓ Input vocabulary size: {len(processor.tokenizer.input_vocab)}")
        print(f"✓ Output vocabulary size: {len(processor.tokenizer.output_vocab)}")
        print(f"✓ Cache saved to: {os.path.join(args.output_dir, args.cache_name)}")
        
        if args.save_tokenizer:
            print(f"✓ Tokenizer saved to: {os.path.join(args.output_dir, args.tokenizer_name)}")
        
        print("Data preparation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)