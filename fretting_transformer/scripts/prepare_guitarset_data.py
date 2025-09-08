#!/usr/bin/env python3
"""
Prepare GuitarSet Data for Fine-tuning

This script loads GuitarSet data, creates train/validation/test splits,
and generates statistics about the dataset. It saves the splits for
reproducible fine-tuning experiments.
"""

import sys
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.guitarset_loader import GuitarSetLoader, GuitarSetExcerpt
from data.unified_tokenizer import UnifiedFrettingTokenizer


def analyze_dataset_characteristics(excerpts: List[GuitarSetExcerpt]) -> Dict:
    """Analyze characteristics of the GuitarSet dataset."""
    
    stats = {
        'total_excerpts': len(excerpts),
        'total_notes': 0,
        'by_style': {},
        'by_player': {},
        'by_mode': {},
        'by_key': {},
        'note_statistics': {
            'durations': [],
            'string_distribution': {i: 0 for i in range(6)},
            'fret_distribution': {},
            'pitch_distribution': {},
        },
        'excerpt_statistics': {
            'notes_per_excerpt': [],
            'duration_per_excerpt': []
        }
    }
    
    for excerpt in excerpts:
        # Count by categories
        stats['by_style'][excerpt.style] = stats['by_style'].get(excerpt.style, 0) + 1
        stats['by_player'][excerpt.player_id] = stats['by_player'].get(excerpt.player_id, 0) + 1
        stats['by_mode'][excerpt.mode] = stats['by_mode'].get(excerpt.mode, 0) + 1
        
        if 'key' in excerpt.metadata:
            key = excerpt.metadata['key']
            stats['by_key'][key] = stats['by_key'].get(key, 0) + 1
        
        # Note-level statistics
        excerpt_notes = len(excerpt.notes)
        stats['total_notes'] += excerpt_notes
        stats['excerpt_statistics']['notes_per_excerpt'].append(excerpt_notes)
        
        if excerpt.notes:
            excerpt_duration = excerpt.notes[-1].start_time + excerpt.notes[-1].duration - excerpt.notes[0].start_time
            stats['excerpt_statistics']['duration_per_excerpt'].append(excerpt_duration)
            
            for note in excerpt.notes:
                stats['note_statistics']['durations'].append(note.duration)
                stats['note_statistics']['string_distribution'][note.string] += 1
                stats['note_statistics']['fret_distribution'][note.fret] = stats['note_statistics']['fret_distribution'].get(note.fret, 0) + 1
                stats['note_statistics']['pitch_distribution'][note.pitch] = stats['note_statistics']['pitch_distribution'].get(note.pitch, 0) + 1
    
    # Calculate summary statistics
    import numpy as np
    
    if stats['note_statistics']['durations']:
        durations = np.array(stats['note_statistics']['durations'])
        stats['duration_stats'] = {
            'mean': float(np.mean(durations)),
            'median': float(np.median(durations)),
            'std': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations))
        }
    
    if stats['excerpt_statistics']['notes_per_excerpt']:
        notes_per_excerpt = np.array(stats['excerpt_statistics']['notes_per_excerpt'])
        stats['notes_per_excerpt_stats'] = {
            'mean': float(np.mean(notes_per_excerpt)),
            'median': float(np.median(notes_per_excerpt)),
            'std': float(np.std(notes_per_excerpt)),
            'min': int(np.min(notes_per_excerpt)),
            'max': int(np.max(notes_per_excerpt))
        }
    
    return stats


def create_balanced_splits(excerpts: List[GuitarSetExcerpt], 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          split_strategy: str = 'player',
                          seed: int = 42) -> Dict:
    """
    Create balanced train/validation/test splits.
    
    Args:
        excerpts: List of GuitarSetExcerpt objects
        train_ratio, val_ratio, test_ratio: Split ratios
        split_strategy: 'player' or 'excerpt' or 'style'
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split information
    """
    import numpy as np
    np.random.seed(seed)
    
    if split_strategy == 'player':
        # Split by players to avoid data leakage
        players = list(set(e.player_id for e in excerpts))
        np.random.shuffle(players)
        
        n_train = int(len(players) * train_ratio)
        n_val = int(len(players) * val_ratio)
        
        train_players = set(players[:n_train])
        val_players = set(players[n_train:n_train + n_val])
        test_players = set(players[n_train + n_val:])
        
        train_excerpts = [e for e in excerpts if e.player_id in train_players]
        val_excerpts = [e for e in excerpts if e.player_id in val_players]
        test_excerpts = [e for e in excerpts if e.player_id in test_players]
        
        split_info = {
            'strategy': 'player',
            'train_players': list(train_players),
            'val_players': list(val_players),
            'test_players': list(test_players)
        }
        
    elif split_strategy == 'style':
        # Split by musical style
        styles = list(set(e.style for e in excerpts))
        np.random.shuffle(styles)
        
        n_train = max(1, int(len(styles) * train_ratio))
        n_val = max(1, int(len(styles) * val_ratio))
        
        train_styles = set(styles[:n_train])
        val_styles = set(styles[n_train:n_train + n_val])
        test_styles = set(styles[n_train + n_val:])
        
        train_excerpts = [e for e in excerpts if e.style in train_styles]
        val_excerpts = [e for e in excerpts if e.style in val_styles]
        test_excerpts = [e for e in excerpts if e.style in test_styles]
        
        split_info = {
            'strategy': 'style',
            'train_styles': list(train_styles),
            'val_styles': list(val_styles),
            'test_styles': list(test_styles)
        }
        
    else:
        # Random excerpt-based split
        np.random.shuffle(excerpts)
        
        n_train = int(len(excerpts) * train_ratio)
        n_val = int(len(excerpts) * val_ratio)
        
        train_excerpts = excerpts[:n_train]
        val_excerpts = excerpts[n_train:n_train + n_val]
        test_excerpts = excerpts[n_train + n_val:]
        
        split_info = {
            'strategy': 'excerpt'
        }
    
    splits = {
        'train': train_excerpts,
        'val': val_excerpts,
        'test': test_excerpts,
        'info': split_info,
        'seed': seed
    }
    
    print(f"Created {split_strategy} splits:")
    print(f"  Train: {len(train_excerpts)} excerpts")
    print(f"  Val: {len(val_excerpts)} excerpts") 
    print(f"  Test: {len(test_excerpts)} excerpts")
    
    return splits


def save_splits_and_stats(splits: Dict, 
                         stats: Dict, 
                         output_dir: str = "/data/andreaguz/guitarset_data") -> None:
    """Save splits and statistics to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits as JSON (metadata only, not the full excerpts)
    splits_metadata = {
        'train_excerpt_ids': [e.excerpt_id for e in splits['train']],
        'val_excerpt_ids': [e.excerpt_id for e in splits['val']],
        'test_excerpt_ids': [e.excerpt_id for e in splits['test']],
        'split_info': splits['info'],
        'seed': splits['seed']
    }
    
    splits_file = os.path.join(output_dir, 'guitarset_splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits_metadata, f, indent=2)
    print(f"Saved splits metadata to: {splits_file}")
    
    # Save full splits as pickle for later use
    splits_pickle = os.path.join(output_dir, 'guitarset_splits.pkl')
    with open(splits_pickle, 'wb') as f:
        pickle.dump(splits, f)
    print(f"Saved full splits to: {splits_pickle}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'guitarset_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved statistics to: {stats_file}")


def print_split_statistics(splits: Dict, stats: Dict) -> None:
    """Print detailed statistics about the splits."""
    
    print("\n" + "="*60)
    print("GUITARSET DATASET STATISTICS")
    print("="*60)
    
    # Overall statistics
    print(f"\nOverall Dataset:")
    print(f"  Total excerpts: {stats['total_excerpts']}")
    print(f"  Total notes: {stats['total_notes']:,}")
    print(f"  Average notes per excerpt: {stats['notes_per_excerpt_stats']['mean']:.1f}")
    print(f"  Average note duration: {stats['duration_stats']['mean']:.3f}s")
    
    # By category
    print(f"\nBy Musical Style:")
    for style, count in stats['by_style'].items():
        print(f"  {style}: {count} excerpts")
    
    print(f"\nBy Player:")
    for player, count in stats['by_player'].items():
        print(f"  Player {player}: {count} excerpts")
        
    print(f"\nBy Mode:")
    for mode, count in stats['by_mode'].items():
        print(f"  {mode}: {count} excerpts")
    
    # Split statistics
    print(f"\nSplit Distribution:")
    for split_name in ['train', 'val', 'test']:
        split_excerpts = splits[split_name]
        split_notes = sum(len(e.notes) for e in split_excerpts)
        print(f"  {split_name.title()}: {len(split_excerpts)} excerpts, {split_notes:,} notes")
        
        # Style distribution within split
        split_styles = {}
        for excerpt in split_excerpts:
            split_styles[excerpt.style] = split_styles.get(excerpt.style, 0) + 1
        
        if split_styles:
            styles_str = ", ".join([f"{style}: {count}" for style, count in split_styles.items()])
            print(f"    Styles: {styles_str}")
    
    # String and fret distribution
    print(f"\nString Distribution (0=low E, 5=high e):")
    for string_num in range(6):
        count = stats['note_statistics']['string_distribution'][string_num]
        percentage = 100 * count / stats['total_notes'] if stats['total_notes'] > 0 else 0
        print(f"  String {string_num}: {count:,} notes ({percentage:.1f}%)")
    
    print(f"\nMost Common Frets:")
    fret_items = sorted(stats['note_statistics']['fret_distribution'].items(), 
                       key=lambda x: x[1], reverse=True)[:10]
    for fret, count in fret_items:
        percentage = 100 * count / stats['total_notes'] if stats['total_notes'] > 0 else 0
        print(f"  Fret {fret}: {count:,} notes ({percentage:.1f}%)")


def main():
    """Main function to prepare GuitarSet data."""
    
    print("=== Preparing GuitarSet Data for Fine-tuning ===")
    
    # Initialize GuitarSet loader
    loader = GuitarSetLoader()
    
    # Load all excerpts
    print("\nLoading all GuitarSet excerpts...")
    excerpts = loader.load_all_excerpts()
    
    if not excerpts:
        print("ERROR: No excerpts loaded from GuitarSet!")
        return
    
    print(f"Loaded {len(excerpts)} excerpts")
    
    # Analyze dataset characteristics
    print("\nAnalyzing dataset characteristics...")
    stats = analyze_dataset_characteristics(excerpts)
    
    # Create splits (try player-based first, then excerpt-based)
    print("\nCreating data splits...")
    
    # Check if we have multiple players
    players = set(e.player_id for e in excerpts)
    if len(players) >= 3:
        print("Using player-based splits to avoid data leakage")
        splits = create_balanced_splits(excerpts, split_strategy='player', 
                                   train_ratio=0.67, val_ratio=0.17, test_ratio=0.17)
    else:
        print("Using excerpt-based splits (insufficient players for player-based splits)")
        splits = create_balanced_splits(excerpts, split_strategy='excerpt')
    
    # Print detailed statistics
    print_split_statistics(splits, stats)
    
    # Save everything
    print("\nSaving splits and statistics...")
    save_splits_and_stats(splits, stats)
    
    # Test tokenization on a sample
    print("\nTesting tokenization on sample data...")
    tokenizer = UnifiedFrettingTokenizer()
    
    if splits['train']:
        sample_excerpt = splits['train'][0]
        print(f"Sample excerpt: {sample_excerpt.excerpt_id}")
        print(f"  Notes: {len(sample_excerpt.notes)}")
        print(f"  Style: {sample_excerpt.style}")
        print(f"  Mode: {sample_excerpt.mode}")
        
        # Show first few notes
        if sample_excerpt.notes:
            print("  First 3 notes:")
            for i, note in enumerate(sample_excerpt.notes[:3]):
                print(f"    {i+1}: t={note.start_time:.3f}s, pitch={note.pitch}, string={note.string}, fret={note.fret}")
    
    print("\n=== GuitarSet Data Preparation Complete ===")
    print("\nNext steps:")
    print("1. Use /data/andreaguz/guitarset_data/guitarset_splits.pkl for fine-tuning")
    print("2. Review /data/andreaguz/guitarset_data/guitarset_statistics.json for insights")
    print("3. Run fine-tuning with the prepared splits")


if __name__ == "__main__":
    main()