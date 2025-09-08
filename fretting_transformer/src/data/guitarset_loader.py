"""
GuitarSet Data Loader

This module loads JAMS files from the GuitarSet dataset and extracts
MIDI note sequences with corresponding tablature (string/fret) information.
Each JAMS file contains 6 note_midi annotations (one per string) with 
precise timing and pitch information.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GuitarSetNote:
    """Represents a single guitar note from GuitarSet with timing and tablature info."""
    start_time: float    # Time in seconds
    duration: float      # Duration in seconds
    pitch: int          # MIDI note number (rounded)
    string: int         # Guitar string (0=low E, 5=high e)
    fret: int          # Fret position (0-24)
    confidence: float   # Optional confidence score


@dataclass
class GuitarSetExcerpt:
    """Represents a complete GuitarSet excerpt with metadata."""
    excerpt_id: str              # e.g., "00_BN1-129-Eb_comp"
    notes: List[GuitarSetNote]   # All notes sorted by time
    metadata: Dict               # File metadata from JAMS
    player_id: int              # Player ID (0-5)
    style: str                  # Musical style 
    mode: str                   # "comp" or "solo"
    tempo: str                  # "slow" or "fast"
    progression: str            # Chord progression type


class GuitarSetLoader:
    """
    Loads and processes JAMS files from GuitarSet dataset.
    
    Each JAMS file contains:
    - 6 note_midi annotations (one per string)
    - Comprehensive metadata and musical structure info
    - Precise timing aligned to hexaphonic pickup recordings
    """
    
    # Standard guitar tuning (MIDI note numbers for open strings)
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4 (Low to High)
    
    def __init__(self, guitarset_path: str = "/data/akshaj/MusicAI/GuitarSet"):
        self.guitarset_path = Path(guitarset_path)
        self.annotation_path = self.guitarset_path / "annotation"
        self.midi_path = self.guitarset_path / "MIDIAnnotations"
        
        # Validate paths
        if not self.annotation_path.exists():
            raise ValueError(f"GuitarSet annotation path not found: {self.annotation_path}")
        
        print(f"GuitarSet loader initialized with path: {guitarset_path}")
    
    def get_all_excerpt_ids(self) -> List[str]:
        """Get all available excerpt IDs in the dataset."""
        jams_files = list(self.annotation_path.glob("*.jams"))
        excerpt_ids = [f.stem for f in jams_files]
        return sorted(excerpt_ids)
    
    def parse_excerpt_metadata(self, excerpt_id: str) -> Dict:
        """Parse metadata from excerpt ID string."""
        # Example: "00_BN1-129-Eb_comp" -> player=0, style=BN, mode=comp, etc.
        parts = excerpt_id.split('_')
        if len(parts) < 2:
            return {'excerpt_id': excerpt_id}
        
        player_part = parts[0]
        
        metadata = {
            'excerpt_id': excerpt_id,
            'player_id': int(player_part),
        }
        
        # Handle different formats
        if len(parts) == 3:
            # Format: "00_BN1-129-Eb_comp"
            track_part = parts[1]
            mode_part = parts[2]
        elif len(parts) == 2:
            # Format: "00_BN1-129-Ebcomp" 
            track_and_mode = parts[1]
            if track_and_mode.endswith('comp'):
                track_part = track_and_mode[:-4]
                mode_part = 'comp'
            elif track_and_mode.endswith('solo'):
                track_part = track_and_mode[:-4]
                mode_part = 'solo'
            else:
                track_part = track_and_mode
                mode_part = 'unknown'
        else:
            # More complex format, reconstruct
            track_part = '_'.join(parts[1:-1])
            mode_part = parts[-1]
        
        metadata['mode'] = mode_part
        
        # Parse track info (style-progression-key)
        if '-' in track_part:
            track_parts = track_part.split('-')
            if len(track_parts) >= 3:
                style_code = track_parts[0]
                progression_num = track_parts[1] 
                key = track_parts[2]
                
                # Map style codes
                style_map = {
                    'BN': 'Bossa Nova',
                    'Ja': 'Jazz',
                    'Fu': 'Funk', 
                    'Ro': 'Rock',
                    'SS': 'Singer-Songwriter'
                }
                
                metadata.update({
                    'style': style_map.get(style_code[:2], style_code),
                    'progression_num': progression_num,
                    'key': key,
                })
        
        return metadata
    
    def load_jams_file(self, jams_file_path: str) -> GuitarSetExcerpt:
        """
        Load a JAMS file and extract all notes with tablature info.
        
        Args:
            jams_file_path: Path to the .jams file
            
        Returns:
            GuitarSetExcerpt with all notes and metadata
        """
        with open(jams_file_path, 'r') as f:
            jams_data = json.load(f)
        
        # Parse excerpt metadata
        excerpt_id = Path(jams_file_path).stem
        metadata = self.parse_excerpt_metadata(excerpt_id)
        metadata.update(jams_data.get('file_metadata', {}))
        
        # Extract notes from all string annotations
        all_notes = []
        
        # Process note_midi annotations (6 per file, one per string)
        note_midi_annotations = [
            a for a in jams_data['annotations'] 
            if a['namespace'] == 'note_midi'
        ]
        
        if len(note_midi_annotations) != 6:
            print(f"Warning: Expected 6 note_midi annotations, got {len(note_midi_annotations)} in {excerpt_id}")
        
        for annotation in note_midi_annotations:
            # Get string info from annotation metadata
            meta = annotation.get('annotation_metadata', {})
            string_index_str = meta.get('data_source', '-1')
            
            try:
                string_index = int(string_index_str)
            except (ValueError, TypeError):
                print(f"Warning: Invalid string index '{string_index_str}' in {excerpt_id}")
                continue
            
            if string_index < 0 or string_index > 5:
                print(f"Warning: Invalid string index {string_index} in {excerpt_id}")
                continue
            
            # Process each note on this string
            for note_data in annotation['data']:
                # Extract pitch and calculate fret position
                pitch_float = note_data['value']
                pitch = int(round(pitch_float))
                
                # Calculate fret from pitch and open string tuning
                open_pitch = self.STANDARD_TUNING[string_index]
                fret = max(0, pitch - open_pitch)  # Ensure non-negative fret
                
                # Extract timing info
                start_time = note_data['time']
                duration = note_data['duration']
                confidence = note_data.get('confidence', 1.0)
                
                # Create note object
                note = GuitarSetNote(
                    start_time=start_time,
                    duration=duration,
                    pitch=pitch,
                    string=string_index,
                    fret=fret,
                    confidence=confidence or 1.0
                )
                
                all_notes.append(note)
        
        # Sort by start time
        all_notes.sort(key=lambda x: x.start_time)
        
        # Create excerpt object
        excerpt = GuitarSetExcerpt(
            excerpt_id=excerpt_id,
            notes=all_notes,
            metadata=metadata,
            player_id=metadata.get('player_id', -1),
            style=metadata.get('style', 'unknown'),
            mode=metadata.get('mode', 'unknown'),
            tempo=metadata.get('tempo', 'unknown'),
            progression=metadata.get('progression_num', 'unknown')
        )
        
        return excerpt
    
    def load_excerpt_by_id(self, excerpt_id: str) -> GuitarSetExcerpt:
        """Load a specific excerpt by its ID."""
        jams_path = self.annotation_path / f"{excerpt_id}.jams"
        
        if not jams_path.exists():
            raise FileNotFoundError(f"JAMS file not found: {jams_path}")
        
        return self.load_jams_file(str(jams_path))
    
    def load_all_excerpts(self, max_excerpts: Optional[int] = None) -> List[GuitarSetExcerpt]:
        """
        Load all excerpts from the dataset.
        
        Args:
            max_excerpts: Maximum number of excerpts to load (for testing)
            
        Returns:
            List of GuitarSetExcerpt objects
        """
        excerpt_ids = self.get_all_excerpt_ids()
        
        if max_excerpts is not None:
            excerpt_ids = excerpt_ids[:max_excerpts]
            
        excerpts = []
        for i, excerpt_id in enumerate(excerpt_ids):
            try:
                excerpt = self.load_excerpt_by_id(excerpt_id)
                excerpts.append(excerpt)
                
                if (i + 1) % 50 == 0:
                    print(f"Loaded {i + 1}/{len(excerpt_ids)} excerpts")
                    
            except Exception as e:
                print(f"Error loading excerpt {excerpt_id}: {e}")
                continue
        
        print(f"Successfully loaded {len(excerpts)} excerpts from GuitarSet")
        return excerpts
    
    def get_dataset_statistics(self, excerpts: Optional[List[GuitarSetExcerpt]] = None) -> Dict:
        """Generate statistics about the loaded dataset."""
        if excerpts is None:
            excerpts = self.load_all_excerpts()
        
        stats = {
            'total_excerpts': len(excerpts),
            'total_notes': sum(len(e.notes) for e in excerpts),
            'styles': {},
            'players': {},
            'modes': {},
            'note_distribution': {i: 0 for i in range(6)},  # Notes per string
            'duration_stats': [],
            'fret_distribution': {},
        }
        
        for excerpt in excerpts:
            # Count by categories
            stats['styles'][excerpt.style] = stats['styles'].get(excerpt.style, 0) + 1
            stats['players'][excerpt.player_id] = stats['players'].get(excerpt.player_id, 0) + 1  
            stats['modes'][excerpt.mode] = stats['modes'].get(excerpt.mode, 0) + 1
            
            # Note statistics
            for note in excerpt.notes:
                stats['note_distribution'][note.string] += 1
                stats['fret_distribution'][note.fret] = stats['fret_distribution'].get(note.fret, 0) + 1
                stats['duration_stats'].append(note.duration)
        
        # Calculate duration statistics
        if stats['duration_stats']:
            durations = np.array(stats['duration_stats'])
            stats['avg_duration'] = float(np.mean(durations))
            stats['median_duration'] = float(np.median(durations))
            stats['min_duration'] = float(np.min(durations))
            stats['max_duration'] = float(np.max(durations))
        
        return stats
    
    def create_train_val_test_splits(
        self, 
        excerpts: Optional[List[GuitarSetExcerpt]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_by: str = 'excerpt',  # 'excerpt' or 'player'
        seed: int = 42
    ) -> Tuple[List[GuitarSetExcerpt], List[GuitarSetExcerpt], List[GuitarSetExcerpt]]:
        """
        Create train/validation/test splits.
        
        Args:
            excerpts: List of excerpts to split
            train_ratio, val_ratio, test_ratio: Split ratios
            split_by: 'excerpt' for random split, 'player' for player-based split
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_excerpts, val_excerpts, test_excerpts)
        """
        if excerpts is None:
            excerpts = self.load_all_excerpts()
        
        np.random.seed(seed)
        
        if split_by == 'player':
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
            
        else:
            # Random excerpt-based split
            np.random.shuffle(excerpts)
            
            n_train = int(len(excerpts) * train_ratio)
            n_val = int(len(excerpts) * val_ratio)
            
            train_excerpts = excerpts[:n_train]
            val_excerpts = excerpts[n_train:n_train + n_val]
            test_excerpts = excerpts[n_train + n_val:]
        
        print(f"Created splits: train={len(train_excerpts)}, val={len(val_excerpts)}, test={len(test_excerpts)}")
        
        return train_excerpts, val_excerpts, test_excerpts


def test_guitarset_loader():
    """Test the GuitarSet loader functionality."""
    print("=== Testing GuitarSet Loader ===")
    
    # Initialize loader
    loader = GuitarSetLoader()
    
    # Test loading single excerpt
    excerpt_ids = loader.get_all_excerpt_ids()
    print(f"Found {len(excerpt_ids)} excerpts in dataset")
    print(f"Sample excerpt IDs: {excerpt_ids[:5]}")
    
    if excerpt_ids:
        # Load first excerpt
        excerpt = loader.load_excerpt_by_id(excerpt_ids[0])
        print(f"\nLoaded excerpt: {excerpt.excerpt_id}")
        print(f"  Notes: {len(excerpt.notes)}")
        print(f"  Style: {excerpt.style}")
        print(f"  Mode: {excerpt.mode}")
        print(f"  Player: {excerpt.player_id}")
        
        # Show first few notes
        print(f"  First 3 notes:")
        for i, note in enumerate(excerpt.notes[:3]):
            print(f"    {i}: t={note.start_time:.3f}s, pitch={note.pitch}, string={note.string}, fret={note.fret}")
    
    # Test loading multiple excerpts
    print(f"\nTesting batch loading (first 5 excerpts)...")
    excerpts = loader.load_all_excerpts(max_excerpts=5)
    
    # Generate statistics
    stats = loader.get_dataset_statistics(excerpts)
    print(f"\nDataset Statistics:")
    print(f"  Total excerpts: {stats['total_excerpts']}")
    print(f"  Total notes: {stats['total_notes']}")
    print(f"  Styles: {stats['styles']}")
    print(f"  Players: {stats['players']}")
    print(f"  Avg note duration: {stats['avg_duration']:.3f}s")
    
    # Test splits
    train, val, test = loader.create_train_val_test_splits(excerpts)
    print(f"\nSplits: train={len(train)}, val={len(val)}, test={len(test)}")
    
    print("=== GuitarSet Loader Test Completed ===")


if __name__ == "__main__":
    test_guitarset_loader()