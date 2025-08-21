"""
Basic Pitch evaluation loader for any dataset with audio and MIDI directories.
Simple, generic loader that matches audio files with corresponding MIDI files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pretty_midi
import librosa

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicPitchLoader:
    """
    Generic data loader for audio + MIDI evaluation.
    Works with any dataset that has separate audio and MIDI directories.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize loader with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate dataset paths are provided and directories exist
        audio_dir_str = self.config['paths'].get('guitarset_audio_dir', '')
        midi_dir_str = self.config['paths'].get('guitarset_midi_dir', '')
        if not audio_dir_str or not audio_dir_str.strip():
            raise ValueError("Config 'paths.guitarset_audio_dir' is empty. Please set it in config.yaml.")
        if not midi_dir_str or not midi_dir_str.strip():
            raise ValueError("Config 'paths.guitarset_midi_dir' is empty. Please set it in config.yaml.")

        self.audio_dir = Path(audio_dir_str)
        self.midi_dir = Path(midi_dir_str)
        
        # Validate directories
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found or not a directory: {self.audio_dir}")
        if not self.midi_dir.is_dir():
            raise FileNotFoundError(f"MIDI directory not found or not a directory: {self.midi_dir}")
        
        # Find and match files
        self.matched_pairs = self._find_matching_pairs()
        
        logger.info(f"✓ Basic Pitch loader initialized")
        logger.info(f"  Audio directory: {self.audio_dir}")
        logger.info(f"  MIDI directory: {self.midi_dir}")
        logger.info(f"  Matched pairs: {len(self.matched_pairs)}")
    
    def _find_matching_pairs(self) -> List[Dict[str, Any]]:
        """Find audio files with matching MIDI files."""
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
        midi_extensions = ['.mid', '.midi']
        
        pairs = []
        
        # Find all audio files
        audio_files = {}
        for ext in audio_extensions:
            for audio_path in self.audio_dir.glob(f"*{ext}"):
                stem = audio_path.stem
                audio_files[stem] = audio_path
        
        # Find all MIDI files
        midi_files = {}
        for ext in midi_extensions:
            for midi_path in self.midi_dir.glob(f"**/*{ext}"):
                midi_files[midi_path.stem] = midi_path

        # Find matching audio files
        for midi_stem, midi_path in midi_files.items():
            audio_stem = f"{midi_stem}_mic"
            if audio_stem in audio_files:
                audio_path = audio_files[audio_stem]
                pairs.append({
                    'filename': midi_stem,
                    'audio_path': audio_path,
                    'midi_path': midi_path
                })
            else:
                logger.warning(f"No matching audio found for: {midi_stem}")
        
        return pairs
    
    def get_all_pairs(self) -> List[Dict[str, Any]]:
        """Get all matched audio-MIDI pairs."""
        return self.matched_pairs
    
    def load_audio(self, audio_path: Path, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        waveform, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return waveform, sr
    
    def load_midi_ground_truth(self, midi_path: Path) -> Dict[str, Any]:
        """Load MIDI ground truth and extract note events."""
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Extract all note events
        note_events = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Skip drums
                for note in instrument.notes:
                    note_events.append((
                        note.start,      # Start time
                        note.end,        # End time  
                        note.pitch,      # MIDI pitch
                        note.velocity    # Velocity
                    ))
        
        # Sort by start time
        note_events.sort(key=lambda x: x[0])
        
        return {
            'midi_data': midi_data,
            'note_events': note_events,
            'duration': midi_data.get_end_time(),
            'num_notes': len(note_events)
        }
    
    def load_pair(self, filename: str, target_sr: int = 44100) -> Dict[str, Any]:
        """Load both audio and MIDI for a specific file."""
        # Find the pair
        pair = None
        for p in self.matched_pairs:
            if p['filename'] == filename:
                pair = p
                break
        
        if pair is None:
            raise ValueError(f"No pair found for: {filename}")
        
        # Load audio
        waveform, sr = self.load_audio(pair['audio_path'], target_sr)
        
        # Load MIDI
        midi_gt = self.load_midi_ground_truth(pair['midi_path'])
        
        return {
            'filename': filename,
            'audio_path': str(pair['audio_path']),
            'midi_path': str(pair['midi_path']),
            'waveform': waveform,
            'sample_rate': sr,
            'audio_duration': len(waveform) / sr,
            'midi_ground_truth': midi_gt
        }


if __name__ == "__main__":
    config_path = "config.yaml"
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        print("Please create config.yaml with audio and MIDI directory paths")
        exit(1)
    
    # Initialize loader
    loader = BasicPitchLoader(config_path)
    
    # Show available files
    print(f"\nFound {len(loader.matched_pairs)} audio-MIDI pairs:")
    for pair in loader.matched_pairs[:5]:  # Show first 5
        print(f"  {pair['filename']}")
    
    if len(loader.matched_pairs) > 5:
        print(f"  ... and {len(loader.matched_pairs) - 5} more")