"""
Basic Pitch wrapper for guitar transcription evaluation on GuitarSet dataset.
Uses Spotify's official Basic Pitch implementation with TensorFlow backend.
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

import numpy as np
import tensorflow as tf

# Basic Pitch imports
from basic_pitch.inference import predict, predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicPitchWrapper:
    """
    Wrapper for Basic Pitch focused on guitar transcription evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Basic Pitch wrapper with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
        
        # Basic Pitch parameters from config
        self.onset_threshold = self.config['basic_pitch']['onset_threshold']
        self.frame_threshold = self.config['basic_pitch']['frame_threshold']
        self.minimum_note_length = self.config['basic_pitch']['minimum_note_length']
        #self.minimum_frequency = self.config['basic_pitch']['minimum_frequency']
        #self.maximum_frequency = self.config['basic_pitch']['maximum_frequency']
        self.midi_tempo = self.config['basic_pitch']['midi_tempo']
        
        # Processing settings
        self.save_predictions = self.config['processing']['save_predictions']
        self.save_model_outputs = self.config['processing']['save_model_outputs']
        self.verbose = self.config['processing']['verbose']
        
        # Output directory
        self.sr = self.config['paths']['sr']
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Pre-load model for efficiency
        self.model = Model(ICASSP_2022_MODEL_PATH)
        
        if self.verbose:
            logger.info("✓ Basic Pitch Guitar Wrapper initialized")
            logger.info(f"  Onset threshold: {self.onset_threshold}")
            logger.info(f"  Frame threshold: {self.frame_threshold}")
            #logger.info(f"  Frequency range: {self.minimum_frequency}-{self.maximum_frequency} Hz")
            logger.info(f"  Output directory: {self.output_dir}")
    
    
    
    def transcribe_single(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe a single audio file using Basic Pitch.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing:
            - model_output: Raw model probabilities
            - midi_data: PrettyMIDI object
            - note_events: List of note events
            - processing_time: Time taken
            - output_files: Paths to saved outputs
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.verbose:
            logger.info(f"Transcribing: {audio_path.name}")
        
        start_time = time.time()
        
        # Create output directory for this file
        file_output_dir = self.output_dir / audio_path.stem
        file_output_dir.mkdir(exist_ok=True)
        
    
        # Run Basic Pitch prediction
        model_output, midi_data, note_events = predict(
            audio_path,
            model_or_model_path=self.model,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold
        )
        
        processing_time = time.time() - start_time
        
        output_files = {}
        
        # Save MIDI file
        if self.save_predictions:
            midi_path = file_output_dir / f"{audio_path.stem}_basic_pitch.mid"
            midi_data.write(str(midi_path))
            output_files['midi'] = str(midi_path)
        
        # Save model outputs (raw probabilities)
        if self.save_model_outputs:
            npz_path = file_output_dir / f"{audio_path.stem}_model_output.npz"
            np.savez(npz_path, **model_output)
            output_files['model_output'] = str(npz_path)
        
        # Save note events as text
        if self.save_predictions:
            notes_path = file_output_dir / f"{audio_path.stem}_note_events.txt"
            self._save_note_events(note_events, notes_path)
            output_files['note_events'] = str(notes_path)
        
        if self.verbose:
            num_notes = len(note_events)
            logger.info(f"  ✓ Processed in {processing_time:.2f}s")
            logger.info(f"  ✓ Detected {num_notes} notes")
            if output_files:
                logger.info(f"  ✓ Saved outputs to: {file_output_dir}")
        
        return {
            'model_output': model_output,
            'midi_data': midi_data,
            'note_events': note_events,
            'processing_time': processing_time,
            'output_files': output_files,
            'audio_path': str(audio_path)
        }
            
        
    
    def transcribe_batch(self, audio_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            Dictionary mapping audio paths to transcription results
        """
        if self.verbose:
            logger.info(f"Batch transcribing {len(audio_paths)} files...")
        
        results = {}
        total_start_time = time.time()
        
        for i, audio_path in enumerate(audio_paths):
            try:
                if self.verbose:
                    logger.info(f"Processing {i+1}/{len(audio_paths)}: {Path(audio_path).name}")
                
                result = self.transcribe_single(audio_path)
                results[audio_path] = result
                
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results[audio_path] = {
                    'error': str(e),
                    'processing_time': 0,
                    'audio_path': audio_path
                }
        
        total_time = time.time() - total_start_time
        successful = len([r for r in results.values() if 'error' not in r])
        
        if self.verbose:
            logger.info(f"✓ Batch processing complete")
            logger.info(f"  Successful: {successful}/{len(audio_paths)}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Average time per file: {total_time/len(audio_paths):.2f}s")
        
        return results
    
    def _save_note_events(self, note_events: List[Tuple], output_path: Path):
        """Save note events to text file."""
        with open(output_path, 'w') as f:
            f.write("# Basic Pitch Note Events\n")
            f.write("# Format: start_time, end_time, midi_note, amplitude\n")
            for start_time, end_time, midi_note, amplitude, pitch_bends in note_events:
                f.write(f"{start_time:.6f}, {end_time:.6f}, {midi_note}, {amplitude:.6f}\n")
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate audio file."""
        path = Path(audio_path)
        return (
            path.exists() and 
            path.suffix.lower() in self.get_supported_formats() and
            path.stat().st_size > 1000  # At least 1KB
        )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Example usage
    config_path = "config.yaml"
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        print("Please create config.yaml with your settings")
        exit(1)
    
    # Initialize wrapper
    wrapper = BasicPitchWrapper(config_path)
    
    # Example single file transcription
    test_audio = "/data/akshaj/MusicAI/GuitarSet/audio_mono-mic/03_BN2-131-B_comp_mic.wav"
    if Path(test_audio).exists():
        result = wrapper.transcribe_single(test_audio)
        print(f"Transcribed {result['audio_path']}")
        print(f"Detected {len(result['note_events'])} notes")
    
    print("Basic Pitch Guitar Wrapper initialized successfully!")
    print("Usage:")
    print("  wrapper = BasicPitchGuitarWrapper('config.yaml')")
    print("  result = wrapper.transcribe_single('guitar.wav')")