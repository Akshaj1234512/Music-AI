import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import soundfile as sf
import numpy as np


class PrecomputedFeaturesDataset(Dataset):
    """
    Dataset for loading precomputed Basic Pitch features alongside audio files.
    
    This dataset is designed for the dual Basic Pitch approach:
    - Loads precomputed Basic Pitch features from offline preprocessing
    - Loads corresponding audio files for other pipeline components (Encodec, CLAP)
    - Handles feature/audio alignment and validation
    """
    
    def __init__(
        self,
        audio_dir: Union[str, Path],
        features_dir: Union[str, Path],
        sample_rate: int = 22050,
        max_duration: Optional[float] = None,
        extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a'],
        validate_features: bool = True
    ):
        """
        Initialize dataset with audio and precomputed features.
        
        Args:
            audio_dir: Directory containing audio files
            features_dir: Directory containing precomputed Basic Pitch features (.pt files)
            sample_rate: Target sample rate for audio loading
            max_duration: Maximum audio duration in seconds (None for no limit)
            extensions: Supported audio file extensions
            validate_features: Whether to validate feature/audio alignment
        """
        self.audio_dir = Path(audio_dir)
        self.features_dir = Path(features_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.extensions = extensions
        
        # Load feature manifest if available
        manifest_path = self.features_dir / 'basic_pitch_features_manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            print(f"✓ Loaded feature manifest: {len(self.manifest['processed_files'])} files")
        else:
            self.manifest = None
            print("⚠ No feature manifest found, will scan directories")
        
        # Find matching audio/feature pairs
        self.samples = self._find_audio_feature_pairs()
        
        if validate_features:
            self._validate_samples()
        
        print(f"✓ Dataset initialized with {len(self.samples)} audio/feature pairs")
    
    def _find_audio_feature_pairs(self) -> List[Dict[str, Path]]:
        """Find matching audio and feature file pairs."""
        samples = []
        
        if self.manifest:
            # Use manifest for reliable pairing
            for rel_path_str in self.manifest['processed_files']:
                rel_path = Path(rel_path_str)
                audio_path = self.audio_dir / rel_path
                feature_path = self.features_dir / rel_path.with_suffix('.pt')
                
                if audio_path.exists() and feature_path.exists():
                    samples.append({
                        'audio_path': audio_path,
                        'feature_path': feature_path,
                        'relative_path': rel_path
                    })
                else:
                    print(f"⚠ Missing files for {rel_path}: audio={audio_path.exists()}, features={feature_path.exists()}")
        else:
            # Scan directories for matches
            for ext in self.extensions:
                for audio_path in self.audio_dir.rglob(f"*{ext}"):
                    rel_path = audio_path.relative_to(self.audio_dir)
                    feature_path = self.features_dir / rel_path.with_suffix('.pt')
                    
                    if feature_path.exists():
                        samples.append({
                            'audio_path': audio_path,
                            'feature_path': feature_path,
                            'relative_path': rel_path
                        })
        
        return samples
    
    def _validate_samples(self) -> None:
        """Validate a subset of samples for feature/audio alignment."""
        if not self.samples:
            return
        
        print("Validating feature/audio alignment...")
        
        # Check a few samples for validation
        validation_samples = min(3, len(self.samples))
        for i in range(validation_samples):
            try:
                sample = self.__getitem__(i)
                audio_len = sample['audio'].shape[-1]
                features_len = sample['basic_pitch_features'].shape[0]
                
                # Basic Pitch uses hop_length=512 by default
                expected_features_len = audio_len // 512 + 1
                
                if abs(features_len - expected_features_len) > 5:  # Allow some tolerance
                    print(f"⚠ Length mismatch in sample {i}: audio={audio_len}, features={features_len} (expected ~{expected_features_len})")
                
            except Exception as e:
                print(f"⚠ Validation error for sample {i}: {e}")
        
        print("✓ Sample validation complete")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load audio and precomputed Basic Pitch features.
        
        Returns:
            Dictionary containing:
            - audio: Audio waveform [time]
            - basic_pitch_features: Precomputed Basic Pitch features [time_frames, 440]
            - metadata: File information
        """
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = sf.read(sample['audio_path'])
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            
            # Resample if needed
            if sr != self.sample_rate:
                # Simple resampling (could use librosa.resample for better quality)
                audio = torch.tensor(audio, dtype=torch.float32)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=int(len(audio) * self.sample_rate / sr),
                    mode='linear',
                    align_corners=False
                ).squeeze()
                audio = audio.numpy()
            
            # Truncate if max_duration specified
            if self.max_duration:
                max_samples = int(self.max_duration * self.sample_rate)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            # Load precomputed Basic Pitch features
            basic_pitch_features = torch.load(sample['feature_path'])
            
            # Concatenate features if they're in dictionary format
            if isinstance(basic_pitch_features, dict):
                # Concatenate in correct order: onset + contour + note
                basic_pitch_tensor = torch.cat([
                    basic_pitch_features['onset'],    # [time, 88]
                    basic_pitch_features['contour'],  # [time, 264]
                    basic_pitch_features['note']      # [time, 88]
                ], dim=-1)  # [time, 440]
            else:
                # Assume already concatenated
                basic_pitch_tensor = basic_pitch_features
            
            # Ensure features are float32
            basic_pitch_tensor = basic_pitch_tensor.float()
            
            return {
                'audio': audio_tensor,
                'basic_pitch_features': basic_pitch_tensor,
                'metadata': {
                    'audio_path': str(sample['audio_path']),
                    'feature_path': str(sample['feature_path']),
                    'relative_path': str(sample['relative_path']),
                    'original_sample_rate': sr,
                    'audio_duration': len(audio_tensor) / self.sample_rate,
                    'feature_frames': basic_pitch_tensor.shape[0]
                }
            }
            
        except Exception as e:
            print(f"Error loading sample {idx} ({sample['relative_path']}): {e}")
            raise e
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading the actual data."""
        return {
            'index': idx,
            'audio_path': str(self.samples[idx]['audio_path']),
            'feature_path': str(self.samples[idx]['feature_path']),
            'relative_path': str(self.samples[idx]['relative_path'])
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching variable-length sequences.
        Pads sequences to the same length within each batch.
        """
        # Find maximum lengths in batch
        max_audio_len = max(sample['audio'].shape[0] for sample in batch)
        max_feature_len = max(sample['basic_pitch_features'].shape[0] for sample in batch)
        
        batch_size = len(batch)
        
        # Initialize padded tensors
        audio_batch = torch.zeros(batch_size, max_audio_len)
        features_batch = torch.zeros(batch_size, max_feature_len, 440)
        
        # Create padding masks
        audio_masks = torch.zeros(batch_size, max_audio_len, dtype=torch.bool)
        feature_masks = torch.zeros(batch_size, max_feature_len, dtype=torch.bool)
        
        metadata_batch = []
        
        for i, sample in enumerate(batch):
            # Audio
            audio_len = sample['audio'].shape[0]
            audio_batch[i, :audio_len] = sample['audio']
            audio_masks[i, :audio_len] = True
            
            # Features
            feature_len = sample['basic_pitch_features'].shape[0]
            features_batch[i, :feature_len] = sample['basic_pitch_features']
            feature_masks[i, :feature_len] = True
            
            # Metadata
            metadata_batch.append(sample['metadata'])
        
        return {
            'audio': audio_batch,
            'basic_pitch_features': features_batch,
            'audio_mask': audio_masks,
            'feature_mask': feature_masks,
            'metadata': metadata_batch
        }