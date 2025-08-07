import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path

# Import the PyTorch Basic Pitch model
try:
    from .basic_pitch_torch.model import BasicPitchTorch
    PYTORCH_BASIC_PITCH_AVAILABLE = True
except ImportError:
    PYTORCH_BASIC_PITCH_AVAILABLE = False


class CleanBasicPitchWrapper(nn.Module):
    """
    Clean Basic Pitch wrapper supporting dual modes:
    
    1. Precomputed Mode: Load precomputed features from offline preprocessing
    2. Online Mode: Real-time inference using PyTorch Basic Pitch
    
    For training pipelines, use precomputed mode for best performance.
    For inference/deployment, use online mode for real-time processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        mode: str = 'precomputed',  # 'precomputed' or 'online'
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        if sample_rate != 22050:
            raise ValueError("Basic Pitch requires 22050 Hz sample rate")
        
        self.sample_rate = sample_rate
        self.device = device
        self.mode = mode
        self.output_dim = 440  # 88 onset + 264 contour + 88 note
        
        if mode == 'online':
            self._init_online_mode(model_path)
        elif mode == 'precomputed':
            print("✓ Basic Pitch wrapper initialized in precomputed mode")
            print("  Use load_precomputed_features() or provide features directly")
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'precomputed' or 'online'")
    
    def _init_online_mode(self, model_path: Optional[str]):
        """Initialize PyTorch Basic Pitch for online inference."""
        if not PYTORCH_BASIC_PITCH_AVAILABLE:
            raise ImportError(
                "PyTorch Basic Pitch not available. Please ensure basic_pitch_torch "
                "is properly installed in models/basic_pitch_torch/"
            )
        
        # Initialize the PyTorch Basic Pitch model
        self.basic_pitch_model = BasicPitchTorch()
        
        # Load pretrained weights
        if model_path is None:
            model_path = Path(__file__).parent / "basic_pitch_torch" / "basic_pitch_pytorch_icassp_2022.pth"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Basic Pitch weights not found at {model_path}")
        
        # Load pretrained weights
        state_dict = torch.load(model_path, map_location='cpu')
        self.basic_pitch_model.load_state_dict(state_dict)
        self.basic_pitch_model.to(self.device)
        self.basic_pitch_model.eval()  # Set to eval mode for inference
        
        print(f"✓ PyTorch Basic Pitch initialized for online inference")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
    
    def forward(self, audio_or_features: torch.Tensor) -> torch.Tensor:
        """
        Process audio or return precomputed features.
        
        Args:
            audio_or_features: Either raw audio [batch, time] or precomputed features [batch, time, 440]
            
        Returns:
            Basic Pitch features [batch, time_frames, 440]
        """
        if self.mode == 'precomputed':
            # Assume input is already processed features
            return audio_or_features
        
        elif self.mode == 'online':
            return self._extract_features_online(audio_or_features)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _extract_features_online(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features using PyTorch Basic Pitch."""
        # Handle batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Ensure audio is on correct device
        audio = audio.to(self.device)
        
        # Run PyTorch Basic Pitch with gradient flow
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.basic_pitch_model(audio)
            
            # Extract outputs: onset [batch, time, 88], contour [batch, time, 264], note [batch, time, 88]
            onset = outputs['onset']      # [batch, time, 88]
            contour = outputs['contour']  # [batch, time, 264]
            note = outputs['note']        # [batch, time, 88]
            
            # Concatenate features: [batch, time, 440]
            features = torch.cat([onset, contour, note], dim=-1)
        
        if squeeze_output:
            features = features.squeeze(0)
            
        return features
    
    def load_precomputed_features(self, feature_path: Union[str, Path]) -> torch.Tensor:
        """
        Load precomputed Basic Pitch features from .pt file.
        
        Args:
            feature_path: Path to .pt file with precomputed features
            
        Returns:
            Concatenated features [time, 440]
        """
        features_dict = torch.load(feature_path, map_location=self.device)
        
        # Concatenate in correct order: onset + contour + note
        features = torch.cat([
            features_dict['onset'],    # [time, 88]
            features_dict['contour'],  # [time, 264]
            features_dict['note']      # [time, 88]
        ], dim=-1)  # [time, 440]
        
        return features
    
    def extract_individual_outputs(self, audio_or_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract individual Basic Pitch outputs.
        
        Returns:
            Dictionary with onset, contour, note tensors
        """
        if self.mode == 'precomputed':
            # Split concatenated features back to components
            if audio_or_features.dim() == 3:
                # [batch, time, 440]
                onset = audio_or_features[..., :88]
                contour = audio_or_features[..., 88:352]  # 88 + 264 = 352
                note = audio_or_features[..., 352:]
            else:
                # [time, 440]
                onset = audio_or_features[..., :88]
                contour = audio_or_features[..., 88:352]
                note = audio_or_features[..., 352:]
            
            return {'onset': onset, 'contour': contour, 'note': note}
        
        elif self.mode == 'online':
            if audio_or_features.dim() == 1:
                audio_or_features = audio_or_features.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            audio_or_features = audio_or_features.to(self.device)
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.basic_pitch_model(audio_or_features)
            
            if squeeze_output:
                outputs = {k: v.squeeze(0) for k, v in outputs.items()}
                
            return outputs
    
    def switch_mode(self, mode: str):
        """Switch between precomputed and online modes."""
        if mode not in ['precomputed', 'online']:
            raise ValueError(f"Invalid mode: {mode}")
        
        if mode == 'online' and not hasattr(self, 'basic_pitch_model'):
            raise RuntimeError("Cannot switch to online mode: PyTorch Basic Pitch not initialized")
        
        self.mode = mode
        print(f"✓ Switched to {mode} mode")


# Backward compatibility - prefers precomputed mode for training
class BasicPitchFeatureExtractor(CleanBasicPitchWrapper):
    """Legacy compatibility wrapper."""
    
    def __init__(self, sample_rate: int = 22050, **kwargs):
        # Default to precomputed mode for training compatibility
        super().__init__(sample_rate=sample_rate, mode='precomputed', **kwargs)