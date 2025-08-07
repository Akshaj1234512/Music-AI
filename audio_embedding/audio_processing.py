import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Optional, Tuple, Union
import warnings

try:
    import nnAudio.features
    NNAUDIO_AVAILABLE = True
except ImportError:
    NNAUDIO_AVAILABLE = False
    warnings.warn("nnAudio not available. Install with: pip install nnAudio")


class AudioProcessor:
    """
    Central audio processing utilities for the guitar embedding pipeline.
    
    Handles:
    - Audio loading and resampling
    - Normalization
    - CQT computation (for Basic Pitch)
    - Mel-spectrogram computation (for CLAP)
    - STFT and other transforms
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.sample_rate = sample_rate
        self.device = device
        
        # Initialize transforms
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize audio transforms."""
        # Mel-spectrogram for CLAP
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            f_min=50,
            f_max=11025
        ).to(self.device)
        
        # CQT for Basic Pitch (using nnAudio if available)
        if NNAUDIO_AVAILABLE:
            self.cqt_transform = nnAudio.features.CQT(
                sr=self.sample_rate,
                hop_length=512,
                fmin=32.7,  # C1
                n_bins=216,  # 6 octaves × 36 bins
                bins_per_octave=36,
                window='hann',
                center=True,
                pad_mode='constant'
            ).to(self.device)
        else:
            self.cqt_transform = None
    
    def load_audio(
        self,
        audio_path: str,
        start_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> torch.Tensor:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds (optional)
            duration: Duration in seconds (optional)
            
        Returns:
            Audio tensor [channels, time]
        """
        # Calculate frame offset and num_frames if time is specified
        frame_offset = int(start_time * self.sample_rate) if start_time else 0
        num_frames = int(duration * self.sample_rate) if duration else -1
        
        # Load audio
        waveform, sr = torchaudio.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=num_frames if num_frames > 0 else None
        )
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def normalize_audio(
        self,
        audio: torch.Tensor,
        method: str = 'peak',
        target_level: float = 0.9
    ) -> torch.Tensor:
        """
        Normalize audio using various methods.
        
        Args:
            audio: Audio tensor
            method: 'peak', 'rms', or 'lufs'
            target_level: Target normalization level
            
        Returns:
            Normalized audio
        """
        if method == 'peak':
            # Peak normalization
            max_val = torch.abs(audio).max()
            if max_val > 0:
                audio = audio * (target_level / max_val)
        
        elif method == 'rms':
            # RMS normalization
            rms = torch.sqrt(torch.mean(audio ** 2))
            if rms > 0:
                audio = audio * (target_level / rms)
        
        elif method == 'lufs':
            # Simplified LUFS (would need proper implementation)
            # For now, fallback to RMS
            return self.normalize_audio(audio, method='rms', target_level=target_level)
        
        return audio
    
    def compute_cqt(
        self,
        audio: torch.Tensor,
        n_bins: int = 216,
        bins_per_octave: int = 36,
        fmin: float = 32.7,
        hop_length: int = 512
    ) -> torch.Tensor:
        """
        Compute Constant-Q Transform.
        
        Args:
            audio: Audio tensor [batch, time] or [time]
            n_bins: Number of frequency bins
            bins_per_octave: Frequency resolution
            fmin: Minimum frequency
            hop_length: Hop length in samples
            
        Returns:
            CQT magnitude [batch, freq_bins, time_frames]
        """
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        if self.cqt_transform is not None and NNAUDIO_AVAILABLE:
            # Use nnAudio (GPU-accelerated)
            cqt_complex = self.cqt_transform(audio)
            cqt_mag = torch.abs(cqt_complex)
        else:
            # Fallback to STFT-based approximation
            # This is a simplified version - for production use proper CQT
            stft = torch.stft(
                audio,
                n_fft=2048,
                hop_length=hop_length,
                window=torch.hann_window(2048).to(audio.device),
                return_complex=True
            )
            
            # Convert to magnitude and interpolate to CQT bins
            stft_mag = torch.abs(stft)
            
            # Interpolate to approximate CQT frequency bins
            # This is a rough approximation!
            freq_bins = stft_mag.shape[1]
            cqt_mag = F.interpolate(
                stft_mag.unsqueeze(1),
                size=(n_bins, stft_mag.shape[-1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        return cqt_mag
    
    def compute_harmonic_cqt(
        self,
        audio: torch.Tensor,
        n_harmonics: int = 8,
        **cqt_kwargs
    ) -> torch.Tensor:
        """
        Compute harmonic CQT stack for better pitch detection.
        
        Args:
            audio: Audio tensor [batch, time]
            n_harmonics: Number of harmonics to stack
            **cqt_kwargs: Arguments for compute_cqt
            
        Returns:
            Harmonic CQT [batch, n_harmonics, freq_bins, time_frames]
        """
        # Compute base CQT
        cqt = self.compute_cqt(audio, **cqt_kwargs)
        batch_size, freq_bins, time_frames = cqt.shape
        
        # Initialize harmonic stack
        harmonic_cqt = torch.zeros(
            batch_size, n_harmonics, freq_bins, time_frames,
            device=audio.device
        )
        
        # First harmonic is original
        harmonic_cqt[:, 0] = cqt
        
        # Compute harmonic shifts
        bins_per_octave = cqt_kwargs.get('bins_per_octave', 36)
        
        for h in range(1, n_harmonics):
            # Calculate shift in bins for this harmonic
            # h+1 because h=0 is fundamental
            octave_shift = np.log2(h + 1)
            bin_shift = int(octave_shift * bins_per_octave)
            
            if bin_shift < freq_bins:
                # Shift the CQT up by the harmonic interval
                harmonic_cqt[:, h, :-bin_shift] = cqt[:, bin_shift:]
                
                # Apply harmonic weighting (higher harmonics are usually weaker)
                harmonic_cqt[:, h] *= 1.0 / (h + 1)
        
        return harmonic_cqt
    
    def compute_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 320
    ) -> torch.Tensor:
        """
        Compute mel-spectrogram for CLAP encoder.
        
        Args:
            audio: Audio tensor [batch, time]
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
            
        Returns:
            Log mel-spectrogram [batch, n_mels, time_frames]
        """
        # Ensure batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        return log_mel_spec
    
    def chunk_audio(
        self,
        audio: torch.Tensor,
        chunk_size: int,
        overlap: int = 0
    ) -> torch.Tensor:
        """
        Split audio into fixed-size chunks with optional overlap.
        
        Args:
            audio: Audio tensor [batch, time]
            chunk_size: Size of each chunk in samples
            overlap: Overlap between chunks in samples
            
        Returns:
            Chunked audio [batch, n_chunks, chunk_size]
        """
        batch_size, total_length = audio.shape
        stride = chunk_size - overlap
        
        # Calculate number of chunks
        n_chunks = (total_length - chunk_size) // stride + 1
        
        # Create chunks
        chunks = []
        for i in range(n_chunks):
            start = i * stride
            end = start + chunk_size
            
            if end <= total_length:
                chunk = audio[:, start:end]
            else:
                # Pad last chunk if necessary
                chunk = F.pad(audio[:, start:], (0, end - total_length))
            
            chunks.append(chunk)
        
        # Stack chunks
        chunked = torch.stack(chunks, dim=1)
        
        return chunked


# Convenience functions
def load_and_preprocess_audio(
    audio_path: str,
    sample_rate: int = 22050,
    normalize: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Convenience function to load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        normalize: Whether to normalize audio
        device: Device to load audio to
        
    Returns:
        Preprocessed audio tensor [1, time]
    """
    processor = AudioProcessor(sample_rate=sample_rate, device=device)
    
    # Load audio
    audio = processor.load_audio(audio_path)
    
    # Normalize if requested
    if normalize:
        audio = processor.normalize_audio(audio, method='peak')
    
    # Move to device
    audio = audio.to(device)
    
    # Remove channel dimension for pipeline compatibility
    audio = audio.squeeze(0)
    
    return audio.unsqueeze(0)  # Add batch dimension