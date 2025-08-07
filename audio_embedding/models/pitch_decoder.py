"""
Frame-to-note post-processing utilities for guitar transcription.

This module contains only the post-processing components that are still needed
after the Kena AI architectural correction. The pitch decoders have been 
integrated directly into the KenaVQVAE system.

DEPRECATED COMPONENTS (kept for backwards compatibility):
- KenaStylePitchDecoder: Integrated into KenaVQVAE
- GuitarPitchModel: Replaced by direct Kena predictions
- KenaDualLoss: Moved to vq_vae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PitchDetection:
    """Frame-level pitch detection output."""
    pitch_probs: torch.Tensor  # [batch, time, 88] pitch probabilities
    onset_probs: torch.Tensor  # [batch, time, 88] onset probabilities  
    frame_probs: torch.Tensor  # [batch, time, 88] frame activations
    confidence: torch.Tensor    # [batch, time] overall confidence


class FrameToNotePP:
    """
    Frame-to-note post-processing following contemporary approaches.
    Converts frame-level predictions to discrete note events.
    
    This is the main component still needed from this module, as it processes
    the output from Kena VQ-VAE's direct predictions.
    """
    
    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.5,
        min_note_length: int = 3,  # frames
        min_gap_length: int = 2,   # frames
        onset_delta: float = 0.2   # onset must be this much higher than previous frame
    ):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_length = min_note_length
        self.min_gap_length = min_gap_length
        self.onset_delta = onset_delta
    
    def __call__(
        self,
        onset_probs: torch.Tensor,
        frame_probs: torch.Tensor,
        pitch_range: Tuple[int, int] = (21, 109)  # A0 to C8
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Convert frame predictions to note events.
        
        Args:
            onset_probs: [batch, time, pitch]
            frame_probs: [batch, time, pitch]
            pitch_range: MIDI pitch range
            
        Returns:
            List of notes per batch: [(pitch, onset_frame, offset_frame), ...]
        """
        batch_notes = []
        batch_size = onset_probs.shape[0]
        
        for b in range(batch_size):
            notes = []
            
            # Process each pitch independently
            for pitch_idx in range(onset_probs.shape[2]):
                pitch_onsets = onset_probs[b, :, pitch_idx].cpu().numpy()
                pitch_frames = frame_probs[b, :, pitch_idx].cpu().numpy()
                
                # Find note segments
                note_segments = self._extract_notes(pitch_onsets, pitch_frames)
                
                # Convert to MIDI pitch
                midi_pitch = pitch_range[0] + pitch_idx
                
                for onset, offset in note_segments:
                    notes.append((midi_pitch, onset, offset))
            
            # Sort by onset time
            notes.sort(key=lambda x: x[1])
            batch_notes.append(notes)
        
        return batch_notes
    
    def _extract_notes(
        self,
        onset_probs: np.ndarray,
        frame_probs: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Extract note segments from a single pitch track."""
        notes = []
        in_note = False
        onset_frame = 0
        
        for i in range(len(onset_probs)):
            # Check for onset
            if not in_note:
                # Need both onset and frame activation
                if onset_probs[i] > self.onset_threshold and frame_probs[i] > self.frame_threshold:
                    # Additional check: onset should be peak
                    if i == 0 or onset_probs[i] > onset_probs[i-1] + self.onset_delta:
                        in_note = True
                        onset_frame = i
            
            # Check for offset
            else:
                if frame_probs[i] < self.frame_threshold:
                    # Check if gap is long enough
                    gap_length = 1
                    for j in range(i+1, min(i+self.min_gap_length, len(frame_probs))):
                        if frame_probs[j] < self.frame_threshold:
                            gap_length += 1
                        else:
                            break
                    
                    if gap_length >= self.min_gap_length:
                        # End note if it's long enough
                        if i - onset_frame >= self.min_note_length:
                            notes.append((onset_frame, i))
                        in_note = False
        
        # Close final note if needed
        if in_note and len(frame_probs) - onset_frame >= self.min_note_length:
            notes.append((onset_frame, len(frame_probs)))
        
        return notes


# DEPRECATED CLASSES - Kept for backwards compatibility only
# These have been replaced by KenaVQVAE's integrated architecture

class KenaStylePitchDecoder(nn.Module):
    """
    DEPRECATED: This functionality is now integrated into KenaVQVAE.
    Use KenaVQVAE directly instead.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("WARNING: KenaStylePitchDecoder is deprecated. Use KenaVQVAE directly.")
        
        # Minimal implementation for compatibility
        self.embedding_dim = kwargs.get('embedding_dim', 768)
        self.n_pitches = kwargs.get('n_pitches', 88)
        
        self.dummy_layer = nn.Linear(self.embedding_dim, self.n_pitches * 2)
    
    def forward(self, embeddings):
        print("WARNING: Using deprecated KenaStylePitchDecoder. Please use KenaVQVAE instead.")
        
        # Dummy output for compatibility
        batch, time, _ = embeddings.shape
        logits = self.dummy_layer(embeddings)
        onset_logits = logits[:, :, :self.n_pitches]
        frame_logits = logits[:, :, self.n_pitches:]
        
        return {
            'onset_logits': onset_logits,
            'frame_logits': frame_logits,
            'onset_probs': torch.sigmoid(onset_logits),
            'frame_probs': torch.sigmoid(frame_logits),
            'confidence': torch.ones(batch, time)
        }


class GuitarPitchModel(nn.Module):
    """
    DEPRECATED: This functionality is now handled by GuitarTranscriptionSystem
    using KenaVQVAE's direct predictions.
    """
    
    def __init__(self, embedding_pipeline, freeze_embeddings=True):
        super().__init__()
        print("WARNING: GuitarPitchModel is deprecated. Use GuitarTranscriptionSystem instead.")
        
        self.embedding_pipeline = embedding_pipeline
        self.post_processor = FrameToNotePP()
    
    def forward(self, audio):
        print("WARNING: Using deprecated GuitarPitchModel. Please use GuitarTranscriptionSystem instead.")
        
        # Try to use Kena's direct predictions if available
        outputs = self.embedding_pipeline(audio)
        
        if 'kena_onset_probs' in outputs:
            return {
                'onset_probs': outputs['kena_onset_probs'],
                'frame_probs': outputs['kena_frame_probs'],
                'confidence': torch.ones_like(outputs['kena_onset_probs'][:, :, 0])
            }
        else:
            # Fallback to dummy output
            batch, time_frames = audio.shape[0], outputs['embeddings'].shape[1]
            return {
                'onset_probs': torch.zeros(batch, time_frames, 88),
                'frame_probs': torch.zeros(batch, time_frames, 88),
                'confidence': torch.zeros(batch, time_frames)
            }
    
    def transcribe(self, audio, return_confidence=False):
        outputs = self.forward(audio)
        notes = self.post_processor(outputs['onset_probs'], outputs['frame_probs'])
        
        # Convert to time
        hop_length = 512
        sample_rate = 22050
        frame_time = hop_length / sample_rate
        
        timed_notes = []
        for batch_notes in notes:
            batch_timed = []
            for pitch, onset_frame, offset_frame in batch_notes:
                onset_time = onset_frame * frame_time
                offset_time = offset_frame * frame_time
                batch_timed.append((pitch, onset_time, offset_time))
            timed_notes.append(batch_timed)
        
        return timed_notes


class KenaDualLoss(nn.Module):
    """
    DEPRECATED: This class has been moved to vq_vae.py as the proper location.
    Import from vq_vae instead.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("WARNING: KenaDualLoss has moved to vq_vae.py. Please import from there instead.")
        
        # Import the real implementation
        from .vq_vae import KenaDualLoss as RealKenaDualLoss
        self.real_loss = RealKenaDualLoss(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        print("WARNING: Using deprecated import. Please import KenaDualLoss from vq_vae.py instead.")
        return self.real_loss(*args, **kwargs)