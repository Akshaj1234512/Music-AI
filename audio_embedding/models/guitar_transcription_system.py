import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .audio_embedding_pipeline import GuitarAudioEmbeddingPipeline
from .embedding_validation_decoder import EmbeddingValidationDecoder
from .pitch_decoder import FrameToNotePP
from .tab_assignment import DynamicProgrammingTabAssignment, TechniqueDetector, TabFormatter, GuitarNote


class GuitarTranscriptionSystem:
    """
    Complete guitar transcription system following Kena AI's approach.
    
    Pipeline:
    1. Audio → Embeddings (our pipeline)
    2. Embeddings → Pitch detection (Kena-style dual-loss)
    3. Pitches → Note events (post-processing)
    4. Notes → Tab assignment (dynamic programming)
    5. Tab → Formatted output
    
    Target: 87% accuracy matching Kena AI's performance
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        sample_rate: int = 22050
    ):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        
        # Initialize components
        self._init_models(model_path)
        self._init_processors()
    
    def _init_models(self, model_path: Optional[str]):
        """Initialize neural network models."""
        # Audio embedding pipeline
        self.embedding_pipeline = GuitarAudioEmbeddingPipeline(
            sample_rate=self.sample_rate,
            device=self.device
        ).to(self.device)
        
        # Audio transcription decoder (for audio-only mode)
        self.audio_decoder = EmbeddingValidationDecoder(
            embedding_dim=768,
            hidden_dim=256,
            n_pitches=88
        ).to(self.device)
        
        # Post-processing for transcription predictions
        self.frame_to_note = FrameToNotePP(
            onset_threshold=0.5,
            frame_threshold=0.5,
            min_note_length=3,
            min_gap_length=2
        )
        
        # Load weights if provided
        if model_path is not None:
            self.load_checkpoint(model_path)
        
        # Set to eval mode
        self.embedding_pipeline.eval()
    
    def _init_processors(self):
        """Initialize post-processing components."""
        # Tab assignment
        self.tab_assigner = DynamicProgrammingTabAssignment()
        
        # Technique detection
        self.technique_detector = TechniqueDetector()
        
        # Tab formatter
        self.tab_formatter = TabFormatter()
    
    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        return_midi: bool = False,
        return_confidence: bool = False,
        format_tab: bool = True
    ) -> Dict[str, any]:
        """
        Transcribe audio to guitar tablature.
        
        Args:
            audio: Audio waveform [time] or [batch, time]
            return_midi: Whether to return MIDI representation
            return_confidence: Whether to return confidence scores
            format_tab: Whether to format as ASCII tab
            
        Returns:
            Dictionary containing:
            - 'tab': Formatted tablature (if format_tab=True)
            - 'notes': List of GuitarNote objects
            - 'midi': MIDI representation (if return_midi=True)
            - 'confidence': Confidence scores (if return_confidence=True)
        """
        # Prepare audio
        audio = self._prepare_audio(audio)
        
        with torch.no_grad():
            # Step 1: Get embeddings from pipeline
            embedding_outputs = self.embedding_pipeline(audio)
            embeddings = embedding_outputs['embeddings']  # [batch, time, 768]
            
            # Step 2: Use audio decoder for transcription (multimodal-compatible)
            decoder_outputs = self.audio_decoder(embeddings)
            onset_probs = decoder_outputs['onset_probs']
            frame_probs = decoder_outputs['frame_probs']
            
            # Step 3: Convert predictions to note events
            notes = self.frame_to_note(onset_probs, frame_probs)
            
            # Process each batch item
            results = []
            for batch_idx, batch_notes in enumerate(notes):
                # Convert to proper format for tab assignment
                formatted_notes = [
                    (pitch + 21, onset * 0.023, offset * 0.023)  # Convert back to MIDI + time
                    for pitch, onset, offset in batch_notes
                ]
                
                # Step 3: Assign strings using DP algorithm
                guitar_notes = self.tab_assigner.assign_strings(formatted_notes)
                
                # Step 4: Detect techniques
                guitar_notes = self.technique_detector.detect_techniques(guitar_notes)
                
                # Prepare result
                result = {'notes': guitar_notes}
                
                # Add formatted tab if requested
                if format_tab:
                    tab_string = self.tab_formatter.notes_to_tab(guitar_notes)
                    result['tab'] = tab_string
                
                # Add MIDI if requested
                if return_midi:
                    midi_data = self._notes_to_midi(guitar_notes)
                    result['midi'] = midi_data
                
                # Add confidence if requested
                if return_confidence:
                    # Use frame probabilities as confidence measure
                    confidence = frame_probs[batch_idx].max(dim=-1)[0].cpu().numpy()
                    result['confidence'] = confidence
                
                results.append(result)
        
        # Return single result if single input
        if len(results) == 1:
            return results[0]
        else:
            return results
    
    def _prepare_audio(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Prepare audio for processing."""
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Move to device
        audio = audio.to(self.device)
        
        return audio
    
    def _notes_to_midi(self, notes: List[GuitarNote]) -> Dict:
        """Convert notes to MIDI representation."""
        # Simple MIDI-like representation
        midi_events = []
        
        for note in notes:
            midi_events.append({
                'type': 'note_on',
                'pitch': note.pitch,
                'time': note.onset_time,
                'velocity': int(note.confidence * 127),
                'string': note.string,
                'fret': note.fret
            })
            
            midi_events.append({
                'type': 'note_off',
                'pitch': note.pitch,
                'time': note.offset_time,
                'string': note.string,
                'fret': note.fret
            })
        
        # Sort by time
        midi_events.sort(key=lambda x: x['time'])
        
        return {
            'events': midi_events,
            'ticks_per_quarter': 480,
            'tempo': 500000  # 120 BPM
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'embedding_pipeline': self.embedding_pipeline.state_dict(),
            'pitch_model': self.pitch_model.state_dict(),
            'config': {
                'sample_rate': self.sample_rate,
                'device': str(self.device)
            }
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model weights
        self.embedding_pipeline.load_state_dict(checkpoint['embedding_pipeline'])
        self.pitch_model.load_state_dict(checkpoint['pitch_model'])
        
        print(f"Loaded checkpoint from {path}")
    
    def process_file(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        save_midi: bool = False
    ) -> Dict:
        """
        Process audio file to tablature.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save tab file (optional)
            save_midi: Whether to save MIDI file
            
        Returns:
            Transcription results
        """
        # Load audio
        from ..utils.audio_processing import load_and_preprocess_audio
        audio = load_and_preprocess_audio(audio_path, sample_rate=self.sample_rate)
        
        # Transcribe
        results = self.transcribe(audio, return_midi=save_midi, format_tab=True)
        
        # Save tab if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(results['tab'])
            print(f"Saved tablature to {output_path}")
        
        # Save MIDI if requested
        if save_midi and 'midi' in results:
            midi_path = output_path.replace('.txt', '.mid') if output_path else 'output.mid'
            # Would need proper MIDI library here
            print(f"MIDI export to {midi_path} (requires midiutil or pretty_midi)")
        
        return results


class AudioTranscriptionTrainer:
    """
    Trainer for the audio transcription system (decoder-focused).
    Trains the audio decoder while keeping embeddings frozen for validation,
    or end-to-end with VQ-VAE for full system optimization.
    """
    
    def __init__(
        self,
        transcription_system: GuitarTranscriptionSystem,
        learning_rate: float = 1e-4,
        freeze_embeddings: bool = True,
        train_vq_vae: bool = False
    ):
        self.system = transcription_system
        self.freeze_embeddings = freeze_embeddings
        self.train_vq_vae = train_vq_vae
        
        # Loss function for transcription
        from .vq_vae import KenaDualLoss
        self.criterion = KenaDualLoss(
            onset_weight=1.0,
            frame_weight=1.0,
            commitment_weight=0.25
        )
        
        # Set up training parameters
        params_to_train = []
        
        # Always train the audio decoder
        params_to_train.extend(list(self.system.audio_decoder.parameters()))
        
        # Optionally train VQ-VAE
        if train_vq_vae:
            params_to_train.extend(list(self.system.embedding_pipeline.kena_vq_vae.parameters()))
        
        # Freeze embeddings if requested
        if freeze_embeddings:
            for param in self.system.embedding_pipeline.parameters():
                param.requires_grad = not train_vq_vae or param in self.system.embedding_pipeline.kena_vq_vae.parameters()
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=5,
            factor=0.5
        )
    
    def train_step(
        self,
        audio: torch.Tensor,
        onset_targets: torch.Tensor,
        frame_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step for audio transcription."""
        self.system.audio_decoder.train()
        if self.train_vq_vae:
            self.system.embedding_pipeline.kena_vq_vae.train()
        
        self.optimizer.zero_grad()
        
        # Forward pass through embedding pipeline
        embedding_outputs = self.system.embedding_pipeline(audio)
        embeddings = embedding_outputs['embeddings']
        
        # Forward pass through audio decoder
        decoder_outputs = self.system.audio_decoder(embeddings)
        onset_logits = decoder_outputs['onset_logits']
        frame_logits = decoder_outputs['frame_logits']
        
        # Get VQ commitment loss if training VQ-VAE
        vq_commitment_loss = embedding_outputs['vq_loss'] if self.train_vq_vae else None
        
        # Calculate loss
        loss, loss_dict = self.criterion(
            onset_logits,
            frame_logits,
            onset_targets,
            frame_targets,
            vq_commitment_loss
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Return losses
        return {k: v.item() for k, v in loss_dict.items()}
    
    def eval_step(
        self,
        audio: torch.Tensor,
        onset_targets: torch.Tensor,
        frame_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Single evaluation step for audio transcription."""
        self.system.audio_decoder.eval()
        if self.train_vq_vae:
            self.system.embedding_pipeline.kena_vq_vae.eval()
        
        with torch.no_grad():
            embedding_outputs = self.system.embedding_pipeline(audio)
            embeddings = embedding_outputs['embeddings']
            
            decoder_outputs = self.system.audio_decoder(embeddings)
            onset_logits = decoder_outputs['onset_logits']
            frame_logits = decoder_outputs['frame_logits']
            
            vq_commitment_loss = embedding_outputs['vq_loss'] if self.train_vq_vae else None
            
            loss, loss_dict = self.criterion(
                onset_logits,
                frame_logits,
                onset_targets,
                frame_targets,
                vq_commitment_loss
            )
        
        return {k: v.item() for k, v in loss_dict.items()}


# Aliases for backward compatibility
KenaVQVAETrainer = AudioTranscriptionTrainer
KenaStyleTrainer = AudioTranscriptionTrainer