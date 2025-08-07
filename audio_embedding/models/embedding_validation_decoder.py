import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .pitch_decoder import FrameToNotePP
from .tab_assignment import DynamicProgrammingTabAssignment, GuitarNote


class EmbeddingValidationDecoder(nn.Module):
    """
    CRNN decoder for validating audio embedding quality.
    
    Based on GAPS paper baseline and your research showing CRNN as 
    "workhorse architecture" and "current practical state-of-the-art".
    
    Architecture:
    - CNN layers: Process your 768-dim embeddings
    - GRU layers: Model temporal dependencies 
    - Multi-head outputs: Onset + Frame predictions (following research)
    - Post-processing: Frame-to-note → Tab assignment
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        gru_hidden: int = 128,
        n_pitches: int = 88,  # Piano range (A0 to C8)
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.n_pitches = n_pitches
        self.bidirectional = bidirectional
        
        # CNN feature extraction (processes embeddings)
        # Following GAPS approach: convolution only along frequency axis
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU for temporal modeling (following GAPS/Kong et al.)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if bidirectional else 0,
            bidirectional=bidirectional
        )
        
        gru_output_dim = gru_hidden * (2 if bidirectional else 1)
        
        # Multi-head outputs (following your research on multi-task learning)
        self.onset_head = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_pitches)
        )
        
        self.frame_head = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_pitches)
        )
        
        # Confidence head (for weighting predictions)
        self.confidence_head = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Guitar-specific pitch bias (learned parameter)
        self.pitch_bias = nn.Parameter(torch.zeros(n_pitches))
        self._init_guitar_bias()
    
    def _init_guitar_bias(self):
        """Initialize bias towards guitar pitch range (E2 to E6)."""
        # Guitar range in piano keys: E2 (MIDI 40) to E6 (MIDI 88)
        # Convert to 0-87 range (A0=0 to C8=87)
        guitar_low = 40 - 21   # E2 relative to A0
        guitar_high = 88 - 21  # E6 relative to A0
        
        # Create gaussian bias centered on guitar range
        midi_range = torch.arange(self.n_pitches).float()
        guitar_center = (guitar_low + guitar_high) / 2
        guitar_spread = (guitar_high - guitar_low) / 4
        
        bias = torch.exp(-(midi_range - guitar_center)**2 / (2 * guitar_spread**2))
        self.pitch_bias.data = bias * 0.2  # Modest bias towards guitar range
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass from embeddings to pitch predictions.
        
        Args:
            embeddings: Audio embeddings [batch, time, embedding_dim]
            
        Returns:
            Dictionary with onset/frame predictions and confidence
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # CNN processing: [batch, time, embed] → [batch, embed, time] → [batch, hidden, time]
        x = embeddings.transpose(1, 2)  # [batch, embed_dim, time]
        cnn_features = self.cnn_layers(x)  # [batch, hidden_dim, time]
        cnn_features = cnn_features.transpose(1, 2)  # [batch, time, hidden_dim]
        
        # GRU temporal modeling
        gru_output, _ = self.gru(cnn_features)  # [batch, time, gru_output_dim]
        
        # Multi-head predictions
        onset_logits = self.onset_head(gru_output)  # [batch, time, n_pitches]
        frame_logits = self.frame_head(gru_output)  # [batch, time, n_pitches]
        confidence_logits = self.confidence_head(gru_output)  # [batch, time, 1]
        
        # Apply guitar bias
        onset_logits = onset_logits + self.pitch_bias
        frame_logits = frame_logits + self.pitch_bias
        
        # Return predictions
        return {
            'onset_logits': onset_logits,
            'frame_logits': frame_logits,
            'confidence_logits': confidence_logits,
            'onset_probs': torch.sigmoid(onset_logits),
            'frame_probs': torch.sigmoid(frame_logits),
            'confidence': torch.sigmoid(confidence_logits).squeeze(-1)
        }


class EmbeddingValidationSystem:
    """
    Complete validation system for testing embedding quality.
    
    Pipeline:
    1. Your embedding pipeline → 768-dim embeddings
    2. CRNN decoder → pitch predictions  
    3. Post-processing → note events
    4. Tab assignment → guitar tablature
    
    This tests if your embeddings contain sufficient information
    for high-quality guitar transcription.
    """
    
    def __init__(
        self,
        embedding_pipeline,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        freeze_embeddings: bool = True
    ):
        self.device = torch.device(device)
        self.embedding_pipeline = embedding_pipeline.to(self.device)
        self.decoder = EmbeddingValidationDecoder().to(self.device)
        
        # Freeze embedding pipeline to test its current quality
        if freeze_embeddings:
            for param in self.embedding_pipeline.parameters():
                param.requires_grad = False
        
        # Post-processing components
        self.frame_to_note = FrameToNotePP(
            onset_threshold=0.5,
            frame_threshold=0.5,
            min_note_length=3,
            min_gap_length=2
        )
        
        self.tab_assigner = DynamicProgrammingTabAssignment()
        
        # Set to eval mode initially
        self.embedding_pipeline.eval()
        self.decoder.eval()
    
    def validate_embeddings(
        self,
        audio: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, any]:
        """
        Test embedding quality by transcribing audio to tablature.
        
        Args:
            audio: Input audio [batch, time] or [time]
            return_intermediate: Return intermediate representations
            
        Returns:
            Dictionary with transcription results and optionally intermediates
        """
        # Prepare audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        
        with torch.no_grad():
            # Step 1: Extract embeddings and Kena predictions from pipeline
            embedding_outputs = self.embedding_pipeline(audio)
            
            # Check if Kena VQ-VAE predictions are available
            if 'kena_onset_probs' in embedding_outputs and 'kena_frame_probs' in embedding_outputs:
                # Use Kena's direct predictions (preferred)
                onset_probs = embedding_outputs['kena_onset_probs']
                frame_probs = embedding_outputs['kena_frame_probs']
                using_kena_direct = True
            else:
                # Fallback to CRNN decoder on embeddings
                embeddings = embedding_outputs['embeddings']  # [batch, time, 768]
                pitch_outputs = self.decoder(embeddings)
                onset_probs = pitch_outputs['onset_probs']
                frame_probs = pitch_outputs['frame_probs']
                using_kena_direct = False
            
            # Step 2: Convert to note events
            notes = self.frame_to_note(onset_probs, frame_probs)
            
            # Step 4: Assign to guitar strings/frets
            results = []
            for batch_notes in notes:
                # Convert to proper format for tab assignment
                formatted_notes = [
                    (pitch + 21, onset * 0.023, offset * 0.023)  # Convert back to MIDI + time
                    for pitch, onset, offset in batch_notes
                ]
                
                # Assign strings
                guitar_notes = self.tab_assigner.assign_strings(formatted_notes)
                results.append(guitar_notes)
        
        # Prepare output
        output = {
            'guitar_notes': results[0] if len(results) == 1 else results,
            'embedding_quality_metrics': self._compute_quality_metrics(
                embedding_outputs, {'onset_probs': onset_probs, 'frame_probs': frame_probs}, results
            ),
            'using_kena_direct': using_kena_direct if 'using_kena_direct' in locals() else False
        }
        
        # Add intermediate representations if requested
        if return_intermediate:
            output['intermediate'] = {
                'embeddings': embeddings,
                'pitch_predictions': pitch_outputs,
                'raw_notes': notes,
                'embedding_components': {
                    'basic_pitch': embedding_outputs.get('pitch_features'),
                    'encodec': embedding_outputs.get('encodec_codes'),
                    'kena_vq_vae': embedding_outputs.get('discrete_codes'),
                    'clap': embedding_outputs.get('semantic_features'),
                    'kena_onset_probs': embedding_outputs.get('kena_onset_probs'),
                    'kena_frame_probs': embedding_outputs.get('kena_frame_probs')
                }
            }
        
        return output
    
    def _compute_quality_metrics(
        self,
        embeddings: torch.Tensor,
        pitch_outputs: Dict[str, torch.Tensor],
        guitar_notes: List[List[GuitarNote]]
    ) -> Dict[str, float]:
        """Compute metrics to assess embedding quality."""
        metrics = {}
        
        # Embedding statistics
        if 'embeddings' in embeddings and isinstance(embeddings['embeddings'], torch.Tensor):
            emb_tensor = embeddings['embeddings']
            metrics['embedding_mean'] = emb_tensor.mean().item()
            metrics['embedding_std'] = emb_tensor.std().item()
            metrics['embedding_sparsity'] = (emb_tensor.abs() < 0.01).float().mean().item()
        else:
            metrics['embedding_mean'] = 0.0
            metrics['embedding_std'] = 0.0  
            metrics['embedding_sparsity'] = 0.0
        
        # Prediction statistics
        onset_probs = pitch_outputs['onset_probs']
        frame_probs = pitch_outputs['frame_probs']
        
        metrics['avg_onset_activation'] = onset_probs.mean().item()
        metrics['avg_frame_activation'] = frame_probs.mean().item()
        metrics['pitch_range_coverage'] = (onset_probs.max(dim=1)[0] > 0.1).float().mean().item()
        
        # Output statistics
        total_notes = sum(len(batch_notes) for batch_notes in guitar_notes)
        metrics['notes_detected'] = total_notes
        
        if total_notes > 0:
            # Pitch range analysis
            all_pitches = [note.pitch for batch_notes in guitar_notes for note in batch_notes]
            metrics['pitch_range_min'] = min(all_pitches)
            metrics['pitch_range_max'] = max(all_pitches)
            metrics['unique_pitches'] = len(set(all_pitches))
            
            # String usage analysis
            string_usage = [0] * 6
            for batch_notes in guitar_notes:
                for note in batch_notes:
                    if note.string is not None:
                        string_usage[note.string] += 1
            metrics['string_usage_entropy'] = self._compute_entropy(string_usage)
        
        return metrics
    
    def _compute_entropy(self, counts: List[int]) -> float:
        """Compute entropy of a distribution."""
        total = sum(counts)
        if total == 0:
            return 0.0
        
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log2(p) for p in probs)
    
    def format_results(self, results: Dict) -> str:
        """Format validation results for display."""
        guitar_notes = results['guitar_notes']
        metrics = results['embedding_quality_metrics']
        
        output = []
        output.append("=" * 60)
        output.append("EMBEDDING VALIDATION RESULTS")
        output.append("=" * 60)
        
        # Prediction source info
        using_kena = results.get('using_kena_direct', False)
        output.append(f"\nPrediction Source: {'Kena VQ-VAE Direct' if using_kena else 'CRNN Decoder'}")
        
        # Embedding quality metrics
        output.append(f"\nEmbedding Quality Metrics:")
        if 'embedding_mean' in metrics:
            output.append(f"  Mean activation: {metrics['embedding_mean']:.4f}")
            output.append(f"  Std deviation: {metrics['embedding_std']:.4f}")
            output.append(f"  Sparsity: {metrics['embedding_sparsity']:.4f}")
        else:
            output.append(f"  Embeddings not available (using Kena direct)")
        
        # Prediction quality
        output.append(f"\nPrediction Quality:")
        output.append(f"  Notes detected: {metrics['notes_detected']}")
        output.append(f"  Avg onset activation: {metrics['avg_onset_activation']:.4f}")
        output.append(f"  Avg frame activation: {metrics['avg_frame_activation']:.4f}")
        output.append(f"  Pitch range coverage: {metrics['pitch_range_coverage']:.4f}")
        
        if metrics['notes_detected'] > 0:
            output.append(f"  Pitch range: {metrics['pitch_range_min']} - {metrics['pitch_range_max']}")
            output.append(f"  Unique pitches: {metrics['unique_pitches']}")
            output.append(f"  String usage entropy: {metrics['string_usage_entropy']:.4f}")
        
        # Sample notes
        output.append(f"\nSample Detected Notes:")
        if isinstance(guitar_notes, list) and len(guitar_notes) > 0:
            for i, note in enumerate(guitar_notes[:5]):
                output.append(f"  Note {i+1}: Pitch {note.pitch}, String {note.string}, "
                            f"Fret {note.fret}, Time {note.onset_time:.2f}s")
        
        return "\n".join(output)


# Training utilities for fine-tuning the decoder
class EmbeddingValidationTrainer:
    """Trainer for fine-tuning the validation decoder while keeping embeddings frozen."""
    
    def __init__(
        self,
        validation_system: EmbeddingValidationSystem,
        learning_rate: float = 1e-4
    ):
        self.system = validation_system
        
        # Only train the decoder, keep embeddings frozen
        decoder_params = list(self.system.decoder.parameters())
        self.optimizer = torch.optim.Adam(decoder_params, lr=learning_rate)
        
        # Loss function (binary cross entropy for multi-label)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_step(
        self,
        audio: torch.Tensor,
        onset_targets: torch.Tensor,
        frame_targets: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step - tests if embeddings support learning."""
        self.system.decoder.train()
        self.optimizer.zero_grad()
        
        # Get embeddings (frozen)
        with torch.no_grad():
            embedding_outputs = self.system.embedding_pipeline(audio)
            embeddings = embedding_outputs['embeddings']
        
        # Decode (trainable)
        outputs = self.system.decoder(embeddings)
        
        # Calculate losses
        onset_loss = self.criterion(outputs['onset_logits'], onset_targets)
        frame_loss = self.criterion(outputs['frame_logits'], frame_targets)
        total_loss = onset_loss + frame_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'onset_loss': onset_loss.item(),
            'frame_loss': frame_loss.item(),
            'total_loss': total_loss.item()
        }