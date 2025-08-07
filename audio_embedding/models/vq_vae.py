import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.
    
    Implements the quantization with straight-through gradient estimation
    and exponential moving average updates for codebook learning.
    """
    
    def __init__(
        self,
        codebook_size: int = 512,
        codebook_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        # EMA updates for codebook
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', torch.zeros(codebook_size, codebook_dim))
        self.register_buffer('initialized', torch.tensor(False))
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor using codebook.
        
        Args:
            inputs: Input tensor [batch, time, dim]
            
        Returns:
            quantized: Quantized output [batch, time, dim]
            indices: Codebook indices [batch, time]
            commitment_loss: VQ commitment loss
        """
        # Flatten input
        batch_size, time_steps, _ = inputs.shape
        flat_input = inputs.reshape(-1, self.codebook_dim)
        
        # Calculate distances to codebook vectors
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        # Get closest codebook entries
        indices = torch.argmin(distances, dim=1)
        indices_reshaped = indices.view(batch_size, time_steps)
        
        # Quantize
        quantized_flat = self.embedding(indices)
        quantized = quantized_flat.view(batch_size, time_steps, self.codebook_dim)
        
        # Update codebook using EMA (only during training)
        if self.training:
            self._update_codebook(flat_input, indices)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), inputs) * self.commitment_cost
        
        return quantized, indices_reshaped, commitment_loss
    
    def _update_codebook(self, flat_input: torch.Tensor, indices: torch.Tensor):
        """Update codebook using exponential moving average."""
        # Convert indices to one-hot
        encodings = F.one_hot(indices, num_classes=self.codebook_size).float()
        
        # Update cluster sizes
        self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encodings, 0)
        
        # Update codebook vectors
        dw = torch.matmul(encodings.t(), flat_input)
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        
        # Normalize
        n = torch.sum(self.ema_cluster_size)
        self.ema_cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (n + self.codebook_size * self.epsilon) * n
        )
        
        self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)


class KenaVQVAE(nn.Module):
    """
    Kena AI's VQ-VAE implementation for multimodal music transcription.
    
    IMPORTANT: Refactored for multimodal architecture compatibility.
    This component focuses on learning high-quality embeddings and discrete 
    representations that can be used by separate transcription decoders.
    
    Architecture:
    - VQ-VAE foundation with discrete latent representations
    - Guitar-specific adaptations (string-aware processing) 
    - Optimized embeddings for both audio-only and multimodal transcription
    - NO direct transcription heads (moved to separate decoders)
    
    For transcription, use EmbeddingValidationDecoder or future MultimodalDecoder.
    """
    
    def __init__(
        self,
        input_dim: int = 360,  # From Basic Pitch
        codebook_size: int = 512,
        codebook_dim: int = 64,
        hidden_dims: List[int] = [512, 512, 512, 512],
        n_strings: int = 6,  # Guitar strings
        use_string_pooling: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.n_strings = n_strings
        self.use_string_pooling = use_string_pooling
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            
            # Downsample every other layer
            if i % 2 == 1 and i < len(hidden_dims) - 1:
                encoder_layers.append(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
                )
            
            in_dim = hidden_dim
        
        # Final encoder projection
        encoder_layers.extend([
            nn.Conv1d(hidden_dims[-1], codebook_dim, kernel_size=1),
            nn.BatchNorm1d(codebook_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=0.25
        )
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        decoder_layers.extend([
            nn.Conv1d(codebook_dim, hidden_dims[-1], kernel_size=1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        ])
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Upsample if needed
            if i % 2 == 1 and i < len(hidden_dims) - 1:
                decoder_layers.append(
                    nn.ConvTranspose1d(
                        hidden_dims[i], hidden_dims[i],
                        kernel_size=4, stride=2, padding=1
                    )
                )
            
            decoder_layers.extend([
                nn.Conv1d(hidden_dims[i], hidden_dims[i-1], kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dims[i-1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
        
        # Final decoder projection
        decoder_layers.append(
            nn.Conv1d(hidden_dims[0], input_dim, kernel_size=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Guitar-specific components
        if use_string_pooling:
            self.string_attention = nn.MultiheadAttention(
                embed_dim=codebook_dim,
                num_heads=8,  # Fixed: 64 ÷ 8 = 8 dimensions per head
                dropout=dropout
            )
        
        # REMOVED: Transcription heads moved to separate decoders
        # This allows embeddings to be used for multimodal fusion
        # Use EmbeddingValidationDecoder or MultimodalDecoder for transcription
    
    # REMOVED: Guitar bias moved to transcription decoders
    # Each decoder can implement its own instrument-specific biases
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to latent representation.
        
        Args:
            x: Input features [batch, time, features]
            
        Returns:
            Encoded features [batch, time, codebook_dim]
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [batch, features, time]
        
        # Encode
        z = self.encoder(x)
        
        # Transpose back
        z = z.transpose(1, 2)  # [batch, time, codebook_dim]
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output features.
        
        Args:
            z: Latent features [batch, time, codebook_dim]
            
        Returns:
            Decoded features [batch, time, features]
        """
        # Transpose for Conv1d
        z = z.transpose(1, 2)  # [batch, codebook_dim, time]
        
        # Decode
        x_recon = self.decoder(z)
        
        # Transpose back
        x_recon = x_recon.transpose(1, 2)  # [batch, time, features]
        
        return x_recon
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Multimodal-compatible forward pass: encode → quantize → embeddings.
        
        Args:
            x: Input features [batch, time, features]
            
        Returns:
            Dictionary containing:
            - z_q: Quantized embeddings [batch, time, codebook_dim] - for embedding fusion
            - indices: Codebook indices [batch, time] - for discrete pattern learning  
            - commitment_loss: VQ commitment loss - for training
            
        For transcription, pass z_q to EmbeddingValidationDecoder or MultimodalDecoder.
        """
        # Encode to latent space
        z_e = self.encode(x)
        
        # Vector quantization
        z_q, indices, commitment_loss = self.quantizer(z_e)
        
        # Apply string-aware processing if enabled
        if self.use_string_pooling:
            z_q_attn, _ = self.string_attention(z_q, z_q, z_q)
            z_q = z_q + z_q_attn
        
        return {
            'z_q': z_q,                    # High-quality embeddings for fusion
            'indices': indices,            # Discrete tokens for pattern learning
            'commitment_loss': commitment_loss  # Training loss
        }
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """Get the current codebook embeddings."""
        return self.quantizer.embedding.weight.data
    
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input directly to codebook indices.
        
        Args:
            x: Input features [batch, time, features]
            
        Returns:
            Codebook indices [batch, time]
        """
        z_e = self.encode(x)
        _, indices, _ = self.quantizer(z_e)
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices.
        
        Args:
            indices: Codebook indices [batch, time]
            
        Returns:
            Decoded features [batch, time, features]
        """
        # Get embeddings from indices
        z_q = self.quantizer.embedding(indices)
        
        # Decode
        x_recon = self.decode(z_q)
        
        return x_recon


class KenaDualLoss(nn.Module):
    """
    Kena AI's dual-objective loss function for transcription decoders.
    
    L_total = L_onset + L_frame + L_consistency
    
    MOVED from VQ-VAE to transcription decoders for multimodal compatibility.
    Use this with EmbeddingValidationDecoder or MultimodalDecoder.
    
    Components:
    - L_onset: Precise detection of note beginnings
    - L_frame: Frame activation for note duration
    - L_consistency: Ensures onset ≤ frame (physical constraint)
    """
    
    def __init__(
        self,
        onset_weight: float = 1.0,
        frame_weight: float = 1.0,
        commitment_weight: float = 0.25,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.onset_weight = onset_weight
        self.frame_weight = frame_weight
        self.commitment_weight = commitment_weight
        self.label_smoothing = label_smoothing
        
        # Use binary cross entropy for onset/frame predictions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(
        self,
        onset_logits: torch.Tensor,
        frame_logits: torch.Tensor,
        onset_targets: torch.Tensor,
        frame_targets: torch.Tensor,
        vq_commitment_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate dual-objective transcription loss.
        
        Args:
            onset_logits: Onset predictions [batch, time, 88]
            frame_logits: Frame predictions [batch, time, 88]
            onset_targets: Onset ground truth [batch, time, 88]
            frame_targets: Frame ground truth [batch, time, 88]
            vq_commitment_loss: Optional VQ commitment loss (if training with VQ-VAE)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            onset_targets = onset_targets * (1 - self.label_smoothing) + self.label_smoothing / 2
            frame_targets = frame_targets * (1 - self.label_smoothing) + self.label_smoothing / 2
        
        # Calculate individual losses
        onset_loss = self.bce(onset_logits, onset_targets).mean()
        frame_loss = self.bce(frame_logits, frame_targets).mean()
        
        # Consistency loss (onset should be <= frame)
        onset_probs = torch.sigmoid(onset_logits)
        frame_probs = torch.sigmoid(frame_logits)
        consistency_loss = F.relu(onset_probs - frame_probs).mean()
        
        # Total loss
        total_loss = (
            self.onset_weight * onset_loss +
            self.frame_weight * frame_loss +
            0.1 * consistency_loss
        )
        
        loss_dict = {
            'onset_loss': onset_loss,
            'frame_loss': frame_loss, 
            'consistency_loss': consistency_loss,
            'total_loss': total_loss
        }
        
        # Add VQ commitment loss if provided (for end-to-end training)
        if vq_commitment_loss is not None:
            vq_loss_weighted = self.commitment_weight * vq_commitment_loss
            total_loss = total_loss + vq_loss_weighted
            loss_dict['vq_commitment_loss'] = vq_commitment_loss
            loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


# Alias for backward compatibility
GuitarVQVAE = KenaVQVAE  
GuitarVQVAELoss = KenaDualLoss