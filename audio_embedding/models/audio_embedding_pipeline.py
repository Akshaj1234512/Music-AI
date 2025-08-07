import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .clean_basic_pitch_wrapper import CleanBasicPitchWrapper
from .huggingface_encodec import HuggingFaceEncodec
from .vq_vae import KenaVQVAE
from .huggingface_clap import HuggingFaceCLAP


class GuitarAudioEmbeddingPipeline(nn.Module):
    """
    Hierarchical audio embedding pipeline for guitar transcription.
    Combines Basic Pitch, Meta Encodec, VQ-VAE, and CLAP for comprehensive audio understanding.
    
    Architecture:
    1. Basic Pitch: Initial feature extraction (onset, contour, note predictions)
    2. Meta Encodec: Audio compression with discrete codebook representation
    3. VQ-VAE: Structured guitar-specific representation learning
    4. CLAP: Semantic audio understanding and style embeddings
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        embedding_dim: int = 768,
        vq_codebook_size: int = 512,
        vq_codebook_dim: int = 64,
        encodec_bandwidth: float = 6.0,  # kbps
        use_clap: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize Basic Pitch wrapper (defaults to precomputed mode for training)
        self.basic_pitch = CleanBasicPitchWrapper(
            sample_rate=sample_rate,
            mode='precomputed',  # Use precomputed features for training
            device=device
        )
        
        self.encodec = HuggingFaceEncodec(
            model_name="facebook/encodec_24khz",
            target_bandwidth=encodec_bandwidth,
            output_dim=128
        )
        
        self.kena_vq_vae = KenaVQVAE(
            input_dim=self.basic_pitch.output_dim,
            codebook_size=vq_codebook_size,
            codebook_dim=vq_codebook_dim,
            hidden_dims=[512, 512, 512, 512]
        )
        
        if use_clap:
            self.clap_encoder = HuggingFaceCLAP(
                output_dim=embedding_dim,
                freeze_model=True
            )
        else:
            self.clap_encoder = None
        
        # Projection layers for unified embedding space
        self.pitch_proj = nn.Linear(self.basic_pitch.output_dim, embedding_dim)
        self.encodec_proj = nn.Linear(self.encodec.output_dim, embedding_dim)
        self.vq_proj = nn.Linear(vq_codebook_dim, embedding_dim)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        audio: torch.Tensor,
        basic_pitch_features: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio through the hierarchical pipeline.
        
        Args:
            audio: Raw audio waveform [batch, time]
            basic_pitch_features: Optional precomputed Basic Pitch features [batch, time_frames, 440]
            return_intermediates: Whether to return intermediate representations
            
        Returns:
            Dictionary containing:
            - embeddings: Final fused embeddings [batch, time_frames, embedding_dim]
            - discrete_codes: VQ-VAE codebook indices [batch, time_frames]
            - pitch_features: Basic Pitch features [batch, time_frames, pitch_dim]
            - semantic_features: CLAP embeddings [batch, embedding_dim] (if enabled)
        """
        outputs = {}
        
        # Stage 1: Basic Pitch feature extraction (use precomputed if available)
        if basic_pitch_features is not None:
            pitch_features = basic_pitch_features
        else:
            pitch_features = self.basic_pitch(audio)
        pitch_embeddings = self.pitch_proj(pitch_features)
        outputs['pitch_features'] = pitch_features
        
        # Stage 2: HuggingFace Encodec compression
        encodec_codes, encodec_embeddings = self.encodec(audio, source_sample_rate=22050)
        encodec_embeddings = self.encodec_proj(encodec_embeddings)
        outputs['encodec_codes'] = encodec_codes
        
        # Stage 3: Kena VQ-VAE (embeddings + discrete tokens for multimodal fusion)
        kena_outputs = self.kena_vq_vae(pitch_features)
        vq_embeddings = self.vq_proj(kena_outputs['z_q'])
        outputs['discrete_codes'] = kena_outputs['indices']
        outputs['vq_loss'] = kena_outputs['commitment_loss']
        
        # Store VQ-VAE embeddings for fusion (no direct transcription)
        # For transcription, use EmbeddingValidationDecoder or MultimodalDecoder
        outputs['vq_embeddings'] = kena_outputs['z_q']  # Raw VQ embeddings
        
        # Stage 4: CLAP semantic encoding (if enabled)
        if self.clap_encoder is not None:
            semantic_features = self.clap_encoder(audio)
            outputs['semantic_features'] = semantic_features
            
            # Expand semantic features to match temporal dimension
            semantic_expanded = semantic_features.unsqueeze(1).repeat(
                1, pitch_embeddings.shape[1], 1
            )
        else:
            semantic_expanded = torch.zeros_like(pitch_embeddings)
        
        # Align temporal dimensions before fusion
        pitch_embeddings, encodec_embeddings, vq_embeddings, semantic_expanded = self._align_temporal_dimensions(
            pitch_embeddings, encodec_embeddings, vq_embeddings, semantic_expanded
        )
        
        # Hierarchical fusion of embeddings
        fused_embeddings = self._fuse_embeddings(
            pitch_embeddings,
            encodec_embeddings,
            vq_embeddings,
            semantic_expanded
        )
        
        outputs['embeddings'] = fused_embeddings
        
        if return_intermediates:
            outputs['pitch_embeddings'] = pitch_embeddings
            outputs['encodec_embeddings'] = encodec_embeddings
            outputs['vq_embeddings'] = vq_embeddings
        
        return outputs
    
    def _fuse_embeddings(
        self,
        pitch_emb: torch.Tensor,
        encodec_emb: torch.Tensor,
        vq_emb: torch.Tensor,
        semantic_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multiple embedding streams using weighted addition.
        
        Args:
            pitch_emb: Basic Pitch embeddings [batch, time, dim]
            encodec_emb: Encodec embeddings [batch, time, dim]
            vq_emb: VQ-VAE embeddings [batch, time, dim]
            semantic_emb: CLAP semantic embeddings [batch, time, dim]
            
        Returns:
            Fused embeddings [batch, time, dim]
        """
        # Weighted fusion (weights can be learned or fixed)
        weights = {
            'pitch': 0.4,
            'encodec': 0.2,
            'vq': 0.3,
            'semantic': 0.1
        }
        
        fused = (
            weights['pitch'] * pitch_emb +
            weights['encodec'] * encodec_emb +
            weights['vq'] * vq_emb +
            weights['semantic'] * semantic_emb
        )
        
        # Apply layer normalization
        fused = self.layer_norm(fused)
        
        return fused
    
    def _align_temporal_dimensions(
        self,
        pitch_emb: torch.Tensor,
        encodec_emb: torch.Tensor,
        vq_emb: torch.Tensor,
        semantic_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align temporal dimensions of different embeddings using interpolation.
        
        Args:
            pitch_emb: Basic Pitch embeddings [batch, time_pitch, dim]
            encodec_emb: Encodec embeddings [batch, time_encodec, dim]
            vq_emb: VQ-VAE embeddings [batch, time_vq, dim]
            semantic_emb: CLAP semantic embeddings [batch, time_semantic, dim]
            
        Returns:
            Aligned embeddings with same temporal dimension
        """
        # Find the minimum temporal dimension to avoid information loss
        time_dims = [
            pitch_emb.shape[1],
            encodec_emb.shape[1], 
            vq_emb.shape[1],
            semantic_emb.shape[1]
        ]
        target_time = min(time_dims)
        
        # Interpolate to target temporal dimension
        def interpolate_temporal(emb: torch.Tensor, target_frames: int) -> torch.Tensor:
            if emb.shape[1] == target_frames:
                return emb
            
            # Transpose to [batch, dim, time] for interpolation
            emb_transposed = emb.transpose(1, 2)
            
            # Interpolate temporal dimension
            emb_interp = F.interpolate(
                emb_transposed, 
                size=target_frames, 
                mode='linear', 
                align_corners=False
            )
            
            # Transpose back to [batch, time, dim]
            return emb_interp.transpose(1, 2)
        
        # Align all embeddings to target temporal dimension
        pitch_aligned = interpolate_temporal(pitch_emb, target_time)
        encodec_aligned = interpolate_temporal(encodec_emb, target_time)
        vq_aligned = interpolate_temporal(vq_emb, target_time)
        semantic_aligned = interpolate_temporal(semantic_emb, target_time)
        
        return pitch_aligned, encodec_aligned, vq_aligned, semantic_aligned
    
    def extract_embeddings(
        self,
        audio: torch.Tensor,
        level: str = 'fused'
    ) -> torch.Tensor:
        """
        Extract embeddings at specified level of the hierarchy.
        
        Args:
            audio: Raw audio waveform [batch, time]
            level: One of ['pitch', 'encodec', 'vq', 'semantic', 'fused']
            
        Returns:
            Embeddings at specified level
        """
        outputs = self.forward(audio, return_intermediates=True)
        
        level_map = {
            'pitch': 'pitch_embeddings',
            'encodec': 'encodec_embeddings',
            'vq': 'vq_embeddings',
            'semantic': 'semantic_features',
            'fused': 'embeddings'
        }
        
        if level not in level_map:
            raise ValueError(f"Invalid level: {level}. Choose from {list(level_map.keys())}")
        
        return outputs[level_map[level]]
    
    def get_discrete_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract discrete tokens from VQ-VAE codebook.
        
        Args:
            audio: Raw audio waveform [batch, time]
            
        Returns:
            Discrete token indices [batch, time_frames]
        """
        outputs = self.forward(audio)
        return outputs['discrete_codes']


# Import actual implementations  
from .clean_basic_pitch_wrapper import CleanBasicPitchWrapper
from .huggingface_encodec import HuggingFaceEncodec
from .vq_vae import KenaVQVAE
from .huggingface_clap import HuggingFaceCLAP