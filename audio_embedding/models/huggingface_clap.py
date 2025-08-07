import torch
import torch.nn as nn
from typing import Optional, Union
import numpy as np

from transformers import ClapModel, ClapProcessor


class HuggingFaceCLAP(nn.Module):
    """
    Simple wrapper for HuggingFace CLAP music model.
    
    Uses laion/larger_clap_music - pretrained specifically for music understanding.
    No guitar-specific modifications needed since CLAP already understands music.
    """
    
    def __init__(
        self,
        model_name: str = "laion/larger_clap_music",
        output_dim: int = 768,
        freeze_model: bool = True
    ):
        super().__init__()
        
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pretrained CLAP model
        print(f"Loading CLAP model: {model_name}")
        self.clap_model = ClapModel.from_pretrained(model_name)
        self.clap_processor = ClapProcessor.from_pretrained(model_name)
        
        # Get CLAP's native output dimension
        self.clap_output_dim = self.clap_model.config.projection_dim
        
        # Add projection layer if dimensions don't match
        if self.clap_output_dim != output_dim:
            self.projection = nn.Linear(self.clap_output_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze pretrained weights if specified
        if freeze_model:
            for param in self.clap_model.parameters():
                param.requires_grad = False
        
        print(f"CLAP loaded: {self.clap_output_dim} → {output_dim} dim")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract music-aware embeddings from audio.
        
        Args:
            audio: Audio waveform [batch, time] or [time]
            
        Returns:
            Music embeddings [batch, output_dim]
        """
        # Handle single audio input
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        embeddings = []
        
        # Process each audio in batch (CLAP processor expects individual audios)
        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()
            
            # Resample from 22050 to 48000 Hz (CLAP requirement)
            if len(audio_np) > 0:
                import torch.nn.functional as F
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                target_length = int(len(audio_np) * 48000 / 22050)
                audio_resampled = F.interpolate(
                    audio_tensor, 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).squeeze().numpy()
            else:
                audio_resampled = audio_np
            
            # CLAP preprocessing
            inputs = self.clap_processor(
                audios=audio_resampled,
                sampling_rate=48000,  # CLAP's expected sample rate
                return_tensors="pt"
            )
            
            # Move to correct device
            inputs = {k: v.to(audio.device) for k, v in inputs.items()}
            
            # Extract audio features
            with torch.no_grad() if not self.training else torch.enable_grad():
                audio_features = self.clap_model.get_audio_features(
                    inputs["input_features"]
                )
                embeddings.append(audio_features)
        
        # Stack batch
        embeddings = torch.cat(embeddings, dim=0)  # [batch, clap_dim]
        
        # Project to target dimension
        embeddings = self.projection(embeddings)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def get_text_embeddings(self, texts: list) -> torch.Tensor:
        """
        Get text embeddings for audio-text similarity.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings [len(texts), output_dim]
        """
        # Process texts
        text_inputs = self.clap_processor(
            text=texts,
            return_tensors="pt"
        )
        
        # Extract text features
        with torch.no_grad():
            text_embeddings = self.clap_model.get_text_features(**text_inputs)
        
        # Project if needed
        text_embeddings = self.projection(text_embeddings)
        
        # Normalize
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def compute_similarity(
        self,
        audio: torch.Tensor,
        texts: list
    ) -> torch.Tensor:
        """
        Compute similarity between audio and text descriptions.
        
        Args:
            audio: Audio waveform [batch, time]
            texts: List of text descriptions
            
        Returns:
            Similarity scores [batch, len(texts)]
        """
        audio_emb = self.forward(audio)
        text_emb = self.get_text_embeddings(texts)
        
        # Compute cosine similarity
        similarity = torch.matmul(audio_emb, text_emb.t())
        
        return similarity
    
    def classify_audio(
        self,
        audio: torch.Tensor,
        candidate_labels: list
    ) -> dict:
        """
        Zero-shot audio classification using text descriptions.
        
        Args:
            audio: Audio waveform [time] (single audio)
            candidate_labels: List of text descriptions/labels
            
        Returns:
            Dictionary with labels and scores
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)  # Remove batch dim for single audio
        
        # Compute similarities
        similarities = self.compute_similarity(audio.unsqueeze(0), candidate_labels)
        probabilities = torch.softmax(similarities, dim=-1).squeeze(0)
        
        # Create results
        results = {}
        for i, label in enumerate(candidate_labels):
            results[label] = probabilities[i].item()
        
        return results


# Guitar-specific text descriptions for technique classification
GUITAR_TECHNIQUE_LABELS = [
    "guitar slide technique",
    "guitar hammer-on technique", 
    "guitar pull-off technique",
    "guitar string bending",
    "guitar vibrato technique",
    "guitar palm muting",
    "clean guitar playing",
    "distorted electric guitar",
    "acoustic guitar fingerpicking",
    "guitar power chords"
]