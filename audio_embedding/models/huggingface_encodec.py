import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import numpy as np

from transformers import EncodecModel, EncodecFeatureExtractor


class HuggingFaceEncodec(nn.Module):
    """
    HuggingFace Encodec wrapper for audio compression and embedding extraction.
    
    Uses Meta's pretrained Encodec models optimized for music and speech.
    Integrates seamlessly with the guitar transcription pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        target_bandwidth: float = 6.0,  # kbps
        output_dim: int = 128,
        freeze_model: bool = True
    ):
        super().__init__()
        
        
        self.model_name = model_name
        self.target_bandwidth = target_bandwidth
        self.output_dim = output_dim
        
        # Load pretrained Encodec model
        print(f"Loading Encodec model: {model_name}")
        self.encodec_model = EncodecModel.from_pretrained(model_name)
        self.encodec_processor = EncodecFeatureExtractor.from_pretrained(model_name)
        
        # Get model specifications
        self.sample_rate = self.encodec_processor.sampling_rate
        # The actual output dimension is the number of codebooks (8 for facebook/encodec_24khz)
        # The audio_codes shape is [batch, channels, n_codebooks, time]
        # After processing, we get [time, n_codebooks], so dimension is n_codebooks = 8
        self.encodec_dim = 8  # Number of codebooks in RVQ
        
        # Store target bandwidth (used in encode method)
        # Note: HuggingFace EncodecModel doesn't have set_target_bandwidth method
        
        # Projection layer if dimensions don't match
        if self.encodec_dim != output_dim:
            self.projection = nn.Linear(self.encodec_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze pretrained weights if specified
        if freeze_model:
            for param in self.encodec_model.parameters():
                param.requires_grad = False
        
        print(f"Encodec loaded: {self.encodec_dim} → {output_dim} dim")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Target bandwidth: {target_bandwidth} kbps")
    
    def _resample_audio(self, audio: torch.Tensor, source_sr: int) -> torch.Tensor:
        """
        Resample audio to Encodec's expected sample rate.
        
        Args:
            audio: Input audio tensor
            source_sr: Source sample rate
            
        Returns:
            Resampled audio tensor
        """
        if source_sr == self.sample_rate:
            return audio
        
        # Simple linear interpolation resampling
        import torch.nn.functional as F
        
        ratio = self.sample_rate / source_sr
        target_length = int(audio.shape[-1] * ratio)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
            resampled = F.interpolate(
                audio, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).squeeze()
        else:
            # Handle batch dimension
            original_shape = audio.shape
            audio = audio.view(-1, 1, audio.shape[-1])  # [batch, 1, time]
            resampled = F.interpolate(
                audio,
                size=target_length,
                mode='linear',
                align_corners=False
            ).view(original_shape[:-1] + (target_length,))
        
        return resampled
    
    def encode(
        self, 
        audio: torch.Tensor, 
        source_sample_rate: int = 22050
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to compressed codes and embeddings.
        
        Args:
            audio: Input audio [batch, time] or [time]
            source_sample_rate: Sample rate of input audio
            
        Returns:
            codes: Compressed codes [batch, time_compressed, n_codebooks]
            embeddings: Audio embeddings [batch, time_compressed, output_dim]
        """
        # Handle single audio input
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        
        # Resample if needed
        if source_sample_rate != self.sample_rate:
            audio = self._resample_audio(audio, source_sample_rate)
        
        codes_list = []
        embeddings_list = []
        
        # Process each audio in batch
        for i in range(batch_size):
            audio_sample = audio[i].detach().cpu().numpy()
            
            # Process through Encodec processor
            inputs = self.encodec_processor(
                raw_audio=audio_sample,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # Move to correct device
            inputs = {k: v.to(audio.device) for k, v in inputs.items()}
            
            # Encode through Encodec
            with torch.no_grad() if not self.training else torch.enable_grad():
                encoder_outputs = self.encodec_model.encode(
                    inputs["input_values"],
                    inputs.get("padding_mask", None),
                    bandwidth=self.target_bandwidth
                )
                
                # Get codes and embeddings
                audio_codes = encoder_outputs.audio_codes  # [1, channels, n_codebooks, time]
                
                # Reshape codes to [time, n_codebooks] 
                # audio_codes shape is [batch, channels, n_codebooks, time]
                codes = audio_codes.squeeze(0).squeeze(0).transpose(0, 1)  # [time, n_codebooks]
                
                # For embeddings, we need to decode the codes to get latent representation
                # Since we don't have direct access to encoder latents, use the codes as embeddings
                # Flatten the codes for embedding use
                embeddings = codes.float()  # [time, n_codebooks]
                
                codes_list.append(codes)
                embeddings_list.append(embeddings)
        
        # Stack batch results
        codes_batch = torch.stack(codes_list, dim=0)  # [batch, time, n_codebooks]
        embeddings_batch = torch.stack(embeddings_list, dim=0)  # [batch, time, channels]
        
        # Project embeddings to target dimension
        embeddings_projected = self.projection(embeddings_batch)
        
        return codes_batch, embeddings_projected
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode compressed codes back to audio.
        
        Args:
            codes: Compressed codes [batch, time, n_codebooks]
            
        Returns:
            Reconstructed audio [batch, time]
        """
        batch_size = codes.shape[0]
        reconstructed_audio = []
        
        for i in range(batch_size):
            # Reshape codes for Encodec: [1, n_codebooks, time]
            codes_sample = codes[i].transpose(0, 1).unsqueeze(0)
            
            with torch.no_grad():
                # Decode through Encodec
                decoder_outputs = self.encodec_model.decode(
                    codes_sample,
                    audio_scales=None  # Use default scaling
                )
                
                # Get reconstructed audio
                audio_values = decoder_outputs.audio_values  # [1, 1, time]
                reconstructed_audio.append(audio_values.squeeze())
        
        return torch.stack(reconstructed_audio, dim=0)
    
    def forward(
        self, 
        audio: torch.Tensor, 
        source_sample_rate: int = 22050
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode audio to codes and embeddings.
        
        Args:
            audio: Input audio [batch, time] or [time]
            source_sample_rate: Sample rate of input audio
            
        Returns:
            codes: Compressed codes [batch, time_compressed, n_codebooks]
            embeddings: Audio embeddings [batch, time_compressed, output_dim]
        """
        return self.encode(audio, source_sample_rate)
    
    def get_compression_info(self, audio_length_samples: int, source_sr: int = 22050) -> Dict[str, float]:
        """
        Get compression statistics for given audio length.
        
        Args:
            audio_length_samples: Length of audio in samples
            source_sr: Source sample rate
            
        Returns:
            Dictionary with compression statistics
        """
        # Calculate compressed length (approximate)
        hop_length = self.sample_rate // 75  # Encodec typically uses ~75 Hz frame rate
        resampled_length = int(audio_length_samples * self.sample_rate / source_sr)
        compressed_frames = resampled_length // hop_length
        
        # Calculate compression ratio
        original_bits = audio_length_samples * 16  # Assume 16-bit audio
        compressed_bits = compressed_frames * self.target_bandwidth * 1000 / 75  # ~75 fps
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'original_samples': audio_length_samples,
            'compressed_frames': compressed_frames,
            'compression_ratio': compression_ratio,
            'bandwidth_kbps': self.target_bandwidth,
            'frame_rate_hz': 75.0  # Approximate
        }
    
    def reconstruct_audio(
        self, 
        audio: torch.Tensor, 
        source_sample_rate: int = 22050
    ) -> torch.Tensor:
        """
        Full reconstruction: encode then decode audio.
        
        Args:
            audio: Input audio [batch, time] or [time]
            source_sample_rate: Sample rate of input audio
            
        Returns:
            Reconstructed audio [batch, time]
        """
        codes, _ = self.encode(audio, source_sample_rate)
        reconstructed = self.decode(codes)
        
        # Resample back to original sample rate if needed
        if source_sample_rate != self.sample_rate:
            reconstructed = self._resample_audio(reconstructed, self.sample_rate)
            # Note: This reverses the resampling direction
            ratio = source_sample_rate / self.sample_rate
            target_length = int(reconstructed.shape[-1] * ratio)
            
            import torch.nn.functional as F
            if reconstructed.dim() == 1:
                reconstructed = reconstructed.unsqueeze(0).unsqueeze(0)
                reconstructed = F.interpolate(
                    reconstructed, 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).squeeze()
            else:
                original_shape = reconstructed.shape
                reconstructed = reconstructed.view(-1, 1, reconstructed.shape[-1])
                reconstructed = F.interpolate(
                    reconstructed,
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).view(original_shape[:-1] + (target_length,))
        
        return reconstructed


# Available Encodec models
ENCODEC_MODELS = {
    "facebook/encodec_24khz": {
        "description": "24kHz model optimized for music and speech",
        "sample_rate": 24000,
        "recommended_bandwidth": [1.5, 3.0, 6.0, 12.0]
    },
    "facebook/encodec_48khz": {
        "description": "48kHz model for high-quality audio",
        "sample_rate": 48000,
        "recommended_bandwidth": [3.0, 6.0, 12.0, 24.0]
    }
}


def list_available_models():
    """List available pretrained Encodec models."""
    print("Available HuggingFace Encodec Models:")
    for model_name, info in ENCODEC_MODELS.items():
        print(f"\n{model_name}:")
        print(f"  Description: {info['description']}")
        print(f"  Sample Rate: {info['sample_rate']} Hz")
        print(f"  Recommended Bandwidths: {info['recommended_bandwidth']} kbps")


if __name__ == "__main__":
    # Demo usage
    list_available_models()
    
    # Test with synthetic audio
    print("\nTesting HuggingFace Encodec...")
    try:
        encodec = HuggingFaceEncodec(
            model_name="facebook/encodec_24khz",
            target_bandwidth=6.0,
            output_dim=128
        )
        
        # Create test audio
        test_audio = torch.randn(1, 22050)  # 1 second at 22kHz
        print(f"Input audio shape: {test_audio.shape}")
        
        # Encode
        codes, embeddings = encodec(test_audio, source_sample_rate=22050)
        print(f"Codes shape: {codes.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Get compression info
        info = encodec.get_compression_info(22050, 22050)
        print(f"Compression ratio: {info['compression_ratio']:.1f}x")
        
        print("✅ HuggingFace Encodec test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")