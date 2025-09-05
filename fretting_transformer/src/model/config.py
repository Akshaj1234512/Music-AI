"""
Model Configuration for Fretting Transformer

Defines the T5 architecture configuration based on the paper specifications:
- Reduced T5 architecture (half of t5-small)
- d_model = 128, d_ff = 1024
- 3 encoder-decoder layers
- 4 attention heads
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import T5Config


@dataclass
class FrettingTransformerConfig:
    """
    Configuration for the Fretting Transformer based on T5 architecture.
    
    Based on paper specifications:
    - Model dimension: 128
    - Feedforward dimension: 1024  
    - 3 encoder and 3 decoder layers
    - 4 attention heads per layer
    """
    
    # Model architecture
    d_model: int = 128
    d_kv: int = 32  # Key/value dimension (d_model // num_heads)
    d_ff: int = 1024
    num_layers: int = 3
    num_decoder_layers: Optional[int] = None  # Same as encoder if None
    num_heads: int = 4
    
    # Vocabulary sizes (will be set from tokenizer)
    vocab_size: int = 0  # Input vocabulary size
    decoder_vocab_size: int = 0  # Output vocabulary size
    
    # Sequence lengths
    max_length: int = 512
    
    # Dropout and regularization
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    
    # Initialization
    initializer_factor: float = 1.0
    
    # Training specific
    use_cache: bool = True
    is_encoder_decoder: bool = True
    
    # Special tokens (will be set from tokenizer)
    pad_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 1  # BOS token
    
    def to_t5_config(self) -> T5Config:
        """
        Convert to HuggingFace T5Config object.
        
        Returns:
            T5Config object with our specifications
        """
        
        # Ensure decoder layers match encoder layers if not specified
        num_decoder_layers = self.num_decoder_layers or self.num_layers
        
        config = T5Config(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            d_kv=self.d_kv,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=self.num_heads,
            relative_attention_num_buckets=32,  # Standard T5 value
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_factor=self.initializer_factor,
            feed_forward_proj="relu",  # Standard T5 activation
            is_encoder_decoder=self.is_encoder_decoder,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            max_length=self.max_length,
        )
        
        # For T5, decoder vocabulary size is same as encoder
        if self.decoder_vocab_size > 0:
            # T5 typically uses the same vocab for encoder and decoder
            # But we handle different vocabs by setting decoder_vocab_size
            config.decoder_vocab_size = self.decoder_vocab_size
        
        return config
    
    @classmethod
    def from_vocab_sizes(cls, input_vocab_size: int, output_vocab_size: int, **kwargs) -> 'FrettingTransformerConfig':
        """
        Create config with vocabulary sizes from tokenizer.
        
        Args:
            input_vocab_size: Size of input (MIDI) vocabulary
            output_vocab_size: Size of output (tablature) vocabulary
            **kwargs: Additional config overrides
            
        Returns:
            FrettingTransformerConfig instance
        """
        config = cls(**kwargs)
        config.vocab_size = input_vocab_size
        config.decoder_vocab_size = output_vocab_size
        
        return config
    
    def get_model_size_info(self) -> Dict[str, Any]:
        """
        Calculate approximate model size information.
        
        Returns:
            Dictionary with model size statistics
        """
        # Calculate approximate parameter count
        # This is a rough estimate based on T5 architecture
        
        # Embedding parameters
        encoder_embed_params = self.vocab_size * self.d_model
        decoder_embed_params = self.decoder_vocab_size * self.d_model
        
        # Encoder parameters (per layer)
        encoder_layer_params = (
            # Self-attention
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            # Feed-forward  
            2 * self.d_model * self.d_ff +
            # Layer norms
            4 * self.d_model
        )
        
        # Decoder parameters (per layer)
        decoder_layer_params = (
            # Self-attention
            4 * self.d_model * self.d_model +
            # Cross-attention
            4 * self.d_model * self.d_model +
            # Feed-forward
            2 * self.d_model * self.d_ff +
            # Layer norms  
            6 * self.d_model
        )
        
        total_encoder_params = encoder_embed_params + self.num_layers * encoder_layer_params
        total_decoder_params = decoder_embed_params + (self.num_decoder_layers or self.num_layers) * decoder_layer_params
        
        total_params = total_encoder_params + total_decoder_params
        
        return {
            'total_parameters': total_params,
            'encoder_parameters': total_encoder_params,
            'decoder_parameters': total_decoder_params,
            'embedding_parameters': encoder_embed_params + decoder_embed_params,
            'parameters_millions': total_params / 1e6,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'input_vocab_size': self.vocab_size,
            'output_vocab_size': self.decoder_vocab_size
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        
        config_dict = {
            'd_model': self.d_model,
            'd_kv': self.d_kv,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'num_heads': self.num_heads,
            'vocab_size': self.vocab_size,
            'decoder_vocab_size': self.decoder_vocab_size,
            'max_length': self.max_length,
            'dropout_rate': self.dropout_rate,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'initializer_factor': self.initializer_factor,
            'use_cache': self.use_cache,
            'is_encoder_decoder': self.is_encoder_decoder,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'decoder_start_token_id': self.decoder_start_token_id
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FrettingTransformerConfig':
        """Load configuration from JSON file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


def create_paper_config(input_vocab_size: int, output_vocab_size: int) -> FrettingTransformerConfig:
    """
    Create the exact configuration used in the paper.
    
    Args:
        input_vocab_size: Input vocabulary size from tokenizer
        output_vocab_size: Output vocabulary size from tokenizer
        
    Returns:
        FrettingTransformerConfig matching paper specs
    """
    return FrettingTransformerConfig.from_vocab_sizes(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        d_model=128,
        d_kv=32,  # 128 / 4 heads
        d_ff=1024,
        num_layers=3,
        num_decoder_layers=3,
        num_heads=4,
        max_length=512,
        dropout_rate=0.1
    )


def create_debug_config(input_vocab_size: int, output_vocab_size: int) -> FrettingTransformerConfig:
    """
    Create a smaller configuration for debugging/testing.
    
    Args:
        input_vocab_size: Input vocabulary size from tokenizer
        output_vocab_size: Output vocabulary size from tokenizer
        
    Returns:
        Smaller FrettingTransformerConfig for testing
    """
    return FrettingTransformerConfig.from_vocab_sizes(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        d_model=64,
        d_kv=16,  # 64 / 4 heads
        d_ff=256,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        max_length=256,
        dropout_rate=0.1
    )


if __name__ == "__main__":
    # Test configuration
    config = create_paper_config(input_vocab_size=1000, output_vocab_size=800)
    
    print("Fretting Transformer Configuration:")
    print(f"d_model: {config.d_model}")
    print(f"d_ff: {config.d_ff}")
    print(f"num_layers: {config.num_layers}")
    print(f"num_heads: {config.num_heads}")
    
    model_info = config.get_model_size_info()
    print(f"\nModel size: {model_info['parameters_millions']:.2f}M parameters")
    print(f"Encoder params: {model_info['encoder_parameters']:,}")
    print(f"Decoder params: {model_info['decoder_parameters']:,}")
    
    # Test T5Config conversion
    t5_config = config.to_t5_config()
    print(f"\nT5Config vocab_size: {t5_config.vocab_size}")
    print(f"T5Config d_model: {t5_config.d_model}")
    print(f"T5Config num_layers: {t5_config.num_layers}")