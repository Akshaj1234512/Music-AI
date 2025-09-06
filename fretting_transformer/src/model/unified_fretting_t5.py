"""
Unified Fretting Transformer Model Implementation

T5-based encoder-decoder model for MIDI-to-tablature transcription using
unified vocabulary compatible with standard T5 architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

try:
    from .config import FrettingTransformerConfig
except ImportError:
    from config import FrettingTransformerConfig


class UnifiedFrettingT5Model(nn.Module):
    """
    T5-based model for MIDI-to-tablature transcription using unified vocabulary.
    
    This version uses standard T5ForConditionalGeneration with a single vocabulary
    containing both MIDI input tokens and TAB output tokens. This approach is
    compatible with standard T5 architecture and eliminates the vocabulary mismatch
    issues that prevented training.
    """
    
    def __init__(self, config: FrettingTransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Create T5 config with unified vocabulary - no custom configurations needed
        t5_config = T5Config(
            vocab_size=config.vocab_size,  # Unified vocabulary size (~468 tokens)
            d_model=config.d_model,
            d_kv=config.d_kv,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_decoder_layers=config.num_decoder_layers or config.num_layers,
            num_heads=config.num_heads,
            relative_attention_num_buckets=32,
            dropout_rate=config.dropout_rate,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_factor=config.initializer_factor,
            feed_forward_proj="relu",
            is_encoder_decoder=config.is_encoder_decoder,
            use_cache=config.use_cache,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            decoder_start_token_id=config.decoder_start_token_id,
            max_length=config.max_length,
        )
        
        # Create standard T5 model - no custom heads or embeddings needed!
        self.model = T5ForConditionalGeneration(t5_config)
        
        print(f"Created standard T5 model with unified vocabulary: {config.vocab_size} tokens")
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Standard T5 forward pass - no custom logic needed with unified vocabulary.
        
        Args:
            input_ids: Input token IDs (MIDI events)
            attention_mask: Mask for input padding
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Mask for decoder padding
            labels: Target token IDs (tablature events)
            return_dict: Whether to return ModelOutput or tuple
            
        Returns:
            Model outputs including loss and logits
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=return_dict
        )
    
    def generate(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_length: Optional[int] = None,
                max_new_tokens: Optional[int] = None,
                min_new_tokens: Optional[int] = None,
                num_beams: int = 4,
                early_stopping: bool = True,
                do_sample: bool = False,
                temperature: float = 1.0,
                top_p: float = 1.0,
                repetition_penalty: float = 1.0,
                **kwargs) -> torch.Tensor:
        """
        Generate tablature sequences from MIDI input using standard T5 generation.
        
        Args:
            input_ids: Input MIDI token sequences
            attention_mask: Attention mask for input
            max_length: Maximum generation length
            max_new_tokens: Maximum new tokens to generate
            min_new_tokens: Minimum new tokens to generate
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop at EOS token
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated token sequences
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Union[int, float]]:
        """
        Get model information including parameter count and configuration.
        
        Returns:
            Dictionary with model statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_millions': total_params / 1e6,
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'd_ff': self.config.d_ff,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'max_length': self.config.max_length,
        }


def create_model_from_tokenizer(tokenizer, config_type: str = 'paper') -> UnifiedFrettingT5Model:
    """
    Create a UnifiedFrettingT5Model from a unified tokenizer.
    
    Args:
        tokenizer: UnifiedFrettingTokenizer instance
        config_type: 'paper' for full model, 'debug' for smaller model
        
    Returns:
        Initialized UnifiedFrettingT5Model
    """
    try:
        from .config import create_paper_config, create_debug_config
    except ImportError:
        from config import create_paper_config, create_debug_config
    
    if config_type == 'paper':
        config = create_paper_config(tokenizer.vocab_size)
    elif config_type == 'debug':
        config = create_debug_config(tokenizer.vocab_size)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")
    
    # Set token IDs from unified tokenizer
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id
    
    return UnifiedFrettingT5Model(config)


def test_unified_model():
    """Test the unified model creation and basic functionality."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from unified_tokenizer import UnifiedFrettingTokenizer
    
    # Create unified tokenizer
    tokenizer = UnifiedFrettingTokenizer()
    
    # Create unified model
    model = create_model_from_tokenizer(tokenizer, 'debug')
    
    print("=== Unified Model Test ===")
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    
    # Create test input (MIDI tokens from unified vocab)
    input_ids = torch.randint(4, 300, (batch_size, seq_len))  # MIDI token range
    attention_mask = torch.ones_like(input_ids)
    
    # Create test output (TAB tokens from unified vocab)
    labels = torch.randint(300, 450, (batch_size, seq_len))  # TAB token range
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print(f"Output loss: {outputs.loss.item():.4f}")
        print(f"Expected loss (ln({tokenizer.vocab_size})): {torch.log(torch.tensor(float(tokenizer.vocab_size))).item():.4f}")
        print(f"Loss is reasonable: {abs(outputs.loss.item() - torch.log(torch.tensor(float(tokenizer.vocab_size))).item()) < 2}")
        
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Expected shape: [{batch_size}, {seq_len}, {tokenizer.vocab_size}]")


if __name__ == "__main__":
    test_unified_model()