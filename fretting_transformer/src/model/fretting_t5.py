"""
Fretting Transformer Model Implementation

T5-based encoder-decoder model for MIDI-to-tablature transcription.
Handles the specific vocabulary and generation requirements for guitar tablature.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

from .config import FrettingTransformerConfig


class FrettingT5Model(nn.Module):
    """
    T5-based model for MIDI-to-tablature transcription.
    
    Wraps HuggingFace T5ForConditionalGeneration with custom configuration
    and utilities specific to guitar tablature generation.
    """
    
    def __init__(self, config: FrettingTransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Convert to T5Config
        t5_config = config.to_t5_config()
        
        # Initialize T5 model from scratch
        self.model = T5ForConditionalGeneration(t5_config)
        
        # Store vocabulary information
        self.input_vocab_size = config.vocab_size
        self.output_vocab_size = config.decoder_vocab_size
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights following T5 initialization."""
        # T5 uses normal initialization with factor scaling
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Forward pass through the model.
        
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
                num_beams: int = 4,
                early_stopping: bool = True,
                do_sample: bool = False,
                temperature: float = 1.0,
                top_p: float = 1.0,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """
        Generate tablature sequences from MIDI input.
        
        Args:
            input_ids: Input MIDI token sequences
            attention_mask: Attention mask for input
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop at EOS token
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token sequences
        """
        
        # Set default values
        if max_length is None:
            max_length = self.config.max_length
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
    
    def encode(self, 
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input MIDI sequences.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Encoded representations
        """
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return encoder_outputs.last_hidden_state
    
    def get_encoder_decoder_states(self, 
                                 input_ids: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 decoder_input_ids: Optional[torch.Tensor] = None,
                                 decoder_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get encoder and decoder hidden states for analysis.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Input attention mask
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Decoder attention mask
            
        Returns:
            Tuple of (encoder_states, decoder_states)
        """
        
        # Get encoder states
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_states = encoder_outputs.last_hidden_state
        
        # Get decoder states if decoder inputs provided
        decoder_states = None
        if decoder_input_ids is not None:
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=attention_mask
            )
            decoder_states = decoder_outputs.last_hidden_state
        
        return encoder_states, decoder_states
    
    def calculate_perplexity(self,
                           input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate perplexity for a batch of sequences.
        
        Args:
            input_ids: Input token sequences
            labels: Target token sequences
            attention_mask: Attention mask
            
        Returns:
            Average perplexity
        """
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
        return perplexity.item()
    
    def save_model(self, save_path: str):
        """
        Save the model and configuration.
        
        Args:
            save_path: Directory to save model
        """
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save the T5 model
        self.model.save_pretrained(save_path)
        
        # Save our custom config
        config_path = os.path.join(save_path, 'fretting_config.json')
        self.config.save(config_path)
    
    @classmethod
    def load_model(cls, load_path: str) -> 'FrettingT5Model':
        """
        Load a saved model.
        
        Args:
            load_path: Directory containing saved model
            
        Returns:
            Loaded FrettingT5Model instance
        """
        import os
        
        # Load custom config
        config_path = os.path.join(load_path, 'fretting_config.json')
        config = FrettingTransformerConfig.load(config_path)
        
        # Create model instance
        model = cls(config)
        
        # Load T5 weights
        model.model = T5ForConditionalGeneration.from_pretrained(load_path)
        
        return model
    
    def get_model_info(self) -> Dict[str, Union[int, float]]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_millions': total_params / 1e6,
            'input_vocab_size': self.input_vocab_size,
            'output_vocab_size': self.output_vocab_size,
            'd_model': self.config.d_model,
            'd_ff': self.config.d_ff,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'max_length': self.config.max_length
        }
    
    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
    
    def freeze_decoder(self):
        """Freeze decoder parameters for fine-tuning."""
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        for param in self.model.lm_head.parameters():
            param.requires_grad = True


def create_model_from_tokenizer(tokenizer, config_type: str = 'paper') -> FrettingT5Model:
    """
    Create a FrettingT5Model from a tokenizer.
    
    Args:
        tokenizer: FrettingTokenizer instance
        config_type: 'paper' for full model, 'debug' for smaller model
        
    Returns:
        Initialized FrettingT5Model
    """
    from .config import create_paper_config, create_debug_config
    
    input_vocab_size, output_vocab_size = tokenizer.get_vocab_sizes()
    
    if config_type == 'paper':
        config = create_paper_config(input_vocab_size, output_vocab_size)
    elif config_type == 'debug':
        config = create_debug_config(input_vocab_size, output_vocab_size)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")
    
    # Set token IDs from tokenizer
    config.pad_token_id = tokenizer.output_token_to_id[tokenizer.config.pad_token]
    config.eos_token_id = tokenizer.output_token_to_id[tokenizer.config.eos_token]
    config.decoder_start_token_id = tokenizer.output_token_to_id[tokenizer.config.bos_token]
    
    return FrettingT5Model(config)


def test_model():
    """Test the model creation and basic functionality."""
    from ..data.tokenizer import FrettingTokenizer
    
    # Create tokenizer
    tokenizer = FrettingTokenizer()
    
    # Create model
    model = create_model_from_tokenizer(tokenizer, 'debug')
    
    print("Model created successfully!")
    model_info = model.get_model_info()
    print(f"Model size: {model_info['parameters_millions']:.2f}M parameters")
    print(f"Input vocab size: {model_info['input_vocab_size']}")
    print(f"Output vocab size: {model_info['output_vocab_size']}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, model_info['input_vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, model_info['output_vocab_size'], (batch_size, seq_len))
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print(f"Loss: {outputs.loss.item():.4f}")
        print(f"Logits shape: {outputs.logits.shape}")
    
    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:1],  # Single sample
            attention_mask=attention_mask[:1],
            max_length=32,
            num_beams=2
        )
        print(f"Generated shape: {generated.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_model()