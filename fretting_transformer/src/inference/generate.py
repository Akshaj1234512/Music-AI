"""
Inference System for Fretting Transformer

Implements chunked generation with context preservation as described in the paper.
Processes input MIDI sequences in chunks of 20 notes (~100 tokens) to handle long sequences.
"""

import torch
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from ..model.fretting_t5 import FrettingT5Model
from ..data.tokenizer import FrettingTokenizer
from ..data.synthtab_loader import Note


@dataclass
class GenerationConfig:
    """Configuration for generation process."""
    
    # Chunking parameters (from paper)
    chunk_size_notes: int = 20  # Notes per chunk
    chunk_size_tokens: int = 100  # Approximate tokens per chunk
    context_overlap_tokens: int = 20  # Tokens to preserve from previous chunk
    
    # Generation parameters
    max_length: int = 512
    num_beams: int = 4
    early_stopping: bool = True
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    
    # Post-processing
    apply_postprocessing: bool = True
    
    # Device
    device: Optional[str] = None


class ChunkedInference:
    """
    Handles chunked inference for long sequences.
    
    From paper: "During inference, chunks of 20 notes are processed.
    The tokens from the last note of the previous chunk are placed at the 
    beginning of the following sequence in both the encoder and decoder 
    to preserve context between the chunks."
    """
    
    def __init__(self, 
                 model: FrettingT5Model,
                 tokenizer: FrettingTokenizer,
                 config: Optional[GenerationConfig] = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        # Set device
        if self.config.device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(self.config.device)
            
        self.model.to(self.device)
        self.model.eval()
    
    def generate_tablature(self, 
                          midi_events: List[Dict],
                          capo: int = 0,
                          tuning: Optional[List[int]] = None) -> Tuple[List[str], List[str]]:
        """
        Generate tablature from MIDI events using chunked inference.
        
        Args:
            midi_events: List of MIDI event dictionaries
            capo: Capo position (0-7)
            tuning: Guitar tuning as MIDI numbers [E,A,D,G,B,E]
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        
        # Tokenize full input sequence
        input_tokens = self._prepare_input_tokens(midi_events, capo, tuning)
        
        # Chunk the input sequence
        input_chunks = self._chunk_input_sequence(input_tokens)
        
        # Generate outputs for each chunk with context
        output_chunks = []
        previous_context = []
        
        for i, chunk in enumerate(input_chunks):
            # Add context from previous chunk
            if i > 0:
                chunk_with_context = previous_context + chunk
            else:
                chunk_with_context = chunk
            
            # Generate output for this chunk
            chunk_output = self._generate_chunk(chunk_with_context)
            
            # Extract new tokens (remove context overlap if present)
            if i > 0:
                # Remove tokens that correspond to the context overlap
                new_output = chunk_output[len(previous_context):]
            else:
                new_output = chunk_output
            
            output_chunks.append(new_output)
            
            # Prepare context for next chunk (last note's tokens)
            previous_context = self._extract_context(chunk, chunk_output)
        
        # Combine all output chunks
        full_output = self._combine_chunks(output_chunks)
        
        return input_tokens, full_output
    
    def _prepare_input_tokens(self, 
                            midi_events: List[Dict],
                            capo: int = 0,
                            tuning: Optional[List[int]] = None) -> List[str]:
        """
        Prepare input tokens with optional conditioning.
        
        Args:
            midi_events: MIDI events to tokenize
            capo: Capo position
            tuning: Guitar tuning
            
        Returns:
            List of input tokens
        """
        tokens = [self.tokenizer.config.bos_token]
        
        # Add conditioning tokens if provided
        if capo > 0:
            tokens.append(f"CAPO<{capo}>")
            
        if tuning is not None:
            tuning_str = ','.join(map(str, tuning))
            tokens.append(f"TUNING<{tuning_str}>")
        
        # Add MIDI event tokens
        midi_tokens = self.tokenizer.encode_midi_events(midi_events)
        tokens.extend(midi_tokens[1:-1])  # Remove BOS/EOS from midi tokens
        
        tokens.append(self.tokenizer.config.eos_token)
        
        return tokens
    
    def _chunk_input_sequence(self, input_tokens: List[str]) -> List[List[str]]:
        """
        Chunk input sequence based on note boundaries.
        
        Args:
            input_tokens: Full input token sequence
            
        Returns:
            List of token chunks
        """
        # Find note boundaries (NOTE_ON tokens)
        note_positions = []
        for i, token in enumerate(input_tokens):
            if token.startswith('NOTE_ON<'):
                note_positions.append(i)
        
        # Create chunks based on note count
        chunks = []
        start_idx = 0
        
        while start_idx < len(input_tokens):
            # Find end position for this chunk
            notes_in_chunk = 0
            end_idx = start_idx
            
            # Count notes in current position onwards
            for i in range(start_idx, len(input_tokens)):
                if input_tokens[i].startswith('NOTE_ON<'):
                    notes_in_chunk += 1
                    
                if notes_in_chunk >= self.config.chunk_size_notes:
                    # Find end of this note (next NOTE_ON or end of sequence)
                    end_idx = i + 1
                    while (end_idx < len(input_tokens) and 
                           not input_tokens[end_idx].startswith('NOTE_ON<')):
                        end_idx += 1
                    break
                end_idx = i + 1
            
            # Extract chunk
            chunk = input_tokens[start_idx:end_idx]
            if chunk:
                chunks.append(chunk)
            
            start_idx = end_idx
            
            # Prevent infinite loop
            if start_idx >= len(input_tokens):
                break
        
        return chunks
    
    def _generate_chunk(self, input_chunk: List[str]) -> List[str]:
        """
        Generate output for a single input chunk.
        
        Args:
            input_chunk: Input token chunk
            
        Returns:
            Generated output tokens
        """
        # Convert tokens to IDs
        input_ids = self.tokenizer.tokens_to_ids(input_chunk, 'input')
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_tensor)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_length=len(input_ids) + self.config.max_length,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.output_token_to_id[self.tokenizer.config.pad_token],
                eos_token_id=self.tokenizer.output_token_to_id[self.tokenizer.config.eos_token]
            )
        
        # Convert back to tokens
        generated_tokens = self.tokenizer.ids_to_tokens(
            generated_ids[0].cpu().tolist(), 'output'
        )
        
        # Remove special tokens
        output_tokens = []
        for token in generated_tokens:
            if token in [self.tokenizer.config.pad_token, 
                        self.tokenizer.config.eos_token,
                        self.tokenizer.config.bos_token]:
                continue
            output_tokens.append(token)
        
        return output_tokens
    
    def _extract_context(self, input_chunk: List[str], output_chunk: List[str]) -> List[str]:
        """
        Extract context tokens from the end of chunks for next iteration.
        
        From paper: "The tokens from the last note of the previous chunk 
        are placed at the beginning of the following sequence"
        
        Args:
            input_chunk: Input tokens for this chunk
            output_chunk: Generated output tokens
            
        Returns:
            Context tokens for next chunk
        """
        # Find the last NOTE_ON token in input
        last_note_idx = -1
        for i in range(len(input_chunk) - 1, -1, -1):
            if input_chunk[i].startswith('NOTE_ON<'):
                last_note_idx = i
                break
        
        if last_note_idx == -1:
            return []
        
        # Extract tokens from last note onwards in input
        input_context = input_chunk[last_note_idx:]
        
        # For output, take corresponding number of tokens from the end
        # This is an approximation - in practice we'd need alignment
        context_length = min(len(input_context), self.config.context_overlap_tokens)
        output_context = output_chunk[-context_length:] if output_chunk else []
        
        return input_context[:context_length]
    
    def _combine_chunks(self, output_chunks: List[List[str]]) -> List[str]:
        """
        Combine output chunks into a single sequence.
        
        Args:
            output_chunks: List of output token chunks
            
        Returns:
            Combined output token sequence
        """
        combined = []
        
        for chunk in output_chunks:
            combined.extend(chunk)
        
        return combined
    
    def generate_from_notes(self, notes: List[Note], 
                           capo: int = 0,
                           tuning: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """
        Generate tablature from Note objects.
        
        Args:
            notes: List of Note objects
            capo: Capo position
            tuning: Guitar tuning
            
        Returns:
            List of (string, fret) tuples
        """
        from ..data.synthtab_loader import SynthTabLoader
        
        # Convert notes to MIDI events
        loader = SynthTabLoader()
        midi_events = loader.notes_to_midi_sequence(notes)
        
        # Generate tablature
        input_tokens, output_tokens = self.generate_tablature(
            midi_events, capo, tuning
        )
        
        # Decode output tokens to (string, fret) pairs
        tab_pairs = self.tokenizer.decode_tablature_tokens(output_tokens)
        
        return tab_pairs
    
    def batch_generate(self, 
                      batch_midi_events: List[List[Dict]],
                      batch_capo: Optional[List[int]] = None,
                      batch_tuning: Optional[List[List[int]]] = None) -> List[List[str]]:
        """
        Generate tablature for a batch of sequences.
        
        Args:
            batch_midi_events: List of MIDI event sequences
            batch_capo: List of capo positions (optional)
            batch_tuning: List of tunings (optional)
            
        Returns:
            List of output token sequences
        """
        results = []
        
        for i, midi_events in enumerate(batch_midi_events):
            capo = batch_capo[i] if batch_capo else 0
            tuning = batch_tuning[i] if batch_tuning else None
            
            _, output_tokens = self.generate_tablature(midi_events, capo, tuning)
            results.append(output_tokens)
        
        return results


class StreamingInference:
    """
    Real-time streaming inference for live MIDI input.
    Maintains context across multiple generate calls.
    """
    
    def __init__(self, 
                 model: FrettingT5Model,
                 tokenizer: FrettingTokenizer,
                 config: Optional[GenerationConfig] = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        self.device = next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        
        # Streaming state
        self.context_tokens = []
        self.reset()
    
    def reset(self):
        """Reset streaming state."""
        self.context_tokens = [self.tokenizer.config.bos_token]
    
    def add_midi_events(self, midi_events: List[Dict]) -> List[str]:
        """
        Add new MIDI events and generate corresponding tablature.
        
        Args:
            midi_events: New MIDI events
            
        Returns:
            Generated tablature tokens for the new events
        """
        # Tokenize new events
        new_tokens = self.tokenizer.encode_midi_events(midi_events)
        new_tokens = new_tokens[1:-1]  # Remove BOS/EOS
        
        # Add to context
        self.context_tokens.extend(new_tokens)
        
        # Limit context size
        max_context = self.config.max_length - 100  # Leave room for generation
        if len(self.context_tokens) > max_context:
            # Keep most recent tokens
            self.context_tokens = (
                [self.tokenizer.config.bos_token] + 
                self.context_tokens[-max_context:]
            )
        
        # Generate for current context
        chunked_inference = ChunkedInference(self.model, self.tokenizer, self.config)
        
        # For streaming, we only process recent context
        recent_context = self.context_tokens[-self.config.chunk_size_tokens:]
        
        input_ids = self.tokenizer.tokens_to_ids(recent_context, 'input')
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_length=len(input_ids) + 50,  # Short generation for streaming
                num_beams=2,  # Faster for real-time
                early_stopping=True
            )
        
        # Convert to tokens and extract new output
        output_tokens = self.tokenizer.ids_to_tokens(
            generated_ids[0].cpu().tolist(), 'output'
        )
        
        # Return only the newly generated tokens
        new_output = [t for t in output_tokens if t not in [
            self.tokenizer.config.pad_token,
            self.tokenizer.config.bos_token,
            self.tokenizer.config.eos_token
        ]]
        
        return new_output[-len(new_tokens):]  # Return corresponding amount


def test_inference():
    """Test the inference system."""
    from ..model.fretting_t5 import create_model_from_tokenizer
    from ..data.tokenizer import FrettingTokenizer
    
    # Create components
    tokenizer = FrettingTokenizer()
    model = create_model_from_tokenizer(tokenizer, 'debug')
    
    # Create chunked inference
    inference = ChunkedInference(model, tokenizer)
    
    print("Inference system created successfully!")
    
    # Test with dummy MIDI events
    midi_events = [
        {'type': 'note_on', 'pitch': 55, 'string': 3, 'fret': 0},
        {'type': 'time_shift', 'delta': 120},
        {'type': 'note_off', 'pitch': 55, 'string': 3, 'fret': 0},
        {'type': 'note_on', 'pitch': 57, 'string': 3, 'fret': 2},
        {'type': 'time_shift', 'delta': 120},
        {'type': 'note_off', 'pitch': 57, 'string': 3, 'fret': 2}
    ]
    
    print("Testing generation...")
    input_tokens, output_tokens = inference.generate_tablature(midi_events)
    
    print(f"Input tokens: {len(input_tokens)}")
    print(f"Output tokens: {len(output_tokens)}")
    print(f"First few output tokens: {output_tokens[:10]}")
    
    # Test tablature decoding
    tab_pairs = tokenizer.decode_tablature_tokens(output_tokens)
    print(f"Generated tablature: {tab_pairs}")
    
    print("Inference test completed successfully!")


if __name__ == "__main__":
    test_inference()