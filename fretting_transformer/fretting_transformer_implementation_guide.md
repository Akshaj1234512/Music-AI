# Fretting-Transformer Implementation Guide

## Overview

This document provides a complete specification for implementing the Fretting-Transformer system for MIDI-to-guitar tablature transcription, based on the Hamberger et al. paper and SynthTab dataset integration.

## System Architecture

### Core Approach
The Fretting-Transformer treats tablature generation as a **text-to-text translation problem** using a T5 encoder-decoder architecture. The system converts MIDI sequences into text tokens, then generates corresponding tablature tokens.

### Model Specifications
```
Architecture: T5 Encoder-Decoder
Model dimension (d_model): 128
Feedforward dimension (d_ff): 1024
Encoder layers: 3
Decoder layers: 3
Attention heads: 4
Optimizer: Adafactor with self-adaptive learning rate
Training sequence length: 512 tokens
Inference chunk size: 20 notes (100 tokens)
```

## Data Processing Pipeline

### Input Data Format
The system expects MIDI files with the following information extracted:
- Start time (in ticks)
- End time (in ticks)
- MIDI pitch number
- String assignment (1-6, where 1 = high E string)
- Fret position (0-24, where 0 = open string)

### Tokenization Scheme

#### Input Tokens (MIDI Events)
```
NOTE_ON<pitch>     # MIDI note start, pitch 0-127
NOTE_OFF<pitch>    # MIDI note end, pitch 0-127
TIME_SHIFT<ticks>  # Time duration in MIDI ticks
CAPO<position>     # Capo position 0-7 (conditional model only)
TUNING<E,A,D,G,B,E> # String tunings in MIDI numbers (conditional model only)
```

#### Output Tokens (Tablature)
```
TAB<string,fret>   # String (1-6) and fret (0-24) combination
TIME_SHIFT<ticks>  # Time duration matching input
```

#### Example Tokenization
**Input:** Single note G3 (MIDI 55) on 3rd string, open fret, held for 120 ticks
```
NOTE_ON<55> TIME_SHIFT<120> NOTE_OFF<55>
```

**Output:** Corresponding tablature
```
TAB<3,0> TIME_SHIFT<120>
```

### Data Preprocessing Steps

1. **MIDI Extraction**
   ```python
   # Extract from MIDI file
   notes = []
   for note in midi_track:
       notes.append({
           'start_time': note.start,
           'end_time': note.end, 
           'pitch': note.pitch,
           'string': note.string,    # From Guitar Pro format
           'fret': note.fret        # From Guitar Pro format
       })
   ```

2. **Sequence Generation**
   ```python
   # Convert to token sequences
   input_tokens = []
   output_tokens = []
   
   for note in sorted_notes:
       # Input sequence
       input_tokens.extend([
           f"NOTE_ON<{note.pitch}>",
           f"TIME_SHIFT<{note.duration}>", 
           f"NOTE_OFF<{note.pitch}>"
       ])
       
       # Output sequence  
       output_tokens.extend([
           f"TAB<{note.string},{note.fret}>",
           f"TIME_SHIFT<{note.duration}>"
       ])
   ```

3. **Chunking for Training**
   - Split sequences into 512-token chunks
   - Maintain note boundaries (don't split individual notes)
   - Add padding tokens as needed

## Model Training

### Training Configuration
```python
training_config = {
    'batch_size': 32,
    'learning_rate': 'adaptive',  # Adafactor handles this
    'epochs': 100,
    'gradient_accumulation_steps': 4,
    'warmup_steps': 1000,
    'max_sequence_length': 512
}
```

### Training Procedure
1. **Standard Model**: Train on standard tuning, no capo
2. **Conditional Model**: Train with CAPO and TUNING tokens

```python
# Training loop pseudocode
for epoch in range(100):
    for batch in dataloader:
        input_ids = tokenizer(batch['input_text'])
        target_ids = tokenizer(batch['target_text'])
        
        outputs = model(
            input_ids=input_ids,
            labels=target_ids
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

### Data Augmentation Strategy
```python
# Capo augmentation: transpose all pitches by capo amount
def augment_capo(notes, capo_position):
    augmented_notes = []
    for note in notes:
        new_pitch = note.pitch + capo_position
        if new_pitch <= 127:  # Valid MIDI range
            augmented_notes.append({
                **note,
                'pitch': new_pitch
            })
    return augmented_notes

# Tuning augmentation: common alternate tunings
TUNINGS = {
    'standard': [64, 59, 55, 50, 45, 40],      # E A D G B E
    'half_step_down': [63, 58, 54, 49, 44, 39], # Eb Ab Db Gb Bb Eb  
    'full_step_down': [62, 57, 53, 48, 43, 38], # D G C F A D
    'drop_d': [62, 59, 55, 50, 45, 40]          # D A D G B E
}
```

## Inference Implementation

### Chunked Inference with Context
```python
def generate_tablature(model, tokenizer, midi_sequence, chunk_size=20):
    notes = extract_notes_from_midi(midi_sequence)
    chunks = create_chunks(notes, chunk_size)
    
    all_outputs = []
    previous_context = []
    
    for i, chunk in enumerate(chunks):
        # Add context from previous chunk
        if i > 0:
            input_tokens = previous_context + chunk
        else:
            input_tokens = chunk
            
        # Generate output
        input_ids = tokenizer.encode(input_tokens)
        output_ids = model.generate(
            input_ids,
            max_length=len(input_ids) + 100,
            num_beams=4,
            early_stopping=True
        )
        
        chunk_output = tokenizer.decode(output_ids)
        all_outputs.append(chunk_output)
        
        # Save context for next chunk (last note tokens)
        previous_context = get_last_note_tokens(input_tokens)
    
    return combine_outputs(all_outputs)
```

## Post-Processing (Critical for Accuracy)

### Pitch Validation and Correction
```python
def post_process_tablature(input_notes, predicted_tablature, tuning=[64, 59, 55, 50, 45, 40]):
    """
    From paper: Post-processing achieves 100% pitch accuracy
    Ensures generated tablature produces correct pitches
    """
    corrected_tablature = []
    
    for i, (input_note, pred_tab_token) in enumerate(zip(input_notes, predicted_tablature)):
        if pred_tab_token.startswith('TAB<'):
            # Parse predicted string/fret
            string, fret = parse_tab_token(pred_tab_token)
            
            # Calculate actual pitch from string/fret
            if 1 <= string <= 6 and 0 <= fret <= 24:
                actual_pitch = tuning[string-1] + fret
                
                if actual_pitch == input_note['pitch']:
                    # Correct prediction
                    corrected_tablature.append(pred_tab_token)
                else:
                    # Find alternative fingering
                    alternative = find_valid_fingering(input_note['pitch'], tuning)
                    corrected_tablature.append(alternative)
            else:
                # Invalid string/fret, find alternative
                alternative = find_valid_fingering(input_note['pitch'], tuning)
                corrected_tablature.append(alternative)
        else:
            # Non-tablature token (TIME_SHIFT), keep as-is
            corrected_tablature.append(pred_tab_token)
    
    return corrected_tablature

def find_valid_fingering(target_pitch, tuning=[64, 59, 55, 50, 45, 40]):
    """Find valid string/fret combination for target pitch"""
    for string_idx, open_pitch in enumerate(tuning):
        fret = target_pitch - open_pitch
        if 0 <= fret <= 24:  # Valid fret range
            return f"TAB<{string_idx+1},{fret}>"
    
    # Fallback: use highest string possible
    return f"TAB<1,{max(0, min(24, target_pitch - tuning[0]))}>"

def parse_tab_token(tab_token):
    """Extract string and fret from TAB<string,fret> token"""
    # Remove 'TAB<' and '>'
    content = tab_token[4:-1]
    string, fret = content.split(',')
    return int(string), int(fret)
```

## Implementation Requirements

### Required Libraries
```python
# Essential dependencies only
torch >= 1.9.0
transformers >= 4.21.0
mido >= 1.2.0  # MIDI file processing

# Optional for evaluation/analysis
datasets >= 2.0.0
```

### Minimal File Structure
```
project/
├── data/
│   ├── train/              # Training MIDI-tablature pairs
│   └── test/               # Test data
├── models/
│   └── fretting_transformer/ # Saved model checkpoints
├── src/
│   ├── data_processing.py  # MIDI loading and tokenization
│   ├── model.py           # T5 model setup  
│   ├── training.py        # Training script
│   └── inference.py       # Generation and post-processing
└── train.py               # Main training script
```

## Critical Implementation Notes

1. **Post-Processing is Essential**: The paper shows post-processing improves accuracy from ~68% to 72%+. Always validate generated tablature produces correct pitches.

2. **Context Preservation**: Include previous chunk context during inference to maintain musical coherence across boundaries.

3. **Token Boundary Respect**: Never split NOTE_ON/TIME_SHIFT/NOTE_OFF sequences when chunking for training.

4. **Standard Tuning Focus**: Start with standard tuning only. Add alternate tunings after basic system works.

5. **Data Format Validation**: Ensure your MIDI-tablature pairs are properly aligned before training.

This specification provides the core implementation requirements based directly on the Hamberger et al. paper, focusing on proven techniques and avoiding speculative additions.