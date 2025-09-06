#!/usr/bin/env python3
"""
Debug dataset processing to understand why no sequences are generated.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.synthtab_loader import SynthTabLoader
from data.unified_tokenizer import UnifiedFrettingTokenizer
from data.unified_dataset import UnifiedFrettingDataProcessor, UnifiedDataConfig

def debug_dataset_processing():
    """Debug step-by-step dataset processing."""
    
    print("=== Debug Dataset Processing ===")
    
    # 1. Test SynthTabLoader directly
    print("1. Testing SynthTabLoader...")
    loader = SynthTabLoader()
    
    # Find JAMS files
    jams_files = loader.find_jams_files('jams')
    print(f"   Found {len(jams_files)} JAMS files")
    if len(jams_files) > 0:
        print(f"   First file: {jams_files[0]}")
    
    # Load dataset
    raw_dataset = loader.load_dataset('jams', max_files=3)
    print(f"   Loaded {len(raw_dataset)} files with notes")
    
    for file_path, notes in raw_dataset.items():
        print(f"   {file_path}: {len(notes)} notes")
        if len(notes) > 0:
            first_note = notes[0]
            print(f"     First note: pitch={first_note.pitch}, string={first_note.string}, fret={first_note.fret}")
        
        # Test MIDI sequence conversion
        midi_events = loader.notes_to_midi_sequence(notes)
        print(f"     MIDI events: {len(midi_events)}")
        if len(midi_events) > 0:
            print(f"     First event: {midi_events[0]}")
            print(f"     Event types: {[e['type'] for e in midi_events[:5]]}")
        
        # Stop after first file for debug
        break
    
    # 2. Test tokenizer
    print("\n2. Testing tokenizer...")
    tokenizer = UnifiedFrettingTokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # Test with sample events
    if len(raw_dataset) > 0:
        file_path, notes = next(iter(raw_dataset.items()))
        midi_events = loader.notes_to_midi_sequence(notes)
        
        if len(midi_events) > 0:
            print(f"   Testing tokenization with {len(midi_events)} events...")
            
            # Convert to structured events manually
            structured_midi = []
            structured_tab = []
            
            for event in midi_events:
                if event['type'] == 'note_on':
                    structured_midi.append({
                        'type': 'note_on',
                        'pitch': event['pitch']
                    })
                    structured_tab.append({
                        'type': 'tab',
                        'string': event['string'],
                        'fret': event['fret']
                    })
                elif event['type'] == 'note_off':
                    structured_midi.append({
                        'type': 'note_off',
                        'pitch': event['pitch']
                    })
                elif event['type'] == 'time_shift':
                    ticks = event.get('ticks', event.get('delta', 0))
                    structured_midi.append({
                        'type': 'time_shift',
                        'ticks': ticks
                    })
                    structured_tab.append({
                        'type': 'time_shift',
                        'ticks': ticks
                    })
            
            print(f"   Structured MIDI events: {len(structured_midi)}")
            print(f"   Structured TAB events: {len(structured_tab)}")
            
            if len(structured_midi) > 0:
                midi_tokens = tokenizer.encode_midi_sequence(structured_midi[:10])  # First 10
                print(f"   MIDI tokens: {midi_tokens}")
                
                midi_ids = tokenizer.tokens_to_ids(midi_tokens)
                print(f"   MIDI IDs: {midi_ids}")
            
            if len(structured_tab) > 0:
                tab_tokens = tokenizer.encode_tab_sequence(structured_tab[:10])  # First 10
                print(f"   TAB tokens: {tab_tokens}")
                
                tab_ids = tokenizer.tokens_to_ids(tab_tokens)
                print(f"   TAB IDs: {tab_ids}")
    
    # 3. Test data config filtering
    print("\n3. Testing data config filtering...")
    config = UnifiedDataConfig()
    print(f"   Min notes per song: {config.min_notes_per_song}")
    print(f"   Max notes per song: {config.max_notes_per_song}")
    
    for file_path, notes in raw_dataset.items():
        note_count = len(notes)
        passes_filter = (note_count >= config.min_notes_per_song and 
                        note_count <= config.max_notes_per_song)
        print(f"   {os.path.basename(file_path)}: {note_count} notes, passes filter: {passes_filter}")
    
    # 4. Test processor directly
    print("\n4. Testing processor...")
    processor = UnifiedFrettingDataProcessor()
    
    # Test processing one song manually
    if len(raw_dataset) > 0:
        file_path, notes = next(iter(raw_dataset.items()))
        print(f"   Processing {os.path.basename(file_path)} with {len(notes)} notes...")
        
        try:
            sequences = processor._process_song(notes)
            print(f"   Generated sequences: {len(sequences)}")
            
            if len(sequences) > 0:
                input_ids, output_ids = sequences[0]
                print(f"   First sequence: input_len={len(input_ids)}, output_len={len(output_ids)}")
                print(f"   Input IDs (first 10): {input_ids[:10]}")
                print(f"   Output IDs (first 10): {output_ids[:10]}")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_dataset_processing()