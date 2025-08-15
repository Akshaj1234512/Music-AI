#!/usr/bin/env python3
"""
MIDI to Audio Converter
Converts the specific MIDI file to synthesized audio using Python
"""

import os
import sys
from pathlib import Path
import mido
import numpy as np
import wave
import struct

def synthesize_midi_audio(midi_file, output_file, sample_rate=44100):
    """Synthesize MIDI file to audio using improved synthesis with anti-aliasing and smooth envelopes"""
    print(f"Synthesizing MIDI to audio: {midi_file.name}")
    
    # Load MIDI file
    mid = mido.MidiFile(midi_file)
    
    # Calculate total duration in seconds
    total_ticks = 0
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, 'time'):
                total_ticks += msg.time
    
    # Convert ticks to seconds (assuming 60 BPM)
    total_seconds = total_ticks / mid.ticks_per_beat * 60 / 60 
    total_samples = int(total_seconds * sample_rate)
    
    print(f"  Total duration: {total_seconds:.2f} seconds")
    print(f"  Total samples: {total_samples:,}")
    
    # Create audio buffer
    audio_buffer = np.zeros(total_samples, dtype=np.float32)
    
    # Track active notes
    active_notes = {}  
    current_sample = 0
    current_tick = 0
    
    # Process MIDI messages to build note timeline
    notes_timeline = []
    for track in mid.tracks:
        for msg in track:
            # Convert ticks to samples
            tick_delta = msg.time
            sample_delta = int(tick_delta / mid.ticks_per_beat * 60 / 60 * sample_rate)
            current_sample += sample_delta
            current_tick += tick_delta
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note on
                note = msg.note
                velocity = msg.velocity
                active_notes[note] = (current_sample, velocity, None)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Note off
                note = msg.note
                if note in active_notes:
                    start_sample, velocity, _ = active_notes[note]
                    notes_timeline.append((start_sample, current_sample, note, velocity))
                    del active_notes[note]
    
    # Add any remaining active notes
    for note, (start_sample, velocity, _) in active_notes.items():
        notes_timeline.append((start_sample, total_samples, note, velocity))
    
    # Sort notes by start time
    notes_timeline.sort(key=lambda x: x[0])
    
    print(f"  Processing {len(notes_timeline)} notes")
    
    # Generate audio for each note with improved synthesis
    for start_sample, end_sample, note, velocity in notes_timeline:
        # Calculate note duration
        duration_samples = end_sample - start_sample
        
        # Skip very short notes (less than 10ms)
        if duration_samples < sample_rate * 0.01:
            continue
        
        # Generate frequency for this note
        frequency = 440.0 * (2 ** ((note - 69) / 12))  
        
        # Skip very high frequencies that cause aliasing
        if frequency > sample_rate / 3:
            continue
        
        # Create time array for this note
        t = np.linspace(0, duration_samples / sample_rate, duration_samples)
        
        # Generate sine wave
        sine_wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply smooth envelope (attack and release)
        attack_samples = min(int(sample_rate * 0.01), duration_samples // 4)  
        release_samples = min(int(sample_rate * 0.01), duration_samples // 4)  
        
        # Create envelope
        envelope = np.ones(duration_samples)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        # Apply envelope and velocity
        note_audio = sine_wave * envelope * (velocity / 127.0) * 0.2
        
        # Add to audio buffer
        end_sample = min(start_sample + duration_samples, total_samples)
        if start_sample < total_samples:
            audio_buffer[start_sample:end_sample] += note_audio[:end_sample-start_sample]
    
    # Normalize audio to prevent clipping
    if np.max(np.abs(audio_buffer)) > 0:
        audio_buffer = audio_buffer / np.max(np.abs(audio_buffer)) * 0.7
    
    # Apply gentle low-pass filter to reduce high-frequency artifacts
    from scipy import signal
    try:
        # Design a low-pass filter
        nyquist = sample_rate / 2
        cutoff = min(nyquist * 0.8, 8000)  
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        audio_buffer = signal.filtfilt(b, a, audio_buffer)
        print(f"  Applied low-pass filter (cutoff: {cutoff:.0f}Hz)")
    except ImportError:
        print("  scipy not available, skipping filter")
    
    # Convert to 16-bit PCM
    audio_16bit = (audio_buffer * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(str(output_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_16bit.tobytes())
    
    print(f"✓ Successfully created: {output_file.name}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    return output_file

def find_corresponding_audio(midi_file):
    """Find the corresponding original audio file"""
    # Extract base name
    base_name = midi_file.stem
    
    # Look for audio files
    audio_dir = Path("/data/akshaj/MusicAI/GuitarSet/audio_mono-mic")
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return None
    
    # Try different audio extensions
    for ext in ['.wav', '.mp3', '.flac']:
        audio_file = audio_dir / f"{base_name}_mic{ext}"
        if audio_file.exists():
            return audio_file
    
    print(f"No corresponding audio file found for: {base_name}")
    return None

def main():
    """Main function"""
    print("MIDI to Audio Converter (Python-based)")
    print("=" * 50)
    
    # Specify the MIDI file
    midi_file = Path("/data/akshaj/MusicAI/GuitarSet/MIDIAnnotations/00_BN1-129-Eb_comp.mid")
    
    if not midi_file.exists():
        print(f"MIDI file not found: {midi_file}")
        print("Please run GuitarSet_MIDI.py first to create the MIDI files")
        return
    
    print(f"Found MIDI file: {midi_file.name}")
    
    # Create output directory
    output_dir = Path("/data/akshaj/MusicAI/GuitarSet/synthesized_audio")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Convert MIDI to audio
    print("\n🎵 Converting MIDI to synthesized audio...")
    output_file = output_dir / f"{midi_file.stem}_synthesized.wav"
    
    try:
        audio_file = synthesize_midi_audio(midi_file, output_file)
        
        if not audio_file:
            print("✗ Conversion failed")
            return
        
        # Find corresponding original audio
        original_audio = find_corresponding_audio(midi_file)
        
        print("\n" + "=" * 50)
        print("CONVERSION COMPLETE!")
        print(f"Synthesized audio: {audio_file}")
        
        if original_audio:
            print(f"Original audio: {original_audio}")
            print(f"  - Synthesized: {audio_file}")
            print(f"  - Original: {original_audio}")
        else:
            print("Original audio not found for comparison")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
