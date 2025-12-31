import h5py
import numpy as np
import scipy.io.wavfile as wav
import pretty_midi
import os
import re

# --- CONFIGURATION ---
# The H5 file you want to test
H5_PATH = "/data/akshaj/MusicAI/workspace/hdf5s/synthtab/2024/train/synthtab_acoustic_gibson_thumb__1 god - grace - gp4__1 - acoustic nylon guitar__gibson_thumb_nonoise_mono_body.flac.h5"

# Where to save the output files
OUTPUT_DIR = "./verification_output"
SAMPLE_RATE = 16000

def parse_value(text, key):
    """Helper to extract values like 'note=45' from the event string."""
    match = re.search(f"{key}=(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None

def reconstruct_files(h5_path, output_dir):
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- RECONSTRUCTING FROM: {os.path.basename(h5_path)} ---")

    with h5py.File(h5_path, 'r') as hf:
        # 1. SAVE AUDIO (WAV)
        print("1. Extracting Audio...")
        waveform = hf['waveform'][:]
        audio_out = os.path.join(output_dir, "reconstructed_audio.wav")
        wav.write(audio_out, SAMPLE_RATE, waveform)
        print(f"   -> Saved: {audio_out}")

        # 2. SAVE MIDI (.MID)
        print("2. Reconstructing MIDI...")
        events = hf['midi_event'][:]
        times = hf['midi_event_time'][:]
        
        # Create a new MIDI object
        pm = pretty_midi.PrettyMIDI()
        # Create an instrument (Acoustic Guitar = Program 25)
        inst = pretty_midi.Instrument(program=25, name="Reconstructed Guitar")
        
        # Dictionary to track active notes: { (pitch, string): (start_time, velocity) }
        active_notes = {}

        # Sort events by time to be safe
        sorted_indices = np.argsort(times)
        
        for i in sorted_indices:
            event_str = events[i].decode('utf-8')
            
            # Parse the string
            # Example: "note_on channel=0 note=45 velocity=80 time=0.00 string=6"
            try:
                pitch = int(parse_value(event_str, "note"))
                vel = int(parse_value(event_str, "velocity"))
                time = float(parse_value(event_str, "time"))
                string_idx = int(parse_value(event_str, "string")) # Keeping track of string just for matching
                
                # We use (pitch, string_idx) as the key to handle the same note played on different strings correctly
                note_key = (pitch, string_idx)

                if "note_on" in event_str:
                    if vel > 0:
                        active_notes[note_key] = (time, vel)
                    else:
                        # Sometimes note_on with vel=0 is treated as note_off
                        if note_key in active_notes:
                            start_time, start_vel = active_notes.pop(note_key)
                            note = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=time)
                            inst.notes.append(note)

                elif "note_off" in event_str:
                    if note_key in active_notes:
                        start_time, start_vel = active_notes.pop(note_key)
                        # Create the Note object
                        note = pretty_midi.Note(velocity=start_vel, pitch=pitch, start=start_time, end=time)
                        inst.notes.append(note)
                        
            except Exception as e:
                print(f"Skipping malformed event: {event_str} ({e})")

        pm.instruments.append(inst)
        midi_out = os.path.join(output_dir, "reconstructed_midi.mid")
        pm.write(midi_out)
        print(f"   -> Saved: {midi_out}")
        print("\n✅ Verification Complete!")

if __name__ == "__main__":
    reconstruct_files(H5_PATH, OUTPUT_DIR)