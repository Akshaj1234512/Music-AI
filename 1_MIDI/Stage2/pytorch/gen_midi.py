import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../../autoth'))
import numpy as np
import argparse
import librosa
import torch
 
from utilities import create_folder, get_filename, int16_to_float32
import config
from inference import PianoTranscription
import time

def transcribe_audio(args):
    """
    Transcribe a single audio file to MIDI and save it to a specific workspace.
    """
    # Arguments & parameters
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    
    # 1. Configuration matches training parameters
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    post_processor_type = args.post_processor_type

    # 2. Setup Output Directory
    # You requested this specific path
    output_dir = '/data/akshaj/MusicAI/workspace/midi'
    
    # Create the folder if it doesn't exist (equivalent to mkdir -p)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # 3. Load Audio
    print(f"Loading audio from: {audio_path}")
    # We must load with sr=sample_rate to match the model's expectation
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # 4. Initialize Transcriptor
    transcriptor = PianoTranscription(
        model_type=model_type, 
        device=device, 
        checkpoint_path=checkpoint_path, 
        segment_samples=segment_samples, 
        post_processor_type=post_processor_type
    )

    # 5. Define Output MIDI Path
    # Extract filename without extension (e.g., "song.wav" -> "song")
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    midi_filename = f"{filename}.mid"
    midi_path = os.path.join(output_dir, midi_filename)

    # 6. Transcribe
    print(f"Transcribing...")
    transcription_time = time.time()
    
    # The transcribe method handles the midi writing internally
    transcribed_dict = transcriptor.transcribe(audio, midi_path)
    
    print(f"Transcription complete! Saved to: {midi_path}")
    print(f"Time taken: {time.time() - transcription_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe a single audio file')
    
    # Required arguments
    parser.add_argument('--model_type', type=str, required=True, help='e.g., Regress_onset_offset_frame_velocity_CRNN')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the input audio file')

    # Optional arguments with defaults
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--post_processor_type', type=str, default='regression', choices=['regression', 'onsets_frames'])

    args = parser.parse_args()
    
    transcribe_audio(args)