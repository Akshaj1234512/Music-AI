#!/usr/bin/env python3

import argparse
import yaml
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import noisereduce as nr

def guitar_denoise(audio, sr, **params):
    default_params = {
        'stationary': False,
        'prop_decrease': 0.7,
        'n_fft': 2048,
        'hop_length': 512,
        'n_std_thresh_stationary': 2.0,
        'freq_mask_smooth_hz': 250,
        'time_mask_smooth_ms': 25,
        'thresh_n_mult_nonstationary': 2,
        'sigmoid_slope_nonstationary': 5,
        'time_constant_s': 1.0,
        'chunk_size': 60000
    }
    final_params = {**default_params, **params}
    return nr.reduce_noise(y=audio, sr=sr, **final_params)

def main():
    parser = argparse.ArgumentParser(description='Denoise WAV files in directory')
    parser.add_argument('input_dir', help='Input directory with WAV files')
    parser.add_argument('output_dir', help='Output directory for denoised files')
    parser.add_argument('--config', help='YAML config file (optional)')
    args = parser.parse_args()

    # Load config
    config_params = {}
    if args.config:
        with open(args.config, 'r') as f:
            config_params = yaml.safe_load(f) or {}

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find WAV files
    wav_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.WAV'))
    print(f"Found {len(wav_files)} WAV files")

    # Process files
    for wav_file in tqdm(wav_files, desc="Denoising"):
        audio, sr = librosa.load(wav_file, sr=None)
        denoised = guitar_denoise(audio, sr, **config_params)
        output_file = output_dir / wav_file.name
        sf.write(output_file, denoised, sr)

    print(f"Completed! Processed {len(wav_files)} files")

if __name__ == '__main__':
    main()