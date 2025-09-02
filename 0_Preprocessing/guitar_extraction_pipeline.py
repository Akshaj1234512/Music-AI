#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil

def extract_guitar(input_wav, output_wav, model='htdemucs_6s', device='cpu', segment=7):
    """
    Extract guitar from audio using Demucs
    
    Args:
        input_wav: Path to input WAV file
        output_wav: Path to output guitar WAV file
        model: Demucs model ('htdemucs_6s' has dedicated guitar stem)
        device: 'cpu' or 'cuda'
        segment: Segment size in seconds (max 7.8 for htdemucs_6s)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create temp directory for Demucs output
    temp_dir = Path('./temp_demucs')
    temp_dir.mkdir(exist_ok=True)
    
    # Run Demucs
    cmd = [
        'python', '-m', 'demucs',
        '-n', model,
        '-o', str(temp_dir),
        '--device', device,
        '--segment', str(segment),
        str(input_wav)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error processing {input_wav}: {result.stderr}")
        return False
    
    # Find guitar output
    stem_name = Path(input_wav).stem
    guitar_file = temp_dir / model / stem_name / 'guitar.wav'
    
    # If no guitar stem, use 'other' (for 4-stem models)
    if not guitar_file.exists():
        guitar_file = temp_dir / model / stem_name / 'other.wav'
    
    if guitar_file.exists():
        shutil.move(str(guitar_file), str(output_wav))
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Extract guitar from WAV files using Demucs')
    parser.add_argument('input_dir', help='Input directory with WAV files')
    parser.add_argument('output_dir', help='Output directory for guitar tracks')
    parser.add_argument('--model', default='htdemucs_6s', help='Demucs model (default: htdemucs_6s)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--segment', type=int, default=7, help='Segment size in seconds (default: 7)')
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find WAV files
    wav_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.WAV'))
    print(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    print(f"Using model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Segment size: {args.segment}s")
    
    # Process files
    success_count = 0
    for wav_file in tqdm(wav_files, desc="Extracting guitar"):
        output_file = output_dir / f"{wav_file.stem}_guitar.wav"
        if extract_guitar(wav_file, output_file, args.model, args.device, args.segment):
            success_count += 1
    
    print(f"Completed! Successfully processed {success_count}/{len(wav_files)} files")

if __name__ == '__main__':
    main()