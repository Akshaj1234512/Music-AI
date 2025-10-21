#!/usr/bin/env python3
"""
Download audio files for GAPS dataset from YouTube.

This script reads the GAPS metadata CSV and downloads the corresponding
YouTube videos as audio files.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import subprocess
import time

def download_gaps_audio(args):
    """Download audio files from YouTube based on GAPS metadata.

    Args:
        dataset_dir: str, path to GAPS dataset directory
        output_dir: str, path to save downloaded audio files
    """
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read metadata CSV
    metadata_path = os.path.join(dataset_dir, 'gaps_v1_metadata.csv')

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed. Please install it with:")
        print("  pip install yt-dlp")
        return

    print(f"Reading metadata from {metadata_path}")

    downloaded = 0
    failed = 0
    skipped = 0
    max_files = args.max_files

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, 1):
            # Stop if we've reached max_files
            if max_files and (downloaded + skipped + failed) >= max_files:
                print(f"\nReached max_files limit ({max_files}), stopping...")
                break
            yt_id = row.get('yt_id', '').strip()
            gpx_name = row.get('gpx_name', '').strip()

            if not yt_id or not gpx_name:
                print(f"Row {i}: Missing yt_id or gpx_name, skipping")
                skipped += 1
                continue

            # Output filename based on gpx_name
            output_file = os.path.join(output_dir, f"{gpx_name}.wav")

            # Skip if already exists
            if os.path.exists(output_file):
                print(f"[{i}] Already exists: {gpx_name}.wav")
                skipped += 1
                continue

            # YouTube URL
            youtube_url = f"https://www.youtube.com/watch?v={yt_id}"

            print(f"[{i}] Downloading: {gpx_name} from {yt_id}")

            # Download with yt-dlp
            # -x: extract audio
            # --audio-format wav: convert to WAV
            # --audio-quality 0: best quality
            # -o: output template
            try:
                subprocess.run([
                    'yt-dlp',
                    '-x',
                    '--audio-format', 'wav',
                    '--audio-quality', '0',
                    '-o', output_file,
                    youtube_url
                ], check=True, capture_output=True)

                downloaded += 1
                print(f"  ✓ Success: {gpx_name}.wav")

                # Add a small delay to avoid rate limiting
                time.sleep(1)

            except subprocess.CalledProcessError as e:
                failed += 1
                print(f"  ✗ Failed: {gpx_name}.wav")
                print(f"    Error: {e.stderr.decode() if e.stderr else str(e)}")

    print("\n" + "="*60)
    print(f"Download Summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total processed: {downloaded + skipped + failed}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download GAPS audio from YouTube')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to GAPS dataset directory (containing gaps_v1_metadata.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save downloaded audio files')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to download (for testing)')

    args = parser.parse_args()
    download_gaps_audio(args)
