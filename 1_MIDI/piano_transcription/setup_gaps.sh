#!/bin/bash

# Setup script for GAPS dataset
# This script downloads audio from YouTube and packs the dataset into HDF5 format

set -e  # Exit on error

# Configuration
GAPS_DIR="/data/subhash/gaps_v1"  # GAPS dataset location (MIDI + metadata)
AUDIO_DIR="/data/subhash/gaps_v1/audio"  # Where to download audio files
WORKSPACE="/data/subhash/MusicAI/workspace"  # Workspace for HDF5 files

# Check if GAPS dataset exists
if [ ! -d "$GAPS_DIR" ]; then
    echo "Error: GAPS dataset not found at: $GAPS_DIR"
    echo "Please download and extract the dataset from:"
    echo "  https://zenodo.org/records/13962272"
    echo "Then extract it to: $GAPS_DIR"
    exit 1
fi

echo "Checking GAPS dataset structure..."
if [ ! -f "$GAPS_DIR/gaps_v1_metadata.csv" ]; then
    echo "Error: Metadata file not found!"
    exit 1
fi

if [ ! -d "$GAPS_DIR/midi" ]; then
    echo "Error: MIDI directory not found!"
    exit 1
fi

echo "✓ GAPS dataset structure looks good"

# Step 1: Download audio from YouTube
echo ""
echo "========================================="
echo "Step 1: Downloading audio from YouTube"
echo "========================================="

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp is not installed. Installing..."
    pip install yt-dlp
fi

# Create audio directory
mkdir -p "$AUDIO_DIR"

# Download audio files
python3 1_MIDI/piano_transcription/download_gaps_audio.py \
    --dataset_dir "$GAPS_DIR" \
    --output_dir "$AUDIO_DIR"

echo ""
echo "Audio download complete!"

# Step 2: Pack dataset to HDF5
echo ""
echo "========================================="
echo "Step 2: Packing GAPS dataset to HDF5"
echo "========================================="

cd 1_MIDI/piano_transcription

python3 utils/gaps_features.py \
    --dataset_dir "$GAPS_DIR" \
    --workspace "$WORKSPACE" \
    --audio_dir "$AUDIO_DIR"

cd ../..

echo ""
echo "========================================="
echo "GAPS Dataset Setup Complete!"
echo "========================================="
echo ""
echo "Dataset Info:"
echo "  Location: $GAPS_DIR"
echo "  Audio: $AUDIO_DIR"
echo "  HDF5 files: $WORKSPACE/hdf5s/gaps/"
echo ""
echo "To finetune on GAPS dataset:"
echo "  cd 1_MIDI/piano_transcription"
echo "  python3 finetune_gaps.py \\"
echo "    --workspace /data/subhash/MusicAI/workspace \\"
echo "    --pretrained_path PT.pth \\"
echo "    --batch_size 4 \\"
echo "    --learning_rate 1e-5 \\"
echo "    --cuda"
echo ""
