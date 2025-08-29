#!/bin/bash

# EGDB Finetuning Setup Script
# This script sets up the complete EGDB finetuning pipeline

set -e

echo "=== EGDB Finetuning Setup ==="

# Check if we're in the right directory
if [ ! -f "pytorch/main.py" ]; then
    echo "Error: Please run this script from the piano_transcription directory"
    exit 1
fi

# Set paths
WORKSPACE="/data/akshaj/MusicAI/workspace"
EGDB_DIR="/data/upasanap/EGDB"  # EGDB dataset location

echo "Workspace: $WORKSPACE"
echo "EGDB directory: $EGDB_DIR"

# Check if EGDB directory exists
if [ ! -d "$EGDB_DIR" ]; then
    echo "Error: EGDB directory not found at $EGDB_DIR"
    echo "Please update the EGDB_DIR variable in this script to point to your EGDB dataset"
    exit 1
fi

# Check EGDB structure
echo "Checking EGDB dataset structure..."
if [ ! -d "$EGDB_DIR/audio_label" ]; then
    echo "Error: MIDI files not found at $EGDB_DIR/audio_label"
    exit 1
fi

if [ ! -d "$EGDB_DIR/audio_DI" ]; then
    echo "Error: Audio files not found at $EGDB_DIR/audio_DI"
    exit 1
fi

echo "EGDB structure verified:"
echo "  - MIDI files: $EGDB_DIR/audio_label/ (240 .midi files numbered 1-240)"
echo "  - Audio files: $EGDB_DIR/audio_*/ (matching numbering 1.wav, 2.wav, etc.)"

# Create workspace directories
echo "Creating workspace directories..."
mkdir -p "$WORKSPACE/hdf5s/egdb"
mkdir -p "$WORKSPACE/checkpoints/egdb"
mkdir -p "$WORKSPACE/logs/egdb"
mkdir -p "$WORKSPACE/statistics/egdb"

# Function to process audio type
process_audio_type() {
    local audio_type=$1
    echo "Processing $audio_type..."
    
    # Check if audio directory exists
    if [ ! -d "$EGDB_DIR/$audio_type" ]; then
        echo "Warning: $audio_type directory not found, skipping..."
        return
    fi
    
    # Pack dataset to HDF5
    echo "Packing $audio_type to HDF5 format..."
    python3 utils/egdb_features.py \
        --dataset_dir "$EGDB_DIR" \
        --workspace "$WORKSPACE" \
        --audio_type "$audio_type"
    
    echo "$audio_type processing complete!"
}

# Process combined dataset (all audio types together)
echo "Processing EGDB combined dataset (all audio types)..."
process_audio_type "combined"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "EGDB Dataset Info:"
echo "  - Total clips: 240 (numbered 1-240)"
echo "  - Training: clips 1-192 (80% split)"
echo "  - Validation: clips 193-240 (20% split)"
echo "  - Audio types: DI, Mesa, Marshall, Ftwin, Plexi, JCjazz"
echo "  - MIDI files: .midi format in audio_label/"
echo "  - Audio files: .wav format in audio_*/"
echo ""
echo "To finetune on combined dataset (all audio types - more training data!):"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type combined --cuda"
echo ""
echo "To finetune on individual audio types:"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type audio_DI --cuda"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type audio_Mesa --cuda"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type audio_Marshall --cuda"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type audio_Ftwin --cuda"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type audio_Plexi --cuda"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type audio_JCjazz --cuda"
echo ""
echo "To use different training parameters:"
echo "python3 finetune_egdb.py --workspace $WORKSPACE --audio_type combined --batch_size 8 --learning_rate 1e-4 --cuda"
echo ""
echo "The finetuned models will be saved in: $WORKSPACE/checkpoints/egdb/"
echo ""
echo "Note: The script automatically maps 1.midi -> 1.wav, 2.midi -> 2.wav, etc."
echo "This matches the EGDB dataset structure where MIDI and audio files have the same numbering."
