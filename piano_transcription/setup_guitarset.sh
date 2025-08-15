#!/bin/bash

# GuitarSet Finetuning Setup Script
# This script sets up the complete GuitarSet finetuning pipeline

set -e

echo "=== GuitarSet Finetuning Setup ==="

# Check if we're in the right directory
if [ ! -f "pytorch/main.py" ]; then
    echo "Error: Please run this script from the piano_transcription directory"
    exit 1
fi

# Set paths
WORKSPACE="/data/akshaj/MusicAI/workspace"
GUITARSET_DIR="/data/akshaj/MusicAI/GuitarSet"
PRETRAINED_MODEL="PT.pth"

echo "Workspace: $WORKSPACE"
echo "GuitarSet directory: $GUITARSET_DIR"
echo "Pretrained model: $PRETRAINED_MODEL"

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained model $PRETRAINED_MODEL not found"
    exit 1
fi

# Create workspace directories
echo "Creating workspace directories..."
mkdir -p "$WORKSPACE/hdf5s/guitarset"
mkdir -p "$WORKSPACE/checkpoints/guitarset"
mkdir -p "$WORKSPACE/logs/guitarset"
mkdir -p "$WORKSPACE/statistics/guitarset"

# Function to process audio type
process_audio_type() {
    local audio_type=$1
    echo "Processing $audio_type..."
    
    # Check if audio directory exists
    if [ ! -d "$GUITARSET_DIR/$audio_type" ]; then
        echo "Warning: $audio_type directory not found, skipping..."
        return
    fi
    
    # Pack dataset to HDF5
    echo "Packing $audio_type to HDF5 format..."
    python3 utils/guitarset_features.py \
        --dataset_dir "$GUITARSET_DIR" \
        --workspace "$WORKSPACE" \
        --audio_type "$audio_type"
    
    echo "$audio_type processing complete!"
}

# Process combined dataset (all audio types together)
echo "Processing GuitarSet combined dataset (all audio types)..."
process_audio_type "combined"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To finetune on combined dataset (all audio types - 4x more data!):"
echo "python3 finetune_guitarset.py --workspace $WORKSPACE --audio_type combined --cuda"
echo ""
echo "To finetune on individual audio types:"
echo "python3 finetune_guitarset.py --workspace $WORKSPACE --audio_type audio_mix --cuda"
echo "python3 finetune_guitarset.py --workspace $WORKSPACE --audio_type audio_hex_debleeded --cuda"
echo "python3 finetune_guitarset.py --workspace $WORKSPACE --audio_type audio_hex_original --cuda"
echo "python3 finetune_guitarset.py --workspace $WORKSPACE --audio_type audio_mono-mic --cuda"
echo ""
echo "To use different training parameters:"
echo "python3 finetune_guitarset.py --workspace $WORKSPACE --audio_type audio_mix --batch_size 8 --learning_rate 1e-4 --cuda"
echo ""
echo "The finetuned models will be saved in: $WORKSPACE/checkpoints/guitarset/"
