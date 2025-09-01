#!/bin/bash

# FretNet Training Script for GPU 6 with custom output directory
# Run with: nohup ./run_training.sh &

# Set GPU to 6
export CUDA_VISIBLE_DEVICES=6

# Set custom results directory
RESULTS_DIR="/data/andreaguz/FretNet_results"
mkdir -p "$RESULTS_DIR"

echo "Starting FretNet training on GPU 6..."
echo "Results will be saved to: $RESULTS_DIR"
echo "Started at: $(date)"

# Navigate to FretNet directory
cd /home/andreaguz/Music-AI/FretNet_Original

# Run training with custom output directory
python six_fold_cv_scripts/experiment.py with \
    gpu_id=0 \
    reset_data=True \
    root_dir="$RESULTS_DIR/experiments/FretNet_GuitarSetPlus_HCQT_X" \
    --print-config \
    --id="gpu6_training" \
    --comment="FretNet training on GPU 6 with clean setup"

echo "Training completed at: $(date)"