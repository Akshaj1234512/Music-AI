#!/usr/bin/env python3
"""
Finetune piano transcription model on GAPS dataset.

This is a convenience wrapper around finetune.py for the GAPS dataset.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
import argparse
from finetune import finetune


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune piano transcription model on GAPS')
    parser.add_argument('--workspace', type=str, required=True,
        help='Directory of your workspace')
    parser.add_argument('--model_type', type=str, default='Note_pedal',
        help='Model type')
    parser.add_argument('--loss_type', type=str, default='note_pedal_combined_bce',
        help='Loss type')
    parser.add_argument('--augmentation', type=str, default='aug', choices=['none', 'aug'],
        help='Data augmentation')
    parser.add_argument('--max_note_shift', type=int, default=2,
        help='Maximum note shift for augmentation')
    parser.add_argument('--max_timing_shift', type=float, default=0.0,
        help='Maximum timing shift for augmentation (in seconds)')
    parser.add_argument('--batch_size', type=int, default=4,
        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
        help='Learning rate (paper uses 1e-5)')
    parser.add_argument('--reduce_iteration', type=int, default=10000,
        help='Reduce learning rate every N iterations (paper uses 10K)')
    parser.add_argument('--resume_iteration', type=int, default=0,
        help='Resume from iteration')
    parser.add_argument('--early_stop', type=int, default=75000,
        help='Early stop at iteration (default 75K for guitar datasets)')
    parser.add_argument('--mini_data', action='store_true', default=False,
        help='Use small subset for debugging')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Use CUDA')
    parser.add_argument('--pretrained_path', type=str, default='PT.pth',
        help='Path to pretrained model')

    args = parser.parse_args()

    # Set GAPS-specific parameters
    args.dataset_name = 'gaps'
    args.hdf5s_dir = os.path.join(args.workspace, 'hdf5s', 'gaps', '2024')

    print(f"Finetuning on GAPS dataset")
    print(f"HDF5 directory: {args.hdf5s_dir}")

    finetune(args)
