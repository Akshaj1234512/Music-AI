#!/usr/bin/env python3
"""
Test Script

This script evaluates the pretrained piano transcription model on  test data.

"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
import numpy as np
import argparse
import h5py
import math
import time
import librosa
import logging
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, get_filename, create_logging,
    StatisticsContainer, RegressionPostProcessor)
from data_generator import Augmentor, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN, Note_pedal
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator
import config


def analyze_onset_timing_bias(model, dataloader, device, output_path, use_postprocessing=True):
    """Analyze onset timing bias using the same method as training evaluation."""
    from pytorch_utils import forward_dataloader
    import matplotlib.pyplot as plt
    
    # Get model outputs using the same method as training
    output_dict = forward_dataloader(model, dataloader, 4)
    
    # Apply post-processing if requested
    if use_postprocessing:
        from utilities import RegressionPostProcessor
        # Create post-processor with the same thresholds as inference.py
        post_processor = RegressionPostProcessor(
            frames_per_second=config.frames_per_second,
            classes_num=config.classes_num,
            onset_threshold=0.4,  # Updated threshold
            offset_threshold=0.3,
            frame_threshold=0.1,
            pedal_offset_threshold=0.2
        )
        
        # Apply post-processing to get final predictions
        processed_outputs = []
        for i in range(len(output_dict['frame_output'])):
            frame_output = output_dict['frame_output'][i:i+1]
            onset_output = output_dict['onset_output'][i:i+1]
            offset_output = output_dict['offset_output'][i:i+1]
            reg_onset_output = output_dict['reg_onset_output'][i:i+1]
            
            processed = post_processor.process(
                frame_output, onset_output, offset_output, reg_onset_output
            )
            processed_outputs.append(processed)
        
        # Use post-processed reg_onset_output for analysis
        reg_onset_output = np.concatenate([p['reg_onset_output'] for p in processed_outputs])
    else:
        reg_onset_output = output_dict['reg_onset_output']
    
    # Calculate individual onset errors using the same method as the MAE calculation
    if 'reg_onset_output' in output_dict.keys():
        # Use the same mask as in the MAE calculation
        mask = (np.sign(reg_onset_output + output_dict['reg_onset_roll'] - 0.01) + 1) / 2
        
        # Get individual errors where mask is 1 (where there are onsets)
        reg_onset_roll = output_dict['reg_onset_roll']
        
        # Flatten and apply mask
        output_flat = reg_onset_output.flatten()
        target_flat = reg_onset_roll.flatten()
        mask_flat = mask.flatten()
        
        # Get only the values where mask is 1 (actual onsets)
        onset_indices = np.where(mask_flat > 0.5)[0]
        
        if len(onset_indices) > 0:
            onset_errors = output_flat[onset_indices] - target_flat[onset_indices]
            
            # Calculate statistics
            mean_error = np.mean(onset_errors)
            std_error = np.std(onset_errors)
            early_count = np.sum(onset_errors < 0)
            late_count = np.sum(onset_errors > 0)
            
            # Count errors within tolerance (in frames, convert to seconds)
            within_50ms = np.sum(np.abs(onset_errors) <= 0.05)  # 0.05 seconds = 50ms
            within_100ms = np.sum(np.abs(onset_errors) <= 0.1)   # 0.1 seconds = 100ms
            within_200ms = np.sum(np.abs(onset_errors) <= 0.2)   # 0.2 seconds = 200ms
            
            # Generate error distribution plots
            plot_error_distributions(onset_errors, output_path)
            
            return {
                'onset_error_mean_seconds': mean_error,
                'onset_error_std_seconds': std_error,
                'onset_error_early_count': early_count,
                'onset_error_late_count': late_count,
                'onset_error_within_50ms': within_50ms,
                'onset_error_within_100ms': within_100ms,
                'onset_error_within_200ms': within_200ms,
                'total_onset_errors': len(onset_errors)
            }
    
    return {
        'onset_error_mean_seconds': 0.0,
        'onset_error_std_seconds': 0.0,
        'onset_error_early_count': 0,
        'onset_error_late_count': 0,
        'onset_error_within_50ms': 0,
        'onset_error_within_100ms': 0,
        'onset_error_within_200ms': 0,
        'total_onset_errors': 0
    }


def plot_error_distributions(onset_errors, output_path):
    """Generate histograms showing the distribution of onset timing errors."""
    import matplotlib.pyplot as plt
    import os
    
    # Convert to milliseconds for better readability
    errors_ms = onset_errors * 1000
    abs_errors_ms = np.abs(errors_ms)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Raw error distribution (signed errors)
    ax1.hist(errors_ms, bins=100, alpha=0.7, edgecolor='black', density=True)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect timing')
    ax1.axvline(np.mean(errors_ms), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(errors_ms):.1f}ms')
    
    # Add tolerance boundaries
    ax1.axvline(-50, color='green', linestyle=':', alpha=0.8, linewidth=2, label='±50ms tolerance')
    ax1.axvline(50, color='green', linestyle=':', alpha=0.8, linewidth=2)
    ax1.axvline(-100, color='blue', linestyle=':', alpha=0.6, linewidth=1.5, label='±100ms tolerance')
    ax1.axvline(100, color='blue', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.axvline(-200, color='purple', linestyle=':', alpha=0.4, linewidth=1, label='±200ms tolerance')
    ax1.axvline(200, color='purple', linestyle=':', alpha=0.4, linewidth=1)
    
    ax1.set_xlabel('Timing Error (milliseconds)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Raw Onset Timing Errors\n(Negative = Early, Positive = Late)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text with tolerance info
    within_50ms_raw = np.sum(np.abs(errors_ms) <= 50)
    within_100ms_raw = np.sum(np.abs(errors_ms) <= 100)
    within_200ms_raw = np.sum(np.abs(errors_ms) <= 200)
    total_raw = len(errors_ms)
    
    stats_text = f'Mean: {np.mean(errors_ms):.1f}ms\nStd: {np.std(errors_ms):.1f}ms\n'
    stats_text += f'Early: {np.sum(errors_ms < 0)} ({np.sum(errors_ms < 0)/len(errors_ms)*100:.1f}%)\n'
    stats_text += f'Late: {np.sum(errors_ms > 0)} ({np.sum(errors_ms > 0)/len(errors_ms)*100:.1f}%)\n'
    stats_text += f'Within ±50ms: {within_50ms_raw} ({within_50ms_raw/total_raw*100:.1f}%)\n'
    stats_text += f'Within ±100ms: {within_100ms_raw} ({within_100ms_raw/total_raw*100:.1f}%)\n'
    stats_text += f'Within ±200ms: {within_200ms_raw} ({within_200ms_raw/total_raw*100:.1f}%)'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Absolute error distribution (MAE)
    ax2.hist(abs_errors_ms, bins=100, alpha=0.7, edgecolor='black', density=True, color='green')
    ax2.axvline(np.mean(abs_errors_ms), color='red', linestyle='-', linewidth=2, 
                label=f'MAE: {np.mean(abs_errors_ms):.1f}ms')
    
    # Add threshold lines with better visibility
    ax2.axvline(50, color='orange', linestyle=':', alpha=0.8, linewidth=2, label='50ms threshold')
    ax2.axvline(100, color='blue', linestyle=':', alpha=0.8, linewidth=2, label='100ms threshold')
    ax2.axvline(200, color='purple', linestyle=':', alpha=0.8, linewidth=2, label='200ms threshold')
    
    ax2.set_xlabel('Absolute Timing Error (milliseconds)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Absolute Onset Timing Errors\n(MAE = Mean Absolute Error)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add accuracy statistics
    within_50ms = np.sum(abs_errors_ms <= 50)
    within_100ms = np.sum(abs_errors_ms <= 100)
    within_200ms = np.sum(abs_errors_ms <= 200)
    total = len(abs_errors_ms)
    
    acc_text = f'Within ±50ms: {within_50ms} ({within_50ms/total*100:.1f}%)\n'
    acc_text += f'Within ±100ms: {within_100ms} ({within_100ms/total*100:.1f}%)\n'
    acc_text += f'Within ±200ms: {within_200ms} ({within_200ms/total*100:.1f}%)'
    ax2.text(0.02, 0.98, acc_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_path, 'onset_error_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Error distribution plots saved to {plot_path}")
    
    # Show plot
    plt.show()


def guitar_test(args):
    """Evaluate piano transcription model on GuitarSet test dataset.

    Args:
      workspace: str, directory of your workspace
      audio_type: str, one of ['combined', 'audio_hex_debleeded', 'audio_hex_original', 'audio_mix', 'audio_mono-mic']
      model_type: str, e.g. 'Note_pedal'
      batch_size: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arguments & parameters
    workspace = args.workspace
    audio_type = args.audio_type
    model_type = args.model_type
    batch_size = args.batch_size
    dataset = args.dataset
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8

    # Paths
    # Updated to use the new folder structure: test, train, val
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset, audio_type, '2024')

    output_path = os.path.join(workspace, 'test_results')
    create_folder(output_path)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

    # Model
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num)

    # Dataset - Use existing data generator but point to GuitarSet HDF5s
    from data_generator import MaestroDataset, Sampler, TestSampler

    test_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second)

    # Use specified split for evaluation
    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split=args.split, segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Load pretrained model
    if args.pretrained_path:
        logging.info('Loading pretrained model from {}'.format(args.pretrained_path))
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logging.info('Pretrained model loaded successfully')

    # Move model to device
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    logging.info('Starting evaluation on test set...')
    
    # Evaluate the model
    with torch.no_grad():
        test_statistics = evaluator.evaluate(test_loader)

    logging.info('Test evaluation completed!')
    logging.info('Test statistics: {}'.format(test_statistics))
    
    # Add detailed onset timing analysis
    logging.info('Starting detailed onset timing analysis...')
    onset_analysis = analyze_onset_timing_bias(model, test_loader, device, output_path, use_postprocessing=True)
    test_statistics.update(onset_analysis)

    # Print detailed results
    print('\n' + '='*60)
    print('GuitarSet Test Results')
    print('='*60)
    
    # Print core metrics
    print('\nCORE METRICS:')
    print(f'Frame AP: {test_statistics.get("frame_ap", 0):.4f}')
    print(f'Onset MAE: {test_statistics.get("reg_onset_mae", 0):.4f} seconds ({test_statistics.get("reg_onset_mae", 0)*1000:.1f} ms)')
    print(f'Offset MAE: {test_statistics.get("reg_offset_mae", 0):.4f} seconds ({test_statistics.get("reg_offset_mae", 0)*1000:.1f} ms)')
    print(f'Velocity MAE: {test_statistics.get("velocity_mae", 0):.4f}')
    
    # Print onset timing analysis
    if 'total_onset_errors' in test_statistics and test_statistics['total_onset_errors'] > 0:
        total_errors = test_statistics['total_onset_errors']
        print(f'\nONSET TIMING ANALYSIS ({total_errors:.0f} onset errors analyzed):')
        print(f'Mean Error: {test_statistics.get("onset_error_mean_seconds", 0):.4f} seconds ({test_statistics.get("onset_error_mean_seconds", 0)*1000:.1f} ms)')
        print(f'Std Error: {test_statistics.get("onset_error_std_seconds", 0):.4f} seconds ({test_statistics.get("onset_error_std_seconds", 0)*1000:.1f} ms)')
        print(f'MAE: {test_statistics.get("reg_onset_mae", 0):.4f} seconds ({test_statistics.get("reg_onset_mae", 0)*1000:.1f} ms)')
        
        print(f'\nERROR DIRECTION:')
        early_pct = (test_statistics.get("onset_error_early_count", 0) / total_errors) * 100
        late_pct = (test_statistics.get("onset_error_late_count", 0) / total_errors) * 100
        print(f'Early predictions: {test_statistics.get("onset_error_early_count", 0):.0f} ({early_pct:.1f}%)')
        print(f'Late predictions: {test_statistics.get("onset_error_late_count", 0):.0f} ({late_pct:.1f}%)')
        
        print(f'\nACCURACY WITHIN TOLERANCE:')
        within_50ms_pct = (test_statistics.get("onset_error_within_50ms", 0) / total_errors) * 100
        within_100ms_pct = (test_statistics.get("onset_error_within_100ms", 0) / total_errors) * 100
        within_200ms_pct = (test_statistics.get("onset_error_within_200ms", 0) / total_errors) * 100
        print(f'Within ±50ms: {test_statistics.get("onset_error_within_50ms", 0):.0f} ({within_50ms_pct:.1f}%)')
        print(f'Within ±100ms: {test_statistics.get("onset_error_within_100ms", 0):.0f} ({within_100ms_pct:.1f}%)')
        print(f'Within ±200ms: {test_statistics.get("onset_error_within_200ms", 0):.0f} ({within_200ms_pct:.1f}%)')
        
        print(f'\nBIAS ANALYSIS:')
        mean_error_ms = test_statistics.get("onset_error_mean_seconds", 0) * 1000
        if abs(mean_error_ms) < 10:
            print(f'No significant bias (mean error: {mean_error_ms:.1f}ms)')
        elif mean_error_ms > 0:
            print(f'Model predicts LATE by {mean_error_ms:.1f}ms on average')
        else:
            print(f'Model predicts EARLY by {abs(mean_error_ms):.1f}ms on average')
    
    print('='*60)

    # Save results to file
    results_file = os.path.join(output_path, f'{filename}_test_results.txt')
    with open(results_file, 'w') as f:
        f.write('Test Results\n')
        f.write('='*50 + '\n')
        for key, value in test_statistics.items():
            if isinstance(value, (int, float)):
                f.write(f'{key}: {value:.4f}\n')
            else:
                f.write(f'{key}: {value}\n')
    
    logging.info(f'Results saved to {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate piano transcription model')
    parser.add_argument('--workspace', type=str, required=True, 
        help='Directory of your workspace')
    parser.add_argument('--dataset', type=str, required=True, 
        help='Directory of your dataset')
    parser.add_argument('--audio_type', type=str, default='combined',
        help='Type of audio to use (folder name in guitarset directory)')
    parser.add_argument('--model_type', type=str, default='Note_pedal',
        help='Model type')
    parser.add_argument('--batch_size', type=int, default=4,
        help='Batch size')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Use CUDA')
    parser.add_argument('--pretrained_path', type=str, default='PT.pth',
        help='Path to pretrained model')
    parser.add_argument('--split', type=str, default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate')
    
    
    args = parser.parse_args()
    args.filename = 'test_guitarset'
    
    guitar_test(args)
