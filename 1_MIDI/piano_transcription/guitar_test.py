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

    # Use 'test' split for evaluation
    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
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

    # Print detailed results
    print('\n' + '='*50)
    print('GuitarSet Test Results')
    print('='*50)
    
    for key, value in test_statistics.items():
        if isinstance(value, (int, float)):
            print(f'{key}: {value:.4f}')
        else:
            print(f'{key}: {value}')
    
    print('='*50)

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
    
    
    args = parser.parse_args()
    args.filename = 'test_guitarset'
    
    guitar_test(args)
