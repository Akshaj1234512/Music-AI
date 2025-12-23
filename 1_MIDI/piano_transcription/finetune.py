#!/usr/bin/env python3
"""
Unified Finetuning Script


This script finetunes the pretrained piano transcription model on any dataset,
following the methodology from the High Resolution Guitar Transcription paper.

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


def finetune(args):
    """Finetune piano transcription model on any dataset.

    Args:
      workspace: str, directory of your workspace
      dataset_name: str, name of the dataset (e.g., 'egdb', 'guitarset')
      hdf5s_dir: str, path to HDF5 files directory
      model_type: str, e.g. 'Note_pedal'
      loss_type: str, e.g. 'note_pedal_combined_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arguments & parameters
    workspace = args.workspace
    dataset_name = args.dataset_name
    hdf5s_dir = args.hdf5s_dir
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    max_timing_shift = args.max_timing_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8

    # Loss function
    loss_func = get_loss_func(loss_type)

    # Paths - hdf5s_dir is now passed as argument

    # Simplified directory structure
    checkpoints_dir = os.path.join(workspace, 'checkpoints')
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs')
    create_folder(logs_dir)

    # Get the log number that create_logging will use
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    if log_files:
        log_numbers = [int(f.split('.')[0]) for f in log_files]
        current_log_number = max(log_numbers) + 1  # Next log number
    else:
        current_log_number = 0  # First log number
    
    # Create checkpoint folder based on the log number we'll use
    log_folder_name = f"log_{current_log_number:04d}"
    log_checkpoints_dir = os.path.join(checkpoints_dir, log_folder_name)
    create_folder(log_checkpoints_dir)
    
    # Now create the logging with the specific log number
    log_path = os.path.join(logs_dir, f'{current_log_number:04d}.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode='w')
    
    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    logging.info(f'Using log number: {current_log_number}')
    logging.info(f'Checkpoints will be saved to: {log_checkpoints_dir}')
    
    logging.info('Dataset: {}'.format(dataset_name))
    logging.info('HDF5 directory: {}'.format(hdf5s_dir))
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

    # Model
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num)

    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect augmentation!')

    # Dataset - Use existing data generator but point to EGDB HDF5s
    from data_generator import MaestroDataset, Sampler, TestSampler

    train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=max_note_shift, max_timing_shift=max_timing_shift, augmentor=augmentor)
    


    evaluate_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=0, max_timing_shift=0)

    # Sampler for training - Use existing sampler
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Sampler for evaluation - Use existing sampler
    evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='val', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        weight_decay=1e-4)

    # Load pretrained model
    if args.pretrained_path:
        logging.info('Loading pretrained model from {}'.format(args.pretrained_path))
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logging.info('Pretrained model loaded successfully')

    # Move model to device
    model.to(device)

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Training loop
    train_bgn_time = time.time()
    iteration = resume_iteration
    
    # Track best validation performance for best checkpoint saving
    best_validation_metric = float('inf') 
    best_validation_iteration = 0

    for batch_data_dict in train_loader:
        # Forward
        model.train()

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        batch_output_dict = model(batch_data_dict['waveform'])

        # Loss
        loss = loss_func(model, batch_output_dict, batch_data_dict)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        loss = loss.item()
        statistics_container.append(iteration, {'loss': loss}, 'train')

        # Print training information
        if iteration % 100 == 0:
            logging.info('Iteration: {}, loss: {:.3f}, lr: {:.2e}'.format(
                iteration, loss, optimizer.param_groups[0]['lr']))

        # Reduce learning rate by 0.9 every 10K steps (following paper methodology)
        if iteration % reduce_iteration == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                logging.info('Reduce learning rate to {:.3e}'.format(param_group['lr']))

        # Save checkpoint
        if iteration % 1000 == 0 and iteration > 0:
            checkpoint_filename = '{}_{}_log{}_iter{}_lr{:.0e}_bs{}.pth'.format(
                dataset_name, loss_type, current_log_number, iteration, learning_rate, batch_size)
            checkpoint_path = os.path.join(log_checkpoints_dir, checkpoint_filename)
            torch.save({
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'dataset_name': dataset_name,
                'model_type': model_type,
                'batch_size': batch_size,
                'augmentation': augmentation,
                'max_note_shift': max_note_shift
            }, checkpoint_path)
            logging.info('Save checkpoint to {}'.format(checkpoint_path))

        # Evaluation
        if iteration % 1000 == 0 and iteration > 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))
            
            train_fin_time = time.time()
            
            # Use the proper SegmentEvaluator for evaluation
            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            validate_statistics = evaluator.evaluate(validate_loader)
            

            
            logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
            logging.info('    Validation statistics: {}'.format(validate_statistics))
            
            # Store statistics
            statistics_container.append(iteration, evaluate_train_statistics, 'train')
            statistics_container.append(iteration, validate_statistics, 'validation')
            statistics_container.dump()
            
            # Also save as readable text file
            stats_text_path = os.path.join(workspace, 'statistics', 'training_log.txt')
            with open(stats_text_path, 'a') as f:
                f.write(f"\n=== Iteration {iteration} ===\n")
                f.write(f"Train: {evaluate_train_statistics}\n")
                f.write(f"Validation: {validate_statistics}\n")
                f.write(f"Loss: {loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}\n")
                f.write("-" * 50 + "\n")
            
            # Check if this is the best validation performance so far
            if validate_statistics:
                # Try to get a performance metric (higher is better)
                performance_metric = None
                for key in ['f1', 'precision', 'recall', 'accuracy']:
                    if key in validate_statistics:
                        performance_metric = validate_statistics[key]
                        break
                
                # If no performance metric found, use negative loss (so higher is better)
                if performance_metric is None:
                    # Look for loss-related metrics
                    for key in ['loss', 'total_loss', 'note_loss']:
                        if key in validate_statistics:
                            performance_metric = -validate_statistics[key]  # Negative so higher is better
                            break
                
                # Update best checkpoint if performance improved
                if performance_metric is not None and performance_metric > best_validation_metric:
                    best_validation_metric = performance_metric
                    best_validation_iteration = iteration
                    
                    # Save best model checkpoint
                    best_checkpoint_path = os.path.join(log_checkpoints_dir, f'best_model_{dataset_name}_log{current_log_number}.pth')
                    torch.save({
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': loss,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'dataset_name': dataset_name,
                        'model_type': model_type,
                        'batch_size': batch_size,
                        'augmentation': augmentation,
                        'max_note_shift': max_note_shift,
                        'validation_metric': performance_metric,
                        'validation_statistics': validate_statistics
                    }, best_checkpoint_path)
                    logging.info('New best model saved! Validation metric: {:.4f} at iteration {}'.format(
                        performance_metric, iteration))
            
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            
            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))
            
            train_bgn_time = time.time()

        iteration += 1

        if iteration >= early_stop:
            break

    train_end_time = time.time()
    logging.info('Training time: {:.3f} s'.format(train_end_time - train_bgn_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune piano transcription model on any dataset')
    parser.add_argument('--workspace', type=str, required=True, 
        help='Directory of your workspace')
    parser.add_argument('--dataset_name', type=str, required=True,
        help='Name of the dataset (e.g., egdb, guitarset)')
    parser.add_argument('--hdf5s_dir', type=str, required=True,
        help='Path to HDF5 files directory')
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
    parser.add_argument('--early_stop', type=int, default=100000,
        help='Early stop at iteration (paper uses 100K steps)')
    parser.add_argument('--mini_data', action='store_true', default=False,
        help='Use small subset for debugging')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Use CUDA')
    parser.add_argument('--pretrained_path', type=str, default='PT.pth',
        help='Path to pretrained model')
    
    args = parser.parse_args()
    
    finetune(args)
