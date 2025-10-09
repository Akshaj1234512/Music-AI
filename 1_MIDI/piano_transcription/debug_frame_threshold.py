#!/usr/bin/env python3

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

import numpy as np
import pickle
from utilities import RegressionPostProcessor
import config

def debug_frame_threshold():
    """Debug frame threshold application"""
    
    # Load a sample pickle file
    probs_dir = '/data/akshaj/MusicAI/workspace/probs/model_type=Note_pedal/augmentation=none/dataset=guitarset/split=val'
    pickle_files = [f for f in os.listdir(probs_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        print("No pickle files found!")
        return
    
    # Load first pickle file
    pickle_path = os.path.join(probs_dir, pickle_files[0])
    print(f"Loading: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        total_dict = pickle.load(f)
    
    frame_output = total_dict['frame_output']
    frame_roll = total_dict['frame_roll']
    
    print(f"Frame output shape: {frame_output.shape}")
    print(f"Frame roll shape: {frame_roll.shape}")
    print(f"Frame output min/max: {frame_output.min():.4f} / {frame_output.max():.4f}")
    print(f"Frame roll min/max: {frame_roll.min():.4f} / {frame_roll.max():.4f}")
    
    # Test different frame thresholds
    for frame_thresh in [0.05, 0.1, 0.2, 0.3, 0.5]:
        print(f"\n--- Frame threshold: {frame_thresh} ---")
        
        # Apply frame threshold (same logic as in calculate_score_for_paper.py)
        y_pred = (np.sign(frame_output - frame_thresh) + 1) / 2
        y_pred[np.where(y_pred==0.5)] = 0
        
        # Trim to same length as ground truth
        y_pred = y_pred[0 : frame_roll.shape[0], :]
        y_true = frame_roll[0 : y_pred.shape[0], :]
        
        # Calculate metrics
        from sklearn import metrics
        tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
        
        print(f"Active frames: {np.sum(y_pred)}")
        print(f"Ground truth frames: {np.sum(y_true)}")
        print(f"Precision: {tmp[0][1]:.4f}")
        print(f"Recall: {tmp[1][1]:.4f}")
        print(f"F1: {tmp[2][1]:.4f}")

if __name__ == '__main__':
    debug_frame_threshold()
