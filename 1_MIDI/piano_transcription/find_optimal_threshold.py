#!/usr/bin/env python3
"""
Script to find the optimal standardized onset threshold that maximizes 
average F1 score across all model variants.
"""

import subprocess
import sys
import re
import numpy as np
from collections import defaultdict

def run_evaluation(model_config, threshold):
    """Run evaluation for a single model configuration and threshold."""
    cmd = [
        'python', 'pytorch/calculate_score_for_paper.py', 'calculate_metrics',
        '--workspace', '/data/akshaj/MusicAI/workspace',
        '--model_type', 'Note_pedal',
        '--augmentation', 'none',
        '--dataset_name', model_config['dataset_name'],
        '--hdf5s_dir', model_config['hdf5s_dir'],
        '--split', 'val',
        '--model_name', model_config['model_name'],
        '--thresholds', str(threshold), '0.3', '0.1'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/data/akshaj/MusicAI/piano_transcription')
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
            return None
            
        # Extract F1 score from output
        output = result.stdout
        f1_match = re.search(r'note_f1: ([\d.]+)%', output)
        if f1_match:
            return float(f1_match.group(1))
        else:
            print(f"Could not extract F1 score from output for {model_config['name']} at threshold {threshold}")
            return None
            
    except Exception as e:
        print(f"Exception running evaluation: {e}")
        return None

def main():
    # Define all model configurations to test
    model_configs = [
        {
            'name': 'GuitarSet Model',
            'dataset_name': 'GuitarSet',
            'hdf5s_dir': '/data/akshaj/MusicAI/workspace/hdf5s/guitarset/combined/2024/val',
            'model_name': 'guitarset_regress_onset_offset_frame_velocity_bce_log40_iter68000_lr1e-05_bs4'
        },
        {
            'name': 'Combined Model on GuitarSet',
            'dataset_name': 'combined',
            'hdf5s_dir': '/data/akshaj/MusicAI/workspace/hdf5s/guitarset/combined/2024/val',
            'model_name': 'combined_regress_onset_offset_frame_velocity_bce_log46_iter28000_lr1e-05_bs4'
        },
        {
            'name': 'EGDB Model',
            'dataset_name': 'egdb',
            'hdf5s_dir': '/data/akshaj/MusicAI/workspace/hdf5s/egdb/combined/2024/val',
            'model_name': 'egdb_regress_onset_offset_frame_velocity_bce_log43_iter36000_lr1e-05_bs4'
        },
        {
            'name': 'Combined Model on EGDB',
            'dataset_name': 'combined',
            'hdf5s_dir': '/data/akshaj/MusicAI/workspace/hdf5s/egdb/combined/2024/val',
            'model_name': 'combined_regress_onset_offset_frame_velocity_bce_log46_iter28000_lr1e-05_bs4'
        },
        {
            'name': 'Combined Model on Combined Dataset',
            'dataset_name': 'combined',
            'hdf5s_dir': '/data/akshaj/MusicAI/workspace/hdf5s/combined/combined/2024/val',
            'model_name': 'combined_regress_onset_offset_frame_velocity_bce_log46_iter28000_lr1e-05_bs4'
        }
    ]
    
    # Test thresholds from 0.2 to 0.3 in steps of 0.025
    thresholds = [0.20, 0.225, 0.25, 0.275, 0.30]
    
    # Store results
    results = defaultdict(dict)
    average_f1_scores = {}
    
    print("Testing onset thresholds from 0.2 to 0.3...")
    print("=" * 60)
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        print("-" * 30)
        
        threshold_results = []
        
        for config in model_configs:
            print(f"  Testing {config['name']}...")
            f1_score = run_evaluation(config, threshold)
            
            if f1_score is not None:
                results[threshold][config['name']] = f1_score
                threshold_results.append(f1_score)
                print(f"    F1 Score: {f1_score:.1f}%")
            else:
                print(f"    Failed to get F1 score")
        
        if threshold_results:
            avg_f1 = np.mean(threshold_results)
            average_f1_scores[threshold] = avg_f1
            print(f"  Average F1 Score: {avg_f1:.1f}%")
    
    # Find optimal threshold
    if average_f1_scores:
        optimal_threshold = max(average_f1_scores.keys(), key=lambda k: average_f1_scores[k])
        optimal_score = average_f1_scores[optimal_threshold]
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\nOptimal standardized threshold: {optimal_threshold}")
        print(f"Average F1 score at optimal threshold: {optimal_score:.1f}%")
        
        print(f"\nDetailed results for threshold {optimal_threshold}:")
        for model_name, f1_score in results[optimal_threshold].items():
            print(f"  {model_name}: {f1_score:.1f}%")
        
        print(f"\nAll threshold results:")
        for threshold in sorted(average_f1_scores.keys()):
            print(f"  Threshold {threshold}: {average_f1_scores[threshold]:.1f}%")
            
        print(f"\nRecommendation: Use threshold {optimal_threshold} for standardized evaluation")
        
    else:
        print("No successful evaluations completed!")

if __name__ == "__main__":
    main()
