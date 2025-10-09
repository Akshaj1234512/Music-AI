#!/usr/bin/env python3
"""
Simplified script to find optimal threshold using only working configurations.
"""

import subprocess
import re
import numpy as np

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
            print(f"Error: {result.stderr}")
            return None
            
        # Extract F1 score from output
        output = result.stdout
        f1_match = re.search(r'note_f1: ([\d.]+)%', output)
        if f1_match:
            return float(f1_match.group(1))
        else:
            print(f"Could not extract F1 score from output")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    # Define working model configurations
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
            'hdf5s_dir': '/data/akshaj/MusicAI/workspace/hdf5s/combined/2024/val',
            'model_name': 'combined_regress_onset_offset_frame_velocity_bce_log46_iter28000_lr1e-05_bs4'
        }
    ]
    
    # Test key thresholds
    thresholds = [0.20, 0.225, 0.25, 0.275, 0.30]
    
    results = {}
    average_f1_scores = {}
    
    print("Testing onset thresholds for optimal standardized value...")
    print("=" * 60)
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        print("-" * 30)
        
        threshold_results = []
        
        for config in model_configs:
            print(f"  {config['name']}...", end=" ")
            f1_score = run_evaluation(config, threshold)
            
            if f1_score is not None:
                results[f"{threshold}_{config['name']}"] = f1_score
                threshold_results.append(f1_score)
                print(f"{f1_score:.1f}%")
            else:
                print("FAILED")
        
        if threshold_results:
            avg_f1 = np.mean(threshold_results)
            average_f1_scores[threshold] = avg_f1
            print(f"  Average F1: {avg_f1:.1f}%")
    
    # Find optimal threshold
    if average_f1_scores:
        optimal_threshold = max(average_f1_scores.keys(), key=lambda k: average_f1_scores[k])
        optimal_score = average_f1_scores[optimal_threshold]
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        
        print(f"\nOptimal standardized threshold: {optimal_threshold}")
        print(f"Average F1 score: {optimal_score:.1f}%")
        
        print(f"\nResults at optimal threshold ({optimal_threshold}):")
        for config in model_configs:
            key = f"{optimal_threshold}_{config['name']}"
            if key in results:
                print(f"  {config['name']}: {results[key]:.1f}%")
        
        print(f"\nAll threshold averages:")
        for threshold in sorted(average_f1_scores.keys()):
            print(f"  {threshold}: {average_f1_scores[threshold]:.1f}%")
            
    else:
        print("No successful evaluations!")

if __name__ == "__main__":
    main()
