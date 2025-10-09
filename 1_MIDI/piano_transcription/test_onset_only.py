#!/usr/bin/env python3
"""
Test onset thresholds from 0.1 to 0.6 in 0.05 increments to find the best F1 score.
Only tests onset threshold since that's what affects P50/R50/F50 metrics.
"""

import subprocess
import sys

def run_threshold_test(onset_thresh):
    """Run a single threshold test and return the metrics."""
    cmd = [
        'python', 'pytorch/calculate_score_for_paper.py', 'calculate_metrics',
        '--workspace', '/data/akshaj/MusicAI/workspace',
        '--model_type', 'Note_pedal',
        '--augmentation', 'none',
        '--dataset_name', 'guitarset',
        '--hdf5s_dir', '/data/akshaj/MusicAI/workspace/hdf5s/guitarset/combined/2024/val',
        '--split', 'val',
        '--thresholds', str(onset_thresh), '0.3', '0.1'  # offset=0.3, frame=0.1 (fixed)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"ERROR: Command failed for onset threshold {onset_thresh}")
            print(f"Error: {result.stderr}")
            return None
        
        # Parse the output to extract metrics
        output = result.stdout
        lines = output.strip().split('\n')
        metrics = {}
        
        for line in lines:
            if 'note_precision:' in line:
                metrics['precision'] = float(line.split(':')[1].replace('%', '').strip())
            elif 'note_recall:' in line:
                metrics['recall'] = float(line.split(':')[1].replace('%', '').strip())
            elif 'note_f1:' in line:
                metrics['f1'] = float(line.split(':')[1].replace('%', '').strip())
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Command timed out for onset threshold {onset_thresh}")
        return None
    except Exception as e:
        print(f"ERROR: Exception for onset threshold {onset_thresh}: {e}")
        return None

def main():
    print("Testing onset thresholds from 0.1 to 0.6...")
    print("Offset=0.3, Frame=0.1 (fixed)")
    print("=" * 50)
    
    # Test onset thresholds from 0.1 to 0.6 in 0.05 increments
    onset_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    results = []
    
    for i, onset_thresh in enumerate(onset_values, 1):
        print(f"\n[{i}/{len(onset_values)}] Testing onset threshold: {onset_thresh}")
        
        metrics = run_threshold_test(onset_thresh)
        
        if metrics:
            results.append({
                'onset': onset_thresh,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
            print(f"  → P: {metrics['precision']:.1f}%, R: {metrics['recall']:.1f}%, F1: {metrics['f1']:.1f}%")
        else:
            print(f"  → FAILED")
    
    # Sort results by F1 score (descending)
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (sorted by F1 score)")
    print("=" * 60)
    print(f"{'Rank':<4} {'Onset':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['onset']:<6} {result['precision']:<10.1f} {result['recall']:<10.1f} {result['f1']:<10.1f}")
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST ONSET THRESHOLD: {best['onset']}")
        print(f"   Precision: {best['precision']:.1f}%")
        print(f"   Recall: {best['recall']:.1f}%")
        print(f"   F1 Score: {best['f1']:.1f}%")
        
        # Save results to file
        with open('onset_threshold_results.txt', 'w') as f:
            f.write("Onset Threshold Test Results (sorted by F1 score)\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Onset':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-" * 50 + "\n")
            for result in results:
                f.write(f"{result['onset']:<6} {result['precision']:<10.1f} {result['recall']:<10.1f} {result['f1']:<10.1f}\n")
        
        print(f"\n📁 Results saved to: onset_threshold_results.txt")
    else:
        print("\n❌ No successful tests completed!")

if __name__ == "__main__":
    main()
