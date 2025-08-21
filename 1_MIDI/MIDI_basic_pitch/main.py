"""
Main script for Basic Pitch guitar transcription evaluation.
Runs end-to-end evaluation: audio -> Basic Pitch -> metrics vs ground truth.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

import yaml
import pretty_midi

from basic_pitch_wrapper import BasicPitchWrapper
from basic_pitch_loader import BasicPitchLoader
from evaluation_metrics import BasicPitchEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_evaluation(config_path: str, output_file: str = None, max_files: int = None):
    """
    Run complete Basic Pitch evaluation pipeline.
    
    Args:
        config_path: Path to YAML configuration file
        output_file: Path to save results JSON (optional)
        max_files: Maximum number of files to process (optional, for testing)
    """
    logger.info("Starting Basic Pitch evaluation pipeline...")
    
    # Initialize components
    logger.info("Initializing components...")
    
    wrapper = BasicPitchWrapper(config_path)
    loader = BasicPitchLoader(config_path) 
    evaluator = BasicPitchEvaluator(config_path)
    
    # Get all available file pairs
    all_pairs = loader.get_all_pairs()
    
    if not all_pairs:
        logger.error("No matching audio-MIDI pairs found!")
        return None
    
    # Limit files if requested (for testing)
    if max_files is not None:
        all_pairs = all_pairs[:max_files]
        logger.info(f"Processing first {len(all_pairs)} files (limited by max_files)")
    
    logger.info(f"Processing {len(all_pairs)} audio-MIDI pairs...")
    
    # Process each file
    results = []
    start_time = time.time()
    
    for i, pair in enumerate(all_pairs):
        filename = pair['filename']
        logger.info(f"Processing {i+1}/{len(all_pairs)}: {filename}")
        
        
        # Load ground truth
        ground_truth = loader.load_midi_ground_truth(pair['midi_path'])
        
        # Load the MIDI file as a pretty_midi object for frame evaluation
        try:
            midi_object = pretty_midi.PrettyMIDI(str(pair['midi_path']))
            ground_truth['midi_object'] = midi_object
        except Exception as e:
            logger.error(f"Could not load MIDI file {pair['midi_path']}: {e}")
            continue
        
        # Run Basic Pitch prediction
        prediction = wrapper.transcribe_single(str(pair['audio_path']))
        
        # Evaluate prediction vs ground truth
        metrics = evaluator.evaluate_prediction(prediction, ground_truth)
        
        # Store result
        result = {
            'filename': filename,
            'audio_path': str(pair['audio_path']),
            'midi_path': str(pair['midi_path']),
            'processing_time': prediction['processing_time'],
            'num_predicted_notes': len(prediction['note_events']),
            'num_ground_truth_notes': ground_truth['num_notes'],
            'audio_duration': ground_truth['duration'],
            'metrics': metrics
        }
        
        results.append(result)
        
        # Log key metrics
        logger.info(f"  ✓ Predicted notes: {result['num_predicted_notes']}")
        logger.info(f"  ✓ Ground truth notes: {result['num_ground_truth_notes']}")
        logger.info(f"  ✓ Note F1: {metrics.get('note_f1', 0):.4f}")
        logger.info(f"  ✓ Frame F1: {metrics.get('frame_f1', 0):.4f}")
        
        
    
    total_time = time.time() - start_time
    
    # Compute aggregate metrics
    logger.info("Computing aggregate metrics...")
    aggregate_results = evaluator.evaluate_batch(results)
    
    # Create final results dictionary
    final_results = {
        'evaluation_summary': {
            'total_files_processed': len(results),
            'total_processing_time': total_time,
            'average_time_per_file': total_time / len(results) if results else 0,
        },
        'aggregate_metrics': aggregate_results.get('aggregate_metrics', {}),
        'individual_results': results,
        'config_used': config_path
    }
    
    # Display results
    display_results(final_results)
    
    # Save results if requested
    if output_file:
        save_results(final_results, output_file)
    
    return final_results


def display_results(results: Dict[str, Any]):
    """Display evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("BASIC PITCH EVALUATION RESULTS")
    print("=" * 60)
    
    summary = results['evaluation_summary']
    print(f"Total time: {summary['total_processing_time']:.1f}s")
    print(f"Average time per file: {summary['average_time_per_file']:.1f}s")
    print()
    
    metrics = results['aggregate_metrics']
    if metrics:
        print("AGGREGATE METRICS:")
        print("-" * 30)
        
        # Note-level metrics
        if 'note_f1_mean' in metrics:
            print(f"Note F1:       {metrics['note_f1_mean']:.4f} ± {metrics.get('note_f1_std', 0):.4f}")
            print(f"Note Precision: {metrics.get('note_precision_mean', 0):.4f} ± {metrics.get('note_precision_std', 0):.4f}")
            print(f"Note Recall:    {metrics.get('note_recall_mean', 0):.4f} ± {metrics.get('note_recall_std', 0):.4f}")
            print()
        
        # Frame-level metrics  
        if 'frame_f1_mean' in metrics:
            print(f"Frame F1:       {metrics['frame_f1_mean']:.4f} ± {metrics.get('frame_f1_std', 0):.4f}")
            print(f"Frame Precision: {metrics.get('frame_precision_mean', 0):.4f} ± {metrics.get('frame_precision_std', 0):.4f}")
            print(f"Frame Recall:    {metrics.get('frame_recall_mean', 0):.4f} ± {metrics.get('frame_recall_std', 0):.4f}")
            print()
        
        # Onset-level metrics
        if 'onset_f1_mean' in metrics:
            print(f"Onset F1:       {metrics['onset_f1_mean']:.4f} ± {metrics.get('onset_f1_std', 0):.4f}")
            print(f"Onset Precision: {metrics.get('onset_precision_mean', 0):.4f} ± {metrics.get('onset_precision_std', 0):.4f}")
            print(f"Onset Recall:    {metrics.get('onset_recall_mean', 0):.4f} ± {metrics.get('onset_recall_std', 0):.4f}")
    
    print("=" * 60)


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Basic Pitch Guitar Transcription Evaluation')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON file')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        print("\nPlease create a config.yaml file with the following structure:")
        print("""
paths:
  guitarset_audio_dir: "/path/to/audio/files"
  guitarset_midi_dir: "/path/to/midi/files"  
  output_dir: "results"

basic_pitch:
  onset_threshold: 0.3
  frame_threshold: 0.3
  minimum_frequency: 75.0
  maximum_frequency: 2000.0

gpu:
  device_id: 0
  use_gpu: true
        """)
        return
    
    # Run evaluation
    try:
        results = run_evaluation(
            config_path=args.config,
            output_file=args.output,
            max_files=args.max_files
        )
        
        if results is not None:
            logger.info("Evaluation completed successfully!")
        else:
            logger.error("Evaluation failed!")
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()