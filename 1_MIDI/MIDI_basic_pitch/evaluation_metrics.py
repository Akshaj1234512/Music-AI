"""
Evaluation metrics for Basic Pitch guitar transcription.
Compares predictions against ground truth MIDI using mir_eval and frame-level metrics.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

import numpy as np
import mir_eval
from sklearn import metrics
import pretty_midi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicPitchEvaluator:
    """
    Evaluator for Basic Pitch predictions against ground truth MIDI.
    Computes frame-level, onset-level, and note-level metrics.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize evaluator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # mir_eval tolerance settings
        self.onset_tolerance = self.config['evaluation']['onset_tolerance']
        self.offset_ratio = self.config['evaluation']['offset_ratio'] 
        self.offset_min_tolerance = self.config['evaluation']['offset_min_tolerance']
        
        # Basic Pitch frame rate (frames per second)
        self.frames_per_second = self.config['basic_pitch'].get('frames_per_second', 50)
        
        logger.info("✓ Basic Pitch evaluator initialized")
        logger.info(f"  Onset tolerance: {self.onset_tolerance}s")
        logger.info(f"  Offset tolerance: {self.offset_ratio} ratio, {self.offset_min_tolerance}s min")
    
    def evaluate_prediction(self, 
                          prediction_result: Dict[str, Any],
                          ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            prediction_result: Result from BasicPitchWrapper.transcribe_single()
            ground_truth: Ground truth from BasicPitchLoader.load_midi_ground_truth()
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics_dict = {}
        
        # Extract data
        model_output = prediction_result['model_output']
        predicted_notes = prediction_result['note_events']
        gt_notes = ground_truth['note_events']
        
        # 1. Frame-level evaluation
        if 'note' in model_output and 'onset' in model_output:
            # Ensure we have the pretty_midi object for frame conversion
            if 'midi_object' in ground_truth:
                frame_metrics = self._evaluate_frame_level(model_output, ground_truth['midi_object'])
                metrics_dict.update(frame_metrics)
            else:
                logger.warning("Skipping frame-level evaluation: 'midi_object' not in ground_truth")
        
        # 2. Note-level evaluation using mir_eval
        note_metrics = self._evaluate_note_level(predicted_notes, gt_notes)
        metrics_dict.update(note_metrics)
        
        return metrics_dict
    
    def _evaluate_frame_level(self, model_output: Dict[str, np.ndarray], 
                            gt_midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Evaluate frame-level predictions against ground truth.
        
        Args:
            model_output: Basic Pitch raw outputs (note, onset probabilities)
            ground_truth: Ground truth MIDI data
            
        Returns:
            Dictionary of frame-level metrics
        """
        # Get Basic Pitch frame predictions
        note_probs = model_output['note']    # [time, 88]
        onset_probs = model_output['onset']  # [time, 88]
        
        # Convert ground truth MIDI to frame representation
        gt_frames = self._midi_to_frames(gt_midi, note_probs.shape[0])
        gt_onsets = self._midi_to_onset_frames(gt_midi, onset_probs.shape[0])
        
        # Apply thresholds to get binary predictions
        frame_threshold = self.config['basic_pitch']['frame_threshold']
        onset_threshold = self.config['basic_pitch']['onset_threshold']
        
        pred_frames = (note_probs >= frame_threshold).astype(int)
        pred_onsets = (onset_probs >= onset_threshold).astype(int)
        
        # Ensure same shape (trim to minimum length)
        min_len = min(pred_frames.shape[0], gt_frames.shape[0])
        pred_frames = pred_frames[:min_len]
        gt_frames = gt_frames[:min_len] 
        pred_onsets = pred_onsets[:min_len]
        gt_onsets = gt_onsets[:min_len]


        
        # Compute frame-level metrics
        frame_metrics = {}
        
        # Frame F1, precision, recall
        frame_f1 = metrics.f1_score(gt_frames.flatten(), pred_frames.flatten(), average='micro')
        frame_precision = metrics.precision_score(gt_frames.flatten(), pred_frames.flatten(), average='micro', zero_division=0)
        frame_recall = metrics.recall_score(gt_frames.flatten(), pred_frames.flatten(), average='micro', zero_division=0)
        
        frame_metrics['frame_f1'] = frame_f1
        frame_metrics['frame_precision'] = frame_precision
        frame_metrics['frame_recall'] = frame_recall
        
        # Onset F1, precision, recall
        onset_f1 = metrics.f1_score(gt_onsets.flatten(), pred_onsets.flatten(), average='micro')
        onset_precision = metrics.precision_score(gt_onsets.flatten(), pred_onsets.flatten(), average='micro', zero_division=0)
        onset_recall = metrics.recall_score(gt_onsets.flatten(), pred_onsets.flatten(), average='micro', zero_division=0)
        
        frame_metrics['onset_f1'] = onset_f1
        frame_metrics['onset_precision'] = onset_precision
        frame_metrics['onset_recall'] = onset_recall
        
        return frame_metrics
    
    def _midi_to_frames(self, midi_data: pretty_midi.PrettyMIDI, num_frames: int) -> np.ndarray:
        """Convert ground truth MIDI to a piano roll using pretty_midi."""
        # The piano roll has a shape of (128, num_frames)
        piano_roll = midi_data.get_piano_roll(fs=self.frames_per_second)
        
        # Ensure piano_roll has the same number of frames as the prediction
        if piano_roll.shape[1] < num_frames:
            # Pad with zeros if shorter
            pad_width = num_frames - piano_roll.shape[1]
            piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant')
        elif piano_roll.shape[1] > num_frames:
            # Truncate if longer
            piano_roll = piano_roll[:, :num_frames]
            
        # Transpose to match Basic Pitch's (num_frames, pitch) shape and slice to 88 keys
        return piano_roll.T[:, 21:109]

    def _midi_to_onset_frames(self, midi_data: pretty_midi.PrettyMIDI, num_frames: int) -> np.ndarray:
        """Convert ground truth MIDI to an onset piano roll."""
        onsets = np.zeros((num_frames, 88), dtype=int)
        
        for note in midi_data.instruments[0].notes:
            start_frame = int(note.start * self.frames_per_second)
            pitch_index = note.pitch - 21  # Convert MIDI pitch to 88-key index
            
            if 0 <= start_frame < num_frames and 0 <= pitch_index < 88:
                onsets[start_frame, pitch_index] = 1
        
        return onsets  # Return 88-key piano range
    
    def _evaluate_note_level(self, predicted_notes: List[Tuple], 
                           gt_notes: List[Tuple]) -> Dict[str, float]:
        """
        Evaluate note-level predictions using mir_eval.
        
        Args:
            predicted_notes: List of (start, end, pitch, amplitude) from Basic Pitch
            gt_notes: List of (start, end, pitch, velocity) from ground truth
            
        Returns:
            Dictionary of note-level metrics
        """
        if not predicted_notes or not gt_notes:
            return {
                'note_f1': 0.0,
                'note_precision': 0.0,
                'note_recall': 0.0
            }
        
        # Convert to mir_eval format
        # Predicted intervals and pitches
        pred_intervals = np.array([[note[0], note[1]] for note in predicted_notes])
        pred_pitches = np.array([mir_eval.util.midi_to_hz(note[2]) for note in predicted_notes])
        
        # Ground truth intervals and pitches  
        gt_intervals = np.array([[note[0], note[1]] for note in gt_notes])
        gt_pitches = np.array([mir_eval.util.midi_to_hz(note[2]) for note in gt_notes])
        
        # Compute note-level metrics with mir_eval
        try:
            precision, recall, f1, overlap = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=gt_intervals,
                ref_pitches=gt_pitches,
                est_intervals=pred_intervals,
                est_pitches=pred_pitches,
                onset_tolerance=self.onset_tolerance,
                offset_ratio=self.offset_ratio,
                offset_min_tolerance=self.offset_min_tolerance
            )
            
            return {
                'note_f1': f1,
                'note_precision': precision, 
                'note_recall': recall
            }
            
        except Exception as e:
            logger.warning(f"mir_eval evaluation failed: {e}")
            return {
                'note_f1': 0.0,
                'note_precision': 0.0,
                'note_recall': 0.0
            }
    
    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate batch of predictions and compute aggregate metrics.
        
        Args:
            results: List of evaluation results from evaluate_prediction()
            
        Returns:
            Dictionary with aggregate metrics and per-file results
        """
        if not results:
            return {'error': 'No results to evaluate'}
        
        # Extract all metrics
        all_metrics = {}
        for result in results:
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Compute aggregate statistics
        aggregate_metrics = {}
        for key, values in all_metrics.items():
            values = [v for v in values if v is not None and not np.isnan(v)]
            if values:
                aggregate_metrics[f'{key}_mean'] = np.mean(values)
                aggregate_metrics[f'{key}_std'] = np.std(values)
                aggregate_metrics[f'{key}_median'] = np.median(values)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'num_files': len(results),
            'per_file_results': results
        }
    
    def format_results(self, metrics: Dict[str, float]) -> str:
        """Format evaluation results for display."""
        lines = ["=" * 50]
        lines.append("BASIC PITCH EVALUATION RESULTS")
        lines.append("=" * 50)
        
        if 'note_f1' in metrics:
            lines.append(f"Note-level Metrics (mir_eval):")
            lines.append(f"  F1 Score:  {metrics.get('note_f1', 0):.4f}")
            lines.append(f"  Precision: {metrics.get('note_precision', 0):.4f}")
            lines.append(f"  Recall:    {metrics.get('note_recall', 0):.4f}")
            lines.append("")
        
        if 'frame_f1' in metrics:
            lines.append(f"Frame-level Metrics:")
            lines.append(f"  F1 Score:  {metrics.get('frame_f1', 0):.4f}")
            lines.append(f"  Precision: {metrics.get('frame_precision', 0):.4f}")
            lines.append(f"  Recall:    {metrics.get('frame_recall', 0):.4f}")
            lines.append("")
        
        if 'onset_f1' in metrics:
            lines.append(f"Onset-level Metrics:")
            lines.append(f"  F1 Score:  {metrics.get('onset_f1', 0):.4f}")
            lines.append(f"  Precision: {metrics.get('onset_precision', 0):.4f}")
            lines.append(f"  Recall:    {metrics.get('onset_recall', 0):.4f}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    config_path = "config.yaml"
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        exit(1)
    
    evaluator = BasicPitchEvaluator(config_path)
    print("Basic Pitch evaluator initialized successfully!")