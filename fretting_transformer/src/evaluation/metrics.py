"""
Evaluation Metrics for Fretting Transformer

Implements the three metrics described in the paper:
1. Pitch Accuracy - How well the model reproduces original pitches
2. Tab Accuracy - Agreement with professionally created ground-truth tablatures  
3. Playability Score - Objective evaluation based on difficulty estimation framework
"""

import math
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np

from ..data.tokenizer import FrettingTokenizer
from ..data.synthtab_loader import Note


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Standard guitar tuning
    standard_tuning: List[int] = None
    
    # Difficulty calculation parameters (from paper)
    alpha: float = 0.25  # Locality factor for fret position difficulty
    max_fret: int = 24
    
    # String numbering (1=high E, 6=low E)
    num_strings: int = 6
    
    def __post_init__(self):
        if self.standard_tuning is None:
            # E4, B3, G3, D3, A2, E2 (MIDI numbers)
            self.standard_tuning = [64, 59, 55, 50, 45, 40]


class FrettingEvaluator:
    """
    Evaluation system implementing the paper's metrics.
    
    From paper: "Evaluating guitar tablatures requires domain-specific metrics,
    as conventional machine learning and NLP metrics miss crucial aspects 
    of musicality and playability."
    """
    
    def __init__(self, 
                 tokenizer: FrettingTokenizer,
                 config: Optional[EvaluationConfig] = None):
        
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        
    def evaluate_sequence(self,
                         input_tokens: List[str],
                         predicted_tokens: List[str], 
                         ground_truth_tokens: List[str],
                         tuning: Optional[List[int]] = None,
                         capo: int = 0) -> Dict[str, float]:
        """
        Evaluate a single sequence with all metrics.
        
        Args:
            input_tokens: Input MIDI tokens
            predicted_tokens: Model-generated tablature tokens
            ground_truth_tokens: Reference tablature tokens
            tuning: Guitar tuning (defaults to standard)
            capo: Capo position
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if tuning is None:
            tuning = self.config.standard_tuning.copy()
        
        # Apply capo to tuning
        effective_tuning = [pitch + capo for pitch in tuning]
        
        # Extract sequences
        input_pitches = self._extract_midi_pitches(input_tokens)
        predicted_tabs = self._extract_tablature_pairs(predicted_tokens)
        ground_truth_tabs = self._extract_tablature_pairs(ground_truth_tokens)
        
        # Calculate metrics
        pitch_accuracy = self.calculate_pitch_accuracy(
            input_pitches, predicted_tabs, effective_tuning
        )
        
        tab_accuracy = self.calculate_tab_accuracy(
            predicted_tabs, ground_truth_tabs
        )
        
        predicted_difficulty = self.calculate_playability_score(predicted_tabs)
        gt_difficulty = self.calculate_playability_score(ground_truth_tabs)
        
        return {
            'pitch_accuracy': pitch_accuracy,
            'tab_accuracy': tab_accuracy,
            'predicted_difficulty': predicted_difficulty,
            'ground_truth_difficulty': gt_difficulty,
            'difficulty_ratio': predicted_difficulty / gt_difficulty if gt_difficulty > 0 else 1.0,
            'sequence_length': len(input_pitches),
            'num_predicted_tabs': len(predicted_tabs),
            'num_ground_truth_tabs': len(ground_truth_tabs)
        }
    
    def calculate_pitch_accuracy(self, 
                               input_pitches: List[int],
                               predicted_tabs: List[Tuple[int, int]],
                               tuning: List[int]) -> float:
        """
        Calculate pitch accuracy metric.
        
        From paper: "The pitch accuracy metric, ranging from 0% to 100%, 
        measures how well the model reproduces the original pitches from 
        the MIDI input. It allows for alternative string-fret combinations 
        as long as the pitch remains correct."
        
        Args:
            input_pitches: Target MIDI pitches
            predicted_tabs: Predicted (string, fret) pairs
            tuning: Guitar tuning
            
        Returns:
            Pitch accuracy as percentage (0.0 to 1.0)
        """
        if not input_pitches or not predicted_tabs:
            return 0.0
        
        # Handle length mismatch
        min_length = min(len(input_pitches), len(predicted_tabs))
        if len(input_pitches) != len(predicted_tabs):
            warnings.warn(
                f"Length mismatch in pitch accuracy: {len(input_pitches)} "
                f"input vs {len(predicted_tabs)} predicted"
            )
        
        correct_count = 0
        
        for i in range(min_length):
            target_pitch = input_pitches[i]
            predicted_tab = predicted_tabs[i]
            
            if self._is_pitch_correct(predicted_tab, target_pitch, tuning):
                correct_count += 1
        
        return correct_count / min_length
    
    def calculate_tab_accuracy(self,
                             predicted_tabs: List[Tuple[int, int]],
                             ground_truth_tabs: List[Tuple[int, int]]) -> float:
        """
        Calculate tab accuracy metric.
        
        From paper: "The tab accuracy, also ranging from 0% to 100%, 
        reflects how well the professionally created ground-truth tablatures 
        agree with the estimated fretting. This metric compares the predicted 
        string-fret combinations with the ground truth."
        
        Args:
            predicted_tabs: Predicted (string, fret) pairs
            ground_truth_tabs: Ground truth (string, fret) pairs
            
        Returns:
            Tab accuracy as percentage (0.0 to 1.0)
        """
        if not predicted_tabs or not ground_truth_tabs:
            return 0.0
        
        # Handle length mismatch
        min_length = min(len(predicted_tabs), len(ground_truth_tabs))
        if len(predicted_tabs) != len(ground_truth_tabs):
            warnings.warn(
                f"Length mismatch in tab accuracy: {len(predicted_tabs)} "
                f"predicted vs {len(ground_truth_tabs)} ground truth"
            )
        
        exact_matches = 0
        
        for i in range(min_length):
            if predicted_tabs[i] == ground_truth_tabs[i]:
                exact_matches += 1
        
        return exact_matches / min_length
    
    def calculate_playability_score(self, tabs: List[Tuple[int, int]]) -> float:
        """
        Calculate playability difficulty score.
        
        From paper: "A modified version of the difficulty estimation framework
        is used to objectively evaluate the playability of tablatures. The scoring 
        system takes into account two types of movement: horizontal shifts along 
        the fretboard (along) and vertical shifts across the strings (across)."
        
        Difficulty formula from paper:
        difficulty(p,q) = along(p,q) + across(p,q)
        where:
        along(p,q) = fret_stretch(p,q) + locality(p,q)  
        across(p,q) = vertical_stretch(p,q)
        
        Args:
            tabs: List of (string, fret) pairs
            
        Returns:
            Average difficulty score (0 to ~18.5 for 24-fret guitar)
        """
        if len(tabs) < 2:
            return 0.0
        
        total_difficulty = 0.0
        
        for i in range(len(tabs) - 1):
            current_tab = tabs[i]
            next_tab = tabs[i + 1]
            
            difficulty = self._calculate_transition_difficulty(current_tab, next_tab)
            total_difficulty += difficulty
        
        # Return average difficulty per transition
        return total_difficulty / (len(tabs) - 1)
    
    def _calculate_transition_difficulty(self, 
                                      tab1: Tuple[int, int], 
                                      tab2: Tuple[int, int]) -> float:
        """
        Calculate difficulty of transition between two tablature positions.
        
        Implements the exact formulas from the paper.
        
        Args:
            tab1: Starting (string, fret) position
            tab2: Ending (string, fret) position
            
        Returns:
            Difficulty score for this transition
        """
        string1, fret1 = tab1
        string2, fret2 = tab2
        
        # Calculate along (horizontal) difficulty
        along_difficulty = self._calculate_along_difficulty(fret1, fret2)
        
        # Calculate across (vertical) difficulty  
        across_difficulty = self._calculate_across_difficulty(string1, string2)
        
        return along_difficulty + across_difficulty
    
    def _calculate_along_difficulty(self, fret1: int, fret2: int) -> float:
        """
        Calculate horizontal (along fretboard) difficulty.
        
        From paper:
        along(p,q) = fret_stretch(p,q) + locality(p,q)
        
        fret_stretch(p,q) = {
            0.50 * |Δfret| if Δfret > 0,
            0.75 * |Δfret| if Δfret ≤ 0
        }
        
        locality(p,q) = α * (p + q), where α = 0.25
        """
        delta_fret = fret2 - fret1
        
        # Fret stretch calculation
        if delta_fret > 0:
            fret_stretch = 0.50 * abs(delta_fret)
        else:
            fret_stretch = 0.75 * abs(delta_fret)
        
        # Locality calculation
        locality = self.config.alpha * (fret1 + fret2)
        
        return fret_stretch + locality
    
    def _calculate_across_difficulty(self, string1: int, string2: int) -> float:
        """
        Calculate vertical (across strings) difficulty.
        
        From paper:
        vertical_stretch(p,q) = {
            0.25 if Δstring ≤ 1,
            0.50 if Δstring > 1
        }
        """
        delta_string = abs(string2 - string1)
        
        if delta_string <= 1:
            return 0.25
        else:
            return 0.50
    
    def _is_pitch_correct(self, 
                         tab: Tuple[int, int], 
                         target_pitch: int, 
                         tuning: List[int]) -> bool:
        """
        Check if tablature produces the correct pitch.
        
        Args:
            tab: (string, fret) pair
            target_pitch: Target MIDI pitch
            tuning: Guitar tuning
            
        Returns:
            True if pitch matches
        """
        string, fret = tab
        
        # Validate ranges
        if not (1 <= string <= len(tuning)) or not (0 <= fret <= self.config.max_fret):
            return False
        
        # Calculate actual pitch
        open_pitch = tuning[string - 1]  # Convert to 0-based indexing
        actual_pitch = open_pitch + fret
        
        return actual_pitch == target_pitch
    
    def _extract_midi_pitches(self, tokens: List[str]) -> List[int]:
        """Extract MIDI pitches from input tokens."""
        pitches = []
        
        for token in tokens:
            if token.startswith('NOTE_ON<') and token.endswith('>'):
                try:
                    pitch_str = token[8:-1]  # Remove 'NOTE_ON<' and '>'
                    pitch = int(pitch_str)
                    pitches.append(pitch)
                except ValueError:
                    warnings.warn(f"Invalid NOTE_ON token: {token}")
        
        return pitches
    
    def _extract_tablature_pairs(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """Extract (string, fret) pairs from tablature tokens."""
        tabs = []
        
        for token in tokens:
            if token.startswith('TAB<') and token.endswith('>'):
                try:
                    content = token[4:-1]  # Remove 'TAB<' and '>'
                    parts = content.split(',')
                    if len(parts) == 2:
                        string = int(parts[0])
                        fret = int(parts[1])
                        tabs.append((string, fret))
                except ValueError:
                    warnings.warn(f"Invalid TAB token: {token}")
        
        return tabs
    
    def evaluate_dataset(self, 
                        predictions: List[Dict[str, List[str]]],
                        ground_truth: List[Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Evaluate predictions on entire dataset.
        
        Args:
            predictions: List of prediction dictionaries with keys:
                        'input_tokens', 'predicted_tokens'
            ground_truth: List of ground truth dictionaries with keys:
                         'input_tokens', 'ground_truth_tokens'
        
        Returns:
            Aggregated evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        all_metrics = []
        
        for pred, gt in zip(predictions, ground_truth):
            try:
                metrics = self.evaluate_sequence(
                    input_tokens=pred.get('input_tokens', gt['input_tokens']),
                    predicted_tokens=pred['predicted_tokens'],
                    ground_truth_tokens=gt['ground_truth_tokens']
                )
                all_metrics.append(metrics)
            except Exception as e:
                warnings.warn(f"Failed to evaluate sequence: {e}")
        
        if not all_metrics:
            return {'error': 'No sequences could be evaluated'}
        
        # Aggregate metrics
        aggregated = {
            'num_sequences': len(all_metrics),
            'pitch_accuracy_mean': np.mean([m['pitch_accuracy'] for m in all_metrics]),
            'pitch_accuracy_std': np.std([m['pitch_accuracy'] for m in all_metrics]),
            'tab_accuracy_mean': np.mean([m['tab_accuracy'] for m in all_metrics]),
            'tab_accuracy_std': np.std([m['tab_accuracy'] for m in all_metrics]),
            'predicted_difficulty_mean': np.mean([m['predicted_difficulty'] for m in all_metrics]),
            'predicted_difficulty_std': np.std([m['predicted_difficulty'] for m in all_metrics]),
            'ground_truth_difficulty_mean': np.mean([m['ground_truth_difficulty'] for m in all_metrics]),
            'ground_truth_difficulty_std': np.std([m['ground_truth_difficulty'] for m in all_metrics]),
            'difficulty_ratio_mean': np.mean([m['difficulty_ratio'] for m in all_metrics]),
            'difficulty_ratio_std': np.std([m['difficulty_ratio'] for m in all_metrics])
        }
        
        # Add percentile statistics
        pitch_accuracies = [m['pitch_accuracy'] for m in all_metrics]
        tab_accuracies = [m['tab_accuracy'] for m in all_metrics]
        
        aggregated.update({
            'pitch_accuracy_p50': np.percentile(pitch_accuracies, 50),
            'pitch_accuracy_p90': np.percentile(pitch_accuracies, 90),
            'pitch_accuracy_p95': np.percentile(pitch_accuracies, 95),
            'tab_accuracy_p50': np.percentile(tab_accuracies, 50),
            'tab_accuracy_p90': np.percentile(tab_accuracies, 90),
            'tab_accuracy_p95': np.percentile(tab_accuracies, 95)
        })
        
        return aggregated
    
    def create_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """
        Create a formatted evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary
            
        Returns:
            Formatted report string
        """
        if 'error' in metrics:
            return f"Evaluation Error: {metrics['error']}"
        
        report = "=== Fretting Transformer Evaluation Report ===\n\n"
        
        report += f"Dataset Statistics:\n"
        report += f"  Number of sequences: {metrics['num_sequences']}\n\n"
        
        report += f"Pitch Accuracy (Paper Metric 1):\n"
        report += f"  Mean: {metrics['pitch_accuracy_mean']:.2%} ± {metrics['pitch_accuracy_std']:.2%}\n"
        report += f"  Median: {metrics['pitch_accuracy_p50']:.2%}\n"
        report += f"  90th percentile: {metrics['pitch_accuracy_p90']:.2%}\n"
        report += f"  95th percentile: {metrics['pitch_accuracy_p95']:.2%}\n\n"
        
        report += f"Tab Accuracy (Paper Metric 2):\n"
        report += f"  Mean: {metrics['tab_accuracy_mean']:.2%} ± {metrics['tab_accuracy_std']:.2%}\n"
        report += f"  Median: {metrics['tab_accuracy_p50']:.2%}\n" 
        report += f"  90th percentile: {metrics['tab_accuracy_p90']:.2%}\n"
        report += f"  95th percentile: {metrics['tab_accuracy_p95']:.2%}\n\n"
        
        report += f"Playability Difficulty (Paper Metric 3):\n"
        report += f"  Predicted Mean: {metrics['predicted_difficulty_mean']:.4f} ± {metrics['predicted_difficulty_std']:.4f}\n"
        report += f"  Ground Truth Mean: {metrics['ground_truth_difficulty_mean']:.4f} ± {metrics['ground_truth_difficulty_std']:.4f}\n"
        report += f"  Difficulty Ratio: {metrics['difficulty_ratio_mean']:.4f} ± {metrics['difficulty_ratio_std']:.4f}\n"
        report += f"    (Ratio < 1.0 means predicted is easier than ground truth)\n\n"
        
        # Add paper comparison context
        report += "Paper Reference Values (Table 5):\n"
        report += "  GuitarToday: Tab Accuracy ~98%, Difficulty ~1.95\n"
        report += "  Leduc: Tab Accuracy ~72%, Difficulty ~4.24\n"
        report += "  DadaGP: Tab Accuracy ~82%, Difficulty ~2.41\n"
        
        return report


def test_evaluation_metrics():
    """Test the evaluation metrics system."""
    from ..data.tokenizer import FrettingTokenizer
    
    tokenizer = FrettingTokenizer()
    evaluator = FrettingEvaluator(tokenizer)
    
    # Test data
    input_tokens = [
        '<BOS>', 'NOTE_ON<55>', 'TIME_SHIFT<120>', 'NOTE_OFF<55>',
        'NOTE_ON<57>', 'TIME_SHIFT<120>', 'NOTE_OFF<57>', '<EOS>'
    ]
    
    predicted_tokens = [
        '<BOS>', 'TAB<3,0>', 'TIME_SHIFT<120>',  # G string, open (correct)
        'TAB<3,2>', 'TIME_SHIFT<120>', '<EOS>'   # G string, 2nd fret (correct)
    ]
    
    ground_truth_tokens = [
        '<BOS>', 'TAB<3,0>', 'TIME_SHIFT<120>',  # Same as predicted
        'TAB<2,2>', 'TIME_SHIFT<120>', '<EOS>'   # Different fingering (B string, 2nd fret)
    ]
    
    print("Testing evaluation metrics...")
    
    metrics = evaluator.evaluate_sequence(
        input_tokens, predicted_tokens, ground_truth_tokens
    )
    
    print("Evaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test difficulty calculation specifically
    tabs = [(3, 0), (3, 2), (2, 2), (1, 0)]  # Some tab sequence
    difficulty = evaluator.calculate_playability_score(tabs)
    print(f"\nPlayability difficulty for sample sequence: {difficulty:.4f}")
    
    print("Evaluation metrics test completed!")


if __name__ == "__main__":
    test_evaluation_metrics()