"""
Post-Processing for Fretting Transformer

Implements pitch validation and correction to achieve 100% pitch accuracy
as described in the paper. Handles overlap correction and neighbor search.
"""

import warnings
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

from ..data.tokenizer import FrettingTokenizer


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    
    # Standard guitar tuning (MIDI note numbers)
    standard_tuning: List[int] = None
    
    # Search parameters
    neighbor_search_window: int = 5  # ±5 notes window for finding alternatives
    max_fret: int = 24  # Maximum fret position
    
    # Correction priorities
    prefer_open_strings: bool = True
    prefer_lower_frets: bool = True
    
    def __post_init__(self):
        if self.standard_tuning is None:
            # E4, B3, G3, D3, A2, E2 (standard tuning)
            self.standard_tuning = [64, 59, 55, 50, 45, 40]


class FrettingPostProcessor:
    """
    Post-processing system for tablature correction.
    
    From paper: "Post-processing algorithm refines the model's output by 
    comparing the estimated note sequence to the corresponding input note sequence.
    It attempts to match each input note to its closest counterpart in the 
    estimated sequence within a configurable window of ±5 notes."
    """
    
    def __init__(self, 
                 tokenizer: FrettingTokenizer,
                 config: Optional[PostProcessConfig] = None):
        
        self.tokenizer = tokenizer
        self.config = config or PostProcessConfig()
        
    def process_tablature(self, 
                         input_tokens: List[str],
                         output_tokens: List[str],
                         tuning: Optional[List[int]] = None,
                         capo: int = 0) -> Tuple[List[str], Dict[str, int]]:
        """
        Post-process generated tablature to ensure pitch accuracy.
        
        Args:
            input_tokens: Input MIDI tokens
            output_tokens: Generated tablature tokens
            tuning: Guitar tuning (defaults to standard)
            capo: Capo position
            
        Returns:
            Tuple of (corrected_tokens, correction_stats)
        """
        if tuning is None:
            tuning = self.config.standard_tuning.copy()
        
        # Apply capo to tuning
        effective_tuning = [pitch + capo for pitch in tuning]
        
        # Extract note sequences
        input_notes = self._extract_input_notes(input_tokens)
        output_tabs = self._extract_output_tabs(output_tokens)
        
        # Align and correct
        corrected_tabs, stats = self._correct_tabs(input_notes, output_tabs, effective_tuning)
        
        # Reconstruct token sequence
        corrected_tokens = self._reconstruct_tokens(output_tokens, corrected_tabs)
        
        return corrected_tokens, stats
    
    def _extract_input_notes(self, input_tokens: List[str]) -> List[int]:
        """
        Extract MIDI note pitches from input tokens.
        
        Args:
            input_tokens: Input token sequence
            
        Returns:
            List of MIDI pitches
        """
        notes = []
        
        for token in input_tokens:
            if token.startswith('NOTE_ON<') and token.endswith('>'):
                # Extract pitch from NOTE_ON<pitch>
                try:
                    pitch_str = token[8:-1]  # Remove 'NOTE_ON<' and '>'
                    pitch = int(pitch_str)
                    notes.append(pitch)
                except ValueError:
                    warnings.warn(f"Invalid NOTE_ON token: {token}")
                    
        return notes
    
    def _extract_output_tabs(self, output_tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Extract (string, fret) pairs from output tokens.
        
        Args:
            output_tokens: Output token sequence
            
        Returns:
            List of (string, fret) tuples
        """
        tabs = []
        
        for token in output_tokens:
            if token.startswith('TAB<') and token.endswith('>'):
                # Extract string,fret from TAB<string,fret>
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
    
    def _correct_tabs(self, 
                     input_notes: List[int],
                     output_tabs: List[Tuple[int, int]],
                     tuning: List[int]) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
        """
        Correct tablature to match input pitches.
        
        Args:
            input_notes: Target MIDI pitches
            output_tabs: Generated (string, fret) pairs
            tuning: Effective guitar tuning
            
        Returns:
            Tuple of (corrected_tabs, correction_statistics)
        """
        corrected_tabs = []
        stats = {
            'total_notes': len(input_notes),
            'exact_matches': 0,
            'overlap_corrections': 0,
            'neighbor_corrections': 0,
            'fallback_corrections': 0,
            'uncorrectable': 0
        }
        
        # Handle length mismatch
        min_length = min(len(input_notes), len(output_tabs))
        if len(input_notes) != len(output_tabs):
            warnings.warn(
                f"Length mismatch: {len(input_notes)} input notes vs "
                f"{len(output_tabs)} output tabs. Processing {min_length} notes."
            )
        
        for i in range(min_length):
            target_pitch = input_notes[i]
            predicted_tab = output_tabs[i] if i < len(output_tabs) else (1, 0)
            
            # Check if current prediction is correct
            if self._is_tab_correct(predicted_tab, target_pitch, tuning):
                corrected_tabs.append(predicted_tab)
                stats['exact_matches'] += 1
                continue
            
            # Try overlap correction (within predicted sequence)
            overlap_correction = self._find_overlap_correction(
                target_pitch, output_tabs, i, tuning
            )
            
            if overlap_correction is not None:
                corrected_tabs.append(overlap_correction)
                stats['overlap_corrections'] += 1
                continue
            
            # Try neighbor search (find valid fingering)
            neighbor_correction = self._find_neighbor_correction(
                target_pitch, tuning
            )
            
            if neighbor_correction is not None:
                corrected_tabs.append(neighbor_correction)
                stats['neighbor_corrections'] += 1
                continue
            
            # Fallback: use any valid fingering
            fallback_correction = self._find_fallback_correction(
                target_pitch, tuning
            )
            
            if fallback_correction is not None:
                corrected_tabs.append(fallback_correction)
                stats['fallback_corrections'] += 1
            else:
                # This should rarely happen
                corrected_tabs.append((1, 0))  # Open high E as last resort
                stats['uncorrectable'] += 1
                warnings.warn(f"Could not correct pitch {target_pitch}")
        
        return corrected_tabs, stats
    
    def _is_tab_correct(self, 
                       tab: Tuple[int, int], 
                       target_pitch: int, 
                       tuning: List[int]) -> bool:
        """
        Check if a tablature produces the correct pitch.
        
        Args:
            tab: (string, fret) tuple
            target_pitch: Target MIDI pitch
            tuning: Guitar tuning
            
        Returns:
            True if tablature produces correct pitch
        """
        string, fret = tab
        
        # Validate string and fret ranges
        if not (1 <= string <= len(tuning)) or not (0 <= fret <= self.config.max_fret):
            return False
        
        # Calculate actual pitch
        open_pitch = tuning[string - 1]  # Convert to 0-based indexing
        actual_pitch = open_pitch + fret
        
        return actual_pitch == target_pitch
    
    def _find_overlap_correction(self, 
                               target_pitch: int,
                               output_tabs: List[Tuple[int, int]],
                               current_index: int,
                               tuning: List[int]) -> Optional[Tuple[int, int]]:
        """
        Find correction within a window of the predicted sequence.
        
        From paper: "within a configurable window of ±5 notes"
        
        Args:
            target_pitch: Target MIDI pitch
            output_tabs: All predicted tabs
            current_index: Current position in sequence
            tuning: Guitar tuning
            
        Returns:
            Corrected (string, fret) or None
        """
        window = self.config.neighbor_search_window
        
        # Search within window around current index
        start_idx = max(0, current_index - window)
        end_idx = min(len(output_tabs), current_index + window + 1)
        
        for i in range(start_idx, end_idx):
            if i != current_index:  # Don't check current position again
                tab = output_tabs[i]
                if self._is_tab_correct(tab, target_pitch, tuning):
                    return tab
        
        return None
    
    def _find_neighbor_correction(self, 
                                target_pitch: int,
                                tuning: List[int]) -> Optional[Tuple[int, int]]:
        """
        Find a valid fingering for the target pitch using neighbor search.
        
        Args:
            target_pitch: Target MIDI pitch
            tuning: Guitar tuning
            
        Returns:
            Valid (string, fret) or None
        """
        valid_fingerings = []
        
        # Find all possible fingerings for this pitch
        for string_idx, open_pitch in enumerate(tuning):
            fret = target_pitch - open_pitch
            
            if 0 <= fret <= self.config.max_fret:
                string_num = string_idx + 1  # Convert to 1-based
                valid_fingerings.append((string_num, fret))
        
        if not valid_fingerings:
            return None
        
        # Apply preferences to choose best fingering
        best_fingering = self._choose_best_fingering(valid_fingerings)
        
        return best_fingering
    
    def _choose_best_fingering(self, 
                             fingerings: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Choose the best fingering based on preferences.
        
        Args:
            fingerings: List of valid (string, fret) options
            
        Returns:
            Best (string, fret) choice
        """
        if not fingerings:
            return (1, 0)  # Fallback
        
        # Sort by preferences
        def fingering_score(tab):
            string, fret = tab
            score = 0
            
            # Prefer open strings
            if self.config.prefer_open_strings and fret == 0:
                score += 100
            
            # Prefer lower frets
            if self.config.prefer_lower_frets:
                score += (self.config.max_fret - fret)
            
            # Prefer middle strings (avoid extreme high/low)
            middle_string = len(self.config.standard_tuning) // 2
            string_preference = abs(string - middle_string)
            score -= string_preference * 5
            
            return score
        
        # Return fingering with highest score
        best_fingering = max(fingerings, key=fingering_score)
        return best_fingering
    
    def _find_fallback_correction(self, 
                                target_pitch: int,
                                tuning: List[int]) -> Optional[Tuple[int, int]]:
        """
        Find any valid fingering as fallback.
        
        Args:
            target_pitch: Target MIDI pitch
            tuning: Guitar tuning
            
        Returns:
            Any valid (string, fret) or None
        """
        # Try each string
        for string_idx, open_pitch in enumerate(tuning):
            fret = target_pitch - open_pitch
            
            # Allow higher frets as fallback
            if 0 <= fret <= 30:  # Extended range for fallback
                return (string_idx + 1, min(fret, self.config.max_fret))
        
        return None
    
    def _reconstruct_tokens(self, 
                          original_tokens: List[str],
                          corrected_tabs: List[Tuple[int, int]]) -> List[str]:
        """
        Reconstruct token sequence with corrected tablature.
        
        Args:
            original_tokens: Original output tokens
            corrected_tabs: Corrected (string, fret) pairs
            
        Returns:
            Corrected token sequence
        """
        corrected_tokens = []
        tab_index = 0
        
        for token in original_tokens:
            if token.startswith('TAB<') and token.endswith('>'):
                # Replace with corrected tab
                if tab_index < len(corrected_tabs):
                    string, fret = corrected_tabs[tab_index]
                    corrected_token = f"TAB<{string},{fret}>"
                    corrected_tokens.append(corrected_token)
                    tab_index += 1
                else:
                    # Keep original if we run out of corrections
                    corrected_tokens.append(token)
            else:
                # Keep non-tab tokens as-is
                corrected_tokens.append(token)
        
        return corrected_tokens
    
    def validate_pitch_accuracy(self, 
                              input_tokens: List[str],
                              output_tokens: List[str],
                              tuning: Optional[List[int]] = None,
                              capo: int = 0) -> Dict[str, float]:
        """
        Validate pitch accuracy of tablature.
        
        Args:
            input_tokens: Input MIDI tokens
            output_tokens: Output tablature tokens
            tuning: Guitar tuning
            capo: Capo position
            
        Returns:
            Accuracy metrics
        """
        if tuning is None:
            tuning = self.config.standard_tuning.copy()
        
        effective_tuning = [pitch + capo for pitch in tuning]
        
        input_notes = self._extract_input_notes(input_tokens)
        output_tabs = self._extract_output_tabs(output_tokens)
        
        if not input_notes or not output_tabs:
            return {'pitch_accuracy': 0.0, 'total_notes': 0}
        
        correct_count = 0
        total_count = min(len(input_notes), len(output_tabs))
        
        for i in range(total_count):
            if self._is_tab_correct(output_tabs[i], input_notes[i], effective_tuning):
                correct_count += 1
        
        pitch_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            'pitch_accuracy': pitch_accuracy,
            'correct_notes': correct_count,
            'total_notes': total_count,
            'input_length': len(input_notes),
            'output_length': len(output_tabs)
        }


def apply_postprocessing(input_tokens: List[str],
                        output_tokens: List[str],
                        tokenizer: FrettingTokenizer,
                        tuning: Optional[List[int]] = None,
                        capo: int = 0) -> Tuple[List[str], Dict[str, Union[int, float]]]:
    """
    Convenience function to apply post-processing.
    
    Args:
        input_tokens: Input MIDI tokens
        output_tokens: Generated tablature tokens
        tokenizer: Tokenizer instance
        tuning: Guitar tuning (optional)
        capo: Capo position
        
    Returns:
        Tuple of (corrected_tokens, metrics)
    """
    processor = FrettingPostProcessor(tokenizer)
    
    # Get accuracy before correction
    before_metrics = processor.validate_pitch_accuracy(
        input_tokens, output_tokens, tuning, capo
    )
    
    # Apply correction
    corrected_tokens, correction_stats = processor.process_tablature(
        input_tokens, output_tokens, tuning, capo
    )
    
    # Get accuracy after correction
    after_metrics = processor.validate_pitch_accuracy(
        input_tokens, corrected_tokens, tuning, capo
    )
    
    # Combine metrics
    final_metrics = {
        'pitch_accuracy_before': before_metrics['pitch_accuracy'],
        'pitch_accuracy_after': after_metrics['pitch_accuracy'],
        'improvement': after_metrics['pitch_accuracy'] - before_metrics['pitch_accuracy'],
        **correction_stats
    }
    
    return corrected_tokens, final_metrics


def test_postprocessing():
    """Test the post-processing system."""
    from ..data.tokenizer import FrettingTokenizer
    
    tokenizer = FrettingTokenizer()
    processor = FrettingPostProcessor(tokenizer)
    
    # Test with sample tokens
    input_tokens = [
        '<BOS>', 'NOTE_ON<55>', 'TIME_SHIFT<120>', 'NOTE_OFF<55>',
        'NOTE_ON<57>', 'TIME_SHIFT<120>', 'NOTE_OFF<57>', '<EOS>'
    ]
    
    # Output with one incorrect tab (should be string 3, fret 0 for pitch 55)
    output_tokens = [
        '<BOS>', 'TAB<2,5>', 'TIME_SHIFT<120>',  # Wrong: 2,5 gives pitch 54
        'TAB<3,2>', 'TIME_SHIFT<120>', '<EOS>'   # Correct: 3,2 gives pitch 57
    ]
    
    print("Testing post-processing...")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    
    # Validate before correction
    before_accuracy = processor.validate_pitch_accuracy(input_tokens, output_tokens)
    print(f"Accuracy before correction: {before_accuracy['pitch_accuracy']:.2%}")
    
    # Apply post-processing
    corrected_tokens, stats = processor.process_tablature(input_tokens, output_tokens)
    print(f"Corrected tokens: {corrected_tokens}")
    print(f"Correction stats: {stats}")
    
    # Validate after correction
    after_accuracy = processor.validate_pitch_accuracy(input_tokens, corrected_tokens)
    print(f"Accuracy after correction: {after_accuracy['pitch_accuracy']:.2%}")
    
    print("Post-processing test completed!")


if __name__ == "__main__":
    test_postprocessing()