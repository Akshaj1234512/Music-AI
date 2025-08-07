import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import heapq


@dataclass
class GuitarNote:
    """Represents a note with guitar-specific information."""
    pitch: int  # MIDI pitch
    onset_time: float
    offset_time: float
    string: Optional[int] = None  # 0-5 (high E to low E)
    fret: Optional[int] = None    # 0-24
    confidence: float = 1.0


class DynamicProgrammingTabAssignment:
    """
    Dynamic programming-based string assignment for guitar tablature.
    
    Based on contemporary approaches that minimize:
    - Finger stretch
    - Position changes
    - String transitions
    - Physical impossibilities
    """
    
    def __init__(
        self,
        tuning: List[int] = [64, 59, 55, 50, 45, 40],  # Standard tuning E-A-D-G-B-E
        n_frets: int = 24,
        max_stretch: int = 5,
        capo: int = 0
    ):
        self.tuning = [t + capo for t in tuning]
        self.n_frets = n_frets
        self.max_stretch = max_stretch
        self.n_strings = len(tuning)
        
        # Pre-compute string-fret to pitch mapping
        self.pitch_map = self._create_pitch_map()
        self.position_map = self._create_position_map()
    
    def _create_pitch_map(self) -> Dict[Tuple[int, int], int]:
        """Create mapping from (string, fret) to MIDI pitch."""
        pitch_map = {}
        for string in range(self.n_strings):
            for fret in range(self.n_frets + 1):
                pitch = self.tuning[string] + fret
                pitch_map[(string, fret)] = pitch
        return pitch_map
    
    def _create_position_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """Create mapping from pitch to possible (string, fret) positions."""
        position_map = {}
        for (string, fret), pitch in self.pitch_map.items():
            if pitch not in position_map:
                position_map[pitch] = []
            position_map[pitch].append((string, fret))
        return position_map
    
    def assign_strings(
        self,
        notes: List[Tuple[int, float, float]],
        prefer_open_strings: bool = True,
        prefer_lower_positions: bool = True
    ) -> List[GuitarNote]:
        """
        Assign strings to notes using dynamic programming.
        
        Args:
            notes: List of (pitch, onset, offset) tuples
            prefer_open_strings: Bias towards open strings
            prefer_lower_positions: Bias towards lower fret positions
            
        Returns:
            List of GuitarNote objects with string/fret assignments
        """
        if not notes:
            return []
        
        # Sort by onset time
        sorted_notes = sorted(notes, key=lambda x: x[1])
        
        # Dynamic programming solution
        n = len(sorted_notes)
        
        # dp[i][pos] = (min_cost, prev_pos)
        # where pos = (string, fret)
        dp = [{} for _ in range(n)]
        
        # Initialize first note
        pitch, onset, offset = sorted_notes[0]
        if pitch not in self.position_map:
            # Pitch out of guitar range
            return []
        
        for pos in self.position_map[pitch]:
            cost = self._position_cost(pos, prefer_open_strings, prefer_lower_positions)
            dp[0][pos] = (cost, None)
        
        # Fill DP table
        for i in range(1, n):
            pitch, onset, offset = sorted_notes[i]
            
            if pitch not in self.position_map:
                continue
            
            prev_pitch, prev_onset, prev_offset = sorted_notes[i-1]
            
            for curr_pos in self.position_map[pitch]:
                min_cost = float('inf')
                best_prev = None
                
                # Try all previous positions
                for prev_pos in dp[i-1]:
                    prev_cost = dp[i-1][prev_pos][0]
                    
                    # Calculate transition cost
                    trans_cost = self._transition_cost(
                        prev_pos, curr_pos,
                        prev_offset, onset,
                        prefer_open_strings, prefer_lower_positions
                    )
                    
                    total_cost = prev_cost + trans_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev = prev_pos
                
                if best_prev is not None:
                    dp[i][curr_pos] = (min_cost, best_prev)
        
        # Backtrack to find optimal path
        if not dp[n-1]:
            return []
        
        # Find best final position
        best_final_pos = min(dp[n-1].keys(), key=lambda p: dp[n-1][p][0])
        
        # Reconstruct path
        path = []
        pos = best_final_pos
        for i in range(n-1, -1, -1):
            path.append(pos)
            if i > 0:
                pos = dp[i][pos][1]
        
        path.reverse()
        
        # Create GuitarNote objects
        guitar_notes = []
        for i, (pitch, onset, offset) in enumerate(sorted_notes):
            if i < len(path):
                string, fret = path[i]
                guitar_notes.append(GuitarNote(
                    pitch=pitch,
                    onset_time=onset,
                    offset_time=offset,
                    string=string,
                    fret=fret
                ))
        
        return guitar_notes
    
    def _position_cost(
        self,
        pos: Tuple[int, int],
        prefer_open: bool,
        prefer_low: bool
    ) -> float:
        """Calculate cost of a single position."""
        string, fret = pos
        cost = 0.0
        
        # Open string preference
        if prefer_open and fret == 0:
            cost -= 2.0
        
        # Lower position preference
        if prefer_low:
            cost += fret * 0.1
        
        # String preference (middle strings slightly preferred)
        string_pref = [0.2, 0.0, 0.0, 0.0, 0.1, 0.3]  # E-A-D-G-B-E
        cost += string_pref[string]
        
        return cost
    
    def _transition_cost(
        self,
        prev_pos: Tuple[int, int],
        curr_pos: Tuple[int, int],
        prev_offset: float,
        curr_onset: float,
        prefer_open: bool,
        prefer_low: bool
    ) -> float:
        """Calculate cost of transitioning between positions."""
        prev_string, prev_fret = prev_pos
        curr_string, curr_fret = curr_pos
        
        cost = 0.0
        
        # Time gap factor (closer notes = higher transition cost)
        time_gap = curr_onset - prev_offset
        time_factor = max(0.1, min(1.0, time_gap * 2))  # Normalize to [0.1, 1.0]
        
        # String change cost
        if prev_string != curr_string:
            string_distance = abs(prev_string - curr_string)
            cost += string_distance * 1.5 * time_factor
        
        # Fret distance cost (only for non-open strings)
        if prev_fret > 0 and curr_fret > 0:
            fret_distance = abs(prev_fret - curr_fret)
            
            # Check if stretch is feasible
            if fret_distance > self.max_stretch:
                cost += 10.0  # High penalty for impossible stretches
            else:
                cost += fret_distance * 0.5 * time_factor
        
        # Position shift cost (hand movement)
        if prev_fret > 0 or curr_fret > 0:
            position_shift = abs(self._get_position(prev_fret) - self._get_position(curr_fret))
            cost += position_shift * 0.3 * time_factor
        
        # Repeated string penalty (for fast passages)
        if prev_string == curr_string and time_gap < 0.1:
            cost += 2.0
        
        # Basic position cost
        cost += self._position_cost(curr_pos, prefer_open, prefer_low)
        
        return cost
    
    def _get_position(self, fret: int) -> int:
        """Get hand position for a fret (grouped by common positions)."""
        if fret == 0:
            return 0
        elif fret <= 3:
            return 1
        elif fret <= 5:
            return 2
        elif fret <= 7:
            return 3
        elif fret <= 10:
            return 4
        elif fret <= 12:
            return 5
        else:
            return 6 + (fret - 12) // 3


class TechniqueDetector:
    """
    Post-processing technique detection for assigned notes.
    Analyzes audio segments to identify playing techniques.
    """
    
    def __init__(self, model=None):
        self.model = model  # Placeholder for trained technique classifier
        
        # Technique transition rules
        self.slide_threshold = 2  # Minimum fret distance for slide
        self.bend_threshold = 0.1  # Minimum pitch variation for bend
    
    def detect_techniques(
        self,
        notes: List[GuitarNote],
        audio_features: Optional[np.ndarray] = None
    ) -> List[GuitarNote]:
        """
        Detect techniques based on note transitions and audio features.
        
        Args:
            notes: List of GuitarNote objects with string/fret assignments
            audio_features: Optional audio features for ML-based detection
            
        Returns:
            Notes with technique labels added
        """
        if len(notes) < 2:
            return notes
        
        # Rule-based technique detection
        for i in range(len(notes) - 1):
            curr_note = notes[i]
            next_note = notes[i + 1]
            
            # Skip if not on same string
            if curr_note.string != next_note.string:
                continue
            
            # Check for slide
            fret_diff = next_note.fret - curr_note.fret
            time_gap = next_note.onset_time - curr_note.offset_time
            
            if abs(fret_diff) >= self.slide_threshold and time_gap < 0.05:
                if fret_diff > 0:
                    curr_note.technique = 'slide_up'
                else:
                    curr_note.technique = 'slide_down'
            
            # Check for hammer-on / pull-off
            elif time_gap < 0.02:  # Very close notes
                if fret_diff > 0 and curr_note.fret > 0:
                    curr_note.technique = 'hammer_on'
                elif fret_diff < 0 and next_note.fret > 0:
                    curr_note.technique = 'pull_off'
        
        # ML-based detection would go here if model is provided
        if self.model is not None and audio_features is not None:
            # Placeholder for ML technique detection
            pass
        
        return notes


class TabFormatter:
    """Format GuitarNote objects into readable tablature."""
    
    @staticmethod
    def notes_to_tab(
        notes: List[GuitarNote],
        duration: float = None,
        resolution: float = 0.25,
        show_timing: bool = True
    ) -> str:
        """
        Convert notes to ASCII tablature.
        
        Args:
            notes: List of GuitarNote objects
            duration: Total duration (auto-detect if None)
            resolution: Time resolution in seconds
            show_timing: Whether to show timing marks
            
        Returns:
            Formatted tablature string
        """
        if not notes:
            return "No notes to display"
        
        # Auto-detect duration
        if duration is None:
            duration = max(n.offset_time for n in notes) + 1.0
        
        n_positions = int(duration / resolution)
        
        # Initialize tab lines
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        tab_lines = [[] for _ in range(6)]
        
        # Create position mapping
        for pos in range(n_positions):
            time = pos * resolution
            
            # Find notes at this position
            active_notes = []
            for note in notes:
                if note.onset_time <= time < note.offset_time:
                    active_notes.append(note)
            
            # Fill tab positions
            for string_idx in range(6):
                found = False
                for note in active_notes:
                    if note.string == string_idx:
                        # Add fret number
                        if len(str(note.fret)) == 1:
                            tab_lines[string_idx].append(str(note.fret))
                        else:
                            tab_lines[string_idx].append(f"({note.fret})")
                        found = True
                        break
                
                if not found:
                    tab_lines[string_idx].append('-')
        
        # Format output
        output = []
        
        # Add timing marks if requested
        if show_timing:
            timing_line = ' ' * 2  # Space for string name
            for pos in range(n_positions):
                if pos % 4 == 0:  # Mark every beat
                    timing_line += str((pos // 4) % 10)
                else:
                    timing_line += ' '
            output.append(timing_line)
        
        # Add tab lines
        for i, name in enumerate(string_names):
            line = name + '|' + ''.join(tab_lines[i]) + '|'
            output.append(line)
        
        return '\n'.join(output)