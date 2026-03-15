import music21 as m21
from typing import List, Tuple, Dict, Any
import logging

# Initialize the logs, keeping them consistent with interface.py.
logger = logging.getLogger("B_Analysis")

# Quarter Length of a 16th Note
QL_16TH = 0.25 

# Group B Phase 2: Analysis
# Core functions: chord sequence splitting, MIDI parsing, and melody density calculation.

def consolidate_chord_sequence(chord_sequence: List[str]) -> List[Tuple[str, float]]:
    """
    Convert the chord sequence of 16th note slices (e.g., ['C', '_', '_', '_', 'G', '_', ...]) into a list of (chord labels, duration QL).
    
    Args:
        chord_sequence (List[str]): Group A predicts a sequence of 16th note slices.
        
    Returns:
        List[Tuple[str, float]]: A list of actual chord progressions.
    """
    if not chord_sequence:
        return []

    consolidated_chords: List[Tuple[str, float]] = []
    current_chord = chord_sequence[0]
    duration_slices = 0

    for label in chord_sequence:
        if label != '_' and label != current_chord:
            if current_chord != '_':
                consolidated_chords.append((current_chord, duration_slices * QL_16TH))
            
            current_chord = label
            duration_slices = 1
        elif label == '_':
            duration_slices += 1
        else:
            duration_slices += 1

    if current_chord != '_':
        consolidated_chords.append((current_chord, duration_slices * QL_16TH))
        
    final_list = [c for c in consolidated_chords if c[0] != '_']
    
    logger.info(f"The chord sequence has been divided into {len(final_list)} fragment(s)")
    return final_list


def get_melody_part(score: m21.stream.Score) -> m21.stream.Part:
    """
    Try to extract the melody part (usually the first part) from the music21.Score object.
    """
    if score.parts:
        return score.parts[0]
    
    logger.warning("Score has no Part, unable to extract melody")
    return m21.stream.Part()


def analyze_melody_density(melody_part: m21.stream.Part, consolidated_chords: List[Tuple[str, float]]) -> List[float]:
    """
    Calculate the note density of the melody within each variable-duration chord segment.
    
    Args:
        melody_part (m21.stream.Part): The extracted melody part.
        consolidated_chords (List[Tuple[str, float]]): A list of chord labels and durations (QL).
        
    Returns:
        List[float]: The melody density score (0.0 to 1.0) for each chord segment.
    """
    if not melody_part or not consolidated_chords:
        logger.error("Melody Part If the chord list is empty, return the density 0.0")
        return [0.0] * len(consolidated_chords)

    all_elements = melody_part.flat.notesAndRests
    density_scores: List[float] = []
    current_time = 0.0
    
    MAX_NOTES_PER_QL = 4.0 

    for i, (chord_label, segment_length) in enumerate(consolidated_chords):
        end_time = current_time + segment_length
        notes_in_segment = 0
        
        for element in all_elements:
            if element.offset >= current_time and element.offset < end_time:
                if isinstance(element, (m21.note.Note, m21.chord.Chord)):
                    notes_in_segment += 1

        max_notes_in_segment = segment_length * MAX_NOTES_PER_QL
        
        if max_notes_in_segment > 0:
            density = min(1.0, notes_in_segment / max_notes_in_segment)
        else:
            density = 0.0
            
        density_scores.append(density)
        logger.debug(f"Segment {i} ({chord_label}, {segment_length:.2f} QL): Notes={notes_in_segment}, Density={density:.2f}")
        
        current_time = end_time

    return density_scores