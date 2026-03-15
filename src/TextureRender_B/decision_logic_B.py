import music21 as m21
from typing import List, Dict, Tuple
import logging
import math
import sys
import os

# 1. Path mounting logic
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"The root directory has been mounted.: {project_root}")
from src.TextureRender_B.data_logic_B import (
    CHORD_TYPES,
    PATTERN_TEMPLATES,
    parse_simplified_chord,
)

logger = logging.getLogger("B_Decision")

# Group B Phase 3: Decision and Rendering

# ----------------------------------------------------
# Velocity Mapping and Calculation
# ----------------------------------------------------

# MIDI Velocity Mapping Parameters
# The MIDI velocity range is 0-127
MAX_VELOCITY = 100 # maximum dynamics of accompaniment notes
MIN_VELOCITY = 40  # Minimum volume of accompaniment notes

# Relative dynamic weights for strong and weak beats (for 4/4 and 3/4 time signatures, based on QuarterLength offset)
# Key: Total duration (QL) of the template
# Value: { Starting QL offset of the beat: Relative dynamic multiplier }
BEAT_VELOCITY_WEIGHTS = {
    4.0: {0.0: 1.0, 1.0: 0.8, 2.0: 0.9, 3.0: 0.7}, # 4/4 beat (1st beat is the strongest, 3rd beat is the second strongest)
    3.0: {0.0: 1.0, 1.0: 0.7, 2.0: 0.8},          # 3/4 time (1 beat is the strongest)
}

def calculate_velocity(segment_offset: float, density: float, template_duration: float) -> int:
    """
Calculates the MIDI velocity value based on the note's starting position in the segment, melody density, and template duration.

Args:

segment_offset (float): The starting time (QL) of the note within the template cycle of the chord segment.

density (float): The melody density of the current segment (0.0 - 1.0).

template_duration (float): The total duration of the template (e.g., 4.0 QL or 3.0 QL).

Returns:

int: The calculated MIDI velocity value (0-127).

"""
    
    # 1. Density Inverse Adjustment
    # The higher the melody density, the lower the accompaniment intensity (1.0 - density)
    density_factor = 1.0 - density
    
    # 2. Metric Weight
    # Calculates the relative position of notes within the template cycle.
    beat_time = segment_offset % template_duration
    
    weights = BEAT_VELOCITY_WEIGHTS.get(template_duration, BEAT_VELOCITY_WEIGHTS[4.0])
    
    metric_weight = 1.0 
    for beat_start, weight in weights.items():
        if abs(beat_time - beat_start) < 0.1:
            metric_weight = weight
            break
            
    # 3. The final strength is calculated comprehensively.
    base_velocity = MAX_VELOCITY - MIN_VELOCITY
    
    # Final strength = Minimum strength + (Strength baseline range * Density counter-modulation factor * Strong and weak beat weights)
    final_velocity = MIN_VELOCITY + (base_velocity * density_factor * metric_weight)
    
    return int(max(MIN_VELOCITY, min(MAX_VELOCITY, final_velocity)))

# ----------------------------------------------------
# 1. Heuristic decision-making: texture selection (Texture Selector)
# ----------------------------------------------------
DENSITY_THRESHOLD = 0.4 # Melody density threshold: Values ​​exceeding this are considered "busy".

def select_texture_pattern(style: str, density_score: float) -> str:
    """
    Heuristic algorithm: Select accompaniment texture template based on style and melody density.
    """

    if density_score > DENSITY_THRESHOLD:
        logger.debug(
            f"Density {density_score:.2f} High, tending towards sparse texture (Sparse_Arpeggio)。"
        )
        return "Sparse_Arpeggio"

    # If the density is low (the melody is sparse), use the default texture style.
    else:
        logger.debug(f"Density {density_score:.2f} Low, using the default texture style: {style}")
        return style


# 2. Preprocessing A: Chord Fragment Integration and Density Calculation
def _consolidate_chords(chord_sequence, quantization_step=0.25):
    if not chord_sequence: return []
    
    consolidated = []
    current_chord = 'C' 
    for token in chord_sequence:
        if token != '_':
            current_chord = token
            break
            
    current_duration = 0.0
    for label in chord_sequence:
        if label == '_' or label == current_chord:
            current_duration += quantization_step
        else:
            if current_chord != '_':
                consolidated.append((current_chord, current_duration))
            current_chord = label
            current_duration = quantization_step
            
    if current_chord != '_':
        consolidated.append((current_chord, current_duration))
    return consolidated


def _calculate_melody_density(
    melody_stream: m21.stream.Stream, consolidated_chords: List[Tuple[str, float]]
) -> List[float]:
    """
    Calculate the melodic density within each chord progression.
    Density = (Number of notes in the progression) / (Progress duration QL) / (Reference tempo)
    """
    density_scores = []
    current_offset = 0.0

    notes_and_rests = melody_stream.flat.notesAndRests

    for chord_label, duration in consolidated_chords:
        segment_notes_count = 0
        segment_start = current_offset
        segment_end = current_offset + duration

        for element in notes_and_rests:
            element_start = element.offset

            if isinstance(element, m21.note.Note) or isinstance(
                element, m21.chord.Chord
            ):
                if segment_start <= element_start < segment_end - 0.001:
                    segment_notes_count += 1

        # Density calculation
        if duration > 0:
            density = math.sqrt(segment_notes_count / duration) * 0.7 
        else:
            density = 0.0

        # Ensure the density does not exceed 1.0.
        density_scores.append(min(density, 1.0))
        current_offset += duration

    return density_scores


# 3. Core Engine: Accompaniment Part Generation (Accompaniment Renderer)
def generate_accompaniment_part(
    consolidated_chords: List[Tuple[str, float]], 
    melody_densities: List[float], 
    selected_style: str
) -> m21.stream.Part:
    
    accompaniment_part = m21.stream.Part()
    accompaniment_part.insert(0, m21.instrument.Piano())
    
    current_offset = 0.0
    
    for i in range(len(consolidated_chords)):
        chord_label, segment_length = consolidated_chords[i]
        density = melody_densities[i]
        
        root_name, chord_type = parse_simplified_chord(chord_label)
        
        if chord_type == 'NoChord' or chord_type not in CHORD_TYPES:
            current_offset += segment_length
            continue
            
        offsets = CHORD_TYPES[chord_type]
        root_midi = m21.pitch.Pitch(root_name + '3').midi

        pattern_name = select_texture_pattern(selected_style, density)
        template = PATTERN_TEMPLATES.get(pattern_name, PATTERN_TEMPLATES["Pop Ballad"])
        template_duration = 3.0 if pattern_name == "Waltz" else 4.0

        time_elapsed_in_segment = 0.0
        
        # The loop continues filling the segment until its duration is reached.
        while time_elapsed_in_segment < segment_length - 0.01:
            for time_offset, index, duration in template:
                # Calculate the relative start time of this note within the current segment.
                note_start_in_segment = time_elapsed_in_segment + time_offset
                
                # If the note is outside the range of the current chord, it is not drawn (to prevent it from covering the next chord).
                if note_start_in_segment >= segment_length - 0.01:
                    break
                
                pitch_midi = (m21.pitch.Pitch(root_name + '2').midi if index == 0 
                             else root_midi) + offsets[index % len(offsets)]
                
                n = m21.note.Note(midi=pitch_midi)
                
                actual_duration = duration
                if note_start_in_segment + duration > segment_length:
                    actual_duration = segment_length - note_start_in_segment
                n.duration.quarterLength = actual_duration
                
                n.volume.velocity = calculate_velocity(note_start_in_segment, density, template_duration)

                # [Core Logic]: Insertion position = Global cumulative offset + Intra-segment relative offset
                accompaniment_part.insert(current_offset + note_start_in_segment, n)

            # Template stepping (e.g., advancing 4 beats in a 4/4 time signature)
            time_elapsed_in_segment += template_duration
            
        # 4. Once the fragment processing is complete, the global offset moves forward, preparing for the next chord.
        current_offset += segment_length
        # --- 【Repair complete】 ---

    return accompaniment_part


# 4. Main B-Group Entry: From raw input to part generation
def render_accompaniment_from_raw_inputs(
    melody_path: str, chord_sequence: List[str], selected_style: str
) -> m21.stream.Part:
    """
    The complete workflow for Group B is: parsing the input, calculating the density, selecting a template, and rendering the Part.
    This is then called by the `render_music` function in `main_pipeline.py`.
    """
    logger.info("Begin Group B preprocessing: chord integration and density calculation...")

    consolidated_chords = _consolidate_chords(chord_sequence)

    try:
        melody_stream = m21.converter.parse(melody_path)
    except Exception as e:
        logger.error(f"Unable to parse melody MIDI file: {e}")
        return m21.stream.Part()

    melody_densities = _calculate_melody_density(melody_stream, consolidated_chords)

    logger.info(
        f"Number of chord segments: {len(consolidated_chords)}, Density fraction example: {melody_densities[:3]}"
    )

    accompaniment_part = generate_accompaniment_part(
        consolidated_chords, melody_densities, selected_style
    )

    return accompaniment_part
