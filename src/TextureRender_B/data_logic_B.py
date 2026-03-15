from typing import List, Dict, Tuple, Any
import music21 as m21

# Group B phase 1: logic)
# Includes: chord structure, simplified chord analysis, and accompaniment texture templates.

# A. Chord Type Definitions
# Key name: Simplified chord type (e.g., Maj, min7, Dom7)
# Value: Semitone pitch offset relative to the root note (0) (MIDI Offsets)
CHORD_TYPES: Dict[str, List[int]] = {
    "Maj": [0, 4, 7],       # Major triad (Root, M3, P5)
    "min": [0, 3, 7],       # Small triad (Root, m3, P5)
    "Dom7": [0, 4, 7, 10],  # Dominant seventh chord (Root, M3, P5, m7)
    "Maj7": [0, 4, 7, 11],  # Major seventh chord (Root, M3, P5, M7)
    "min7": [0, 3, 7, 10],  # Seventh Chord (Root, m3, P5, m7)
    "dim": [0, 3, 6],       # Diminished triads (Root, m3, d5)
    "sus4": [0, 5, 7],      # Hanging fourth chord (Root, P4, P5)
    "Maj6": [0, 4, 7, 9],   # Major sixth chord (Root, M3, P5, M6)
    "min6": [0, 3, 7, 9],   # Small Six Harmonics (Root, m3, P5, M6)
    "NoChord": [],          # [New Feature] The type corresponding to the rest symbol '0' does not contain any pitch shift.
}

# B. Chord Parsing Function
def parse_simplified_chord(chord_label: str):
    if not chord_label or chord_label in ['0', '_', 'None', 'N.C.']:
        return 'C', 'NoChord'
    
    # 1. Preprocess the possible formats of group A (e.g., C:maj -> C).
    clean_label = str(chord_label).split(':')[0].split('/')[0]
    
    # 2. Extract the root note (processing in C# or Bb)
    if len(clean_label) >= 2 and clean_label[1] in ('#', 'b', '-'):
        root_name = clean_label[:2].replace('b', '-')
        suffix = clean_label[2:]
    else:
        root_name = clean_label[0]
        suffix = clean_label[1:]

    # 3. Enhanced suffix recognition (compatible with Am, C, G7)
    s = suffix.lower()
    if s in ['', 'maj', 'major']: chord_type = 'Maj'
    elif any(x in s for x in ['m', 'min']): chord_type = 'min'
    elif '7' in s: chord_type = 'Dom7'
    else: chord_type = 'Maj'
    
    return root_name, chord_type

# C. Accompaniment Pattern Templates
# Structure: (Time Offset QL, Chord Tone Index, Duration QL)
# Chord Tone Index: 0 = Root, 1 = Third, 2 = Fifth, 3 = Seventh...
PATTERN_TEMPLATES: Dict[str, List[Tuple[float, int, float]]] = {
    # Style 1: Pop Ballad - Root note + arpeggio, 4/4 time signature
    "Pop Ballad": [
        (0.0, 0, 1.0), # 1 Beat: Root note (Bass)
        (1.0, 1, 0.5), # 2 Begin with the beat: three notes
        (1.5, 2, 0.5), # 2 Half a beat: Five notes
        (2.0, 3, 0.5), # 3 Begin with the seventh note (or repeat the root note).
        (2.5, 2, 0.5), # 3 Half a beat: Five notes
        (3.0, 1, 1.0), # 4 Beat: Three notes
    ],
    
    # Style 2: Waltz - Root note + chord tone, 3/4 time signature
    "Waltz": [
        (0.0, 0, 1.0), # 1 beat: Root note (Bass)
        (1.0, 1, 1.0), # 2 Beat: Third note (usually a chord tone, which can be repeated)
        (2.0, 2, 1.0), # 3 beats: pentatonic scale (usually chord tones)
    ],
    
    # Style 3: Sparse Arpeggio - The default option for high-density arpeggios, 4/4 time signature
    # Longer arpeggio intervals and fewer notes to avoid conflict with the melody.
    "Sparse_Arpeggio": [
        (0.0, 0, 1.0), # 1 beat: root note
        (2.0, 1, 0.5), # 3 beats: three notes
        (2.5, 2, 0.5), # 3.5 beats: Five notes
        (3.5, 3, 0.5), # 4.5 beats: Seven notes
    ],
    
    "March": [
        (0.0, 0, 0.5),  # 1 beat: Root note (strong beat)
        (0.5, 1, 0.5), # Weak beat: Chord tone
        (1.0, 2, 0.5), # 2 beat: Fifth (Strong beat)
        (1.5, 1, 0.5), # Weak beat: Chord tone
        (2.0, 0, 0.5), # 3 beat: Root (Strong beat)
        (2.5, 1, 0.5), # Weak beat: Chord tone
        (3.0, 2, 0.5), # 4 beat: Fifth (Strong beat)
        (3.5, 1, 0.5), # Weak beat: Chord tone
    ],
    
    "Jazz Swing": [
        (0.0, 0, 0.75), #1 beat: Root note (long)
        (1.0, 1, 0.25), #2 beat: Third note (short) - Swing rhythm
        (1.5, 2, 0.75), #2.5 beat: Fifth note (long)
        (2.5, 3, 0.25), #3.5 beat: Seventh note (short)
        (3.0, 2, 1.0), #4 beat: Fifth note (long)
    ]
}