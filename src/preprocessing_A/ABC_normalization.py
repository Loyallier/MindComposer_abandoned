import music21
import os
import glob
from tqdm import tqdm
from collections import Counter

# ===========================
# 1. Basic configuration and parameters
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
OUTPUT_FILE = os.path.join(INTERIM_DIR, "training_data_aligned.txt")

# Special Token
BAR_TOKEN = "<BAR>"
REST_TOKEN = "0"
HOLD_TOKEN = "_"

# Normalization parameter (user-specified)
TARGET_CENTROID = 63     # Target focus (55+86)/2
RANGE_TOLERANCE = 1      # Tolerance (semitone)
HARD_MIN_PITCH = 41        # Strict lower limit (G2)
HARD_MAX_PITCH = 86        # Strict upper limit (C#7)

os.makedirs(INTERIM_DIR, exist_ok=True)

# ===========================
# 2. Core logic function
# ===========================

def get_melody_pitches(stream_obj):
    """
    Strictly extract melody pitches:
    1. If there are multiple parts, only extract Part 0 (the main melody).
    2. Extract only the highest notes of the Note and Chord (in the case of double stops).
    3. Exclude accompaniment markings (Harmony) and percussion.
    """
    # 1. Isolate Melody Part
    if stream_obj.hasPartLikeStreams():
        melody_stream = stream_obj.parts[0]
    else:
        melody_stream = stream_obj

    pitches = []
    # 2. Flattening extraction (only for melody layers)
    # `recurse()` is more robust than `flatten()`, excluding metadata.
    for el in melody_stream.recurse().notes:
        if isinstance(el, music21.note.Note):
            pitches.append(el.pitch.midi)
        elif isinstance(el, music21.chord.Chord):
            # For double notes/chords in a melody, take the highest note as the melody outline.
            pitches.append(el.pitches[-1].midi)
            
    return pitches

def analyze_and_transpose(score, song_id="Unknown"):
    try:
        # --- Step 1: Extract the original melody fingerprint ---
        raw_pitches = get_melody_pitches(score)
        
        if not raw_pitches:
            return None, "Empty Melody"

        raw_min = min(raw_pitches)
        raw_max = max(raw_pitches)
        raw_range = raw_max - raw_min 

        # --- Step 2: Key Norm Normalization ---
        # Note: analyze('key') will analyze all voices, which is correct because tonality is global.
        key = score.analyze("key")
        
        if key.mode in ['minor', 'dorian', 'phrygian', 'locrian']:
            target_tonic_name = 'A'
        else:
            target_tonic_name = 'C'

        target_key = music21.key.Key(target_tonic_name)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        
        # The whole piece modulates
        transposed_score = score.transpose(interval)

        # --- Step 3: Integrity Check ---
        # Extract the melody fingerprint again (Note: You must extract it again using get_melody_pitches)
        trans_pitches = get_melody_pitches(transposed_score)
        
        if not trans_pitches: 
            return None, "Lost Notes After Transpose"

        trans_min = min(trans_pitches)
        trans_max = max(trans_pitches)
        trans_range = trans_max - trans_min

        # Verification: Only a 1.5 semitone error is allowed.
        diff = abs(raw_range - trans_range)
        if diff > RANGE_TOLERANCE:
            return None, f"Range Integrity Fail (Diff: {diff:.2f})"

        # --- Step 4: Octave Normalization Based on Melodic Center ---
        # Use the melody's median, not the median of the entire score.
        current_centroid = (trans_max + trans_min) / 2.0
        
        # Calculate the difference between the distance to the target (70.5).
        dist_to_target = TARGET_CENTROID - current_centroid
        
        # Round to the nearest octave.
        octaves_to_shift = round(dist_to_target / 12.0)
        shift_amount = int(octaves_to_shift * 12)

        if shift_amount != 0:
            final_score = transposed_score.transpose(shift_amount)
        else:
            final_score = transposed_score

        # --- Step 5: Final Boundary Check ---
        final_pitches = get_melody_pitches(final_score)
        f_min = min(final_pitches)
        f_max = max(final_pitches)
        
        # Strict boundaries: 44-97
        if f_min < HARD_MIN_PITCH or f_max > HARD_MAX_PITCH:
            return None, f"Bounds Violation ({f_min}-{f_max})"

        return final_score, "PASS"

    except Exception as e:
        return None, f"Music21 Exception: {str(e)}"

def sample_stream(score, step_size=0.25):
    """
    [V5.0 Revised Version] Dual-track sampling: Melody + Chord
    Fixed: Returns two lists to match m, c = sample_stream(...)
    """
    melody_tokens = []
    chord_tokens = []
    
    # 1. Structural cleaning
    try:
        # Eliminate the Part/Voice hierarchy and force exposure of Measure.
        parts = music21.instrument.partitionByInstrument(score)
        if parts:
            # If there are multiple instruments, the first one is usually chosen as the main melody and its chords.
            score_to_process = parts.parts[0]
        else:
            score_to_process = score
            
        score_to_process = score_to_process.makeMeasures()
        measures = list(score_to_process.recurse().getElementsByClass(music21.stream.Measure))
    except Exception as e:
        # The structure is extremely chaotic, making it impossible to extract subsections.
        return [], []

    if not measures:
        return [], []

    # Record the chord from the previous moment to fill in the underscores.
    last_chord = "N.C." 

    for m in measures:
        m_len = m.duration.quarterLength
        steps = int(round(m_len / step_size))
        if steps <= 0: continue
        
        # === Key: Retrieve Notes and ChordSymbols within a section ===
        # Use `flat` to retrieve all elements within the section, sorted by offset.
        m_flat = m.flat
        
        # Prefetch all chord symbols for this section
        # Chords in ABC files are typically Harmony objects
        chord_objs = list(m_flat.getElementsByClass(music21.harmony.ChordSymbol))
        
        for i in range(steps):
            offset = i * step_size
            current_abs_offset = m.offset + offset # Absolute Time (Backup)
            
            # --- Track 1: Melody ---
            melody_token = "0" # Default: Rest
            
            # Get the note at the current time point
            elements = m_flat.getElementsByOffset(offset, mustBeginInSpan=False)
            
            # A. Attack Detection
            attack_found = False
            for el in elements:
                if abs(el.offset - offset) < 0.01:
                    if isinstance(el, music21.note.Note):
                        melody_token = str(el.pitch.midi)
                        attack_found = True
                        break
                    elif isinstance(el, music21.chord.Chord):
                        # Take the highest note
                        melody_token = str(el.sortAscending().notes[-1].pitch.midi)
                        attack_found = True
                        break
            
            # B. Sustain Detection
            if not attack_found:
                for el in elements:
                    if isinstance(el, (music21.note.Note, music21.chord.Chord)):
                        # If the current point is within the duration of the note (start < now < end)
                        if el.offset < offset < (el.offset + el.duration.quarterLength):
                            melody_token = "_"
                            break
            
            melody_tokens.append(melody_token)

            # --- Track 2: Chord ---
            # Logic: Find the current offset and the nearest chord symbol before it.
            # ABC chords are usually only marked when they change, so it's necessary to keep the state consistent.
            
            found_new_chord = False
            # Is a new chord beginning within this very short time window?
            for ch in chord_objs:
                if abs(ch.offset - offset) < 0.01:
                    last_chord = ch.figure
                    found_new_chord = True
                    break
            
            if found_new_chord:
                chord_tokens.append(last_chord)
            else:
                chord_tokens.append("_")

        # --- End of Measure ---
        melody_tokens.append("<BAR>")
        chord_tokens.append("<BAR>")
        
    return melody_tokens, chord_tokens

def process_all_files():
    files = glob.glob(os.path.join(RAW_DIR, "*.abc"))
    print(f"Start processing {len(files)} files...")
    print(f"   - Target Centroid: MIDI {TARGET_CENTROID}")
    print(f"   - Hard Bounds: {HARD_MIN_PITCH} ~ {HARD_MAX_PITCH}")

    stats = {
        "processed": 0,
        "valid": 0,
        "dropped": 0
    }
    drop_reasons = Counter()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for file_path in tqdm(files, desc="Processing"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                stream_obj = music21.converter.parse(content, format="abc")
                
                # Processing single tracks vs. multiple tracks
                songs = []
                if isinstance(stream_obj, music21.stream.Opus):
                    songs = list(stream_obj)
                else:
                    songs = [stream_obj]

                for i, song in enumerate(songs):
                    stats["processed"] += 1
                    
                    # 1. Analysis and Transposition
                    final_score, message = analyze_and_transpose(song)

                    if final_score is None:
                        stats["dropped"] += 1
                        # Record error reason keywords
                        if "Bounds" in message: reason = "Bounds Violation"
                        elif "Range" in message: reason = "Range Integrity Fail"
                        elif "Empty" in message: reason = "Empty Score"
                        else: reason = "Other Error"
                        drop_reasons[reason] += 1
                        continue

                    # 2. Sample serialization
                    try:
                        m, c = sample_stream(final_score)
                        
                        if len(m) < 10:
                            stats["dropped"] += 1
                            drop_reasons["Too Short"] += 1
                            continue

                        line = f"{' '.join(m)} | {' '.join(c)}"
                        f_out.write(line + "\n")
                        stats["valid"] += 1
                    except Exception:
                        stats["dropped"] += 1
                        drop_reasons["Sampling Error"] += 1

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

    # === Final statistics ===
    print("\n" + "="*40)
    print(f"Final processing report")
    print("="*40)
    print(f"Total processed tracks: {stats['processed']}")
    print(f"Successfully output:   {stats['valid']} ({(stats['valid']/max(1, stats['processed']))*100:.1f}%)")
    print(f"Number of items to be removed:   {stats['dropped']}")
    print("-" * 40)
    print("Distribution of reasons for exclusion:")
    for reason, count in drop_reasons.most_common():
        print(f"   - {reason}: {count}")
    print("="*40)
    print(f"The results have been saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_files()