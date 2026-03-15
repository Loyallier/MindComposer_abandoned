# import os
# import glob
# import math
# from music21 import converter, note, chord, harmony, stream

"""
data_format_converter.py Final version: ABC to time series data conversion script.
Includes path correction, robustness optimization, dynamic quantization, chord filtering (keeping only Major, Minor, and Seventh chords), and concise output.
"""

import os
import glob
import math
import traceback
from typing import List, Dict, Any, Tuple, Optional
from music21 import converter, note, chord, harmony, stream, expressions

# =======================================================
# I. Helper functions and tools
# =======================================================


def find_project_root() -> str:
    """The project root directory is determined dynamically, assuming the script is located at... src/ 下"""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.dirname(script_dir)


def safe_expand_repeats(s: stream.Score) -> stream.Score:
    """Try expandingRepeats; if that fails, revert to the original. score"""
    try:
        return s.expandRepeats()
    except Exception:
        return s


def is_int_str(s: str) -> bool:
    """Determine if a string can be parsed into an integer. MIDI"""
    try:
        int(s)
        return True
    except Exception:
        return False


def parse_text_from_textexpression(te) -> str:
    """Retrieve text from TextExpression (compatible with different versions of music21)"""
    try:
        if hasattr(te, "content") and te.content:
            return str(te.content).strip()
        if hasattr(te, "text") and te.text:
            return str(te.text).strip()
        return str(te).strip()
    except Exception:
        return str(te).strip()


def safe_chord_name(obj) -> Optional[str]:
    """
    Try to extract simplified chord names from multiple objects and perform strict filtering:
    Keep only Major, Minor, and Seventh chords.
    """
    chord_name = None

    try:
        if isinstance(obj, harmony.ChordSymbol):
            name_only = str(obj.figure).split("/")[0]
            quality = obj.quality

            # --- Core chord selection logic ---
            if quality in ("major", "minor"):
                chord_name = name_only
            elif "seventh" in quality:
                chord_name = name_only

        elif isinstance(obj, expressions.TextExpression):
            txt = parse_text_from_textexpression(obj)
            chord_name = txt.strip().strip('"').split("/")[0]

        elif isinstance(obj, chord.Chord):
            try:
                root = obj.root()
                if root:
                    chord_name = root.name
            except Exception:
                pass

    except Exception:
        return None

    if chord_name:
        lower_name = chord_name.lower()
        if "dim" in lower_name or "aug" in lower_name or "sus" in lower_name:
            return None

        return chord_name

    return None


def detect_min_step(s_flat: stream.Score, default_step: float = 0.25) -> float:
    """
    Automatically detect the smallest tick unit for dynamic quantization of step_size.
    """
    min_d = default_step * 2

    try:
        for n in s_flat.recurse().notesAndRests:
            d = n.duration.quarterLength
            if d is None or d <= 1e-9:
                continue
            if d < min_d:
                min_d = d

        potential_step = min_d / 4.0

        if potential_step < 0.015625:
            return 0.015625

        if 0.0625 <= potential_step < 0.125:
            return 0.0625
        if 0.125 <= potential_step < 0.25:
            return 0.125

        return default_step

    except Exception:
        return default_step


# =======================================================
# II. Core parsing logic
# =======================================================


def parse_abc_to_dataset(
    file_path: str, default_step: float = 0.25
) -> List[Dict[str, List[str]]]:
    """
    Parse all the music in a single ABC file, and quantize and serialize it.
    """
    dataset = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(100)
            if "X:" not in header and "T:" not in header:
                return []
    except Exception as e:
        print(f" File read error {os.path.basename(file_path)}: {e}")
        return []

    try:
        opus = converter.parse(file_path, format="abc", forceSource=True)
    except Exception as e:
        print(f" Overall parsing failed {os.path.basename(file_path)}: {e}")
        return []

    scores: List[stream.Score]
    if isinstance(opus, stream.Score):
        scores = [opus]
    else:
        scores = list(opus)

    print(f" Processing: {os.path.basename(file_path)} (including {len(scores)} head)")

    # --- 4. Core processing loop (single track exception capture, hiding internal error printing) ---
    for s in scores:
        try:
            s_expanded = safe_expand_repeats(s)
            s_flat = s_expanded.flatten()

            if s_flat.duration.quarterLength < 4.0:
                continue

            step_size = detect_min_step(s_flat, default_step)
            total_steps = int(math.ceil(s_flat.duration.quarterLength / step_size))

            melody_seq = ["0"] * total_steps
            chord_seq = ["_"] * total_steps

            # --- Melody processing ---
            notes_and_rests = s_flat.notesAndRests
            for n in notes_and_rests:
                start_idx = int(round(n.offset / step_size))

                end_idx = int(
                    math.floor((n.offset + n.duration.quarterLength) / step_size + 1e-9)
                )

                if start_idx >= total_steps:
                    continue
                end_idx = min(end_idx, total_steps)

                if end_idx <= start_idx:
                    continue

                if isinstance(n, note.Note):
                    melody_seq[start_idx] = str(n.pitch.midi)
                    for i in range(start_idx + 1, end_idx):
                        melody_seq[i] = "_"
                elif isinstance(n, note.Rest):
                    for i in range(start_idx, end_idx):
                        melody_seq[i] = "0"

            # --- Chord processing ---
            chord_elements = s_flat.getElementsByClass(
                (harmony.ChordSymbol, expressions.TextExpression, chord.Chord)
            )

            for c_obj in chord_elements:
                start_idx = int(round(c_obj.offset / step_size))
                if start_idx >= total_steps:
                    continue

                chord_name = safe_chord_name(c_obj)

                if chord_name:
                    chord_seq[start_idx] = chord_name

            # 5. Data quality check: Filter out sequences whose melodies consist entirely of rests.
            has_valid_note = any(is_int_str(t) and int(t) > 0 for t in melody_seq)

            if has_valid_note:
                dataset.append({"melody": melody_seq, "harmony": chord_seq})

        except Exception as e:
            continue

    return dataset


# =======================================================
# III. Main execution block (path management and execution)
# =======================================================

if __name__ == "__main__":
    PROJECT_ROOT = find_project_root()

    # --- 1. Path configuration---
    data_folder = os.path.join(PROJECT_ROOT, "data", "raw")
    output_dir = os.path.join(PROJECT_ROOT, "data", "interim")
    output_filename = "training_data_aligned.txt"
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_folder):
        print(
            f" Error: Raw data folder '{data_folder}' not found. Please confirm that your raw ABC files are located in data/raw/."
        )
        exit()

    # --- 2. File search logic ---
    search_path_txt = os.path.join(data_folder, "*.txt")
    search_path_abc = os.path.join(data_folder, "*.abc")

    input_files = glob.glob(search_path_txt) + glob.glob(search_path_abc)

    print(f" {len(input_files)} files were found in '{data_folder}'.")

    all_data = []

    # 3. Iterative processing of all files
    for f in input_files:
        data = parse_abc_to_dataset(f, default_step=0.25)
        all_data.extend(data)

    print(f"\n Processing complete! A total of {len(all_data)} tracks of data were extracted.")

    # 4. Write to output file
    if len(all_data) > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            for song in all_data:
                if isinstance(song, dict) and "melody" in song and "harmony" in song:
                    x_str = " ".join(song["melody"])
                    y_str = " ".join(song["harmony"])
                    f.write(f"{x_str}|{y_str}\n")
        print(f" Saved to {output_path}")
    else:
        print(" No data was retrieved. Please check the error message above.")
