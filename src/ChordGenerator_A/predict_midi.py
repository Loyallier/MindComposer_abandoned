import os
import sys
import music21
import math
import json  # Need to read vocab

# Path mounting
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A import utils
from src.ChordGenerator_A.inference import AIComposer

# Import preprocessing logic
try:
    from src.preprocessing_A.ABC_normalization import sample_stream
except ImportError:
    raise ImportError("❌ [Fatal] Unable to import src/preprocessing_A/ABC_normalization.py")


class MidiPreprocessor:
    """
    Responsibilities: MIDI parsing, key analysis, smart octave matching, transposition
    """

    def __init__(self):
        # Load vocabulary for smart coverage calculation
        self.vocab = utils.load_vocab(path.VOCAB_PATH)
        # Extract all valid digit Keys from the melody vocabulary (excluding special symbols)
        self.valid_vocab_pitches = set()
        for k in self.vocab["melody"].keys():
            if k.isdigit():
                self.valid_vocab_pitches.add(int(k))

        if not self.valid_vocab_pitches:
            raise ValueError("❌ Vocabulary anomaly: No valid melody Pitch Keys found")

    def get_melody_stream(self, score):
        # Prioritize finding if a Part is explicitly marked as Melody
        if score.hasPartLikeStreams():
            # Simple strategy: take the Part with the most notes, or the first Part
            # Optimization: Usually the treble is the melody; taking the Part with a higher average pitch might be more accurate, but for now, we take Part 0
            return score.parts[0]
        return score

    def _calculate_vocab_coverage(self, pitches):
        """Calculate the coverage rate of the note list within the vocabulary"""
        if not pitches:
            return 0.0
        in_vocab_count = sum(1 for p in pitches if p in self.valid_vocab_pitches)
        return in_vocab_count / len(pitches)

    def _smart_octave_shift(self, stream):
        """
        Algorithm: Try -2, -1, 0, +1, +2 octaves, selecting the offset with the highest vocabulary coverage and closest distance to the center
        """
        # 1. Extract all Pitches
        raw_pitches = [
            p.midi
            for n in stream.recurse().notes
            for p in (n.pitches if n.isChord else [n.pitch])
        ]

        if not raw_pitches:
            raise ValueError("❌ No valid notes detected in MIDI track")

        best_shift = 0
        max_coverage = -1.0

        # Search range: -24 to +24 (two octaves)
        candidates = [-24, -12, 0, 12, 24]

        for shift in candidates:
            shifted_pitches = [p + shift for p in raw_pitches]
            coverage = self._calculate_vocab_coverage(shifted_pitches)

            # Logic: Prioritize high coverage; if coverage is the same, choose the smaller absolute shift value (minimum change)
            if coverage > max_coverage:
                max_coverage = coverage
                best_shift = shift
            elif coverage == max_coverage:
                if abs(shift) < abs(best_shift):
                    best_shift = shift

        if best_shift != 0:
            print(
                f"    Smart Octave Correction: Offset {best_shift} semitones (Coverage: {max_coverage:.2%})"
            )
            return stream.transpose(best_shift), best_shift

        return stream, 0

    def process(self, midi_path):
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"❌ MIDI file not found: {midi_path}")

        try:
            score = music21.converter.parse(midi_path)
        except Exception as e:
            raise ValueError(f"Music21 parsing failed: {e}")

        original_stream = self.get_melody_stream(score)

        # --- 1. Key Analysis and Smart Transpose ---
        try:
            key = original_stream.analyze("key")
        except:
            # Very short sequences might fail key analysis, default to C Major
            key = music21.key.Key("C")

        if key.mode in ["minor", "dorian", "phrygian", "locrian"]:
            target_tonic = "A"
        else:
            target_tonic = "C"

        target_key = music21.key.Key(target_tonic)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        transposed_stream = original_stream.transpose(interval)

        # Record the offset of the first step of transposition
        recover_semitones_key = -interval.semitones

        # --- 2. Smart Octave Normalization (Vocab Coverage Maximization) ---
        final_stream, octave_shift_semitones = self._smart_octave_shift(
            transposed_stream
        )

        # Total restoration amount = -(Transposition offset + Octave offset)
        total_recover_semitones = recover_semitones_key - octave_shift_semitones

        # --- 3. Sampling and Tokenization ---
        melody_tokens_flat, _ = sample_stream(final_stream)

        if not melody_tokens_flat:
            raise ValueError("No valid Tokens obtained after preprocessing (file might be too short or empty)")

        # Reorganize into measure structure
        measure_tokens = []
        current_bar = []
        for token in melody_tokens_flat:
            if token == config.BAR_TOKEN:
                if current_bar:
                    measure_tokens.append(current_bar)
                current_bar = []
            else:
                current_bar.append(token)

        # [Robustness] Handle remaining tail or single measure cases
        if current_bar:
            measure_tokens.append(current_bar)

        # [Robustness] Protection for extremely short sequences: If no BAR token exists, measure_tokens might be empty after the loop
        # However, sample_stream should have handled basic tokens. If flat only has notes and no bar, it will be captured by current_bar.
        # Final non-empty check here
        if not measure_tokens:
            raise ValueError("Unable to split measures, data structure anomaly")

        return measure_tokens, total_recover_semitones


class ChordPredictor:
    def __init__(self):
        self.composer = AIComposer()
        self.preprocessor = MidiPreprocessor()
        self.window_size = 4
        self.BAR_POS_IDX = config.POS_VOCAB_SIZE - 2
        
    def _prepare_window_input(self, m_tokens, m_pos):
        flat_input = []
        flat_pos = []
        for bar_toks, bar_ps in zip(m_tokens, m_pos):
            flat_input.extend(bar_toks + [config.BAR_TOKEN])
            flat_pos.extend(bar_ps + [self.BAR_POS_IDX])
        return flat_input, flat_pos

    def restore_chord(self, chord_token, semitones):
        if chord_token in [
            config.PAD_TOKEN,
            config.SOS_TOKEN,
            config.EOS_TOKEN,
            config.UNK_TOKEN,
            "_",
            "0",
            config.BAR_TOKEN,
            "N.C.",
        ]:
            return chord_token

        try:
            import re

            root_match = re.match(r"^([A-G][b#-]?)", chord_token)
            if not root_match:
                return chord_token

            root_str = root_match.group(1)
            suffix = chord_token[len(root_str) :]

            n = music21.note.Note(root_str)
            # semitones might be float, force cast to int
            transposed_n = n.transpose(int(semitones))
            new_root = transposed_n.name.replace("-", "b")

            return new_root + suffix
        except:
            return chord_token

    def run(self, midi_path):
        # At the beginning of run()
        print(f"[Config] Window Size: {self.window_size} | Device: {self.composer.device}")
        
        # 1. Preprocessing
        try:
            measure_structs, recover_semitones = self.preprocessor.process(midi_path)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return []

        num_measures = len(measure_structs)

        # 2. Position Calculation (Smart Position)
        # utils needs to be able to handle extremely short sequences (<1 measure)
        measure_positions = utils.generate_smart_position_indices(measure_structs)

        final_chords_per_measure = [None] * num_measures

        print(
            f"[Predictor] Processing: {os.path.basename(midi_path)} | Measures: {num_measures}"
        )

        # === 3. Dynamic Sliding Window Logic (Robust for Short Sequences) ===

        # If total length is less than window size, predict as a single batch directly
        if num_measures <= self.window_size:
            input_seq, pos_seq = self._prepare_window_input(
                measure_structs, measure_positions
            )
            raw_output, _ = self.composer.predict(input_seq, pos_seq)

            # EOS Processing
            if config.EOS_TOKEN in raw_output:
                eos_idx = raw_output.index(config.EOS_TOKEN)
                raw_output[eos_idx:] = ["0"] * (len(raw_output) - eos_idx)

            # Assign back to measures (Robust Slicing)
            ptr = 0
            for i in range(num_measures):
                # Theoretical length = number of notes + 1 (BAR)
                length = len(measure_structs[i]) + 1
                seg = raw_output[ptr : ptr + length]
                # Pad to prevent out-of-bounds
                if len(seg) < length:
                    seg = seg + ["0"] * (length - len(seg))
                final_chords_per_measure[i] = seg
                ptr += length

        else:
            # Standard sliding window (Keep original logic)
            step = 1
            for i in range(0, num_measures, step):
                end_idx = min(i + self.window_size, num_measures)
                # Only meaningful if remaining length is sufficient, or at the final segment
                current_window_measures = measure_structs[i:end_idx]
                current_window_positions = measure_positions[i:end_idx]

                input_seq, pos_seq = self._prepare_window_input(
                    current_window_measures, current_window_positions
                )
                raw_output, _ = self.composer.predict(input_seq, pos_seq)

                if config.EOS_TOKEN in raw_output:
                    eos_idx = raw_output.index(config.EOS_TOKEN)
                    raw_output[eos_idx:] = ["0"] * (len(raw_output) - eos_idx)

                ptr = 0
                for local_idx, m_tokens in enumerate(current_window_measures):
                    global_idx = i + local_idx
                    length = len(m_tokens) + 1
                    seg = raw_output[ptr : ptr + length]
                    if len(seg) < length:
                        seg = seg + ["0"] * (length - len(seg))

                    if final_chords_per_measure[global_idx] is None:
                        final_chords_per_measure[global_idx] = seg
                    ptr += length

        # === 4. Concatenation and Restoration ===
        final_result_flat = []
        for chords in final_chords_per_measure:
            if chords is None:
                # Fallback: if a measure was not predicted (extremely rare), fill empty
                final_result_flat.append("N.C.")
                continue

            for token in chords:
                if token in [config.BAR_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN]:
                    continue
                final_result_flat.append(self.restore_chord(token, recover_semitones))

        return final_result_flat


if __name__ == "__main__":
    # Simple self-test
    p = ChordPredictor()
    # Simulate an extremely short file path; actual run requires file to exist
    print("Predictor initialized. Ready for robustness test.")