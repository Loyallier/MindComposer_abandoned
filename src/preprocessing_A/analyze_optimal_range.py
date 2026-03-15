import os

# ================= Configuration =================
DATA_FILE = (
    r"E:\VScode_Programs\！Projects\AI_AS1_G\data\interim\training_data_aligned.txt"
)

# Objective: Find a window of length N that can cover the most musical notes.
WINDOW_SIZES = [36, 40, 48]


def load_data(filepath):
    """Read all melodies Token"""
    all_melodies = []
    if not os.path.exists(filepath):
        print(f"The file does not exist.: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loading {len(lines)} Data entries...")
    for line in lines:
        if "|" not in line:
            continue
        melody_part = line.split("|")[0].strip()
        tokens = melody_part.split()
        pitches = [int(t) for t in tokens if t.isdigit() and t != "0"]
        if pitches:
            all_melodies.append(pitches)

    return all_melodies


def simulate_best_octave(melodies, target_center=66):
    """
    Simulation: If each song is shifted to the octave closest to target_center,
    what will happen to the global pitch range?
    """
    shifted_pitches = []
    shifts_applied = []

    for song in melodies:
        if not song:
            continue
        avg_pitch = sum(song) / len(song)

        # Calculate how many octaves of shift are needed to approach target_center (F4/F#4)
        diff = target_center - avg_pitch
        shift_octaves = round(diff / 12)
        shift_semitones = int(shift_octaves * 12)

        # Apply translation
        new_song = [p + shift_semitones for p in song]
        shifted_pitches.extend(new_song)
        shifts_applied.append(shift_octaves)

    return shifted_pitches, shifts_applied


def analyze():
    melodies = load_data(DATA_FILE)
    if not melodies:
        return

    # 1. Original distribution statistics
    raw_pitches = [p for song in melodies for p in song]
    raw_min, raw_max = min(raw_pitches), max(raw_pitches)
    raw_unique = len(set(raw_pitches))

    print("\n --- Current status of raw data ---")
    print(f"   vocal range: {raw_min} ~ {raw_max} (span {raw_max - raw_min})")
    print(f"   Vocabulary size: {raw_unique} different pitches")
    print(
        f"   LowNote (<52) percentage: {sum(1 for p in raw_pitches if p < 52) / len(raw_pitches):.2%}"
    )

    # 2. Simulation optimal convergence
    opt_pitches, shifts = simulate_best_octave(melodies, target_center=66)
    opt_min, opt_max = min(opt_pitches), max(opt_pitches)
    opt_unique = len(set(opt_pitches))

    print("\n --- Simulated optimal octave convergence (Target Center = 66) ---")
    print(f"   Optimized range: {opt_min} ~ {opt_max} (span {opt_max - opt_min})")
    print(f"   Optimized vocabulary: {opt_unique} different pitches")

    # 3. Coverage calculation
    sorted_p = sorted(opt_pitches)
    cut_idx = int(len(sorted_p) * 0.01)  # 1%
    core_min = sorted_p[cut_idx]
    core_max = sorted_p[-cut_idx]

    print(f"   Core area (98% coverage): {core_min} ~ {core_max}")
    print(f"   Recommended LowNote threshold: {core_min}")
    print(f"   Recommended HighNote threshold: {core_max}")

    # 4. Recommended strategy
    print("\n💡 --- Experts recommend (AI_G) ---")
    print(
        f"   1. Modify normalization.py and adjust TARGET_CENTROID to 66 or 67 (F4/G4)."
    )
    print(
        f"   2. 70.5 (Bb4) It was clearly too high, forcing the bass frequencies to shift upwards, turning them into tenor frequencies."
    )
    print(
        f"   3. In clean_data.py, set LowNote to {core_min} and HighNote to {core_max}."
    )


if __name__ == "__main__":
    analyze()
