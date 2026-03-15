import music21
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def get_stats(score, name):
    """Extract statistical features from a stream: average pitch, lowest pitch, highest pitch"""
    pitches = [p.midi for p in score.flatten().pitches if isinstance(p, music21.pitch.Pitch)]
    if not pitches:
        return None
    return {
        "mean": np.mean(pitches),
        "min": np.min(pitches),
        "max": np.max(pitches),
        "range": np.max(pitches) - np.min(pitches)
    }

def run_diagnosis():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    
    report_data = []

    print("🕵️ Normalization logic diagnostics in progress...")
    
    for file_path in tqdm(files, desc="Scanning"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            s_orig = music21.converter.parse(content, format="abc")
            
            score = s_orig[0] if isinstance(s_orig, music21.stream.Opus) else s_orig

            # --- Phase 0: Raw Data ---
            stats_orig = get_stats(score, "Original")
            if not stats_orig: continue

            # --- Phase 1: Tonal Normalization Only (Key Norm) ---
            key = score.analyze("key")
            mode = key.mode
            if mode == "minor":
                target = music21.key.Key("a", "minor")
            else:
                target = music21.key.Key("C", "major")
            interval = music21.interval.Interval(key.tonic, target.tonic)
            s_key = score.transpose(interval)
            
            stats_key = get_stats(s_key, "KeyNorm")

            # --- Phase 2: Your Elevator Algorithm (Octave Norm) ---
            melody_pitches = [p.midi for p in s_key.flatten().pitches]
            current_min = min(melody_pitches)
            TARGET_FLOOR = 55
            shift_amount = 0
            if current_min < TARGET_FLOOR:
                diff = TARGET_FLOOR - current_min
                octaves = np.ceil(diff / 12.0)
                shift_amount = int(octaves * 12)
            
            s_octave = s_key.transpose(shift_amount)
            stats_final = get_stats(s_octave, "Final")

            report_data.append({
                "file": filename,
                "orig_key": f"{key.tonic.name} {key.mode}",
                "orig_mean": stats_orig['mean'],
                "key_mean": stats_key['mean'],
                "final_mean": stats_final['mean'],
                "final_min": stats_final['min'],
                "final_max": stats_final['max'],
                "shifted": shift_amount
            })

        except Exception as e:
            continue

    df = pd.DataFrame(report_data)
    
    print("\n Diagnostic Report Summary:")
    print("-" * 30)
    print(f"Total number of files: {len(df)}")
    print(f"Standard deviation of final mean pitch: {df['final_mean'].std():.2f} (The smaller the better)")
    print(f"The range of the lowest note (Min): {df['final_min'].min()} - {df['final_min'].max()}")
    
    # Anomalies detected: The final average pitch deviates too far from 60 (Central C).
    outliers = df[(df['final_mean'] < 53) | (df['final_mean'] > 77)]
    if not outliers.empty:
        print(f"\n {len(outliers)} severely anomalous samples were found. (Mean < 53 或 > 77):")
        print(outliers[['file', 'orig_key', 'final_mean', 'final_min', 'shifted']].head())
    else:
        print("\n No extreme anomalies were found in the mean pitch shift.")

if __name__ == "__main__":
    run_diagnosis()