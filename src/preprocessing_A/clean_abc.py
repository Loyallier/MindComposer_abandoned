import music21
import os
import glob
import re
from tqdm import tqdm

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

TARGET_FILES = ["reelsh-l.txt", "reelsd-g.txt"]


def split_abc_content(content):
    lines = content.splitlines()
    songs = []
    current_song_lines = []

    for line in lines:
        # Remove any potential marker interference (if it's on a separate line).
        if line.strip().startswith("[source"):
            clean_line = re.sub(r"\\", "", line).strip()
            if not clean_line:
                continue
            line = clean_line

        # Detect X: Mark as the start of a new track.
        if re.match(r"^X:\s*\d+", line.strip()):
            if current_song_lines:
                songs.append("\n".join(current_song_lines))
            current_song_lines = [line]
        else:
            current_song_lines.append(line)

    if current_song_lines:
        songs.append("\n".join(current_song_lines))

    return songs


def clean_and_save():
    print("Surgical cleaning begins...")
    print(f"Target directory: {RAW_DIR}")

    total_removed = 0
    total_kept = 0

    files_to_process = [os.path.join(RAW_DIR, f) for f in TARGET_FILES]

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"file not found: {file_path}")
            continue

        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}...")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 1. Disassembled tracks
        raw_songs = split_abc_content(content)
        valid_songs = []
        file_removed_count = 0

        # 2. Verify one by one
        for song_str in tqdm(raw_songs, desc=f"Scanning {filename}"):
            if not song_str.strip():
                continue

            try:
                # === Core logic: Attempt to analyze ===
                # If this step reports an error, it means the song has a fatal grammatical error.
                # We don't need to know where the error is; we just need to know it's "broken."
                s = music21.converter.parse(song_str, format="abc")

                # An extra double check: for unclosed square brackets (specifically, for the error you mentioned).
                # While parse usually catches them, explicit checks are safer.
                if song_str.count("[") != song_str.count("]"):
                    raise ValueError("Unbalanced brackets mismatch")

                valid_songs.append(song_str)

            except Exception as e:
                # print(f"   ❌ Drop song X:{get_x_number(song_str)} - Reason: {str(e)[:50]}...")
                file_removed_count += 1

        # 3. Overwrite the file (save the cleaned version).
        if valid_songs:
            new_content = "\n\n".join(valid_songs)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(
                f"   {filename} Processing complete: {len(valid_songs)} songs retained, {file_removed_count} songs removed (invalid tracks)."
            )
            total_kept += len(valid_songs)
            total_removed += file_removed_count
        else:
            print(f"   {filename} Empty after processing! No data written.")

    print("\n" + "=" * 40)
    print(f"Cleaning Summary:")
    print(f"   - Total repertoire: {total_kept}")
    print(f"   - Total removal of bad curves: {total_removed}")
    print("   - The original file has been updated in place to the cleaned version.")
    print("=" * 40)


def get_x_number(song_str):
    """Auxiliary function: Extract the X number for log purposes"""
    match = re.search(r"^X:\s*(\d+)", song_str, re.MULTILINE)
    return match.group(1) if match else "?"


if __name__ == "__main__":
    clean_and_save()
