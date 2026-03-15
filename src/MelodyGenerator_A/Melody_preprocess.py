import os
import re

# ================= CONFIGURATION =================
RAW_DATA_DIR = os.path.join('data', 'raw')
OUTPUT_DIR = os.path.join('data', 'processed')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'dataset.txt')

# Define whitelist character set (based on standard ABC music theory)
# Includes: uppercase and lowercase letters, numbers, spaces, newlines
# Musical symbols: | : [ ] / - ^ = , ' . > ( )
VALID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n|:[]/-^=_,'().>")

# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_line_strict(line):
    """
    Strict cleaning:
    1. Remove comments (%)
    2. Remove chords ("...")
    3. Keep only characters in the whitelist
    4. Compress extra spaces
    """
    # 1. Remove comments
    line = line.split('%')[0]
    
    # 2. Remove chords (content wrapped in double quotes)
    line = re.sub(r'"[^"]*"', '', line)
    
    # 3. Whitelist filtering
    # Check character by character, keep if in whitelist
    line = ''.join([c for c in line if c in VALID_CHARS])
    
    # 4. Strip leading and trailing whitespace
    return line.strip()

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split tracks by X:
    raw_songs = re.split(r'\n\s*X:', content)
    processed_songs = []

    for song in raw_songs:
        if not song.strip():
            continue
            
        lines = song.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Perform strict cleaning first
            cleaned_line = clean_line_strict(line)
            
            if not cleaned_line:
                continue

            # Simple logic to determine if it is a required Header or melody
            # Since it's cleaned, ensure M: and K: structures are not destroyed (colon and letters are in whitelist)
            
            is_header = False
            # Check if it is M: or K:
            # Regex explanation: Start of line is a letter followed immediately by a colon
            if re.match(r'^[A-Z]:', cleaned_line):
                if cleaned_line.startswith('M:') or cleaned_line.startswith('K:'):
                    filtered_lines.append(cleaned_line)
                is_header = True
            else:
                # Non-Header line, i.e., melody body
                filtered_lines.append(cleaned_line)

        # Keep only when both Header and melody exist (prevents empty songs)
        if len(filtered_lines) > 2:
            song_text = '\n'.join(filtered_lines)
            processed_songs.append(song_text)

    return processed_songs

def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: {RAW_DATA_DIR} not found.")
        return

    all_songs = []
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.abc')]
    
    print(f"Scanning {len(files)} files...")

    for filename in files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        songs = process_file(filepath)
        all_songs.extend(songs)

    # Write to file
    # Use <|endoftext|> as a delimiter
    # Note: This delimiter is added after cleaning and is not restricted by the whitelist; this is the correct logic.
    full_content = '\n<|endoftext|>\n'.join(all_songs) + '\n<|endoftext|>'
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"Strict filtering complete.")
    print(f"Valid Songs Extracted: {len(all_songs)}")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Data validation (Sanity Check)
    print("-" * 30)
    print("Sample Output (First 5 lines):")
    print("\n".join(full_content.split('\n')[:5]))
    print("-" * 30)

if __name__ == "__main__":
    main()