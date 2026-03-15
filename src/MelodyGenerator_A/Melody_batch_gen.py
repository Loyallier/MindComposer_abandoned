import os
import torch
import random
import time
import music21
from datetime import datetime

# Import custom modules from the same directory
from Melody_model import MelodyGPT, GPTConfig
from Melody_tokenizer import MelodyTokenizer
from Melody_inference import generate_melody_with_params

# ================= CONFIGURATION =================
# Number of songs to generate
NUM_SONGS = 1000

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path configuration
# Assuming this script is running in src/MelodyGenerator_A/
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOCAB_PATH = os.path.join(BASE_PROJECT_DIR, "data", "processed", "Melody_vocab.json")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "Melody_ckpt.pt")
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "samples", "test_melody_1000")

# Model hyperparameters (Must match training config)
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.0

# Randomization pools
KEYS = ['C', 'G', 'D', 'A', 'F', 'Bb', 'Eb', 'Am', 'Em', 'Dm', 'Gm']
METERS = ['4/4', '3/4', '2/4', '6/8', '12/8']

def safe_save_midi(abc_content, output_path):
    """
    Attempts to convert ABC content to MIDI using music21.
    Returns: (bool success, str error_message)
    """
    try:
        # 1. Validation: Ensure X: header exists (Music21 requirement)
        if "X:" not in abc_content:
            abc_content = "X:1\n" + abc_content

        # 2. Parsing
        # forceSource=True helps bypass some internal caching issues
        stream = music21.converter.parse(abc_content, format='abc', forceSource=True)

        # 3. Handling Opus objects (collections) vs Scores
        if isinstance(stream, music21.stream.Opus):
            stream = stream[0]
            
        # 4. Write to file
        stream.write('midi', fp=output_path)
        return True, None
        
    except Exception as e:
        return False, str(e)

def main():
    # 1. Setup Directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[System] Device: {DEVICE}")
    print(f"[Batch] Output Directory: {OUTPUT_DIR}")
    print(f"[Batch] Target: {NUM_SONGS} songs")

    # 2. Load Tokenizer
    print("[Batch] Loading Tokenizer...")
    tokenizer = MelodyTokenizer()
    if os.path.exists(VOCAB_PATH):
        tokenizer.load_vocab(VOCAB_PATH)
        print(f"[Batch] Vocab loaded. Size: {tokenizer.vocab_size}")
    else:
        print(f"[Error] Vocab file not found at {VOCAB_PATH}")
        return

    # 3. Load Model (Load once, run many times)
    print(f"[Batch] Loading Model from {CKPT_PATH}...")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT
    )
    model = MelodyGPT(config)
    
    try:
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("[Batch] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load model checkpoint: {e}")
        return

    # 4. Statistics Counters
    stats = {
        'total_attempts': 0,
        'generation_success': 0,
        'midi_conversion_success': 0,
        'midi_conversion_errors': 0,
        'error_details': []
    }

    print("-" * 50)
    print("[Batch] Starting Generation Loop...")
    start_time = time.time()

    # 5. Generation Loop
    for i in range(1, NUM_SONGS + 1):
        stats['total_attempts'] += 1
        
        # Randomize Parameters
        key = random.choice(KEYS)
        meter = random.choice(METERS)
        # Randomize temperature slightly for variety (0.7 to 0.95)
        temp = round(random.uniform(0.7, 0.95), 2)
        
        print(f"[{i}/{NUM_SONGS}] Generating (Key:{key}, Meter:{meter}, T:{temp})...", end="", flush=True)

        try:
            # A. Generate Text (ABC Notation)
            abc_content = generate_melody_with_params(
                model=model,
                tokenizer=tokenizer,
                key=key,
                meter=meter,
                max_tokens=600,
                temp=temp
            )
            stats['generation_success'] += 1

            # Define filenames
            timestamp = datetime.now().strftime("%H%M%S")
            safe_meter = meter.replace('/', '-')
            filename_base = f"song_{i:03d}_{key}_{safe_meter}_{timestamp}"
            abc_path = os.path.join(OUTPUT_DIR, filename_base + ".abc")
            midi_path = os.path.join(OUTPUT_DIR, filename_base + ".mid")

            # B. Save ABC File
            with open(abc_path, "w", encoding="utf-8") as f:
                f.write(abc_content)

            # C. Convert to MIDI
            success, error_msg = safe_save_midi(abc_content, midi_path)
            
            if success:
                stats['midi_conversion_success'] += 1
                print(" -> [OK]")
            else:
                stats['midi_conversion_errors'] += 1
                stats['error_details'].append(f"Song {i}: {error_msg}")
                print(" -> [MIDI FAIL]")

            # Optional: Clear GPU cache periodically to prevent fragmentation
            if i % 20 == 0 and DEVICE == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f" -> [GEN FAIL] {e}")
            stats['error_details'].append(f"Song {i} (Generation): {str(e)}")

    # 6. Final Report
    total_time = time.time() - start_time
    success_rate = (stats['midi_conversion_success'] / stats['total_attempts']) * 100 if stats['total_attempts'] > 0 else 0

    print("\n" + "=" * 50)
    print("BATCH GENERATION REPORT")
    print("=" * 50)
    print(f"Total Songs Requested : {NUM_SONGS}")
    print(f"Text Generation OK    : {stats['generation_success']}")
    print(f"MIDI Conversion OK    : {stats['midi_conversion_success']}")
    print(f"MIDI Conversion Fail  : {stats['midi_conversion_errors']}")
    print("-" * 30)
    print(f"Music21 Success Rate  : {success_rate:.2f}%")
    print("-" * 30)
    print(f"Total Time Elapsed    : {total_time:.2f} seconds")
    print(f"Average Time Per Song : {total_time/NUM_SONGS:.2f} seconds")
    print(f"Output Directory      : {OUTPUT_DIR}")
    
    if stats['error_details']:
        print("\n[Error Log Sample (First 5 errors)]:")
        for err in stats['error_details'][:5]:
            print(f" - {err}")
        if len(stats['error_details']) > 5:
            print(f" ... and {len(stats['error_details']) - 5} more errors.")
    print("=" * 50)

if __name__ == "__main__":
    main()