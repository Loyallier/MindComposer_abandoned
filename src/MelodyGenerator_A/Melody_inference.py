import os
import torch
import music21
from datetime import datetime
import re

# Import custom modules
from Melody_model import MelodyGPT, GPTConfig
from Melody_tokenizer import MelodyTokenizer

# ================= CONFIGURATION =================
# Hardware and Paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.join("data", "processed")
VOCAB_PATH = os.path.join(BASE_DIR, "Melody_vocab.json")
CKPT_PATH = os.path.join("src", "MelodyGenerator_A", "Melody_ckpt.pt")
OUTPUT_DIR = os.path.join("samples", "Melody Outputs")

# Model Parameters (Must strictly match Melody_train.py)
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384

# Generation Parameters
MAX_NEW_TOKENS = 500  # Generation length
TEMPERATURE = 0.8  # 0.8 balances creativity and accuracy (set to 1.0 for overfit verification)
TOP_K = 5  # Truncated sampling to prevent generating low-probability gibberish

# User Customization Area
TARGET_KEY = "C"  # Target key (e.g., C, G, D, Am)
TARGET_METER = "3/4"  # Target meter (e.g., 4/4, 3/4, 6/8)


# =========================================
def extract_first_song(full_text):
    """
    Logic upgrade: Strict truncation.
    Terminates immediately if a new Header (M:, K:, X:, T:) or a pure numeric index is detected after the body.
    """
    lines = full_text.split("\n")
    valid_lines = []

    # Status flag: whether the core Header (K:) has been read
    # Since the Seed usually contains M: and K:, we need to allow these at the beginning
    seen_key_signature = False

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 1. Check if it is a pure number (e.g., '48', '50' as dataset indices)
        # If it appears after the first line and looks like an index, truncate.
        if i > 0 and re.match(r"^\d+$", line):
            # print(f"[Debug] Detected Index '{line}', truncating.")
            break

        # 2. Check if it is a new Header
        # If we have passed the K: stage (entering the melody) but encounter M:, K:, or X:, it indicates a second song.
        is_header = re.match(r"^[A-Z]:", line)

        if seen_key_signature:
            # After entering the melody section, if M: K: X: T: etc. are encountered, treat as a new song start
            if is_header:
                # print(f"[Debug] Detected new Header '{line}', truncating.")
                break

        # 3. Collect lines
        valid_lines.append(line)

        # 4. Update status
        # Once K: is read, mark the Header stage as ended (or nearly ended); subsequent non-Header lines are melody.
        if line.startswith("K:"):
            seen_key_signature = True

    return "\n".join(valid_lines)


def generate_melody_with_params(
    model, tokenizer, key="G", meter="4/4", max_tokens=500, temp=0.8
):
    """
    Wrapper function: Generate melody based on specified key and meter.
    """
    # 1. Build Seed
    # Format strictly follows cleaning logic: separated by newlines
    seed_text = f"M:{meter}\nK:{key}\n"
    print(f"[Inference] Generating with Seed: {repr(seed_text)}")

    # 2. Encoding
    start_ids = tokenizer.encode(seed_text)
    x = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    # 3. Generation
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=40)

    # 4. Decoding
    full_text = tokenizer.decode(y[0].tolist())

    # 5. Extract single song
    clean_abc = extract_first_song(full_text)

    # 6. Fallback: If X:1 is missing after extraction, add it (Required by Music21)
    if "X:" not in clean_abc:
        clean_abc = "X:1\n" + clean_abc

    return clean_abc


def save_midi(abc_str, filename):
    """
    Revised version: MIDI conversion with enhanced fault tolerance.
    """
    try:
        # 1. Ensure X: index exists (Strictly required by Music21)
        if not re.search(
            r"\nX:\s*\d+", "\n" + abc_str
        ) and not abc_str.lstrip().startswith("X:"):
            # If no X: marker, manually prepend it
            abc_str = "X:1\n" + abc_str

        print(f"[Converter] Parsing ABC content (Length: {len(abc_str)} chars)...")

        # 2. Parsing
        # Using forceSource=True can sometimes bypass certain cache errors, though usually automatic in parse method
        s = music21.converter.parse(abc_str, format="abc")

        # 3. Handle Opus (Multiple songs) vs Score (Single song)
        # Even after truncation, Music21 sometimes returns an Opus object
        if isinstance(s, music21.stream.Opus):
            # If it's a collection, take the first one
            stream_to_write = s[0]
        else:
            stream_to_write = s

        # 4. Write
        stream_to_write.write("midi", fp=filename)
        print(f"[Converter] Success! MIDI saved to: {filename}")
        return True

    except music21.abcFormat.ABCParsingException as e:
        print(f"[Error] ABC Syntax Error: {e}")
        return False
    except Exception as e:
        # Catch stream errors like 'TimeSignature already found'
        print(f"[Error] Music21 Logic Error: {e}")
        print(">> Suggestion: The generated ABC might contain conflicting headers.")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[System] Device: {DEVICE}")

    # 1. Load Tokenizer
    tokenizer = MelodyTokenizer()
    try:
        tokenizer.load_vocab(VOCAB_PATH)
    except FileNotFoundError:
        print(f"[Error] Vocab file not found at {VOCAB_PATH}. Cannot decode output.")
        return

    # 2. Initialize Model and Load Weights
    print(f"[Inference] Loading model from {CKPT_PATH}...")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=0.0,  # Dropout not needed for inference
    )
    model = MelodyGPT(config)

    # Load State Dict
    try:
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()  # Switch to evaluation mode
        print("[Inference] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load model checkpoint: {e}")
        return

    # Call the wrapped generation function
    abc_content = generate_melody_with_params(
        model=model,
        tokenizer=tokenizer,
        key=TARGET_KEY,
        meter=TARGET_METER,
        max_tokens=500,  # Long enough to include a full song
        temp=0.9,  # Slightly increase randomness for more variation
    )

    # Post-processing and Saving
    print("-" * 30)
    print(f"[Inference] Extracted Song ({TARGET_KEY}, {TARGET_METER}):")
    print(abc_content)
    print("-" * 30)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(OUTPUT_DIR, f"melody_{TARGET_KEY}_{timestamp}")

    # Save ABC
    with open(output_base + ".abc", "w", encoding="utf-8") as f:
        f.write(abc_content)

    # Convert to MIDI (Reusing previous save_midi)
    save_midi(abc_content, output_base + ".mid")


if __name__ == "__main__":
    main()