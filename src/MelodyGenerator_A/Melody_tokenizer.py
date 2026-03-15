import os
import json
import torch  # Assuming you use PyTorch for Transformer training

# ================= CONFIGURATION =================
# Follow the Melody_ prefix rule
DATA_DIR = os.path.join('data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'dataset.txt')
VOCAB_FILE = os.path.join(DATA_DIR, 'Melody_vocab.json')

# =========================================

class MelodyTokenizer:
    def __init__(self):
        self.stoi = {} # String to Integer
        self.itos = {} # Integer to String
        self.vocab_size = 0

    def build_vocab(self, text_data):
        """
        Build vocabulary based on text data
        """
        # Get all unique characters and sort them to ensure determinism
        chars = sorted(list(set(text_data)))
        self.vocab_size = len(chars)
        
        # Build mapping
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        print(f"[Tokenizer] Vocab built. Size: {self.vocab_size}")
        print(f"[Tokenizer] Sample chars: {chars[:10]}")

    def save_vocab(self, filepath):
        """
        Save vocabulary to JSON
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            # Note: JSON keys must be strings; itos keys (int) will be converted to str during saving
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f, indent=4)
        print(f"[Tokenizer] Vocab saved to {filepath}")

    def load_vocab(self, filepath):
        """
        Load vocabulary from JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocab file not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.stoi = data['stoi']
            # When loading back, itos keys are strings and need to be converted back to int
            self.itos = {int(k): v for k, v in data['itos'].items()}
            self.vocab_size = len(self.stoi)
        
        print(f"[Tokenizer] Vocab loaded. Size: {self.vocab_size}")

    def encode(self, text):
        """
        Convert string to list of integers
        """
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        """
        Convert integer list (or Tensor) to string
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.itos[i] for i in indices])

def main():
    # 1. Read cleaned data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run Melody_preprocess.py first.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text_data = f.read()

    # 2. Initialize and build Tokenizer
    tokenizer = MelodyTokenizer()
    tokenizer.build_vocab(text_data)

    # 3. Save vocabulary
    tokenizer.save_vocab(VOCAB_FILE)

    # 4. Sanity Check: Encoding/Decoding logic
    print("-" * 30)
    print("Sanity Check: Encoding/Decoding")
    
    # Test with the first 50 characters
    sample_text = text_data[:50]
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {repr(sample_text)}")
    print(f"Encoded : {encoded}")
    print(f"Decoded : {repr(decoded)}")
    
    # Logical verification: Must be perfectly restored
    if sample_text == decoded:
        print(">> TEST PASSED: Perfect reconstruction.")
    else:
        print(">> TEST FAILED: Data mismatch.")
    print("-" * 30)

    # 5. Demonstrate Tensor conversion
    tensor_data = torch.tensor(encoded, dtype=torch.long)
    print(f"Tensor Shape: {tensor_data.shape}")
    print(f"Tensor Type:  {tensor_data.dtype}")

if __name__ == "__main__":
    main()