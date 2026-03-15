import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pickle # Used for potential caching; not used in this example but kept for extensibility
from Melody_tokenizer import MelodyTokenizer # Ensure filename case sensitivity matches

class MelodyDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size=128):
        """
        Initialize the dataset
        :param data_path: Path to the preprocessed dataset.txt
        :param tokenizer: Instance of the Tokenizer already built and loaded with a vocabulary
        :param block_size: Context window size (Context Length)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 1. Read data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 2. Tokenization (Convert full data to Tensor)
        # For 1,000 songs, the data volume is small; loading everything directly into memory is most efficient
        print(f"[Dataset] Tokenizing {len(raw_text)} characters...")
        self.data_ids = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)
        print(f"[Dataset] Tokenization complete. Total tokens: {len(self.data_ids)}")

    def __len__(self):
        # Dataset length = Total tokens - block_size
        # This ensures the last sample has a complete block_size + 1 (used for target)
        return len(self.data_ids) - self.block_size

    def __getitem__(self, idx):
        # Logic:
        # We need to extract a segment of length block_size + 1
        # chunk = [token_i, token_i+1, ..., token_i+block_size]
        # x (input) = chunk[:-1] (excludes the last one)
        # y (target) = chunk[1:]  (excludes the first one, i.e., shifted by one position)
        
        chunk = self.data_ids[idx : idx + self.block_size + 1]
        
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y

def create_dataloaders(data_path, vocab_path, block_size=128, batch_size=64, train_split=0.9):
    """
    Factory function: Creates DataLoaders for training and validation sets
    """
    # 1. Prepare Tokenizer
    tokenizer = MelodyTokenizer()
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        # If no existing vocab, it should logically throw an error or be built temporarily.
        # For rigor, we require Melody_tokenizer.py to be run first to generate the vocab.
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}. Run Melody_tokenizer.py first.")

    # 2. Instantiate Dataset
    full_dataset = MelodyDataset(data_path, tokenizer, block_size)

    # 3. Split training/validation sets
    # For overfitting experiments, you can typically use all data for training.
    # For logical rigor, we keep the standard splitting interface.
    # If you want pure overfitting, you can set train_split to 1.0 (requires handling remainder errors; 0.99 suggested)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. Create DataLoaders
    # pin_memory=True accelerates data transfer for GPU training
    # For Windows, num_workers=0 is recommended to avoid multiprocess errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    print(f"[DataLoader] Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    print(f"[DataLoader] Batch size: {batch_size} | Block size: {block_size}")
    
    return train_loader, val_loader, tokenizer

# ================= Test Code =================
if __name__ == "__main__":
    # Mock paths
    PROCESSED_DATA = os.path.join('data', 'processed', 'dataset.txt')
    VOCAB_FILE = os.path.join('data', 'processed', 'Melody_vocab.json')

    # Test parameters
    BATCH_SIZE = 4 # Small batch for testing
    BLOCK_SIZE = 10 # Small window for testing

    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            PROCESSED_DATA, 
            VOCAB_FILE, 
            block_size=BLOCK_SIZE, 
            batch_size=BATCH_SIZE
        )

        # Get one batch to check data shape
        x_batch, y_batch = next(iter(train_loader))
        
        print("-" * 30)
        print("Data Shape Check:")
        print(f"Input (x) shape: {x_batch.shape}") # Expected: [4, 10]
        print(f"Target (y) shape: {y_batch.shape}")# Expected: [4, 10]
        
        print("\nLogic Check (First sample in batch):")
        print(f"x[0]: {x_batch[0].tolist()}")
        print(f"y[0]: {y_batch[0].tolist()}")
        
        decoded_x = tokenizer.decode(x_batch[0])
        decoded_y = tokenizer.decode(y_batch[0])
        
        print(f"\nText x: {repr(decoded_x)}")
        print(f"Text y: {repr(decoded_y)}")
        
        # Logic verification: The first character of y should be the second character of x
        if decoded_x[1:] == decoded_y[:-1]:
            print(">> LOGIC VERIFIED: Target is correctly shifted.")
        else:
            print(">> LOGIC ERROR: Target mismatch.")
        print("-" * 30)

    except Exception as e:
        print(f"Test failed: {e}")