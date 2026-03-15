import json
import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src import path
from src.ChordGenerator_A import config

class MusicDataset(Dataset):
    def __init__(self, data_path, augment=False):
        """
        [V3.3 Update] Added augment parameter
        augment=True: Enable data augmentation (noise injection) for training set
        augment=False: Keep original data for validation set
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.augment = augment 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get original list (Note: this is a reference, modifying it affects data in memory)
        src_list = item["input"]
        
        # [V3.3] Dynamic melody noise injection
        # Only triggered when augment=True
        if self.augment:
            src_list = src_list.copy() 
            
            # Avoid SOS at the beginning and EOS at the end
            seq_len = len(src_list)
            if seq_len > 2:
                for i in range(1, seq_len - 1):
                    # Skip special symbols (e.g., BAR=config.BAR_TOKEN or its ID)
                    # Are IDs or Tokens stored in src_list?
                    # According to tokenize_data.py, IDs (int) are stored in the json
                    # Assuming the ID corresponding to config.BAR_TOKEN should not be masked
                    # [V3.5] Reduce difficulty: 0.15 -> 0.10 (10% noise)
                    if random.random() < 0.10:
                        # Replace with "0" (ID for UNK/PAD, usually 0)
                        # Force the model to "guess" this note
                        src_list[i] = config.PAD_IDX

        # Convert to Tensor
        src = torch.LongTensor(src_list)
        pos = torch.LongTensor(item["position"])
        trg = torch.LongTensor(item["target"])
        length = item["length"]

        return src, pos, trg, length
    
def collate_fn(batch):
    """
    [Key Function] Batch Padding
    """
    # 1. Unpack (Added pos_batch)
    src_batch, pos_batch, trg_batch, lengths = zip(*batch)

    # 2. Padding
    # Use 0 (the <PAD> in vocab) for src and trg
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=config.PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=config.PAD_IDX)
    
    # [V3.1 New] Use POS_PAD_IDX (32) for pos padding
    # Because 0 is a valid position index (the first beat), it cannot be used for Padding
    pos_padded = pad_sequence(pos_batch, batch_first=True, padding_value=config.POS_PAD_IDX)

    # 3. Convert to Tensor
    lengths = torch.LongTensor(lengths)

    return src_padded, pos_padded, trg_padded, lengths


# --- Test Code ---
if __name__ == "__main__":
    # Test loading
    import os
    if os.path.exists(path.TRAIN_DATASET_PATH):
        dataset = MusicDataset(str(path.TRAIN_DATASET_PATH))
        print(f"Dataset size: {len(dataset)}")

        # Simulate fetching a sample
        src, pos, trg, l = dataset[0]
        print(f"Sample 0:")
        print(f" - Src shape: {src.shape}")
        print(f" - Pos shape: {pos.shape} (Example: {pos[:10].tolist()})")
        print(f" - Trg shape: {trg.shape}")
    else:
        print("Dataset file not found, please run tokenize_data.py first")