# import json
# import os
# import torch

# # 👇 Import config (Unified constants)
# try:
#     from src.ChordGenerator_A import config
# except ImportError:
#     import src.ChordGenerator_A.config as config

import json
import os
import torch
from collections import Counter

# Unified reference: Start from src
from src import path
from src.ChordGenerator_A import config

# ================= General Utility Functions =================

def load_vocab(vocab_path):
    """
    General function to load the dictionary.
    Must return dictionary content; never just 'pass'!
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path} (Please check the path in config.py)")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Critical! Must return the loaded content
    return vocab

def clean_melody_token(token):
    """Clean melody Token"""
    token = str(token).strip()
    
    # Use definitions from config
    if token in ["_", "0", config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.BAR_TOKEN]: 
        return token
    
    if token.isdigit(): 
        return token
    
    return config.UNK_TOKEN

def token_to_tensor(token_list, vocab_stoi, device):
    """Convert characters to Tensor"""
    sos_id = vocab_stoi.get(config.SOS_TOKEN)
    eos_id = vocab_stoi.get(config.EOS_TOKEN)
    unk_id = vocab_stoi.get(config.UNK_TOKEN, 3)
    
    ids = [sos_id] + [vocab_stoi.get(t, unk_id) for t in token_list] + [eos_id]
    
    tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    length = torch.LongTensor([len(ids)])
    
    return tensor, length

def calculate_positions(token_list):
    """
    [V3.1] Inference Helper: Calculate Absolute Position IDs for a token sequence.
    Logic must match tokenize_data.py (Smart Alignment).
    """
    pos_seq = []
    
    # Simple logic: We assume the input slice is a valid window.
    # We reset the counter to 0 after every <BAR>.
    # Note: Inference is harder because we don't know the "Dominant Length" of the whole song easily.
    # For generated chunks, we assume standard alignment (0, 1, 2...) unless specific logic is added.
    
    current_tick = 0
    for token in token_list:
        if token == config.BAR_TOKEN:
            pos_seq.append(config.POS_VOCAB_SIZE - 2) # BAR_POS_IDX (31)
            current_tick = 0 # Reset for next bar
        else:
            # Clamp to max beat index (30)
            idx = min(current_tick, config.POS_VOCAB_SIZE - 3) 
            pos_seq.append(idx)
            current_tick += 1
            
    return pos_seq

def token_to_tensor_v3(token_list, vocab_stoi, device):
    """
    [V3.1] Convert tokens AND positions to tensors.
    """
    # 1. Clean and Map Tokens
    clean_seq = [clean_melody_token(t) for t in token_list]
    
    sos_id = vocab_stoi.get(config.SOS_TOKEN)
    eos_id = vocab_stoi.get(config.EOS_TOKEN)
    unk_id = vocab_stoi.get(config.UNK_TOKEN, 0)
    
    # ID Sequence: [SOS] + [Body] + [EOS]
    ids = [sos_id] + [vocab_stoi.get(t, unk_id) for t in clean_seq] + [eos_id]
    
    # 2. Calculate Positions
    # Body positions
    body_pos = calculate_positions(clean_seq)
    
    # Add SOS/EOS positions (Mapping to BAR_POS_IDX=31)
    bar_pos_idx = config.POS_VOCAB_SIZE - 2 
    pos_ids = [bar_pos_idx] + body_pos + [bar_pos_idx]
    
    # 3. To Tensor
    src_tensor = torch.LongTensor(ids).unsqueeze(0).to(device) # [1, seq_len]
    pos_tensor = torch.LongTensor(pos_ids).unsqueeze(0).to(device) # [1, seq_len]
    length = torch.LongTensor([len(ids)])
    
    return src_tensor, pos_tensor, length

# ... (Retain original functions like load_vocab, clean_melody_token, etc.) ...

def get_dominant_bar_length(measure_tokens_list):
    """Count dominant length (Reused from tokenize_data.py)"""
    if not measure_tokens_list: return 16
    lengths = [len(b) for b in measure_tokens_list]
    counts = Counter(lengths)
    # Return the most frequent length
    return counts.most_common(1)[0][0]

def generate_smart_position_indices(measure_tokens_list):
    """
    [V3.1 Core Logic Reuse]
    Generates absolute position indices for the entire piece (List of Lists).
    Includes: Pickup (anacrusis) right-alignment + standard left-alignment + BAR position handling.
    """
    dominant_len = get_dominant_bar_length(measure_tokens_list)
    all_pos_indices = []
    
    max_beat = config.POS_VOCAB_SIZE - 3 # 30
    bar_pos_idx = config.POS_VOCAB_SIZE - 2 # 31 (for BAR)

    for i, bar_tokens in enumerate(measure_tokens_list):
        bar_len = len(bar_tokens)
        current_bar_pos = []
        
        # --- Pickup Handling (Pickup Logic) ---
        # Logic: First measure & shorter than dominant length & shorter than 10
        if i == 0 and bar_len < dominant_len and bar_len < 10:
            start_idx = max(0, dominant_len - bar_len)
            # Generate: [14, 15]
            current_bar_pos = [min(start_idx + k, max_beat) for k in range(bar_len)]
        else:
            # Standard left-alignment: [0, 1, 2...]
            current_bar_pos = [min(k, max_beat) for k in range(bar_len)]
            
        all_pos_indices.append(current_bar_pos)
        
    return all_pos_indices

def token_to_tensor_v3_with_pos(token_list, pos_list, vocab_stoi, device):
    """
    [V3.1 Inference] Receives pre-calculated tokens and positions, converts to Tensor
    """
    clean_seq = [clean_melody_token(t) for t in token_list]
    
    sos_id = vocab_stoi.get(config.SOS_TOKEN)
    eos_id = vocab_stoi.get(config.EOS_TOKEN)
    unk_id = vocab_stoi.get(config.UNK_TOKEN, 0)
    bar_pos_idx = config.POS_VOCAB_SIZE - 2 # 31
    
    # 1. Token IDs: [SOS] + Body + [EOS]
    ids = [sos_id] + [vocab_stoi.get(t, unk_id) for t in clean_seq] + [eos_id]
    
    # 2. Pos IDs: [31] + Body + [31]
    # Note: pos_list must already include 31 corresponding to <BAR> (if <BAR> exists in tokens)
    # However, based on predict_midi logic, the pos_list passed in is pure numerical; 
    # we manually pad 31 when token=<BAR>
    
    final_pos = [bar_pos_idx] # SOS pos
    
    # Needs careful alignment
    # token_list: ['60', '62', '<BAR>']
    # pos_list:   [0,    1,    31]  <-- Ensure alignment when passing from external source
    final_pos.extend(pos_list)
    final_pos.append(bar_pos_idx) # EOS pos
    
    src_tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    pos_tensor = torch.LongTensor(final_pos).unsqueeze(0).to(device)
    length = torch.LongTensor([len(ids)])
    
    return src_tensor, pos_tensor, length

def get_current_tf_ratio(epoch):
    """
    [V3.4 Engineering] Calculate Teacher Forcing Ratio for the current Epoch
    Strategy: Linear Decay
    """
    # If the decay end epoch has not been reached
    if epoch < config.TF_DECAY_EPOCHS:
        # Calculate decay step
        decay_step = (config.TF_START_RATIO - config.TF_END_RATIO) / config.TF_DECAY_EPOCHS
        current_ratio = config.TF_START_RATIO - (epoch * decay_step)
        return max(config.TF_END_RATIO, current_ratio)
    else:
        # After decay epochs, maintain the minimum value
        return config.TF_END_RATIO