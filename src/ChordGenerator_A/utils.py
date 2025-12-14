# import json
# import os
# import torch

# # 👇 引入 config (统一常量)
# try:
#     from src.ChordGenerator_A import config
# except ImportError:
#     import src.ChordGenerator_A.config as config

import json
import os
import torch
from collections import Counter

# ✅ 统一引用：从 src 开始
from src import path
from src.ChordGenerator_A import config

# ================= 通用工具函数 =================

def load_vocab(vocab_path):
    """
    加载字典的通用函数
    ✅ 必须返回字典内容，绝对不能 pass！
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"❌ 找不到字典文件: {vocab_path} (请检查 config.py 里的路径)")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 👇 关键！必须把读到的东西返回出去
    return vocab

def clean_melody_token(token):
    """清洗旋律 Token"""
    token = str(token).strip()
    
    # 使用 config 里的定义
    if token in ["_", "0", config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.BAR_TOKEN]: 
        return token
    
    if token.isdigit(): 
        return token
    
    return config.UNK_TOKEN

def token_to_tensor(token_list, vocab_stoi, device):
    """字符转 Tensor"""
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

# ... (保留原有的 load_vocab, clean_melody_token 等函数) ...

def get_dominant_bar_length(measure_tokens_list):
    """统计众数长度 (复用自 tokenize_data.py)"""
    if not measure_tokens_list: return 16
    lengths = [len(b) for b in measure_tokens_list]
    counts = Counter(lengths)
    # 返回出现次数最多的长度
    return counts.most_common(1)[0][0]

def generate_smart_position_indices(measure_tokens_list):
    """
    [V3.1 核心逻辑复用]
    为整首曲子生成绝对位置索引 (List of Lists)。
    包含: 弱起右对齐 + 标准左对齐 + BAR位置处理
    """
    dominant_len = get_dominant_bar_length(measure_tokens_list)
    all_pos_indices = []
    
    max_beat = config.POS_VOCAB_SIZE - 3 # 30
    bar_pos_idx = config.POS_VOCAB_SIZE - 2 # 31 (用于 BAR)

    for i, bar_tokens in enumerate(measure_tokens_list):
        bar_len = len(bar_tokens)
        current_bar_pos = []
        
        # --- 弱起处理 (Pickup Logic) ---
        # 逻辑：第一小节 & 短于众数 & 短于10
        if i == 0 and bar_len < dominant_len and bar_len < 10:
            start_idx = max(0, dominant_len - bar_len)
            # 生成: [14, 15]
            current_bar_pos = [min(start_idx + k, max_beat) for k in range(bar_len)]
        else:
            # 标准左对齐: [0, 1, 2...]
            current_bar_pos = [min(k, max_beat) for k in range(bar_len)]
            
        all_pos_indices.append(current_bar_pos)
        
    return all_pos_indices

def token_to_tensor_v3_with_pos(token_list, pos_list, vocab_stoi, device):
    """
    [V3.1 Inference] 接收外部计算好的 token 和 pos，转为 Tensor
    """
    clean_seq = [clean_melody_token(t) for t in token_list]
    
    sos_id = vocab_stoi.get(config.SOS_TOKEN)
    eos_id = vocab_stoi.get(config.EOS_TOKEN)
    unk_id = vocab_stoi.get(config.UNK_TOKEN, 0)
    bar_pos_idx = config.POS_VOCAB_SIZE - 2 # 31
    
    # 1. Token IDs: [SOS] + Body + [EOS]
    ids = [sos_id] + [vocab_stoi.get(t, unk_id) for t in clean_seq] + [eos_id]
    
    # 2. Pos IDs: [31] + Body + [31]
    # 注意：pos_list 必须已经包含了 <BAR> 对应的 31 (如果 token 里有 <BAR>)
    # 但根据 predict_midi 的逻辑，我们传入的 pos_list 是纯数值，遇到 token=<BAR> 时我们需要手动补 31
    
    final_pos = [bar_pos_idx] # SOS pos
    
    # 这里需要非常小心地对齐
    # token_list: ['60', '62', '<BAR>']
    # pos_list:   [0,    1,    31]  <-- 外部传入时要保证对齐
    final_pos.extend(pos_list)
    final_pos.append(bar_pos_idx) # EOS pos
    
    src_tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    pos_tensor = torch.LongTensor(final_pos).unsqueeze(0).to(device)
    length = torch.LongTensor([len(ids)])
    
    return src_tensor, pos_tensor, length

def get_current_tf_ratio(epoch):
    """
    [V3.4 Engineering] 计算当前 Epoch 的 Teacher Forcing Ratio
    策略: 线性衰减 (Linear Decay)
    """
    # 如果还没到衰减结束轮次
    if epoch < config.TF_DECAY_EPOCHS:
        # 计算衰减步长
        decay_step = (config.TF_START_RATIO - config.TF_END_RATIO) / config.TF_DECAY_EPOCHS
        current_ratio = config.TF_START_RATIO - (epoch * decay_step)
        return max(config.TF_END_RATIO, current_ratio)
    else:
        # 超过衰减轮次，保持最低值
        return config.TF_END_RATIO