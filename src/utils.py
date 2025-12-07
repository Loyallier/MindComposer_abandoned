# 文件: src/utils.py

import json
import os
import torch

# ================= 1. 全局常量 (Single Source of Truth) =================
# 以后要改特殊符，只改这里一个地方
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "0"      # 未知/脏数据统一映射为休止符

# ================= 2. 通用工具函数 =================

def load_vocab(vocab_path):
    """加载字典的通用函数"""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"❌ 找不到字典文件: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def clean_melody_token(token):
    """
    【核心清洗逻辑】
    确保 训练(train) 和 推理(inference) 使用完全相同的清洗标准。
    """
    token = str(token).strip()
    
    # 保留合法的特殊符
    if token in ["_", "0", PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]: 
        return token
    
    # 保留数字 (MIDI pitch)
    if token.isdigit(): 
        return token
    
    # 其他所有脏数据 -> 强制转为休止符 (UNK_TOKEN)
    return UNK_TOKEN

def token_to_tensor(token_list, vocab_stoi, device):
    """
    推理专用：把字符列表 -> 加上SOS/EOS -> 转为 Tensor
    """
    # 1. 获取特殊 ID
    sos_id = vocab_stoi.get(SOS_TOKEN)
    eos_id = vocab_stoi.get(EOS_TOKEN)
    unk_id = vocab_stoi.get(UNK_TOKEN, 3) # 默认 ID 3 是 _ 或 0，视字典而定
    
    # 2. 转换 (Handle Unknowns)
    ids = [sos_id] + [vocab_stoi.get(t, unk_id) for t in token_list] + [eos_id]
    
    # 3. 变 Tensor (Batch Dim)
    tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    length = torch.LongTensor([len(ids)])
    
    return tensor, length