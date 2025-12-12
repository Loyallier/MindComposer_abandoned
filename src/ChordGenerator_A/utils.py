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