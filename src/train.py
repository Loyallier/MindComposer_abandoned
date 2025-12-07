import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time
import sys # 新增：用于优雅退出

# 导入你自己写的模块
from model import Encoder, Decoder, Seq2Seq
from dataset import MusicDataset, collate_fn
from utils import load_vocab

import config

# ================= 1. 配置区域 =================
DATA_PATH = r"data\processed\dataset_encoded.json"
VOCAB_PATH = r"data\processed\vocab.json"

# 保存两个模型：一个是 Loss 最低的(Best)，一个是最后时刻的(Last)
MODEL_SAVE_PATH = r"models\best_model.pth"
LAST_MODEL_PATH = r"models\checkpoints\last_checkpoint.pth" 

# # 超参数
# BATCH_SIZE = 32         # 切片后可以开大 Batch
# LEARNING_RATE = 0.001   
# N_EPOCHS = 100          
# CLIP = 1                

# # 模型参数
# ENC_EMB_DIM = 64
# DEC_EMB_DIM = 64
# HIDDEN_DIM = 128
# DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 辅助函数 =================
def load_vocab_dims(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        input_dim = len(vocab['melody'])
        output_dim = len(vocab['harmony'])
        print(f"📖 字典加载成功 | 旋律词表: {input_dim}, 和弦词表: {output_dim}")
        return input_dim, output_dim

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg, lengths) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, lengths)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# ================= 3. 主程序 =================
if __name__ == "__main__":
    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"🚀 正在使用设备: {device}")
    
    if not os.path.exists(VOCAB_PATH):
        print(f"❌ 错误：找不到文件 {VOCAB_PATH}")
        sys.exit(1)
        
    INPUT_DIM, OUTPUT_DIM = load_vocab_dims(VOCAB_PATH)

    print("📦正在加载数据...")
    dataset = MusicDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    print(f"✅ 数据加载完毕，共 {len(dataset)} 条训练切片")

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print("🧠 模型初始化完成")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n🔥 开始训练！(随时按 Ctrl+C 安全停止)")
    print("-" * 65)
    print(f"{'Epoch':<10} | {'Time':<10} | {'Loss':<10} | {'Status'}")
    print("-" * 65)
    
    best_loss = float('inf')
    
    # 🛡️【关键修改】这里加了 try...except 块
    try:
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            
            train_loss = train(model, dataloader, optimizer, criterion, CLIP)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            save_msg = ""
            # 如果是历史最佳，保存 best_model
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                save_msg = "💾 Best Saved"
                
            print(f"{epoch+1:<10} | {int(epoch_mins)}m {int(epoch_secs)}s    | {train_loss:.4f}     | {save_msg}")

    except KeyboardInterrupt:
        print("\n" + "=" * 65)
        print("🛑 检测到用户中断 (Ctrl+C)！正在进行安全退出处理...")
        print("=" * 65)
        
    finally:
        # 无论正常结束还是强行打断，这里都会执行
        print("\n📊 训练总结:")
        
        # 1. 确认最佳模型是否存在
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"✅ 历史最佳模型已安全保存至: {MODEL_SAVE_PATH} (Loss: {best_loss:.4f})")
        else:
            print(f"⚠️  尚未产生最佳模型 (Loss 尚未下降)。")

        # 2. 保存最后时刻的模型 (Checkpoint)
        # 有时候"最佳 Loss"的模型可能有点过拟合，保留"最后时刻"的模型作为备选很有用
        torch.save(model.state_dict(), LAST_MODEL_PATH)
        print(f"💾 中断时的模型状态已保存至: {LAST_MODEL_PATH}")
        print("-" * 65)
        print("👉 现在你可以运行 interface.py 了！")