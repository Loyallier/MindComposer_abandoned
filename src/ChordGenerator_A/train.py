import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import time
import sys

# 挂载根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"🔧 [System] Root mounted: {project_root}")

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A import utils
from src.ChordGenerator_A.model import Encoder, Decoder, Seq2Seq
from src.ChordGenerator_A.dataset import MusicDataset, collate_fn

# ================= 工具函数 =================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 [System] Seed locked: {seed}")

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=config.INIT_WEIGHT_STD)
        else:
            nn.init.constant_(param.data, 0)

def train(model, iterator, optimizer, criterion, clip, device):
    """
    V3.1 训练循环: 必须处理 pos (位置编码)
    """
    model.train()
    epoch_loss = 0
    
    # Unpack 4 items: src, pos, trg, lengths
    for i, (src, pos, trg, lengths) in enumerate(iterator):
        src = src.to(device)
        pos = pos.to(device) # ✅ [V3.1] 位置数据上微
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # ✅ [V3.1] 传入 pos
        output = model(src, pos, trg, lengths, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO)
        
        # Reshape for Loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    """
    V3.1 验证循环
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, pos, trg, lengths) in enumerate(iterator):
            src = src.to(device)
            pos = pos.to(device) # ✅ [V3.1]
            trg = trg.to(device)
            
            # ✅ [V3.1] 传入 pos
            output = model(src, pos, trg, lengths, teacher_forcing_ratio=0) 
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 环境初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"🚀 [Device] Using: {device}")
    set_seed(config.SEED)
    
    # 2. 加载数据 (V3.1: 双文件加载)
    print("📦 [Data] Loading Vocab...")
    vocab = utils.load_vocab(config.VOCAB_PATH)
    harmony_stoi = vocab['harmony']
    INPUT_DIM = len(vocab['melody'])
    OUTPUT_DIM = len(harmony_stoi)
    
    # ✅ [V3.1] 位置参数
    POS_DIM = config.POS_VOCAB_SIZE # e.g., 33
    POS_EMB_DIM = config.POS_EMB_DIM # e.g., 32

    print(f"   - Melody Vocab: {INPUT_DIM}")
    print(f"   - Harmony Vocab: {OUTPUT_DIM}")
    print(f"   - Position Dim: {POS_DIM} -> {POS_EMB_DIM}")

    print("📦 [Data] Loading Datasets (Split Mode)...")
    # ✅ [V3.1] 分别加载训练集和验证集 (防止泄露)
    if not os.path.exists(config.TRAIN_DATASET_PATH) or not os.path.exists(config.VAL_DATASET_PATH):
        print(f"❌ 找不到数据集文件! 请先运行 tokenize_data.py")
        sys.exit(1)

    train_dataset = MusicDataset(config.TRAIN_DATASET_PATH)
    val_dataset = MusicDataset(config.VAL_DATASET_PATH)
    
    print(f"📊 数据加载完成:")
    print(f"   - 训练集: {len(train_dataset)} samples")
    print(f"   - 验证集: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    # 3. 初始化模型
    # ✅ [V3.1] 传入 pos_dim 和 pos_emb_dim
    enc = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT, POS_DIM, POS_EMB_DIM)
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model.apply(init_weights)
    
    # 加载逻辑 (V3.1 默认 RESUME_TRAINING = False)
    if config.RESUME_TRAINING:
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"🔄 [Resume] Loading: {config.MODEL_SAVE_PATH}")
            try:
                model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
                print("✅ Weights loaded.")
            except Exception as e:
                print(f"⚠️ Load failed ({e}). Starting fresh.")
        else:
            print(f"✨ [Resume] No file found. Starting fresh.")
    else:
        print("🆕 [Fresh] Config forces fresh start (Architecture changed).")

    # 4. 优化器与 Loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Loss 权重设置
    loss_weights = torch.ones(OUTPUT_DIM).to(device)
    
    # A. 延音权重 (Hold)
    hold_id = harmony_stoi.get("_")
    if hold_id is not None:
        loss_weights[hold_id] = config.HOLD_LOSS_WEIGHT
        print(f"⚖️  Hold Weight: {config.HOLD_LOSS_WEIGHT}")
        
    # B. 屏蔽 BAR (如果需要)
    bar_id = harmony_stoi.get(config.BAR_TOKEN)
    if bar_id is not None:
        loss_weights[bar_id] = 0.0 # 强制不计算 BAR 的 Loss
        print(f"⚖️  BAR Weight: 0.0 (Masked)")

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)

    # 5. 训练循环
    print(f"\n🔥 Training Start (Target: {config.N_EPOCHS} Epochs)")
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Time':<10} | {'Train Loss':<12} | {'Val Loss':<12} | {'Status'}")
    print("-" * 95)
    
    best_valid_loss = float('inf')
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        for epoch in range(config.N_EPOCHS):
            start_time = time.time()
            
            train_loss = train(model, train_loader, optimizer, criterion, config.CLIP, device)
            valid_loss = evaluate(model, val_loader, criterion, device)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            status_msg = []
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                status_msg.append("🏆 Best!")
            
            # Checkpoint (每 5 轮或每轮)
            if (epoch + 1) % 5 == 0 or (epoch + 1) == config.N_EPOCHS:
                ep_path = os.path.join(checkpoint_dir, f"ep_{epoch+1:03d}_vl_{valid_loss:.4f}.pth")
                torch.save(model.state_dict(), ep_path)

            print(f"{epoch+1:<6} | {int(epoch_mins)}m {int(epoch_secs)}s    | {train_loss:.4f}       | {valid_loss:.4f}       | {' '.join(status_msg)}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        
    finally:
        print(f"✅ Training Finished. Best Val Loss: {best_valid_loss:.4f}")