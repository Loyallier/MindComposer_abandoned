import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import time
import sys
# 强行把根目录加入 sys.path
# 1. 获取所在的目录 (src/ChordGenerator_A)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 往上跳两级，找到根目录 (src -> Root)
project_root = os.path.dirname(os.path.dirname(current_dir))
# 3. 加入路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"🔧 已将根目录挂载: {project_root}")

# ✅ 统一引用：全部基于 src 根路径
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
    print(f"🔒 随机种子已锁定为: {seed}")

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=config.INIT_WEIGHT_STD)
        else:
            nn.init.constant_(param.data, 0)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg, lengths) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, lengths, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 环境初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"🚀 正在使用设备: {device}")
    set_seed(config.SEED)
    
    # 2. 加载数据
    vocab = utils.load_vocab(config.VOCAB_PATH)
    harmony_stoi = vocab['harmony']
    INPUT_DIM = len(vocab['melody'])
    OUTPUT_DIM = len(harmony_stoi)

    print("📦 正在加载数据...")
    dataset = MusicDataset(config.DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    print(f"✅ 数据加载完毕，共 {len(dataset)} 条样本")
    
    # 3. 初始化模型
    enc = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # 先应用默认随机初始化
    model.apply(init_weights)
    
    # 🌟【核心修改】根据 config 决定是否加载旧模型
    if config.RESUME_TRAINING:
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"🔄 [继续训练模式] 发现旧模型，正在加载: {config.MODEL_SAVE_PATH}")
            try:
                # weights_only=True 保证安全
                model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
                print("✅ 旧模型加载成功，将在此基础上继续微调！")
            except Exception as e:
                print(f"⚠️ 旧模型加载失败 ({e})，将回退到从头训练。")
        else:
            print(f"✨ [继续训练模式] 但未找到文件 {config.MODEL_SAVE_PATH}，将从头开始。")
    else:
        print("🆕 [全新训练模式] 配置已禁用 RESUME，强制从头开始训练。")

    print("🧠 模型准备完毕")

    # 4. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. 配置权重
    print("⚖️ 配置 Loss 权重...")
    loss_weights = torch.ones(OUTPUT_DIM).to(device)
    hold_id = harmony_stoi.get("_")
    if hold_id is not None:
        loss_weights[hold_id] = config.HOLD_LOSS_WEIGHT
        print(f"   -> HOLD权重 = {config.HOLD_LOSS_WEIGHT}")
    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)

    # 6. 训练循环
    print(f"\n🔥 开始训练！(目标: {config.N_EPOCHS} Epochs)")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Time':<10} | {'Loss':<8} | {'Status'}")
    print("-" * 75)
    
    best_loss = float('inf')
    
    # 如果是继续训练，最好把 best_loss 设为一个较大的数，或者尝试去读之前的 log
    # 这里为了简单，我们还是让它重新刷 best_loss，只有比当前更低才保存
    
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_model_path = os.path.join(config.MODEL_DIR, 'last_checkpoint.pth')

    try:
        for epoch in range(config.N_EPOCHS):
            start_time = time.time()
            
            train_loss = train(model, dataloader, optimizer, criterion, config.CLIP)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            status_msg = []
            
            # 💾 存档: 每轮都存 (epoch_001.pth)
            epoch_save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), epoch_save_path)

            if train_loss < best_loss:
                best_loss = train_loss
                os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                status_msg.append("🏆Best")
                
            print(f"{epoch+1:<8} | {int(epoch_mins)}m {int(epoch_secs)}s    | {train_loss:.4f}   | {' '.join(status_msg)}")

    except KeyboardInterrupt:
        print("\n🛑 用户中断训练。")
        
    finally:
        print("\n📊 训练总结:")
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"✅ 最佳模型: {config.MODEL_SAVE_PATH} (Loss: {best_loss:.4f})")
        
        torch.save(model.state_dict(), last_model_path)
        print(f"💾 中断现场: {last_model_path}")