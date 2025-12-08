import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np # 需要安装 numpy (pip install numpy)
import os
import time

# 导入配置和模块
try:
    from src import config
    from src.model import Encoder, Decoder, Seq2Seq
    from src.dataset import MusicDataset, collate_fn
    from src.utils import load_vocab
except ImportError:
    import config
    from model import Encoder, Decoder, Seq2Seq
    from dataset import MusicDataset, collate_fn
    from utils import load_vocab

# ================= 工具函数 =================

def set_seed(seed):
    """
    固定所有随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 随机种子已锁定为: {seed}")

def init_weights(m):
    """
    初始化模型权重
    参数 std 从 config 读取
    """
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
        
        # 🌟 修改点：显式传入 teacher_forcing_ratio
        output = model(src, trg, lengths, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO)
        
        # 展平输出以计算 Loss
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
    
    # 🌟 修改点：设置随机种子
    set_seed(config.SEED)
    
    # 2. 加载数据与字典
    vocab = load_vocab(config.VOCAB_PATH)
    harmony_stoi = vocab['harmony']
    INPUT_DIM = len(vocab['melody'])
    OUTPUT_DIM = len(harmony_stoi)

    print("📦 正在加载数据...")
    dataset = MusicDataset(config.DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    print(f"✅ 数据加载完毕，共 {len(dataset)} 条训练样本")
    
    # 3. 初始化模型
    enc = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # 初始化权重
    model.apply(init_weights)
    print("🧠 模型初始化完成")

    # 4. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. 定义损失函数权重 (Anti-Lazy Strategy)
    print("⚖️ 正在配置损失函数权重...")
    loss_weights = torch.ones(OUTPUT_DIM).to(device)
    
    # 获取 "_" (HOLD) 的 ID
    hold_id = harmony_stoi.get("_")
    
    if hold_id is not None:
        # 🌟 修改点：权重值从 config 读取
        loss_weights[hold_id] = config.HOLD_LOSS_WEIGHT
        print(f"   -> 已应用配置: HOLD权重 = {config.HOLD_LOSS_WEIGHT}")
    else:
        print("⚠️ 警告: 字典中未找到 '_'，无法应用防偷懒策略！")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)

    # 6. 开始训练循环
    print(f"\n🔥 开始训练！(目标: {config.N_EPOCHS} Epochs, LR: {config.LEARNING_RATE})")
    print("-" * 65)
    print(f"{'Epoch':<10} | {'Time':<10} | {'Loss':<10} | {'Status'}")
    print("-" * 65)
    
    best_loss = float('inf')
    last_model_path = os.path.join(config.MODEL_DIR, 'last_checkpoint.pth')

    try:
        for epoch in range(config.N_EPOCHS):
            start_time = time.time()
            
            train_loss = train(model, dataloader, optimizer, criterion, config.CLIP)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            save_msg = ""
            if train_loss < best_loss:
                best_loss = train_loss
                os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                save_msg = "💾 Best Saved"
                
            print(f"{epoch+1:<10} | {int(epoch_mins)}m {int(epoch_secs)}s    | {train_loss:.4f}     | {save_msg}")

    except KeyboardInterrupt:
        print("\n🛑 用户中断训练。")
        
    finally:
        print("\n📊 训练总结:")
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"✅ 最佳模型: {config.MODEL_SAVE_PATH} (Loss: {best_loss:.4f})")
        else:
            print("⚠️ 未保存最佳模型。")

        # 保存最后状态
        torch.save(model.state_dict(), last_model_path)
        print(f"💾 最终状态: {last_model_path}")