import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# ================= 辅助函数 =================
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
        
        # output: [batch_size, trg_len, output_dim]
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
    # 0. 准备设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # 清理显存
    print(f"🚀 正在使用设备: {device}")
    
    # 1. 加载字典 (使用 config 里的路径)
    vocab = load_vocab(config.VOCAB_PATH)
    harmony_stoi = vocab['harmony']
    INPUT_DIM = len(vocab['melody'])
    OUTPUT_DIM = len(harmony_stoi)

    print("📦 正在加载数据...")
    # 👇 修改点：使用 config.DATASET_PATH
    dataset = MusicDataset(config.DATASET_PATH)
    # 👇 修改点：使用 config.BATCH_SIZE
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    print(f"✅ 数据加载完毕，共 {len(dataset)} 条训练切片")
    
    # 2. 初始化模型 (使用 config 里的参数)
    enc = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    # 👇 修改点：使用 config.LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ==========================================
    # 🌟【核心】防止 AI 偷懒的权重设置 (你之前漏了这段)
    # ==========================================
    print("⚖️ 正在配置损失函数权重...")
    loss_weights = torch.ones(OUTPUT_DIM).to(device)
    
    # 降低 "_" (HOLD) 的权重，逼迫模型去猜和弦变化
    hold_id = harmony_stoi.get("_")
    if hold_id is not None:
        loss_weights[hold_id] = 0.025 
        print(f"   -> 已降低 '_' (ID: {hold_id}) 的权重，防止模型偷懒。")
    
    # 将权重传给 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)
    # ==========================================

    print(f"\n🔥 开始训练！(目标轮数: {config.N_EPOCHS})")
    print("-" * 65)
    print(f"{'Epoch':<10} | {'Time':<10} | {'Loss':<10} | {'Status'}")
    print("-" * 65)
    
    best_loss = float('inf')
    
    # 定义备用模型保存路径 (基于 config.MODEL_DIR)
    last_model_path = os.path.join(config.MODEL_DIR, 'last_checkpoint.pth')

    try:
        # 👇 修改点：使用 config.N_EPOCHS
        for epoch in range(config.N_EPOCHS):
            start_time = time.time()
            
            # 👇 修改点：使用 config.CLIP
            train_loss = train(model, dataloader, optimizer, criterion, config.CLIP)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            save_msg = ""
            if train_loss < best_loss:
                best_loss = train_loss
                # 确保目录存在
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

        # 保存最后的中断点
        torch.save(model.state_dict(), last_model_path)
        print(f"💾 最后状态已保存至: {last_model_path}")