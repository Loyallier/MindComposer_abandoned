import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time

# 导入你自己写的模块
from model import Encoder, Decoder, Seq2Seq
from dataset import MusicDataset, collate_fn

# ================= 1. 配置区域 (Configuration) =================
# 路径设置 (使用你确认过的路径)
# 注意：在字符串前加 r 可以防止反斜杠被转义
DATA_PATH = r"data\processed\dataset_encoded.json"
VOCAB_PATH = r"data\processed\vocab.json"
MODEL_SAVE_PATH = "best_model.pth"

# 超参数 (Hyperparameters)
BATCH_SIZE = 1         # 一次训练几首歌 (显存不够就改小，比如 8 或 4)
LEARNING_RATE = 0.001   # 学习率
N_EPOCHS = 20          # 训练总轮数 (建议先设 20 跑跑看，效果好再加)
CLIP = 1                # 梯度裁剪 (防止 LSTM 梯度爆炸)

# 模型参数 (尽量保持与 check_model.py 一致)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.5

# 设备配置 (优先使用 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 辅助函数 =================
def load_vocab_dims(vocab_path):
    """自动读取字典大小，避免手动填错"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        input_dim = len(vocab['melody'])
        output_dim = len(vocab['harmony'])
        print(f"📖 字典加载成功 | 旋律词表: {input_dim}, 和弦词表: {output_dim}")
        return input_dim, output_dim

def init_weights(m):
    """初始化权重，帮助模型更快收敛"""
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg, lengths) in enumerate(iterator):
        # 放到 GPU/CPU 上
        src = src.to(device)
        trg = trg.to(device)
        # lengths 必须在 CPU 上用于 pack_padded_sequence，但在 Dataset 里已经是 CPU 了
        
        optimizer.zero_grad()
        
        # Forward Pass
        # trg 用作 teacher forcing 的输入
        output = model(src, trg, lengths)
        
        # Output: [batch_size, trg_len, output_dim]
        # Target: [batch_size, trg_len]
        
        # 切掉 <SOS> token，因为我们需要预测的是 <SOS> 之后的内容
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        # 计算 Loss (会自动忽略 pad token)
        loss = criterion(output, trg)
        
        # Backward Pass
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# ================= 3. 主程序 =================
if __name__ == "__main__":
    print(f"🚀 正在使用设备: {device}")
    
    # 1. 自动获取维度
    if not os.path.exists(VOCAB_PATH):
        print(f"❌ 错误：找不到文件 {VOCAB_PATH}")
        exit()
    INPUT_DIM, OUTPUT_DIM = load_vocab_dims(VOCAB_PATH)

    # 2. 准备数据
    print("📦正在加载数据...")
    dataset = MusicDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    print(f"✅ 数据加载完毕，共 {len(dataset)} 首乐曲")

    # 3. 初始化模型
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # 初始化权重
    model.apply(init_weights)
    print("🧠 模型初始化完成")

    # 4. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 关键：ignore_index=0 告诉模型 "不要管 PAD (填充位)，预测错了也不扣分"
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 5. 开始训练循环
    print("\n🔥 开始训练！(按 Ctrl+C 可以强行停止，但尽量等它跑完)")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, dataloader, optimizer, criterion, CLIP)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # 如果 Loss 创新低，就保存模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_msg = "💾 模型已保存!"
        else:
            save_msg = ""
            
        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s | Loss: {train_loss:.3f} | {save_msg}')

    print("-" * 60)
    print(f"🎉 训练结束！最佳模型已保存为: {MODEL_SAVE_PATH}")
    print("👉 下一步：写 predict.py 来生成真正的音乐！")