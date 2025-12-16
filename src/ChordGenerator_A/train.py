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

# ✅ [V3.6 新增] 准确率计算函数
def calculate_accuracy(output, trg):
    """
    计算准确率 (忽略 Padding 和 BAR)
    output: [batch_size * seq_len, output_dim]
    trg:    [batch_size * seq_len]
    """
    # 获取预测的最大概率索引
    preds = output.argmax(dim=1, keepdim=True) 
    
    # trg 必须和 preds 形状一致
    correct = preds.eq(trg.view_as(preds))
    
    # 忽略 PAD (0) 和 BAR (假设BAR不计入准确率，或者计入均可)
    # 我们这里只忽略 PAD
    non_pad_mask = trg.ne(config.PAD_IDX)
    
    # 最终计算
    correct_non_pad = correct.squeeze(1) & non_pad_mask
    acc = correct_non_pad.sum().float() / non_pad_mask.sum().float()
    
    return acc

# ✅ [V3.4] 调度函数放回此处，确保独立运行
def get_current_tf_ratio(epoch):
    """
    计算当前 Epoch 的 Teacher Forcing Ratio
    """
    if epoch < config.TF_DECAY_EPOCHS:
        decay_step = (
            config.TF_START_RATIO - config.TF_END_RATIO
        ) / config.TF_DECAY_EPOCHS
        current_ratio = config.TF_START_RATIO - (epoch * decay_step)
        return max(config.TF_END_RATIO, current_ratio)
    else:
        return config.TF_END_RATIO


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 [System] Seed locked: {seed}")


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=config.INIT_WEIGHT_STD)
        else:
            nn.init.constant_(param.data, 0)


def train(model, iterator, optimizer, criterion, clip, device, tf_ratio):
    """
    V3.1 训练循环: 必须处理 pos (位置编码)
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0 # ✅ V3.6 新增

    # Unpack 4 items: src, pos, trg, lengths
    for i, (src, pos, trg, lengths) in enumerate(iterator):
        src = src.to(device)
        pos = pos.to(device)  # ✅ [V3.1] 位置数据上微
        trg = trg.to(device)

        optimizer.zero_grad()

        # ✅ [V3.1] 传入 pos
        # ✅ 将 tf_ratio 传给 model.forward
        output = model(src, pos, trg, lengths, teacher_forcing_ratio=tf_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        # ✅ [V3.6] 计算准确率
        acc = calculate_accuracy(output, trg)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item() # [V3.6] ✅ 累加

    return epoch_loss / len(iterator), epoch_acc / len(iterator)  # [V3.6] ✅ 返回平均准确率


def evaluate(model, iterator, criterion, device):
    """
    V3.6 验证循环 (修复返回值)
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0 # ✅ V3.6 新增

    with torch.no_grad():
        for i, (src, pos, trg, lengths) in enumerate(iterator):
            src = src.to(device)
            pos = pos.to(device)
            trg = trg.to(device)

            output = model(src, pos, trg, lengths, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            acc = calculate_accuracy(output, trg) # ✅ 计算
            
            epoch_loss += loss.item()
            epoch_acc += acc.item() # ✅ 累加

    # 🚨 [关键修复] 必须返回两个值，匹配主程序的解包
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # --- A. 初始化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"🚀 [Device] Using: {device}")
    set_seed(config.SEED)

    # --- B. 打印配置 ---
    print("\n" + "=" * 50)
    print("🔧 V3.5 Configuration Summary")
    print("=" * 50)
    print(f"   • Hidden Dim     : {config.HIDDEN_DIM} (Should be 256 for V3.5)")
    print(f"   • Dropout        : {config.DROPOUT}")
    print(f"   • Learning Rate  : {config.LEARNING_RATE}")
    print(f"   • Weight Decay   : {config.WEIGHT_DECAY}")
    print(f"   • Label Smoothing: {config.LABEL_SMOOTHING}")
    print(
        f"   • TF Strategy    : {config.TF_START_RATIO}->{config.TF_END_RATIO} / {config.TF_DECAY_EPOCHS} eps"
    )
    print("=" * 50 + "\n")

    # --- C. 加载数据 ---
    print("📦 [Data] Loading Vocab...")
    vocab = utils.load_vocab(config.VOCAB_PATH)
    harmony_stoi = vocab["harmony"]
    INPUT_DIM = len(vocab["melody"])
    OUTPUT_DIM = len(harmony_stoi)
    POS_DIM = config.POS_VOCAB_SIZE
    POS_EMB_DIM = config.POS_EMB_DIM

    if not os.path.exists(config.TRAIN_DATASET_PATH):
        print(f"❌ 找不到数据文件，请先运行 tokenize_data.py")
        sys.exit(1)

    # V3.3+ 开启增强 V3.6 训练时关闭增强
    train_dataset = MusicDataset(config.TRAIN_DATASET_PATH, augment=False)
    val_dataset = MusicDataset(config.VAL_DATASET_PATH, augment=False)

    print(f"📊 数据加载完成:")
    print(f"   - Train: {len(train_dataset)} (Noise Off)")
    print(f"   - Val:   {len(val_dataset)} (Clean)")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    # --- D. 初始化模型 ---
    enc = Encoder(
        INPUT_DIM,
        config.ENC_EMB_DIM,
        config.HIDDEN_DIM,
        config.DROPOUT,
        POS_DIM,
        POS_EMB_DIM,
    )
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HIDDEN_DIM, config.DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    # --- E. 加载策略 ---
    if config.RESUME_TRAINING:
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"🔄 [Resume] Loading: {config.MODEL_SAVE_PATH}")
            try:
                state_dict = torch.load(
                    config.MODEL_SAVE_PATH, map_location=device, weights_only=True
                )
                model.load_state_dict(state_dict)
                print("✅ Weights loaded.")
            except Exception as e:
                print(f"⚠️ Load failed ({e}). Starting fresh.")
        else:
            print(f"✨ [Resume] No file found. Starting fresh.")
    else:
        print("🆕 [Fresh] Config forces fresh start.")

    # --- F. 优化器 ---
    print(f"🔧 Optimizer: AdamW (Weight Decay: {config.WEIGHT_DECAY})")
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    loss_weights = torch.ones(OUTPUT_DIM).to(device)
    hold_id = harmony_stoi.get("_")
    if hold_id is not None:
        loss_weights[hold_id] = config.HOLD_LOSS_WEIGHT
    bar_id = harmony_stoi.get(config.BAR_TOKEN)
    if bar_id is not None:
        loss_weights[bar_id] = 0.0

    criterion = nn.CrossEntropyLoss(
        ignore_index=0, weight=loss_weights, label_smoothing=config.LABEL_SMOOTHING
    )

    # --- G. 训练循环 ---
    # 获取起始轮次 (默认0)
    start_epoch = getattr(config, "START_EPOCH", 0)

    print(f"\n🔥 Training Start (Epoch {start_epoch + 1} -> {config.N_EPOCHS})")
    print("-" * 115)
    print(
        f"{'Epoch':<6} | {'Time':<10} | {'Train Loss':<12} | {'Val Loss':<12} | {'TF Ratio':<10} | {'Status'}"
    )
    print("-" * 115)

    best_valid_loss = float("inf")

    # Checkpoint 目录 (自动包含版本号或时间戳是个好习惯，这里沿用您的路径)
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints/V3.6")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, config.N_EPOCHS):
            start_time = time.time()

            # 1. 计算 Ratio
            current_tf_ratio = get_current_tf_ratio(epoch)

            # 2. 训练
            train_loss, train_acc = train(
                model,
                train_loader,
                optimizer,
                criterion,
                config.CLIP,
                device,
                tf_ratio=current_tf_ratio,
            )

            # 3. 验证
            valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            status_msg = []

            # 保存 Best Model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                status_msg.append("🏆 Best!")

            # 保存 Checkpoint
            if (epoch + 1) % 1 == 0 or (epoch + 1) == config.N_EPOCHS:
                cp_name = f"epoch_{epoch + 1:03d}_loss_{valid_loss:.4f}.pth"
                cp_path = os.path.join(checkpoint_dir, cp_name)
                torch.save(model.state_dict(), cp_path)
                status_msg.append(f"💾 Saved")

            # ✅ [优化] 打印格式包含准确率
            # 格式: Train Loss (Acc) | Val Loss (Acc)
            print(
                f"{epoch + 1:<6} | {int(epoch_mins)}m {int(epoch_secs)}s    | "
                f"{train_loss:.4f} ({train_acc:.2%}) | "
                f"{valid_loss:.4f} ({valid_acc:.2%}) | "
                f"{current_tf_ratio:.2f}       | {' '.join(status_msg)}"
            )

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

    finally:
        print(f"✅ Training Finished. Best Val Loss: {best_valid_loss:.4f}")
