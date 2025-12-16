import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入自定义模块
from Melody_model import MelodyGPT, GPTConfig
from Melody_dataset import create_dataloaders
from Melody_tokenizer import MelodyTokenizer

# ================= 训练超参数配置 =================
# 硬件与路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.join('data', 'processed', 'dataset.txt')
VOCAB_PATH = os.path.join('data', 'processed', 'Melody_vocab.json')
CKPT_PATH = os.path.join('src', 'MelodyGenerator_A', 'Melody_ckpt.pt')

# 模型参数 (需与 Melody_model.py 中的设计匹配)
BLOCK_SIZE = 256      # 上下文窗口
BATCH_SIZE = 64       # RTX 4060 8GB 可轻松承载 64-128
N_LAYER = 6           # 层数
N_HEAD = 6            # 注意力头数
N_EMBD = 384          # 嵌入维度
DROPOUT = 0.0         # 0.0 以强制过拟合/记忆

# 优化器参数
LEARNING_RATE = 5e-4  # 较高的学习率加速收敛
MAX_ITERS = 5000      # 训练迭代次数
EVAL_INTERVAL = 500   # 每隔多少步评估一次
WARMUP_STEPS = 100    # 预热步数

# =================================================

def estimate_loss(model, dataloaders, eval_iters=50):
    """
    评估函数：计算训练集和验证集的平均Loss
    """
    out = {}
    model.eval()
    for split, loader in dataloaders.items():
        losses = torch.zeros(eval_iters)
        # 获取一个迭代器
        loader_iter = iter(loader)
        for k in range(eval_iters):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                X, Y = next(loader_iter)
                
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    print(f"[System] Device: {DEVICE}")
    torch.manual_seed(1337) # 复现性

    # 1. 准备数据
    print("[Train] Loading Data...")
    # 注意：过拟合实验中，我们主要关注 train loss。
    # 为了逻辑严谨，我们仍然划分验证集，但可以预期 val loss 可能会上升 (如果是纯过拟合)。
    train_loader, val_loader, tokenizer = create_dataloaders(
        DATA_PATH, VOCAB_PATH, 
        block_size=BLOCK_SIZE, 
        batch_size=BATCH_SIZE
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    vocab_size = tokenizer.vocab_size
    print(f"[Train] Vocab Size: {vocab_size}")

    # 2. 初始化模型
    print("[Train] Initializing Model...")
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT
    )
    model = MelodyGPT(config)
    model.to(DEVICE)
    
    # 3. 优化器 (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    print(f"[Train] Starting training for {MAX_ITERS} iterations...")
    iter_num = 0
    t0 = time.time()
    
    # 创建无限循环的迭代器
    train_iter = iter(train_loader)

    while iter_num < MAX_ITERS:
        # 获取Batch
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        # 评估阶段
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, dataloaders)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 前向传播 (混合精度)
        # RTX 4060 支持 AMP，能显著提速
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(X, Y)

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iter_num += 1

    # 5. 结束训练与保存
    t1 = time.time()
    print(f"[Train] Finished in {(t1-t0):.2f} seconds.")
    
    print(f"[Train] Saving model to {CKPT_PATH}...")
    torch.save(model.state_dict(), CKPT_PATH)

    # 6. 生成测试 (Sanity Check)
    print("-" * 30)
    print("[Train] Generating sample melody (Overfit Check)...")
    model.eval()
    
    # 这里的Prompt可以是一个典型的ABC文件头，看看它能否续写
    # M:4/4
    # K:G
    start_str = "\nM:4/4\nK:G\n" 
    start_ids = tokenizer.encode(start_str)
    context = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=200, temperature=0.8)
        
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    print("-" * 30)

if __name__ == "__main__":
    main()