import os
import torch
import random
import time
from datetime import datetime

# 导入同级目录下的模块
from Melody_model import MelodyGPT, GPTConfig
from Melody_tokenizer import MelodyTokenizer
# 复用 Melody_inference 中的核心函数
from Melody_inference import generate_melody_with_params, save_midi

# ================= 配置区 =================
# 生成数量
NUM_SONGS = 100

# 硬件配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径配置 (假设脚本在 src/MelodyGenerator_A/ 下运行)
# 我们需要往上退两级找到 dataset 和 samples 目录
BASE_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
VOCAB_PATH = os.path.join(BASE_PROJECT_DIR, "data", "processed", "Melody_vocab.json")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "Melody_ckpt.pt")
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "samples", "test_melody")

# 模型参数 (需与训练时一致)
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384

# 随机池
KEYS = ['C', 'G', 'D', 'A', 'F', 'Bb', 'Eb', 'Am', 'Em', 'Dm', 'Gm']
METERS = ['4/4', '3/4', '2/4', '6/8', '12/8']
# =========================================

def main():
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Batch] Output directory ready: {OUTPUT_DIR}")
    print(f"[Batch] Target: {NUM_SONGS} songs")
    print(f"[System] Device: {DEVICE}")

    # 2. 加载 Tokenizer
    print("[Batch] Loading Tokenizer...")
    tokenizer = MelodyTokenizer()
    if os.path.exists(VOCAB_PATH):
        tokenizer.load_vocab(VOCAB_PATH)
    else:
        print(f"[Error] Vocab not found at {VOCAB_PATH}")
        return

    # 3. 加载模型 (只加载一次！)
    print(f"[Batch] Loading Model from {CKPT_PATH}...")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=0.0
    )
    model = MelodyGPT(config)
    
    try:
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("[Batch] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # 4. 批量生成循环
    print("-" * 40)
    print(f"[Batch] Starting generation loop...")
    
    start_time = time.time()
    success_count = 0

    for i in range(1, NUM_SONGS + 1):
        # 随机选择参数
        key = random.choice(KEYS)
        meter = random.choice(METERS)
        # 稍微随机化温度，增加多样性 (0.75 ~ 0.95)
        temp = round(random.uniform(0.75, 0.95), 2)

        print(f"\n[{i}/{NUM_SONGS}] Generating (Key: {key}, Meter: {meter}, Temp: {temp})...")

        try:
            # 调用推理逻辑
            abc_content = generate_melody_with_params(
                model=model,
                tokenizer=tokenizer,
                key=key,
                meter=meter,
                max_tokens=600, # 稍微加大一点，防止长拍号被截断
                temp=temp
            )

            # 文件命名：序号_调性_拍号_时间戳
            timestamp = datetime.now().strftime("%H%M%S")
            # 处理拍号文件名 (把 4/4 变成 4-4)
            safe_meter = meter.replace('/', '-')
            filename_base = f"song_{i:03d}_{key}_{safe_meter}_{timestamp}"
            output_path_base = os.path.join(OUTPUT_DIR, filename_base)

            # 保存 ABC
            with open(output_path_base + ".abc", "w", encoding="utf-8") as f:
                f.write(abc_content)

            # 保存 MIDI (使用 inference 中定义的 save_midi)
            midi_success = save_midi(abc_content, output_path_base + ".mid")
            
            if midi_success:
                success_count += 1
            
            # 可选：清理一下 GPU 缓存（如果是非常大的模型需要，这里其实不太需要，但为了保险）
            if i % 20 == 0 and DEVICE == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Error] Failed at song {i}: {e}")
            continue

    total_time = time.time() - start_time
    print("-" * 40)
    print(f"[Batch] Complete!")
    print(f"[Batch] Success: {success_count}/{NUM_SONGS}")
    print(f"[Batch] Time elapsed: {total_time:.2f}s (Avg: {total_time/NUM_SONGS:.2f}s/song)")
    print(f"[Batch] Files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()