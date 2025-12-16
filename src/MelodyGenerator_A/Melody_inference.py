import os
import torch
import music21
from datetime import datetime
import re

# 导入自定义模块
from Melody_model import MelodyGPT, GPTConfig
from Melody_tokenizer import MelodyTokenizer

# ================= 配置区 =================
# 硬件与路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.join('data', 'processed')
VOCAB_PATH = os.path.join(BASE_DIR, 'Melody_vocab.json')
CKPT_PATH = os.path.join('src', 'MelodyGenerator_A', 'Melody_ckpt.pt')
OUTPUT_DIR = os.path.join('outputs')

# 模型参数 (必须严格匹配 Melody_train.py)
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384

# 生成参数
MAX_NEW_TOKENS = 500  # 生成长度
TEMPERATURE = 0.8     # 0.8 平衡创造性与准确性 (过拟合验证可设为 1.0)
TOP_K = 40            # 截断采样，防止生成极低概率的乱码

# =========================================

def extract_first_song(full_text):
    """
    逻辑清洗：从生成的长文本中截取第一首完整的ABC曲目
    """
    # 1. 移除特殊 Token
    text = full_text.replace('<|endoftext|>', '')

    # 2. 尝试按 X: 分割 (ABC标准曲目分隔符)
    # 如果生成结果包含多首曲子 (例如 "X:1... X:2..."), 我们只取 X:1 部分
    # 注意：我们的Seed不包含 X:，所以开头可能是 "M:4/4..."
    
    # 策略：如果文本中出现了第二次 "X:" (或第一次出现在中间)，则截断
    # 找到所有的 "X:" 位置
    x_indices = [m.start() for m in re.finditer(r'\nX:', text)]
    
    if len(x_indices) > 0:
        # 如果开头没有 X: (因为Seed没写)，但中间出现了 X: (第二首)，截断到第一个 X: 之前
        # 如果开头就是 X:，那么截断到第二个 X: 之前
        if x_indices[0] == 0 or text.strip().startswith('X:'):
             if len(x_indices) > 1:
                 text = text[:x_indices[1]] # 取第一首，丢弃第二首开头之后的内容
        else:
            # 开头没有 X: (是 Seed 直接开始的)，后面出现了 X: (第二首)
            text = text[:x_indices[0]]
            
    return text.strip()

def save_midi(abc_str, filename):
    """
    修正版：增强容错性的 MIDI 转换
    """
    try:
        # 1. 确保有 X: 索引 (Music21 强制要求)
        if not re.search(r'\nX:\s*\d+', "\n" + abc_str) and not abc_str.lstrip().startswith("X:"):
            # 如果没有 X: 标记，手动在最前面添加
            abc_str = "X:1\n" + abc_str

        print(f"[Converter] Parsing ABC content (Length: {len(abc_str)} chars)...")
        
        # 2. 解析
        # 使用 forceSource=True 有时能绕过某些缓存错误，但在 parse 方法中通常是自动的
        s = music21.converter.parse(abc_str, format='abc')
        
        # 3. 处理 Opus (多首曲子) vs Score (单首)
        # 即使我们截断了，Music21 有时仍会返回 Opus 对象
        if isinstance(s, music21.stream.Opus):
            # 如果是作品集，取第一首
            stream_to_write = s[0]
        else:
            stream_to_write = s
            
        # 4. 写入
        stream_to_write.write('midi', fp=filename)
        print(f"[Converter] Success! MIDI saved to: {filename}")
        return True
        
    except music21.abcFormat.ABCParsingException as e:
        print(f"[Error] ABC Syntax Error: {e}")
        return False
    except Exception as e:
        # 捕获类似 'TimeSignature already found' 的流错误
        print(f"[Error] Music21 Logic Error: {e}")
        print(">> Suggestion: The generated ABC might contain conflicting headers.")
        return False
    
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[System] Device: {DEVICE}")

    # 1. 加载 Tokenizer
    tokenizer = MelodyTokenizer()
    try:
        tokenizer.load_vocab(VOCAB_PATH)
    except FileNotFoundError:
        print(f"[Error] Vocab file not found at {VOCAB_PATH}. Cannot decode output.")
        return

    # 2. 初始化模型并加载权重
    print(f"[Inference] Loading model from {CKPT_PATH}...")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=0.0 # 推理时不需要 Dropout
    )
    model = MelodyGPT(config)
    
    # 加载 State Dict
    try:
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval() # 切换到评估模式
        print("[Inference] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load model checkpoint: {e}")
        return

    # 3. 定义种子 (Seed)
    # 经典 Nottingham 风格开头: 4/4拍, G大调 (数据集中最常见的组合)
    # 注意：必须以换行符结尾，模拟 dataset.txt 的格式
    seed_text = "M:4/4\nK:G\n"
    print(f"[Inference] Seed Prompt: {repr(seed_text)}")

    # 编码种子
    start_ids = tokenizer.encode(seed_text)
    x = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    # 4. 生成 (Generation)
    print(f"[Inference] Generating {MAX_NEW_TOKENS} tokens...")
    with torch.no_grad():
        # 调用模型的 generate 方法
        y = model.generate(x, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)

    # 解码
    full_generated_text = tokenizer.decode(y[0].tolist())
    
    # ================= 关键修改点 =================
    # 5. 后处理
    print("-" * 30)
    print("[Inference] Post-processing generated text...")
    
    # 提取第一首有效曲目
    clean_abc = extract_first_song(full_generated_text)
    
    # 保存 ABC 文本 (保留清洗后的版本)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    abc_filename = os.path.join(OUTPUT_DIR, f"melody_{timestamp}.abc")
    midi_filename = os.path.join(OUTPUT_DIR, f"melody_{timestamp}.mid")

    with open(abc_filename, 'w', encoding='utf-8') as f:
        f.write(clean_abc)
    print(f"[Inference] Cleaned ABC saved to {abc_filename}")

    # 转换为 MIDI
    save_midi(clean_abc, midi_filename)

    # 打印结果
    print("-" * 30)
    print("Final ABC Content:")
    print(clean_abc)
    print("-" * 30)

if __name__ == "__main__":
    main()