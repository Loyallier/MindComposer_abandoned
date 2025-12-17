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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.join("data", "processed")
VOCAB_PATH = os.path.join(BASE_DIR, "Melody_vocab.json")
CKPT_PATH = os.path.join("src", "MelodyGenerator_A", "Melody_ckpt.pt")
OUTPUT_DIR = os.path.join("samples", "Melody Outputs")

# 模型参数 (必须严格匹配 Melody_train.py
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384

# 生成参数
MAX_NEW_TOKENS = 500  # 生成长度
TEMPERATURE = 0.8  # 0.8 平衡创造性与准确性 (过拟合验证可设为 1.0)
TOP_K = 5  # 截断采样，防止生成极低概率的乱码

# 用户自定义区
TARGET_KEY = "C"  # 目标调性 (例如 C, G, D, Am)
TARGET_METER = "3/4"  # 目标拍号 (例如 4/4, 3/4, 6/8)


# =========================================
def extract_first_song(full_text):
    """
    逻辑升级：严格截断。
    一旦在正文之后检测到新的 Header (M:, K:, X:, T:) 或 纯数字索引，立即终止。
    """
    lines = full_text.split("\n")
    valid_lines = []

    # 状态标记：是否已经读取过了核心Header (K:)
    # 因为Seed通常包含 M: 和 K:，我们需要允许开头出现这两个
    seen_key_signature = False

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 1. 检查是否是纯数字 (例如 '48', '50' 这种数据集索引)
        # 如果出现在第一行之后，且看起来像索引，截断
        if i > 0 and re.match(r"^\d+$", line):
            # print(f"[Debug] Detected Index '{line}', truncating.")
            break

        # 2. 检查是否是新的 Header
        # 如果我们已经过了 K: 阶段（开始进入旋律了），却又遇到了 M: 或 K: 或 X:，说明是第二首
        is_header = re.match(r"^[A-Z]:", line)

        if seen_key_signature:
            # 进入旋律部分后，如果遇到 M: K: X: T: 等，视为新歌开始
            if is_header:
                # print(f"[Debug] Detected new Header '{line}', truncating.")
                break

        # 3. 收集行
        valid_lines.append(line)

        # 4. 更新状态
        # 一旦读到 K:，标记 Header 阶段结束（或即将结束），后续非 Header 行视为旋律
        if line.startswith("K:"):
            seen_key_signature = True

    return "\n".join(valid_lines)


def generate_melody_with_params(
    model, tokenizer, key="G", meter="4/4", max_tokens=500, temp=0.8
):
    """
    封装函数：根据指定调性和节拍生成旋律
    """
    # 1. 构建种子 (Seed)
    # 格式严格遵循清洗逻辑：换行符分隔
    seed_text = f"M:{meter}\nK:{key}\n"
    print(f"[Inference] Generating with Seed: {repr(seed_text)}")

    # 2. 编码
    start_ids = tokenizer.encode(seed_text)
    x = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    # 3. 生成
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=40)

    # 4. 解码
    full_text = tokenizer.decode(y[0].tolist())

    # 5. 截取单曲
    clean_abc = extract_first_song(full_text)

    # 6. 兜底：如果截取后丢失了 X:1，补上 (Music21 需要)
    if "X:" not in clean_abc:
        clean_abc = "X:1\n" + clean_abc

    return clean_abc


def save_midi(abc_str, filename):
    """
    修正版：增强容错性的 MIDI 转换
    """
    try:
        # 1. 确保有 X: 索引 (Music21 强制要求)
        if not re.search(
            r"\nX:\s*\d+", "\n" + abc_str
        ) and not abc_str.lstrip().startswith("X:"):
            # 如果没有 X: 标记，手动在最前面添加
            abc_str = "X:1\n" + abc_str

        print(f"[Converter] Parsing ABC content (Length: {len(abc_str)} chars)...")

        # 2. 解析
        # 使用 forceSource=True 有时能绕过某些缓存错误，但在 parse 方法中通常是自动的
        s = music21.converter.parse(abc_str, format="abc")

        # 3. 处理 Opus (多首曲子) vs Score (单首)
        # 即使我们截断了，Music21 有时仍会返回 Opus 对象
        if isinstance(s, music21.stream.Opus):
            # 如果是作品集，取第一首
            stream_to_write = s[0]
        else:
            stream_to_write = s

        # 4. 写入
        stream_to_write.write("midi", fp=filename)
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
        dropout=0.0,  # 推理时不需要 Dropout
    )
    model = MelodyGPT(config)

    # 加载 State Dict
    try:
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()  # 切换到评估模式
        print("[Inference] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load model checkpoint: {e}")
        return

    # 调用封装好的生成函数
    abc_content = generate_melody_with_params(
        model=model,
        tokenizer=tokenizer,
        key=TARGET_KEY,
        meter=TARGET_METER,
        max_tokens=500,  # 足够长以包含完整一首
        temp=0.9,  # 稍微提高随机性以获得更多变化
    )

    # 后处理与保存
    print("-" * 30)
    print(f"[Inference] Extracted Song ({TARGET_KEY}, {TARGET_METER}):")
    print(abc_content)
    print("-" * 30)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(OUTPUT_DIR, f"melody_{TARGET_KEY}_{timestamp}")

    # 保存 ABC
    with open(output_base + ".abc", "w", encoding="utf-8") as f:
        f.write(abc_content)

    # 转换为 MIDI (复用之前的 save_midi)
    save_midi(abc_content, output_base + ".mid")


if __name__ == "__main__":
    main()
