import json
import os
import re
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

# ✅ 统一引用
from src import path
from src.ChordGenerator_A import config


# ================= 配置区域 =================
# 使用 config 或 path 里的路径，不要硬编码
INPUT_FILE = str(path.DATA_INTERIM_DIR / "training_data_aligned.txt")
OUTPUT_VOCAB = str(path.VOCAB_PATH)
OUTPUT_DATA = str(path.DATASET_PATH)


# 【新增】切片配置
CHUNK_SIZE = 256  # 窗口大小：256个16分音符 (约16小节)
MIN_LEN = 32  # 最小长度：太短的碎片丢弃

# 特殊 Token 定义
# config.PAD_TOKEN = "<PAD>" # ID 0
# config.SOS_TOKEN = "<SOS>" # ID 1
# config.EOS_TOKEN = "<EOS>" # ID 2
SPECIAL_TOKENS = [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN]

# 异名同音映射表
ENHARMONIC_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "E#": "F",
    "B#": "C",
    "Fb": "E",
}


def clean_melody_token(token):
    """清洗旋律: 非数字强制转为休止符 0"""
    token = token.strip()
    if token in ["_", "0", config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN]:
        return token
    if token.isdigit():
        return token
    return "0"


def simplify_chord(chord_str):
    """清洗和弦: 归一化为 Root + m/7"""
    chord_str = chord_str.strip()
    if chord_str in ["_", "0", "N.C.", "rest"]:
        if chord_str == "rest":
            return "0"
        return chord_str

    # 去除转位
    if "/" in chord_str:
        chord_str = chord_str.split("/")[0]

    # 正则解析
    match = re.match(r"^([A-G][#b]?)(.*)$", chord_str)
    if not match:
        return "0"

    root = match.group(1)
    raw_suffix = match.group(2)

    # 异名同音处理
    if root in ENHARMONIC_MAP:
        root = ENHARMONIC_MAP[root]

    # 后缀归类
    final_suffix = ""
    if "m" in raw_suffix and "maj" not in raw_suffix:
        final_suffix = "m"
    elif "7" in raw_suffix and "maj" not in raw_suffix:
        final_suffix = "7"
    else:
        final_suffix = ""  # 其他都归为大三和弦

    return root + final_suffix


def slice_sequence(m_seq, h_seq, chunk_size, min_len):
    """
    【新增核心函数】重叠切片
    将长歌切分为多个 chunk_size 的片段，步长为 chunk_size // 2
    """
    slices = []
    total_len = len(m_seq)

    # 步长设为窗口的一半 (50% Overlap)，增加数据连贯性
    stride = chunk_size // 2

    # 如果歌本身很短，直接返回（或者丢弃）
    if total_len <= chunk_size:
        if total_len >= min_len:
            slices.append((m_seq, h_seq))
        return slices

    # 滑动窗口切分
    for i in range(0, total_len, stride):
        m_chunk = m_seq[i : i + chunk_size]
        h_chunk = h_seq[i : i + chunk_size]

        # 只有片段足够长才保留
        if len(m_chunk) >= min_len:
            slices.append((m_chunk, h_chunk))

    return slices


def load_and_clean_data(file_path):
    print(f"正在读取并清洗 {file_path} ...")
    clean_pairs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = line.split("|")
            raw_m = parts[0].strip().split()
            raw_h = parts[1].strip().split()

            # 1. 清洗
            clean_m = [clean_melody_token(t) for t in raw_m]
            clean_h = [simplify_chord(t) for t in raw_h]

            # 2. 强制对齐长度
            min_len = min(len(clean_m), len(clean_h))
            if min_len < MIN_LEN:
                continue
            final_m = clean_m[:min_len]
            final_h = clean_h[:min_len]

            # 3. 【调用切片】
            chunks = slice_sequence(final_m, final_h, CHUNK_SIZE, MIN_LEN)
            clean_pairs.extend(chunks)

    print(f"处理完成: 生成了 {len(clean_pairs)} 条训练片段 (Max Len: {CHUNK_SIZE})")
    return clean_pairs


def build_vocab(sequences, vocab_name):
    unique_tokens = set()
    for seq in sequences:
        unique_tokens.update(seq)
    sorted_tokens = sorted(list(unique_tokens))

    token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    start_id = len(SPECIAL_TOKENS)
    for i, token in enumerate(sorted_tokens):
        token_to_id[token] = start_id + i

    print(f"[{vocab_name}] 字典大小: {len(token_to_id)}")
    return token_to_id


def encode_sequence(seq, vocab):
    return (
        [vocab[config.SOS_TOKEN]] + [vocab[t] for t in seq] + [vocab[config.EOS_TOKEN]]
    )


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    # 1. 加载、清洗、切片
    data_pairs = load_and_clean_data(INPUT_FILE)
    m_seqs = [p[0] for p in data_pairs]
    h_seqs = [p[1] for p in data_pairs]

    # 2. 构建字典
    melody_vocab = build_vocab(m_seqs, "Melody")
    harmony_vocab = build_vocab(h_seqs, "Harmony")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_VOCAB), exist_ok=True)

    # 保存字典
    full_vocab = {"melody": melody_vocab, "harmony": harmony_vocab}
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(full_vocab, f, indent=4)

    # 3. 数字化
    encoded_dataset = []
    for idx, (m, h) in enumerate(zip(m_seqs, h_seqs)):
        m_ids = encode_sequence(m, melody_vocab)
        h_ids = encode_sequence(h, harmony_vocab)

        encoded_dataset.append(
            {"id": idx, "length": len(m_ids), "input": m_ids, "target": h_ids}
        )

    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(encoded_dataset, f)

    print(f"✅ 数字化完成！\n字典: {OUTPUT_VOCAB}\n数据: {OUTPUT_DATA}")
    print("👉 现在可以去运行 train.py 了，记得把 BATCH_SIZE 改回 32 或 64！")


if __name__ == "__main__":
    main()
