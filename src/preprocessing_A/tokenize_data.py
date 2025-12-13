import json
import os
import sys
from collections import Counter
import re

# ================= 1. 环境 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

try:
    from src import path
    from src.ChordGenerator_A import config
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================= 2. 配置 =================
INPUT_FILE = os.path.join(path.DATA_INTERIM_DIR, "training_data_cleaned.txt")
OUTPUT_VOCAB = path.VOCAB_PATH
OUTPUT_DATA = path.DATASET_PATH

# 🌟 滑窗配置
SLICE_WINDOW_BARS = 4
SLICE_STEP_BARS = 1

RESERVED_TOKENS = [
    config.PAD_TOKEN,
    config.SOS_TOKEN,
    config.EOS_TOKEN,
    "_",
    config.BAR_TOKEN,
    "0",
]

# ================= 3. 工具 =================


def normalize_chord(token):
    # ... (保持原有的和弦标准化逻辑，完全不变) ...
    if token in RESERVED_TOKENS or token == "0" or token == "_":
        return token
    enharmonic_map = {
        "Db": "C#",
        "D-": "C#",
        "Eb": "D#",
        "E-": "D#",
        "Gb": "F#",
        "G-": "F#",
        "Ab": "G#",
        "A-": "G#",
        "Bb": "A#",
        "B-": "A#",
        "Cb": "B",
        "E#": "F",
        "B#": "C",
        "Fb": "E",
    }
    match = re.match(r"^([A-G][b#-]?)", token)
    if not match:
        return config.UNK_TOKEN
    root = match.group(1)
    suffix = token[len(root) :]
    if root in enharmonic_map:
        root = enharmonic_map[root]
    new_suffix = ""
    if "m" in suffix and "maj" not in suffix:
        new_suffix = "m"
    if "7" in suffix and "maj" not in suffix and "m" not in new_suffix:
        new_suffix += "7"
    elif "7" in suffix and "m" in new_suffix:
        new_suffix = "m7"
    if "m" in new_suffix:
        final_suffix = "m"
    elif "7" in new_suffix:
        final_suffix = "7"
    else:
        final_suffix = ""
    return root + final_suffix


def split_into_bars_smart(seq):
    """
    🌟 智能分节：保留弱起 (Pickup) 和 尾部 (Tail)
    逻辑：
    1. <bar> 是小节的分隔符/结束符。
    2. 只有当确实遇到 <bar> token 时，才将其作为该小节的结尾。
    3. 尾部如果只是残留音符，就保持原样，不强行补 <bar>。
       模型将学习到：[Note, Note, EOS] = 结束，而不是 [Note, Note, BAR, EOS]。
    """
    bars = []
    current_segment = []

    for token in seq:
        if token == config.BAR_TOKEN:
            # 遇到 bar，将其加入当前片段，标志着一个小节的结束
            current_segment.append(token)
            bars.append(current_segment)
            current_segment = []
        else:
            current_segment.append(token)

    # 处理尾部 (Tail)
    if current_segment:
        # 🌟 修正：不再强行 append(config.BAR_TOKEN)
        # 如果尾部没有 bar，那就是没有。让它裸露着。
        # 这样模型预测时会倾向于直接预测 EOS。
        bars.append(current_segment)

    return bars


def build_vocab(sequences, vocab_name, min_freq=1, is_harmony=False):
    counter = Counter()
    for seq in sequences:
        if is_harmony:
            cleaned = [normalize_chord(t) for t in seq]
            counter.update(cleaned)
        else:
            counter.update(seq)
    token_to_id = {}
    current_id = 0
    for token in RESERVED_TOKENS:
        token_to_id[token] = current_id
        current_id += 1
    valid_tokens = []
    for token, count in counter.items():
        if token in RESERVED_TOKENS:
            continue
        if count >= min_freq:
            valid_tokens.append(token)
    try:
        valid_tokens.sort(key=lambda x: int(x) if x.isdigit() else 999)
    except:
        valid_tokens.sort()
    for token in valid_tokens:
        token_to_id[token] = current_id
        current_id += 1
    return token_to_id


def encode_sequence(seq, vocab, is_harmony=False):
    encoded = [vocab[config.SOS_TOKEN]]
    unk_id = vocab["0"]
    for token in seq:
        if is_harmony:
            token = normalize_chord(token)
        encoded.append(vocab.get(token, unk_id))
    encoded.append(vocab[config.EOS_TOKEN])
    return encoded


# ================= 4. 主程序 =================


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到清洗后的文件: {INPUT_FILE}")
        return

    print(f"🔢 启动智能切片 (保留弱起/尾部)...")
    print(f"   - 窗口大小: {SLICE_WINDOW_BARS} 小节")

    all_melody_bars = []
    all_harmony_bars = []

    total_raw_songs = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" | ")
            m_seq = parts[0].split()
            h_seq = parts[1].split()

            # 🌟 使用新的智能分节
            m_bars = split_into_bars_smart(m_seq)
            h_bars = split_into_bars_smart(h_seq)

            # 必须再次校验长度（因为弱起和尾部的存在，这里要极其小心）
            if len(m_bars) != len(h_bars):
                # 如果长度不对，通常是ABC处理时 bar 符号不对齐
                # 这种数据无法用于配对训练，只能舍弃
                continue

            all_melody_bars.append(m_bars)
            all_harmony_bars.append(h_bars)
            total_raw_songs += 1

    print(f"   - 原始歌曲: {total_raw_songs} 首")
    # ================= 🕵️‍♂️ 调试探针 (DEBUG PROBE) =================
    print("\n🕵️‍♂️ [DEBUG] 数据形态抽查 (前 5 首):")
    for i in range(min(5, len(all_melody_bars))):
        m_bars = all_melody_bars[i]
        bar_len = len(m_bars)
        # 计算切片数
        if bar_len <= SLICE_WINDOW_BARS:
            slices = 1
        else:
            slices = (bar_len - SLICE_WINDOW_BARS) // SLICE_STEP_BARS + 1

        print(f"   - ID {i}: 检测到 {bar_len} 小节 -> 预计生成 {slices} 个切片")
        # 打印前2个小节的内容，检查 Token 是否正常
        first_bar_sample = m_bars[0] if m_bars else []
        print(f"     [First Bar]: {first_bar_sample}")
    print("======================================================\n")
    # ============================================================

    # 构建词表
    flat_melody = [token for song in all_melody_bars for bar in song for token in bar]
    flat_harmony = [token for song in all_harmony_bars for bar in song for token in bar]

    melody_vocab = build_vocab([flat_melody], "Melody", min_freq=1)
    harmony_vocab = build_vocab([flat_harmony], "Harmony", min_freq=20, is_harmony=True)

    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump({"melody": melody_vocab, "harmony": harmony_vocab}, f, indent=4)
    print(f"💾 词表已构建")

    # 执行切片
    encoded_data = []
    slice_count = 0

    for song_idx, (m_bars, h_bars) in enumerate(zip(all_melody_bars, all_harmony_bars)):
        num_bars = len(m_bars)

        # 🌟 策略调整：对于短歌，直接整首放入，不做 Padding (LSTM处理变长序列是强项，只要batch内pad即可)
        # 如果歌比窗口还短，就当做一个单独的样本
        if num_bars <= SLICE_WINDOW_BARS:
            m_slice_seq = [t for bar in m_bars for t in bar]
            h_slice_seq = [t for bar in h_bars for t in bar]
            encoded_entry = {
                "id": slice_count,
                "origin_song_id": song_idx,
                "slice_start_bar": 0,
                "length": len(m_slice_seq) + 2,
                "input": encode_sequence(m_slice_seq, melody_vocab),
                "target": encode_sequence(h_slice_seq, harmony_vocab, True),
            }
            encoded_data.append(encoded_entry)
            slice_count += 1
            continue

        # 正常长歌切片
        for i in range(0, num_bars - SLICE_WINDOW_BARS + 1, SLICE_STEP_BARS):
            m_slice_bars = m_bars[i : i + SLICE_WINDOW_BARS]
            h_slice_bars = h_bars[i : i + SLICE_WINDOW_BARS]

            m_slice_seq = [t for bar in m_slice_bars for t in bar]
            h_slice_seq = [t for bar in h_slice_bars for t in bar]

            encoded_entry = {
                "id": slice_count,
                "origin_song_id": song_idx,
                "slice_start_bar": i,
                "length": len(m_slice_seq) + 2,
                "input": encode_sequence(m_slice_seq, melody_vocab),
                "target": encode_sequence(h_slice_seq, harmony_vocab, True),
            }
            encoded_data.append(encoded_entry)
            slice_count += 1

            # 🌟 额外补充：如果最后还剩一点尾巴没被覆盖到（因为步长问题）
            # 比如 10小节，窗口4，步长3 -> [0:4], [3:7], [6:10] (完美)
            # 比如 10小节，窗口4，步长4 -> [0:4], [4:8] ... 剩下 [8:10] 没用
            # 如果你希望全覆盖，可以加逻辑。但步长为1通常能保证覆盖绝大部分。

    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(encoded_data, f)

    print(f"\n🚀 切片完成报告:")
    print(f"   - ✅ 生成训练样本 (Slices): {len(encoded_data)}")
    print(
        f"   - 数据膨胀倍数: {len(encoded_data) / total_raw_songs if total_raw_songs > 0 else 0:.1f}x"
    )


if __name__ == "__main__":
    main()
