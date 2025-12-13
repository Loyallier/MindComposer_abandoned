import json
import os
import sys
from collections import Counter
import re
import statistics

# ================= 1. 环境与导入 =================
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

# ================= 2. 配置与阈值 =================
INPUT_FILE = os.path.join(path.DATA_INTERIM_DIR, "training_data_aligned.txt")
OUTPUT_VOCAB = path.VOCAB_PATH
OUTPUT_DATA = path.DATASET_PATH

SUSPICION_THRESHOLD = 100

# 🌟 罚分权重表
PENALTY_SCORES = {
    "LowNote": 10,
    "HighNote": 10,
    "Chromatic": 25,
    "WeirdChord": 15,
    "EmptyBar": 50,
    "WrongKey": 100,
    "BadRhythm_Hard": 100,
    "BadRhythm_Soft": 80
}

DIATONIC_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}
SAFE_ROOTS = {"C", "D", "E", "F", "G", "A", "B", "Bb", "A#", "Eb", "D#"}
RESERVED_TOKENS = [
    config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, "_", config.BAR_TOKEN, "0",
]

# ================= 3. 辅助功能 =================

def transpose_melody_seq(seq, semitones):
    new_seq = []
    for token in seq:
        # 🌟 修正：排除 "0" (休止符)
        if token.isdigit() and token != "0":
            try:
                val = int(token)
                new_seq.append(str(val + semitones))
            except:
                new_seq.append(token)
        else:
            new_seq.append(token)
    return new_seq

def get_melody_range(seq):
    """计算旋律音域 (Max - Min), 忽略休止符 '0'"""
    notes = []
    for token in seq:
        # 🌟 修正：排除 "0"
        if token.isdigit() and token != "0":
            notes.append(int(token))
    if not notes:
        return 0
    return max(notes) - min(notes)

# ================= 4. 异常检测器 =================

class SuspicionDetector:
    def __init__(self):
        self.dropped_count = 0
        self.rescued_count = 0
        self.reasons = Counter()
        self.lownote_ranges = []

    def check(self, melody_seq, harmony_seq):
        score = 0
        reasons = []

        total_notes = 0
        chromatic_notes = 0
        
        # --- A. 结构一致性检测 ---
        bar_indices = [i for i, x in enumerate(melody_seq) if x == config.BAR_TOKEN]
        if len(bar_indices) > 1:
            segment_lengths = []
            for i in range(len(bar_indices) - 1):
                length = bar_indices[i+1] - bar_indices[i] - 1
                segment_lengths.append(length)
            
            unique_lengths = set(segment_lengths)
            count_unique = len(unique_lengths)

            if count_unique > 2:
                score += PENALTY_SCORES["BadRhythm_Hard"]
                reasons.append(f"InconsistentBar(Hard, lens={unique_lengths})")
            elif count_unique == 2:
                score += PENALTY_SCORES["BadRhythm_Soft"]
                reasons.append(f"InconsistentBar(Soft, lens={unique_lengths})")

        # --- B. 检查旋律 ---
        for token in melody_seq:
            # 🌟 修正：严格排除 "0"
            if token.isdigit() and token != "0":
                val = int(token)
                total_notes += 1

                if val < 53:
                    score += PENALTY_SCORES["LowNote"]
                    reasons.append(f"LowNote({val})")

                if val > 88:
                    score += PENALTY_SCORES["HighNote"]
                    reasons.append(f"HighNote({val})")

                if (val % 12) not in DIATONIC_PITCH_CLASSES:
                    chromatic_notes += 1

        if total_notes > 0:
            chromatic_ratio = chromatic_notes / total_notes
            if chromatic_ratio > 0.15:
                score += PENALTY_SCORES["WrongKey"]
                reasons.append(f"WrongKey({chromatic_ratio:.1%})")
            elif chromatic_ratio > 0.05:
                score += PENALTY_SCORES["Chromatic"]
                reasons.append(f"Chromatic({chromatic_ratio:.1%})")

        # --- C. 检查和弦 ---
        current_bar_chords = []
        for token in harmony_seq:
            if token == config.BAR_TOKEN:
                if current_bar_chords and all(t in ["_", "0"] for t in current_bar_chords):
                    score += PENALTY_SCORES["EmptyBar"]
                    reasons.append("EmptyBar")
                current_bar_chords = []
                continue

            current_bar_chords.append(token)

            if token not in RESERVED_TOKENS and token != "0":
                match = re.match(r"^([A-G][b#-]?)", token)
                if match:
                    root = match.group(1)
                    if (root not in SAFE_ROOTS and root.replace("-", "b") not in SAFE_ROOTS):
                        score += PENALTY_SCORES["WeirdChord"]
                        reasons.append(f"WeirdChord({root})")

        return score, reasons


# ================= 5. 和弦标准化 (不变) =================
def normalize_chord(token):
    if token in RESERVED_TOKENS or token == "0" or token == "_":
        return token
    enharmonic_map = {
        "Db": "C#", "D-": "C#", "Eb": "D#", "E-": "D#",
        "Gb": "F#", "G-": "F#", "Ab": "G#", "A-": "G#",
        "Bb": "A#", "B-": "A#", "Cb": "B", "E#": "F",
        "B#": "C", "Fb": "E",
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


# ================= 6. 主程序 =================

def main():
    if not os.path.exists(path.DATA_PROCESSED_DIR):
        os.makedirs(path.DATA_PROCESSED_DIR)
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print("🕵️‍♂️ 启动增强版异常检测 (V6 - Fix Rest '0' Bug)...")
    
    detector = SuspicionDetector()
    valid_melody = []
    valid_harmony = []
    total_lines = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split(" | ")
            if len(parts) != 2:
                continue
            m_seq = parts[0].split()
            h_seq = parts[1].split()

            total_lines += 1

            # --- 第一轮检测 ---
            score, reasons = detector.check(m_seq, h_seq)

            # --- 统计 LowNote 歌曲的音域 (无论死活) ---
            has_lownote = any("LowNote" in r for r in reasons)
            if has_lownote:
                pitch_range = get_melody_range(m_seq)
                detector.lownote_ranges.append(pitch_range)

            # --- 救援尝试 ---
            if has_lownote:
                m_seq_rescued = transpose_melody_seq(m_seq, 12)
                score_new, reasons_new = detector.check(m_seq_rescued, h_seq)
                if score_new < score and score_new < SUSPICION_THRESHOLD:
                    m_seq = m_seq_rescued
                    score = score_new
                    reasons = reasons_new
                    detector.rescued_count += 1

            # --- 最终判决 ---
            if score >= SUSPICION_THRESHOLD:
                detector.dropped_count += 1
                cleaned_reasons = [r.split("(")[0] for r in reasons]
                detector.reasons.update(cleaned_reasons)
                continue

            valid_melody.append(m_seq)
            valid_harmony.append(h_seq)

    print(f"\n📊 检测报告:")
    print(f"   - 原始: {total_lines}")
    print(f"   - 有效: {len(valid_melody)}")
    print(f"   - 剔除: {detector.dropped_count} (剔除率: {detector.dropped_count / total_lines:.1%})")
    print(f"   - 🚑 成功救援: {detector.rescued_count} 例")

    # 🌟 专项分析：LowNote
    print(f"\n📉 LowNote 专项尸检 (已排除休止符 '0'):")
    if detector.lownote_ranges:
        avg_range = statistics.mean(detector.lownote_ranges)
        max_range = max(detector.lownote_ranges)
        min_range = min(detector.lownote_ranges)
        print(f"   - 触发 LowNote 的歌曲数量: {len(detector.lownote_ranges)}")
        print(f"   - 平均音域 (Range): {avg_range:.2f} 半音 (约 {avg_range/12:.1f} 个八度)")
        print(f"   - 最大音域: {max_range} 半音")
        print(f"   - 最小音域: {min_range} 半音")
    else:
        print(f"   - ✅ 无 LowNote 样本 (太棒了!)")

    print(f"\n💀 详细死因统计 (剔除样本中出现的错误):")
    sorted_reasons = sorted(detector.reasons.items(), key=lambda x: x[1], reverse=True)
    for r, c in sorted_reasons:
        print(f"   ❌ {r}: {c} 次")

    # 构建词表
    melody_vocab = build_vocab(valid_melody, "Melody", min_freq=1)
    harmony_vocab = build_vocab(valid_harmony, "Harmony", min_freq=20, is_harmony=True)

    full_vocab = {"melody": melody_vocab, "harmony": harmony_vocab}
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(full_vocab, f, indent=4)
    print(f"\n💾 词表已保存 (Melody: {len(melody_vocab)}, Harmony: {len(harmony_vocab)})")

    # 编码
    encoded_data = []
    for idx, (m, h) in enumerate(zip(valid_melody, valid_harmony)):
        encoded_data.append(
            {
                "id": idx,
                "length": len(m) + 2,
                "input": encode_sequence(m, melody_vocab),
                "target": encode_sequence(h, harmony_vocab, True),
            }
        )

    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(encoded_data, f)
    print("🎉 处理完成！")

if __name__ == "__main__":
    main()