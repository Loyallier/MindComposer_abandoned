import json
import os
import sys
from collections import Counter
import re
import random

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

# ================= 2. 配置 (V3.2) =================
INPUT_FILE = os.path.join(path.DATA_INTERIM_DIR, "training_data_cleaned.txt")
OUTPUT_VOCAB = path.VOCAB_PATH
OUTPUT_TRAIN_DATA = path.DATA_PROCESSED_DIR / "train_data.json"
OUTPUT_VAL_DATA = path.DATA_PROCESSED_DIR / "val_data.json"

SLICE_WINDOW_BARS = 4
SLICE_STEP_BARS = 1
RANDOM_SEED = 42
VAL_RATIO = 0.1

BAR_POS_IDX = 31
MAX_BEAT_IDX = 30 
POS_PAD_IDX = 32

RESERVED_TOKENS = [
    config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN,
    "_", config.BAR_TOKEN, "0",
]

# ================= 3. 工具 =================

def normalize_chord(token):
    if token in RESERVED_TOKENS or token == "0" or token == "_": return token
    enharmonic_map = {"Db":"C#", "D-":"C#", "Eb":"D#", "E-":"D#", "Gb":"F#", "G-":"F#", "Ab":"G#", "A-":"G#", "Bb":"A#", "B-":"A#", "Cb":"B", "E#": "F", "B#": "C", "Fb": "E"}
    match = re.match(r"^([A-G][b#-]?)", token)
    if not match: return config.UNK_TOKEN
    root = match.group(1)
    suffix = token[len(root):]
    if root in enharmonic_map: root = enharmonic_map[root]
    new_suffix = ""
    if "m" in suffix and "maj" not in suffix: new_suffix = "m"
    if "7" in suffix and "maj" not in suffix and "m" not in new_suffix: new_suffix += "7"
    elif "7" in suffix and "m" in new_suffix: new_suffix = "m7"
    if "m" in new_suffix: final_suffix = "m"
    elif "7" in new_suffix: final_suffix = "7"
    else: final_suffix = ""
    return root + final_suffix

def split_to_bars_raw(seq):
    bars = []
    curr = []
    for token in seq:
        if token == config.BAR_TOKEN:
            bars.append(curr)
            curr = []
        else:
            curr.append(token)
    if curr: bars.append(curr)
    return bars

def get_dominant_bar_length(bars_tokens):
    if not bars_tokens: return 16
    lengths = [len(b) for b in bars_tokens]
    counts = Counter(lengths)
    return counts.most_common(1)[0][0]

def build_vocab(sequences, vocab_name, min_freq=1, is_harmony=False):
    counter = Counter()
    for seq in sequences:
        if is_harmony:
            cleaned = [normalize_chord(t) for t in seq]
            counter.update(cleaned)
        else:
            counter.update(seq)
    token_to_id = {}
    curr_id = 0
    for t in RESERVED_TOKENS:
        token_to_id[t] = curr_id
        curr_id += 1
    valid = [t for t, c in counter.items() if t not in RESERVED_TOKENS and c >= min_freq]
    try: valid.sort(key=lambda x: int(x) if x.isdigit() else 999)
    except: valid.sort()
    for t in valid:
        token_to_id[t] = curr_id
        curr_id += 1
    return token_to_id

# ✅ [修正] 专门的 Input 编码器 (带位置)
def encode_input(seq, pos_seq, vocab):
    sos_id = vocab[config.SOS_TOKEN]
    eos_id = vocab[config.EOS_TOKEN]
    unk_id = vocab["0"]
    
    encoded_ids = [sos_id]
    encoded_pos = [BAR_POS_IDX] # SOS -> 31
    
    # 正常 zip 遍历
    for token, pos in zip(seq, pos_seq):
        encoded_ids.append(vocab.get(token, unk_id))
        encoded_pos.append(pos)
        
    encoded_ids.append(eos_id)
    encoded_pos.append(BAR_POS_IDX) # EOS -> 31
    
    return encoded_ids, encoded_pos

# ✅ [修正] 专门的 Target 编码器 (不带位置，修复 Zip 空列表问题)
def encode_target(seq, vocab):
    sos_id = vocab[config.SOS_TOKEN]
    eos_id = vocab[config.EOS_TOKEN]
    unk_id = vocab["0"]
    
    encoded_ids = [sos_id]
    
    for token in seq:
        token = normalize_chord(token)
        encoded_ids.append(vocab.get(token, unk_id))
    
    encoded_ids.append(eos_id)
    return encoded_ids

def process_dataset(song_lines, melody_vocab, harmony_vocab, start_id=0):
    encoded_data = []
    slice_count = start_id
    skipped_count = 0
    
    for song_idx, line in enumerate(song_lines):
        parts = line.strip().split(" | ")
        m_seq = parts[0].split()
        h_seq = parts[1].split()

        m_bars = split_to_bars_raw(m_seq)
        h_bars = split_to_bars_raw(h_seq)
        
        if len(m_bars) != len(h_bars) or not m_bars:
            continue

        dominant_len = get_dominant_bar_length(m_bars)
        m_pos_bars = []
        
        for i, bar in enumerate(m_bars):
            bar_len = len(bar)
            if i == 0 and bar_len < dominant_len and bar_len < 10:
                start_idx = max(0, dominant_len - bar_len)
                pos_seq = [min(start_idx + k, MAX_BEAT_IDX) for k in range(bar_len)]
            else:
                pos_seq = [min(k, MAX_BEAT_IDX) for k in range(bar_len)]
            m_pos_bars.append(pos_seq)

        def save_slice(m_b, m_p_b, h_b, s_start):
            nonlocal slice_count
            flat_inp = []
            flat_pos = []
            flat_trg = []
            
            for mb, mp, hb in zip(m_b, m_p_b, h_b):
                flat_inp.extend(mb + [config.BAR_TOKEN])
                flat_pos.extend(mp + [BAR_POS_IDX]) 
                flat_trg.extend(hb + [config.BAR_TOKEN])
            
            # 编码 (分别调用修复后的函数)
            enc_inp, enc_pos = encode_input(flat_inp, flat_pos, melody_vocab)
            enc_trg = encode_target(flat_trg, harmony_vocab) # ✅ Fix
            
            entry = {
                "id": slice_count,
                "origin_song_idx": song_idx,
                "slice_start_bar": s_start,
                "length": len(enc_inp),
                "input": enc_inp,
                "position": enc_pos,
                "target": enc_trg
            }
            encoded_data.append(entry)
            slice_count += 1

        num_bars = len(m_bars)
        if num_bars <= SLICE_WINDOW_BARS:
            save_slice(m_bars, m_pos_bars, h_bars, 0)
        else:
            for i in range(0, num_bars - SLICE_WINDOW_BARS + 1, SLICE_STEP_BARS):
                save_slice(
                    m_bars[i : i + SLICE_WINDOW_BARS], 
                    m_pos_bars[i : i + SLICE_WINDOW_BARS],
                    h_bars[i : i + SLICE_WINDOW_BARS], 
                    i
                )

    return encoded_data, slice_count, skipped_count

# ================= 4. 主程序 =================

def main():
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_VOCAB), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_DATA), exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到: {INPUT_FILE}")
        return

    print(f"🔒 [V3.2 Fixed] 正在修复数据编码 (Zip Bug Fix)...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if "|" in line]
    
    print(f"📦 原始歌曲: {len(all_lines)}")

    print("📖 构建词表...")
    temp_m, temp_h = [], []
    for line in all_lines:
        p = line.strip().split(" | ")
        temp_m.extend(p[0].split())
        temp_h.extend(p[1].split())
    
    mv = build_vocab([temp_m], "Melody")
    hv = build_vocab([temp_h], "Harmony", min_freq=20, is_harmony=True)
    
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump({"melody": mv, "harmony": hv}, f, indent=4)

    random.seed(RANDOM_SEED)
    random.shuffle(all_lines)
    val_idx = int(len(all_lines) * VAL_RATIO)
    val_lines = all_lines[:val_idx]
    train_lines = all_lines[val_idx:]
    
    print(f"📊 训练集: {len(train_lines)} | 验证集: {len(val_lines)}")

    train_data, c1, s1 = process_dataset(train_lines, mv, hv, 0)
    val_data, c2, s2 = process_dataset(val_lines, mv, hv, c1)

    with open(OUTPUT_TRAIN_DATA, "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open(OUTPUT_VAL_DATA, "w", encoding="utf-8") as f:
        json.dump(val_data, f)

    print(f"\n🚀 完成:")
    print(f"   - Train Slices: {len(train_data)}")
    print(f"   - Val Slices:   {len(val_data)}")
    print(f"   - 请重新运行 check_data_integrity.py 确认 Target 不再只是 [1, 2]")

if __name__ == "__main__":
    main()