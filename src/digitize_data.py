import json
import os
import re

# ================= 配置区域 =================
INPUT_FILE = r"data\interim\training_data_aligned.txt"
OUTPUT_VOCAB = r"data\processed\vocab.json"
OUTPUT_DATA = r"data\processed\dataset_encoded.json"

# 特殊 Token 定义 (ID固定)
PAD_TOKEN = "<PAD>" # ID 0
SOS_TOKEN = "<SOS>" # ID 1
EOS_TOKEN = "<EOS>" # ID 2
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

# 异名同音映射表 (统一转为 # 号)
ENHARMONIC_MAP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B",  "E#": "F",  "B#": "C",  "Fb": "E"
}

def clean_melody_token(token):
    """
    清洗旋律 Token:
    1. 必须是数字 (MIDI) 或 "_" 或 "0"
    2. 如果是奇怪的字符，强制转为 "0" (休止符)，保持节拍对齐
    """
    token = token.strip()
    if token in ["_", "0"]:
        return token
    
    # 尝试判断是否为数字
    if token.isdigit():
        return token
    
    # 如果是其他脏数据（比如 abc 里的 'z', '>', '|' 残留），转为休止
    return "0"

def simplify_chord(chord_str):
    """
    【核心逻辑】和弦归一化
    将任意复杂和弦映射到: Root + (m) + (7) 或 _ 或 0
    """
    chord_str = chord_str.strip()
    
    # 1. 处理特殊标记
    if chord_str in ["_", "0", "N.C.", "rest"]:
        if chord_str == "rest": return "0"
        return chord_str
    
    # 2. 预处理：去除转位 (Slash Chords) -> 取 '/' 前面的部分
    if "/" in chord_str:
        chord_str = chord_str.split("/")[0]
        
    # 3. 正则解析：分离 根音(Root) 和 后缀(Suffix)
    # 匹配: 大写字母A-G + 可选的#或b + 剩余所有字符
    match = re.match(r"^([A-G][#b]?)(.*)$", chord_str)
    
    if not match:
        # 无法识别的格式，视为无和弦
        return "0"
        
    root = match.group(1)
    raw_suffix = match.group(2)
    
    # 4. 异名同音转换 (Eb -> D#)
    if root in ENHARMONIC_MAP:
        root = ENHARMONIC_MAP[root]
        
    # 5. 后缀归类 (Mapping Logic)
    final_suffix = ""
    
    # 优先级 1: 小调 (m)
    # 只要包含 'm' 且不包含 'maj' (排除 maj7)，就是小三和弦
    # 兼容: m, min, m7, m9, m11 -> m
    if 'm' in raw_suffix and 'maj' not in raw_suffix:
        final_suffix = "m"
        
    # 优先级 2: 属七 (7)
    # 包含 '7' 且没有 'maj' 且已经被排除掉 'm' -> 属七
    # 兼容: 7, 7sus4, 9, 13 (通常 9 和 13 包含 7 的功能，这里简化为 7 或 大三)
    # 为了保险，我们只认明确写了 7 的
    elif '7' in raw_suffix and 'maj' not in raw_suffix:
        final_suffix = "7"
    
    # 优先级 3: 其他统统归为 大三和弦 (无后缀)
    # 兼容: maj7, sus4, add9, 6, dim, aug (简化处理)
    else:
        final_suffix = ""
        
    return root + final_suffix

def load_and_clean_data(file_path):
    print(f"正在读取并清洗 {file_path} ...")
    clean_pairs = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            raw_m = parts[0].strip().split()
            raw_h = parts[1].strip().split()
            
            # --- 步骤 1: 清洗 Token ---
            clean_m = [clean_melody_token(t) for t in raw_m]
            clean_h = [simplify_chord(t) for t in raw_h]
            
            # --- 步骤 2: 长度强制对齐 ---
            # 取两者中最短的长度，截断多余的 (保证 1:1)
            min_len = min(len(clean_m), len(clean_h))
            
            if min_len < 4: # 太短的片段丢弃
                continue
                
            final_m = clean_m[:min_len]
            final_h = clean_h[:min_len]
            
            clean_pairs.append((final_m, final_h))
            
    print(f"清洗完成: 保留了 {len(clean_pairs)} 条有效数据。")
    return clean_pairs

def build_vocab(sequences, vocab_name):
    """构建字典"""
    unique_tokens = set()
    for seq in sequences:
        unique_tokens.update(seq)
    
    sorted_tokens = sorted(list(unique_tokens))
    
    # 特殊符放在前 0,1,2
    token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    start_id = len(SPECIAL_TOKENS)
    
    for i, token in enumerate(sorted_tokens):
        token_to_id[token] = start_id + i
        
    print(f"[{vocab_name}] 字典大小: {len(token_to_id)} (含特殊符)")
    return token_to_id

def encode_sequence(seq, vocab):
    """转数字: [SOS, id, id..., EOS]"""
    # 如果遇到字典里没有的脏数据（理论上build_vocab已经覆盖，但为了保险），用 PAD(0) 或 忽略
    # 这里直接映射，因为我们是用清洗后的数据建的字典
    return [vocab[SOS_TOKEN]] + [vocab[t] for t in seq] + [vocab[EOS_TOKEN]]

def main():
    # 1. 清洗数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    data_pairs = load_and_clean_data(INPUT_FILE)
    
    # 解压为两个列表
    m_seqs = [p[0] for p in data_pairs]
    h_seqs = [p[1] for p in data_pairs]

    # 2. 构建字典
    melody_vocab = build_vocab(m_seqs, "Melody")
    harmony_vocab = build_vocab(h_seqs, "Harmony")
    
    # 打印和弦列表供检查 (确认没有奇怪的和弦混进去)
    print("\n--- 最终生成的和弦列表 (请检查是否只有标准格式) ---")
    chord_list = [k for k, v in harmony_vocab.items() if k not in SPECIAL_TOKENS]
    print(chord_list)
    print("--------------------------------------------------\n")

    # 保存字典
    full_vocab = {"melody": melody_vocab, "harmony": harmony_vocab}
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(full_vocab, f, indent=4)

    # 3. 数字化并保存
    encoded_dataset = []
    for idx, (m, h) in enumerate(zip(m_seqs, h_seqs)):
        m_ids = encode_sequence(m, melody_vocab)
        h_ids = encode_sequence(h, harmony_vocab)
        
        encoded_dataset.append({
            "id": idx,
            "length": len(m_ids), 
            "input": m_ids,
            "target": h_ids
        })

    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(encoded_dataset, f)
        
    print(f"✅ 处理完成！\n字典: {OUTPUT_VOCAB}\n数据: {OUTPUT_DATA}")

if __name__ == "__main__":
    main()