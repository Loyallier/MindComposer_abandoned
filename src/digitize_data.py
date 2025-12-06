import json
import os

# ================= 配置区域 =================
# 输入文件：这就是你上传的那个 txt 文件的路径
INPUT_FILE = "data_preprocessing\\training_data_aligned.txt"

# 输出文件：生成的两个 JSON
OUTPUT_VOCAB = "vocab.json"
OUTPUT_DATA = "dataset_encoded.json"

# 特殊 Token 定义 (这里的 ID 必须固定)
# 0: PAD (填充，用于 Batch对齐)
# 1: SOS (Start Of Sequence，告诉模型开始生成)
# 2: EOS (End Of Sequence，告诉模型生成结束)
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]

def load_raw_data(file_path):
    """读取 txt 文件，拆分旋律和和弦"""
    melody_seqs = []
    harmony_seqs = []
    
    print(f"正在读取 {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line: continue
            
            # 你的数据格式是: 旋律 | 和弦
            parts = line.split("|")
            if len(parts) != 2:
                print(f"警告: 第 {line_num+1} 行格式错误，已跳过。")
                continue
                
            # split() 默认按空格切分
            m_seq = parts[0].strip().split()
            h_seq = parts[1].strip().split()
            
            # 简单校验长度是否对齐 (这对 Seq2Seq 很重要)
            if len(m_seq) != len(h_seq):
                print(f"警告: 第 {line_num+1} 行长度不对齐 (Melody:{len(m_seq)}, Harmony:{len(h_seq)}) - 建议检查清洗脚本")
                # 即使不对齐，我们暂时也收录，交给 DataLoader 处理或之后清洗
            
            melody_seqs.append(m_seq)
            harmony_seqs.append(h_seq)
            
    return melody_seqs, harmony_seqs

def build_vocab(sequences, vocab_name="vocab"):
    """
    构建词汇表 (Token -> ID)
    """
    unique_tokens = set()
    for seq in sequences:
        unique_tokens.update(seq)
    
    # 排序，保证每次运行生成的 ID 顺序一致
    sorted_tokens = sorted(list(unique_tokens))
    
    # 1. 先加入特殊 Token (ID: 0, 1, 2)
    token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    
    # 2. 再加入数据中的 Token (ID: 3, 4, ...)
    start_id = len(SPECIAL_TOKENS)
    for i, token in enumerate(sorted_tokens):
        token_to_id[token] = start_id + i
        
    print(f"[{vocab_name}] 构建完成: 包含 {len(token_to_id)} 个 Token (含特殊符)")
    return token_to_id

def encode_sequence(seq, vocab):
    """
    将字符列表转换为数字列表，并加上 SOS 和 EOS
    Input:  ['60', '_', '62']
    Output: [1, 5, 3, 6, 2]  (假设 SOS=1, 60=5, _=3, 62=6, EOS=2)
    """
    # 遇到字典里没有的词（比如新数据出现的），临时用 UNK 处理或者报错
    # 这里我们假设训练集构建的字典覆盖全集
    return [vocab["<SOS>"]] + [vocab[token] for token in seq] + [vocab["<EOS>"]]

def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    m_seqs, h_seqs = load_raw_data(INPUT_FILE)
    print(f"成功加载 {len(m_seqs)} 条数据对。")

    # 2. 构建两个独立的字典
    # 注意：即便 X 和 Y 都有 "0" 或 "_"，它们在不同的字典里 ID 可能不同，互不干扰
    melody_vocab = build_vocab(m_seqs, "Melody")
    harmony_vocab = build_vocab(h_seqs, "Harmony")

    # 保存字典 (以后预测推理时要用)
    full_vocab = {
        "melody": melody_vocab,
        "harmony": harmony_vocab
    }
    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(full_vocab, f, indent=4)
    print(f"字典已保存至: {OUTPUT_VOCAB}")

    # 3. 数字化所有数据
    encoded_data = []
    for idx, (m, h) in enumerate(zip(m_seqs, h_seqs)):
        m_ids = encode_sequence(m, melody_vocab)
        h_ids = encode_sequence(h, harmony_vocab)
        
        encoded_data.append({
            "id": idx,
            "length": len(m_ids), # 记录长度，供 Pack_Padded_Sequence 使用
            "input": m_ids,
            "target": h_ids
        })

    # 4. 保存最终的训练数据
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(encoded_data, f)
    print(f"数字化数据已保存至: {OUTPUT_DATA}")
    print("-" * 30)
    print("准备工作完成！下一步：编写 Dataset 和 DataLoader。")

if __name__ == "__main__":
    main()