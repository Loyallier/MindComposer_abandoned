import os
import re
import glob

def extract_chords_from_abc_file(file_path):
    """
    从单个ABC文件中提取和弦序列。
    返回一个列表，其中每个元素是一首曲子的和弦字符串（以空格分隔）。
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 1. 以 "X:" 分割曲目 (ABC文件中 X: 是曲目的索引号，标志着新曲子的开始)
    # 过滤掉第一部分（通常是文件头或空的）
    raw_tunes = [t for t in content.split('X:') if t.strip()]

    extracted_sequences = []

    for tune in raw_tunes:
        # 2. 分离元数据（Header）和正文（Body）
        # ABC记谱法的正文通常在 K: (Key) 标签之后开始
        if 'K:' in tune:
            # 找到 K: 所在的位置，之后的内容为乐谱正文
            header, body = tune.split('K:', 1)
            # 有时候 K: 后面还有调号参数，取换行符之后的内容作为正文
            if '\n' in body:
                _, body = body.split('\n', 1)
        else:
            # 如果没有 K:，可能是不标准的片段，尝试直接处理
            body = tune

        # 3. 使用正则表达式提取引号内的内容
        # ABC中和弦通常在双引号内，例如 "Gm", "D7"
        # 正则解释: "([A-G][^"]*)"
        # 匹配以 A-G 开头的内容（排除掉像 "slowly" 这种非和弦标记），直到遇到下一个引号
        # 你的源文件中有些像 "g" 或 "f#" 的小写标记，通常是低音走向而非和弦根音，
        # 如果你想保留它们，去掉 [A-G] 限制即可。这里我保留了 [A-G] 以确保提取的是和弦。
        chords = re.findall(r'"([A-G][^"]*)"', body)

        # 4. 数据清洗
        # 移除一些可能的特殊字符或多余空格
        clean_chords = [c.strip() for c in chords if c.strip()]

        # 如果这首曲子提取到了和弦，加入列表
        if clean_chords:
            # 将列表转换为空格分隔的字符串
            sequence_str = ' '.join(clean_chords)
            extracted_sequences.append(sequence_str)

    return extracted_sequences

def process_dataset(input_dir, output_file):
    """
    遍历目录下的所有 .txt 或 .abc 文件，提取和弦并保存。
    """
    all_sequences = []
    
    # 获取所有 .txt 和 .abc 文件
    files = glob.glob(os.path.join(input_dir, '*.txt')) + \
            glob.glob(os.path.join(input_dir, '*.abc'))

    print(f"找到 {len(files)} 个文件，开始处理...")

    for file_path in files:
        try:
            sequences = extract_chords_from_abc_file(file_path)
            all_sequences.extend(sequences)
            print(f" - 已处理: {os.path.basename(file_path)} (提取了 {len(sequences)} 首曲子)")
        except Exception as e:
            print(f" ! 处理文件出错 {file_path}: {e}")

    # 写入结果文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for seq in all_sequences:
            f.write(seq + '\n')

    print(f"\n处理完成！")
    print(f"总共提取了 {len(all_sequences)} 条和弦序列。")
    print(f"训练数据已保存至: {output_file}")

# ================= 使用说明 =================
# 1. 将你的 ABC/TXT 文件放在一个文件夹里，例如 'raw_data'
# 2. 设置下面的 INPUT_DIR 为该文件夹路径
# 3. 运行代码

if __name__ == "__main__":
    # 修改这里：你的20个文件所在的文件夹路径
    # 如果文件就在当前目录下，可以使用 '.'
    INPUT_DIR = 'E:\VScode_Programs\！Projects\AI_AS1_G\dataset' 
    
    # 输出文件名
    OUTPUT_FILE = 'chords_dataset_for_training.txt'

    # 创建示例文件（为了演示，如果你已经有文件可以注释掉这一块）
    # (此处代码仅为演示逻辑，实际运行时会读取你目录下的文件)
    
    process_dataset(INPUT_DIR, OUTPUT_FILE)