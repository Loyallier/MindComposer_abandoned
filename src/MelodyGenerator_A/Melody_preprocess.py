import os
import re

# ================= 配置区 =================
RAW_DATA_DIR = os.path.join('data', 'raw')
OUTPUT_DIR = os.path.join('data', 'processed')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'dataset.txt')

# 定义白名单字符集 (基于标准ABC乐理)
# 包含：大小写字母、数字、空格、换行
# 乐理符号：| : [ ] / - ^ = , ' . > ( )
VALID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n|:[]/-^=_,'().>")

# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_line_strict(line):
    """
    严格清洗：
    1. 移除注释 (%)
    2. 移除和弦 ("...")
    3. 仅保留白名单内的字符
    4. 压缩多余空格
    """
    # 1. 移除注释
    line = line.split('%')[0]
    
    # 2. 移除和弦 (双引号包裹内容)
    line = re.sub(r'"[^"]*"', '', line)
    
    # 3. 白名单过滤
    # 逐字符检查，如果在白名单里则保留
    line = ''.join([c for c in line if c in VALID_CHARS])
    
    # 4. 去除首尾空白
    return line.strip()

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 按 X: 分割曲目
    raw_songs = re.split(r'\n\s*X:', content)
    processed_songs = []

    for song in raw_songs:
        if not song.strip():
            continue
            
        lines = song.split('\n')
        filtered_lines = []
        
        for line in lines:
            # 先进行严格清洗
            cleaned_line = clean_line_strict(line)
            
            if not cleaned_line:
                continue

            # 简单的逻辑判断是否为需要的Header或旋律
            # 因为清洗过，所以必须确保 M: 和 K: 结构没被破坏 (冒号在白名单里，字母在白名单里)
            
            is_header = False
            # 检查是否是 M: 或 K:
            # 正则解释：行首是字母 紧接冒号
            if re.match(r'^[A-Z]:', cleaned_line):
                if cleaned_line.startswith('M:') or cleaned_line.startswith('K:'):
                    filtered_lines.append(cleaned_line)
                is_header = True
            else:
                # 非Header行，即旋律体
                filtered_lines.append(cleaned_line)

        # 只有当既有Header又有旋律时才保留 (防止空歌曲)
        if len(filtered_lines) > 2: 
            song_text = '\n'.join(filtered_lines)
            processed_songs.append(song_text)

    return processed_songs

def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: {RAW_DATA_DIR} not found.")
        return

    all_songs = []
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.abc')]
    
    print(f"Scanning {len(files)} files...")

    for filename in files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        songs = process_file(filepath)
        all_songs.extend(songs)

    # 写入文件
    # 使用 <|endoftext|> 作为分隔符
    # 注意：这个分隔符是在清洗后添加的，不受白名单限制，这是正确的逻辑。
    full_content = '\n<|endoftext|>\n'.join(all_songs) + '\n<|endoftext|>'
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"Strict filtering complete.")
    print(f"Valid Songs Extracted: {len(all_songs)}")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # 数据验证 (Sanity Check)
    print("-" * 30)
    print("Sample Output (First 5 lines):")
    print("\n".join(full_content.split('\n')[:5]))
    print("-" * 30)

if __name__ == "__main__":
    main()