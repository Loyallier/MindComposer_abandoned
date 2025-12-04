import re

def clean_chord_dataset(input_file, output_file):
    """
    清洗和弦数据集：
    1. 移除 这种元数据
    2. 移除旋律残留 (如 A2, B/2)
    3. 移除乐谱符号 (|, \)
    """
    
    cleaned_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"原始行数: {len(lines)}")
    
    for line in lines:
        # --- 步骤 1: 移除 标记 ---
        # 使用正则表达式匹配 ，并替换为空格
        line = re.sub(r'\\', ' ', line)
        
        # --- 步骤 2: 移除反斜杠 '\' ---
        # 修复之前的 SyntaxError：直接用 string.replace 更安全，不用正则
        line = line.replace('\\', ' ')
        
        # --- 步骤 3: 逐词过滤 ---
        tokens = line.strip().split()
        valid_tokens = []
        
        for token in tokens:
            # A. 过滤特殊符号
            if any(char in token for char in r'|()'): 
                continue
            
            # B. 过滤旋律时值 (如 /2, /4)
            # 正则解释: 斜杠后跟数字，通常是 ABC 的时值标记
            if re.search(r'/\d', token): 
                continue
                
            # C. 过滤包含数字但不是和弦数字的 (如 A2)
            # 真正的和弦数字通常是 7, 9, 11, 13, 5, 6, 4(sus4), 2(sus2)
            # 如果出现 A2 这种，大概率是旋律。为了保险，我们只允许常见的和弦数字。
            # 这里做一个简单的启发式过滤：如果包含数字，且数字前不是和弦性质字符，可能是旋律
            # 简单策略：如果长度只有2且第二个字符是数字 (如 A2, C3)，通常是旋律
            # (除了 E5, D5 这种强力和弦，但在民谣里很少见，即使误删也比保留脏数据好)
            if len(token) == 2 and token[1].isdigit() and token[1] not in ['7', '6', '9', '5']:
                 continue

            # D. 再次检查是否以 A-G 开头 (过滤掉像 "slowly" 这种词)
            if not re.match(r'^[A-G]', token):
                continue
                
            # E. (可选) 简化和弦：去掉转位 (如 D/F# -> D)
            # 这对初学者模型很有帮助，能显著提高准确率
            token = token.split('/')[0] 
            
            valid_tokens.append(token)
        
        # 如果这一行还有剩下的有效和弦，加入结果
        if valid_tokens:
            cleaned_lines.append(' '.join(valid_tokens))
            
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
            
    print(f"清洗完成！")
    print(f"有效数据行数: {len(cleaned_lines)}")
    # 打印前5行预览，让你放心
    print("--- 数据预览 (前5行) ---")
    for l in cleaned_lines[:5]:
        print(l)

# ================= 运行 =================
if __name__ == "__main__":
    # 确保文件名和你本地的一致
    INPUT_FILE = 'chords_dataset_for_training.txt' 
    OUTPUT_FILE = 'chords_dataset_clean.txt'
    
    try:
        clean_chord_dataset(INPUT_FILE, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到文件 {INPUT_FILE}，请确认文件路径是否正确。")