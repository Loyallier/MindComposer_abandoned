import os
import sys
from collections import Counter

# ================= 配置 =================
# 请确保这里指向你当前正在使用的清洗后文件
INPUT_FILE = r"data/interim/training_data_cleaned.txt" 
# 或者你可以改为 training_data_aligned.txt 查看源头

def analyze_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"🔬 正在分析文件结构: {file_path}")
    print("=" * 60)

    total_songs = 0
    bar_length_counter = Counter()
    songs_with_giant_bars = 0
    songs_with_no_bars = 0
    
    # 记录前 3 个异常样本用于展示
    example_failures = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split(" | ")
            melody_seq = parts[0].split()
            
            total_songs += 1
            
            # --- 核心分析逻辑 ---
            # 1. 拆分小节
            # 我们假设 <BAR> 是分隔符。注意：如果末尾有 <BAR>，split 会产生一个空串，需过滤
            raw_segments = []
            current_seg = []
            bar_count = 0
            
            for token in melody_seq:
                if token == "<BAR>" or token == "config.BAR_TOKEN": # 兼容可能的变量名残留
                    bar_count += 1
                    if current_seg:
                        raw_segments.append(current_seg)
                    current_seg = []
                else:
                    current_seg.append(token)
            
            # 处理残留（如果最后没有 BAR）
            if current_seg:
                raw_segments.append(current_seg)

            if bar_count == 0:
                songs_with_no_bars += 1
            
            # 2. 统计每个片段的长度
            has_error = False
            song_bar_lengths = []
            
            for seg in raw_segments:
                length = len(seg)
                bar_length_counter[length] += 1
                song_bar_lengths.append(length)
                
                # 定义“巨大异常”：超过 32 个 token (即超过 2 个 4/4 小节)
                if length > 32:
                    has_error = True
            
            if has_error:
                songs_with_giant_bars += 1
                if len(example_failures) < 3:
                    example_failures.append({
                        "id": line_idx,
                        "bar_count": bar_count,
                        "lengths": song_bar_lengths
                    })

    # ================= 报告输出 =================
    print(f"📊 分析结果 (总歌曲数: {total_songs})")
    print("-" * 30)
    print(f"❌ 完全无 <BAR> 的歌曲: {songs_with_no_bars}")
    print(f"❌ 含有巨大畸形小节 (>32 tokens) 的歌曲: {songs_with_giant_bars}")
    
    print("\n[Token 长度分布 Top 10] (理想值应该是 16, 12, 8 等)")
    for length, count in bar_length_counter.most_common(100):
        divisible_by_4 = "✅" if length % 4 == 0 and length > 0 else "❌"
        # 16 = 4/4拍, 12 = 3/4拍 或 6/8拍
        remark = ""
        if length == 16: remark = "(标准 4/4)"
        elif length == 12: remark = "(3/4 或 6/8)"
        elif length == 8: remark = "(2/4)"
        
        print(f"   长度 {length:3d}: {count:5d} 次  | 整除4? {divisible_by_4} {remark}")

    print("\n[畸形样本示例]")
    for fail in example_failures:
        print(f"   🔹 Song ID {fail['id']}: 共 {fail['bar_count']} 个BAR。")
        print(f"      内部长度分布: {fail['lengths']}")
        print("      (典型症状: 长度动辄上百，说明中间的 BAR 丢了)")

if __name__ == "__main__":
    # 获取项目根目录，确保能找到文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 src/preprocessing_A/ 或类似位置，根据你的结构调整
    # 如果找不到文件，请手动修改 INPUT_FILE 为绝对路径
    analyze_file(INPUT_FILE)