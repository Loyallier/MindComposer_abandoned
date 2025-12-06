import os
import glob
import math
from music21 import converter, note, chord, harmony, stream

def parse_abc_to_dataset(file_path, step_size=0.25):
    # --- 修复点 1: dataset 必须初始化为空列表 ---
    dataset = [] 

    # --- 预检查文件内容 ---
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.read(100) 
        if "X:" not in header and "T:" not in header:
            print(f"⏩ 跳过非乐谱文件: {os.path.basename(file_path)}")
            return []

    try:
        opus = converter.parse(file_path, format='abc', forceSource=True)
    except Exception as e:
        print(f"⚠️ 解析出错 {os.path.basename(file_path)}: {e}")
        return []

    if isinstance(opus, stream.Score):
        scores = [opus]
    else:
        scores = opus

    print(f"✅ 正在处理: {os.path.basename(file_path)} (包含 {len(scores)} 首)")

    for s in scores:
        try:
            s_flat = s.expandRepeats().flatten()
            if s_flat.duration.quarterLength < 4.0:
                continue

            total_steps = int(math.ceil(s_flat.duration.quarterLength / step_size))
            
            melody_seq = ['0'] * total_steps 
            chord_seq = ['_'] * total_steps 
            
            # --- 旋律处理 ---
            notes = s_flat.notesAndRests
            for n in notes:
                start_idx = int(round(n.offset / step_size))
                end_idx = int(round((n.offset + n.duration.quarterLength) / step_size))
                
                if start_idx >= total_steps: continue
                end_idx = min(end_idx, total_steps)
                
                if isinstance(n, note.Note):
                    melody_seq[start_idx] = str(n.pitch.midi)
                    for i in range(start_idx + 1, end_idx):
                        melody_seq[i] = '_' 
                elif isinstance(n, note.Rest):
                    for i in range(start_idx, end_idx):
                        melody_seq[i] = '0'

            # --- 和弦处理 ---
            chords = s_flat.getElementsByClass(harmony.ChordSymbol)
            for c in chords:
                start_idx = int(round(c.offset / step_size))
                if start_idx >= total_steps: continue
                chord_name = c.figure.split('/')[0] 
                chord_seq[start_idx] = chord_name
            
            if len(melody_seq) > 0 and len(chord_seq) > 0:
                dataset.append({
                    "melody": melody_seq,
                    "harmony": chord_seq
                })

        except Exception as e:
            continue
            
    return dataset

if __name__ == "__main__":
    # --- 修复点 2: 指定数据所在的子文件夹 ---
    # 假设你的 txt 文件在 dataset 文件夹里
    data_folder = "dataset" 
    
    # 检查文件夹是否存在
    if not os.path.exists(data_folder):
        # 如果 dataset 文件夹不存在，尝试在当前目录找
        print(f"⚠️ 没找到 '{data_folder}' 文件夹，尝试在当前目录查找...")
        data_folder = "."

    # 构建搜索路径：dataset/*.txt
    search_path_txt = os.path.join(data_folder, "*.txt")
    search_path_abc = os.path.join(data_folder, "*.abc")
    
    input_files = glob.glob(search_path_txt) + glob.glob(search_path_abc)
    
    # 排除输出文件
    input_files = [f for f in input_files if "training_data_aligned" not in f]

    print(f"📂 在 '{data_folder}' 中找到了 {len(input_files)} 个文件")

    all_data = []
    for f in input_files:
        data = parse_abc_to_dataset(f)
        all_data.extend(data)
        
    print(f"\n🎉 处理完成！共提取 {len(all_data)} 首曲目数据。")
    
    if len(all_data) > 0:
        output_filename = "training_data_aligned.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            for song in all_data:
                if isinstance(song, dict) and "melody" in song:
                    x_str = " ".join(song["melody"])
                    y_str = " ".join(song["harmony"])
                    f.write(f"{x_str}|{y_str}\n")
        print(f"💾 已保存至 {output_filename}")
    else:
        print("❌ 依然没有提取到数据。请确认：")
        print(f"1. 你的 morris.txt 等文件是否在 '{os.path.abspath(data_folder)}' 里面？")