import os
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ================= 配置 =================
# 指向你目前 interim 文件夹里存放 ABC 或者 TXT 的路径
# 注意：这里我们直接分析你 clean 之前的 "aligned" 数据，或者原始数据
# 为了准确，建议分析 "training_data_aligned.txt" (如果格式是 Token 化的)
DATA_FILE = r"E:\VScode_Programs\！Projects\AI_AS1_G\data\interim\training_data_aligned.txt"

# 目标：找到一个长度为 N 的窗口，能覆盖最多的音符
WINDOW_SIZES = [36, 40, 48]  # 测试 3个八度、3.5个八度、4个八度

def load_data(filepath):
    """读取所有旋律 Token"""
    all_melodies = []
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"📂 正在加载 {len(lines)} 条数据...")
    for line in lines:
        if "|" not in line: continue
        melody_part = line.split("|")[0].strip()
        tokens = melody_part.split()
        # 提取 MIDI 音高 (排除 0, _, BAR)
        pitches = [int(t) for t in tokens if t.isdigit() and t != '0']
        if pitches:
            all_melodies.append(pitches)
            
    return all_melodies

def simulate_best_octave(melodies, target_center=66):
    """
    模拟：如果把每首歌都平移到离 target_center 最近的八度，
    全局的音高范围会变成什么样？
    """
    shifted_pitches = []
    shifts_applied = []
    
    for song in melodies:
        if not song: continue
        avg_pitch = sum(song) / len(song)
        
        # 计算需要移动多少个八度才能接近 target_center (F4/F#4)
        diff = target_center - avg_pitch
        shift_octaves = round(diff / 12)
        shift_semitones = int(shift_octaves * 12)
        
        # 应用平移
        new_song = [p + shift_semitones for p in song]
        shifted_pitches.extend(new_song)
        shifts_applied.append(shift_octaves)
        
    return shifted_pitches, shifts_applied

def analyze():
    melodies = load_data(DATA_FILE)
    if not melodies: return

    # 1. 原始分布统计
    raw_pitches = [p for song in melodies for p in song]
    raw_min, raw_max = min(raw_pitches), max(raw_pitches)
    raw_unique = len(set(raw_pitches))
    
    print("\n📊 --- 原始数据现状 ---")
    print(f"   音域范围: {raw_min} ~ {raw_max} (跨度 {raw_max - raw_min})")
    print(f"   词表大小: {raw_unique} 个不同音高")
    print(f"   LowNote (<52) 占比: {sum(1 for p in raw_pitches if p < 52) / len(raw_pitches):.2%}")
    
    # 2. 模拟最优收敛
    # 我们尝试将所有歌对齐到 MIDI 66 (F#4) —— 这是流行歌旋律的绝对中心
    opt_pitches, shifts = simulate_best_octave(melodies, target_center=66)
    opt_min, opt_max = min(opt_pitches), max(opt_pitches)
    opt_unique = len(set(opt_pitches))
    
    print("\n🧪 --- 模拟最优八度收敛 (Target Center = 66) ---")
    print(f"   优化后范围: {opt_min} ~ {opt_max} (跨度 {opt_max - opt_min})")
    print(f"   优化后词表: {opt_unique} 个不同音高")
    
    # 3. 覆盖率计算
    # 计算 [opt_min, opt_max] 区间内，抛弃首尾 1% 极端值后的核心区间
    sorted_p = sorted(opt_pitches)
    cut_idx = int(len(sorted_p) * 0.01) # 1%
    core_min = sorted_p[cut_idx]
    core_max = sorted_p[-cut_idx]
    
    print(f"   核心区间 (98% 覆盖): {core_min} ~ {core_max}")
    print(f"   建议 LowNote 阈值: {core_min}")
    print(f"   建议 HighNote 阈值: {core_max}")
    
    # 4. 推荐策略
    print("\n💡 --- 专家建议 (AI_G) ---")
    print(f"   1. 修改 normalization.py，将 TARGET_CENTROID 调整为 66 或 67 (F4/G4)。")
    print(f"   2. 70.5 (Bb4) 显然太高了，导致低音部被迫上移，变成了次中音。")
    print(f"   3. 在 clean_data.py 中，将 LowNote 设为 {core_min}，HighNote 设为 {core_max}。")

if __name__ == "__main__":
    analyze()