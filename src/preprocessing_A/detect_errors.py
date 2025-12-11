import music21
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# 路径配置 (复用你的配置)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def get_stats(score, name):
    """提取一个流的统计特征：平均音高、最低音、最高音"""
    pitches = [p.midi for p in score.flatten().pitches if isinstance(p, music21.pitch.Pitch)]
    if not pitches:
        return None
    return {
        "mean": np.mean(pitches),
        "min": np.min(pitches),
        "max": np.max(pitches),
        "range": np.max(pitches) - np.min(pitches)
    }

def run_diagnosis():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    # 为了快速测试，你可以切片 files[:50]
    
    report_data = []

    print("🕵️ 正在进行归一化逻辑诊断...")
    
    for file_path in tqdm(files, desc="Scanning"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # 解析原始流
            s_orig = music21.converter.parse(content, format="abc")
            
            # 如果包含多首曲子，这里只取第一首做测试
            score = s_orig[0] if isinstance(s_orig, music21.stream.Opus) else s_orig

            # --- 阶段 0: 原始数据 ---
            stats_orig = get_stats(score, "Original")
            if not stats_orig: continue

            # --- 阶段 1: 仅调性归一化 (Key Norm) ---
            key = score.analyze("key")
            mode = key.mode
            if mode == "minor":
                target = music21.key.Key("a", "minor")
            else:
                target = music21.key.Key("C", "major")
            interval = music21.interval.Interval(key.tonic, target.tonic)
            s_key = score.transpose(interval)
            
            stats_key = get_stats(s_key, "KeyNorm")

            # --- 阶段 2: 你的电梯算法 (Octave Norm) ---
            # 复刻你的逻辑
            melody_pitches = [p.midi for p in s_key.flatten().pitches] # 简化版
            current_min = min(melody_pitches)
            TARGET_FLOOR = 55
            shift_amount = 0
            if current_min < TARGET_FLOOR:
                diff = TARGET_FLOOR - current_min
                octaves = np.ceil(diff / 12.0)
                shift_amount = int(octaves * 12)
            
            s_octave = s_key.transpose(shift_amount)
            stats_final = get_stats(s_octave, "Final")

            report_data.append({
                "file": filename,
                "orig_key": f"{key.tonic.name} {key.mode}",
                "orig_mean": stats_orig['mean'],
                "key_mean": stats_key['mean'], # 观察转调后平均音高是否剧烈波动
                "final_mean": stats_final['mean'],
                "final_min": stats_final['min'],
                "final_max": stats_final['max'],
                "shifted": shift_amount
            })

        except Exception as e:
            continue

    # 转为 DataFrame 分析
    df = pd.DataFrame(report_data)
    
    print("\n📊 诊断报告摘要:")
    print("-" * 30)
    print(f"总文件数: {len(df)}")
    print(f"最终平均音高 (Mean) 的标准差: {df['final_mean'].std():.2f} (越小越好)")
    print(f"最终最低音 (Min) 的范围: {df['final_min'].min()} - {df['final_min'].max()}")
    
    # 检测异常值：最终平均音高 偏离 60 (Central C) 太远的
    outliers = df[(df['final_mean'] < 53) | (df['final_mean'] > 77)]
    if not outliers.empty:
        print(f"\n⚠️ 发现 {len(outliers)} 个严重异常样本 (Mean < 53 或 > 77):")
        print(outliers[['file', 'orig_key', 'final_mean', 'final_min', 'shifted']].head())
    else:
        print("\n✅ 没有发现极端异常的平均音高偏移。")

if __name__ == "__main__":
    run_diagnosis()