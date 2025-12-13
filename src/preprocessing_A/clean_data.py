import os
import sys
from collections import Counter
import re

# ================= 1. 环境与导入 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

try:
    from src import path
    from src.ChordGenerator_A import config
except ImportError as e:
    # Fallback for standalone execution
    class ConfigMock:
        PAD_TOKEN = "<PAD>"
        SOS_TOKEN = "<SOS>"
        EOS_TOKEN = "<EOS>"
        BAR_TOKEN = "<BAR>"
    config = ConfigMock()

# ================= 2. 配置 =================
INPUT_FILE = os.path.join(path.DATA_INTERIM_DIR, "training_data_aligned.txt")
OUTPUT_FILE = os.path.join(path.DATA_INTERIM_DIR, "training_data_cleaned.txt")

# 阈值设定：允许总扣分在 100 以内 (既然 EmptyBar 只有 30，两三个空小节是可以存活的)
SUSPICION_THRESHOLD = 100

PENALTY_SCORES = {
    "LowNote": 10,       
    "HighNote": 10,
    "Chromatic": 25,
    "WeirdChord": 15,
    "EmptyBar": 30,      # [用户指令] 放宽至 30
    "NoHarmony": 999,    # 致命错误：全曲无和弦
    "WrongKey": 100,
    "BadRhythm_Hard": 100,
    "BadRhythm_Soft": 50,
}

# [新增配置] 合法小节长度白名单
# 涵盖: 1拍(4), 2拍(8), 2拍半(10), 3拍(12), 3拍半(14), 4拍(16), 4拍半(18), 5拍(20), 6拍(24), 8拍(32)
VALID_FULL_BAR_LENGTHS = {4, 8, 10, 12, 14, 16, 18, 20, 24, 32}

DIATONIC_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}
SAFE_ROOTS = {"C", "D", "E", "F", "G", "A", "B", "Bb", "A#", "Eb", "D#"}
RESERVED_TOKENS = {
    config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, 
    "_", config.BAR_TOKEN, "0", "|", "N.C."
}

# ================= 3. 核心工具类 =================

def transpose_melody_seq(seq, semitones):
    """移调工具 (排除休止符 '0' 和特殊符)"""
    new_seq = []
    for token in seq:
        if token.isdigit() and token != "0":
            try:
                val = int(token)
                new_seq.append(str(val + semitones))
            except:
                new_seq.append(token)
        else:
            new_seq.append(token)
    return new_seq

class SuspicionDetector:
    def __init__(self):
        self.dropped_count = 0
        self.rescued_low_count = 0
        self.rescued_high_count = 0
        self.reasons = Counter()

    def check(self, melody_seq, harmony_seq):
        score = 0
        reasons = []
        
        # --- Pre-Check: 全曲和弦有效性 ---
        real_chords = [t for t in harmony_seq if t not in RESERVED_TOKENS and t != "0"]
        if len(real_chords) == 0:
            return 9999, ["NoHarmony"]

        # A. 结构一致性与非法长度检查 (V3.1 Revised)
        bar_indices = [i for i, x in enumerate(melody_seq) if x == config.BAR_TOKEN]
        
        if len(bar_indices) > 2:
            segment_lengths = []
            start = 0
            for end in bar_indices:
                length = end - start
                segment_lengths.append(length)
                start = end + 1 
            
            # segment_lengths = [Bar1, Bar2, ..., BarN]
            # 丢弃最后一个小节（通常是不完全小节），检查主体部分
            if len(segment_lengths) > 1:
                # 只检查中间的小节 (Bar 2 到 Bar N-1)
                # 因为 Bar 1 可能是弱起，Bar N 可能是截断，不参与“非法长度”校验
                middle_bars = segment_lengths[1:-1]
                
                # [原逻辑保留] 拍号稳定性检查
                unique_lengths = set(middle_bars)
                if len(unique_lengths) > 2:
                    score += PENALTY_SCORES["BadRhythm_Hard"] # +100
                    reasons.append(f"InconsistentBar(Hard)")
                elif len(unique_lengths) == 2:
                    score += PENALTY_SCORES["BadRhythm_Soft"] # +50 (允许一次变奏)
                    reasons.append(f"InconsistentBar(Soft)")

                # [新逻辑] 非法长度精准打击
                # 只要中间出现任何一个奇怪长度 (如 7, 9, 13, 22)，直接处决
                for l in middle_bars:
                    if l not in VALID_FULL_BAR_LENGTHS:
                        score += 100
                        reasons.append(f"WeirdLength({l})")
                        break # 发现一个脏小节就足够了

        # B. 旋律检查
        total_notes = 0
        chromatic_notes = 0
        
        for token in melody_seq:
            if token.isdigit() and token != "0":
                val = int(token)
                total_notes += 1
                
                # [Expert Fix] 下调至 48 (C3)。
                # 48 是“小字组 C”，是这一音域归一化策略下的合理底线。
                if val < 45: 
                    score += PENALTY_SCORES["LowNote"]
                    reasons.append(f"LowNote({val})")
                
                if val > 81: 
                    score += PENALTY_SCORES["HighNote"]
                    reasons.append(f"HighNote({val})")
                    
                if (val % 12) not in DIATONIC_PITCH_CLASSES:
                    chromatic_notes += 1

        if total_notes > 0:
            chromatic_ratio = chromatic_notes / total_notes
            if chromatic_ratio > 0.15:
                score += PENALTY_SCORES["WrongKey"]
                reasons.append(f"WrongKey({chromatic_ratio:.1%})")
            elif chromatic_ratio > 0.05:
                score += PENALTY_SCORES["Chromatic"]
                reasons.append(f"Chromatic({chromatic_ratio:.1%})")

        # C. 和弦与小节空置检查 (保持 V3.0 逻辑)
        current_bar_chords = []
        for token in harmony_seq:
            if token == config.BAR_TOKEN:
                if current_bar_chords:
                    is_pure_rest = all(t == "0" for t in current_bar_chords)
                    if is_pure_rest:
                        score += PENALTY_SCORES["EmptyBar"]
                        reasons.append("EmptyBar(Rest)")
                current_bar_chords = []
                continue
            current_bar_chords.append(token)
            
            if token not in RESERVED_TOKENS and token != "0":
                match = re.match(r"^([A-G][b#-]?)", token)
                if match:
                    root = match.group(1)
                    norm_root = root.replace("-", "b")
                    if norm_root not in SAFE_ROOTS:
                        score += PENALTY_SCORES["WeirdChord"]
                        reasons.append(f"WeirdChord({root})")

        return score, reasons

# ================= 4. 主程序 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print("🧹 [V3.0] 启动数据清洗 - 策略修正版...")
    print(f"   - LowNote 阈值: < 52 (保护 A Minor E3)")
    print(f"   - EmptyBar 惩罚: 30分 (仅针对全休止 0)")
    
    detector = SuspicionDetector()
    valid_lines = []
    total_lines = 0
    length_mismatch_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split(" | ")
        if len(parts) != 2:
            continue
        m_seq = parts[0].split()
        h_seq = parts[1].split()

        total_lines += 1

        if len(m_seq) != len(h_seq):
            length_mismatch_count += 1
            continue

        # 1. 检测
        best_score, best_reasons = detector.check(m_seq, h_seq)
        best_m_seq = m_seq
        rescue_type = None

        # 2. 救援 (Low/High)
        has_lownote = any("LowNote" in r for r in best_reasons)
        has_highnote = any("HighNote" in r for r in best_reasons)

        if has_lownote:
            m_try = transpose_melody_seq(m_seq, 12)
            s_try, r_try = detector.check(m_try, h_seq)
            if s_try < best_score and s_try < SUSPICION_THRESHOLD:
                best_score = s_try
                best_reasons = r_try
                best_m_seq = m_try
                rescue_type = "low"

        if has_highnote:
            m_try_high = transpose_melody_seq(m_seq, -12)
            s_try_high, r_try_high = detector.check(m_try_high, h_seq)
            if s_try_high < best_score and s_try_high < SUSPICION_THRESHOLD:
                best_score = s_try_high
                best_reasons = r_try_high
                best_m_seq = m_try_high
                rescue_type = "high"

        # 3. 裁决
        if best_score >= SUSPICION_THRESHOLD:
            detector.dropped_count += 1
            cleaned_reasons = [r.split("(")[0] for r in best_reasons]
            detector.reasons.update(cleaned_reasons)
            continue

        if rescue_type == "low": detector.rescued_low_count += 1
        elif rescue_type == "high": detector.rescued_high_count += 1

        valid_lines.append(f"{' '.join(best_m_seq)} | {' '.join(h_seq)}")

    # 写入
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    print(f"\n📊 清洗报告 (V3.0 User Adjusted)")
    print(f"   - 原始: {total_lines}")
    print(f"   - ✅ 最终有效: {len(valid_lines)} ({(len(valid_lines)/max(1, total_lines)):.1%})")
    print(f"   - 🗑️ 剔除总数: {detector.dropped_count}")
    print(f"   - 🚑 救援 Low/High: {detector.rescued_low_count} / {detector.rescued_high_count}")
    print(f"\n💀 主要死因:")
    for r, c in detector.reasons.most_common(5):
        print(f"   ❌ {r}: {c}")

if __name__ == "__main__":
    main()