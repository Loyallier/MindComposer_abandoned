import music21
import os
import glob
from tqdm import tqdm
import math
import numpy as np
from collections import Counter

# ===========================
# 1. 基础配置与参数
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
OUTPUT_FILE = os.path.join(INTERIM_DIR, "training_data_aligned.txt")

# 特殊 Token
BAR_TOKEN = "<BAR>"
REST_TOKEN = "0"
HOLD_TOKEN = "_"

# 归一化参数 (用户指定)
TARGET_CENTROID = 63     # 目标重心 (55+86)/2
RANGE_TOLERANCE = 1      # 允许误差 (半音)
HARD_MIN_PITCH = 41        # 严格下限 (G2)
HARD_MAX_PITCH = 86        # 严格上限 (C#7)

os.makedirs(INTERIM_DIR, exist_ok=True)

# ===========================
# 2. 核心逻辑函数
# ===========================

def get_melody_pitches(stream_obj):
    """
    严谨提取旋律音高：
    1. 如果有多声部，只取 Part 0 (主旋律)。
    2. 仅提取 Note 和 Chord 的最高音(双音情况)。
    3. 排除伴奏符号 (Harmony) 和打击乐。
    """
    # 1. 声部隔离 (Isolate Melody Part)
    if stream_obj.hasPartLikeStreams():
        # 通常 Part 0 是高音/主旋律，Part 1 是低音/伴奏
        melody_stream = stream_obj.parts[0]
    else:
        melody_stream = stream_obj

    pitches = []
    # 2. 扁平化提取 (只针对旋律层)
    # recurse() 比 flatten() 更稳健，排除 Metadata
    for el in melody_stream.recurse().notes:
        if isinstance(el, music21.note.Note):
            pitches.append(el.pitch.midi)
        elif isinstance(el, music21.chord.Chord):
            # 对于旋律中的双音/和弦，取最高音作为旋律轮廓
            pitches.append(el.pitches[-1].midi)
            
    return pitches

def analyze_and_transpose(score, song_id="Unknown"):
    """
    处理管线：提取旋律指纹 -> 调性归一 -> 完整性校验 -> 重心归一(70.5) -> 边界校验
    """
    try:
        # --- 步骤 1: 提取原始旋律指纹 ---
        raw_pitches = get_melody_pitches(score)
        
        if not raw_pitches:
            return None, "Empty Melody"

        raw_min = min(raw_pitches)
        raw_max = max(raw_pitches)
        raw_range = raw_max - raw_min 

        # --- 步骤 2: 调性归一化 (Key Norm) ---
        # 注意：analyze('key') 会分析所有声部，这是正确的，因为调性是全局的
        key = score.analyze("key")
        
        if key.mode in ['minor', 'dorian', 'phrygian', 'locrian']:
            target_tonic_name = 'A'
        else:
            target_tonic_name = 'C'

        target_key = music21.key.Key(target_tonic_name)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        
        # 整首曲子转调
        transposed_score = score.transpose(interval)

        # --- 步骤 3: 完整性严厉检测 (Integrity Check) ---
        # 再次提取旋律指纹 (注意：必须重新用 get_melody_pitches 提取)
        trans_pitches = get_melody_pitches(transposed_score)
        
        if not trans_pitches: 
            return None, "Lost Notes After Transpose"

        trans_min = min(trans_pitches)
        trans_max = max(trans_pitches)
        trans_range = trans_max - trans_min

        # 校验：只允许 1.5 半音的误差
        diff = abs(raw_range - trans_range)
        if diff > RANGE_TOLERANCE:
            return None, f"Range Integrity Fail (Diff: {diff:.2f})"

        # --- 步骤 4: 基于旋律重心的八度归一化 (Centroid Norm) ---
        # 使用旋律的中值，而不是整个 Score 的中值
        current_centroid = (trans_max + trans_min) / 2.0
        
        # 计算距离目标 (70.5) 的差值
        dist_to_target = TARGET_CENTROID - current_centroid
        
        # 四舍五入到最近的八度
        octaves_to_shift = round(dist_to_target / 12.0)
        shift_amount = int(octaves_to_shift * 12)

        if shift_amount != 0:
            final_score = transposed_score.transpose(shift_amount)
        else:
            final_score = transposed_score

        # --- 步骤 5: 最终边界检查 ---
        final_pitches = get_melody_pitches(final_score)
        f_min = min(final_pitches)
        f_max = max(final_pitches)
        
        # 你的严格边界: 44-97
        if f_min < HARD_MIN_PITCH or f_max > HARD_MAX_PITCH:
            return None, f"Bounds Violation ({f_min}-{f_max})"

        return final_score, "PASS"

    except Exception as e:
        return None, f"Music21 Exception: {str(e)}"

def sample_stream(score, step_size=0.25):
    """
    [V5.0 修正版] 双轨采样：Melody + Chord
    修复：返回两个列表以匹配 m, c = sample_stream(...)
    """
    melody_tokens = []
    chord_tokens = []
    
    # 1. 结构清洗
    try:
        # 消除 Part/Voice 层级，强制暴露 Measure
        parts = music21.instrument.partitionByInstrument(score)
        if parts:
            # 如果有多乐器，通常取第一个作为主旋律及其和弦
            score_to_process = parts.parts[0]
        else:
            score_to_process = score
            
        score_to_process = score_to_process.makeMeasures()
        measures = list(score_to_process.recurse().getElementsByClass(music21.stream.Measure))
    except Exception as e:
        # 结构极度混乱，无法提取小节
        return [], []

    if not measures:
        return [], []

    # 记录上一个时刻的和弦，用于填充 "_"
    last_chord = "N.C." 

    for m in measures:
        m_len = m.duration.quarterLength
        steps = int(round(m_len / step_size))
        if steps <= 0: continue
        
        # === 关键：分别获取小节内的 Note 和 ChordSymbol ===
        # 使用 flat 获取该小节内所有元素，按 offset 排序
        m_flat = m.flat
        
        # 预抓取该小节所有和弦符号
        # ABC 文件中和弦通常是 Harmony 对象
        chord_objs = list(m_flat.getElementsByClass(music21.harmony.ChordSymbol))
        
        for i in range(steps):
            offset = i * step_size
            current_abs_offset = m.offset + offset # 绝对时间（备用）
            
            # --- Track 1: Melody ---
            melody_token = "0" # Default: Rest
            
            # 获取当前时间点的音符
            elements = m_flat.getElementsByOffset(offset, mustBeginInSpan=False)
            
            # A. Attack Detection
            attack_found = False
            for el in elements:
                if abs(el.offset - offset) < 0.01: # 刚好在这里开始
                    if isinstance(el, music21.note.Note):
                        melody_token = str(el.pitch.midi)
                        attack_found = True
                        break
                    elif isinstance(el, music21.chord.Chord):
                        # 取最高音
                        melody_token = str(el.sortAscending().notes[-1].pitch.midi)
                        attack_found = True
                        break
            
            # B. Sustain Detection
            if not attack_found:
                for el in elements:
                    if isinstance(el, (music21.note.Note, music21.chord.Chord)):
                        # 如果当前点在音符持续范围内 (start < now < end)
                        if el.offset < offset < (el.offset + el.duration.quarterLength):
                            melody_token = "_"
                            break
            
            melody_tokens.append(melody_token)

            # --- Track 2: Chord ---
            # 逻辑：查找当前 offset 及其之前最近的一个和弦符号
            # ABC 的和弦通常只在变化时标记，所以需要保持状态
            
            found_new_chord = False
            # 在当前极短的时间窗内是否有新和弦开始？
            for ch in chord_objs:
                if abs(ch.offset - offset) < 0.01:
                    # 简化和弦名称：把 "Am7" 变成 "Am" 或保留原样
                    # 这里先用 figure (e.g., "C", "G7")
                    last_chord = ch.figure
                    found_new_chord = True
                    break
            
            if found_new_chord:
                chord_tokens.append(last_chord)
            else:
                chord_tokens.append("_") # 和弦延续

        # --- End of Measure ---
        melody_tokens.append("<BAR>")
        chord_tokens.append("<BAR>") # 和弦序列也要对齐小节线
        
    return melody_tokens, chord_tokens

def process_all_files():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    print(f"🚀 开始处理 {len(files)} 个文件...")
    print(f"   - 目标重心 (Target Centroid): MIDI {TARGET_CENTROID}")
    print(f"   - 严格边界 (Hard Bounds): {HARD_MIN_PITCH} ~ {HARD_MAX_PITCH}")

    stats = {
        "processed": 0,
        "valid": 0,
        "dropped": 0
    }
    drop_reasons = Counter()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for file_path in tqdm(files, desc="Processing"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                stream_obj = music21.converter.parse(content, format="abc")
                
                # 处理单曲 vs 多曲
                songs = []
                if isinstance(stream_obj, music21.stream.Opus):
                    songs = list(stream_obj)
                else:
                    songs = [stream_obj]

                for i, song in enumerate(songs):
                    stats["processed"] += 1
                    
                    # 1. 分析与转调
                    final_score, message = analyze_and_transpose(song)

                    if final_score is None:
                        stats["dropped"] += 1
                        # 记录错误原因关键词
                        if "Bounds" in message: reason = "Bounds Violation"
                        elif "Range" in message: reason = "Range Integrity Fail"
                        elif "Empty" in message: reason = "Empty Score"
                        else: reason = "Other Error"
                        drop_reasons[reason] += 1
                        continue

                    # 2. 采样序列化
                    try:
                        m, c = sample_stream(final_score)
                        
                        if len(m) < 10:
                            stats["dropped"] += 1
                            drop_reasons["Too Short"] += 1
                            continue

                        line = f"{' '.join(m)} | {' '.join(c)}"
                        f_out.write(line + "\n")
                        stats["valid"] += 1
                    except Exception:
                        stats["dropped"] += 1
                        drop_reasons["Sampling Error"] += 1

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

    # === 最终统计 ===
    print("\n" + "="*40)
    print(f"📊 最终处理报告")
    print("="*40)
    print(f"总处理曲目: {stats['processed']}")
    print(f"✅ 成功输出:   {stats['valid']} ({(stats['valid']/max(1, stats['processed']))*100:.1f}%)")
    print(f"❌ 剔除数量:   {stats['dropped']}")
    print("-" * 40)
    print("📉 剔除原因分布:")
    for reason, count in drop_reasons.most_common():
        print(f"   - {reason}: {count}")
    print("="*40)
    print(f"结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_files()