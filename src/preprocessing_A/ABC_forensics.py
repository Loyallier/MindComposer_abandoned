import music21
import os
import glob
import math
import numpy as np
from collections import Counter
from tqdm import tqdm

# === 配置路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# === 阈值设置 ===
CONFIDENCE_THRESHOLD = 0.5  # 调性分析置信度阈值 (0-1)，低于此值报警
TARGET_CENTROID = 66        # 目标重心 MIDI (F#4)
PIANO_MIN = 21              # A0
PIANO_MAX = 108             # C8

def analyze_strict(score, filename, song_id):
    """
    对单首曲目进行深层核磁共振检查
    返回: (Status, Details_Dict)
    """
    try:
        # --- 1. 原始指纹提取 ---
        # 修复点：使用 p.pitch.midi 而不是 p.midi
        raw_elements = [
            p.pitch.midi for p in score.flatten().getElementsByClass(music21.note.Note)
        ]
        
        # 如果是空曲子 (只有休止符或和弦?)
        if not raw_elements:
            # 尝试提取和弦最高音作为补充
            chords = score.flatten().getElementsByClass(music21.chord.Chord)
            if chords:
                 for c in chords:
                     raw_elements.append(c.pitches[-1].midi)
            
            if not raw_elements:
                return "REJECT", {"reason": "Empty Score (No Notes)"}

        raw_min = min(raw_elements)
        raw_max = max(raw_elements)
        raw_range = raw_max - raw_min
        
        # --- 2. 调性与置信度分析 ---
        key = score.analyze("key")
        confidence = key.correlationCoefficient
        
        confidence_flag = confidence < CONFIDENCE_THRESHOLD

        # 策略：大调类->C, 小调类->A
        if key.mode in ['minor', 'dorian', 'phrygian', 'locrian']:
            target_tonic = 'A'
        else:
            target_tonic = 'C'
        
        target_key = music21.key.Key(target_tonic)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        
        # 执行转调
        transposed_score = score.transpose(interval)

        # --- 3. 转调后指纹核验 (Integrity Check) ---
        trans_elements = [
            p.pitch.midi for p in transposed_score.flatten().getElementsByClass(music21.note.Note)
        ]
        
        # 如果转调后找不到音符，尝试找和弦
        if not trans_elements:
             chords = transposed_score.flatten().getElementsByClass(music21.chord.Chord)
             for c in chords:
                 trans_elements.append(c.pitches[-1].midi)

        if not trans_elements: 
            return "REJECT", {"reason": "Transposition Lost Notes"}

        trans_min = min(trans_elements)
        trans_max = max(trans_elements)
        trans_range = trans_max - trans_min
        
        # [严厉检测]：音域宽度必须绝对一致
        if raw_range != trans_range:
             # 允许 1 个半音的微小误差 (music21 处理变音记号时的偶发行为)
             if abs(raw_range - trans_range) > 1:
                 return "REJECT", {
                     "reason": "Structure Integrity Fail", 
                     "info": f"Range {raw_range}->{trans_range}"
                 }

        # --- 4. 重心归一化 ---
        current_centroid = (trans_max + trans_min) / 2.0
        diff = TARGET_CENTROID - current_centroid
        octaves_shift = round(diff / 12.0)
        shift_amount = int(octaves_shift * 12)
        
        final_score = transposed_score.transpose(shift_amount)
        
        # 获取最终音高
        final_elements = [p.pitch.midi for p in final_score.flatten().getElementsByClass(music21.note.Note)]
        if not final_elements:
             chords = final_score.flatten().getElementsByClass(music21.chord.Chord)
             for c in chords:
                 final_elements.append(c.pitches[-1].midi)
        
        final_min = min(final_elements)
        final_max = max(final_elements)

        # --- 5. 最终边界检查 ---
        if final_min < PIANO_MIN or final_max > PIANO_MAX:
            return "REJECT", {
                "reason": "Out of Bounds",
                "info": f"Final Range {final_min}-{final_max}"
            }

        # --- 6. 成功返回 ---
        return "PASS", {
            "file": filename,
            "id": song_id,
            "orig_key": f"{key.tonic.name}{key.mode}",
            "confidence": round(confidence, 2),
            "confidence_low": confidence_flag,
            "octave_shift": int(octaves_shift),
            "final_min": final_min,
            "final_max": final_max,
            "final_mean": round(np.mean(final_elements), 2)
        }

    except Exception as e:
        return "ERROR", {"reason": f"{type(e).__name__}: {str(e)}"}


def run_comprehensive_diagnosis():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    print(f"🕵️  全面诊断启动 (v2修正版): 扫描 {len(files)} 个文件集...")
    print(f"    - 目标重心: MIDI {TARGET_CENTROID}")
    print("="*60)

    stats = {
        "total_songs": 0,
        "passed": 0,
        "rejected": 0,
        "errors": 0
    }
    
    reject_reasons = Counter()
    confidence_warnings = 0
    shift_distribution = Counter()
    logs = []
    
    # 新增：记录 Crash 的具体原因，防止再次全军覆没
    crash_reasons = Counter()

    for file_path in tqdm(files, desc="Diagnosing"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            stream = music21.converter.parse(content, format="abc")
            
            songs = []
            if isinstance(stream, music21.stream.Opus):
                songs = list(stream)
            else:
                songs = [stream]

            for i, song in enumerate(songs):
                stats["total_songs"] += 1
                song_id = song.id if hasattr(song, 'id') else f"song_{i}"

                status, data = analyze_strict(song, filename, song_id)

                if status == "PASS":
                    stats["passed"] += 1
                    shift_distribution[data["octave_shift"]] += 1
                    if data["confidence_low"]:
                        confidence_warnings += 1
                
                elif status == "REJECT":
                    stats["rejected"] += 1
                    reject_reasons[data["reason"]] += 1
                    if "Structure" in data["reason"] or "Bounds" in data["reason"]:
                        logs.append(f"❌ {data['reason'].ljust(25)} | {filename} | {data.get('info', '')}")
                
                elif status == "ERROR":
                    stats["errors"] += 1
                    reject_reasons["Crash"] += 1
                    crash_reasons[data["reason"]] += 1

        except Exception as e:
            print(f"FILE LOAD ERROR: {filename} - {e}")

    # === 生成报告 ===
    print("\n" + "="*60)
    print(f"📊 诊断报告摘要 (Total Songs Checked: {stats['total_songs']})")
    print("="*60)
    
    print(f"✅ 通过 (Passed):   {stats['passed']} ({(stats['passed']/max(1, stats['total_songs']))*100:.1f}%)")
    print(f"❌ 拒绝 (Rejected): {stats['rejected']}")
    print(f"⚠️ 错误 (Errors):   {stats['errors']}")
    print("-" * 30)
    
    if stats['rejected'] > 0:
        print("📉 拒绝原因细分:")
        for reason, count in reject_reasons.most_common():
            if reason != "Crash":
                print(f"   - {reason}: {count}")

    if stats['errors'] > 0:
        print("\n💀 Crash 具体报错信息 (Top 5):")
        for reason, count in crash_reasons.most_common(5):
            print(f"   - {reason}: {count} 例")

    print("-" * 30)
    print(f"🧠 调性分析预警 (Low Confidence < {CONFIDENCE_THRESHOLD}): {confidence_warnings} 例")
    print("-" * 30)
    print("🏗️  八度搬运分布:")
    for shift in sorted(shift_distribution.keys()):
        print(f"   {shift:+d} Octaves: {shift_distribution[shift]} songs")

    print("="*60)
    if logs:
        print("🔍 拒绝样本示例:")
        for l in logs[:5]:
            print(l)
    print("="*60)

if __name__ == "__main__":
    run_comprehensive_diagnosis()