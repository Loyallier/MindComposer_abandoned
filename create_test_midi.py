import music21 as m21
from typing import List, Tuple, Union

# ----------------------------------------------------
# 原始数据
# ----------------------------------------------------

# '60' 等是 MIDI 音高。'_' 表示休止符或延长音。
# 每一行代表一个固定的时间单位（例如：一个四分音符 Quarter Length = 1.0）
MELODY_DATA: List[Union[int, str]] = [
    60, '_', '_', '_', 
    62, '_', '_', '_', 
    64, '_', '_', '_', 
    65, '_', '_', '_', 
    67, '_', '_', '_', 
    69, '_', '_', '_', 
    71, '_', '_', '_', 
    72, '_', '_', '_', 
    '_', '_', '_', '_',
    '_', '_', '_', '_',
    0 # 特殊标记 0 表示休止
]

# 对应的和弦标签。'_' 表示和弦持续。
CHORD_DATA: List[str] = [
    'G7', '_', '_', 'C',
    '_', '_', 'C', '_',
    'F', '_', 'C', '_',
    'F', '_', 'B', 'B', # 注意：B 和弦可能需要修正为 Bmaj 或 Bmin 以符合规范，这里按原样处理
    '_', '_', 'B', '_',
    '_', 'F', '_', '_',
    '_', 'F', 'F', 'F',
    'F', '_', '_', 'B',
    'F', '_', 'F', '_',
    'F', 'F', '_', 'B',
    'F' # 对应旋律的 0
]

# ----------------------------------------------------
# 步骤二：数据处理函数
# ----------------------------------------------------
def create_score_from_data(
    melody_pitches: List[Union[int, str]], 
    chord_labels: List[str], 
    output_path: str = "test_melody.mid"
) -> str:
    """
    根据旋律和和弦数据创建一个 music21 Score 对象并保存为 MIDI 文件。
    
    旋律部分: 将 MIDI 音高转换为 music21 Note 对象。
    和弦部分: 将和弦标签转换为 music21 Harmony/ChordSymbol 对象。
    """
    # 假设每个时间步长为 1.0 (四分音符 Quarter Length)
    STEP_DURATION = 1.0
    
    # 1. 创建旋律 Part
    melody_part = m21.stream.Part()
    last_note = None # 用于处理持续音('_')
    
    for pitch in melody_pitches:
        if pitch == '_':
            # 延长上一个音符的持续时间
            if last_note:
                last_note.duration.quarterLength += STEP_DURATION
        elif pitch == 0:
            # 标记为 0 的休止符
            rest = m21.note.Rest(quarterLength=STEP_DURATION)
            melody_part.append(rest)
            last_note = None # 重置
        else:
            # 新音符
            note = m21.note.Note(pitch)
            note.quarterLength = STEP_DURATION
            melody_part.append(note)
            last_note = note
            
    # 2. 创建和弦 Stream (用于和弦标记)
    chord_stream = m21.stream.Measure()
    
    # 和弦 Stream 应该独立于旋律，只包含和弦记号。
    # 旋律和和弦数据的长度必须相等
    if len(melody_pitches) != len(chord_labels):
        raise ValueError("旋律和和弦数据长度不匹配！")
        
    last_chord = None
    
    for label in chord_labels:
        if label == '_':
            # 延续上一个和弦，不插入新的和弦记号
            if last_chord:
                last_chord.quarterLength += STEP_DURATION
        else:
            # 遇到新的和弦记号或非 '_' 标签
            # 结束上一个和弦的持续时间（如果存在）
            if last_chord and last_chord.quarterLength == 0:
                 last_chord.quarterLength = STEP_DURATION
                 
            try:
                # 使用 m21.harmony.ChordSymbol 来标记和弦
                new_chord = m21.harmony.ChordSymbol(label)
            except m21.harmony.HarmonyException:
                # 无法解析的和弦（例如：B，应该用 Bmaj 或 Bm）
                print(f"警告：无法解析和弦标签 '{label}'，使用默认 Maj.")
                new_chord = m21.harmony.ChordSymbol(f"{label}maj")
                
            new_chord.quarterLength = STEP_DURATION # 初始时长
            chord_stream.append(new_chord)
            last_chord = new_chord


    # 3. 创建 Score 对象并插入两个 Part
    final_score = m21.stream.Score()
    # 插入和弦 Part/Stream
    # 注意：为了让 MIDI 文件能够被 m21.converter.parse() 成功读取为包含旋律的 Part，
    # 我们只将旋律 Part 写入 MIDI，和弦标记通常不会被 MIDI 阅读器作为音符。
    # 旋律 MIDI 文件通常只包含旋律 Part。
    # 
    # 但是，为了测试 A 组的输入（如果 A 组需要旋律和和弦信息），我们可以只返回旋律 Part，
    # 或者返回一个包含旋律和文本/和弦 Stream 的 Score。
    
    # A 组的输入是 melody_midi_path，所以这个 MIDI 文件应该只包含旋律。
    
    # 4. 写入 MIDI 文件（只包含旋律 Part）
    melody_part.write('midi', fp=output_path)

    print(f"✅ 已成功创建仅包含旋律的 MIDI 文件: {output_path}")
    print("注意：和弦标记未写入 MIDI，因为 MIDI 文件只用于传递旋律给 A 组预测。")
    print("如果需要查看和弦标记，请查看 Score 文件 (如 MusicXML)")
    
    return output_path

# ----------------------------------------------------
# 步骤三：执行创建
# ----------------------------------------------------
if __name__ == "__main__":
    
    # 使用函数创建文件
    file_path = create_score_from_data(MELODY_DATA, CHORD_DATA, "test_melody.mid")
    
    # 建议您用 MusicXML 格式输出一次，以便可视化检查旋律和和弦标记是否正确
    # score = create_score_from_data(MELODY_DATA, CHORD_DATA, "test_melody.xml")
    
    print("\n下一步：将 'test_melody.mid' 文件路径用于测试您的 interface.py。")