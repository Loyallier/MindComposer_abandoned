import os
import music21 as m21
from typing import List, Dict, Tuple
import logging
import math
from data_logic_B import CHORD_TYPES, PATTERN_TEMPLATES, parse_simplified_chord # 导入 Group B 知识库

# 初始化日志
logger = logging.getLogger("B_Decision")

# Group B Phase 3: Decision and Rendering
# 核心功能：根据旋律密度和风格选择织体，并生成 music21 Part。

# 1. 启发式决策：织体选择 (Texture Selector)
DENSITY_THRESHOLD = 0.4 # 旋律密度阈值：超过此值视为“繁忙”

def select_texture_pattern(style: str, density_score: float) -> str:
    """
    启发式算法：根据风格和旋律密度选择伴奏织体模板。
    """
    
    # 如果密度高（旋律繁忙），倾向于选择稀疏的伴奏，避免冲突
    if density_score > DENSITY_THRESHOLD:
        logger.debug(f"密度 {density_score:.2f} 较高，倾向于稀疏织体 (Sparse_Arpeggio)。")
        # 无论什么风格，都使用通用的稀疏琶音来避免冲突
        return "Sparse_Arpeggio" 
        
    # 如果密度低（旋律稀疏），使用风格默认的织体
    else:
        logger.debug(f"密度 {density_score:.2f} 较低，使用风格默认织体: {style}")
        return style

# 2. 预处理 A: 和弦片段整合与密度计算 (Pre-processing)
def _consolidate_chords(chord_sequence: List[str], quantization_step: float = 0.25) -> List[Tuple[str, float]]:
    """
    将 16分音符切片和弦序列 (e.g., ['C', '_', '_', 'G']) 
    整合为 (和弦标签, 时长 QL) 列表 [(C, 0.75), (G, 0.25)]。
    """
    consolidated = []
    if not chord_sequence:
        return consolidated
        
    current_chord = None
    current_duration = 0.0
    
    for chord in chord_sequence:
        if chord == '_':
            # 延音：增加当前和弦的时长
            current_duration += quantization_step
        elif current_chord is None:
            # 序列的第一个有效和弦
            current_chord = chord
            current_duration = quantization_step
        elif chord == current_chord:
            # 相同和弦：增加当前和弦的时长
            current_duration += quantization_step
        else:
            # 遇到新和弦：保存上一个和弦，并开始新的计时
            if current_duration > 0 and current_chord != '_':
                consolidated.append((current_chord, current_duration))
            
            # 重置计时器
            current_chord = chord
            current_duration = quantization_step
            
    # 确保最后一个和弦被添加
    if current_duration > 0 and current_chord != '_':
        consolidated.append((current_chord, current_duration))
        
    return consolidated

def _calculate_melody_density(melody_stream: m21.stream.Stream, consolidated_chords: List[Tuple[str, float]]) -> List[float]:
    """
    计算每个和弦片段内的旋律密度。
    密度 = (该片段内的音符数) / (片段时长 QL) / (参考速率)
    """
    density_scores = []
    current_offset = 0.0
    
    # 提取旋律中的所有音符和休止符
    notes_and_rests = melody_stream.flat.notesAndRests
    
    for chord_label, duration in consolidated_chords:
        segment_notes_count = 0
        segment_start = current_offset
        segment_end = current_offset + duration
        
        # 遍历旋律中的事件，计算落在当前片段内的音符数
        for element in notes_and_rests:
            element_start = element.offset
            
            # 检查是否是音符且其起始点在当前片段内 (使用小误差范围)
            if isinstance(element, m21.note.Note) or isinstance(element, m21.chord.Chord):
                if segment_start <= element_start < segment_end - 0.001: 
                    segment_notes_count += 1

        # 密度计算
        if duration > 0:
            # 使用一个缩放因子 (0.5) 和平方根来将密度值映射到 0-1 范围，并平滑变化
            density = math.sqrt(segment_notes_count / duration) * 0.7 
        else:
            density = 0.0
            
        # 确保密度不超过 1.0
        density_scores.append(min(density, 1.0))
        current_offset += duration
        
    return density_scores

# 3. 核心引擎：伴奏 Part 生成 (Accompaniment Renderer)
def generate_accompaniment_part(
    consolidated_chords: List[Tuple[str, float]], 
    melody_densities: List[float], 
    selected_style: str
) -> m21.stream.Part:
    """
    根据和弦序列、旋律密度和风格，生成左手伴奏 music21.Part。
    """
    accompaniment_part = m21.stream.Part()
    accompaniment_part.insert(0, m21.instrument.Piano()) # 设置乐器
    
    current_offset = 0.0
    num_chords = len(consolidated_chords)
    
    if num_chords != len(melody_densities):
        logger.error("和弦片段数量与密度数量不匹配，生成可能出错！")
        return accompaniment_part

    for i in range(num_chords):
        chord_label, segment_length = consolidated_chords[i]
        density = melody_densities[i]
        
        # 1. 解析和弦标签 (例如: 'G7' -> 'G', 'Dom7')
        root_name, chord_type = parse_simplified_chord(chord_label)
        
        if chord_type not in CHORD_TYPES:
            logger.warning(f"未知和弦类型: {chord_type}，跳过该片段。")
            current_offset += segment_length
            continue
            
        # 【新增】处理 NoChord/休止符号 '0'
        if chord_type == 'NoChord':
            logger.debug(f"和弦 {chord_label} 为 NoChord/休止，跳过该片段，不生成伴奏。")
            current_offset += segment_length
            continue
            
        offsets = CHORD_TYPES[chord_type]
        # 根音定在 C3 (MIDI 48) 八度，作为琶音或其他音符的基准
        # 注意：这里的 root_midi 是 C3 八度的根音 (例如 D3 = 50, A3 = 57)
        root_midi = m21.pitch.Pitch(root_name + '3').midi 
        
        logger.debug(f"--- DIAGNOSIS: 和弦 {chord_label}, 根音 {root_name}, 偏移量 {offsets} ---")

        # 2. 决策：选择织体模板
        pattern_name = select_texture_pattern(selected_style, density)
        template = PATTERN_TEMPLATES.get(pattern_name)
        
        if not template:
            logger.warning(f"风格 {pattern_name} 无模板，回退到 Pop Ballad。")
            template = PATTERN_TEMPLATES["Pop Ballad"]
            pattern_name = "Pop Ballad" 
            
        logger.debug(f"和弦 {chord_label}, 时长 {segment_length:.2f} QL, 织体 {pattern_name}")

        # 3. 实例化音符并插入 Part (关键步骤：循环填充整个和弦时长)
        time_elapsed_in_segment = 0.0
        template_duration = 4.0 
        
        if pattern_name == "Waltz":
            template_duration = 3.0
        
        while time_elapsed_in_segment < segment_length:
            
            last_note_end_time = time_elapsed_in_segment
            
            for time_offset, index, duration in template:
                
                note_start_time = time_elapsed_in_segment + time_offset
                
                if note_start_time >= segment_length:
                    break 
                
                note_end_time = note_start_time + duration
                
                if note_end_time > segment_length:
                    duration = segment_length - note_start_time
                
                offset_value = offsets[index % len(offsets)]
                
                # 【音高修正】
                if index == 0:
                    # 根音（index=0）降八度 (C2)，然后加上偏移量 (通常为 0)
                    note_midi = (root_midi - 12) + offset_value
                else:
                    # 其他音符保持 C3 八度
                    note_midi = root_midi + offset_value
                
                # =========================================================
                # 【核心修改】: 显式创建 Pitch 对象，确保音高被正确注册
                # =========================================================
                note_pitch = m21.pitch.Pitch(midi=note_midi)
                
                note = m21.note.Note(pitch=note_pitch) # 使用 pitch 参数创建 Note 对象
                note.duration.quarterLength = duration
                
                logger.debug(
                    f"   音符: 索引={index}, 偏移量={offset_value}, MIDI={note_midi} ({note_pitch.nameWithOctave}), QL={duration}"
                )

                # 插入到 Part 中
                accompaniment_part.insert(current_offset + note_start_time, note)
                
                last_note_end_time = current_offset + note_start_time + duration

            
            time_elapsed_in_segment += template_duration
            
            if time_elapsed_in_segment > segment_length and time_elapsed_in_segment - template_duration == last_note_end_time:
                break

        current_offset += segment_length

    return accompaniment_part

# 4. B组主入口：从原始输入到生成 Part (Main B-Group Entry)
def render_accompaniment_from_raw_inputs(
    melody_path: str, 
    chord_sequence: List[str], 
    selected_style: str
) -> m21.stream.Part:
    """
    B组的完整流程：解析输入，计算密度，选择模板，渲染 Part。
    供 main_pipeline.py 中的 render_music 函数调用。
    """
    logger.info("开始 B 组预处理：整合和弦和计算密度...")
    
    # 1. 和弦整合
    consolidated_chords = _consolidate_chords(chord_sequence)
    
    # 2. 解析旋律 MIDI 文件
    try:
        melody_stream = m21.converter.parse(melody_path)
    except Exception as e:
        logger.error(f"无法解析旋律 MIDI 文件: {e}")
        return m21.stream.Part()
    
    # 3. 密度计算
    melody_densities = _calculate_melody_density(melody_stream, consolidated_chords)
    
    logger.info(f"和弦片段数量: {len(consolidated_chords)}, 密度分数示例: {melody_densities[:3]}")

    # 4. 渲染伴奏 Part
    accompaniment_part = generate_accompaniment_part(
        consolidated_chords, 
        melody_densities, 
        selected_style
    )
    
    return accompaniment_part

# ----------------------------------------------------
# 5. 测试主流程 (Test Block)
# ----------------------------------------------------
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG) 

    # 1. 设置输入参数
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    melody_filename = "test_melody.mid" 
    melody_file_path = os.path.join(base_dir, melody_filename)
    
    # 模拟 Group A 输出的和弦序列 (与您上次运行的相同)
    example_chord_sequence = [
        'A', '_', 'D', '_', 'D', '_', '_', '_', '_', '_', '_', '_', 'D', 'D', 
        'D', '_', '_', 'D', 'D', '_', '_', 'D', 'A7', '_', '_', 'D', '_', '0', 
        '_', 'D', '0', 'D', '_', '0', '_', '0', '_', 'D', '0', '0', '_', '_', 
        'D', '0', '_', '_', '0', '_', '_', '0', '_', 'D', '0', '_', '0', '0', 
        'D', '_', '0', '_', '0', '0', '_', '_', 'D', '_', '0', '_', 'D', '_', 
        'D', 'D', 'D', '_', 'D', 'D', '_', 'D', 'D', '_', 'D', 'D', 'D', 'D', 
        '_', 'D', 'D', '_', 'A7', '_', '_', 'D', '0', '_', '0', '0', '_', '_', 
        '_', '_', '0', '_', '0', '_', 'D', '_', 'D', 'D', 'D', 'D', 'D', 'D', 
        'D', 'D', '_', 'D', '_', 'D', 'D', 'A7', '_', 'D', 'D', '0', '0', 'D', 
        '0', '_', 'D', 'D', '_', '0', 'D', 'D', '0', 'D', '_', '0', '0', '_', 
        '_', '_', '0', 'D', '0', '_', 'D', 'D', '0', 'D', '_', '_', 'D', '_', 
        'D', 'D', 'G', 'G', 'D', 'A7', 'D', 'D', '_', '0', '_', '0', '_', '0', 
        'D', '_', '_', '0', 'D', '0', '_', '_', '_', 'D', 'D', 'D', '0', '_', 
        'D', '_', 'D', 'D', '_', '0', 'D', 'D', '_', '_', '_', 'G', 'D', 'D', 
        'D', 'A7', '_', '_', '_', '0', 'D', '_', 'D', '0', 'D', '_', '_', '_', 
        'D', 'D', 'D', 'D', 'D', 'A7', 'A7', 'D', 'A7', '_', 'A7', 'D', 'D', 
        'A7', 'D', 'D', 'D', 'D', '_', 'D', 'D', 'D', 'D', 'D', 'D', 'A7', 'A7', 
        'D', 'D', 'D', 'D', 'D', '_'
    ]
    
    test_style = "Pop Ballad" 

    # 2. 调用主函数
    print(f"\n--- 🚀 开始渲染 {melody_filename} 的伴奏 ({test_style}) ---")
    print(f"--- 📂 尝试从路径: {melody_file_path} 读取文件 ---") 
    
    try:
        accompaniment_part = render_accompaniment_from_raw_inputs(
            melody_file_path,
            example_chord_sequence, 
            test_style
        )
        
        # 3. 检查和输出结果
        if len(accompaniment_part.flat.notesAndRests) > 0:
            print("\n--- ✅ 伴奏 Part 成功生成 ---")
            
            score = m21.stream.Score()
            score.insert(0, m21.metadata.Metadata())
            score.metadata.title = f"Accompaniment for {test_style}"
            score.insert(0, accompaniment_part) 
            
            output_midi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"accompaniment_output_{test_style.replace(' ', '_')}.mid")
            
            # 【关键】: 使用 musicxml 模块来强制刷新 MIDI 缓存，确保音高正确
            score.write('midi', fp=output_midi_path)
            
            print(f"伴奏已保存到: {output_midi_path}")
            
        else:
            print("\n--- ⚠️ 伴奏 Part 为空，检查文件路径、和弦数据或日志中的错误 ---")
            
    except m21.converter.ConverterException:
        print(f"\n--- ❌ 错误：无法找到或解析文件 {melody_filename} ---")
        print(f"请检查文件是否存在于路径: {melody_file_path}。")
    except Exception as e:
        print(f"\n--- ❌ 发生其他错误: {e} ---")