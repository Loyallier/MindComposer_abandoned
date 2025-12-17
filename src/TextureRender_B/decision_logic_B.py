import music21 as m21
from typing import List, Dict, Tuple
import logging
import math
import sys
import os

# 1. 路径挂载逻辑 (保持不变)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"🔧 已将根目录挂载: {project_root}")
from src.TextureRender_B.data_logic_B import (
    CHORD_TYPES,
    PATTERN_TEMPLATES,
    parse_simplified_chord,
)  # 导入 Group B 知识库

# 初始化日志
logger = logging.getLogger("B_Decision")

# Group B Phase 3: Decision and Rendering

# ----------------------------------------------------
# 新增：MIDI 力度值映射和计算逻辑 (Velocity Mapping and Calculation)
# ----------------------------------------------------

# MIDI 力度值映射参数 (Velocity Mapping Parameters)
# MIDI 力度范围是 0-127
MAX_VELOCITY = 100 # 伴奏音符最大力度
MIN_VELOCITY = 40  # 伴奏音符最小力度

# 强弱拍相对力度权重 (用于 4/4 和 3/4 拍，基于 QuarterLength 偏移)
# Key: 模板的总时长 (QL)
# Value: { 拍子的起始 QL 偏移: 相对力度乘数 }
BEAT_VELOCITY_WEIGHTS = {
    4.0: {0.0: 1.0, 1.0: 0.8, 2.0: 0.9, 3.0: 0.7}, # 4/4 拍 (1拍最强，3拍次强)
    3.0: {0.0: 1.0, 1.0: 0.7, 2.0: 0.8},          # 3/4 拍 (1拍最强)
}

def calculate_velocity(segment_offset: float, density: float, template_duration: float) -> int:
    """
    根据音符在片段中的起始位置、旋律密度和模板时长计算 MIDI 力度值。
    
    Args:
        segment_offset (float): 音符在和弦片段内模板周期中的起始时间 (QL)。
        density (float): 当前片段的旋律密度 (0.0 - 1.0)。
        template_duration (float): 模板的总时长 (如 4.0 QL 或 3.0 QL)。
        
    Returns:
        int: 计算出的 MIDI 力度值 (0-127)。
    """
    
    # 1. 密度反向调节 (Density Inverse Adjustment)
    # 旋律密度越高，伴奏力度越小 (1.0 - density)
    density_factor = 1.0 - density
    
    # 2. 强弱拍权重 (Metric Weight)
    # 计算音符在模板周期中的相对位置
    beat_time = segment_offset % template_duration
    
    # 获取对应模板时长的权重，默认 4/4
    weights = BEAT_VELOCITY_WEIGHTS.get(template_duration, BEAT_VELOCITY_WEIGHTS[4.0])
    
    metric_weight = 1.0 
    # 查找最接近的拍子起始点来应用强弱拍权重
    for beat_start, weight in weights.items():
        if abs(beat_time - beat_start) < 0.1: # 允许微小误差
            metric_weight = weight
            break
            
    # 3. 综合计算最终力度
    base_velocity = MAX_VELOCITY - MIN_VELOCITY
    
    # 最终力度 = 最小力度 + (力度基准范围 * 密度反调因子 * 强弱拍权重)
    final_velocity = MIN_VELOCITY + (base_velocity * density_factor * metric_weight)
    
    # 确保力度在有效范围内
    return int(max(MIN_VELOCITY, min(MAX_VELOCITY, final_velocity)))

# ----------------------------------------------------
# 1. 启发式决策：织体选择 (Texture Selector)
# ----------------------------------------------------
DENSITY_THRESHOLD = 0.4 # 旋律密度阈值：超过此值视为“繁忙”

def select_texture_pattern(style: str, density_score: float) -> str:
    """
    启发式算法：根据风格和旋律密度选择伴奏织体模板。
    """

    # 如果密度高（旋律繁忙），倾向于选择稀疏的伴奏，避免冲突
    if density_score > DENSITY_THRESHOLD:
        logger.debug(
            f"密度 {density_score:.2f} 较高，倾向于稀疏织体 (Sparse_Arpeggio)。"
        )
        # 无论什么风格，都使用通用的稀疏琶音来避免冲突
        return "Sparse_Arpeggio"

    # 如果密度低（旋律稀疏），使用风格默认的织体
    else:
        logger.debug(f"密度 {density_score:.2f} 较低，使用风格默认织体: {style}")
        return style


# 2. 预处理 A: 和弦片段整合与密度计算 (Pre-processing)
def _consolidate_chords(chord_sequence, quantization_step=0.25):
    if not chord_sequence: return []
    
    consolidated = []
    # 找到第一个非空的有效和弦
    current_chord = 'C' 
    for token in chord_sequence:
        if token != '_':
            current_chord = token
            break
            
    current_duration = 0.0
    for label in chord_sequence:
        if label == '_' or label == current_chord:
            current_duration += quantization_step
        else:
            # 只有当 current_chord 有效时才添加
            if current_chord != '_':
                consolidated.append((current_chord, current_duration))
            current_chord = label
            current_duration = quantization_step
            
    # 确保最后一个片段
    if current_chord != '_':
        consolidated.append((current_chord, current_duration))
    return consolidated


def _calculate_melody_density(
    melody_stream: m21.stream.Stream, consolidated_chords: List[Tuple[str, float]]
) -> List[float]:
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
            if isinstance(element, m21.note.Note) or isinstance(
                element, m21.chord.Chord
            ):
                if segment_start <= element_start < segment_end - 0.001:
                    segment_notes_count += 1

        # 密度计算
        if duration > 0:
            # 使用一个缩放因子 (0.7) 和平方根来将密度值映射到 0-1 范围，并平滑变化
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
    
    accompaniment_part = m21.stream.Part()
    accompaniment_part.insert(0, m21.instrument.Piano())
    
    # 1. 初始化全局偏移量（追踪整个曲子的时间线）
    current_offset = 0.0
    
    for i in range(len(consolidated_chords)):
        chord_label, segment_length = consolidated_chords[i]
        density = melody_densities[i]
        
        # 2. 解析和弦：确保调用了能识别 'Am' 的增强版解析器
        root_name, chord_type = parse_simplified_chord(chord_label)
        
        if chord_type == 'NoChord' or chord_type not in CHORD_TYPES:
            current_offset += segment_length
            continue
            
        offsets = CHORD_TYPES[chord_type]
        root_midi = m21.pitch.Pitch(root_name + '3').midi

        # 3. 获取模板：比如 Pop Ballad (4拍一循环)
        pattern_name = select_texture_pattern(selected_style, density)
        template = PATTERN_TEMPLATES.get(pattern_name, PATTERN_TEMPLATES["Pop Ballad"])
        template_duration = 3.0 if pattern_name == "Waltz" else 4.0

        # --- 【关键修复：段内计时重置】 ---
        # 每一个新和弦片段（如片段1的6.0QL）开始时，必须重置段内流逝时间
        time_elapsed_in_segment = 0.0
        
        # 循环填充，直到填满这个片段的时长
        while time_elapsed_in_segment < segment_length - 0.01:
            for time_offset, index, duration in template:
                # 计算这个音符在当前片段内的相对起始时间
                note_start_in_segment = time_elapsed_in_segment + time_offset
                
                # 如果这个音符已经超出当前和弦的范围，就不画了（防止覆盖下一个和弦）
                if note_start_in_segment >= segment_length - 0.01:
                    break
                
                # 计算音高
                # index 0 通常是 Bass 音（低八度），其他是和弦内音
                pitch_midi = (m21.pitch.Pitch(root_name + '2').midi if index == 0 
                             else root_midi) + offsets[index % len(offsets)]
                
                # 创建音符对象
                n = m21.note.Note(midi=pitch_midi)
                
                # 时长修正：确保最后一个音符在和弦切换点被切齐
                actual_duration = duration
                if note_start_in_segment + duration > segment_length:
                    actual_duration = segment_length - note_start_in_segment
                n.duration.quarterLength = actual_duration
                
                # 力度计算
                n.volume.velocity = calculate_velocity(note_start_in_segment, density, template_duration)

                # 【核心逻辑】：插入位置 = 全局累计偏移 + 段内相对偏移
                accompaniment_part.insert(current_offset + note_start_in_segment, n)

            # 模板步进（例如 4/4 拍就推进 4 拍）
            time_elapsed_in_segment += template_duration
            
        # 4. 片段处理完毕，全局偏移向前推进，准备接下一个和弦
        current_offset += segment_length
        # --- 【修复结束】 ---

    return accompaniment_part


# 4. B组主入口：从原始输入到生成 Part (Main B-Group Entry)
def render_accompaniment_from_raw_inputs(
    melody_path: str, chord_sequence: List[str], selected_style: str
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
        # 如果解析失败，返回一个空的 Part
        return m21.stream.Part()

    # 3. 密度计算
    melody_densities = _calculate_melody_density(melody_stream, consolidated_chords)

    logger.info(
        f"和弦片段数量: {len(consolidated_chords)}, 密度分数示例: {melody_densities[:3]}"
    )

    # 4. 渲染伴奏 Part
    accompaniment_part = generate_accompaniment_part(
        consolidated_chords, melody_densities, selected_style
    )

    return accompaniment_part
