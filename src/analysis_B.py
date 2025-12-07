import music21 as m21
from typing import List, Tuple, Dict, Any
import logging

# 初始化日志，与 interface.py 保持一致
logger = logging.getLogger("B_Analysis")

# 16分音符的 QuarterLength
QL_16TH = 0.25 

# Group B Phase 2: Analysis
# 核心功能：和弦序列分割、MIDI 解析、旋律密度计算。

def consolidate_chord_sequence(chord_sequence: List[str]) -> List[Tuple[str, float]]:
    """
    将 16 分音符切片的和弦序列 (如 ['C', '_', '_', '_', 'G', '_', ...]) 
    转换为 (和弦标签, 持续时长 QL) 的列表。
    
    Args:
        chord_sequence (List[str]): Group A 预测的 16 分音符切片序列。
        
    Returns:
        List[Tuple[str, float]]: 实际的和弦片段列表。
    """
    if not chord_sequence:
        return []

    consolidated_chords: List[Tuple[str, float]] = []
    current_chord = chord_sequence[0]
    duration_slices = 0

    for label in chord_sequence:
        if label != '_' and label != current_chord:
            # 遇到新的非 HOLD 和弦，或从 HOLD 状态结束
            if current_chord != '_':
                # 记录前一个和弦的 (标签, 时长)
                consolidated_chords.append((current_chord, duration_slices * QL_16TH))
            
            # 开启新的和弦段
            current_chord = label
            duration_slices = 1
        elif label == '_':
            # 遇到延音 HOLD，则增加当前和弦的持续时间
            duration_slices += 1
        else:
            # 遇到与当前和弦相同的标签，增加持续时间
            duration_slices += 1

    # 处理最后一个和弦段
    if current_chord != '_':
        consolidated_chords.append((current_chord, duration_slices * QL_16TH))
        
    # 清理: 如果第一个和弦就是 HOLD，则不记录。
    final_list = [c for c in consolidated_chords if c[0] != '_']
    
    logger.info(f"和弦序列已分割为 {len(final_list)} 个片段。")
    return final_list


def get_melody_part(score: m21.stream.Score) -> m21.stream.Part:
    """
    尝试从 music21.Score 对象中提取旋律 Part (通常是第一个 Part)。
    """
    if score.parts:
        return score.parts[0]
    
    logger.warning("Score 中没有 Part，无法提取旋律。")
    return m21.stream.Part()


def analyze_melody_density(melody_part: m21.stream.Part, consolidated_chords: List[Tuple[str, float]]) -> List[float]:
    """
    计算旋律在每个可变时长和弦片段内的音符密度。
    
    Args:
        melody_part (m21.stream.Part): 提取出的旋律 Part。
        consolidated_chords (List[Tuple[str, float]]): (和弦标签, 时长 QL) 列表。
        
    Returns:
        List[float]: 对应每个和弦片段的旋律密度分数（0.0 到 1.0）。
    """
    if not melody_part or not consolidated_chords:
        logger.error("旋律 Part 或和弦列表为空，密度返回 0.0。")
        return [0.0] * len(consolidated_chords)

    all_elements = melody_part.flat.notesAndRests
    density_scores: List[float] = []
    current_time = 0.0
    
    # 假设最大密度阈值：每 1 拍 (QL=1.0) 最多 4 个音符 (16分音符)
    MAX_NOTES_PER_QL = 4.0 

    for i, (chord_label, segment_length) in enumerate(consolidated_chords):
        end_time = current_time + segment_length
        notes_in_segment = 0
        
        # 遍历该片段内的所有音符和休止符
        for element in all_elements:
            # 检查元素是否在当前时间范围内
            if element.offset >= current_time and element.offset < end_time:
                # 仅计算实际的音符/和弦
                if isinstance(element, (m21.note.Note, m21.chord.Chord)):
                    notes_in_segment += 1

        # 计算密度：音符数量 / 该片段的最大预期音符数量
        # 最大预期音符数量 = 片段时长 * 每拍最大音符数
        max_notes_in_segment = segment_length * MAX_NOTES_PER_QL
        
        if max_notes_in_segment > 0:
            # 密度分数不能超过 1.0
            density = min(1.0, notes_in_segment / max_notes_in_segment)
        else:
            density = 0.0 # 时长为 0 
            
        density_scores.append(density)
        logger.debug(f"Segment {i} ({chord_label}, {segment_length:.2f} QL): Notes={notes_in_segment}, Density={density:.2f}")
        
        current_time = end_time

    return density_scores