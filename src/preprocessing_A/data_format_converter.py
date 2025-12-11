# import os
# import glob
# import math
# from music21 import converter, note, chord, harmony, stream

# def parse_abc_to_dataset(file_path, step_size=0.25):
#     # --- 修复点 1: dataset 必须初始化为空列表 ---
#     dataset = []

#     # --- 预检查文件内容 ---
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         header = f.read(100)
#         if "X:" not in header and "T:" not in header:
#             print(f"⏩ 跳过非乐谱文件: {os.path.basename(file_path)}")
#             return []

#     try:
#         opus = converter.parse(file_path, format='abc', forceSource=True)
#     except Exception as e:
#         print(f"⚠️ 解析出错 {os.path.basename(file_path)}: {e}")
#         return []

#     if isinstance(opus, stream.Score):
#         scores = [opus]
#     else:
#         scores = opus

#     print(f"✅ 正在处理: {os.path.basename(file_path)} (包含 {len(scores)} 首)")

#     for s in scores:
#         try:
#             s_flat = s.expandRepeats().flatten()
#             if s_flat.duration.quarterLength < 4.0:
#                 continue

#             total_steps = int(math.ceil(s_flat.duration.quarterLength / step_size))

#             melody_seq = ['0'] * total_steps
#             chord_seq = ['_'] * total_steps

#             # --- 旋律处理 ---
#             notes = s_flat.notesAndRests
#             for n in notes:
#                 start_idx = int(round(n.offset / step_size))
#                 end_idx = int(round((n.offset + n.duration.quarterLength) / step_size))

#                 if start_idx >= total_steps: continue
#                 end_idx = min(end_idx, total_steps)

#                 if isinstance(n, note.Note):
#                     melody_seq[start_idx] = str(n.pitch.midi)
#                     for i in range(start_idx + 1, end_idx):
#                         melody_seq[i] = '_'
#                 elif isinstance(n, note.Rest):
#                     for i in range(start_idx, end_idx):
#                         melody_seq[i] = '0'

#             # --- 和弦处理 ---
#             chords = s_flat.getElementsByClass(harmony.ChordSymbol)
#             for c in chords:
#                 start_idx = int(round(c.offset / step_size))
#                 if start_idx >= total_steps: continue
#                 chord_name = c.figure.split('/')[0]
#                 chord_seq[start_idx] = chord_name

#             if len(melody_seq) > 0 and len(chord_seq) > 0:
#                 dataset.append({
#                     "melody": melody_seq,
#                     "harmony": chord_seq
#                 })

#         except Exception as e:
#             continue

#     return dataset

# if __name__ == "__main__":
#     # --- 修复点 2: 指定数据所在的子文件夹 ---
#     # 假设你的 txt 文件在 dataset 文件夹里
#     data_folder = "dataset"

#     # 检查文件夹是否存在
#     if not os.path.exists(data_folder):
#         # 如果 dataset 文件夹不存在，尝试在当前目录找
#         print(f"⚠️ 没找到 '{data_folder}' 文件夹，尝试在当前目录查找...")
#         data_folder = "."

#     # 构建搜索路径：dataset/*.txt
#     search_path_txt = os.path.join(data_folder, "*.txt")
#     search_path_abc = os.path.join(data_folder, "*.abc")

#     input_files = glob.glob(search_path_txt) + glob.glob(search_path_abc)

#     # 排除输出文件
#     input_files = [f for f in input_files if "training_data_aligned" not in f]

#     print(f"📂 在 '{data_folder}' 中找到了 {len(input_files)} 个文件")

#     all_data = []
#     for f in input_files:
#         data = parse_abc_to_dataset(f)
#         all_data.extend(data)

#     print(f"\n🎉 处理完成！共提取 {len(all_data)} 首曲目数据。")

#     if len(all_data) > 0:
#         output_filename = "training_data_aligned.txt"
#         with open(output_filename, "w", encoding="utf-8") as f:
#             for song in all_data:
#                 if isinstance(song, dict) and "melody" in song:
#                     x_str = " ".join(song["melody"])
#                     y_str = " ".join(song["harmony"])
#                     f.write(f"{x_str}|{y_str}\n")
#         print(f"💾 已保存至 {output_filename}")
#     else:
#         print("❌ 依然没有提取到数据。请确认：")
#         print(f"1. 你的 morris.txt 等文件是否在 '{os.path.abspath(data_folder)}' 里面？")

# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_format_converter.py
最终版：ABC -> 时序序列数据转换脚本。
包含路径修正、鲁棒性优化、动态量化、和弦筛选 (只保留 Major, Minor, Seventh) 和简洁输出。
"""

import os
import glob
import math
import traceback
from typing import List, Dict, Any, Tuple, Optional
from music21 import converter, note, chord, harmony, stream, expressions

# =======================================================
# I. 辅助函数和工具
# =======================================================


def find_project_root() -> str:
    """动态确定项目根目录，假设脚本位于 src/ 下"""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.dirname(script_dir)


def safe_expand_repeats(s: stream.Score) -> stream.Score:
    """尝试 expandRepeats，失败则回退原始 score"""
    try:
        return s.expandRepeats()
    except Exception:
        return s


def is_int_str(s: str) -> bool:
    """判断字符串是否可以被解析成整数 MIDI"""
    try:
        int(s)
        return True
    except Exception:
        return False


def parse_text_from_textexpression(te) -> str:
    """从 TextExpression 中取得文本（兼容不同版本 music21）"""
    try:
        if hasattr(te, "content") and te.content:
            return str(te.content).strip()
        if hasattr(te, "text") and te.text:
            return str(te.text).strip()
        return str(te).strip()
    except Exception:
        return str(te).strip()


def safe_chord_name(obj) -> Optional[str]:
    """
    尝试从多种对象中提取简化和弦名，并进行严格过滤：
    只保留 Major (大三), Minor (小三) 和 Seventh (七和弦)。
    """
    chord_name = None

    try:
        if isinstance(obj, harmony.ChordSymbol):
            # 获取质量和名称
            name_only = str(obj.figure).split("/")[0]
            quality = obj.quality  # music21 内部质量标识

            # --- 核心和弦筛选逻辑 ---
            if quality in ("major", "minor"):
                chord_name = name_only
            elif "seventh" in quality:
                chord_name = name_only
            # 其他质量 (diminished, augmented, sus, etc.) 返回 None

        elif isinstance(obj, expressions.TextExpression):
            # 对于 TextExpression，我们只能依赖文本本身进行保守过滤
            txt = parse_text_from_textexpression(obj)
            chord_name = txt.strip().strip('"').split("/")[0]

        elif isinstance(obj, chord.Chord):
            # 对于 Chord 实例，提取根音，但无法准确判断质量
            try:
                root = obj.root()
                if root:
                    chord_name = root.name
            except Exception:
                pass

    except Exception:
        return None

    # 最终检查：过滤掉名称中包含 'dim', 'aug', 'sus' 的和弦，以防漏网之鱼
    if chord_name:
        lower_name = chord_name.lower()
        if "dim" in lower_name or "aug" in lower_name or "sus" in lower_name:
            return None

        return chord_name

    return None


def detect_min_step(s_flat: stream.Score, default_step: float = 0.25) -> float:
    """
    自动检测最小节拍单位，用于动态量化 step_size。
    """
    min_d = default_step * 2

    try:
        for n in s_flat.recurse().notesAndRests:
            d = n.duration.quarterLength
            if d is None or d <= 1e-9:
                continue
            if d < min_d:
                min_d = d

        potential_step = min_d / 4.0

        # 限制最小为 1/64 拍 (0.015625)，防止序列过长
        if potential_step < 0.015625:
            return 0.015625

        # 尽量使用 music21 标准的 step (0.25, 0.125, 0.0625)
        if 0.0625 <= potential_step < 0.125:
            return 0.0625
        if 0.125 <= potential_step < 0.25:
            return 0.125

        return default_step

    except Exception:
        return default_step


# =======================================================
# II. 核心解析逻辑
# =======================================================


def parse_abc_to_dataset(
    file_path: str, default_step: float = 0.25
) -> List[Dict[str, List[str]]]:
    """
    解析单个 ABC 文件中的所有乐曲，进行量化和序列化。
    """
    dataset = []

    # --- 1. 文件预检查 ---
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(100)
            if "X:" not in header and "T:" not in header:
                return []
    except Exception as e:
        print(f"⚠️ 文件读取错误 {os.path.basename(file_path)}: {e}")
        return []

    # --- 2. 尝试解析整个文件 ---
    try:
        opus = converter.parse(file_path, format="abc", forceSource=True)
    except Exception as e:
        print(f"⚠️ 整体解析失败 {os.path.basename(file_path)}: {e}")
        return []

    # --- 3. 确定曲目集合 ---
    scores: List[stream.Score]
    if isinstance(opus, stream.Score):
        scores = [opus]
    else:
        scores = list(opus)

    print(f"✅ 正在处理: {os.path.basename(file_path)} (包含 {len(scores)} 首)")

    # --- 4. 核心处理循环 (单曲目异常捕获，隐藏内部错误打印) ---
    for s in scores:
        try:
            # 展开重复段落并扁平化 (使用 safe_expand_repeats)
            s_expanded = safe_expand_repeats(s)
            s_flat = s_expanded.flatten()

            # 乐曲长度过滤 (< 4 拍)
            if s_flat.duration.quarterLength < 4.0:
                continue

            # 自动检测量化步长
            step_size = detect_min_step(s_flat, default_step)
            total_steps = int(math.ceil(s_flat.duration.quarterLength / step_size))

            # 初始化序列
            melody_seq = ["0"] * total_steps
            chord_seq = ["_"] * total_steps

            # --- 旋律处理 ---
            notes_and_rests = s_flat.notesAndRests
            for n in notes_and_rests:
                start_idx = int(round(n.offset / step_size))

                # 精度优化：使用 math.floor 和容差 1e-9
                end_idx = int(
                    math.floor((n.offset + n.duration.quarterLength) / step_size + 1e-9)
                )

                if start_idx >= total_steps:
                    continue
                end_idx = min(end_idx, total_steps)

                if end_idx <= start_idx:
                    continue

                if isinstance(n, note.Note):
                    melody_seq[start_idx] = str(n.pitch.midi)
                    for i in range(start_idx + 1, end_idx):
                        melody_seq[i] = "_"
                elif isinstance(n, note.Rest):
                    for i in range(start_idx, end_idx):
                        melody_seq[i] = "0"

            # --- 和弦处理 ---
            chord_elements = s_flat.getElementsByClass(
                (harmony.ChordSymbol, expressions.TextExpression, chord.Chord)
            )

            for c_obj in chord_elements:
                start_idx = int(round(c_obj.offset / step_size))
                if start_idx >= total_steps:
                    continue

                # 使用带筛选逻辑的函数
                chord_name = safe_chord_name(c_obj)

                if chord_name:
                    chord_seq[start_idx] = chord_name

            # 5. 数据质量检查：过滤掉旋律全为休止符的序列
            has_valid_note = any(is_int_str(t) and int(t) > 0 for t in melody_seq)

            if has_valid_note:
                dataset.append({"melody": melody_seq, "harmony": chord_seq})
            # else: 隐藏 "旋律数据全为休止符或为空" 的打印

        except Exception as e:
            # 隐藏单曲目错误警告
            continue

    return dataset


# =======================================================
# III. 主执行块 (路径管理与运行)
# =======================================================

if __name__ == "__main__":
    # --- 动态获取项目根路径 ---
    PROJECT_ROOT = find_project_root()

    # --- 1. 路径配置 ---
    data_folder = os.path.join(PROJECT_ROOT, "data", "raw")
    output_dir = os.path.join(PROJECT_ROOT, "data", "interim")
    output_filename = "training_data_aligned.txt"
    output_path = os.path.join(output_dir, output_filename)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入文件夹是否存在
    if not os.path.exists(data_folder):
        print(
            f"⚠️ 错误: 找不到原始数据文件夹 '{data_folder}'。请确认您的原始 ABC 文件位于 data/raw/。"
        )
        exit()

    # --- 2. 文件搜索逻辑 ---
    search_path_txt = os.path.join(data_folder, "*.txt")
    search_path_abc = os.path.join(data_folder, "*.abc")

    input_files = glob.glob(search_path_txt) + glob.glob(search_path_abc)

    print(f"📂 在 '{data_folder}' 中找到了 {len(input_files)} 个文件")

    all_data = []

    # 3. 迭代处理所有文件
    for f in input_files:
        data = parse_abc_to_dataset(f, default_step=0.25)
        all_data.extend(data)

    print(f"\n🎉 处理完成！共提取 {len(all_data)} 首曲目数据。")

    # 4. 写入输出文件
    if len(all_data) > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            for song in all_data:
                if isinstance(song, dict) and "melody" in song and "harmony" in song:
                    x_str = " ".join(song["melody"])
                    y_str = " ".join(song["harmony"])
                    f.write(f"{x_str}|{y_str}\n")
        print(f"💾 已保存至 {output_path}")
    else:
        print("❌ 依然没有提取到数据。请检查上面的错误信息。")
