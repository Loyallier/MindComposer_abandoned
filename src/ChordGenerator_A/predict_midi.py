import os
import sys
import music21
import math
import json  # 需要读取 vocab

# 路径挂载
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A import utils
from src.ChordGenerator_A.inference import AIComposer

# 引入预处理逻辑
try:
    from src.preprocessing_A.ABC_normalization import sample_stream
except ImportError:
    raise ImportError("❌ [Fatal] 无法导入 src/preprocessing_A/ABC_normalization.py")


class MidiPreprocessor:
    """
    职责: MIDI 解析、调性分析、智能八度匹配、转调
    """

    def __init__(self):
        # 加载词表以进行智能覆盖率计算
        self.vocab = utils.load_vocab(path.VOCAB_PATH)
        # 提取旋律词表中所有的有效数字 Key (排除特殊符号)
        self.valid_vocab_pitches = set()
        for k in self.vocab["melody"].keys():
            if k.isdigit():
                self.valid_vocab_pitches.add(int(k))

        if not self.valid_vocab_pitches:
            raise ValueError("❌ 词表异常: 未找到有效的旋律 Pitch Key")

    def get_melody_stream(self, score):
        # 优先查找是否有 Part 明确标记为 Melody
        if score.hasPartLikeStreams():
            # 简单策略：取音符最多的那个 Part，或者是第一个 Part
            # 优化：通常高音部是旋律，取平均音高较高的 Part 可能更准，但这里暂时取 Part 0
            return score.parts[0]
        return score

    def _calculate_vocab_coverage(self, pitches):
        """计算音符列表在词表中的覆盖率"""
        if not pitches:
            return 0.0
        in_vocab_count = sum(1 for p in pitches if p in self.valid_vocab_pitches)
        return in_vocab_count / len(pitches)

    def _smart_octave_shift(self, stream):
        """
        算法: 尝试 -2, -1, 0, +1, +2 个八度，选择词表覆盖率最高且距离中心最近的偏移
        """
        # 1. 提取所有 Pitch
        raw_pitches = [
            p.midi
            for n in stream.recurse().notes
            for p in (n.pitches if n.isChord else [n.pitch])
        ]

        if not raw_pitches:
            raise ValueError("❌ MIDI 轨道中未检测到有效音符")

        best_shift = 0
        max_coverage = -1.0

        # 搜索范围: -24 到 +24 (两个八度)
        candidates = [-24, -12, 0, 12, 24]

        for shift in candidates:
            shifted_pitches = [p + shift for p in raw_pitches]
            coverage = self._calculate_vocab_coverage(shifted_pitches)

            # 逻辑: 优先选覆盖率高的; 如果覆盖率相同，选 shift 绝对值小的 (改动最少的)
            if coverage > max_coverage:
                max_coverage = coverage
                best_shift = shift
            elif coverage == max_coverage:
                if abs(shift) < abs(best_shift):
                    best_shift = shift

        if best_shift != 0:
            print(
                f"   ℹ️ 智能八度修正: 偏移 {best_shift} 半音 (覆盖率: {max_coverage:.2%})"
            )
            return stream.transpose(best_shift), best_shift

        return stream, 0

    def process(self, midi_path):
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"❌ MIDI文件未找到: {midi_path}")

        try:
            score = music21.converter.parse(midi_path)
        except Exception as e:
            raise ValueError(f"Music21 解析失败: {e}")

        original_stream = self.get_melody_stream(score)

        # --- 1. 调性分析与转调 (Smart Transpose) ---
        try:
            key = original_stream.analyze("key")
        except:
            # 极短序列可能无法分析调性，默认为 C Major
            key = music21.key.Key("C")

        if key.mode in ["minor", "dorian", "phrygian", "locrian"]:
            target_tonic = "A"
        else:
            target_tonic = "C"

        target_key = music21.key.Key(target_tonic)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        transposed_stream = original_stream.transpose(interval)

        # 记录第一步转调的偏移
        recover_semitones_key = -interval.semitones

        # --- 2. 智能八度归一化 (Vocab Coverage Maximization) ---
        final_stream, octave_shift_semitones = self._smart_octave_shift(
            transposed_stream
        )

        # 总还原量 = -(转调偏移 + 八度偏移)
        total_recover_semitones = recover_semitones_key - octave_shift_semitones

        # --- 3. 采样与 Token 化 ---
        melody_tokens_flat, _ = sample_stream(final_stream)

        if not melody_tokens_flat:
            raise ValueError("❌ 预处理后未获得有效 Token (文件可能过短或为空)")

        # 重组为小节结构
        measure_tokens = []
        current_bar = []
        for token in melody_tokens_flat:
            if token == config.BAR_TOKEN:
                if current_bar:
                    measure_tokens.append(current_bar)
                current_bar = []
            else:
                current_bar.append(token)

        # [Robustness] 处理末尾残余或单小节情况
        if current_bar:
            measure_tokens.append(current_bar)

        # [Robustness] 极短序列保护: 如果没有 BAR token，上面循环后 measure_tokens 可能为空
        # 但 sample_stream 应该处理了基础 token。如果 flat 只有音符没有 bar，会被 current_bar 捕获。
        # 这里做最后的非空检查
        if not measure_tokens:
            raise ValueError("❌ 无法切分小节，数据结构异常")

        return measure_tokens, total_recover_semitones


class ChordPredictor:
    def __init__(self):
        self.composer = AIComposer()
        self.preprocessor = MidiPreprocessor()
        self.window_size = 4
        self.BAR_POS_IDX = config.POS_VOCAB_SIZE - 2
        
    def _prepare_window_input(self, m_tokens, m_pos):
        flat_input = []
        flat_pos = []
        for bar_toks, bar_ps in zip(m_tokens, m_pos):
            flat_input.extend(bar_toks + [config.BAR_TOKEN])
            flat_pos.extend(bar_ps + [self.BAR_POS_IDX])
        return flat_input, flat_pos

    def restore_chord(self, chord_token, semitones):
        if chord_token in [
            config.PAD_TOKEN,
            config.SOS_TOKEN,
            config.EOS_TOKEN,
            config.UNK_TOKEN,
            "_",
            "0",
            config.BAR_TOKEN,
            "N.C.",
        ]:
            return chord_token

        try:
            import re

            root_match = re.match(r"^([A-G][b#-]?)", chord_token)
            if not root_match:
                return chord_token

            root_str = root_match.group(1)
            suffix = chord_token[len(root_str) :]

            n = music21.note.Note(root_str)
            # 这里 semitones 可能是 float，强转 int
            transposed_n = n.transpose(int(semitones))
            new_root = transposed_n.name.replace("-", "b")

            return new_root + suffix
        except:
            return chord_token

    def run(self, midi_path):
        # 在 run() 方法开头
        print(f"🔧 [Config] Window Size: {self.window_size} | Device: {self.composer.device}")
        
        # 1. 预处理
        try:
            measure_structs, recover_semitones = self.preprocessor.process(midi_path)
        except Exception as e:
            print(f"⚠️ 预处理失败: {e}")
            return []

        num_measures = len(measure_structs)

        # 2. 计算位置 (Smart Position)
        # 针对极短序列 (<1 小节)，utils 需要能处理
        measure_positions = utils.generate_smart_position_indices(measure_structs)

        final_chords_per_measure = [None] * num_measures

        print(
            f"🤖 [Predictor] 处理: {os.path.basename(midi_path)} | 小节数: {num_measures}"
        )

        # === 3. 动态滑窗逻辑 (Robust for Short Sequences) ===

        # 如果总长度小于窗口，直接作为单个 batch 预测
        if num_measures <= self.window_size:
            input_seq, pos_seq = self._prepare_window_input(
                measure_structs, measure_positions
            )
            raw_output = self.composer.predict(input_seq, pos_seq)

            # EOS 处理
            if config.EOS_TOKEN in raw_output:
                eos_idx = raw_output.index(config.EOS_TOKEN)
                raw_output[eos_idx:] = ["0"] * (len(raw_output) - eos_idx)

            # 分配回小节 (Robust Slicing)
            ptr = 0
            for i in range(num_measures):
                # 理论长度 = 音符数 + 1 (BAR)
                length = len(measure_structs[i]) + 1
                seg = raw_output[ptr : ptr + length]
                # 补全防止越界
                if len(seg) < length:
                    seg = seg + ["0"] * (length - len(seg))
                final_chords_per_measure[i] = seg
                ptr += length

        else:
            # 标准滑窗 (保持原逻辑)
            step = 1
            for i in range(0, num_measures, step):
                end_idx = min(i + self.window_size, num_measures)
                # 只有当剩余长度足够才有意义，或者到了最后一段
                current_window_measures = measure_structs[i:end_idx]
                current_window_positions = measure_positions[i:end_idx]

                input_seq, pos_seq = self._prepare_window_input(
                    current_window_measures, current_window_positions
                )
                raw_output = self.composer.predict(input_seq, pos_seq)

                if config.EOS_TOKEN in raw_output:
                    eos_idx = raw_output.index(config.EOS_TOKEN)
                    raw_output[eos_idx:] = ["0"] * (len(raw_output) - eos_idx)

                ptr = 0
                for local_idx, m_tokens in enumerate(current_window_measures):
                    global_idx = i + local_idx
                    length = len(m_tokens) + 1
                    seg = raw_output[ptr : ptr + length]
                    if len(seg) < length:
                        seg = seg + ["0"] * (length - len(seg))

                    if final_chords_per_measure[global_idx] is None:
                        final_chords_per_measure[global_idx] = seg
                    ptr += length

        # === 4. 拼接与还原 ===
        final_result_flat = []
        for chords in final_chords_per_measure:
            if chords is None:
                # 兜底：如果某小节未被预测到（极少见），填空
                final_result_flat.append("N.C.")
                continue

            for token in chords:
                if token in [config.BAR_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN]:
                    continue
                final_result_flat.append(self.restore_chord(token, recover_semitones))

        return final_result_flat


if __name__ == "__main__":
    # 简单的自我测试
    p = ChordPredictor()
    # 模拟一个极短的文件路径，实际运行需存在文件
    print("✅ Predictor initialized. Ready for robustness test.")
