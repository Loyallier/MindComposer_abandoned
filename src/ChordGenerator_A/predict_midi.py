import os
import sys
import music21

# ================= 1. 路径挂载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A import utils  # ✅ 复用核心逻辑
from src.ChordGenerator_A.inference import AIComposer

# 复用数据处理模块
try:
    from src.preprocessing_A.ABC_normalization import sample_stream
except ImportError:
    raise ImportError("❌ 无法导入数据处理模块，请检查 src/preprocessing_A/ABC_normalization.py")

# ================= 2. 预处理类 =================

class MidiPreprocessor:
    def __init__(self):
        self.target_centroid = 63  
        self.hard_min = 45
        self.hard_max = 86
        
    def get_melody_stream(self, score):
        if score.hasPartLikeStreams():
            return score.parts[0]
        return score

    def process(self, midi_path):
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"❌ MIDI文件未找到: {midi_path}")

        print(f"🎼 解析 MIDI: {os.path.basename(midi_path)}")
        try:
            score = music21.converter.parse(midi_path)
        except Exception as e:
            raise ValueError(f"Music21 解析失败: {e}")

        melody_stream = self.get_melody_stream(score)
        
        # --- 1. 智能转调 (Smart Transpose) ---
        key = melody_stream.analyze('key')
        if key.mode in ['minor', 'dorian', 'phrygian', 'locrian']:
            target_tonic = 'A'
        else:
            target_tonic = 'C'
            
        target_key = music21.key.Key(target_tonic)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        transposed_stream = melody_stream.transpose(interval)
        recover_semitones = -interval.semitones
        
        print(f"   - 转调: {key.tonic.name} -> {target_tonic} (还原需 {recover_semitones})")
        
        # --- 2. 八度归一化 ---
        pitches = [p.midi for n in transposed_stream.recurse().notes for p in (n.pitches if n.isChord else [n.pitch])]
        if not pitches: raise ValueError("❌ 无效音符")
            
        avg_pitch = sum(pitches) / len(pitches)
        dist = self.target_centroid - avg_pitch
        shift_semitones = int(round(dist / 12) * 12)
        final_stream = transposed_stream.transpose(shift_semitones)
        
        # --- 3. 采样与切分 ---
        melody_tokens_flat, _ = sample_stream(final_stream)
        
        measure_tokens = []
        current_bar = []
        for token in melody_tokens_flat:
            if token == config.BAR_TOKEN:
                if current_bar: measure_tokens.append(current_bar)
                current_bar = []
            else:
                current_bar.append(token)
        if current_bar: measure_tokens.append(current_bar)
            
        print(f"   - 预处理完成: {len(measure_tokens)} 小节")
        return measure_tokens, recover_semitones

# ================= 3. 预测引擎 (Fixed) =================

# ... (前部分代码保持不变，包括 Imports 和 Preprocessor) ...
# ... (Preprocessor 类保持不变) ...

class ChordPredictor:
    def __init__(self):
        self.composer = AIComposer()
        self.preprocessor = MidiPreprocessor()
        self.window_size = 4
        self.BAR_POS_IDX = config.POS_VOCAB_SIZE - 2 
        
    def restore_chord(self, chord_token, semitones):
        # (保持不变)
        if chord_token in [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.UNK_TOKEN, "_", "0", "<BAR>"]:
            return chord_token
        try:
            import re
            root_match = re.match(r"^([A-G][b#-]?)", chord_token)
            if not root_match: return chord_token
            root_str = root_match.group(1)
            suffix = chord_token[len(root_str):]
            n = music21.note.Note(root_str)
            transposed_n = n.transpose(semitones)
            return transposed_n.name + suffix
        except:
            return chord_token

    def _prepare_window_input(self, m_tokens, m_pos):
        flat_input = []
        flat_pos = []
        for bar_toks, bar_ps in zip(m_tokens, m_pos):
            flat_input.extend(bar_toks + [config.BAR_TOKEN])
            flat_pos.extend(bar_ps + [self.BAR_POS_IDX])
        return flat_input, flat_pos

    def run(self, midi_path):
        # 1. 预处理
        measure_structs, recover_semitones = self.preprocessor.process(midi_path)
        num_measures = len(measure_structs)
        
        # 2. 计算位置
        measure_positions = utils.generate_smart_position_indices(measure_structs)
        
        final_chords_per_measure = [None] * num_measures 
        print(f"🤖 启动 V3.1 预测 (Window={self.window_size}, Force Downbeat=ON)...")
        
        # === 滑窗预测 (逻辑保持不变) ===
        if num_measures <= self.window_size:
            input_seq, pos_seq = self._prepare_window_input(measure_structs, measure_positions)
            raw_output = self.composer.predict(input_seq, pos_seq)
            
            ptr = 0
            for i in range(num_measures):
                length = len(measure_structs[i])
                # 容错截取
                seg = raw_output[ptr : ptr+length] if ptr < len(raw_output) else ["_"] * length
                final_chords_per_measure[i] = seg
                ptr += length
        else:
            for i in range(0, num_measures - self.window_size + 1):
                window_measures = measure_structs[i : i+self.window_size]
                window_positions = measure_positions[i : i+self.window_size]
                
                input_seq, pos_seq = self._prepare_window_input(window_measures, window_positions)
                raw_output = self.composer.predict(input_seq, pos_seq)
                
                window_chords_split = []
                ptr = 0
                for m in window_measures:
                    length = len(m)
                    seg = raw_output[ptr : ptr+length] if ptr < len(raw_output) else ["_"] * length
                    window_chords_split.append(seg)
                    ptr += length
                
                if i == 0:
                    final_chords_per_measure[0] = window_chords_split[0]
                    final_chords_per_measure[1] = window_chords_split[1]
                    final_chords_per_measure[2] = window_chords_split[2]
                else:
                    target_idx = i + 2
                    if target_idx < num_measures:
                        final_chords_per_measure[target_idx] = window_chords_split[2]
            
            final_chords_per_measure[-2] = window_chords_split[-2]
            final_chords_per_measure[-1] = window_chords_split[-1]

        # 3. 还原与输出 (Cleaned for Group B)
        print(f"🔄 还原调性 (偏移 {recover_semitones})...")
        result_flat = []
        
        for idx, chords in enumerate(final_chords_per_measure):
            if chords is None:
                chords = ["_"] * len(measure_structs[idx])
            
            restored = [self.restore_chord(c, recover_semitones) for c in chords]
            
            # ✅ [修正] 不再添加 [config.BAR_TOKEN]
            # 我们只添加和弦内容，保持列表平铺且无缝
            result_flat.extend(restored)
            
        return result_flat

if __name__ == "__main__":
    # 测试入口保持不变...
    pass