import os
import sys
import music21
import math

# ================= 1. 路径挂载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/ChordGenerator_A -> src -> Root
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import path
from src.ChordGenerator_A import config
from src.ChordGenerator_A.inference import AIComposer

# 🔄 复用数据处理模块 (确保采样逻辑与训练完全一致)
try:
    from src.preprocessing_A.ABC_normalization import sample_stream
except ImportError:
    raise ImportError("❌ 无法导入数据处理模块，请检查 src/preprocessing_A/ABC_normalization.py 是否存在")

# ================= 2. 预处理类 (Pre-processor) =================

class MidiPreprocessor:
    def __init__(self):
        # 训练时的归一化参数
        self.target_centroid = 63  
        self.hard_min = 45
        self.hard_max = 86
        
    def get_melody_stream(self, score):
        """提取单声部旋律 (逻辑同 ABC_normalization)"""
        if score.hasPartLikeStreams():
            return score.parts[0]
        return score

    def process(self, midi_path):
        """
        核心流程: 加载 -> 转调(C/Am) -> 八度归一 -> 采样(Token化)
        返回: 
          - measure_tokens: List[List[str]] (按小节分组的 token)
          - recover_semitones: int (还原需要的半音数)
        """
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
        
        # 确定目标调性
        if key.mode in ['minor', 'dorian', 'phrygian', 'locrian']:
            target_tonic = 'A'
        else:
            target_tonic = 'C'
            
        target_key = music21.key.Key(target_tonic)
        interval = music21.interval.Interval(key.tonic, target_key.tonic)
        
        # 执行转调
        transposed_stream = melody_stream.transpose(interval)
        
        # 记录还原所需的距离 (反向)
        recover_semitones = -interval.semitones
        print(f"   - 原调: {key.tonic.name} {key.mode} -> 目标: {target_tonic} (偏移 {interval.semitones})")
        
        # --- 2. 八度归一化 (Centroid Normalization) ---
        # 计算当前重心
        pitches = [p.midi for n in transposed_stream.recurse().notes for p in (n.pitches if n.isChord else [n.pitch])]
        if not pitches:
            raise ValueError("❌ 未检测到有效音符")
            
        avg_pitch = sum(pitches) / len(pitches)
        dist = self.target_centroid - avg_pitch
        shift_octaves = round(dist / 12)
        shift_semitones = int(shift_octaves * 12)
        
        final_stream = transposed_stream.transpose(shift_semitones)
        
        # 边界检查 (Hard Clamp) - 如果越界太多，强行拉回
        # 这里为了保证推理不崩，我们只做检查，不做强行删除，因为这是推理不是清洗
        final_pitches = [p.midi for n in final_stream.recurse().notes for p in (n.pitches if n.isChord else [n.pitch])]
        if min(final_pitches) < self.hard_min or max(final_pitches) > self.hard_max:
            print(f"⚠️ 警告: 音域 ({min(final_pitches)}-{max(final_pitches)}) 超出训练范围 ({self.hard_min}-{self.hard_max})，可能影响效果。")

        # --- 3. 采样与切分 (Sampling) ---
        # 复用 ABC_normalization.sample_stream
        # 注意：sample_stream 返回的是 flat list (带 <BAR>)
        # 我们需要把它还原成 List[List] 结构以便滑窗
        
        melody_tokens_flat, _ = sample_stream(final_stream)
        
        # 重构小节结构
        measure_tokens = []
        current_bar = []
        for token in melody_tokens_flat:
            if token == config.BAR_TOKEN:
                if current_bar: # 排除空小节
                    measure_tokens.append(current_bar)
                current_bar = []
            else:
                current_bar.append(token)
        
        # 处理尾部残留
        if current_bar:
            measure_tokens.append(current_bar)
            
        print(f"   - 预处理完成: 共 {len(measure_tokens)} 小节")
        return measure_tokens, recover_semitones

# ================= 3. 预测引擎 (Engine) =================

class ChordPredictor:
    def __init__(self):
        self.composer = AIComposer() # 加载模型
        self.preprocessor = MidiPreprocessor()
        self.window_size = 4
        
    def restore_chord(self, chord_token, semitones):
        """和弦转调还原"""
        if chord_token in [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.UNK_TOKEN, "_", "0", "<BAR>"]:
            return chord_token
            
        try:
            import re
            root_match = re.match(r"^([A-G][b#-]?)", chord_token)
            if not root_match: return chord_token
            
            root_str = root_match.group(1)
            suffix = chord_token[len(root_str):]
            
            # 使用 music21 的转调逻辑
            n = music21.note.Note(root_str)
            transposed_n = n.transpose(semitones)
            # 修正 enharmonics (比如把 E# 变成 F)
            new_root = transposed_n.name
            
            return new_root + suffix
        except:
            return chord_token

    def run(self, midi_path):
        # 1. 预处理
        measure_structs, recover_semitones = self.preprocessor.process(midi_path)
        num_measures = len(measure_structs)
        
        # 初始化结果容器 (按小节存储)
        final_chords_per_measure = [None] * num_measures 
        
        print(f"🤖 启动滑窗预测 (Window={self.window_size})...")
        
        # === 核心滑窗逻辑 ===
        
        # A. 短曲特例 (不足 4 小节)
        if num_measures <= self.window_size:
            input_seq = []
            for m in measure_structs:
                input_seq.extend(m)
                input_seq.append(config.BAR_TOKEN)
            
            raw_output = self.composer.predict(input_seq)
            
            # 切分回小节
            ptr = 0
            for i in range(num_measures):
                length = len(measure_structs[i])
                final_chords_per_measure[i] = raw_output[ptr : ptr+length]
                ptr += length
                
        # B. 长曲滑窗
        else:
            # 循环次数: N - Window + 1
            for i in range(0, num_measures - self.window_size + 1):
                # 1. 构建 Input (包含 BAR)
                window_measures = measure_structs[i : i+self.window_size]
                input_seq = []
                for m in window_measures:
                    input_seq.extend(m)
                    input_seq.append(config.BAR_TOKEN)
                
                # 2. 预测 (inference.py 已去除输出中的 BAR)
                raw_output = self.composer.predict(input_seq)
                
                # 3. 将输出切割成 4 个小节
                window_chords_split = []
                ptr = 0
                for m in window_measures:
                    length = len(m)
                    # 防止越界 (容错)
                    seg = raw_output[ptr : ptr+length] if ptr < len(raw_output) else ["_"] * length
                    window_chords_split.append(seg)
                    ptr += length
                
                # 4. === 拼接策略 (Selection Logic) ===
                if i == 0:
                    # 前 4 小节中的前 3 小节直接取用
                    final_chords_per_measure[0] = window_chords_split[0]
                    final_chords_per_measure[1] = window_chords_split[1]
                    final_chords_per_measure[2] = window_chords_split[2]
                else:
                    # 之后的每次滑窗，只取第 3 小节 (Index 2)
                    # 对应的全局索引是 i + 2
                    target_idx = i + 2
                    if target_idx < num_measures:
                        final_chords_per_measure[target_idx] = window_chords_split[2]
            
            # 5. 处理尾部 (Last Slice Logic)
            # 最后一个窗口覆盖了 [N-4, N-3, N-2, N-1]
            # 上面的循环在 i = N-4 时结束
            # 那时我们要了 window_chords_split[2] 即 N-2
            # 还剩下 N-1 (最后一小节) 没填
            # 策略：保留最后一切片的后两小节 -> 即保留 [3] (N-1)
            # (注意：N-2 已经被上面覆盖了，但覆盖也没事，这里可以覆盖最后两个)
            
            final_chords_per_measure[-2] = window_chords_split[-2] # 覆盖确认
            final_chords_per_measure[-1] = window_chords_split[-1] # 填补最后

        # 2. 还原与输出
        print(f"🔄 还原调性 (偏移 {recover_semitones} 半音)...")
        result_flat = []
        
        for idx, chords in enumerate(final_chords_per_measure):
            if chords is None:
                # 理论不应触发
                print(f"⚠️ 第 {idx} 小节和弦缺失，自动补全")
                chords = ["_"] * len(measure_structs[idx])
            
            restored = [self.restore_chord(c, recover_semitones) for c in chords]
            result_flat.extend(restored)
            
        return result_flat

# ================= 4. 测试入口 =================

if __name__ == "__main__":
    # 指向你的测试文件
    TEST_MIDI = os.path.join(path.SRC_DIR, "test_melody.mid")
    
    if os.path.exists(TEST_MIDI):
        predictor = ChordPredictor()
        try:
            result = predictor.run(TEST_MIDI)
            print("\n🎹 生成结果 (Top 50 Tokens):")
            print(result[:50])
            print(f"📏 总长度: {len(result)}")
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ 请先准备测试文件: {TEST_MIDI}")