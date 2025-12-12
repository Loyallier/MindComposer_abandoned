import os
import sys
import music21

# ================= 1. 路径挂载 =================
# 获取当前脚本所在目录 (Project Root)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 确保根目录在 sys.path 中，以便能导入 src
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入 A 组的核心预测模块
# 依赖: src/ChordGenerator_A/predict_midi.py
try:
    from src.ChordGenerator_A.predict_midi import ChordPredictor
except ImportError as e:
    print(f"❌ interface_A 导入错误: {e}")
    print("请确保 src/ChordGenerator_A/predict_midi.py 存在且没有语法错误")
    sys.exit(1)

# ================= 2. 接口类定义 =================

class GeneratorA:
    def __init__(self):
        """
        初始化 A 组生成器。
        核心组件: self.predictor (ChordPredictor)
        """
        print("🎹 初始化 Generator A (Harmony Model)...")
        try:
            # 实例化预测器，加载 PyTorch 模型
            self.predictor = ChordPredictor()
        except Exception as e:
            raise RuntimeError(f"ChordPredictor 初始化失败: {e}")

    def generate(self, input_midi_path, output_midi_path):
        """
        【独立运行模式】
        功能: 读取 MIDI -> 预测和弦 -> 简单合成 (Merge) -> 保存 MIDI
        注意: 这个方法主要用于单独测试 A 组效果。
              在 interface.py 的总流程中，我们主要使用 self.predictor.run() 获取数据。
        """
        if not os.path.exists(input_midi_path):
            raise FileNotFoundError(f"输入文件不存在: {input_midi_path}")

        print(f"🎵 [Model A] 正在处理: {os.path.basename(input_midi_path)}")

        # 1. 获取预测结果 (List[str])
        # 调用 predict_midi.py 中的核心逻辑
        chord_sequence = self.predictor.run(input_midi_path)
        
        print(f"   -> 预测完成，共 {len(chord_sequence)} 个时间步")

        # 2. 合成 MIDI (将和弦写入新轨道)
        self._merge_and_save(input_midi_path, chord_sequence, output_midi_path)
        
        return output_midi_path

    def _merge_and_save(self, original_midi_path, chord_tokens, output_path):
        """
        内部工具: 将和弦序列简单地写入 MIDI 轨道 (柱式和弦)
        """
        try:
            score = music21.converter.parse(original_midi_path)
            
            # 创建一个新的 Part 用于存放和弦
            chord_part = music21.stream.Part()
            chord_part.id = 'AI_Chords_Basic'
            chord_part.insert(0, music21.instrument.Piano())
            
            # 步长必须与训练/预测时的采样率一致 (0.25 = 16分音符)
            step_size = 0.25
            current_offset = 0.0
            
            for token in chord_tokens:
                # 过滤特殊符号
                if token not in ["_", "0", "<BAR>", "<PAD>", "<SOS>", "<EOS>", "N.C."]:
                    try:
                        # 解析和弦符号 (e.g., "Am7")
                        h = music21.harmony.ChordSymbol(token)
                        # 获取音符并压低八度 (伴奏通常在低音区)
                        c_notes = h.pitches
                        if c_notes:
                            chord_obj = music21.chord.Chord(c_notes)
                            chord_obj.duration.quarterLength = step_size
                            chord_obj.transpose(-12, inPlace=True) 
                            chord_part.insert(current_offset, chord_obj)
                    except:
                        pass # 忽略解析失败的和弦
                
                # 处理延音 "_"
                elif token == "_":
                    try:
                        last_el = chord_part.getElementsByClass(music21.chord.Chord).last()
                        if last_el:
                            last_el.duration.quarterLength += step_size
                    except:
                        pass
                
                current_offset += step_size

            # 合并到总谱
            score.insert(0, chord_part)
            score.write('midi', fp=output_path)
            print(f"💾 [Model A] 独立结果已保存: {output_path}")

        except Exception as e:
            print(f"⚠️ MIDI 合成失败: {e}")

# ================= 3. 测试入口 =================
if __name__ == "__main__":
    # 简单测试
    in_file = os.path.join("src", "test_melody.mid")
    out_file = "output_model_A_standalone.mid"
    
    if os.path.exists(in_file):
        gen = GeneratorA()
        gen.generate(in_file, out_file)
    else:
        print(f"⚠️ 请准备测试文件: {in_file}")