import os
import sys
import music21

# ================= 1. 路径环境配置 =================
# 获取当前脚本所在目录 (Project Root)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入 A 组与 B 组接口
try:
    from interface_A import GeneratorA
    from interface_B import render_accompaniment
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请检查 interface_A.py 和 interface_B.py 是否都在根目录。")
    sys.exit(1)

# ================= 2. 总控类 (The Conductor) =================

class AI_Music_Generator:
    def __init__(self):
        print("🤖 初始化 AI 音乐生成管线...")
        
        # 初始化 A 组模型 (和弦脑)
        try:
            self.model_a = GeneratorA()
            print("✅ Model A (Harmony) 就绪")
        except Exception as e:
            print(f"❌ Model A 初始化失败: {e}")
            sys.exit(1)

    def process(self, input_midi_path, output_midi_path, style="Pop Ballad"):
        """
        全流程生成: Melody -> Chords -> Texture -> Final MIDI
        """
        if not os.path.exists(input_midi_path):
            raise FileNotFoundError(f"找不到输入文件: {input_midi_path}")

        print("\n" + "="*40)
        print(f"🚀 开始处理: {os.path.basename(input_midi_path)}")
        print(f"🎹 目标风格: {style}")
        print("="*40)

        # --- Phase 1: AI_A 生成和弦序列 ---
        print("Step 1: 正在预测和声 (Model A)...")
        try:
            # 获取和弦 Token 列表
            chord_tokens = self.model_a.predictor.run(input_midi_path)
            print(f"   -> 生成了 {len(chord_tokens)} 个和弦切片")
            
            # 简单展示前几个和弦
            preview = [c for c in chord_tokens if c not in []]
            print(f"   -> 和弦预览: {preview}...")
        except Exception as e:
            print(f"❌ Phase 1 失败: {e}")
            return False

        # --- Phase 2: AI_B 渲染伴奏织体 ---
        print("\nStep 2: 正在渲染织体 (Model B)...")
        try:
            # 生成伴奏 Part (Flat Stream)
            accompaniment_part = render_accompaniment(
                melody_midi_path=input_midi_path,
                chord_sequence=chord_tokens,
                style=style
            )
            
            # 检查是否有音符
            if not accompaniment_part or len(accompaniment_part.flatten().notes) == 0:
                print("⚠️ 警告: B 组返回了空轨道 (可能风格不支持或渲染失败)。")
        except Exception as e:
            print(f"❌ Phase 2 失败: {e}")
            return False

        # --- Phase 3: 合成与导出 ---
        print("\nStep 3: 合成最终总谱...")
        try:
            # 1. 读取原始旋律
            original_score = music21.converter.parse(input_midi_path)
            
            # 提取主旋律 Part
            if original_score.hasPartLikeStreams():
                melody_part = original_score.parts[0]
            else:
                melody_part = original_score
            
            # 2. 关键修复：结构化伴奏轨道
            print("   -> 正在构建伴奏小节结构 (makeMeasures)...")
            try:
                # 强制量化并划分小节 (InPlace)
                accompaniment_part.makeMeasures(inPlace=True)
            except Exception as e:
                print(f"⚠️ 伴奏小节划分警告: {e} (尝试继续)")
            
            # 3. 关键修复：安全检查旋律轨道
            # 【Fix】使用 getElementsByClass 替代 hasMeasureStream
            try:
                has_measures = len(melody_part.getElementsByClass(music21.stream.Measure)) > 0
                if not has_measures:
                     print("   -> 旋律轨道未检测到小节，正在自动划分...")
                     melody_part.makeMeasures(inPlace=True)
            except Exception as e:
                print(f"⚠️ 旋律小节检查警告: {e}")

            # 4. 命名轨道
            melody_part.id = 'Melody'
            melody_part.partName = 'Melody'
            accompaniment_part.id = 'Accompaniment'
            accompaniment_part.partName = f'Style: {style}'

            # 5. 构建总谱
            final_score = music21.stream.Score()
            
            # 插入元数据
            final_score.insert(0, music21.metadata.Metadata())
            final_score.metadata.title = f"AI Generation - {style}"
            final_score.metadata.composer = "AI_G Pipeline"

            # 插入轨道
            final_score.insert(0, melody_part)
            final_score.insert(0, accompaniment_part)

            # 6. 写入文件
            final_score.write('midi', fp=output_midi_path)
            print(f"💾 最终成品已保存: {output_midi_path}")
            return True

        except Exception as e:
            print(f"❌ Phase 3 (合成) 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

# ================= 测试入口 =================

if __name__ == "__main__":
    # 配置
    input_file = r"samples\test_result.mid"     
    output_file = r"samples\output_test_melody.mid"
    target_style = "Pop Ballad"            

    # 运行
    pipeline = AI_Music_Generator()
    
    # 检查测试文件
    if not os.path.exists(input_file):
        alt_path = os.path.join("src", "test_melody.mid")
        if os.path.exists(alt_path):
            input_file = alt_path
        else:
            print(f"⚠️ 请准备测试文件: {input_file}")
            sys.exit(0)

    success = pipeline.process(input_file, output_file, style=target_style)
    
    if success:
        print("\n🎉 全流程执行成功！请播放 final_output.mid")
    else:
        print("\n💀 流程中断。")