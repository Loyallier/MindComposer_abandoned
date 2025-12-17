import os
import sys
import music21

# ================= 1. 环境挂载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入 A 组预测器
try:
    from src.ChordGenerator_A.predict_midi import ChordPredictor
    print("✅ Group A (ChordPredictor) 导入成功")
except ImportError as e:
    print(f"❌ Group A 导入失败: {e}")
    sys.exit(1)

# 导入 B 组接口
try:
    from interface_B import render_accompaniment
    print("✅ Group B (Interface) 导入成功")
except ImportError as e:
    print(f"❌ Group B 导入失败: {e}")
    render_accompaniment = None

# ================= 2. 配置 =================
TEST_MIDI_PATH = os.path.join("samples", "test_melody.mid")
OUTPUT_MIDI_PATH = os.path.join("samples", "final_integrated_output.mid")
TARGET_STYLE = "Pop Ballad" 

def main():
    print("\n🔗 启动全链路集成测试 (Pipeline Verification - Fixed)...")
    
    if not os.path.exists(TEST_MIDI_PATH):
        print(f"❌ 找不到测试文件: {TEST_MIDI_PATH}")
        return

    # --- Step 1: Group A (和弦预测) ---
    print("\n[Step 1] 正在运行 Group A 模型预测和弦...")
    try:
        predictor = ChordPredictor()
        chord_tokens = predictor.run(TEST_MIDI_PATH)
        
        if not chord_tokens:
            print("❌ Group A 返回了空列表！")
            return
            
        print(f"   ✅ A组预测成功。长度: {len(chord_tokens)} tokens")
        print(f"   -> 预览: {chord_tokens[:10]}...")
        
    except Exception as e:
        print(f"❌ Group A 运行出错: {e}")
        return

    # --- Step 2: Group B (织体渲染) ---
    print(f"\n[Step 2] 正在调用 Group B 接口渲染伴奏...")
    
    if render_accompaniment is None:
        print("❌ B组接口不可用，跳过。")
        return

    try:
        # 调试代码
        print(f"   [DEBUG] A组原始 Tokens 预览: {chord_tokens[:16]}")
        
        # 模拟 B 组内部合并逻辑进行检查 (请确保路径正确)
        from src.TextureRender_B.decision_logic_B import _consolidate_chords
        test_consolidated = _consolidate_chords(chord_tokens)
        print(f"   [DEBUG] 合并后的片段数: {len(test_consolidated)}")
        for i, (ch, dur) in enumerate(test_consolidated[:5]):
            print(f"      - 片段 {i}: 和弦={ch}, 时长={dur}QL")
            
        # B组返回的是一个 music21.stream.Part 对象
        accompaniment_part = render_accompaniment(
            melody_midi_path=TEST_MIDI_PATH,
            chord_sequence=chord_tokens,
            style=TARGET_STYLE
        )
        
        if accompaniment_part is None or len(accompaniment_part.flatten().notes) == 0:
            print("⚠️ Group B 返回了空轨道。")
        else:
            print(f"   ✅ B组渲染成功 (Music21 Part)。")
            
    except Exception as e:
        print(f"❌ Group B 运行出错: {e}")
        return

    # --- Step 3: 合成与导出 (修复版) ---
    print("\n[Step 3] 标准化合成最终 MIDI...")
    try:
        # 1. 读取原旋律
        original_score = music21.converter.parse(TEST_MIDI_PATH)
        if original_score.hasPartLikeStreams():
            melody_part = original_score.parts[0]
        else:
            melody_part = original_score
            
        melody_part.id = "Melody"
        melody_part.partName = "Original Melody"

        # 2. 伴奏轨道标准化 (关键修复！！！)
        accompaniment_part.id = "Accompaniment"
        accompaniment_part.partName = f"AI Gen ({TARGET_STYLE})"
        
        print("   -> 正在构建小节结构 (makeMeasures)...")
        # 🚨 B组返回的通常是 Flat Stream (无小节线)，直接写 MIDI 会报错
        # 必须先划分小节
        try:
            accompaniment_part.makeMeasures(inPlace=True)
        except Exception as e:
            print(f"   ⚠️ 伴奏小节划分警告: {e}")

        # 3. 组装 Score
        final_score = music21.stream.Score()
        
        # 插入元数据
        md = music21.metadata.Metadata()
        md.title = "AI Pipeline Output"
        final_score.insert(0, md)
        
        # 插入轨道
        final_score.insert(0, melody_part)
        final_score.insert(0, accompaniment_part)

        # 4. 保存
        os.makedirs(os.path.dirname(OUTPUT_MIDI_PATH), exist_ok=True)
        final_score.write('midi', fp=OUTPUT_MIDI_PATH)
        print(f"🎉 流程跑通！文件已保存至: {OUTPUT_MIDI_PATH}")

    except Exception as e:
        print(f"❌ 合成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()