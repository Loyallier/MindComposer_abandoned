import os
import sys
from typing import List, Dict, Tuple, Any

# 1. 配置路径（与 interface_A.py 相似）
# 强行把根目录加入 sys.path，确保能导入 B 组代码
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 B 组代码在 TextureRender_B/
project_root = os.path.dirname(os.path.dirname(current_dir)) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入 B 组核心功能
try:
    # 导入 B 组的主入口函数
    from TextureRender_B.decision_logic_B import render_accompaniment_from_raw_inputs
    import music21 as m21
except ImportError as e:
    print(f"❌ Group B 导入错误: {e}")
    print("请确保你的文件结构包含 TextureRender_B/decision_logic_B.py")
    sys.exit(1)

# ================= 全局接口函数 =================

def render_accompaniment(
    melody_midi_path: str, 
    chord_sequence: List[str], 
    style: str
) -> m21.stream.Part:
    """
    【供 interface.py 调用的主函数】
    
    输入:
        melody_midi_path (str): 原始旋律 MIDI 文件路径。
        chord_sequence (List[str]): A 组预测的 16 分音符切片和弦序列。
        style (str): 用户选择的风格（例如 'Pop Ballad', 'Waltz'）。
    
    输出:
        m21.stream.Part: 包含完整伴奏的 music21 Part 对象。
    """

    # 1. 简单的输入校验
    if not os.path.exists(melody_midi_path):
        print(f"⚠️ 旋律文件未找到: {melody_midi_path}")
        return m21.stream.Part()

    # 2. 调用 B 组核心渲染逻辑
    try:
        accompaniment_part = render_accompaniment_from_raw_inputs(
            melody_midi_path, 
            chord_sequence, 
            style
        )
        print(f"✅ Group B 伴奏 Part 生成完成。")
        return accompaniment_part
        
    except Exception as e:
        print(f"❌ Group B 渲染过程中出错: {e}")
        # 如果出错，返回一个空的 Part，保证主流程不中断
        return m21.stream.Part()

# ================= 本地测试代码 (需要 music21 环境) =================
if __name__ == "__main__":
    print("--- 正在测试 interface_B.py 连通性 ---")
    
    # 注意：这里的测试需要一个真实的 MIDI 文件作为输入。
    # 假设 'test_melody.mid' 存在
    DUMMY_MIDI_PATH = "test_melody.mid" 

    # 创建一个假的 MIDI 文件用于测试（如果不存在）
    if not os.path.exists(DUMMY_MIDI_PATH):
        try:
            s = m21.stream.Stream()
            s.append(m21.note.Note('C4', quarterLength=4.0))
            s.write('midi', fp=DUMMY_MIDI_PATH)
            print(f"🔧 已创建临时的假 MIDI 文件: {DUMMY_MIDI_PATH}")
        except Exception as e:
            print(f"❌ 无法创建 dummy MIDI 文件，跳过测试。({e})")
            sys.exit(0)


    test_chords = ["C", "_", "_", "_", "G7", "_", "_", "_", "Am", "_", "_", "_", "F", "_", "_", "_"]
    test_style = "Pop Ballad"
    
    print(f"🎵 输入风格: {test_style}")
    print(f"🎹 输入和弦序列 (前8个): {test_chords[:8]}...")
    
    result_part = render_accompaniment(DUMMY_MIDI_PATH, test_chords, test_style)
    
    if result_part.notes:
        print(f"✅ 成功生成伴奏 Part，包含 {len(result_part.notes)} 个音符。")
    else:
        print(f"⚠️ 生成的伴奏 Part 为空。")

    # 清理垃圾
    if os.path.exists(DUMMY_MIDI_PATH):
        # os.remove(DUMMY_MIDI_PATH) # 生产环境中不应自动删除，这里保持注释
        pass