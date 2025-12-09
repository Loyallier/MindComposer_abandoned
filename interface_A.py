import os
import sys
# 强行把根目录加入 sys.path
# 1. 获取所在的目录 (src/ChordGenerator_A)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 往上跳两级，找到根目录 (src -> Root)
project_root = os.path.dirname(os.path.dirname(current_dir))
# 3. 加入路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"🔧 已将根目录挂载: {project_root}")

try:
    from src.ChordGenerator_A.inference import AIComposer
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保你的文件结构包含 src/ChordGenerator_A/inference.py")
    sys.exit(1)

# ================= 全局单例变量 =================
# 这种写法保证模型只会被加载一次，而不是每次调用函数都加载
_composer_instance = None

def get_composer():
    """
    获取或初始化 AIComposer 实例 (单例模式)
    """
    global _composer_instance
    if _composer_instance is None:
        print("⏳ 正在初始化 AI Composer 模型...")
        try:
            _composer_instance = AIComposer()
            print("✅ 模型加载完成，准备生成！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    return _composer_instance

def generate_harmony(melody_sequence, temperature=1.0):
    """
    【供 B 组和 UI 调用的主函数】
    
    输入:
        melody_sequence: List[int] 或 List[str]
        例如: [60, 62, 64, 60, 0, 0, 67] 
        (代表 MIDI 音高，0 代表休止符/延音)
    
    输出:
        List[str]
        例如: ['C', 'C', 'Dm', 'C', 'G7', 'G7', 'C']
        (返回与输入长度一一对应的和弦名称列表)
    """
    composer = get_composer()
    if not composer:
        return []

    # 1. 简单的容错处理：把 int 转为 str，适配模型输入
    # 模型训练时使用的是字符串 token (如 '60', '0')
    melody_tokens = [str(note) for note in melody_sequence]

    # 2. 调用核心预测功能
    try:
        # top_k=3 保证了一定的随机性但不会乱弹
        chord_sequence = composer.predict(melody_tokens, temperature=temperature, top_k=3)
        return chord_sequence
    except Exception as e:
        print(f"⚠️ 生成过程中出错: {e}")
        # 如果出错，返回全空列表或全 'N.C.' (根据需求)
        return ["N.C."] * len(melody_sequence)

# ================= 本地测试代码 =================
if __name__ == "__main__":
    # 模拟一段《小星星》旋律: 1 1 5 5 6 6 5 (C大调: 60, 60, 67, 67, 69, 69, 67)
    test_melody = [60, 60, 67, 67, 69, 69, 67, 0]
    
    print("-" * 30)
    print(f"🎵 输入旋律: {test_melody}")
    result = generate_harmony(test_melody)
    print(f"🎹 生成和弦: {result}")
    print("-" * 30)