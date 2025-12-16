import os
import sys

# ================= 1. 路径挂载 (确保能导入 src) =================
# 获取当前脚本所在目录 (假设在项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 验证路径
try:
    from src import path
    from src.ChordGenerator_A import config
    from src.ChordGenerator_A.predict_midi import ChordPredictor
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保脚本位于项目根目录，且 src 文件夹结构正确。")
    sys.exit(1)

# ================= 2. 测试逻辑 =================

def print_full_list(data_list, items_per_line=8):
    """
    辅助函数：完整打印列表，避免被省略号截断，同时保持可读性
    """
    print(f"\n📄 完整结果 (Total: {len(data_list)} tokens):")
    print("-" * 60)
    
    # 将列表分块打印
    for i in range(0, len(data_list), items_per_line):
        chunk = data_list[i : i + items_per_line]
        # 格式化打印：索引 + 内容
        print(f"[{i:03d}] " + "  ".join(f"{str(x):<6}" for x in chunk))
        
    print("-" * 60 + "\n")

def run_test(midi_path):
    print(f"🚀 [Test] 启动 Group A 预测测试")
    print(f"📂 输入文件: {midi_path}")
    
    # 1. 检查文件
    if not os.path.exists(midi_path):
        print(f"❌ 错误: 找不到文件 {midi_path}")
        print("💡 请将一个 MIDI 文件重命名为 test.mid 放入 data/raw/ 目录，或修改代码中的路径。")
        return

    # 2. 初始化预测器
    try:
        predictor = ChordPredictor() # 这里会自动加载模型
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 执行预测
    try:
        # 核心调用
        result_list = predictor.run(midi_path)
        
        # 4. 结果验证
        if not result_list:
            print("⚠️ 警告: 预测结果为空！请检查输入 MIDI 是否过短或无音符。")
            return
            
        # 检查是否包含非法符号 (如 EOS, SOS, BAR) - 理论上 predict_midi 应该清洗掉了
        invalid_tokens = [t for t in result_list if t in [config.SOS_TOKEN, config.EOS_TOKEN, config.BAR_TOKEN]]
        if invalid_tokens:
            print(f"⚠️ 警告: 结果中包含未清洗的特殊符号: {invalid_tokens}")
        else:
            print("✅ 格式检查通过: 无特殊 Token。")

        # 5. 完整打印
        print_full_list(result_list)
        
        # 6. 简单统计
        chord_count = sum(1 for x in result_list if x not in ["_", "0", "N.C."])
        print(f"📊 统计: 有效和弦数: {chord_count} | 保持/空: {len(result_list) - chord_count}")

    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # ===在此处修改测试文件的路径===
    # 默认路径: data/raw/test.mid
    test_file = os.path.join("samples", "audit_detailed", "case_1_1_Melody.mid")
    
    # 如果您想手动指定绝对路径，请取消注释下一行：
    # test_file = r"C:\Your\Path\To\melody.mid"
    
    run_test(test_file)