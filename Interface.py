import os
import time
import logging
from typing import List, Dict, Tuple, Optional

# ==========================================
# 0. 全局配置与日志 (Global Config)
# ==========================================
# 调试开关：True = 使用假数据测试 UI；False = 调用真实 AI/逻辑代码
# 【关键】Day 1-7 设为 True，Day 8 联调时改为 False
USE_MOCK_MODELS = True 

# 配置日志输出，看起来更专业
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Group %(name)s] - %(message)s'
)

# 定义允许的风格（B组要把这些实现了）,根据你的喜好改动，记得保证全文件名字统一
# 题外话，我们这次用的Nottingham Music Database都是英美民谣，风格加上去有概率变奇怪。
VALID_STYLES = ["Pop Ballad", "Waltz", "March", "Jazz"]

# ==========================================
# 1. Group A: AI 预测接口
# ==========================================
def predict_chords(melody_midi_path: str) -> List[str]:
    """
    [Group A 核心任务]
    输入一段单音旋律 MIDI，预测对应的和弦标签序列。
    
    Args:
        melody_midi_path (str): 用户上传的 MIDI 文件路径。
        
    Returns:
        List[str]: 和弦标签列表，例如 ['C_Maj', 'A_Min', ...]。
    """
    logger = logging.getLogger("A")
    logger.info(f"正在分析旋律文件: {melody_midi_path}")

    if USE_MOCK_MODELS:
        # --- 模拟模式 (Mock Mode) ---
        time.sleep(1.5) # 假装 AI 在思考
        logger.info("模拟预测完成")
        return ["C_Maj", "G_Maj", "A_Min", "E_Min", "F_Maj", "C_Maj", "F_Maj", "G_Dom7"]
    
    else:
        # --- 真实模式 (Real Mode) ---
        # TODO: A组在这里填入真实代码
        # 1. import src.predict
        # 2. model = load_model(...)
        # 3. result = model.predict(melody_midi_path)
        # return result
        raise NotImplementedError("A组的真实模型尚未接入！请将 USE_MOCK_MODELS 设为 True")

# ==========================================
# 2. Group B: 逻辑生成接口
# ==========================================
def render_accompaniment(
    original_midi_path: str, 
    chord_sequence: List[str], 
    style: str, 
    output_midi_path: str
) -> str:
    """
    [Group B 核心任务]
    根据和弦标签和风格，生成左手伴奏，并与右手旋律合并。
    
    Args:
        original_midi_path (str): 原始旋律文件（用于提取右手）。
        chord_sequence (List[str]): A组预测出的和弦列表。
        style (str): 用户选择的风格（必须在 VALID_STYLES 中）。
        output_midi_path (str): 结果保存的路径。
        
    Returns:
        str: 生成的 MIDI 文件路径 (成功时返回 output_midi_path)。
    """
    logger = logging.getLogger("B")
    
    if style not in VALID_STYLES:
        logger.warning(f"风格 '{style}' 未定义，回退到默认风格")
        style = VALID_STYLES[0]

    logger.info(f"正在渲染伴奏... 风格: {style}, 和弦数: {len(chord_sequence)}")

    if USE_MOCK_MODELS:
        # --- 模拟模式 ---
        time.sleep(2) # 假装在合成
        # 创建一个空文件占位，防止报错
        with open(output_midi_path, 'w') as f:
            f.write("Dummy MIDI content")
        logger.info(f"生成完毕: {output_midi_path}")
        return output_midi_path
    
    else:
        # --- 真实模式 ---
        # TODO: B组在这里填入真实代码
        # 1. right_hand = music21.converter.parse(original_midi_path)
        # 2. left_hand = generate_notes(chord_sequence, style)
        # 3. combined = right_hand + left_hand
        # 4. combined.write('midi', fp=output_midi_path)
        # return output_midi_path
        raise NotImplementedError("B组的生成逻辑尚未接入！")

def generate_score_pdf(midi_path: str) -> str:
    """
    [Group B 额外任务] 将 MIDI 转为 PDF 乐谱 (可选，若太难可只返回图片路径)
    """
    if USE_MOCK_MODELS:
        return "assets/mock_score.png" # UI组需要自己放一张假图片在这里
    else:
        # 真实转换逻辑
        return midi_path.replace(".mid", ".pdf")

# ==========================================
# 3. Group C: 主控流水线 (Pipeline)
# ==========================================
def run_music_gen_pipeline(uploaded_file_path: str, selected_style: str) -> Dict:
    """
    [Group C 调用入口]
    串联 A组 和 B组 的工作，处理异常。
    """
    logger = logging.getLogger("C")
    logger.info(">>> 开始处理用户请求 <<<")
    
    # 定义输出路径 (保存在 samples 文件夹)
    os.makedirs("samples", exist_ok=True)
    timestamp = int(time.time())
    final_midi_path = os.path.join("samples", f"gen_{timestamp}.mid")

    try:
        # Step 1: AI 预测
        chords = predict_chords(uploaded_file_path)
        
        # Step 2: 生成伴奏
        output_file = render_accompaniment(
            original_midi_path=uploaded_file_path,
            chord_sequence=chords,
            style=selected_style,
            output_midi_path=final_midi_path
        )
        
        # Step 3: 生成乐谱预览 (可选)
        pdf_file = generate_score_pdf(output_file)

        logger.info(">>> 流程成功结束 <<<")
        return {
            "success": True,
            "midi_file": output_file,
            "pdf_file": pdf_file,
            "chord_labels": chords, # 用于在前端展示给老师看
            "message": "生成成功"
        }

    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "系统内部错误，请检查日志"
        }

# ==========================================
# 4. 自测入口 (Self-Check)
# ==========================================
if __name__ == "__main__":
    # 在终端直接运行 python interface.py 即可测试流程是否通畅
    print("--- 接口自测模式 ---")
    test_result = run_music_gen_pipeline("test_melody.mid", "Waltz")
    print(f"执行结果: {test_result}")