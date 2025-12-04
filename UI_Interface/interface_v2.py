import os
import time
import logging
import shutil
from typing import List, Dict, Any, Union

# ==========================================
# 0. 全局配置 (Global Configuration)
# ==========================================
# 【关键开关】
# True = 演示模式 (返回假数据，用于UI开发和流程跑通)
# False = 实战模式 (调用真实模型和生成算法，Day 8 联调时切换)
USE_MOCK_MODELS = True 

# 路径配置
OUTPUT_DIR = "generated_outputs"
LOG_DIR = "logs"

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "system.log")),
        logging.StreamHandler()
    ]
)

# 允许的风格列表 (B组需实现这些风格的模板)
VALID_STYLES = ["Pop Ballad", "Waltz", "March", "Jazz Swing"]

# ==========================================
# 1. Group A: AI 预测模块 (The Brain)
# ==========================================
def predict_harmony(melody_midi_path: str) -> List[str]:
    """
    [A组 核心任务]
    输入: 用户的旋律 MIDI 文件路径
    输出: 一个与时间对齐的和弦标签列表 (16分音符切片)
    
    格式示例: ['C', '_', '_', '_', 'G7', '_', '_', '_', ...]
    ('_' 代表延音 HOLD)
    """
    logger = logging.getLogger("Group_A_AI")
    logger.info(f"正在分析旋律: {melody_midi_path} ...")

    if USE_MOCK_MODELS:
        # --- Mock 模式 (模拟 AI) ---
        time.sleep(1) # 假装在推理
        logger.info("模拟预测完成")
        
        # 返回一个假的 4 小节和弦序列 (假设 4/4 拍, 每小节 16 个切片)
        # C大调 -> G大调 -> A小调 -> F大调
        mock_seq =  ['C'] + ['_'] * 15 + \
                    ['G'] + ['_'] * 15 + \
                    ['Am'] + ['_'] * 15 + \
                    ['F'] + ['_'] * 15
        return mock_seq

    else:
        # --- Real 模式 (真实 AI) ---
        # TODO: [A组填空]
        # 1. 加载 pytorch 模型 (建议在函数外预加载，避免每次调用都加载)
        # 2. 使用 music21 或 pretty_midi 读取 midi_path
        # 3. 将旋律切片化 (Quantization step=0.25)
        # 4. 喂给模型预测
        # 5. 将预测的数字 ID 转回和弦文本 (vocab decode)
        
        # from src.inference import ChordPredictor
        # predictor = ChordPredictor("models/best_checkpoint.pt")
        # return predictor.predict(melody_midi_path)
        
        raise NotImplementedError("A组模型尚未接入！请检查 USE_MOCK_MODELS 开关。")

# ==========================================
# 2. Group B: 逻辑生成模块 (The Hands)
# ==========================================
def render_music(
    melody_midi_path: str, 
    chord_sequence: List[str], 
    style: str
) -> str:
    """
    [B组 核心任务]
    输入: 原始旋律 + A组预测的和弦序列 + 用户选择的风格
    输出: 最终生成的 MIDI 文件路径
    """
    logger = logging.getLogger("Group_B_Logic")
    logger.info(f"开始生成伴奏... 风格: {style}")
    
    # 构造输出文件名
    filename = f"result_{int(time.time())}_{style.replace(' ', '_')}.mid"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if USE_MOCK_MODELS:
        # --- Mock 模式 (模拟生成) ---
        time.sleep(1)
        
        # 直接把用户上传的文件复制过去，假装是生成结果
        # 这样 UI 播放时至少能听到声音 (虽然只有旋律)
        try:
            shutil.copy(melody_midi_path, output_path)
            logger.info(f"模拟生成完毕: {output_path}")
        except Exception as e:
            # 如果复制失败，创建一个空的 dummy 文件
            with open(output_path, 'wb') as f:
                f.write(b'MThd...') 
            logger.warning(f"无法复制源文件，创建空文件: {e}")
            
        return output_path

    else:
        # --- Real 模式 (真实渲染) ---
        # TODO: [B组填空]
        # 1. 使用 music21 读取 melody_midi_path
        # 2. 解析 chord_sequence (处理 '_' 延音标记，计算每个和弦的时长)
        # 3. 根据 style 调用对应的伴奏模板 (Pattern)
        # 4. 将生成的伴奏音符写入新的 Track
        # 5. 合并旋律和伴奏，保存到 output_path
        
        # from src.generator import AccompanimentGenerator
        # gen = AccompanimentGenerator()
        # gen.render(melody_midi_path, chord_sequence, style, output_path)
        # return output_path
        
        raise NotImplementedError("B组生成逻辑尚未接入！")

# ==========================================
# 3. Group C: 主控接口 (Main Pipeline)
# ==========================================
def generate_song(uploaded_file_path: str, selected_style: str = "Pop Ballad") -> Dict[str, Any]:
    """
    [UI 组唯一调用的接口]
    """
    logger = logging.getLogger("Pipeline")
    
    # 结果容器
    result = {
        "success": False,
        "message": "",
        "midi_path": None,
        "chord_preview": [] # 用于在前端展示和弦走向
    }

    try:
        # 1. 校验
        if not os.path.exists(uploaded_file_path):
            raise FileNotFoundError("未找到上传文件")
        
        if selected_style not in VALID_STYLES:
            logger.warning(f"未知风格 {selected_style}，使用默认风格")
            selected_style = VALID_STYLES[0]

        # 2. 执行 A 组任务 (预测)
        logger.info(">>> Stage 1: AI Prediction <<<")
        chords = predict_harmony(uploaded_file_path)
        
        # 为了前端展示好看，我们把 '_' 过滤掉，只返回纯和弦列表给 UI 显示
        # 例如: ['C', 'G7', 'Am']
        display_chords = [c for c in chords if c != '_']
        result["chord_preview"] = display_chords[:10] # 只展示前10个，避免刷屏

        # 3. 执行 B 组任务 (生成)
        logger.info(">>> Stage 2: Rule-based Generation <<<")
        final_midi = render_music(uploaded_file_path, chords, selected_style)
        
        # 4. 完成
        result["success"] = True
        result["midi_path"] = final_midi
        result["message"] = "生成成功！请点击播放或下载。"
        logger.info(f"流程结束。输出: {final_midi}")

    except Exception as e:
        logger.error(f"流程异常: {str(e)}", exc_info=True)
        result["message"] = f"生成失败: {str(e)}"

    return result

# ==========================================
# 4. 测试入口 (Debug)
# ==========================================
if __name__ == "__main__":
    print("--- 正在测试接口连通性 ---")
    # 创建一个假的 MIDI 文件用于测试
    dummy_input = "test_input.mid"
    with open(dummy_input, 'w') as f: f.write("dummy")
    
    # 模拟调用
    res = generate_song(dummy_input, "Waltz")
    
    print("\n返回结果:")
    print(res)
    
    # 清理垃圾
    if os.path.exists(dummy_input): os.remove(dummy_input)