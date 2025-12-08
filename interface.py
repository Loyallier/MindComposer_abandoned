import os
import time
import logging
import shutil
from typing import List, Dict, Any, Union
# 【新增】导入 music21 和 B 组生成逻辑
import music21 as m21
from decision_logic_B import render_accompaniment_from_raw_inputs # 导入 B 组主入口函数

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
# Group A Section
# ==========================================

# 1. 定义一个私有变量，用来存模型实例，防止每次点击都重新加载
# (把它放在函数外面，这样全组都能看到 A 组有个全局模型变量，但不影响别人)
_group_a_model = None 

def predict_harmony(melody_list):
    """
    【A组任务】根据旋律预测和弦
    """
    global _group_a_model # 声明我们要使用上面那个全局变量

    # Step 1: 懒加载 (Lazy Import & Init)
    # 只有第一次运行这个函数时，才会去 import 和加载模型
    # 这样不会污染文件顶部的 import 区域，也不会拖慢程序启动速度
    if _group_a_model is None:
        try:
            print("⏳ [Group A] 正在唤醒 AI 模型...")
            # 动态导入，避免在文件头引入 src 包依赖
            from src.inference import AIComposer 
            _group_a_model = AIComposer()
            print("✅ [Group A] 模型就绪。")
        except Exception as e:
            print(f"❌ [Group A] 模型加载失败: {e}")
            return [] # 出错返回空列表，保证系统不崩

    # Step 2: 执行预测
    try:
        if not melody_list:
            return []
        
        # 调用 A 组核心逻辑
        return _group_a_model.predict(melody_list)
        
    except Exception as e:
        print(f"❌ [Group A] 预测出错: {e}")
        return []
        
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
        
        try:
            # 1. 调用 B 组的主入口函数，生成伴奏 Part
            accompaniment_part = render_accompaniment_from_raw_inputs(
                melody_midi_path, 
                chord_sequence, 
                style
            )
            
            # 2. 读取原始旋律 Score
            original_score = m21.converter.parse(melody_midi_path)
            
            # 3. 创建最终 Score 并合并旋律和伴奏
            final_score = m21.stream.Score()
            
            # 找到原始旋律 part (通常是第一个)
            if original_score.parts:
                final_score.insert(0, original_score.parts[0])
            else:
                logger.warning("无法从 MIDI 文件中提取旋律 Part。仅包含伴奏。")
                
            # 插入伴奏 Part (确保插入到正确的位置，通常是 0.0 offset)
            final_score.insert(0, accompaniment_part)

            # 4. 保存为 MIDI 文件
            final_score.write('midi', fp=output_path)
            
            logger.info(f"真实生成完毕: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"B组生成逻辑出错: {e}", exc_info=True)
            # 为了不阻断流程，可以在失败时创建一个空的 Part 并保存
            m21.stream.Part().write('midi', fp=output_path)
            raise RuntimeError(f"音乐生成失败: {e}")

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