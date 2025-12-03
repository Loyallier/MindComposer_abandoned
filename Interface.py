import os
import time
import random

# ==========================================
# 1. 定义数据结构 (Data Structures)
# ==========================================
# 为了防止大家数据格式打架，我们约定一下：
# 和弦标签就是一个简单的字符串列表，例如: ['C_Maj', 'G_Dom7', 'A_Min']

# ==========================================
# 2. 给 Group A (模型组) 的接口规范
# ==========================================
def predict_chords_from_melody(midi_path):
    """
    [Group A 负责实现]
    输入: 用户上传的 MIDI 文件路径
    输出: 一个预测好的和弦标签列表 (List of Strings)
    """
    print(f"Running AI Model on {midi_path}...")
    
    # --- TODO: A组在这里加载模型，读取MIDI，返回预测结果 ---
    # 目前是 Mock (假数据)，为了让 UI 能跑通
    # 模拟 AI 思考了 1 秒
    time.sleep(1) 
    
    # 假装这是 AI 预测出来的 8 个小节的和弦
    mock_prediction = ["C_Maj", "G_Maj", "A_Min", "F_Maj", 
                       "C_Maj", "F_Maj", "G_Dom7", "C_Maj"]
    return mock_prediction

# ==========================================
# 3. 给 Group B (逻辑组) 的接口规范
# ==========================================
def generate_music_file(chord_sequence, style, output_path):
    """
    [Group B 负责实现]
    输入: 
        1. chord_sequence: A组预测出来的和弦列表
        2. style: 用户在 UI 上选的风格 (例如 'Pop', 'Waltz')
        3. output_path: 生成文件的保存路径
    输出: 
        成功生成的 MIDI 文件路径 (String)
    """
    print(f"Generating music with Style: {style}...")
    
    # --- TODO: B组在这里调用 music21，根据和弦生成音符，导出文件 ---
    # 目前是 Mock (假数据)
    
    # 这一步本来应该生成文件，现在我们先假设文件已经生成了
    # 实际开发时，B组要保证 output_path 真的产生了一个 .mid 文件
    if not os.path.exists(output_path):
        # 创建一个空文件骗过 UI，防止报错
        with open(output_path, 'w') as f:
            f.write("This is a dummy MIDI file.")
            
    return output_path

def generate_sheet_music(midi_path):
    """
    [Group B 负责实现]
    输入: 生成好的 MIDI 路径
    输出: 对应的 PDF 或 PNG 乐谱路径
    """
    # --- TODO: B组调用 music21 的 write('musicxml.pdf') ---
    return "assets/sample_sheet.png" # 返回一个假图片的路径用于测试

# ==========================================
# 4. 给 Group C (UI组) 的调用入口
# ==========================================
def run_pipeline(user_uploaded_file, selected_style):
    """
    [Group C 调用这个函数]
    这是主流程函数，它把 A组 和 B组 的工作串联起来。
    """
    
    # 1. 定义输出文件名
    timestamp = int(time.time())
    output_midi = f"output_{timestamp}.mid"
    
    try:
        # Step 1: 调用 A组 (AI 预测)
        chords = predict_chords_from_melody(user_uploaded_file)
        
        # Step 2: 调用 B组 (生成 MIDI)
        final_midi_path = generate_music_file(chords, selected_style, output_midi)
        
        # Step 3: 调用 B组 (生成乐谱)
        sheet_music_path = generate_sheet_music(final_midi_path)
        
        return {
            "success": True,
            "midi_path": final_midi_path,
            "pdf_path": sheet_music_path,
            "chords": chords, # 把和弦也返回给 UI，可以显示在界面上，很酷
            "message": "生成成功！"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "生成失败，请检查日志。"
        }

# ==========================================
# 测试代码 (直接运行这个文件看看通不通)
# ==========================================
if __name__ == "__main__":
    # 模拟 UI 组的调用
    print("--- 测试开始 ---")
    result = run_pipeline("test_input.mid", "Waltz")
    print("返回结果:", result)
    print("--- 测试结束 ---")