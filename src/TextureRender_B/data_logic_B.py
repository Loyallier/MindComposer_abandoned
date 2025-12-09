from typing import List, Dict, Tuple, Any
import music21 as m21

# Group B phase 1: logic)
# 包含：和弦结构、简化和弦解析、伴奏织体模板。

# A. 和弦类型定义 (Chord Type Definitions)
# 键名：简化和弦类型（例如：Maj, min7, Dom7）
# 值：相对于根音（0）的半音音高偏移量（MIDI Offsets）
CHORD_TYPES: Dict[str, List[int]] = {
    "Maj": [0, 4, 7],       # 大三和弦 (Root, M3, P5)
    "min": [0, 3, 7],       # 小三和弦 (Root, m3, P5)
    "Dom7": [0, 4, 7, 10],  # 属七和弦 (Root, M3, P5, m7)
    "Maj7": [0, 4, 7, 11],  # 大七和弦 (Root, M3, P5, M7)
    "min7": [0, 3, 7, 10],  # 小七和弦 (Root, m3, P5, m7)
    "dim": [0, 3, 6],       # 减三和弦 (Root, m3, d5)
    "sus4": [0, 5, 7],      # 挂四和弦 (Root, P4, P5)
    "Maj6": [0, 4, 7, 9],   # 大六和弦 (Root, M3, P5, M6)
    "min6": [0, 3, 7, 9],   # 小六和弦 (Root, m3, P5, M6)
    "NoChord": [],          # 【新增】休止符号 '0' 对应的类型，不含任何音高偏移
}

# B. 和弦解析函数 (Chord Parsing Function)
def parse_simplified_chord(chord_label: str) -> Tuple[str, str]:
    """
    将简化的和弦标签解析为根音名称和和弦类型。
    例如: 'C#m7' -> ('C#', 'min7')
    """
    
    # 【新增】特殊符号处理：'0' 和 '_' 不通过标准解析
    if chord_label == '0':
        return 'C', 'NoChord' # 根音用 C 占位，类型为 NoChord
    if chord_label == '_':
        # 此函数不应处理 '_'，如果传入，说明调用逻辑有误，返回一个默认值
        return 'C', 'Maj' 

    root_name = ""
    chord_type = ""
    
    # 提取可能的根音（A-G，可能带 # 或 b）
    # 优先处理双字符根音 (e.g., 'A#', 'Bb')
    if len(chord_label) >= 2 and chord_label[1] in ('#', 'b', '-'): 
        root_name = chord_label[:2].replace('b', '-')
        remaining = chord_label[2:]
    elif chord_label and chord_label[0].isalpha(): 
        # 简单根音（A, B, C, D, E, F, G）
        root_name = chord_label[0]
        remaining = chord_label[1:]
    else:
        return 'C', 'Maj' # 无法识别时默认返回 C Maj

    # 查找匹配的和弦类型 - 优先检查最长的后缀
    if remaining.startswith('m7'): chord_type = 'min7'
    elif remaining.startswith('min7'): chord_type = 'min7'
    elif remaining.startswith('maj7'): chord_type = 'Maj7'
    elif remaining.startswith('sus4'): chord_type = 'sus4'
    elif remaining.startswith('dim'): chord_type = 'dim'
    elif remaining.startswith('Maj6'): chord_type = 'Maj6'
    elif remaining.startswith('min6'): chord_type = 'min6'
    elif remaining.startswith('7'): chord_type = 'Dom7' # 【修改】处理 '7' 后缀
    elif remaining.startswith('m'): chord_type = 'min' # 【修改】处理 'm' 后缀
    elif remaining.startswith('min'): chord_type = 'min'
    elif not remaining: 
        # 【修改】无后缀，默认为大三和弦
        chord_type = 'Maj'
    else:
        # 无法识别的后缀，默认为大三和弦
        chord_type = 'Maj'
        
    return root_name, chord_type

# C. 伴奏织体模板 (Accompaniment Pattern Templates)
# 结构：(时间偏移 QL, 和弦音索引, 时长 QL)
# 和弦音索引: 0=根音, 1=三度音, 2=五度音, 3=七度音...
PATTERN_TEMPLATES: Dict[str, List[Tuple[float, int, float]]] = {
    # 风格 1: 流行民谣 (Pop Ballad) - 根音+琶音分解，4/4 拍
    "Pop Ballad": [
        (0.0, 0, 1.0), # 1 拍：根音 (Bass)
        (1.0, 1, 0.5), # 2 拍开始：三音
        (1.5, 2, 0.5), # 2 拍半：五音
        (2.0, 3, 0.5), # 3 拍开始：七音 (或重复根音)
        (2.5, 2, 0.5), # 3 拍半：五音
        (3.0, 1, 1.0), # 4 拍：三音
    ],
    
    # 风格 2: 华尔兹 (Waltz) - 根音+和弦音，3/4 拍
    "Waltz": [
        (0.0, 0, 1.0), # 1 拍：根音 (Bass)
        (1.0, 1, 1.0), # 2 拍：三音 (通常是和弦音，可以重复)
        (2.0, 2, 1.0), # 3 拍：五音 (通常是和弦音)
    ],
    
    # 风格 3: 稀疏琶音 (Sparse Arpeggio) - 密度高时的默认选项，4/4 拍
    # 琶音间隔较长，音符较少，避免与旋律冲突
    "Sparse_Arpeggio": [
        (0.0, 0, 1.0), # 1 拍：根音
        (2.0, 1, 0.5), # 3 拍：三音
        (2.5, 2, 0.5), # 3 拍半：五音
        (3.5, 3, 0.5), # 4 拍半：七音
    ],
    
    "March": [
        (0.0, 0, 0.5),  # 1 拍：根音 (强拍)
        (0.5, 1, 0.5),  # 弱拍：和弦音
        (1.0, 2, 0.5),  # 2 拍：五音 (强拍)
        (1.5, 1, 0.5),  # 弱拍：和弦音
        (2.0, 0, 0.5),  # 3 拍：根音 (强拍)
        (2.5, 1, 0.5),  # 弱拍：和弦音
        (3.0, 2, 0.5),  # 4 拍：五音 (强拍)
        (3.5, 1, 0.5),  # 弱拍：和弦音
    ],
    
    "Jazz Swing": [
        (0.0, 0, 0.75), # 1 拍：根音 (长)
        (1.0, 1, 0.25), # 2 拍：三音 (短) - Swing 节奏
        (1.5, 2, 0.75), # 2 拍半：五音 (长)
        (2.5, 3, 0.25), # 3 拍半：七音 (短)
        (3.0, 2, 1.0),  # 4 拍：五音 (长)
    ]
}