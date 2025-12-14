import os
import sys
# ✅ 统一引用：从 src 包导入 path
# 强行把根目录加入 sys.path
# 1. 获取所在的目录 (src/ChordGenerator_A)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 往上跳两级，找到根目录 (src -> Root)
project_root = os.path.dirname(os.path.dirname(current_dir))
# 3. 加入路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"🔧 已将根目录挂载: {project_root}")
from src import path

# ---------------------------------------------------------

# ================= 1. 路径映射 =================
# 直接使用 path 模块定义的路径
VOCAB_PATH = path.VOCAB_PATH
TRAIN_DATASET_PATH = path.TRAIN_DATASET_PATH
VAL_DATASET_PATH = path.VAL_DATASET_PATH
MODEL_SAVE_PATH = path.BEST_MODEL_PATH
MODEL_DIR = path.MODELS_DIR

# ================= 2. 模型参数 (最重要！只有这里定义一次) =================
# 以后想改模型大小，只改这里，Train 和 Inference 会自动同步
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HIDDEN_DIM = 256  # 如果觉得模型太笨，就把这个改成 256
DROPOUT = 0.5

# [V3.1] 位置编码参数
POS_EMB_DIM = 32      # 位置向量维度 (与旋律拼接到一起)
POS_VOCAB_SIZE = 33   # 词表大小: 0-30(位置) + 31(BAR/SOS/EOS) + 32(PAD)
POS_PAD_IDX = 32      # Padding 专用的 Index

# ================= 3. 训练参数(超参数) =================
BATCH_SIZE = 72
LEARNING_RATE = 0.0005
N_EPOCHS = 100
CLIP = 1

# 【新增控制开关】
# True = 尝试加载 best_model.pth 继续训练
# False = 忽略旧模型，强制从头开始
RESUME_TRAINING = False
START_EPOCH = 0  # 您上次大概训练到的轮数

# 【进阶设置 - 新增】
SEED = 42  # 随机种子 (保证每次训练结果可复现)
TEACHER_FORCING_RATIO = 0.5  # 教学强制率 (0.5 = 50%概率用真实答案纠正模型)
INIT_WEIGHT_STD = 0.01  # 权重初始化的标准差

# TF 策略参数 (V3.5 放缓衰减)
TF_START_RATIO = 1.0
TF_END_RATIO = 0.0
# 📈 [V3.5] 让老师多带一会儿，80轮才彻底放手
TF_DECAY_EPOCHS = 80

# [V3.3] 保持 Hold 权重，因为节奏已经学会了，不需要再动这个
HOLD_LOSS_WEIGHT = 0.1

# [V3.3 新增] 权重衰减 (L2 Regularization)
WEIGHT_DECAY = 1e-4

# [V3.3 新增] 标签平滑
LABEL_SMOOTHING = 0.1

# ================= 4. 特殊 Token 定义 (新增) =================
# 必须与 vocab.json 和 utils.py 里的定义保持一致
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "0"
BAR_TOKEN = "<BAR>"
# ✅ [V3.1 新增] 必须明确定义 PAD_IDX (dataset.py 依赖此变量)
PAD_IDX = 0

# ================= 5. 推理参数 (新增) =================
# 控制生成的随机性和多样性
INFERENCE_TEMP = 1.0    # 温度 (0.8=保守, 1.2=狂野)
INFERENCE_TOP_K = 3     # 仅在概率最高的 K 个候选中采样