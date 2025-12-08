# 文件: src/config.py

import os

# ================= 1. 路径配置 =================
# 这样写的好处是：无论你在哪里运行代码，路径都是相对于项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')
DATASET_PATH = os.path.join(DATA_DIR, 'dataset_encoded.json')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

# ================= 2. 模型参数 (最重要！只有这里定义一次) =================
# 以后想改模型大小，只改这里，Train 和 Inference 会自动同步
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HIDDEN_DIM = 128  # 如果觉得模型太笨，就把这个改成 256
DROPOUT = 0.5

# ================= 3. 训练参数(超参数) =================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS = 100
CLIP = 1

# 【进阶设置 - 新增】
SEED = 42                   # 随机种子 (保证每次训练结果可复现)
TEACHER_FORCING_RATIO = 0.5 # 教学强制率 (0.5 = 50%概率用真实答案纠正模型)
INIT_WEIGHT_STD = 0.01      # 权重初始化的标准差

# 【核心策略 - 新增】
# 防止 AI 偷懒的惩罚权重
# 如果模型一直输出 "_"，就把这个数调小 (如 0.01)
# 如果模型太乱动，就把这个数调大 (如 0.1)
HOLD_LOSS_WEIGHT = 0.04

# ================= 4. 特殊 Token 定义 (新增) =================
# 必须与 vocab.json 和 utils.py 里的定义保持一致
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "0"

# ================= 5. 其他配置 =================
temperature=1.0