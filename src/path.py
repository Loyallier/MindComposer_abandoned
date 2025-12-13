from pathlib import Path

# =========================================================
# 🧭 核心逻辑: 自动定位锚点
# =========================================================

# 1. 获取当前文件 (src/path.py) 的绝对路径
_CURRENT_FILE = Path(__file__).resolve()

# 2. 定位项目根目录 (根据当前文件位置往上找两层)
# _CURRENT_FILE.parent 是 'src' 文件夹
# _CURRENT_FILE.parent.parent 是 项目根目录 (E:/.../)
ROOT_DIR = _CURRENT_FILE.parent.parent

# =========================================================
# 📂 目录定义 (基于 ROOT_DIR)
# =========================================================

# 源代码目录
SRC_DIR = ROOT_DIR / "src"

# 数据目录
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INTERIM_DIR = DATA_DIR / "interim"

# 模型目录
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# 日志目录
LOGS_DIR = ROOT_DIR / "logs"

# 输出目录
OUTPUTS_DIR = ROOT_DIR / "generated_outputs"

# =========================================================
# 📄 具体文件路径 (给代码直接调用)
# str(path.)
# =========================================================

# 1. 字典文件 (训练和推理都需要)
VOCAB_PATH = DATA_PROCESSED_DIR / "vocab.json"

# 2. [V3.1 更新] 编码后的数据集 (已拆分为训练集和验证集)
# 替代原本单一的 dataset_encoded.json
TRAIN_DATASET_PATH = DATA_PROCESSED_DIR / "train_data.json"
VAL_DATASET_PATH = DATA_PROCESSED_DIR / "val_data.json"

# 3. 最佳模型权重 (推理需要)
# 指向 models/best_model.pth
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# 4. 最后一次检查点 (继续训练需要)
LAST_CHECKPOINT_PATH = MODELS_DIR / "last_checkpoint.pth"

# =========================================================
# 🛠️ 辅助功能
# =========================================================

def validate_paths():
    """
    检查关键文件是否存在，防止运行一半报错。
    返回 True 表示一切正常，False 表示有文件缺失。
    """
    missing = []
    
    # 检查字典
    if not VOCAB_PATH.exists():
        missing.append(f"❌ 字典缺失: {VOCAB_PATH}")
    else:
        print(f"✅ 字典路径: {VOCAB_PATH}")

    # 检查模型 (如果是推理模式，模型必须存在)
    if not BEST_MODEL_PATH.exists():
        missing.append(f"⚠️ 模型缺失: {BEST_MODEL_PATH} (如果是首次训练请忽略)")
    else:
        print(f"✅ 模型路径: {BEST_MODEL_PATH}")

    if missing:
        print("\n".join(missing))
        return False
    
    return True

# 当直接运行此文件时，进行自我检查
if __name__ == "__main__":
    print(f"📍 项目根目录: {ROOT_DIR}")
    validate_paths()