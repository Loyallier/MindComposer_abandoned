from pathlib import Path

# =========================================================
# Core Logic: Automatic Anchor Positioning
# =========================================================

# 1. Get the absolute path of the current file (src/path.py)
_CURRENT_FILE = Path(__file__).resolve()

# 2. Locate the project root directory (Go up two levels from the current file)
# _CURRENT_FILE.parent is the 'src' folder
# _CURRENT_FILE.parent.parent is the Project Root directory (E:/.../)
ROOT_DIR = _CURRENT_FILE.parent.parent

# =========================================================
# Directory Definitions (Based on ROOT_DIR)
# =========================================================

# Source code directory
SRC_DIR = ROOT_DIR / "src"

# Data directories
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INTERIM_DIR = DATA_DIR / "interim"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Logs directory
LOGS_DIR = ROOT_DIR / "logs"

# Output directory
OUTPUTS_DIR = ROOT_DIR / "generated_outputs"

# =========================================================
# Specific File Paths (For direct code access)
# str(path.)
# =========================================================

# 1. Vocabulary file (Required for both training and inference)
VOCAB_PATH = DATA_PROCESSED_DIR / "vocab.json"

# 2. [V3.1 Update] Encoded datasets (Split into training and validation sets)
# Replaces the original single dataset_encoded.json
TRAIN_DATASET_PATH = DATA_PROCESSED_DIR / "train_data.json"
VAL_DATASET_PATH = DATA_PROCESSED_DIR / "val_data.json"

# 3. Best model weights (Required for inference)
# Points to models/best_model.pth
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# 4. Last checkpoint (Required to resume training)
LAST_CHECKPOINT_PATH = MODELS_DIR / "last_checkpoint.pth"

# =========================================================
# Helper Functions
# =========================================================

def validate_paths():
    """
    Check if critical files exist to prevent errors during runtime.
    Returns True if everything is normal, False if files are missing.
    """
    missing = []
    
    # Check vocabulary
    if not VOCAB_PATH.exists():
        missing.append(f"❌ Vocabulary missing: {VOCAB_PATH}")
    else:
        print(f"✅ Vocabulary path: {VOCAB_PATH}")

    # Check model (For inference mode, the model must exist)
    if not BEST_MODEL_PATH.exists():
        missing.append(f"⚠️ Model missing: {BEST_MODEL_PATH} (Ignore if this is the first training session)")
    else:
        print(f"✅ Model path: {BEST_MODEL_PATH}")

    if missing:
        print("\n".join(missing))
        return False
    
    return True

# Perform self-check when running this file directly
if __name__ == "__main__":
    print(f"📍 Project Root: {ROOT_DIR}")
    validate_paths()