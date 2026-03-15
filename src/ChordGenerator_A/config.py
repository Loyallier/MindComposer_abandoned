import os
import sys
# Unified reference: Import path from src package
# Force the root directory into sys.path
# 1. Get the current directory (src/ChordGenerator_A)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up two levels to find the root directory (src -> Root)
project_root = os.path.dirname(os.path.dirname(current_dir))
# 3. Add to path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Root directory mounted: {project_root}")
from src import path

# ---------------------------------------------------------

# ================= 1. Path Mapping =================
# Use paths defined directly in the path module
VOCAB_PATH = path.VOCAB_PATH
TRAIN_DATASET_PATH = path.TRAIN_DATASET_PATH
VAL_DATASET_PATH = path.VAL_DATASET_PATH
MODEL_SAVE_PATH = path.BEST_MODEL_PATH
MODEL_DIR = path.MODELS_DIR

# ================= 2. Model Parameters (Crucial! Defined here once) =================
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HIDDEN_DIM = 256  
DROPOUT = 0.5

# [V3.1] Positional Encoding Parameters
POS_EMB_DIM = 32
POS_VOCAB_SIZE = 33 
POS_PAD_IDX = 32 

# ================= 3. Training Parameters (Hyperparameters) =================
BATCH_SIZE = 72
LEARNING_RATE = 0.0005
N_EPOCHS = 100
CLIP = 1

# [Control Switches]
# True = Attempt to load best_model.pth to continue training
# False = Ignore old model, force start from scratch
RESUME_TRAINING = False
START_EPOCH = 0  # The epoch number where you last stopped training

# [Advanced Settings - New]
SEED = 42  # Random seed (ensures training results are reproducible)
TEACHER_FORCING_RATIO = 0.5  # Teacher forcing ratio (0.5 = 50% chance to use ground truth to correct the model)
INIT_WEIGHT_STD = 0.01  # Standard deviation for weight initialization

# TF Strategy Parameters (V3.5 Slower Decay)
TF_START_RATIO = 1.0
TF_END_RATIO = 0.0
# [V3.5] Keep the teacher guidance longer; fully let go only after 80 epochs
TF_DECAY_EPOCHS = 80

# [V3.3] Keep Hold weight as rhythm is already learned and doesn't need much change
HOLD_LOSS_WEIGHT = 0.12

# [V3.3 New] Weight Decay (L2 Regularization)
WEIGHT_DECAY = 1e-4

# [V3.3 New] Label Smoothing
LABEL_SMOOTHING = 0.01

# ================= 4. Special Token Definitions (New) =================
# Must remain consistent with definitions in vocab.json and utils.py
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "0"
BAR_TOKEN = "<BAR>"
# [V3.1 New] Must explicitly define PAD_IDX (dataset.py depends on this variable)
PAD_IDX = 0
ZERO_IDX = 5  # Index for "0"

# ================= 5. Inference Parameters (New) =================
# Controls randomness and diversity of generation
INFERENCE_TEMP = 1.0    # Temperature
INFERENCE_TOP_K = 1     # Sample only from the top K most probable candidates