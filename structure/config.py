# config.py
# Central configuration file for the FinQA project

# --- Paths and File Names ---
BASE_PROJECT_DIR = r"E:\1.apps\obsidian_folder\Research\Research_code\FinQA"
DATA_DIR = "./dataset"
TRAIN_FILE = f"{DATA_DIR}/train.json"
DEV_FILE = f"{DATA_DIR}/dev.json"
TEST_FILE = f"{DATA_DIR}/test.json"

# SUGGESTION: Changed path to be more descriptive of the Mistral+LoRA model
MODEL_SAVE_PATH = "./fingpt_mistral_lora_final"

PREDICTIONS_FILE = "predictions_final.json"
EVALUATION_SCRIPT_PATH = "./code/evaluate/evaluate.py"

# --- Model & Tokenizer ---
DEVICE = "cuda" # or "cpu"

# --- Training Hyperparameters ---
NUM_TRAIN_EPOCHS = 5

# REQUIRED CHANGE: Batch size must be 1 for a 6GB GPU.
PER_DEVICE_TRAIN_BATCH_SIZE = 1

# REQUIRED CHANGE: Evaluation batch size also must be 1.
PER_DEVICE_EVAL_BATCH_SIZE = 1

WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# --- Data Preprocessing ---
# REQUIRED CHANGE: Sequence length of 2048 is too large for 6GB VRAM.
# 1024 is a much safer starting point.
MAX_INPUT_LENGTH = 1024