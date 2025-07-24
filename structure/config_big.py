# config.py
# Central configuration file for the FinQA project

# --- Paths and File Names ---
BASE_PROJECT_DIR = r"E:\1.apps\obsidian_folder\Research\Research_code\FinQA"  #<-- ADJUST THIS
# DATA_DIR = "../dataset"
# TRAIN_FILE = f"{DATA_DIR}/train.json"
# DEV_FILE = f"{DATA_DIR}/dev.json"
# TEST_FILE = f"{DATA_DIR}/test.json"
# MODEL_SAVE_PATH = "../finqa_t5_final_model"
# PREDICTIONS_FILE = "predictions_final.json"
# EVALUATION_SCRIPT_PATH = "../code/evaluate/evaluate.py"

DATA_DIR = "./dataset"
TRAIN_FILE = f"{DATA_DIR}/train.json"
DEV_FILE = f"{DATA_DIR}/dev.json"
TEST_FILE = f"{DATA_DIR}/test.json"
MODEL_SAVE_PATH = "./finqa_t5_final_model"
PREDICTIONS_FILE = "predictions_final.json"
EVALUATION_SCRIPT_PATH = "./code/evaluate/evaluate.py"

# --- Model & Tokenizer ---
DEVICE = "cuda" # or "cpu"

# --- Training Hyperparameters ---
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# --- Data Preprocessing ---
MAX_INPUT_LENGTH = 2048
MAX_TARGET_LENGTH = 128