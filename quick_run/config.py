# config.py
# Central configuration file for the FinQA project

# --- Paths and File Names ---
BASE_PROJECT_DIR = r"E:\1.apps\obsidian_folder\Research\Research_code\FinQA"
DATA_DIR = "./dataset"
TRAIN_FILE = f"{DATA_DIR}/train.json"
DEV_FILE = f"{DATA_DIR}/dev.json"
TEST_FILE = f"{DATA_DIR}/test.json"

# >>> SUGGESTION: Use a separate path for this test run to avoid overwriting your work.
MODEL_SAVE_PATH = "./fingpt_mistral_lora_test_run"

PREDICTIONS_FILE = "predictions_final.json"
EVALUATION_SCRIPT_PATH = "./code/evaluate/evaluate.py"

# --- Model & Tokenizer ---
DEVICE = "cuda" # or "cpu"

# --- Training Hyperparameters (MINIMAL SETTINGS FOR BUG CHECKING) ---

# >>> CHANGE: Only one epoch is needed for a quick test run.
NUM_TRAIN_EPOCHS = 1

PER_DEVICE_TRAIN_BATCH_SIZE = 1 # Keep at 1 for 6GB GPU
PER_DEVICE_EVAL_BATCH_SIZE = 1  # Keep at 1 for 6GB GPU

# >>> CHANGE: Reduce warmup steps for a very short training run.
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01

# >>> CHANGE: Set logging/eval/save steps to be very low.
# This ensures that these events actually trigger during our short test run.
LOGGING_STEPS = 4
EVAL_STEPS = 8
SAVE_STEPS = 8

SAVE_TOTAL_LIMIT = 2

# --- Data Preprocessing ---
MAX_INPUT_LENGTH = 1024 # Keep as is, or reduce to 512 if you face memory issues