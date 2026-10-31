# ============================================================
# Stack Overflow scraping
# ============================================================
API_KEY = "rl_esZBRi5i8oYYf3LNUVN9nvjpB"
SITE = "stackoverflow"
TAG = "python"
MIN_SCORE = 5
PAGESIZE = 100
TARGET_COUNT = 4000
SLEEP_BETWEEN_REQUESTS = 3

# ============================================================
# Data cleaning
# ============================================================
MIN_ANSWER_CHARS = 100
MIN_QUESTION_CHARS = 20
INCLUDE_QUESTION_BODY_THRESHOLD = 60
VAL_SPLIT_RATIO = 0.20
SEED = 42

# ============================================================
# Model
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512

MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
USE_GRADIENT_CHECKPOINTING = False   # not needed at 1.5B + 4-bit

# ============================================================
# LoRA
# ============================================================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50
SEED = 42

# ============================================================
# Paths
# ============================================================
RAW_DATA_PATH = "data/raw/raw_qa.json"
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
PLOTS_DIR = "outputs/plots"
TRAIN_LOG_PATH = "outputs/plots/train_log.json"