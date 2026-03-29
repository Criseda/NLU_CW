"""
config_continue.py — Configuration for continuing RoBERTa training from checkpoint

Key parameters:
  - MODEL_NAME: roberta-large
  - Load checkpoint and continue fine-tuning
  - Train until early stopping (patience configurable)
"""

import os

# ── Load environment variables (HF_TOKEN) ────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from dotenv import load_dotenv
    env_path = os.path.join(ROOT_DIR, ".env")
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "roberta-large"
NUM_LABELS = 1
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

# ── Tokenisation ───────────────────────────────────────────────────────────────
MAX_LENGTH = 512
TRUNCATION = "longest_first"

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS          = 20      # Continue training, early stopping will stop when appropriate
BATCH_SIZE      = 8
GRAD_ACCUM      = 2
LEARNING_RATE   = 5e-6
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.01
MAX_GRAD_NORM   = 1.0
FP16            = True

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_STEPS      = 500
SAVE_STEPS      = 500
METRIC_FOR_BEST = "f1"

# ── Early Stopping ─────────────────────────────────────────────────────────────
PATIENCE        = 5       # Number of epochs without improvement before stopping

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR    = os.path.join(ROOT_DIR, "data", "training_data", "AV")
TRAIN_FILE  = os.path.join(DATA_DIR, "train.csv")
DEV_FILE    = os.path.join(DATA_DIR, "dev.csv")

MODEL_SAVE_DIR  = os.path.join(ROOT_DIR, "models", "solution2", "roberta_model")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "outputs", "solution2", "roberta_model")

# ── Checkpoint to continue from ────────────────────────────────────────────────
# Path to the best_model.pt from previous training
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
