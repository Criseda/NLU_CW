"""
config.py — Configuration for vanilla XLNet cross-encoder

Key parameters:
  - MODEL_NAME: xlnet-large (340M params, permutation LM)
  - No stylistic features — pure semantic learning
  - Standard dropout regularization (0.1)
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
MODEL_NAME = "xlnet-large-cased"             # 340M params, permutation LM (no stylistic features)
NUM_LABELS = 1                               # Binary → BCEWithLogitsLoss
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

# ── Tokenisation ───────────────────────────────────────────────────────────────
MAX_LENGTH = 512          # Standard for AV task
TRUNCATION = "longest_first"

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS          = 20      # Early stopping (patience=3) will prevent overfitting
BATCH_SIZE      = 8       # XLNet-large is ~340M params
GRAD_ACCUM      = 2       # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
LEARNING_RATE   = 1e-5    # Standard for task-specific fine-tuning
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1     # 10% warmup
MAX_GRAD_NORM   = 1.0
FP16            = True    # Mixed-precision for speed

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_STEPS      = 500     # Evaluate on dev set every N steps
SAVE_STEPS      = 500
METRIC_FOR_BEST = "f1"    # Save checkpoint with best macro F1

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR    = os.path.join(ROOT_DIR, "data", "training_data", "AV")
TRAIN_FILE  = os.path.join(DATA_DIR, "train.csv")
DEV_FILE    = os.path.join(DATA_DIR, "dev.csv")

MODEL_SAVE_DIR  = os.path.join(ROOT_DIR, "models", "solution2", "xlnet_model")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "outputs", "solution2", "xlnet_model")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

