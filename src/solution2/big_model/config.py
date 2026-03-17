"""
config.py — All hyperparameters and paths for the big model (DeBERTa-v3-large cross-encoder).
Edit this file to tune training without touching any other file.
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
MODEL_NAME = "microsoft/deberta-v3-large"   # Best single model on NLU benchmarks
NUM_LABELS = 1                               # Binary → BCEWithLogitsLoss
HF_TOKEN   = os.environ.get("HF_TOKEN", None) # Optional HuggingFace token for rate limits

# ── Features ───────────────────────────────────────────────────────────────────
USE_FEATURES    = True    # Use 97-dim stylometric features
FEATURE_DIM     = 97      # Dimensionality of precomputed features
FEATURES_DIR    = os.path.join(ROOT_DIR, "data", "features")
FEATURES_TRAIN  = os.path.join(FEATURES_DIR, "all_features_train.npy")
FEATURES_DEV    = os.path.join(FEATURES_DIR, "all_features_dev.npy")

# ── Tokenisation ───────────────────────────────────────────────────────────────
MAX_LENGTH = 512          # Fits most AV pairs comfortably; raise to 1024 with truncation="longest_first"
TRUNCATION = "longest_first"  # Shrinks the longer text first when pair exceeds MAX_LENGTH

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS          = 5
BATCH_SIZE      = 16       # Per-device; reduce if OOM
GRAD_ACCUM      = 1       # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
LEARNING_RATE   = 1.5e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1    # 10 % of total steps used for LR warmup
MAX_GRAD_NORM   = 1.0
FP16            = True    # Mixed-precision — set False on CPU

# ── Triplet Loss ───────────────────────────────────────────────────────────────
LOSS_TYPE       = "triplet"  # "triplet" or "bce"
TRIPLET_MARGIN  = 0.5    # Margin for triplet loss: loss = max(0, d(a,p) - d(a,n) + margin)
EMBEDDING_DIM   = 256    # Final embedding dimension for triplet encoder

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_STEPS      = 500     # Evaluate on dev set every N steps
SAVE_STEPS      = 500
METRIC_FOR_BEST = "f1"    # Save checkpoint with best F1

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR    = os.path.join(ROOT_DIR, "data", "training_data", "AV")
TRAIN_FILE  = os.path.join(DATA_DIR, "train.csv")
DEV_FILE    = os.path.join(DATA_DIR, "dev.csv")

MODEL_SAVE_DIR  = os.path.join(ROOT_DIR, "models", "solution2", "big_model")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "outputs", "solution2", "big_model")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42