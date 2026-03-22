"""
config.py — Hyperparameters and paths for the local Bi-Encoder small model.
"""

import os
import torch

# ── Model ──────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is an extremely fast, 6-layer model optimized for sentence embeddings.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
NUM_LABELS = 1

# ── Tokenisation ───────────────────────────────────────────────────────────────
# Bi-Encoders encode text INDEPENDENTLY. So this is max length PER TEXT, not combined.
MAX_LENGTH = 256  

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS          = 5
BATCH_SIZE      = 16      
GRAD_ACCUM      = 1
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
MAX_GRAD_NORM   = 1.0

# Setup hardware backend for Apple Silicon (MPS) if available
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_STEPS      = 200     
SAVE_STEPS      = 200
METRIC_FOR_BEST = "f1"    

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR    = os.path.join(ROOT_DIR, "data", "training_data", "AV")
TRAIN_FILE  = os.path.join(DATA_DIR, "train.csv")
DEV_FILE    = os.path.join(DATA_DIR, "dev.csv")

MODEL_SAVE_DIR  = os.path.join(ROOT_DIR, "models", "solution2", "small_model")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "outputs", "solution2", "small_model")

# Ensure output directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
