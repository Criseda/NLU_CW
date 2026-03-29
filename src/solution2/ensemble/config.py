"""
config.py — Configuration for Neural Network meta-learner ensemble.
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Model and output directories ────────────────────────────────────────────
ENSEMBLE_MODEL_DIR = os.path.join(ROOT_DIR, "models", "solution2", "ensemble")
ENSEMBLE_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "solution2", "ensemble")

# ── Neural Network Architecture ─────────────────────────────────────────────
HIDDEN_SIZE = 128         # Hidden layer size (carefully sized to prevent overfitting)
DROPOUT = 0.3             # Dropout rate
LEARNING_RATE = 0.001     # Adam learning rate
WEIGHT_DECAY = 1e-4       # L2 regularization
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda"  # Will auto-detect if GPU is available

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
