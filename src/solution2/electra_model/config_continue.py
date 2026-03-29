"""
config_continue.py — Configuration for continuing ELECTRA-large training from checkpoint

Key parameters:
  - MODEL_NAME: google/electra-large-discriminator
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

# ── Model & Data Paths ─────────────────────────────────────────────────────
MODEL_NAME = "google/electra-large-discriminator"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DATA_DIR = os.path.join(ROOT_DIR, "data", "training_data", "AV")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
DEV_FILE = os.path.join(DATA_DIR, "dev.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models", "solution2", "electra_model")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "solution2", "electra_model")

# ── Model Hyperparameters ──────────────────────────────────────────────────
MAX_LENGTH = 512
TRUNCATION = True

EPOCHS = 20                       # Continue training, early stopping will stop when appropriate
BATCH_SIZE = 8                    # Per GPU
GRAD_ACCUM_STEPS = 2             # Gradient accumulation
LEARNING_RATE = 5e-6           # Slightly elevated for discriminator fine-tuning
WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.01

# ── Early Stopping ─────────────────────────────────────────────────────────
PATIENCE = 5                      # Number of epochs without improvement before stopping

# ── Checkpoint to continue from ────────────────────────────────────────────────
# Path to the best_model.pt from previous training
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

# ── Training & Inference ───────────────────────────────────────────────────
SEED = 42
