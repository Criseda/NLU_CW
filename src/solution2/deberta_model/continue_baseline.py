"""
Run one extra fine-tuning epoch starting from the baseline lr=5e-6 checkpoint.

Usage:
    python -m src.solution2.deberta_model.continue_baseline
"""

import os

from . import config
from .train import train


CHECKPOINT_PATH = os.path.join(
    config.ROOT_DIR,
    "models",
    "solution2",
    "deberta_model",
    "baseline_lr_5e6",
    "best_model.pt",
)


def main() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    config.LEARNING_RATE = 5e-6
    config.EPOCHS = 10
    config.WARMUP_RATIO = 0.0
    config.MODEL_SAVE_DIR = os.path.join(
        config.ROOT_DIR,
        "models",
        "solution2",
        "deberta_model",
        "baseline_lr_5e6_continue_no_warmup",
    )
    config.OUTPUT_DIR = os.path.join(
        config.ROOT_DIR,
        "outputs",
        "solution2",
        "deberta_model",
        "baseline_lr_5e6_continue_no_warmup",
    )

    print("[continue] Starting from baseline checkpoint with one extra epoch.")
    print(f"[continue] Checkpoint: {CHECKPOINT_PATH}")
    print(
        f"[continue] LR={config.LEARNING_RATE} | Epochs={config.EPOCHS} | "
        f"Warmup={config.WARMUP_RATIO}"
    )

    train(checkpoint_path=CHECKPOINT_PATH)


if __name__ == "__main__":
    main()