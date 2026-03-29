"""
src/solution2/deberta_model — DeBERTa-v3-large cross-encoder for AV (Category C, Solution 2).

Quick start:
    # Train:
    python -m src.solution2.deberta_model.train

    # Predict on dev set:
    python -m src.solution2.deberta_model.predict

    # Predict on a custom CSV:
    python -m src.solution2.deberta_model.predict --input data/... --split test
"""

from .model import AVCrossEncoder, load_model
from .predict import predict_probs

__all__ = ["AVCrossEncoder", "load_model", "predict_probs"]
