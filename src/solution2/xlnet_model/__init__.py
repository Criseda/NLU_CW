"""XLNet-large cross-encoder for authorship verification."""
from .model import load_model
from .predict import predict_probs

__all__ = ["load_model", "predict_probs"]
