"""
feature_loader.py — Load and normalize pre-computed stylometric features.

Features shape: (num_samples, 97)
Includes: character n-grams, compression, function words, vocabulary, syntax,
          surface features, readability, unmasking, and impostor features.
"""

import numpy as np
from pathlib import Path


def load_and_normalize_features(npy_path: str) -> np.ndarray:
    """
    Load features from .npy file and normalize using z-score normalization.
    
    Args:
        npy_path: Path to .npy file containing features of shape (num_samples, 97)
    
    Returns:
        Normalized features array of shape (num_samples, 97) as float32
    """
    if not Path(npy_path).exists():
        raise FileNotFoundError(f"Feature file not found: {npy_path}")
    
    features = np.load(npy_path).astype(np.float32)
    
    # Z-score normalization per feature: (x - mean) / (std + eps)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features_normalized = (features - mean) / (std + 1e-8)
    
    return features_normalized.astype(np.float32)


def load_features_optional(npy_path: str) -> np.ndarray | None:
    """
    Load features if file exists, return None otherwise.
    
    Args:
        npy_path: Path to .npy file
    
    Returns:
        Normalized features array or None if file doesn't exist
    """
    path = Path(npy_path)
    if path.exists():
        return load_and_normalize_features(npy_path)
    return None
