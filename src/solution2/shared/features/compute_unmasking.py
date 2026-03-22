"""
compute_unmasking.py — Information-theoretic "unmasking" features (8 per pair).

Adapted from Alexios's solution1/compute_unmasking.py.

Features:
  1. Shannon entropy difference:  |H(a) - H(b)|
  2. Cross-entropy a → b:         H(a, b)
  3. Cross-entropy b → a:         H(b, a)
  4. KL divergence KL(a || b)
  5. KL divergence KL(b || a)
  6. Jensen-Shannon Divergence (JSD)
  7. Rényi entropy (order 2) difference: |R2(a) - R2(b)|
  8. Zlib compression ratio difference:  |C(a)/|a| - C(b)/|b||

All distributions are computed over character unigrams (add-1 smoothed, 256 bins).

Usage:
    from src.solution2.shared.features.compute_unmasking import extract_unmasking_features
    feats = extract_unmasking_features(df)   # df has columns: text_1, text_2
    # feats.shape == (len(df), 8)
"""

import zlib
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


def _char_dist(text: str) -> np.ndarray:
    """Character unigram probability distribution (add-1 smoothed, 256 chars)."""
    counts = np.zeros(256, dtype=np.float64)
    for ch in text.encode("utf-8", errors="replace"):
        counts[ch] += 1
    counts += 1          # Laplace smoothing
    counts /= counts.sum()
    return counts


def _shannon_entropy(p: np.ndarray) -> float:
    """H(p) in nats."""
    return float(-np.sum(p * np.log(p + 1e-12)))


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(p, q) = -sum p * log(q)."""
    return float(-np.sum(p * np.log(q + 1e-12)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = sum p * log(p/q)."""
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence."""
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _renyi_entropy_2(p: np.ndarray) -> float:
    """Rényi entropy of order 2: -log(sum p²)."""
    return float(-math.log(max(np.sum(p ** 2), 1e-12)))


def _zlib_ratio(text: str) -> float:
    """Compression ratio: compressed_size / original_size."""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw)) / len(raw)


def _features_single(a: str, b: str) -> list:
    pa = _char_dist(a)
    pb = _char_dist(b)
    ha = _shannon_entropy(pa)
    hb = _shannon_entropy(pb)
    return [
        abs(ha - hb),                                       # 1  Shannon entropy diff
        _cross_entropy(pa, pb),                             # 2  Cross-entropy a→b
        _cross_entropy(pb, pa),                             # 3  Cross-entropy b→a
        _kl(pa, pb),                                        # 4  KL(a || b)
        _kl(pb, pa),                                        # 5  KL(b || a)
        _jsd(pa, pb),                                       # 6  JSD
        abs(_renyi_entropy_2(pa) - _renyi_entropy_2(pb)),   # 7  Rényi-2 diff
        abs(_zlib_ratio(a) - _zlib_ratio(b)),               # 8  Compression ratio diff
    ]


def extract_unmasking_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract 8 information-theoretic features for each row in df.
    df must have columns: text_1, text_2.
    Returns np.ndarray of shape (len(df), 8).
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[features] Unmasking", leave=False):
        a = str(row["text_1"]) if pd.notna(row["text_1"]) else ""
        b = str(row["text_2"]) if pd.notna(row["text_2"]) else ""
        rows.append(_features_single(a, b))

    arr = np.array(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=50.0, neginf=-50.0)
    return arr
