import numpy as np
import argparse
import zlib
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def _char_distribution(text: str) -> np.ndarray:
    """
    Compute a normalised character unigram probability distribution
    over all 256 possible byte values (covers ASCII + Unicode chars).
    """
    counts = np.zeros(256, dtype=np.float64)
    for byte in text.encode('utf-8', errors='replace'):
        counts[byte] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(256, dtype=np.float64) / 256  # uniform fallback
    return counts / total


def _shannon_entropy(p: np.ndarray) -> float:
    """H(p) = -sum(p * log2(p)) over non-zero elements."""
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """H(p, q) = -sum(p * log2(q)) — how well q predicts p."""
    return float(-np.sum(p * np.log2(q + eps)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) = sum(p * log(p/q))."""
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (q[nz] + eps))))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — symmetric, bounded in [0, 1]."""
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))


def _renyi_entropy_2(p: np.ndarray, eps: float = 1e-12) -> float:
    """Rényi entropy of order 2: R2 = -log2(sum(p^2))."""
    return float(-np.log2(np.sum(p ** 2) + eps))


def _compression_ratio(text: str) -> float:
    """Compressed size / original size — lower = more compressible = more repetitive."""
    encoded = text.encode('utf-8')
    if len(encoded) == 0:
        return 1.0
    return len(zlib.compress(encoded, level=9)) / len(encoded)


def information_theoretic_features(text_a: str, text_b: str) -> list:
    """
    Compute 8 information-theoretic features for a pair of short texts.

    Replaces Unmasking (Koppel & Schler, 2004), which requires long texts
    (>1500 words for chunking). The AV dataset has avg ~100 words per text,
    making Unmasking inapplicable. These features operate on character
    distributions and work on texts of any length.

    Features:
      1. |H(a) - H(b)|       — Shannon entropy difference
      2. H(p_b, p_a)         — cross-entropy (a's distrib. predicts b's chars)
      3. H(p_a, p_b)         — cross-entropy (b's distrib. predicts a's chars)
      4. KL(p_a || p_b)      — KL divergence
      5. KL(p_b || p_a)      — KL divergence (reverse direction)
      6. JSD(p_a, p_b)       — Jensen-Shannon divergence (symmetric)
      7. |R2(a) - R2(b)|     — Rényi entropy (order 2) difference
      8. |C(a) - C(b)|       — zlib compression ratio difference

    References: Halvani et al. (2016), Cilibrasi & Vitányi (2005)
    """
    if not text_a or not text_b:
        return [0.0] * 8

    p_a = _char_distribution(text_a)
    p_b = _char_distribution(text_b)

    h_a = _shannon_entropy(p_a)
    h_b = _shannon_entropy(p_b)

    return [
        abs(h_a - h_b),                                                   # 1. entropy diff
        _cross_entropy(p_b, p_a),                                         # 2. cross-entropy a→b
        _cross_entropy(p_a, p_b),                                         # 3. cross-entropy b→a
        _kl_divergence(p_a, p_b),                                         # 4. KL(a||b)
        _kl_divergence(p_b, p_a),                                         # 5. KL(b||a)
        _jsd(p_a, p_b),                                                   # 6. JSD
        abs(_renyi_entropy_2(p_a) - _renyi_entropy_2(p_b)),               # 7. Rényi diff
        abs(_compression_ratio(text_a) - _compression_ratio(text_b)),     # 8. compression ratio diff
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compute 8 information-theoretic features per AV pair (short-text replacement for Unmasking)."
    )
    parser.add_argument("--input",  required=True, help="Path to raw CSV (text_1, text_2 columns)")
    parser.add_argument("--output", required=True, help="Path to output .npy array (N, 8)")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    print(f"Computing information-theoretic features for {len(df)} pairs with {args.n_jobs} workers ...")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(information_theoretic_features)(row["text_1"], row["text_2"])
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    arr = np.array(results, dtype=np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)

    print(f"Saved info-theoretic features {arr.shape} to {output_path}")
    print(f"NaN rate: {np.isnan(arr).mean():.1%}")
    print(f"Feature means: {np.nanmean(arr, axis=0).round(4)}")


if __name__ == "__main__":
    main()
