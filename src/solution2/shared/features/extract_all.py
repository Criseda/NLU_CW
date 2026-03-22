"""
extract_all.py — Orchestrator that computes all 97 features for a CSV and saves a .npy file.

Feature layout (97 total):
  [  0: 84]  Stylometric features        (compute_stylometric.py)
  [ 84: 92]  Info-theoretic / Unmasking  (compute_unmasking.py)  — Alexios's code
  [ 92: 97]  General Impostor (GI)       (compute_imposter.py)   — Alexios's code

Usage:
    # Recompute dev features (saved to all_features_dev.npy in project root):
    .venv/bin/python -m src.solution2.shared.features.extract_all --split dev

    # Recompute train features:
    .venv/bin/python -m src.solution2.shared.features.extract_all --split train

    # Custom CSV → custom output:
    .venv/bin/python -m src.solution2.shared.features.extract_all \\
        --csv path/to/data.csv --out path/to/features.npy
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.solution2.bi_encoder import config
from .compute_stylometric import extract_stylometric_features
from .compute_unmasking import extract_unmasking_features
from .compute_imposter import extract_impostor_features

ROOT_DIR   = config.ROOT_DIR
TRAIN_CSV  = config.TRAIN_FILE
DEV_CSV    = config.DEV_FILE
TRAIN_NPY  = os.path.join(ROOT_DIR, "all_features_train.npy")
DEV_NPY    = os.path.join(ROOT_DIR, "all_features_dev.npy")


def extract_features(df: pd.DataFrame, desc: str = "") -> np.ndarray:
    """
    Compute all 97 features for every row in df.
    df must have columns: text_1, text_2.

    Returns np.ndarray of shape (len(df), 97).
      [  0: 84] — stylometric
      [ 84: 92] — info-theoretic (unmasking)
      [ 92: 97] — General Impostor
    """
    print(f"\n  Computing stylometric features (84) …")
    t0 = time.time()
    stylo = extract_stylometric_features(df)
    print(f"  Done. Shape: {stylo.shape} | Time: {time.time()-t0:.1f}s")

    print(f"\n  Computing unmasking features (8) …")
    t0 = time.time()
    unmask = extract_unmasking_features(df)
    print(f"  Done. Shape: {unmask.shape} | Time: {time.time()-t0:.1f}s")

    print(f"\n  Computing General Impostor features (5) …")
    t0 = time.time()
    gi = extract_impostor_features(df)
    print(f"  Done. Shape: {gi.shape} | Time: {time.time()-t0:.1f}s")

    all_feats = np.concatenate([stylo, unmask, gi], axis=1)  # (N, 97)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e6, neginf=-1e6)

    assert all_feats.shape[1] == 97, f"Expected 97 features, got {all_feats.shape[1]}"
    return all_feats


def main():
    parser = argparse.ArgumentParser(description="Extract 97 features from an AV CSV.")
    parser.add_argument("--split", choices=["train", "dev"], default="dev",
                        help="Which built-in split to process (default: dev)")
    parser.add_argument("--csv", default=None,
                        help="Custom CSV path (overrides --split)")
    parser.add_argument("--out", default=None,
                        help="Output .npy path (overrides default)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
        out_path = args.out or csv_path.replace(".csv", "_features.npy")
    elif args.split == "train":
        csv_path = TRAIN_CSV
        out_path = args.out or TRAIN_NPY
    else:
        csv_path = DEV_CSV
        out_path = args.out or DEV_NPY

    print("=" * 60)
    print(f"  extract_all — 97 features")
    print(f"  Input : {csv_path}")
    print(f"  Output: {out_path}")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}")

    t_start = time.time()
    feats = extract_features(df, desc=args.split)

    np.save(out_path, feats)
    print(f"\n  Saved → {out_path}")
    print(f"  Shape: {feats.shape} | dtype: {feats.dtype}")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
