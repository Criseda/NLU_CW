"""
predict.py — Solution A Inference Script
=========================================
Provides two public functions for use by the demo notebook:

    extract_features(test_csv, features_dir, n_jobs) -> np.ndarray (N, 97)
    predict_from_csv(test_csv, model_dir, features_dir, n_jobs) -> pd.DataFrame

The stacking inference function (predict_with_stack) is imported directly
from evaluate.py where it lives alongside the training evaluation code.

CLI usage:
    python src/solution1/predict.py \\
        --test_csv  data/test_data/AV/test.csv \\
        --output    outputs/Group_17_A.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Reuse existing pipeline modules — no logic is duplicated here
from src.solution1.training.preprocess import process_pair
from src.solution1.training.feature_extraction import extract_all_features
from src.solution1.training.compute_unmasking import information_theoretic_features
from src.solution1.training.evaluate import predict_with_stack


# ---------------------------------------------------------------------------
# GI inference — defined here because training's compute_imposter.py builds
# a fresh corpus from the input data, which is wrong at inference time.
# At inference we must use the saved training corpus so the feature values
# are in the same space the model was trained on.
# ---------------------------------------------------------------------------

def _gi_inference(vec_a, vec_b, corpus_vectors, n_trials=100, n_impostors=50, seed=42):
    """GI features for a single pair using the saved training corpus as impostors."""
    rng = np.random.default_rng(seed)
    n_texts = corpus_vectors.shape[0]

    if n_texts < n_impostors:
        return [0.5, 0.5, 0.5, 0.5, 0.0]

    sim_ab = cosine_similarity(vec_a, vec_b)[0, 0]
    scores_a, scores_b = [], []

    for _ in range(n_trials):
        idx = rng.choice(n_texts, size=n_impostors, replace=False)
        impostors = corpus_vectors[idx]
        scores_a.append(1.0 if sim_ab > cosine_similarity(vec_a, impostors)[0].max() else 0.0)
        scores_b.append(1.0 if sim_ab > cosine_similarity(vec_b, impostors)[0].max() else 0.0)

    gi_a = float(np.mean(scores_a))
    gi_b = float(np.mean(scores_b))
    return [gi_a, gi_b, (gi_a + gi_b) / 2, min(gi_a, gi_b), float(np.std(scores_a + scores_b))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    test_csv: str | Path,
    features_dir: str | Path,
    n_jobs: int = 4,
) -> np.ndarray:
    """
    Run the full feature extraction pipeline on a raw test CSV.

    Reuses preprocess.py, feature_extraction.py, and compute_unmasking.py
    from the training pipeline. GI features use the saved training corpus.

    Returns:
        X: np.ndarray of shape (N, 97)
    """
    features_dir = Path(features_dir)

    # 1. Load and preprocess with spaCy
    print("Preprocessing with spaCy ...")
    df = pd.read_csv(test_csv)
    rows = df.to_dict(orient="records")
    processed = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_pair)(row) for row in tqdm(rows, desc="spaCy")
    )

    # 2. Stylometric features (84) — reuses extract_all_features from feature_extraction.py
    print("Extracting stylometric features (84) ...")
    X_style = np.vstack([
        extract_all_features(p) for p in tqdm(processed, desc="stylometric")
    ])

    # 3. Info-theoretic features (8) — reuses information_theoretic_features from compute_unmasking.py
    print("Extracting info-theoretic features (8) ...")
    X_info = np.vstack([
        information_theoretic_features(p["text_1"]["raw"], p["text_2"]["raw"])
        for p in tqdm(processed, desc="info-theoretic")
    ])

    # 4. GI features (5) — uses saved training corpus (see _gi_inference docstring)
    print("Loading GI corpus ...")
    tfidf_vec = joblib.load(features_dir / "gi_tfidf_vectorizer.joblib")
    corpus_vecs = joblib.load(features_dir / "gi_corpus_vectors.joblib")
    print(f"  corpus shape: {corpus_vecs.shape}")

    test_vecs_1 = tfidf_vec.transform([p["text_1"]["raw"] for p in processed])
    test_vecs_2 = tfidf_vec.transform([p["text_2"]["raw"] for p in processed])

    print("Computing GI features (5) ...")
    X_gi = np.vstack([
        _gi_inference(test_vecs_1[i], test_vecs_2[i], corpus_vecs, seed=42 + i)
        for i in tqdm(range(len(df)), desc="GI")
    ])

    X = np.concatenate([X_style, X_info, X_gi], axis=1).astype(np.float32)
    print(f"Feature matrix: {X.shape}")
    return X


def predict_from_csv(
    test_csv: str | Path,
    model_dir: str | Path,
    features_dir: str | Path,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    End-to-end: raw test CSV -> submission-ready DataFrame.

    Notebook usage:
        from src.solution1.predict import predict_from_csv
        df = predict_from_csv("data/test_data/AV/test.csv",
                              "src/solution1/models/full",
                              "src/solution1/features")
        df.to_csv("outputs/Group_17_A.csv", index=False)
    """
    X = extract_features(test_csv, features_dir, n_jobs)
    y_pred, _, _, _ = predict_with_stack(X, Path(model_dir))
    return pd.DataFrame({"prediction": y_pred.astype(int)})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Solution A — generate test predictions")
    parser.add_argument("--test_csv",     default="data/test_data/AV/test.csv")
    parser.add_argument("--output",       default="outputs/Group_17_A.csv")
    parser.add_argument("--model_dir",    default="src/solution1/models/full")
    parser.add_argument("--features_dir", default="src/solution1/features")
    parser.add_argument("--n_jobs",       type=int, default=4)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = predict_from_csv(args.test_csv, args.model_dir, args.features_dir, args.n_jobs)
    df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} predictions to {out}")
    print(f"Class distribution: {df['prediction'].mean():.2%} positive")


if __name__ == "__main__":
    main()
