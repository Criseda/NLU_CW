import numpy as np
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from tqdm import tqdm


def unmasking_features(
    text_a: str,
    text_b: str,
    n_rounds: int = 10,
    k_remove: int = 6,
    chunk_size: int = 500,
) -> list:
    """
    Compute unmasking degradation curve for a single pair.
    Returns 8 features derived from the accuracy-vs-round curve.
    Designed to be run in parallel across pairs on HPC.
    """
    # Chunk both texts
    words_a = text_a.split()
    words_b = text_b.split()
    chunks_a = [' '.join(words_a[i:i+chunk_size])
                for i in range(0, len(words_a), chunk_size) if len(words_a[i:i+chunk_size]) > 50]
    chunks_b = [' '.join(words_b[i:i+chunk_size])
                for i in range(0, len(words_b), chunk_size) if len(words_b[i:i+chunk_size]) > 50]

    # Need at least 3 chunks per side for meaningful CV
    if len(chunks_a) < 3 or len(chunks_b) < 3:
        return [np.nan] * 8  # imputed with training median later

    labels = [0] * len(chunks_a) + [1] * len(chunks_b)
    all_chunks = chunks_a + chunks_b

    # Vectorise: char 3-grams, top 1000 features
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 3), max_features=1000)
    X = vec.fit_transform(all_chunks).toarray().astype(float)
    y = np.array(labels)

    n_folds = min(5, min(len(chunks_a), len(chunks_b)))
    accuracies = []

    # Degradation loop
    for round_i in range(n_rounds):
        if X.shape[1] < k_remove:
            # No features left — pad with last accuracy
            accuracies.append(accuracies[-1] if accuracies else 0.5)
            continue

        clf = LinearSVC(max_iter=5000, dual='auto')
        try:
            scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
            acc = float(scores.mean())
        except Exception:
            acc = 0.5
        accuracies.append(acc)

        # Remove top-k most discriminative features (highest |weight|)
        clf.fit(X, y)
        importances = np.abs(clf.coef_[0])
        top_k_idx = np.argsort(importances)[-k_remove:]
        mask = np.ones(X.shape[1], dtype=bool)
        mask[top_k_idx] = False
        X = X[:, mask]

    # Extract curve features
    accs = np.array(accuracies)
    drops = np.diff(accs)
    slope = float(np.polyfit(range(len(accs)), accs, 1)[0]) if len(accs) > 1 else 0.0

    return [
        float(accs[0]),                                          # initial accuracy
        float(accs[-1]),                                         # final accuracy
        slope,                                                   # degradation slope (KEY signal)
        float(np.min(drops)) if len(drops) > 0 else 0.0,        # max single-round drop
        float(np.trapz(accs)),                                   # AUC under curve
        float(accs[0] - accs[-1]),                               # total drop
        float(next((i for i, a in enumerate(accs) if a < 0.6), len(accs))),  # collapse round
        float(np.std(drops)) if len(drops) > 0 else 0.0,        # drop smoothness
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Path to raw CSV (with text_1, text_2 columns)")
    parser.add_argument("--output",  required=True, help="Path to output .npy array")
    parser.add_argument("--n_jobs",  type=int, default=1)
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    print(f"Computing unmasking features for {len(df)} pairs with {args.n_jobs} workers ...")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(unmasking_features)(row["text_1"], row["text_2"])
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    arr = np.array(results, dtype=np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"Saved unmasking features {arr.shape} to {output_path}")
    print(f"NaN rate: {np.isnan(arr).mean():.1%}  (pairs too short to unmask)")


if __name__ == "__main__":
    main()
