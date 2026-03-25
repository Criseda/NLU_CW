import numpy as np
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm


def general_impostor_features(
    idx_a: int,
    idx_b: int,
    corpus_vectors,           # sparse TF-IDF matrix, shape (N_texts, V)
    n_trials: int = 100,
    n_impostors: int = 50,
    seed: int = 42,
) -> list:
    """
    General Impostor Method (Koppel & Winter, 2014).
    Returns 5 features based on how consistently text_b beats random impostors
    when compared to text_a, and vice versa.
    """
    rng = np.random.default_rng(seed)
    n_texts = corpus_vectors.shape[0]
    exclude = {idx_a, idx_b}
    candidate_pool = [i for i in range(n_texts) if i not in exclude]

    if len(candidate_pool) < n_impostors:
        # Not enough impostors in the corpus — return neutral scores
        return [0.5, 0.5, 0.5, 0.5, 0.0]

    vec_a = corpus_vectors[idx_a]   # sparse row
    vec_b = corpus_vectors[idx_b]

    scores_a, scores_b = [], []

    for _ in range(n_trials):
        impostor_idxs = rng.choice(candidate_pool, size=n_impostors, replace=False)
        impostor_vecs = corpus_vectors[impostor_idxs]

        sim_ab = cosine_similarity(vec_a, vec_b)[0, 0]

        # From text_a's perspective: does text_b beat all impostors?
        sim_a_imps = cosine_similarity(vec_a, impostor_vecs)[0]
        scores_a.append(1.0 if sim_ab > sim_a_imps.max() else 0.0)

        # From text_b's perspective: does text_a beat all impostors?
        sim_b_imps = cosine_similarity(vec_b, impostor_vecs)[0]
        scores_b.append(1.0 if sim_ab > sim_b_imps.max() else 0.0)

    gi_a = float(np.mean(scores_a))
    gi_b = float(np.mean(scores_b))

    return [
        gi_a,                              # GI from text_a's perspective
        gi_b,                              # GI from text_b's perspective
        (gi_a + gi_b) / 2,                 # average
        min(gi_a, gi_b),                   # conservative (min) estimate
        float(np.std(scores_a + scores_b)), # confidence (low std = stable signal)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Path to raw CSV")
    parser.add_argument("--output",  required=True, help="Path to output .npy")
    parser.add_argument("--n_jobs",  type=int, default=1)
    parser.add_argument("--n_trials",    type=int, default=100)
    parser.add_argument("--n_impostors", type=int, default=50)
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    # Build corpus: all individual texts (text_1 and text_2 from training set)
    print("Building corpus TF-IDF matrix ...")
    all_texts = list(df["text_1"]) + list(df["text_2"])
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=5000)
    corpus_vectors = tfidf.fit_transform(all_texts)
    # Row i corresponds to all_texts[i]:
    #   rows 0..N-1 → text_1 values
    #   rows N..2N-1 → text_2 values

    # Save vectorizer for inference-time use
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, output_path.parent / "gi_tfidf_vectorizer.joblib")
    joblib.dump(corpus_vectors, output_path.parent / "gi_corpus_vectors.joblib")

    # Pair i corresponds to text_1[i] at index i, text_2[i] at index N+i
    n = len(df)
    print(f"Computing GI features for {n} pairs ({args.n_trials} trials each) ...")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(general_impostor_features)(
            i, n + i, corpus_vectors,
            n_trials=args.n_trials,
            n_impostors=args.n_impostors,
            seed=42 + i,
        )
        for i in tqdm(range(n))
    )

    arr = np.array(results, dtype=np.float32)
    np.save(output_path, arr)
    print(f"Saved GI features {arr.shape} to {output_path}")


if __name__ == "__main__":
    main()
