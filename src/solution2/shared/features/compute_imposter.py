"""
compute_imposter.py — General Impostor (GI) features (5 per pair).

Adapted from Alexios's solution1/compute_imposter.py.

The General Impostor method measures how consistently a pair outscores random
"impostor" texts from the corpus. A higher score = more likely same author.

For each pair (a, b) we draw N random texts from the corpus and compute:
  - GI_a: fraction of impostors where sim(a, b) > sim(a, impostor)
  - GI_b: fraction of impostors where sim(a, b) > sim(b, impostor)

Features (5):
  1. GI score from a's perspective
  2. GI score from b's perspective
  3. Average GI score
  4. Minimum (conservative) GI score
  5. Absolute difference / 2 (proxy for std dev)

Similarity: cosine on character-level TF-IDF (char_wb 2–4-grams, 2000 features).

Usage:
    from src.solution2.shared.features.compute_imposter import extract_impostor_features
    feats = extract_impostor_features(df)   # df has columns: text_1, text_2
    # feats.shape == (len(df), 5)
"""

import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Default number of impostors to sample per pair
N_IMPOSTORS = 25


def _build_tfidf(corpus: list, max_features: int = 2000) -> TfidfVectorizer:
    """Fit a character-level TF-IDF vectorizer on the full corpus."""
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=max_features,
        sublinear_tf=True,
    )
    vec.fit(corpus)
    return vec


def extract_impostor_features(
    df: pd.DataFrame,
    corpus_texts: list = None,
    n_impostors: int = N_IMPOSTORS,
    tfidf_max_features: int = 2000,
) -> np.ndarray:
    """
    Extract 5 General Impostor features for each row in df.

    Parameters
    ----------
    df              : DataFrame with columns text_1, text_2
    corpus_texts    : Pool of impostor texts. Defaults to all text_1 + text_2 in df.
    n_impostors     : Number of random impostors per comparison.
    tfidf_max_features : Vocabulary size for char-level TF-IDF.

    Returns np.ndarray of shape (len(df), 5).
    """
    texts_1 = df["text_1"].fillna("").tolist()
    texts_2 = df["text_2"].fillna("").tolist()

    if corpus_texts is None:
        corpus_texts = texts_1 + texts_2

    print("  [features] GI: Fitting TF-IDF on corpus …")
    vec = _build_tfidf(corpus_texts, max_features=tfidf_max_features)

    print("  [features] GI: Vectorising impostor pool …")
    impostor_matrix = vec.transform(corpus_texts)   # sparse (2N, vocab)
    n_corpus = impostor_matrix.shape[0]

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[features] Impostor", leave=False):
        a = str(row["text_1"]) if pd.notna(row["text_1"]) else ""
        b = str(row["text_2"]) if pd.notna(row["text_2"]) else ""

        a_vec = vec.transform([a])
        b_vec = vec.transform([b])

        sampled = random.sample(range(n_corpus), min(n_impostors, n_corpus))
        imp_block = impostor_matrix[sampled]

        sim_ab  = float(cosine_similarity(a_vec, b_vec)[0, 0])
        sims_a  = cosine_similarity(a_vec, imp_block).flatten()
        sims_b  = cosine_similarity(b_vec, imp_block).flatten()

        gi_a = float(np.mean(sim_ab > sims_a))
        gi_b = float(np.mean(sim_ab > sims_b))

        rows.append([
            gi_a,                   # 1  GI from a's perspective
            gi_b,                   # 2  GI from b's perspective
            (gi_a + gi_b) / 2,      # 3  Average GI
            min(gi_a, gi_b),        # 4  Minimum (conservative)
            abs(gi_a - gi_b) / 2,   # 5  Std dev proxy
        ])

    arr = np.array(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr
