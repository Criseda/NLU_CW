"""
compute_stylometric.py — Stylometric features (84 per pair).

Covers the following feature blocks:
  Block 1 — Character N-gram similarity  (16 features, n=2,3,4,5)
  Block 2 — Compression distance         (6 features)
  Block 3 — Function word frequencies    (13 features)
  Block 4 — Vocabulary richness          (15 features)
  Block 5 — Syntactic proxies            (9 features, regex-based)
  Block 6 — Surface statistics           (20 features)
  Block 7 — Readability scores           (5 features)

No heavy NLP dependencies (no spacy, no nltk). Uses stdlib + scipy + sklearn.

Usage:
    from src.solution2.shared.features.compute_stylometric import extract_stylometric_features
    feats = extract_stylometric_features(df)   # df has columns: text_1, text_2
    # feats.shape == (len(df), 84)
"""

import re
import zlib
import bz2
import lzma
import collections
import math

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


# ── Function words (EN) ─────────────────────────────────────────────────────────
FUNCTION_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "which",
    "me", "when", "make", "can", "like", "time", "no", "just", "him",
    "know", "take", "people", "into", "year", "your", "good", "some",
    "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after",
    "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day",
    "most", "us",
][:100]   # keep top-100 for feature computation

TOP_FW = 10  # number of individual function-word diff features


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def _char_ngram_freq(text: str, n: int) -> dict:
    """Character n-gram frequency dictionary."""
    freq = collections.Counter()
    for i in range(len(text) - n + 1):
        freq[text[i:i + n]] += 1
    total = sum(freq.values()) or 1
    return {k: v / total for k, v in freq.items()}


def _dict_to_vec(d1: dict, d2: dict) -> tuple:
    """Align two frequency dicts to the same key set → numpy arrays."""
    keys = set(d1) | set(d2)
    v1 = np.array([d1.get(k, 0.0) for k in keys], dtype=np.float64)
    v2 = np.array([d2.get(k, 0.0) for k in keys], dtype=np.float64)
    return v1, v2


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    if not np.any(v1) or not np.any(v2):
        return 0.0
    return float(1 - cosine_dist(v1, v2))


def _jsd_from_vecs(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _burrows_delta(v1: np.ndarray, v2: np.ndarray) -> float:
    """Simplified Burrows' Delta: Manhattan distance on z-scored vectors."""
    combined = np.vstack([v1, v2])
    mu = combined.mean(axis=0)
    sigma = combined.std(axis=0) + 1e-12
    z1 = (v1 - mu) / sigma
    z2 = (v2 - mu) / sigma
    return float(np.mean(np.abs(z1 - z2)))


def _pearson(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.std() < 1e-9 or v2.std() < 1e-9:
        return 0.0
    r, _ = pearsonr(v1, v2)
    return float(r) if not math.isnan(r) else 0.0


def _ncd(a: str, b: str, compressor) -> float:
    ra = len(compressor(a.encode("utf-8")))
    rb = len(compressor(b.encode("utf-8")))
    rab = len(compressor((a + b).encode("utf-8")))
    rba = len(compressor((b + a).encode("utf-8")))
    denom = max(ra, rb)
    if denom == 0:
        return 0.0
    return (min(rab, rba) - min(ra, rb)) / denom


def _syllable_count(word: str) -> int:
    """Simple syllable counter."""
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


# ── Block 1: Character N-gram similarity (16 features) ─────────────────────────

def _char_ngram_features(a: str, b: str) -> list:
    feats = []
    for n in [2, 3, 4, 5]:
        fa = _char_ngram_freq(a, n)
        fb = _char_ngram_freq(b, n)
        v1, v2 = _dict_to_vec(fa, fb)
        feats.append(_cosine_sim(v1, v2))
        feats.append(_jsd_from_vecs(v1, v2))
        feats.append(_burrows_delta(v1, v2))
        feats.append(_pearson(v1, v2))
    return feats  # 16


# ── Block 2: Compression distance (6 features) ─────────────────────────────────

def _compression_features(a: str, b: str) -> list:
    zlib_comp  = lambda x: zlib.compress(x)
    bz2_comp   = lambda x: bz2.compress(x)
    lzma_comp  = lambda x: lzma.compress(x)
    return [
        _ncd(a, b, zlib_comp),
        _ncd(b, a, zlib_comp),
        _ncd(a, b, bz2_comp),
        _ncd(b, a, bz2_comp),
        _ncd(a, b, lzma_comp),
        _ncd(b, a, lzma_comp),
    ]  # 6


# ── Block 3: Function word frequencies (13 features) ───────────────────────────

def _fw_vec(text: str) -> np.ndarray:
    words = _tokenize(text)
    total = len(words) or 1
    return np.array([words.count(w) / total for w in FUNCTION_WORDS], dtype=np.float64)


def _function_word_features(a: str, b: str) -> list:
    va = _fw_vec(a)
    vb = _fw_vec(b)
    diffs = np.abs(va - vb)
    top10 = sorted(diffs, reverse=True)[:TOP_FW]
    return [
        _cosine_sim(va, vb),
        float(np.sum(diffs)),                # Manhattan
        *top10,                              # 10 individual top diffs
        float(np.mean(diffs)),               # mean diff
    ]  # 13


# ── Block 4: Vocabulary richness (15 features) ─────────────────────────────────

def _vocab_features(a: str, b: str) -> list:
    wa = _tokenize(a)
    wb = _tokenize(b)
    sa = set(wa)
    sb = set(wb)

    def ttr(words):  return len(set(words)) / (len(words) or 1)
    def hapax(words):
        freq = collections.Counter(words)
        return sum(1 for v in freq.values() if v == 1) / (len(words) or 1)
    def yule_k(words):
        freq = collections.Counter(words)
        n = len(words) or 1
        m2 = sum(v * (v - 1) for v in freq.values())
        return 10_000 * m2 / (n * (n - 1) + 1e-9)
    def simpsons_d(words):
        freq = collections.Counter(words)
        n = len(words) or 1
        return sum(v * (v - 1) for v in freq.values()) / (n * (n - 1) + 1e-9)

    jaccard = len(sa & sb) / (len(sa | sb) + 1e-9)
    overlap_a = len(sa & sb) / (len(sa) + 1e-9)
    overlap_b = len(sa & sb) / (len(sb) + 1e-9)
    dice = 2 * len(sa & sb) / (len(sa) + len(sb) + 1e-9)

    return [
        ttr(wa), ttr(wb), abs(ttr(wa) - ttr(wb)),
        hapax(wa), hapax(wb), abs(hapax(wa) - hapax(wb)),
        yule_k(wa), yule_k(wb), abs(yule_k(wa) - yule_k(wb)),
        simpsons_d(wa), simpsons_d(wb), abs(simpsons_d(wa) - simpsons_d(wb)),
        jaccard, overlap_a, dice,
    ]  # 15


# ── Block 5: Syntactic proxies (9 features, regex-based) ───────────────────────

def _syntactic_features(a: str, b: str) -> list:
    def feats(text):
        sents = re.split(r"[.!?]+", text)
        sents = [s.strip() for s in sents if s.strip()]
        n_sents = len(sents) or 1
        words = _tokenize(text)
        avg_sent_len = len(words) / n_sents
        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
        comma_rate   = text.count(",") / (len(text) or 1)
        clause_marks = len(re.findall(r"\b(which|who|that|because|although|if|when|where|while)\b", text, re.I))
        clause_rate  = clause_marks / n_sents
        return [avg_sent_len, avg_word_len, comma_rate, clause_rate, n_sents]
    fa = feats(a)
    fb = feats(b)
    diffs = [abs(x - y) for x, y in zip(fa, fb)]  # 5 diffs
    return diffs + [fa[0], fb[0], fa[4], fb[4]]    # 5 + 4 = 9


# ── Block 6: Surface statistics (20 features) ──────────────────────────────────

def _surface_features(a: str, b: str) -> list:
    def feats(text):
        n = len(text) or 1
        words = text.split()
        n_words = len(words) or 1
        lens = [len(w) for w in words]
        return [
            np.mean(lens) if lens else 0.0,           # avg word len
            np.std(lens) if lens else 0.0,            # std word len
            max(lens) if lens else 0.0,               # max word len
            min(lens) if lens else 0.0,               # min word len
            text.count(".") / n,                      # period rate
            text.count("!") / n,                      # exclamation rate
            text.count("?") / n,                      # question rate
            text.count(",") / n,                      # comma rate
            text.count(";") / n,                      # semicolon rate
            sum(c.isupper() for c in text) / n,       # uppercase ratio
            sum(c.isdigit() for c in text) / n,       # digit ratio
            len(re.findall(r"[^a-zA-Z0-9\s]", text)) / n,  # special char ratio
            n_words / n,                              # word density
            len(re.split(r"[.!?]+", text)) / n,       # sentence density
        ]
    fa = feats(a)
    fb = feats(b)
    diffs = [abs(x - y) for x, y in zip(fa, fb)]
    return fa[:3] + fb[:3] + diffs  # 3 + 3 + 14 = 20


# ── Block 7: Readability (5 features) ──────────────────────────────────────────

def _readability_features(a: str, b: str) -> list:
    def fk_grade(text):
        words = _tokenize(text)
        sents = re.split(r"[.!?]+", text)
        sents = [s for s in sents if s.strip()]
        n_words = len(words) or 1
        n_sents = len(sents) or 1
        n_sylls = sum(_syllable_count(w) for w in words)
        return 0.39 * (n_words / n_sents) + 11.8 * (n_sylls / n_words) - 15.59

    def avg_syllables(text):
        words = _tokenize(text)
        return np.mean([_syllable_count(w) for w in words]) if words else 1.0

    return [
        abs(fk_grade(a) - fk_grade(b)),
        abs(avg_syllables(a) - avg_syllables(b)),
        fk_grade(a),
        fk_grade(b),
        avg_syllables(a),
    ]  # 5


# ── Master extractor ────────────────────────────────────────────────────────────

def _features_single(a: str, b: str) -> list:
    return (
        _char_ngram_features(a, b)     +  # 16
        _compression_features(a, b)    +  # 6
        _function_word_features(a, b)  +  # 13
        _vocab_features(a, b)          +  # 15
        _syntactic_features(a, b)      +  # 9
        _surface_features(a, b)        +  # 20
        _readability_features(a, b)       # 5
    )   # = 84


def extract_stylometric_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract 84 stylometric features for each row in df.
    df must have columns: text_1, text_2.
    Returns np.ndarray of shape (len(df), 84).
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[features] Stylometric", leave=False):
        a = str(row["text_1"]) if pd.notna(row["text_1"]) else ""
        b = str(row["text_2"]) if pd.notna(row["text_2"]) else ""
        rows.append(_features_single(a, b))

    arr = np.array(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return arr
