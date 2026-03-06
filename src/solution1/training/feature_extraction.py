import pickle
import argparse
import re
import zlib, bz2, lzma
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cityblock
from scipy.stats import pearsonr, zscore
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from tqdm import tqdm
import textstat

FUNCTION_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "any",
    "these", "give", "most", "us",
]

def char_ngram_features(text_a: str, text_b: str, ns=(2, 3, 4, 5), top_k=3000) -> list:
    """16 features: cosine sim, JSD, Burrows Delta, Pearson r x 4 n-gram orders."""
    from scipy.spatial.distance import jensenshannon
    features = []
    for n in ns:
        vec = CountVectorizer(analyzer='char', ngram_range=(n, n), max_features=top_k)
        try:
            counts = vec.fit_transform([text_a, text_b]).toarray().astype(float)
        except ValueError:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue

        p_a = counts[0] / (counts[0].sum() + 1e-10)
        p_b = counts[1] / (counts[1].sum() + 1e-10)

        # Cosine similarity 
        cos_sim = 1 - cosine(p_a, p_b) if p_a.sum() > 0 and p_b.sum() > 0 else 0.0
        features.append(cos_sim)

        # Jensen-Shannon Divergence
        jsd = jensenshannon(p_a, p_b)
        features.append(float(jsd) if not np.isnan(jsd) else 0.0)

        # Burrows' Delta (mean absolute z-score difference)
        stacked = np.vstack([p_a, p_b])
        z = zscore(stacked, axis=0, nan_policy='omit')
        z = np.nan_to_num(z, 0)
        features.append(float(np.mean(np.abs(z[0] - z[1]))))

        # Pearson r
        corr, _ = pearsonr(p_a, p_b)
        features.append(float(corr) if not np.isnan(corr) else 0.0)

    return features  # length 16

def _ncd(a_bytes: bytes, b_bytes: bytes, compressor) -> float:
    c_a = len(compressor(a_bytes))
    c_b = len(compressor(b_bytes))
    c_ab = len(compressor(a_bytes + b_bytes))
    return (c_ab - min(c_a, c_b)) / (max(c_a, c_b) + 1e-10)

def compression_features(text_a: str, text_b: str) -> list:
    """6 features: NCD with 3 compressors x 2 directions."""
    a = text_a.encode('utf-8')
    b = text_b.encode('utf-8')
    return [
        _ncd(a, b, zlib.compress),
        _ncd(a, b, bz2.compress),
        _ncd(a, b, lzma.compress),
        _ncd(b, a, zlib.compress),
        _ncd(b, a, bz2.compress),
        _ncd(b, a, lzma.compress),
    ]  # length 6

def function_word_features(words_a: list, words_b: list) -> list:
    """~13 features: function word frequency vector similarities."""
    from scipy.spatial.distance import jensenshannon

    def fw_vector(words):
        fw_lower = [w.lower() for w in words]
        total = len(fw_lower) + 1e-10
        return np.array([fw_lower.count(fw) / total for fw in FUNCTION_WORDS])

    v_a = fw_vector(words_a)
    v_b = fw_vector(words_b)

    # Cosine similarity
    cos_sim = 1 - cosine(v_a, v_b) if v_a.sum() > 0 and v_b.sum() > 0 else 0.0

    # Manhattan distance
    manhattan = float(cityblock(v_a, v_b))

    # Max absolute difference
    max_diff = float(np.max(np.abs(v_a - v_b)))

    # Top-10 individual differences (fixed indices — top 10 most variable function words)
    top10_diffs = list(np.abs(v_a - v_b)[:10])  # first 10; swap for MI-selected in full run

    return [cos_sim, manhattan, max_diff] + top10_diffs  # length 13

def _yule_k(words: list) -> float:
    """Yule's K — vocabulary richness robust to text length."""
    if not words:
        return 0.0
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    n = len(words)
    sum_fi2 = sum(v * v for v in freq.values())
    return 10_000 * (sum_fi2 - n) / (n * n + 1e-10)

def _honore_r(words: list) -> float:
    """Honore's R."""
    n = len(words)
    if n == 0:
        return 0.0
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    v = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)
    denom = 1 - (hapax / (v + 1e-10))
    return 100 * np.log(n + 1e-10) / (denom + 1e-10)

def _brunet_w(words: list) -> float:
    """Brunet's W."""
    n = len(words)
    v = len(set(words))
    if n == 0 or v == 0:
        return 0.0
    return n ** (v ** -0.172)

def vocabulary_features(words_a: list, words_b: list) -> list:
    """~15 features: lexical richness metrics, pairwise diffs and Jaccard."""

    def metrics(words):
        n = len(words) + 1e-10
        types = set(words)
        v = len(types) + 1e-10
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        hapax = [w for w, c in freq.items() if c == 1]
        ttr = len(types) / n
        hapax_ratio = len(hapax) / v
        yule = _yule_k(words)
        simpson = sum((c / n) ** 2 for c in freq.values())
        honore = _honore_r(words)
        brunet = _brunet_w(words)
        return np.array([ttr, hapax_ratio, yule, simpson, honore, brunet]), set(types), set(hapax)

    m_a, types_a, hapax_a = metrics(words_a)
    m_b, types_b, hapax_b = metrics(words_b)

    # 6 absolute differences
    abs_diffs = list(np.abs(m_a - m_b))

    # 6 min/max ratios (how similar they are relative to the larger)
    ratios = [min(a, b) / (max(a, b) + 1e-10) for a, b in zip(m_a, m_b)]

    # Jaccard of word types
    jacc_words = len(types_a & types_b) / (len(types_a | types_b) + 1e-10)

    # Jaccard of word bigram sets
    def bigrams(words):
        return set(zip(words[:-1], words[1:]))
    jacc_bigrams = len(bigrams(words_a) & bigrams(words_b)) / (len(bigrams(words_a) | bigrams(words_b)) + 1e-10)

    # Shared hapax ratio
    shared_hapax = len(hapax_a & hapax_b) / (len(hapax_a | hapax_b) + 1e-10)

    return abs_diffs + ratios + [jacc_words, jacc_bigrams, shared_hapax]  # length 15

def syntactic_features(pos_a: list, pos_b: list, sents_a: list, sents_b: list) -> list:
    """~9 features: POS distribution similarities and sentence complexity."""
    from scipy.spatial.distance import jensenshannon
    from collections import Counter

    def pos_vector(tags, vocab):
        counts = Counter(tags)
        total = sum(counts.values()) + 1e-10
        return np.array([counts.get(t, 0) / total for t in vocab])

    # Build a shared POS vocabulary from both sequences
    all_pos = sorted(set(pos_a) | set(pos_b))
    if not all_pos:
        return [0.0] * 9

    v_a = pos_vector(pos_a, all_pos)
    v_b = pos_vector(pos_b, all_pos)

    # POS unigram cosine similarity
    cos_uni = 1 - cosine(v_a, v_b) if v_a.sum() > 0 and v_b.sum() > 0 else 0.0

    # POS unigram JSD
    jsd_uni = jensenshannon(v_a, v_b)
    jsd_uni = float(jsd_uni) if not np.isnan(jsd_uni) else 0.0

    # POS bigram cosine similarity
    def pos_bigrams(tags):
        return [f"{a}_{b}" for a, b in zip(tags[:-1], tags[1:])]
    bg_a = pos_bigrams(pos_a)
    bg_b = pos_bigrams(pos_b)
    all_bg = sorted(set(bg_a) | set(bg_b))
    if all_bg:
        bv_a = pos_vector(bg_a, all_bg)
        bv_b = pos_vector(bg_b, all_bg)
        cos_bi = 1 - cosine(bv_a, bv_b) if bv_a.sum() > 0 and bv_b.sum() > 0 else 0.0
    else:
        cos_bi = 0.0

    # Major POS class ratio diffs: NOUN, VERB, ADJ, ADV
    def ratio(tags, label):
        return tags.count(label) / (len(tags) + 1e-10)
    noun_diff = abs(ratio(pos_a, "NOUN") - ratio(pos_b, "NOUN"))
    verb_diff = abs(ratio(pos_a, "VERB") - ratio(pos_b, "VERB"))
    adj_diff  = abs(ratio(pos_a, "ADJ")  - ratio(pos_b, "ADJ"))
    adv_diff  = abs(ratio(pos_a, "ADV")  - ratio(pos_b, "ADV"))

    # Average sentence length diff (proxy for syntactic complexity)
    def avg_sent_len(sents):
        lengths = [len(s.split()) for s in sents if s.strip()]
        return np.mean(lengths) if lengths else 0.0
    sent_len_diff = abs(avg_sent_len(sents_a) - avg_sent_len(sents_b))

    # Sentence length std diff (variability in rhythm)
    def std_sent_len(sents):
        lengths = [len(s.split()) for s in sents if s.strip()]
        return np.std(lengths) if len(lengths) > 1 else 0.0
    sent_std_diff = abs(std_sent_len(sents_a) - std_sent_len(sents_b))

    return [cos_uni, jsd_uni, cos_bi, noun_diff, verb_diff, adj_diff, adv_diff,
            sent_len_diff, sent_std_diff]  # length 9


def surface_features(text_a: str, text_b: str, words_a: list, words_b: list,
                     sents_a: list, sents_b: list) -> list:
    """~20 features: word/sentence length, punctuation, contractions, etc."""
    from collections import Counter

    PUNCT_MARKS = list('.,;:!?-—"\'()')
    CONTRACTION_PATTERN = re.compile(r"'(re|s|t|ll|ve|d|m)\b")

    def word_length_stats(words):
        lengths = [len(w) for w in words if w]
        if not lengths:
            return 0.0, 0.0
        return np.mean(lengths), np.std(lengths)

    def punct_vector(text):
        total = len(text) + 1e-10
        return np.array([text.count(p) / total for p in PUNCT_MARKS])

    wl_mean_a, wl_std_a = word_length_stats(words_a)
    wl_mean_b, wl_std_b = word_length_stats(words_b)

    # Avg/std sentence length diffs
    def sent_stats(sents):
        lengths = [len(s.split()) for s in sents if s.strip()]
        if not lengths:
            return 0.0, 0.0
        return np.mean(lengths), np.std(lengths)

    sl_mean_a, sl_std_a = sent_stats(sents_a)
    sl_mean_b, sl_std_b = sent_stats(sents_b)

    # Punctuation cosine similarity + per-mark absolute diffs
    pv_a = punct_vector(text_a)
    pv_b = punct_vector(text_b)
    punct_cos = 1 - cosine(pv_a, pv_b) if pv_a.sum() > 0 and pv_b.sum() > 0 else 0.0
    punct_diffs = list(np.abs(pv_a - pv_b))  # 13 per-mark diffs

    # Contraction rate diff
    def contraction_rate(text, words):
        return len(CONTRACTION_PATTERN.findall(text)) / (len(words) + 1e-10)
    contract_diff = abs(contraction_rate(text_a, words_a) - contraction_rate(text_b, words_b))

    # Digit rate diff
    def digit_rate(text):
        return sum(c.isdigit() for c in text) / (len(text) + 1e-10)
    digit_diff = abs(digit_rate(text_a) - digit_rate(text_b))

    # Uppercase ratio diff
    def upper_rate(text):
        letters = [c for c in text if c.isalpha()]
        return sum(c.isupper() for c in letters) / (len(letters) + 1e-10)
    upper_diff = abs(upper_rate(text_a) - upper_rate(text_b))

    return [
        abs(wl_mean_a - wl_mean_b),   # avg word length diff
        abs(wl_std_a  - wl_std_b),    # std word length diff
        abs(sl_mean_a - sl_mean_b),   # avg sentence length diff
        abs(sl_std_a  - sl_std_b),    # std sentence length diff
        punct_cos,                     # punctuation distribution cosine
        *punct_diffs,                  # 13 per-mark diffs
        contract_diff,
        digit_diff,
        upper_diff,
    ]  # length 20

def readability_features(text_a: str, text_b: str) -> list:
    """features: absolute differences of readability indices."""
    def safe(fn, text):
        try:
            return float(fn(text))
        except Exception:
            return 0.0

    metrics_a = [
        safe(textstat.flesch_kincaid_grade, text_a),
        safe(textstat.coleman_liau_index, text_a),
        safe(textstat.automated_readability_index, text_a),
        safe(textstat.smog_index, text_a),
        safe(textstat.avg_syllables_per_word, text_a),
    ]
    metrics_b = [
        safe(textstat.flesch_kincaid_grade, text_b),
        safe(textstat.coleman_liau_index, text_b),
        safe(textstat.automated_readability_index, text_b),
        safe(textstat.smog_index, text_b),
        safe(textstat.avg_syllables_per_word, text_b),
    ]
    return [abs(a - b) for a, b in zip(metrics_a, metrics_b)]  # length 5



def extract_all_features(pair: dict) -> np.ndarray:
    """Combine all feature groups for a single preprocessed pair dict."""
    ta = pair["text_1"]["raw"]
    tb = pair["text_2"]["raw"]
    wa = pair["text_1"]["words"]
    wb = pair["text_2"]["words"]
    pa = pair["text_1"]["pos_tags"]
    pb = pair["text_2"]["pos_tags"]
    sa = pair["text_1"]["sentences"]
    sb = pair["text_2"]["sentences"]

    feats = (
        char_ngram_features(ta, tb)          # 16
        + compression_features(ta, tb)        #  6
        + function_word_features(wa, wb)      # 13
        + vocabulary_features(wa, wb)         # 15
        + syntactic_features(pa, pb, sa, sb)  #  9
        + surface_features(ta, tb, wa, wb, sa, sb)  # 20
        + readability_features(ta, tb)        #  5
    )  # total 84

    arr = np.array(feats, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to preprocessed .pkl cache")
    parser.add_argument("--output", required=True, help="Path to output .npy feature matrix")
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading cache from {args.input} ...")
    with open(args.input, "rb") as f:
        pairs = pickle.load(f)

    print(f"Extracting features from {len(pairs)} pairs with {args.n_jobs} workers ...")
    features = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(extract_all_features)(pair) for pair in tqdm(pairs)
    )

    X = np.vstack(features)
    labels = np.array([p["label"] for p in pairs])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, X)
    np.save(output_path.parent / "labels_trial.npy", labels)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
