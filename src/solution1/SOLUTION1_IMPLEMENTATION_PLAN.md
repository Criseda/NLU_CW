# Solution 1 (Category A): Classical ML for Authorship Verification
## Detailed Implementation Plan — Partner 1

---

## 1. Design Philosophy

The core insight of authorship verification is that **style is unconscious**. Authors make deliberate choices about *what* they write (topic, vocabulary) but have far less control over *how* they write (punctuation habits, function word usage, character-level patterns, sentence rhythm). A strong classical AV system should capture these involuntary stylistic signatures and measure how similar they are between two texts.

Our approach has two layers:

1. **Multi-view stylometric feature engineering**: Extract features at multiple linguistic levels — character, lexical, syntactic, and document — compute similarity/divergence measures between each text in the pair.

2. **Computationally expensive, literature-grounded AV meta-features**: Implement two landmark techniques from the AV research community — **Unmasking** (Koppel & Schler, 2004) and the **General Impostor Method** (Koppel & Winter, 2014). These are powerful but prohibitively slow on a laptop — **this is where the University of Manchester CSF (HPC cluster) becomes essential**.

These two layers are fused via **stacked generalisation**: two base classifiers (linear + tree-based) combined through a learned meta-classifier.

### Literature Grounding

- Stamatatos (2009), *A Survey of Modern Authorship Attribution Methods*
- **Koppel & Schler (2004), *Authorship Verification as a One-Class Classification Problem*** — introduces the Unmasking technique
- **Koppel & Winter (2014), *Determining if Two Documents are Written by the Same Author*** — introduces the General Impostor Method
- Kestemont et al. (2020), *Overview of the PAN 2020 Authorship Verification Task*
- Halvani et al. (2016), *Authorship Verification for Different Languages, Genres and Topics*
- Cilibrasi & Vitányi (2005), *Clustering by Compression* — NCD
- Mosteller & Wallace (1964) — function word analysis

---

## 2. Preprocessing

Minimal preprocessing is critical. Heavy normalisation destroys stylistic signal.

```
Input: (text_a, text_b, label)

Steps:
1. Strip leading/trailing whitespace only.
2. Unicode normalisation (NFKC) to collapse equivalent characters.
3. Sentence segmentation via spaCy (en_core_web_sm) — needed for syntactic features.
4. Tokenisation: both word-level (spaCy) and character-level (raw string).
5. POS tagging via spaCy.
6. Do NOT lowercase — capitalisation patterns are a stylistic feature.
7. Do NOT remove punctuation — punctuation usage is a strong authorship marker.
8. Do NOT remove stopwords — function words are among the most discriminative features.
```

**Rationale:** Every normalisation step removes potential signal. In authorship tasks, features that are "noise" in other NLP tasks (punctuation frequency, capitalisation habits, stopword ratios) are primary signal.

**HPC note:** Preprocessing (especially spaCy POS tagging) for 27K+ pairs is slow. On CSF, parallelise with `joblib.Parallel` across CPU cores or submit as a SLURM array job. Cache all POS tags and sentence boundaries to disk as a `.pkl` so this only runs once.

---

## 3. Feature Engineering — Part 1: Stylometric Features

For each pair `(text_a, text_b)`, we compute a fixed-length feature vector. Features in this section are **fast to compute** and form the baseline feature set.

### 3.1 Character N-gram Profiles

**Literature basis:** Character n-grams are the single most effective feature type for authorship tasks (Stamatatos, 2009; Kestemont et al., 2016). They capture morphological habits, punctuation patterns, and subword preferences without requiring any linguistic annotation.

**Extraction:**
- For each text, compute frequency distributions of character n-grams for n ∈ {2, 3, 4, 5}.
- Normalise to relative frequencies (sum to 1).
- Restrict to top-k most frequent n-grams across the training corpus (e.g., k=2000–5000 per n).

**Similarity features (per n-gram order):**
| Feature | Formula | Intuition |
|---|---|---|
| Cosine similarity | cos(p_a, p_b) | Directional similarity of distributions |
| Burrows' Delta | Σ\|z_a_i - z_b_i\| / V | Mean absolute z-score difference; standard in stylometry |
| Jensen-Shannon Divergence | 0.5·KL(p_a \|\| m) + 0.5·KL(p_b \|\| m) | Symmetric, bounded, information-theoretic |
| Correlation | pearsonr(p_a, p_b) | Linear co-variation of frequencies |

**Yields:** 4 similarity metrics × 4 n-gram orders = **16 features**.

```python
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import pearsonr, zscore
import numpy as np

def char_ngram_features(text_a, text_b, ns=[2, 3, 4, 5], top_k=3000):
    features = []
    for n in ns:
        vec = CountVectorizer(analyzer='char', ngram_range=(n, n), max_features=top_k)
        counts = vec.fit_transform([text_a, text_b]).toarray().astype(float)
        p_a = counts[0] / (counts[0].sum() + 1e-10)
        p_b = counts[1] / (counts[1].sum() + 1e-10)
        features.append(1 - cosine(p_a, p_b))
        features.append(jensenshannon(p_a, p_b))
        stacked = np.vstack([p_a, p_b])
        z = zscore(stacked, axis=0, nan_policy='omit')
        z = np.nan_to_num(z, 0)
        features.append(np.mean(np.abs(z[0] - z[1])))
        corr, _ = pearsonr(p_a, p_b)
        features.append(corr if not np.isnan(corr) else 0.0)
    return features
```

**Important:** In the final pipeline, the `CountVectorizer` vocabulary should be **fit on the entire training corpus**, not per-pair. Pre-fit and save the vocabulary.

---

### 3.2 Compression-Based Features (Normalized Compression Distance)

**Literature basis:** Cilibrasi & Vitányi (2005). NCD has been used competitively in PAN authorship verification tasks. It's parameter-free, language-independent, and captures patterns that resist explicit specification.

**Intuition:** If two texts share an author, compressing their concatenation should be efficient relative to compressing them individually — the compressor exploits shared patterns.

**Formula:**
```
NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
```

```python
import zlib, bz2, lzma

def ncd(text_a, text_b, compressor):
    a = text_a.encode('utf-8')
    b = text_b.encode('utf-8')
    c_a, c_b = len(compressor(a)), len(compressor(b))
    c_ab = len(compressor(a + b))
    return (c_ab - min(c_a, c_b)) / (max(c_a, c_b) + 1e-10)

def compression_features(text_a, text_b):
    return [
        ncd(text_a, text_b, zlib.compress),
        ncd(text_a, text_b, bz2.compress),
        ncd(text_a, text_b, lzma.compress),
        ncd(text_b, text_a, zlib.compress),
        ncd(text_b, text_a, bz2.compress),
        ncd(text_b, text_a, lzma.compress),
    ]
```

**Yields:** **6 features** (3 compressors × 2 concatenation directions).

---

### 3.3 Function Word Distributions

**Literature basis:** Mosteller & Wallace (1964); Argamon & Levitan (2005).

**Intuition:** Function words (the, of, and, a, to, in, is, that, ...) are used unconsciously and are topic-independent. Their relative frequencies form a reliable stylistic fingerprint.

**Extraction:**
- Fixed list of ~100 English function words.
- Relative frequency per text (count / total words).
- Compute pairwise similarity.

**Features:**
| Feature | Description |
|---|---|
| Cosine similarity | Of function word frequency vectors |
| Manhattan distance | Sum of absolute differences |
| Max absolute difference | Largest single function word discrepancy |
| Top-10 individual differences | For the 10 most discriminative function words (selected via MI on training set) |

**Yields:** ~**13 features**.

---

### 3.4 Vocabulary & Lexical Richness

**Features computed per text, then differenced:**

| Feature | Description |
|---|---|
| Type-Token Ratio (TTR) | \|V\| / N — basic vocabulary diversity |
| Hapax Legomena Ratio | hapax / \|V\| — words used exactly once |
| Yule's K | Vocabulary richness, robust to text length |
| Simpson's D | Probability two random words are the same |
| Honore's R | 100·log(N) / (1 - hapax/\|V\|) |
| Brunet's W | N^(V^-0.172) |

**Pairwise features:**
- Absolute difference of each metric → 6 features.
- Ratio min/max of each metric → 6 features.
- Jaccard similarity of word sets → 1 feature.
- Jaccard similarity of bigram sets → 1 feature.
- Shared hapax ratio → 1 feature.

**Yields:** ~**15 features**.

---

### 3.5 Syntactic Features (POS-Based)

**Features:**
| Feature | Description |
|---|---|
| POS unigram cosine similarity | Similarity of POS tag frequency vectors |
| POS bigram cosine similarity | Captures syntactic transition patterns |
| POS unigram JSD | Jensen-Shannon divergence of POS distributions |
| Major POS class ratio diffs | Noun/verb/adj/adv proportion differences (4 features) |
| Average dependency depth diff | Captures sentence complexity |
| Subordinate clause ratio diff | SBAR-like structure proportion |

**Yields:** ~**9 features**.

---

### 3.6 Surface Stylometrics

| Feature | Description |
|---|---|
| Avg/std word length diff | Word length distribution differences |
| Avg/std sentence length diff | Sentence rhythm and variability |
| Punctuation frequency cosine sim | Distribution over .,;:!?-—"'() etc. |
| Individual punctuation diffs | Per-mark absolute differences (~10 marks) |
| Contraction usage rate diff | Frequency of 're, 's, n't, 'll, etc. |
| Digit usage rate diff | Formatting preferences |
| Uppercase ratio diff | Capitalisation habits |

**Yields:** ~**20 features**.

---

### 3.7 Readability Metrics

| Metric | Description |
|---|---|
| Flesch-Kincaid Grade Level diff | Grade level of writing |
| Coleman-Liau Index diff | Character-count based readability |
| Automated Readability Index diff | Sentence and word complexity |
| SMOG Index diff | Based on polysyllable count |
| Average syllable count diff | Per word |

**Yields:** **5 features**.

---

## 4. Feature Engineering — Part 2: HPC-Powered AV Meta-Features

These are the **high-creativity, high-compute features** that set this solution apart. Each technique is a landmark method in the AV literature, but is computationally prohibitive on a laptop because it requires training many classifiers *per pair*. With access to Manchester CSF, we can run them at scale.

---

### 4.1 Unmasking (Koppel & Schler, 2004)

This is arguably the most important technique in classical authorship verification. It was designed specifically for AV and addresses a fundamental problem: two texts by different authors on the *same topic* may look superficially similar, while texts by the *same author* on different topics may look superficially different. Unmasking distinguishes deep stylistic similarity from shallow topical similarity.

**Algorithm (per pair):**

```
Input: text_a, text_b

1. Split text_a into chunks of ~500 words → chunks_a
   Split text_b into chunks of ~500 words → chunks_b

2. Represent each chunk as a character 3-gram frequency vector (top 1000 features).

3. ROUND 0: Train a linear SVM to classify chunks_a vs chunks_b
   using cross-validation. Record accuracy_0.

4. ROUND 1: Remove the top-k most discriminative features
   (highest absolute SVM weights). Retrain. Record accuracy_1.

5. Repeat for R rounds (typically R=10, k=5-10 features removed per round).

6. Output: the degradation curve [accuracy_0, accuracy_1, ..., accuracy_R].
```

**Interpretation:**
- **Same author**: Accuracy drops rapidly. The differences are superficial (topic words); once stripped, underlying style is shared.
- **Different author**: Accuracy stays high. Differences are deeply stylistic and persist after feature removal.

**Features extracted from the degradation curve:**
| Feature | Description |
|---|---|
| accuracy_0 | Initial separability |
| accuracy_R | Final accuracy after R rounds |
| Slope | Linear regression slope of degradation curve — **the key signal** |
| Max single-round drop | Largest accuracy decrease in one round |
| AUC under curve | Area under accuracy-vs-round curve |
| Total drop | accuracy_0 - accuracy_R |
| Collapse round | First round where accuracy < 0.6 |
| Drop std | Smoothness of degradation |

**Yields:** ~**8 features**.

**Computational cost & why this needs HPC:**
- Each pair: ~10 rounds × ~5-fold CV = ~50 classifier fits per pair.
- 27K training pairs → ~1.35 million classifier fits.
- Single laptop core: ~20-40 hours.
- **CSF (32-core node): ~1-2 hours via `joblib.Parallel`.**

```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def unmasking_features(text_a, text_b, n_rounds=10, k_remove=6, chunk_size=500):
    """
    Compute unmasking degradation curve for a single pair.
    Designed to be run in parallel across pairs on HPC.
    """
    # 1. Chunk both texts
    words_a = text_a.split()
    words_b = text_b.split()
    chunks_a = [' '.join(words_a[i:i+chunk_size])
                for i in range(0, len(words_a), chunk_size)]
    chunks_b = [' '.join(words_b[i:i+chunk_size])
                for i in range(0, len(words_b), chunk_size)]

    # Need at least 3 chunks per side for meaningful CV
    if len(chunks_a) < 3 or len(chunks_b) < 3:
        return [np.nan] * 8  # fallback: impute later

    labels = [0]*len(chunks_a) + [1]*len(chunks_b)
    all_chunks = chunks_a + chunks_b

    # 2. Vectorise: char 3-grams
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 3), max_features=1000)
    X = vec.fit_transform(all_chunks).toarray().astype(float)
    y = np.array(labels)

    accuracies = []
    n_folds = min(5, min(len(chunks_a), len(chunks_b)))

    for round_i in range(n_rounds):
        if X.shape[1] < k_remove:
            accuracies.append(accuracies[-1] if accuracies else 0.5)
            continue

        clf = LinearSVC(max_iter=5000, dual='auto')
        try:
            scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
            acc = scores.mean()
        except:
            acc = 0.5
        accuracies.append(acc)

        # Remove top-k most discriminative features
        clf.fit(X, y)
        importances = np.abs(clf.coef_[0])
        top_k_idx = np.argsort(importances)[-k_remove:]
        mask = np.ones(X.shape[1], dtype=bool)
        mask[top_k_idx] = False
        X = X[:, mask]

    # Extract features from curve
    accs = np.array(accuracies)
    slope = np.polyfit(range(len(accs)), accs, 1)[0] if len(accs) > 1 else 0
    drops = np.diff(accs)

    return [
        accs[0],                                    # initial accuracy
        accs[-1],                                   # final accuracy
        slope,                                      # degradation slope (KEY)
        np.min(drops) if len(drops) > 0 else 0,    # max single-round drop
        np.trapz(accs),                             # AUC of curve
        accs[0] - accs[-1],                         # total drop
        next((i for i, a in enumerate(accs) if a < 0.6), len(accs)),  # collapse round
        np.std(drops) if len(drops) > 0 else 0,    # drop smoothness
    ]
```

**CSF parallelisation:**
```python
from joblib import Parallel, delayed

# On a 32-core CSF node:
all_unmasking = Parallel(n_jobs=32, verbose=10)(
    delayed(unmasking_features)(row['text_a'], row['text_b'])
    for _, row in train_df.iterrows()
)
unmasking_array = np.array(all_unmasking)
np.save('features/unmasking_train.npy', unmasking_array)
```

**SLURM job script:**
```bash
#!/bin/bash --login
#SBATCH --job-name=unmasking
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=multicore

module load anaconda3/2024.02
conda activate av_env

python compute_unmasking_features.py \
    --input data/train.csv \
    --output features/unmasking_train.npy \
    --n_jobs 32
```

---

### 4.2 General Impostor Method (GI) (Koppel & Winter, 2014)

The GI method asks: "Is text_a more similar to text_b than to random other texts (impostors)?" If consistently yes across many random impostor sets, the pair likely shares an author.

**Algorithm (per pair):**

```
Input: text_a, text_b, corpus of all training texts

For trial = 1 to T (e.g., T=100):
    1. Sample m impostor texts randomly from the corpus (excluding text_a, text_b).
    2. Compute similarity(text_a, text_b) using char n-gram cosine sim.
    3. Compute similarity(text_a, impostor_i) for each impostor.
    4. score_trial = 1 if text_b is most similar to text_a
                       among {text_b, impostor_1, ..., impostor_m}
                     0 otherwise.

Final GI score = mean(score_trial over all T trials)
```

**Features:**
| Feature | Description |
|---|---|
| GI score (text_a perspective) | Proportion of trials where text_b beats all impostors for text_a |
| GI score (text_b perspective) | Same but querying from text_b's side |
| GI score (averaged) | Mean of both perspectives |
| GI score (min) | min(GI_a, GI_b) — conservative estimate |
| GI score std | Standard deviation across trials — confidence |

**Yields:** **5 features**.

**Computational cost & why this needs HPC:**
- Per pair: T=100 trials × m=50 similarity computations.
- 27K pairs × 100 trials = 2.7M similarity lookups.
- Single core: ~10-20 hours. **CSF (32 cores): ~30-60 minutes.**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def general_impostor_features(
    idx_a, idx_b,
    corpus_vectors,    # Pre-computed char n-gram TF-IDF matrix (sparse)
    n_trials=100,
    n_impostors=50,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng(42)

    n_texts = corpus_vectors.shape[0]
    exclude = {idx_a, idx_b}
    candidate_pool = [i for i in range(n_texts) if i not in exclude]

    vec_a = corpus_vectors[idx_a]
    vec_b = corpus_vectors[idx_b]

    scores_a, scores_b = [], []

    for _ in range(n_trials):
        impostor_idxs = rng.choice(candidate_pool, size=n_impostors, replace=False)
        impostor_vecs = corpus_vectors[impostor_idxs]

        sim_ab = cosine_similarity(vec_a, vec_b)[0, 0]
        sim_a_imps = cosine_similarity(vec_a, impostor_vecs)[0]
        scores_a.append(1.0 if sim_ab > sim_a_imps.max() else 0.0)

        sim_b_imps = cosine_similarity(vec_b, impostor_vecs)[0]
        scores_b.append(1.0 if sim_ab > sim_b_imps.max() else 0.0)

    gi_a, gi_b = np.mean(scores_a), np.mean(scores_b)

    return [gi_a, gi_b, (gi_a + gi_b) / 2, min(gi_a, gi_b), np.std(scores_a + scores_b)]
```

**Pre-computation step (run once on CSF):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Build corpus of ALL individual texts
all_texts = list(train_df['text_a']) + list(train_df['text_b'])

tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=5000)
corpus_vectors = tfidf.fit_transform(all_texts)

joblib.dump(tfidf, 'models/gi_tfidf_vectorizer.joblib')
joblib.dump(corpus_vectors, 'models/gi_corpus_vectors.joblib')
```

**Important for inference:** At test time, the impostor pool is the training corpus (already saved). Test texts are vectorised with the same fitted TF-IDF. No label leakage — only comparing against known texts.

---

### 4.3 Feature Summary (Full Pipeline)

| Group | Features | Linguistic Level | Compute |
|---|---|---|---|
| Character n-grams (§3.1) | 16 | Sub-word | CPU (fast) |
| Compression / NCD (§3.2) | 6 | Holistic | CPU (medium) |
| Function words (§3.3) | 13 | Lexical (closed-class) | CPU (fast) |
| Vocabulary richness (§3.4) | 15 | Lexical | CPU (fast) |
| Syntactic / POS (§3.5) | 9 | Syntactic | CPU (medium) |
| Surface stylometrics (§3.6) | 20 | Mechanical | CPU (fast) |
| Readability (§3.7) | 5 | Document | CPU (fast) |
| **Unmasking (§4.1)** | **8** | **Meta-stylistic** | **HPC ×32 cores** |
| **General Impostor (§4.2)** | **5** | **Meta-comparative** | **HPC ×32 cores** |
| **Total** | **~97** | **Multi-level** | |

---

## 5. Model Architecture: Stacked Ensemble

### 5.1 Base Classifiers

Two base models with different inductive biases on the full 97-feature vector:

**Model A1 — Logistic Regression (ElasticNet)**
- Linear model. ElasticNet provides simultaneous feature selection and regularisation.
- Hyperparameters: `C`, `l1_ratio`.

**Model A2 — Gradient Boosted Trees (LightGBM)**
- Non-linear model capturing feature interactions and thresholds.
- **GPU training on CSF** for fast iteration during tuning.
- Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.

### 5.2 Stacking (Meta-Classifier)

1. **Stratified 5-fold CV** on training set.
2. Per fold: train A1 and A2, generate **out-of-fold probability predictions**.
3. Train **Logistic Regression meta-classifier** on (P_A1, P_A2) → label.
4. Retrain A1 and A2 on **full training set** for inference.

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│                   Input Pair                         │
│                (text_a, text_b)                       │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌───────────┐ ┌──────────────────┐
│ Stylometric  │ │ Unmasking │ │ General Impostor │
│ Features     │ │ (HPC)     │ │ Method (HPC)     │
│ §3.1 – §3.7 │ │ §4.1      │ │ §4.2             │
│  84 dims     │ │  8 dims   │ │  5 dims          │
└──────┬───────┘ └─────┬─────┘ └────────┬─────────┘
       └───────────────┼────────────────┘
                       ▼
              ┌──────────────────┐
              │  97-dim feature  │
              │     vector       │
              └───────┬──────────┘
                      │
           ┌──────────┼──────────┐
           ▼                     ▼
  ┌──────────────────┐  ┌────────────────────┐
  │  Model A1        │  │  Model A2          │
  │  Logistic Reg    │  │  LightGBM          │
  │  (ElasticNet)    │  │  (GPU on CSF)      │
  │       ↓          │  │        ↓           │
  │   P(same|A1)     │  │    P(same|A2)      │
  └────────┬─────────┘  └────────┬───────────┘
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │  Meta-Classifier    │
           │  Logistic Regression│
           │         ↓           │
           │  Final P(same)      │
           │  Threshold @ 0.5    │
           └─────────────────────┘
```

---

## 6. Training Strategy & HPC Workflow

### 6.1 CSF Environment Setup

```bash
# On CSF login node:
module load anaconda3/2024.02
conda create -n av_env python=3.11 -y
conda activate av_env

pip install scikit-learn lightgbm optuna scipy numpy pandas spacy joblib textstat matplotlib seaborn

# LightGBM with GPU support:
pip install lightgbm --install-option=--gpu

python -m spacy download en_core_web_sm
```

### 6.2 Compute Pipeline (Phased)

```
Phase A: Preprocessing  [CSF, 1 CPU node, ~30 min]
   train.csv → spaCy processing → cached_pos_tags.pkl, cached_sentences.pkl

Phase B: Fast Features  [CSF or local, ~1 hr]
   cached data → §3.1-§3.7 → stylometric_features_train.npy

Phase C: Unmasking      [CSF, 32-core CPU node, ~2 hrs wall time]
   train.csv → unmasking_features_train.npy

Phase D: Impostor       [CSF, 32-core CPU node, ~1 hr wall time]
   train.csv + corpus_vectors → impostor_features_train.npy

Phase E: Train & Tune   [CSF GPU node, ~4-6 hrs wall time]
   Concatenate all features → 97-dim matrix
   Optuna (500 trials) → best hyperparameters
   Train final A1, A2, meta-classifier → save models

Phase F: Evaluate        [CSF or local, ~2 hrs]
   Ablation, bootstrap significance, error analysis
```

### 6.3 Hyperparameter Tuning (Optuna on CSF)

**Logistic Regression:**
```python
{
    'C': loguniform(1e-4, 1e2),
    'l1_ratio': uniform(0.0, 1.0),
    'solver': 'saga',
    'penalty': 'elasticnet',
    'max_iter': 5000
}
```

**LightGBM (GPU on CSF):**
```python
{
    'n_estimators': int_uniform(100, 3000),
    'max_depth': int_uniform(3, 12),
    'learning_rate': loguniform(1e-3, 0.3),
    'num_leaves': int_uniform(15, 256),
    'min_child_samples': int_uniform(5, 100),
    'subsample': uniform(0.5, 1.0),
    'colsample_bytree': uniform(0.3, 1.0),
    'reg_alpha': loguniform(1e-4, 10.0),
    'reg_lambda': loguniform(1e-4, 10.0),
    'device': 'gpu',
}
```

**Protocol:**
- **500 Optuna trials** per model (feasible on CSF; infeasible on laptop).
- Stratified 5-fold CV, scoring on macro F1.
- TPE sampler with MedianPruner.
- After tuning: retrain on full training set, validate on dev set.

**SLURM script:**
```bash
#!/bin/bash --login
#SBATCH --job-name=optuna_lgbm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu

module load anaconda3/2024.02
conda activate av_env

python tune_hyperparams.py \
    --model lgbm \
    --features features/all_features_train.npy \
    --labels data/train_labels.npy \
    --n_trials 500 \
    --output tuning/lgbm_study.db
```

### 6.4 Feature Scaling

- **Logistic Regression:** StandardScaler. Fit on training data only.
- **LightGBM:** No scaling needed.

---

## 7. Evaluation Plan (Targeting 3 Marks)

### 7.1 Core Metrics
- Macro F1 (primary), per-class Precision/Recall, Accuracy, ROC-AUC, PR-AUC.

### 7.2 Statistical Significance
- **McNemar's test**: ensemble vs. each base model alone.
- **Bootstrap confidence intervals**: 10,000 resamples (feasible on HPC; most groups do 100-1000).

### 7.3 Ablation Study

| Ablated Group | Dev F1 | Δ from Full |
|---|---|---|
| Full model (97 features) | — | — |
| − Character n-grams | | |
| − Compression (NCD) | | |
| − Function words | | |
| − Vocabulary richness | | |
| − Syntactic (POS) | | |
| − Surface stylometrics | | |
| − Readability | | |
| − **Unmasking** | | |
| − **General Impostor** | | |

9 ablation experiments × full retrain — trivial on HPC.

### 7.4 Cross-Validation Variance
- Mean ± std F1 across 5 folds.
- Compare variance: base models vs. ensemble (ensemble should be lower — empirical argument for ensembling).

### 7.5 Error Analysis
- 50+ misclassified dev pairs, categorised by failure mode:
  - Short texts (insufficient signal).
  - Topic-dominated pairs (same topic → false positive).
  - Same-genre different-author pairs.
  - Cases where unmasking and GI disagree.
- Feature values for errors vs. correct predictions.

### 7.6 Calibration
- Reliability diagram + Expected Calibration Error (ECE).

---

## 8. Inference Pipeline (Demo Notebook)

Must be **self-contained and runnable by markers** who do NOT have HPC.

```python
# demo_solution1.ipynb — Structure

# Cell 1: Install dependencies
# !pip install lightgbm scikit-learn scipy spacy joblib textstat
# !python -m spacy download en_core_web_sm

# Cell 2: Load saved artefacts
# - scaler.joblib, char_ngram_vocabs.joblib
# - gi_tfidf_vectorizer.joblib, gi_corpus_vectors.joblib
# - model_a1.joblib, model_a2.joblib, meta_classifier.joblib

# Cell 3: Define feature extraction functions

# Cell 4: Load test data
# test_df = pd.read_csv("test_data.csv")

# Cell 5: Extract features (with pre-compute fallback)
# if pre-computed features exist, load them; else compute live

# Cell 6: Base model probabilities
# p_a1 = model_a1.predict_proba(scaler.transform(X_test))[:, 1]
# p_a2 = model_a2.predict_proba(X_test)[:, 1]

# Cell 7: Stack and predict
# predictions = meta_classifier.predict(np.column_stack([p_a1, p_a2]))

# Cell 8: Save
# pd.DataFrame({'id': test_df['id'], 'label': predictions}).to_csv("Group_X_A.csv", index=False)
```

**Critical: test-time feasibility.** Unmasking is the bottleneck. For 6K test pairs on a laptop (4 cores), ~30-60 min. Pre-compute on CSF and cache with a fallback:

```python
import os
if os.path.exists('features/unmasking_test.npy'):
    print("Loading pre-computed unmasking features...")
    unmasking_feats = np.load('features/unmasking_test.npy')
else:
    print("Computing unmasking features (~30-60 min)...")
    unmasking_feats = compute_unmasking_parallel(test_df, n_jobs=4)
```

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Unmasking needs long texts to chunk (min ~1500 words) | If text < 1500 words, return NaN → impute with training median |
| CSF queue delays | Submit jobs early; do local work in parallel |
| Feature extraction bugs → train/test mismatch | Unit tests: vector length, no NaN/Inf, determinism |
| LightGBM overfitting | Early stopping, max_depth cap, monitor train-dev gap |
| Demo notebook too slow for markers | Pre-compute expensive features for test set on CSF |
| Text length imbalance → noisy features for short texts | Add text length (both + ratio) as explicit features |
| GI impostor pool quality | Use full training corpus; document in model card |

---

## 10. File Structure

```
solution1/
├── README.md
├── training/
│   ├── preprocess.py
│   ├── feature_extraction.py       # §3.1-3.7
│   ├── compute_unmasking.py        # §4.1 (HPC)
│   ├── compute_impostor.py         # §4.2 (HPC)
│   ├── train_base_models.py
│   ├── train_meta_classifier.py
│   ├── tune_hyperparams.py
│   └── evaluate.py
├── slurm/
│   ├── preprocess.sh
│   ├── unmasking.sh
│   ├── impostor.sh
│   └── tuning.sh
├── demo/
│   └── demo_solution1.ipynb
├── models/                         # .joblib (or OneDrive if >10MB)
├── features/                       # .npy arrays (OneDrive if large)
├── model_card_solution1.md
└── results/
    ├── dev_metrics.json
    ├── ablation_results.csv
    ├── confusion_matrix.png
    ├── calibration_plot.png
    └── bootstrap_ci.json
```

---

## 11. Creativity Argument (For Model Card & Poster)

1. **Unmasking (Koppel & Schler, 2004):** Landmark AV technique that separates deep stylistic similarity from superficial topical overlap. Computationally expensive — only feasible at scale with HPC. Rarely seen in coursework.

2. **General Impostor Method (Koppel & Winter, 2014):** State-of-the-art classical AV. Contextualises similarity against background corpus — addresses the problem that absolute similarity thresholds are unreliable.

3. **Multi-view stylometric features** across 7 linguistic levels, grounded in computational stylometry literature.

4. **Compression-based features (NCD):** Parameter-free, information-theoretic, language-independent.

5. **Stacked generalisation** with heterogeneous base learners (linear + gradient boosted), fused via learned meta-classifier.

6. **Rigorous evaluation:** 10K-resample bootstrap CIs, ablation across all 9 feature groups, calibration analysis, structured error taxonomy.

---

## 12. Implementation Timeline

| # | Task | Where | Est. Time |
|---|---|---|---|
| 1 | CSF env setup, conda, test SLURM | CSF | 2 hrs |
| 2 | Data loading, preprocessing, spaCy caching | CSF | 2-3 hrs |
| 3 | Char n-gram + NCD features (§3.1-3.2) | Local | 3-4 hrs |
| 4 | Function word + vocab features (§3.3-3.4) | Local | 3-4 hrs |
| 5 | Baseline A1+A2 with fast features → **checkpoint on dev** | Local | 1-2 hrs |
| 6 | Implement unmasking (§4.1), test on 100 pairs locally | Local | 3-4 hrs |
| 7 | **Submit unmasking SLURM job (full train set)** | CSF | 2-4 hrs wall |
| 8 | Implement GI (§4.2), test locally | Local | 2-3 hrs |
| 9 | **Submit GI SLURM job (full train set)** | CSF | 1-2 hrs wall |
| 10 | Meanwhile: syntactic + surface + readability (§3.5-3.7) | Local | 3-4 hrs |
| 11 | Combine all 97 features, retrain | Local/CSF | 1-2 hrs |
| 12 | Optuna tuning (500 trials) | CSF GPU | 4-6 hrs wall |
| 13 | Train stacking meta-classifier | Local | 1-2 hrs |
| 14 | Ablation study (9 experiments) | CSF | 2-3 hrs |
| 15 | Bootstrap significance + error analysis + calibration | Local/CSF | 3-4 hrs |
| 16 | Build demo notebook, end-to-end test | Local | 2-3 hrs |
| 17 | Write model card | Local | 2 hrs |

**Total: ~35-45 hrs active work + ~10-15 hrs HPC wall time (background).**

**Critical path:** Steps 6-9 (HPC features). Submit SLURM jobs early, do local work in parallel.

---

## 13. CSF Tips

- **Storage:** `$HOME` for code/models, `/scratch/` for large feature arrays.
- **Modules:** Always `module load anaconda3/2024.02` in SLURM scripts.
- **GPU partition:** `--partition=gpu --gres=gpu:1` for LightGBM tuning.
- **CPU partition:** `--partition=multicore --cpus-per-task=32` for unmasking/impostor.
- **Monitoring:** `squeue -u $USER`, `sacct -j <JOBID>`.
- **Array jobs** for per-pair parallelism across nodes:
  ```bash
  #SBATCH --array=0-26   # 27 chunks of ~1000 pairs
  python compute_unmasking.py --chunk_id $SLURM_ARRAY_TASK_ID
  ```
