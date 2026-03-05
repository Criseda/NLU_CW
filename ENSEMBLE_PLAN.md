# Authorship Verification (Track C) - Ensemble Strategy Plan

## Overview
Technical strategy and work breakdown for COMP34812 NLU Coursework. Track: **Track C (Authorship Verification)**.

Deliverables:
- **Solution 1 (Category A):** Classical ML Ensemble.
- **Solution 2 (Category C):** Transformer Ensemble.

Strategy: Model ensembling. Each member develops independent models, which are then fused to improve performance and meet the "creativity" marking criterion.

---

## Work Breakdown

### Phase 1: Base Model Development

#### Partner 1 (Classical ML Leader)
1. **Model A1 (Classical ML):** TF-IDF (word/char n-grams) + Linear Classifier (Logistic Regression / SVM).
2. **Model A2 (Classical ML):** Stylometric features (POS tags, readability) + Tree-based Classifier (Random Forest / XGBoost).
   - *Key tasks:* Preprocessing, feature engineering, n-gram optimization, regularisation tuning.

#### Partner 2 (Transformer Leader - High Compute)
1. **Model C1 (Transformers - Large):** Large pre-trained language model (e.g., RoBERTa-large or DeBERTa-v3-large).
   - *Key tasks:* Training via University shared cluster, fine-tuning, optimizing learning rate and batch size for large sequence pairs.

#### Partner 3 (Transformer Support - Local Compute)
1. **Model C2 (Transformers - Small/Efficient):** Smaller, efficient architecture (e.g., DistilBERT, MiniLM).
   - *Key tasks:* Local training and iteration, evaluating cross-attention mechanisms, and fast prototyping.

*Note: If time permits after Phase 1, Partners 2 and 3 will assist Partner 1 with Classical ML feature engineering and optimizations.*

---

## Phase 2: Ensembling

### Solution 1: Classical ML (Category A)
- **Inputs:** Outputs from Model A1 and Model A2.
- **Fusion:** Soft voting (averaging probabilities) or Stacking (training a meta-classifier like Logistic Regression on A1/A2 outputs).

### Solution 2: Transformers (Category C)
- **Inputs:** Logits from Model C1 and Model C2.
- **Fusion:** Soft voting (averaging continuous outputs before 0.5 thresholding).

---

## Phase 3: Deliverables

- **Demo Notebooks:**
  - `demo_solution1.ipynb`: Load A1/A2, extract features, apply fusion, output `Group_X_A.csv`.
  - `demo_solution2.ipynb`: Load C1/C2, tokenize, apply fusion, output `Group_X_C.csv`.
- **Model Cards:** Document the ensemble system, detailing both base models and the fusion method.
- **Poster:** Diagram illustrating the dual-pipeline ensemble architecture.

---

## Marking Criteria Alignment
- **Creativity (6 Marks):** Combining distinct feature pipelines (TF-IDF + Stylometry) and architectures demonstrates advanced methodology beyond standard baselines.
- **Competitive Performance (6 Marks):** Ensembling orthogonal models reduces variance and typically yields higher F1-scores.
- **Implementation:** Clean separation of base models allows parallel development and avoids merge conflicts.
