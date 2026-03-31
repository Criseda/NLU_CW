---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
language:
- en
license: other
tags:
- authorship-verification
- stylometry
- stacking-ensemble
- nlu
model_name: Solution 1 (Stylometric Stacking Ensemble)
---

# Model Card for Solution 1 (Stylometric Stacking Ensemble)

This model is a traditional machine learning stacking ensemble designed for the COMP34812 NLU Shared Task (Authorship Verification). It combines stylometric, info-theoretic, and General Impostor features across 97 dimensions to determine if two text sequences were written by the same author.

## Model Details

### Model Description

The model employs a two-tier stacking architecture. The base layer consists of a Logistic Regression (ElasticNet) and a LightGBM Classifier trained on a rich set of 97 hand-engineered features. A Logistic Regression meta-classifier then optimally blends these predictions.

- **Developed by:** Group 17 (NLU Coursework Track C)
- **Model type:** Stacking Ensemble (Traditional ML — Category A)
- **Language(s) (NLP):** English
- **License:** Individual academic use for COMP34812 coursework.
- **Finetuned from model:** N/A (Traditional feature-based ML)

### Model Sources

- **Model A2 (LightGBM):** [Google Drive](https://drive.google.com/file/d/1rWwn__rn9BwFgdaxINDbklYOCgHquY1z/view?usp=sharing)
- **Corpus Vectors (GI):** [Google Drive](https://drive.google.com/file/d/193TCG1-I4zSeW2t0peWCi3hxAPZ2FIeX/view?usp=sharing)

## Uses

### Direct Use

The primary use case is pairwise Authorship Verification (AV): determining if two English text sequences were written by the same individual. It is intended for use in computational linguistics research and evaluation within the COMP34812 NLU Shared Task.

### Out-of-Scope Use

This model is not intended for cross-lingual authorship attribution or multi-author detection in a single document. It may not generalise well to extremely short snippets (<50 words) where stylometric signals are sparse.

## Bias, Risks, and Limitations

Stylometric patterns vary significantly by age, region, and education level. The platform of writing (e.g., social media vs. academic) also influences structural markers.

### Recommendations

Users should be aware that stylometric signals become noisy in extremely short texts. For optimal results, ensure target texts are at least 100 tokens long.

## How to Get Started with the Model

Refer to the demo notebook `notebooks/demo_solution1.ipynb` for end-to-end instructions. The notebook automatically downloads large artefacts via `gdown` and runs the full inference pipeline from a raw CSV.

## Training Details

### Training Data

The model was trained on the **Official NLU Authorship Verification Track (C) Training set** (27,643 sequence pairs).

### Training Procedure

#### Preprocessing

Minimal NFKC normalisation was applied to preserve stylometric signals such as punctuation, casing, and spacing. Texts were parsed with spaCy (`en_core_web_sm`) to extract syntactic and POS-based features.

#### Feature Engineering (97 dimensions)

| Group | Features | Dimensions |
| :--- | :--- | :---: |
| Character N-grams (TF-IDF cosine) | Char 2–4-gram similarity | 20 |
| Compression (NCD) | Normalised compression distance variants | 6 |
| Function words | Frequency distributions of closed-class words | 20 |
| Vocabulary richness | TTR, MSTTR, Yule's K, Hapax ratio, etc. | 8 |
| Syntactic (POS) | POS tag distribution differences | 13 |
| Surface stylometrics | Avg sentence/word length, punctuation rate, etc. | 10 |
| Readability | Flesch, Gunning Fog, SMOG, etc. | 7 |
| Info-theoretic | Cross-entropy, KL divergence, Rényi entropy variants | 8 |
| General Impostor (GI) | Koppel & Winter (2014) impostor method | 5 |
| **Total** | | **97** |

#### Training Hyperparameters

- **Base Model A1:** Logistic Regression with ElasticNet regularisation (`C=1.0, l1_ratio=0.5`)
- **Base Model A2:** LightGBM Classifier (`n_estimators=500, learning_rate=0.05, num_leaves=63`)
- **Meta-Classifier:** Logistic Regression trained on OOF probability outputs of A1 and A2
- **Cross-validation:** 5-fold stratified for OOF generation

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on the **Official NLU Authorship Verification Track (C) Development set** (5,993 pairs, 51.0% positive).

#### Metrics

- **Macro F1 Score:** Primary metric
- **Accuracy, Balanced Accuracy, AUC-ROC:** Secondary metrics
- **Brier Score:** Calibration quality
- **Bootstrap 95% CI:** 1,000-resample non-parametric bootstrap for statistical robustness

### Results

#### Core Metrics on Dev Set

| Metric | Score | 95% CI Lower | 95% CI Upper |
| :--- | :---: | :---: | :---: |
| **Macro F1** | **0.7083** | 0.6975 | 0.7202 |
| Accuracy | 0.7085 | 0.6978 | 0.7203 |
| Balanced Accuracy | 0.7092 | 0.6982 | 0.7213 |
| Weighted F1 | 0.7082 | 0.6975 | 0.7200 |
| Precision | 0.7335 | 0.7161 | 0.7497 |
| Recall | 0.6728 | 0.6570 | 0.6891 |
| AUC-ROC | 0.7932 | 0.7827 | 0.8041 |
| Brier Score | 0.1859 | 0.1812 | 0.1904 |

The tight bootstrap CIs (F1 bounded between 69.7%–72.0%) confirm the model's performance is statistically stable and not attributable to a favourable data split.

#### Confusion Matrix

| | Predicted Positive | Predicted Negative |
| :--- | :---: | :---: |
| **Actual Positive** | 2,056 (TP) | 1,000 (FN) |
| **Actual Negative** | 747 (FP) | 2,190 (TN) |

#### Base Model Comparison

The stacking meta-classifier improves over each base model individually:

| Model | Accuracy | Macro F1 | AUC-ROC |
| :--- | :---: | :---: | :---: |
| A1 — Logistic Regression (ElasticNet) | 0.6277 | 0.6247 | 0.6887 |
| A2 — LightGBM | 0.7077 | 0.7074 | 0.7935 |
| **Stacked Meta (Final)** | **0.7085** | **0.7083** | **0.7932** |

#### Calibration

Expected Calibration Error (ECE) = **0.031** — indicating the model's probability outputs are well-calibrated and reliably reflect true confidence levels.

### Ablation Study

A 5-fold stratified cross-validation ablation was run on the 27,643-pair training set to quantify each feature group's contribution in isolation (OOF Macro F1):

| Feature Subset | N Features | OOF Macro F1 |
| :--- | :---: | :---: |
| All 97 features | 97 | **0.6936** |
| Stylometric only (84) | 84 | 0.6832 |
| HPC features only (13) | 13 | 0.6249 |
| — Character N-grams | 20 | 0.6406 |
| — Vocabulary richness | 8 | 0.6163 |
| — Compression (NCD) | 6 | 0.5987 |
| — Syntactic (POS) | 13 | 0.5964 |
| — Info-theoretic | 8 | 0.5876 |
| — General Impostor (GI) | 5 | 0.5763 |
| — Surface stylometrics | 10 | 0.5721 |
| — Readability | 7 | 0.5716 |
| — Function words | 20 | 0.5714 |

**Key findings:**

- Removing the HPC features (GI + info-theoretic) causes a drop of >1.0% F1, justifying the compute overhead.
- Character N-grams are the strongest standalone signal (0.641 F1 alone), but individual feature groups plateau well below the full ensemble.
- The stacking architecture enables complementary features to combine symbiotically, driving the collective F1 from ~0.64 to ~0.708.

### Evaluation Scripts

Evaluation was conducted using `src/solution1/training/evaluate.py`, which produces:

- Full metric suite with 95% bootstrap confidence intervals (1,000 resamples)
- Per-feature-group ablation study (5-fold stratified CV)
- Calibration curve and ECE
- Confusion matrix
- JSON results summary

## Ethical Considerations

- **Sensitive Data:** The model uses structural features; no explicit PII is extracted.
- **Risks:** Potential for deanonymisation of anonymous texts.
- **Mitigation:** Focus on high-level structural features rather than individual word frequencies.

## Technical Specifications

### Compute Infrastructure

#### Software

- Python 3.10+
- Scikit-learn
- LightGBM
- spaCy (`en_core_web_sm`)
- joblib

## Model Card Authors

- Group 17

## Model Card Contact

For coursework-related enquiries, contact the Group 17 owners.
