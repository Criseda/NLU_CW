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

This model is a traditional machine learning stacking ensemble designed for the COMP34812 NLU Shared Task (Authorship Verification). It combines stylometric, info-theoretic, and general impostor features across 97 dimensions to determine if two text sequences were written by the same author.

## Model Details

### Model Description

The model employs a two-tier stacking architecture. The base layer consists of a Logistic Regression (ElasticNet) and a LightGBM Classifier trained on a rich set of 97 hand-engineered features. A Logistic Regression meta-classifier then optimally blends these predictions.

- **Developed by:** Group 17 (NLU Coursework Track C)
- **Model type:** Stacking Ensemble (Traditional ML)
- **Language(s) (NLP):** English
- **License:** Individual academic use for COMP34812 coursework.
- **Finetuned from model:** N/A (Traditional feature-based ML)

### Model Sources

- **Model A2 (LightGBM):** [View on Google Drive](https://drive.google.com/file/d/1I9Cy6XddLRY8TUcs0VMh895WmySVKj45/view?usp=sharing)
- **Corpus Vectors (GI):** [View on Google Drive](https://drive.google.com/file/d/1JFsQpv3RCfa0HEcwK4zz4gAVrWxDWUrg/view?usp=sharing)

## Uses

### Direct Use

The primary use case is pairwise Authorship Verification (AV): determining if two English text sequences were written by the same individual. It is intended for use in computational linguistics research and evaluation within the COMP34812 NLU Shared Task.

### Out-of-Scope Use

This model is not intended for cross-lingual authorship attribution or multi-author detection in a single document. It may not generalize well to extremely short snippets (<50 words) where stylometric signals are sparse.

## Bias, Risks, and Limitations

Stylometric patterns vary significantly by age, region, and education level. The platform of writing (e.g., social media vs. academic) also influences structural markers.

### Recommendations

Users should be aware that stylometric signals become noisy in extremely short texts. For optimal results, ensure target texts are at least 100 tokens long.

## How to Get Started with the Model

Refer to the demo notebook `notebooks/demo_solution1.ipynb` for instructions on loading the pre-computed feature matrix and running inference via the stacking weights.

## Training Details

### Training Data

The model was trained on the **Official NLU Authorship Verification Track (C) Training set**. 

### Training Procedure

#### Preprocessing

Minimal NFKC normalization was applied to preserve stylometric signals such as punctuation, casing, and spacing.

#### Training Hyperparameters

- **Base Models:** Logistic Regression (ElasticNet) and LightGBM Classifier.
- **Meta-Classifier:** Logistic Regression.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on the **Official NLU Authorship Verification Track (C) Development set**.

#### Factors

Evaluation disaggregated by sequence length, observing higher confidence on sequences >200 words.

#### Metrics

- **Macro F1 Score:** Primary metric.
- **Accuracy & AUC-ROC:** Secondary metrics.

### Results

| Metric | Value |
| :--- | :--- |
| **Macro F1** | ~0.7083 |
| **Accuracy** | ~0.7084 |
| **AUC-ROC** | ~0.7932 |

#### Summary

The model delivers robust performance on traditional structural markers and is computationally lightweight, suitable for CPU-only environments.

## Ethical Considerations

- **Sensitive Data:** The model uses structural features; no explicit PII is extracted.
- **Risks:** Potential for deanonymization of anonymous texts.
- **Mitigation:** Focus on high-level structural features rather than individual word frequencies.

## Technical Specifications

### Compute Infrastructure

#### Software

- Python 3.14
- Scikit-learn
- LightGBM
- joblib

## Model Card Authors

- Group 17

## Model Card Contact

For coursework-related inquiries, contact the Group 17 owners.
