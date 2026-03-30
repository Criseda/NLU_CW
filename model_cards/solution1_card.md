# Model Card - Solution 1 (Stylometric Stacking Ensemble)

## 1. Model Details
- **Developer**: Group 17 (NLU Coursework Track C)
- **Model Version**: 1.0.0 (Release Candidate)
- **Model Type**: Stacking Ensemble Learner using a Logistic Regression meta-classifier over traditional stylometric and info-theoretic features.
- **Base Models**: Logistic Regression (ElasticNet) and LightGBM Classifier.
- **Features**: 97-dimensional vector (84 Stylometric, 8 Info-theoretic, 5 Generalised Impostor metrics).

## 2. Intended Use
- **Primary Use Case**: Pairwise Authorship Verification (AV) — determining if two English text sequences were written by the same individual.
- **Intended Users**: Computational Linguistics researchers and markers for the COMP34812 NLU Shared Task.
- **Out-of-Scope**: Cross-lingual authorship attribution or multi-author detection in a single document.

## 3. Factors
- **Demographics**: Stylometric patterns vary by age, region, and education level.
- **Environmental**: The platform of writing (e.g., social media vs. academic) significantly influences structural markers like punctuation and contraction usage.
- **Instrumentation**: Performance is sensitive to the tokenization strategy (spaCy provided).

## 4. Metrics
- **Macro F1 Score**: Primary metric for the shared task, balancing precision and recall across both 'same' and 'different' author classes.
- **Accuracy & AUC-ROC**: Secondary metrics to measure overall classification correctness and probability calibration.

## 5. Evaluation Data
- **Dataset**: Official NLU Authorship Verification Track (C) Development set.
- **Selection**: Composed of balanced pairs representing typical authorship verification challenges.

## 6. Training Data
- **Dataset**: Official NLU Authorship Verification Track (C) Training set.
- **Preprocessing**: Minimal NFKC normalization to preserve stylometric signals (punctuation, casing, and spacing).

## 7. Quantitative Analyses
- **Macro F1**: ~0.7083
- **Accuracy**: ~0.7084
- **AUC-ROC**: ~0.7932
- **Performance breakdown**: Higher confidence on long sequences (>200 words); increased variance on short snippets (<50 words).

## 8. Ethical Considerations
- **Sensitive Data**: The model is trained on textual features; no explicit PII (Personally Identifiable Information) is used as a feature, though writing style is inherently personal.
- **Risks**: Potential for deanonymization of anonymous texts or biased filtering of specific dialects.
- **Mitigation**: Focus on high-level structural features rather than individual word frequencies to reduce overfitting to specific users.

## 9. Caveats and Recommendations
- **Stylometric Noise**: Stylometric signals become noisy in extremely short texts.
- **Recommendation**: For optimal results, ensure target texts are at least 100 tokens long.
- **Inference**: Computationally lightweight; suitable for CPU-only environments.

---

### Model Links
- **Corpus Vectors**: [View on Google Drive](https://drive.google.com/file/d/1JFsQpv3RCfa0HEcwK4zz4gAVrWxDWUrg/view?usp=sharing)
- **Model A2 (LightGBM)**: [View on Google Drive](https://drive.google.com/file/d/1I9Cy6XddLRY8TUcs0VMh895WmySVKj45/view?usp=sharing)
