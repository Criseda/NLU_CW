# Model Card for Solution 1 (Extracted Features + Stacking)

## Model Details
- **Architecture**: Stacking Ensemble Learner over extracted Stylometric and Info-theoretic features.
  - **Feature Extractor**: Concatenates 97 numerical features extracted from each text pair.
    - 84 Stylometric features (char-ngrams, function words, syntactics, etc.)
    - 8 Information-theoretic metrics (replacing traditional unmasking)
    - 5 Generalised Impostor method metrics
  - **Base Models**: 
    - A1: Logistic Regression with ElasticNet penalty (SAGA solver, L1 ratio 0.5)
    - A2: LightGBM Classifier (n_estimators=2244, depth=12)
  - **Meta-Classifier**: Logistic Regression (blends predictions from A1 and A2)

## Training Data
Trained on the official NLU Authorship Verification (AV) track train split.

## Performance
Evaluated on the official dev set split during hyperparameter tuning:
- **Macro F1**: ~0.7083
- **Accuracy**: ~0.7084
- **AUC-ROC**: ~0.7932

## Intended Use
Used to generate predictions for whether two given texts were written by the same author. The computationally lightweight nature of this architecture ensures inference can execute on standard CPU architecture without requiring heavy GPU inference.

## Limitations
Dependent on the robustness of exact feature metrics; performance can degrade on very short texts where stylometric signals (like function word distribution) become noisy or sparse.
