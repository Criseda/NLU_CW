# Model Card: Classical Stylometry (Solution 1)

This card tracks the development of the Classical Machine Learning pipeline for Authorship Verification (Track C / Category C).

## Approach Overview
Solution 1 focuses on interpretable, linguistically-grounded features rather than deep neural representations. We extract a high-dimensional feature vector $\mathbf{x}$ for each text pair and compute the absolute difference $|x_{T1} - x_{T2}|$ as input to a classifier.

### Key Features
*   **Lexical Diversity:** Yule's K, Simpson's D, and Type-Token Ratio (TTR).
*   **Frequency Analysis:** Top 100 function words and punctuation mark distribution.
*   **Syntactic Complexity:** Average sentence length and word length variance.

### Classifier Configuration
*   **Algorithm:** Gradient Boosted Trees (XGBoost)
*   **Hyperparameters:** `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`.
*   **Validation:** 5-fold cross-validation on the AV training set.

## Evaluation Results

| Metric         | Result |
| :------------- | :----- |
| Accuracy       | 0.7612 |
| Macro F1 Score | 0.7521 |
| Macro Precision| 0.7580 |
| Macro Recall   | 0.7490 |

## Implementation
The feature extraction logic is contained in `src.solution1`, utilizing native Python string processing and `scikit-learn` for classification.
