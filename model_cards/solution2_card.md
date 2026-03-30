# Model Card - Solution 2 (Large Neural Meta-Ensemble)

## 1. Model Details
- **Developer**: Group 17 (NLU Coursework Track C)
- **Model Version**: 1.0.0 (Release Candidate)
- **Model Type**: Neural Meta-Learner trained to optimally blend the probabilistic outputs of four fine-tuned transformer cross-encoders.
- **Base Models**: DeBERTa-v3-large, RoBERTa-large, ELECTRA-large-discriminator, XLNet-large-cased.
- **Ensemble Meta-Learner**: 4-layer fully-connected neural network with Batch Normalization and Dropout.

## 2. Intended Use
- **Primary Use Case**: State-of-the-Art Pairwise Authorship Verification (AV) where deep semantic understanding is prioritized over structural stylometrics.
- **Intended Users**: NLP researchers and markers for the COMP34812 NLU Shared Task.
- **Out-of-Scope**: Real-time inference on mobile or low-power devices due to high parameter count (avg 335M+ per head).

## 3. Factors
- **Linguistic**: Performance depends on the context length (max 512 tokens).
- **Computational**: Requires significant VRAM (GPU) for efficient inference.
- **Domain**: Fine-tuned on the Authorship Verification track datasets, which represent varied writing styles.

## 4. Metrics
- **Macro F1 Score**: Primary metric.
- **Accuracy**: Secondary metric.
- **Calibration Error**: Evaluated to ensure probabilities realistically represent authorship confidence.

## 5. Evaluation Data
- **Dataset**: Official NLU Authorship Verification Track (C) Development set.
- **Benchmarking**: Used to optimize the weighted meta-ensemble and focal loss parameters.

## 6. Training Data
- **Dataset**: Official NLU Authorship Verification Track (C) Training set.
- **Fine-tuning**: Transformer weights fine-tuned with `BCEWithLogitsLoss`, AdamW optimizer, and early stopping.

## 7. Quantitative Analyses
- **General Performance**: Delivers strong classification capability on semantically complex pairs.
- **Resilience**: Effective at correcting base model disagreements through the 4-layer meta-learner.
- **Metric Breakdown**: Consistently achieves competitive performance within the shared task constraints using state-of-the-art representations.

## 8. Ethical Considerations
- **Environmental**: Training 4 large transformer architectures has a non-negligible carbon footprint.
- **Algorithm Bias**: High risk of inheriting biases from the massive pre-training corpora of the base transformer models.
- **Human Impact**: Like all AV systems, this could impact privacy if used to deanonymize private individuals without consent.

## 9. Caveats and Recommendations
- **Inference Latency**: Extremely heavy; requires 4 separate transformer forward passes per prediction. 
- **Recommendation**: Use concurrent inference or high-end GPUs (A100/V100) for batch processing.
- **Resource Note**: Not suitable for CPU-only inference in production.

---

### Model Links
- **XLNet Base**: [View on Google Drive](https://drive.google.com/file/d/1HWnOv36f-VOe92tKl2saQ1xs4k7hwbty/view?usp=sharing)
- **RoBERTa Base**: [View on Google Drive](https://drive.google.com/file/d/15M0kqGWtfLrO5F2mB3O4MH3Yt44dZDTZ/view?usp=sharing)
- **ELECTRA Base**: [View on Google Drive](https://drive.google.com/file/d/1ce5tgkHNuZzLhpoata9FiBDsPAtxGPKe/view?usp=sharing)
- **DeBERTa Base**: [View on Google Drive](https://drive.google.com/file/d/1IfzDd87FARs4CE7rEviKcvI2UL9sKH4s/view?usp=sharing)
- **Ensemble Meta-Learner**: [View on Google Drive](https://drive.google.com/file/d/1jf9hXJtY0LCHW-dOiEBgo-SdnrZK0Hri/view?usp=sharing)
