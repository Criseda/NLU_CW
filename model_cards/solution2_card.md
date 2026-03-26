# Model Card: Solution 2 (Small Bi-Encoder)

This card tracks the development and evaluation of the small Bi-Encoder model (`all-MiniLM-L6-v2`) for Authorship Verification.

## Experiment Leaderboard

We are currently running a hyperparameter sweep to optimize F1 score.

| Rank  | Run Name                | Best F1    | Learning Rate | Weight Decay | Max Length | Notes                       |
| :---- | :---------------------- | :--------- | :------------ | :----------- | :--------- | :-------------------------- |
| **🏆** | **Baseline**            | **0.7734** | 2e-5          | 0.01         | 256        | Original `train.py` run     |
| 🥈     | `robust_wd01`           | **0.7723** | 2e-5          | 0.1          | 256        | Higher regularization       |
| 🥉     | `long_context_512`      | **0.7579** | 1e-5          | 0.05         | 512        | Full context window         |
| 4th   | `patient_finetuner_8ep` | **0.7275** | 5e-6          | 0.05         | 256        | Finished slowest learner    |

## Hyperparameter Sweep Strategy

The objective is to find a configuration that achieves higher F1 and AUC than the baseline while reducing overfitting.

2. **Context:** Increasing `MAX_LENGTH` to 512 to ensure no authorship markers are truncated.
3. **Patience:** Slower learning rates (5e-6) with more epochs (8) for finer tuning.
4. **Stability:** Larger batch size (32) for smoother gradient updates.

## Model Details
- **Base Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture:** Bi-Encoder with Cosine Similarity loss (CrossEntropy on pairs)
- **Training Device:** Apple MPS (Metal Performance Shaders)

## Project Integration
This Bi-Encoder serves as a critical diversity signal in our **0.8615 F1 ensemble**, providing global structural representations that complement the token-level focus of the larger Cross-Encoders. 
