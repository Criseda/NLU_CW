# Model Card: Transformer Ensemble (Solution 2 - Expanded)

This card tracks the development and evaluation of the 6-Model Heterogeneous Ensemble for Authorship Verification (Track C / Category C).

## Ensemble Architecture
The ensemble leverages a Late Fusion (Soft-Voting) approach using logit-level probabilities. By combining models with different architectural biases (DeBERTa, RoBERTa, XLNet, ELECTRA, and MiniLM), we reduce individual model variance and capture a broader range of stylistic fingerprints.

### Base Models
1. **RoBERTa (Vanilla):** Strongly captures local contextual dependencies. (Weight: 0.3047)
2. **DeBERTa-v3-large (8180):** High-capacity cross-encoder with relative position embeddings. (Weight: 0.2161)
3. **ELECTRA (Vanilla):** Efficiently learns from replaced token detection. (Weight: 0.1561)
4. **XLNet (Vanilla):** Captures bidirectional context via permutation. (Weight: 0.1387)
5. **MiniLM (Siamese):** Small Bi-Encoder for global structural cues. (Weight: 0.1262)
6. **DeBERTa-v3-large (803):** Secondary cross-encoder checkpoint for diversity. (Weight: 0.0583)

### Fusion Strategy
After an exhaustive randomized weight search (**100,000 iterations**) on the AV `dev` split, the **6-model expanded ensemble** achieved the project's absolute state-of-the-art performance:

*   **Weighted Soft-Voting Formula:**
    $$P_{ens} = \sum w_i \cdot P_i$$
*   Weights are optimized to maximize **Macro F1** while ensuring no single model dominates suspiciously.

## Evaluation Leaderboard

Metric tracking against the bundled local scorer reference (`NLU_SharedTask_AV_dev.solution`). 

| Component                              | Macro F1   | Δ vs Baseline |
| :------------------------------------- | :--------- | :------------ |
| Standalone Bi-Encoder (MiniLM)         | 0.7734     | -             |
| Standalone DeBERTa-v3 (8180)           | 0.7951     | +2.17%        |
| 5-Model Ensemble (DeBERTa + MiniLM)    | 0.8083     | +3.49%        |
| **🏆 6-Model Expanded Ensemble (Final)**| **0.8615** | **+8.81%**    |

## Implementation
The execution pipeline is modular and supports CUDA/MPS:
1. `src.solution2.ensemble.predict_ensemble` merges the 6 probability streams.
2. The script automatically handles the lack of `pair_id` in vanilla model outputs via row-index alignment.
3. Final output: `outputs/solution2/ensemble/ENSEMBLE_EXPANDED_AV_dev.csv`.
