# Model Card: Transformer Ensemble (Solution 2)

This card tracks the development and evaluation of the Ensembled Transformer architecture for Authorship Verification (Track C / Category C).

## Ensemble Architecture
The ensemble leverages a Late Fusion (Soft-Voting) approach running entirely over probabilities rather than embeddings. Both models predict the probability of a "Same Author" classification, which are then fused via a weighted average.

### Base Models
1. **Small Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local F1: 0.7734)
2. **Big Model (Original):** `microsoft/deberta-v3-large` (Local F1: 0.7862)
3. **Big Model (7931):** `microsoft/deberta-v3-large` (Local F1: 0.7701)
4. **Big Model (803):** `microsoft/deberta-v3-large` (Local F1: 0.7932)
5. **Big Model (8180):** `microsoft/deberta-v3-large` (Local F1: 0.8183)

### Fusion Strategy
After extensive empirical tuning against the local AV `dev` split, a **5-model weighted soft-voting** mechanism was selected:
*   `Model 8180 Weight:` **0.50**
*   `Model 803 Weight:` **0.40**
*   `Model 7931 Weight:` **0.00**
*   `Original Big Weight:` **0.00**
*   `Small Model Weight:` **0.10**

## Evaluation Leaderboard

Metric tracking against the bundled local scorer reference (`25_DEV_NLI.csv` / `NLU_SharedTask_AV_dev.solution`). 

| Component                        | Accuracy | Macro Precision | Macro Recall | Macro F1   |
| :------------------------------- | :------- | :-------------- | :----------- | :--------- |
| **ENSEMBLE (2-Model Weighted)**  | 0.7922   | 0.7961          | 0.7910       | 0.7910     |
| **ENSEMBLE (4-Model Weighted)**  | 0.8012   | 0.8014          | 0.8009       | 0.8010     |
| **🏆 ENSEMBLE (5-Model, 8180 Opt)**| 0.8053   | 0.8117          | 0.8038       | **0.8037** |

## Implementation
The execution pipeline is entirely automated within the repository:
1. `src.solution2.bi_encoder.predict` runs inference using Apple MPS.
2. `src.solution2.big_model.predict` runs inference using Apple MPS / CUDA.
3. `src.solution2.ensemble.predict_ensemble` merges the dataframes and calculates the weighted thresholding.
4. `bash src/solution2/ensemble/run_pipeline.sh` strings these modules together seamlessly.
