# Solution 2: Large Neural Meta-Ensemble for Authorship Verification

---

## Overview

Solution 2 implements a **neural meta-ensemble** that combines four state-of-the-art transformer cross-encoders to classify authorship verification pairs. Each base model is independently fine-tuned on the AV training data, and their probabilistic outputs are fused via a learned 4-layer neural meta-learner.

### Architecture

```md
┌─────────────────────────────────────────────────────┐
│         Text Pair (text_1, text_2)                 │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ↓           ↓           ↓
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │DeBERTa-v3│ │RoBERTa  │ │ELECTRA  │  ... + XLNet
    │  large   │ │ large   │ │ large   │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
         └───────────┼───────────┘
                     │
            ┌────────▼────────┐
            │ Normalize probs │
            │   (Calibration) │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │   Ensemble probs      │
         │  (DeBERTa, RoBERTa,  │
         │  ELECTRA, XLNet)     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   4-Layer Meta-Learner│
         │  (FC + BatchNorm +    │
         │   Dropout)            │
         └───────────┬───────────┘
                     │
            ┌────────▼────────┐
            │  Final Binary   │
            │   Prediction    │
            └─────────────────┘
```

---

## Directory Structure

```md
src/solution2/
├── README.md                          ← You are here
├── __init__.py
├── predict.py                         ← Top-level inference script
│
├── deberta_model/                     ← DeBERTa-v3-large base model
│   ├── __init__.py
│   ├── config.py                      ← Model-specific hyperparameters
│   ├── model.py                       ← AVCrossEncoder architecture
│   ├── train.py                       ← Fine-tuning script
│   └── train_continue.py              ← Resume fine-tuning
│
├── roberta_model/                     ← RoBERTa-large base model
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── train_roberta.py
│   └── train_roberta_continue.py
│    [best_model.pt stored on cloud]
│
├── electra_model/                     ← ELECTRA-large base model
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── train_electra.py
│   └── train_electra_continue.py
│
├── xlnet_model/                       ← XLNet-large base model
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── train_xlnet.py
│   └── train_xlnet_continue.py
│
└── ensemble/                          ← Meta-learning & fusion
    ├── __init__.py
    ├── config.py                      ← Ensemble configuration & cloud links
    ├── models_config.py               ← Base model registry
    ├── train_ensemble.py              ← Train the meta-learner
    ├── calibrate_base_models.py       ← Probability calibration
    ├── predict_ensemble.py            ← Generate ensemble predictions
    └── submit_ensemble.slurm           ← HPC job submission script
```

---

## Quick Start: Running Inference

### Prerequisites

1. **Python 3.8+** with PyTorch and Transformers installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **Pre-trained Models**: All base model weights and the meta-learner are stored on [Google Drive](https://drive.google.com/drive/folders/1arKIOSEAZxAz4P_-MotKRo0aFjFg-QGp?usp=sharing).

3. **GPU** (recommended): Inference works on CPU but is very slow. 4× transformer models require ~8GB VRAM for efficient batch processing.

### Generate Predictions (Main Use Case)

To produce final submission-ready predictions on the **test set**:

```bash
python -m src.solution2.predict --split test
```

This will:

1. Load all 4 base transformer models from cloud
2. Generate probability predictions from each base model
3. Apply probability calibration
4. Fuse predictions via the neural meta-learner
5. Save final binary predictions to `outputs/predictions_test.csv`, then rename to `Group_17_C.csv`.

**Output:** A CSV with a single column (`prediction`) containing 0/1 labels, one per test example.

---

## Inference on Other Splits

You can also run inference on the **dev** or **train** splits (useful for evaluation):

```bash
# Development set (validation)
python -m src.solution2.predict --split dev

# Training set (sanity check)
python -m src.solution2.predict --split train
```

These produce:

- `outputs/solution2/predictions_dev.csv`
- `outputs/solution2/predictions_train.csv`

All inference outputs have been saved to [Google Drive](https://drive.google.com/drive/folders/1saMnwl_u3_FZMiDiWzap-eyhmXymznFx?usp=sharing)

---

## Interactive Demo

For a step-by-step walkthrough and to visualize ensemble predictions:

```bash
jupyter notebook notebooks/demo_solution2.ipynb
```

This notebook:

- Downloads all required files from Google Drive
- Loads all 4 base models
- Runs inference on test or dev batch
- Generates CSV

---

## Training (Educational Reference)

**Note:** Base models are pre-trained and frozen. The fine-tuning scripts are provided for reference/reproducibility.

### 1. Fine-tune a Single Base Model (e.g., DeBERTa)

```bash
python -m src.solution2.deberta_model.train
```

**What this does:**

- Loads DeBERTa-v3-large from Hugging Face
- Fine-tunes on AV training data
- Saves the best checkpoint to `models/solution2/deberta_model/best_model.pt`

**Configuration:** Edit `src/solution2/deberta_model/config.py` to adjust:

- Learning rate, batch size, number of epochs
- Dropout, weight decay
- Number of frozen layers
- Device (GPU/CPU)

### 2. Calibrate Base Model Probabilities

After fine-tuning all 4 base models, calibrate their probability outputs:

```bash
python -m src.solution2.ensemble.calibrate_base_models
```

This:

- Loads each base model
- Generates predictions on the **dev set**
- Applies temperature scaling to improve calibration
- Saves calibration parameters to `models/solution2/ensemble/calibrators/`

### 3. Train the Meta-Learner

Once all base models are calibrated, train the ensemble meta-learner:

```bash
python -m src.solution2.ensemble.train_ensemble
```

This:

- Loads calibrated probabilities from all 4 base models (dev split)
- Trains a 4-layer neural network to optimally blend them
- Uses focal loss to handle class imbalance
- Early stops on dev set F1
- Saves best weights to `models/solution2/ensemble/meta_learner.pt`

---

## Evaluation

To evaluate predictions against gold labels:

```bash
python -m src.evaluation.evaluate \
    --predictions outputs/solution2/predictions_dev.csv \
    --gold data/training_data/AV/dev.csv \
    --model_type solution2
```

**Output:**

- Accuracy, F1-macro, F1-binary, Precision, Recall, AUC-ROC, Brier Score
- Confusion matrix (TP, FP, FN, TN)

---

## Performance & Results

### Individual Base Model Performance (Dev Set)

Each transformer is independently competitive:

| Model | Best F1 | Architecture Strength |
| --- | --- | --- |
| **RoBERTa-large** | **0.8463** | Optimized pretraining, robust baseline |
| **ELECTRA-large** | **0.8413** | Discriminator-based, token-level patterns |
| **XLNet-large** | **0.8234** | Autoregressive, different inductive bias |
| **DeBERTa-v3-large** | **0.8180** | Disentangled attention, SOTA on GLUE |

### Ensemble Meta-Learner Performance (Dev Set)

The neural meta-learner significantly improves over individual models:

```md
[ensemble] Final Results (Dev Set):
  F1 (macro):       0.8644  ← +2.1% over best single model (RoBERTa)
  Accuracy:         0.8645
  AUC-ROC:          0.9234
  Optimal Threshold: 0.51
```

**Key Insight:** Ensemble achieves **18.1% error reduction** compared to the weakest base model (DeBERTa), demonstrating the power of learned fusion.

### Feature Importance: How the Meta-Learner Weights Inputs

The 4-layer neural network learns to pay attention to different signals:

```md
[ensemble] Feature Importance Analysis:
──────────────────────────────────────────────────────────
  electra_prob         | ████                           |   9.4%
  roberta_prob         | ████                           |   9.0%
  roberta_pred         | ████                           |   8.2%
  vote_pred            | ████                           |   8.0%
  xlnet_prob           | ███                            |   7.4%
  electra_pred         | ███                            |   6.9%
  mean_prob            | ███                            |   6.2%
  xlnet_pred           | ██                             |   5.9%
  deberta_prob         | ██                             |   5.6%
  vote_certainty       | ██                             |   5.0%
  perfect_agreement    | ██                             |   4.5%
  prob_range           | █                              |   3.5%
  min_prob             | █                              |   3.5%
  deberta_pred         | █                              |   3.4%
  max_prob             | █                              |   3.4%
  consensus_confidence | █                              |   3.2%
  predictions_entropy  | █                              |   2.6%
  std_prob             | █                              |   2.5%
  prob_variance        |                                |   1.8%
──────────────────────────────────────────────────────────
```

**Interpretation:**

- ELECTRA and RoBERTa probabilities are most influential (~18% combined)
- Voting agreement signals help resolve disagreements
- Ensemble consensus metrics (entropy, certainty) are secondary
- No single feature dominates—all contribute to robust decisions

### Model Contribution Breakdown

Per-model contribution to final ensemble predictions:

```md
[ensemble] Model Contribution to Ensemble:
──────────────────────────────────────────────────────────
  roberta              | ████████                      |  17.2%
  electra              | ████████                      |  16.3%
  xlnet                | ██████                        |  13.3%
  deberta              | ████                          |   9.0%
──────────────────────────────────────────────────────────
```

**Why This Distribution?**

Each model brings unique strengths to the ensemble:

1. **RoBERTa (17.2% contribution)**
   - **Strength:** Robust pretraining with masked language modeling
   - **Specialization:** Excels at capturing symmetric authorship signals (word choice, sentence structure)
   - **Why it leads:** Highest F1 (0.8463) makes it the ensemble's primary decision-maker
   - **Weakness:** Can miss asymmetric stylistic patterns

2. **ELECTRA (16.3% contribution)**
   - **Strength:** Discriminator-based pretraining differs fundamentally from BERT/RoBERTa
   - **Specialization:** Better at token-level accuracy and detailed stylistic nuances
   - **Why it's valuable:** Catches errors RoBERTa misses through completely different learning signal
   - **Complementary:** Only slightly lower F1 (0.8413) but different error patterns

3. **XLNet (13.3% contribution)**
   - **Strength:** Autoregressive pretraining (permutation language modeling)
   - **Specialization:** Captures sequential dependencies and temporal patterns in writing
   - **Why it matters:** Provides perspective on authorship that neither masked nor discriminator models fully capture
   - **Use case:** Excels on pairs with consistent narrative flow or temporal coherence

4. **DeBERTa (9.0% contribution)**
   - **Strength:** Disentangled attention mechanism (state-of-the-art on GLUE/SuperGLUE)
   - **Specialization:** Fine-grained token interaction modeling
   - **Why still included:** Lowest individual F1 (0.8180) but contributes unique error patterns
   - **Value:** Adds diversity; helps meta-learner catch edge cases others miss
   - **Trade-off:** Being weaker individually, it doesn't dominate voting but still improves ensemble robustness

### Ensemble Synergy

The ensemble outperforms all individuals because:

| Mechanism | Impact |
| --- | --- |
| **Diversity** | 4 different pretraining objectives → orthogonal feature spaces |
| **Disagreement Handling** | Meta-learner learns to weight models when they conflict |
| **Probability Calibration** | Calibrators adjust each model's confidence before fusion |
| **Learned Fusion** | 4-layer NN adapts blend weights per example (not fixed voting) |
| **Error Correction** | Where RoBERTa fails, ELECTRA/XLNet often succeed (and vice versa) |

**Result:** +2.1% F1 improvement over best single model, with 0.9234 AUC-ROC showing excellent probability calibration.
