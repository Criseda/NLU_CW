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

2. **Pre-trained Models**: All base model weights and the meta-learner are stored on Google Drive (see [Model Links](#model-links) below). These are automatically downloaded when running inference.

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
