---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
language:
- en
license: other
tags:
- authorship-verification
- transformer
- ensemble
- nlu
- deberta
- roberta
- xlnet
- electra
model_name: Solution 2 (Large Neural Meta-Ensemble)
---

# Model Card for Solution 2 (Large Neural Meta-Ensemble)

This model is a state-of-the-art neural meta-ensemble designed for the COMP34812 NLU Shared Task (Authorship Verification). It optimally blends the probabilistic outputs of four fine-tuned transformer cross-encoders.

## Model Details

### Model Description

The architecture uses a 4-layer fully-connected neural network with Batch Normalization and Dropout to ensemble the predictions from DeBERTa-v3-large, RoBERTa-large, ELECTRA-large-discriminator, and XLNet-large-cased.

- **Developed by:** Group 17 (NLU Coursework Track C)
- **Model type:** Deep Learning based (Transformer Neural Meta-Ensemble)
- **Language(s) (NLP):** English
- **License:** Individual academic use for COMP34812 coursework.
- **Finetuned from model:** DeBERTa-v3-large, RoBERTa-large, ELECTRA-large-discriminator, XLNet-large-cased.

### Model Sources

- **DeBERTa Base:** [View on Google Drive](https://drive.google.com/file/d/1IfzDd87FARs4CE7rEviKcvI2UL9sKH4s/view?usp=sharing)
- **XLNet Base:** [View on Google Drive](https://drive.google.com/file/d/1HWnOv36f-VOe92tKl2saQ1xs4k7hwbty/view?usp=sharing)
- **RoBERTa Base:** [View on Google Drive](https://drive.google.com/file/d/15M0kqGWtfLrO5F2mB3O4MH3Yt44dZDTZ/view?usp=sharing)
- **ELECTRA Base:** [View on Google Drive](https://drive.google.com/file/d/1ce5tgkHNuZzLhpoata9FiBDsPAtxGPKe/view?usp=sharing)
- **Ensemble Meta-Learner:** [View on Google Drive](https://drive.google.com/file/d/1jf9hXJtY0LCHW-dOiEBgo-SdnrZK0Hri/view?usp=sharing)

## Uses

### Direct Use

Pairwise Authorship Verification (AV) where deep semantic understanding is prioritized over structural stylometrics.

### Out-of-Scope Use

Real-time inference on mobile or low-power devices due to high parameter count (avg 335M+ per head).

## Bias, Risks, and Limitations

Performance depends on the context length (max 512 tokens). There is an inherent risk of inheriting biases from the massive pre-training corpora of the base models.

### Recommendations

For optimal results, ensure target texts are at least 100 tokens long. Requires significant VRAM for efficient inference.

### Known Limitations

- **Context length:** Limited to 512 subword tokens (sequences longer than this are truncated)
- **Text type:** Optimized for prose; may struggle with code, highly structured text, or non-standard English
- **Language:** English only (trained on English authorship data)
- **Bias inheritance:** Model inherits biases from large pre-training corpora (primarily English web text)
- **Out-of-domain performance:** Limited evaluation on text types significantly different from training distribution

## How to Get Started with the Model

Refer to the demo notebook `notebooks/demo_solution2.ipynb` for instructions on loading the fine-tuned base models and the neural meta-ensemble.

### Code Example

```python
import torch
from pathlib import Path
from src.solution2.ensemble.predict_ensemble import predict_probs

# Load and run inference
results_df = predict_probs(split='dev')
predictions = results_df[['prediction']].copy()
predictions.to_csv('my_predictions.csv', index=False)
```

### Input/Output Specifications

**Input CSV format (for custom data):**

- Columns: `text_1`, `text_2`, `label` (label can be 0 or 1)
- Max length: 512 tokens per text (enforced by tokenizer)
- Minimum recommended: 100 tokens per text pair for optimal performance

**Output CSV format:**

- Single column: `prediction` (binary: 0 or 1)
- Optional: `probability` (float 0-1, confidence score)

### Inference Speed

- **Per-pair latency (L40S, batch_size=32):** ~50-100ms
- **Per-pair latency (CPU, batch_size=1):** ~10-15 seconds
- **Memory footprint:** 8GB VRAM for all 4 base models + meta-learner

### Reproducibility

- **Random seed:** 42
- **PyTorch version:** 2.0+
- **Transformers version:** 4.30+
- **Scikit-learn version:** 1.3+

### Training Code Examples

**1. Fine-tune a Single Base Model (e.g., DeBERTa):**

```bash
python -m src.solution2.deberta_model.train
```

Or programmatically:

```python
from src.solution2.deberta_model.train import main
from src.solution2.deberta_model import config

# Customize config if needed
config.LEARNING_RATE = 2e-5
config.EPOCHS = 4
config.BATCH_SIZE = 32

main()
```

**2. Calibrate All Base Models:**

```bash
python -m src.solution2.ensemble.calibrate_base_models
```

This generates calibrated probability predictions from all 4 base models on the dev set and saves calibrators.

**3. Train the Ensemble Meta-Learner:**

```bash
python -m src.solution2.ensemble.train_ensemble
```

Or programmatically:

```python
from src.solution2.ensemble.train_ensemble import main

main()
# Saves trained meta-learner to models/solution2/ensemble/meta_learner.pt
```

**4. Complete Pipeline (Training + Inference):**

```bash
# Step 1: Fine-tune all 4 transformers
python -m src.solution2.deberta_model.train
python -m src.solution2.roberta_model.train_roberta
python -m src.solution2.electra_model.train_electra
python -m src.solution2.xlnet_model.train_xlnet

# Step 2: Calibrate
python -m src.solution2.ensemble.calibrate_base_models

# Step 3: Train ensemble
python -m src.solution2.ensemble.train_ensemble

# Step 4: Generate predictions
python -m src.solution2.predict --split test
```

## Training Details

### Training Data

The model was trained on the **Official NLU Authorship Verification Track (C) Training set**.

### Training Procedure

#### Preprocessing

Hugging Face default tokenizers for each transformer architecture were used.

#### Training Hyperparameters

**Base Transformers (DeBERTa, RoBERTa, ELECTRA, XLNet):**

| Hyperparameter | DeBERTa | RoBERTa | ELECTRA | XLNet |
| --- | --- | --- | --- | --- |
| Max Epochs | 5 | 20* | 20* | 20* |
| Batch Size | 8 | 8 | 8 | 8 |
| Gradient Accumulation | 2 | 2 | 2 | 2 |
| **Effective Batch Size** | 16 | 16 | 16 | 16 |
| Learning Rate | 1e-5 | 1e-5 | 1.5e-5 | 1e-5 |
| Weight Decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Warmup Ratio | 10% | 10% | 10% | 10% |
| Max Gradient Norm | 1.0 | 1.0 | 1.0 | 1.0 |
| Loss Function | BCEWithLogitsLoss | BCEWithLogitsLoss | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Mixed Precision (FP16) | ✓ | ✓ | ✓ | ✓ |
| Evaluation Interval | 500 steps | 500 steps | 500 steps | 500 steps |
| Early Stopping Patience | - | 3 epochs | 3 epochs | 3 epochs |

*\*Early stopping will halt training if F1 on dev set plateaus.*

**Ensemble Meta-Learner (4-layer NN):**

| Hyperparameter | Value |
| --- | --- |
| Hidden Layer Size | 128 |
| Dropout | 0.3 |
| Batch Normalization | Yes (between layers) |
| Learning Rate | 0.001 |
| Weight Decay (L2) | 1e-4 |
| Max Epochs | 100 |
| Batch Size | 32 |
| Early Stopping Patience | 15 epochs |
| Optimizer | AdamW |
| Loss Function | Focal Loss (α=0.25, γ=2.0) |
| Metric Monitored | Macro F1 Score |
| Random Seed | 42 |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on the **Official NLU Authorship Verification Track (C) Development set**.

#### Metrics

- **Macro F1 Score:** Primary metric.
- **Accuracy:** Secondary metric.
- **Calibration Error:** Evaluated to ensure probabilities realistically represent authorship confidence.

### Results

#### Individual Base Model Performance (Dev Set)

| Model | F1 Score | Key Strength |
| --- | --- | --- |
| RoBERTa-large | 0.8463 | Optimized pretraining, robust baseline |
| ELECTRA-large | 0.8413 | Discriminator-based, token-level patterns |
| XLNet-large | 0.8234 | Autoregressive, different inductive bias |
| DeBERTa-v3-large | 0.8180 | Disentangled attention, SOTA on GLUE |

#### Ensemble Meta-Learner Performance (Dev Set)

- **F1 (macro):** 0.8644 → **+2.1% improvement** over best single model (RoBERTa)
- **Accuracy:** 0.8645
- **AUC-ROC:** 0.9234
- **Error reduction:** 18.1% compared to weakest base model (DeBERTa)

The neural meta-learner learns to weight models dynamically: RoBERTa (17.2% contribution) and ELECTRA (16.3%) lead, while XLNet (13.3%) and DeBERTa (9.0%) provide complementary diversity. This learned fusion significantly outperforms fixed voting schemes.

## Environmental Impact

Training 4 large transformer architectures has a non-negligible carbon footprint.

- **Hardware Type:** GPU (L40S/A100) used for training.
- **Inference Requirement:** GPUs recommended for batch processing.

## Technical Specifications

### Compute Infrastructure

#### Hardware (Training)

- **GPUs:** Nvidia L40S or A100 (recommended)
- **Total Training Time:** ~6 hours per transformer model (varies by hardware and hyperparameters)

#### Software

- Python 3.14
- PyTorch
- Transformers
- Scikit-learn
- CUDA 12.4.1 (if using GPU)

## Model Card Authors

- Group 17

## Model Card Contact

For coursework-related inquiries, contact the Group 17 owners.
