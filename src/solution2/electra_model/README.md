# ELECTRA-Large Cross-Encoder for Authorship Verification

A discriminative-pretraining cross-encoder using ELECTRA-large for binary authorship verification.

## Why ELECTRA?

**Discriminative Pretraining Advantages**:
- Detects replaced tokens during pretraining (discriminator task) vs masked LM (BERT)
- Better sample efficiency: learns from all tokens, not just masked ones
- Usually outperforms BERT/RoBERTa on fine-tuning tasks of similar size
- **Less overhyped than DeBERTa** — genuine, stable performance gains
- **Excels on stylistic/discriminative tasks** — ideal for authorship verification
- Potential to reach **0.89-0.90 F1** with proper tuning

## Architecture

**Simple Cross-Encoder**:
- Input: `[CLS] text_1 [SEP] text_2 [SEP]`
- Encoder: ELECTRA-large discriminator (358M parameters)
- Classifier: Hidden → ReLU → Dropout → 1 logit
- Loss: BCEWithLogitsLoss (binary cross-entropy)
- Optimization: Macro F1, early stopping (patience=3)

## Configuration

**File**: `config.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_NAME` | `google/electra-large-discriminator` | Hugging Face ELECTRA discriminator |
| `EPOCHS` | 10 | Max epochs; early stopping triggers earlier |
| `BATCH_SIZE` | 8 | Per GPU |
| `GRAD_ACCUM` | 2 | Gradient accumulation steps |
| `LEARNING_RATE` | 1.5e-5 | Slightly elevated for discriminator fine-tuning |
| `MAX_LENGTH` | 512 | Token sequence length |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping |
| `SEED` | 42 | Reproducibility |

## Training

**Script**: `train_electra.py`

### Features
- ✅ Multi-GPU via `nn.DataParallel`
- ✅ Macro F1 evaluation metric (grading metric)
- ✅ Early stopping with patience=3
- ✅ Linear warmup + decay scheduler
- ✅ Gradient clipping & accumulation
- ✅ Best checkpoint saving

### Usage

#### Via SLURM
```bash
sbatch submit_train.slurm
```

#### Direct (development)
```bash
python -m src.solution2.electra_model.train_electra
```

### Expected Outputs
- Checkpoint: `outputs/best_model.pt`
- Logs: Console output with epoch metrics

## Prediction

**Script**: `predict_electra.py`

### Usage
```bash
python -m src.solution2.electra_model.predict_electra \
    --checkpoint src/solution2/electra_model/outputs/best_model.pt \
    --input data/dev.csv \
    --output predictions/dev_predictions.csv
```

### Output CSV Format
| Column | Type | Notes |
|--------|------|-------|
| `text_1` | str | Original text 1 |
| `text_2` | str | Original text 2 |
| `label` | int | Ground truth (0 or 1) if available |
| `prediction` | int | Model prediction (0 or 1) |
| `probability` | float | Sigmoid probability score |

### Metrics Reported
- **Binary F1**: Standard F1 score for binary classification
- **Macro F1**: Macro-averaged F1 (grading metric)
- **Accuracy**: Percent correct predictions

## SLURM Job Submission

### Training Job
```bash
sbatch submit_train.slurm
```
- Wallclock: 12 hours
- GPU: 2× L40S (48GB VRAM each)
- CPU: 6 cores
- Output logs: `logs/train_<JOBID>.out`

### Prediction Job
```bash
sbatch submit_predict.slurm
```
- Wallclock: 2 hours
- GPU: 1× L40S (sufficient for inference)
- CPU: 4 cores
- Runs predictions on dev.csv and test.csv (if exists)
- Output: `predictions/dev_predictions.csv`, `predictions/test_predictions.csv`

## Model Comparison

### Ensemble Strategy
ELECTRA-large serves as the **discriminative pretraining specialist** in the ensemble:

| Model | Architecture | Key Feature | Pretraining | F1 Baseline |
|-------|--------------|-------------|------------|------------|
| big_model | DeBERTa-v3-large | Disentangled attention | Masked LM | 0.8176 |
| special_model | XLNet-large + stylistic | Semantic + stylometry | Permutation LM | TBD |
| roberta_model | RoBERTa-large | Pure transformer baseline | Masked LM | TBD |
| electra_model | ELECTRA-large (discriminator) | Discriminative pretraining | Token detection | **TBD (0.89-0.90 target)** |

**Expected Synergy**:
- DeBERTa: Strongest baseline (0.8176)
- ELECTRA: Discriminative specialist (untapped potential)
- XLNet+stylistic: Novel semantic-stylometric fusion
- RoBERTa: Traditional stable baseline

Ensemble weights: Grid search recommended for optimal combination. **ELECTRA may be the key to breaking 0.89**.

## Dependencies
- `torch>=2.0`
- `transformers>=4.30` (includes ELECTRA support)
- `numpy`
- `pandas`
- `scikit-learn`

## Notes
- ELECTRA discriminator uses token_type_ids for segment encoding (handled automatically)
- Learning rate is slightly elevated (1.5e-5) due to discriminative pretraining efficiency
- Early stopping patience=3 prevents overfitting on dev set
- Macro F1 metric aligns with official grading criteria
- **Token detection pretraining** makes ELECTRA more sample-efficient than masked LM models

### Expected Performance Estimates
```
Macro F1: 0.80-0.82 (conservative)
Macro F1: 0.85-0.90 (optimistic with tuning)
Binary F1: 0.82-0.84
Accuracy: 0.79-0.81
```

## References
- ELECTRA Paper: https://openreview.net/pdf?id=r1xMH1BtvB
- Discriminative pretraining significantly outperforms masked LM on downstream tasks
- Particularly strong on tasks requiring fine-grained discrimination (like authorship verification)
