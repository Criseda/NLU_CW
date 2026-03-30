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

- Checkpoint: `outputs/electra_model/best_model.pt`
- Logs: Console output with epoch metrics

### Output CSV Format

| Column | Type | Notes |
|--------|------|-------|
| `text_1` | str | Original text 1 |
| `text_2` | str | Original text 2 |
| `label` | int | Ground truth (0 or 1) if available |
| `prediction` | int | Model prediction (0 or 1) |
| `probability` | float | Sigmoid probability score |

## SLURM Job Submission

### Training Job

```bash
sbatch submit_train.slurm
```

- Wallclock: 12 hours
- GPU: 2× L40S (48GB VRAM each)
- CPU: 6 cores
- Output logs: `logs/train_<JOBID>.out`
