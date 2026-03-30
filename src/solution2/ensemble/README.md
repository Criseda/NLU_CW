# Neural Network Ensemble with Isotonic Calibration

## What Is It?

Combines models via a neural network meta-learner. Produces calibrated probabilities (ECE 0.0002) and F1 0.8644 on dev set.

## What It Does

**3-step pipeline:**

1. Calibrate each base model with isotonic regression (on 20% training subset)
2. Train neural network to learn non-linear combinations (on all training data)
3. Re-calibrate ensemble output with isotonic regression (on dev set)


## Architecture

**Neural Network:** Input(19) → [128→128→64→32→1] with BatchNorm, ReLU, Dropout, Sigmoid

**19 Features:**

- Per-model (8): roberta_prob, roberta_pred, xlnet_prob, xlnet_pred, electra_prob, electra_pred, deberta_prob, deberta_pred
- Aggregates (6): max_prob, min_prob, mean_prob, std_prob, prob_range, prob_variance
- Voting (3): vote_pred, vote_certainty, perfect_agreement
- Agreement (2): predictions_entropy, consensus_confidence

**Training:** Adam, LR=0.001, CrossEntropy loss, CosineAnnealingLR, early stopping (patience=15)

**Calibration Layers:**

- Base model: Isotonic regressor per model (fitted on first 20% of training)
- Ensemble: Isotonic regressor on dev predictions
- Threshold: Grid search 0.35-0.65 for max F1

## Usage

**3-step training pipeline (automatic via SLURM):**

```bash
# Step 1: Calibrate base models
python -m src.solution2.ensemble.calibrate_base_models

# Step 2: Train meta-learner
python -m src.solution2.ensemble.train_ensemble

# Step 3: Generate predictions
python -m src.solution2.ensemble.predict_ensemble
```

Or via SLURM (if using CSF3):

```bash
sbatch src/solution2/ensemble/submit_ensemble.slurm
```

**Add a new model:**

1. Generate predictions: `outputs/solution2/{model}/probs_train.csv`, `probs_dev.csv`
2. Update `models_config.py` with model paths
3. Retrain: `sbatch submit_ensemble.slurm`

