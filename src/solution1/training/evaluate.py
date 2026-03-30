"""
evaluate.py — Solution 1 Evaluation Script
==========================================
Evaluates the full AV stacking pipeline on a held-out feature set.
Produces:
  - Full metrics (accuracy, F1, ROC-AUC, balanced accuracy)
  - 95% bootstrap confidence intervals (1000 resamples)
  - Per-feature-group ablation study
  - Calibration curve
  - Confusion matrix
  - JSON results summary

Usage:
  # Evaluate on a held-out split:
  python evaluate.py \
      --features src/solution1/features/all_features_dev.npy \
      --labels   data/dev_labels.npy \
      --model_dir src/solution1/models/full \
      --output   src/solution1/results/eval_results.json

  # Evaluate on training OOF predictions (no dev features needed):
  python evaluate.py \
      --oof_mode \
      --features src/solution1/features/all_features_train.npy \
      --labels   data/train_labels.npy \
      --model_dir src/solution1/models/full \
      --output   src/solution1/results/oof_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    balanced_accuracy_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# ── Feature group slices (indices into the 97-dim vector) ──────────────────
FEATURE_GROUPS = {
    "char_ngrams":     slice(0, 20),    # §3.1 — 20 features
    "compression":     slice(20, 26),   # §3.2 — 6 features
    "function_words":  slice(26, 46),   # §3.3 — 20 features
    "vocab_richness":  slice(46, 54),   # §3.4 — 8 features
    "syntactic":       slice(54, 67),   # §3.5 — 13 features
    "surface":         slice(67, 77),   # §3.6 — 10 features
    "readability":     slice(77, 84),   # §3.7 — 7 features
    "info_theoretic":  slice(84, 92),   # info-theoretic (replaced unmasking) — 8 features
    "impostor":        slice(92, 97),   # §4.2 GI — 5 features
}

STYLOMETRIC_SLICE = slice(0, 84)   # All 84 "fast" stylometric features
HPC_SLICE         = slice(84, 97)  # 13 HPC-derived features


# ── Metric computation ──────────────────────────────────────────────────────

from src.evaluation.evaluate import compute_metrics as base_compute_metrics

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all scalar evaluation metrics for a set of predictions."""
    metrics = base_compute_metrics(y_true, y_pred, y_prob)
    # Ensure they are rounded for backward compatibility
    return {k: round(v, 6) for k, v in metrics.items()}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Non-parametric bootstrap confidence intervals for all metrics.
    Returns dict of {metric: {"mean": ..., "lower": ..., "upper": ...}}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_metrics = {k: [] for k in compute_metrics(y_true, y_pred, y_prob)}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp, yprob = y_true[idx], y_pred[idx], y_prob[idx]
        # Skip degenerate samples (only one class)
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp, yprob)
        for k, v in m.items():
            boot_metrics[k].append(v)

    ci_results = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        ci_results[k] = {
            "mean":  round(float(arr.mean()), 6),
            "lower": round(float(np.percentile(arr, 100 * alpha / 2)), 6),
            "upper": round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 6),
        }
    return ci_results


# ── Ablation study ──────────────────────────────────────────────────────────

def ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Train a LightGBM classifier on each feature subset via stratified K-fold CV.
    Returns OOF accuracy for each subset.
    """
    ablation_results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Feature subsets to evaluate
    subsets = {
        "all_97":          np.arange(97),
        "stylometric_84":  np.arange(84),
        "hpc_features_13": np.arange(84, 97),
        **{name: np.arange(97)[sl] for name, sl in FEATURE_GROUPS.items()},
    }

    for subset_name, feature_idx in subsets.items():
        X_sub = X[:, feature_idx]
        if len(feature_idx) == 0:
            ablation_results[subset_name] = None
            continue

        # Impute NaNs within subset
        imputer = SimpleImputer(strategy='median')
        X_sub = imputer.fit_transform(X_sub)

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            random_state=42, verbose=-1, n_jobs=1,
        )

        oof_preds = np.zeros(len(y), dtype=int)
        for train_idx, val_idx in skf.split(X_sub, y):
            model.fit(X_sub[train_idx], y[train_idx])
            oof_preds[val_idx] = model.predict(X_sub[val_idx])

        acc   = float(accuracy_score(y, oof_preds))
        f1    = float(f1_score(y, oof_preds, average='macro'))
        ablation_results[subset_name] = {
            "n_features": int(len(feature_idx)),
            "oof_accuracy": round(acc, 6),
            "oof_f1_macro": round(f1, 6),
        }
        print(f"  [{subset_name:20s}] n={len(feature_idx):3d}  acc={acc:.4f}  f1={f1:.4f}")

    return ablation_results


# ── Calibration ──────────────────────────────────────────────────────────────

def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability diagram data + Expected Calibration Error (ECE)."""
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    # ECE: weighted mean absolute calibration error
    bin_sizes, edges = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    bin_sizes = bin_sizes.astype(float)
    weights = bin_sizes / bin_sizes.sum() if bin_sizes.sum() > 0 else np.ones_like(bin_sizes)
    ece = float(np.sum(weights[:len(fraction_pos)] * np.abs(fraction_pos - mean_predicted)))
    return {
        "fraction_positive": fraction_pos.round(6).tolist(),
        "mean_predicted":    mean_predicted.round(6).tolist(),
        "ece":               round(ece, 6),
    }


# ── Inference pipeline ── ────────────────────────────────────────────────────

def predict_with_stack(X_raw: np.ndarray, model_dir: Path) -> tuple:
    """
    Run the full stacking inference pipeline.
    Returns (y_pred, y_prob_stack) using saved imputer + scaler + models.
    """
    imputer = joblib.load(model_dir / "imputer.joblib")
    scaler  = joblib.load(model_dir / "scaler.joblib")
    model_a1 = joblib.load(model_dir / "model_a1.joblib")
    model_a2 = joblib.load(model_dir / "model_a2.joblib")
    meta     = joblib.load(model_dir / "meta_classifier.joblib")

    X = imputer.transform(X_raw)
    X_scaled = scaler.transform(X)

    prob_a1 = model_a1.predict_proba(X_scaled)[:, 1]
    prob_a2 = model_a2.predict_proba(X)[:, 1]

    X_meta   = np.column_stack([prob_a1, prob_a2])
    y_prob   = meta.predict_proba(X_meta)[:, 1]
    y_pred   = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob, prob_a1, prob_a2


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Solution 1 Evaluation Script")
    parser.add_argument("--features",   required=True, help="Path to (N, 97) .npy feature matrix")
    parser.add_argument("--labels",     required=True, help="Path to labels .npy")
    parser.add_argument("--model_dir",  required=True, help="Directory containing trained models")
    parser.add_argument("--output",     required=True, help="Path to output JSON file")
    parser.add_argument("--oof_mode",   action="store_true",
                        help="Evaluate on OOF predictions instead of running inference. "
                             "Use when evaluating on training data itself.")
    parser.add_argument("--ablation",   action="store_true",
                        help="Run per-feature-group ablation study (adds ~5 mins)")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    X = np.load(args.features)
    y = np.load(args.labels).astype(int)
    print(f"Loaded features: {X.shape}, Labels: {y.shape}")
    print(f"Class balance: {y.mean():.2%} positive\n")

    # ── Predictions ─────────────────────────────────────────────────────────
    if args.oof_mode:
        print("OOF mode — using saved OOF predictions ...")
        prob_a1 = np.load(model_dir / "oof_probs_a1.npy")
        prob_a2 = np.load(model_dir / "oof_probs_a2.npy")
        meta    = joblib.load(model_dir / "meta_classifier.joblib")
        X_meta  = np.column_stack([prob_a1, prob_a2])
        y_prob  = meta.predict_proba(X_meta)[:, 1]
        y_pred  = (y_prob >= 0.5).astype(int)
    else:
        print("Running full inference pipeline ...")
        y_pred, y_prob, prob_a1, prob_a2 = predict_with_stack(X, model_dir)

    # ── Core metrics ─────────────────────────────────────────────────────────
    print("─" * 60)
    print("EVALUATION RESULTS")
    print("─" * 60)
    metrics = compute_metrics(y, y_pred, y_prob)
    for name, val in metrics.items():
        print(f"  {name:20s} {val:.4f}")

    # ── Bootstrap CI ─────────────────────────────────────────────────────────
    print(f"\nBootstrap CI ({args.n_bootstrap} resamples) ...")
    ci = bootstrap_ci(y, y_pred, y_prob, n_resamples=args.n_bootstrap)
    print("  Metric               Mean     95% CI")
    print("  " + "─" * 50)
    for name, vals in ci.items():
        print(f"  {name:20s} {vals['mean']:.4f}   [{vals['lower']:.4f}, {vals['upper']:.4f}]")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix:")
    print(f"  TP={tp:5d}  FP={fp:5d}")
    print(f"  FN={fn:5d}  TN={tn:5d}")

    # ── Individual model metrics ─────────────────────────────────────────────
    pred_a1 = (prob_a1 >= 0.5).astype(int)
    pred_a2 = (prob_a2 >= 0.5).astype(int)
    print(f"\nBase model comparison:")
    print(f"  A1 (LogReg)   acc={accuracy_score(y, pred_a1):.4f}  "
          f"f1={f1_score(y, pred_a1, average='macro'):.4f}  "
          f"auc={roc_auc_score(y, prob_a1):.4f}")
    print(f"  A2 (LightGBM) acc={accuracy_score(y, pred_a2):.4f}  "
          f"f1={f1_score(y, pred_a2, average='macro'):.4f}  "
          f"auc={roc_auc_score(y, prob_a2):.4f}")
    print(f"  Stacked meta  acc={metrics['accuracy']:.4f}  "
          f"f1={metrics['f1_macro']:.4f}  "
          f"auc={metrics['roc_auc']:.4f}")

    # ── Calibration ─────────────────────────────────────────────────────────
    calibration = compute_calibration(y, y_prob)
    print(f"\nCalibration ECE: {calibration['ece']:.4f}")

    # ── Ablation ─────────────────────────────────────────────────────────────
    ablation_results = {}
    if args.ablation:
        print("\nRunning ablation study ...")
        ablation_results = ablation_study(X, y)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "mode":         "oof" if args.oof_mode else "held_out",
        "n_samples":    int(len(y)),
        "class_balance": round(float(y.mean()), 4),
        "metrics":      metrics,
        "bootstrap_ci": ci,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "base_models": {
            "a1_logreg": {
                "accuracy": round(float(accuracy_score(y, pred_a1)), 6),
                "f1_macro": round(float(f1_score(y, pred_a1, average='macro')), 6),
                "roc_auc":  round(float(roc_auc_score(y, prob_a1)), 6),
            },
            "a2_lgbm": {
                "accuracy": round(float(accuracy_score(y, pred_a2)), 6),
                "f1_macro": round(float(f1_score(y, pred_a2, average='macro')), 6),
                "roc_auc":  round(float(roc_auc_score(y, prob_a2)), 6),
            },
        },
        "calibration":  calibration,
        "ablation":     ablation_results,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
