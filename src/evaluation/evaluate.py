"""
src/evaluation/evaluate.py — Unified Evaluation Module for NLU Share Task

This script centralizes all evaluation logic to meet the coursework spec requirement
of separating training and evaluation into distinct modules.

It provides both:
1. Reusable metric functions imported by `train.py` scripts during per-epoch validation.
2. A standalone CLI to evaluate any generated predictions CSV against a gold labels CSV.

Usage (CLI):
    python src/evaluation/evaluate.py \
        --predictions outputs/solution2/predictions_dev.csv \
        --gold data/training_data/AV/dev.csv \
        --model_type solution2
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Core evaluation function used by both solutions during training and standalone evaluation.
    Computes a standard suite of classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted')),
        "f1_binary": float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    
    # Probabilistic metrics if probabilities are provided
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except ValueError:
            # Handles cases where y_true only has one class during certain batches
            metrics["roc_auc"] = float('nan')
            metrics["brier_score"] = float('nan')
            
    return metrics


def evaluate_csv(predictions_path: str, gold_path: str, model_type: str) -> dict:
    """
    Evaluates a saved predictions CSV against a gold labels CSV.
    """
    pred_df = pd.read_csv(predictions_path)
    gold_df = pd.read_csv(gold_path)
    
    if len(pred_df) != len(gold_df):
        raise ValueError(f"Mismatch in rows: Predictions ({len(pred_df)}) vs Gold ({len(gold_df)})")
        
    y_true = gold_df['label'].values
    y_pred = pred_df['prediction'].values
    y_prob = pred_df['probability'].values if 'probability' in pred_df.columns else None
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Calculate confusion matrix strings for CLI output
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['confusion_matrix'] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for NLU Coursework")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--gold", required=True, help="Path to gold labels CSV")
    parser.add_argument("--model_type", required=False, default="unified", help="Solution type (for logging)")
    parser.add_argument("--output", required=False, help="Optional JSON path to save metrics")
    
    args = parser.parse_args()
    
    print(f"\n--- Unified Evaluation ({args.model_type}) ---")
    print(f"Predictions: {args.predictions}")
    print(f"Gold Labels: {args.gold}")
    
    try:
        metrics = evaluate_csv(args.predictions, args.gold, args.model_type)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return

    print("\n[Metrics Summary]")
    for key, val in metrics.items():
        if key == 'confusion_matrix':
            continue
        print(f"  {key:20s}: {val:.4f}")
        
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\n[Confusion Matrix]")
        print(f"  TP={cm['tp']:5d}  FP={cm['fp']:5d}")
        print(f"  FN={cm['fn']:5d}  TN={cm['tn']:5d}")
        
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to {out_path}")

if __name__ == "__main__":
    main()
