"""
calibrate_base_models.py — Calibrate individual base model predictions.

Fits isotonic regression on each model's training predictions to improve calibration
before they enter the ensemble. Uses train_calib set to avoid data leakage.

Usage:
    python -m src.solution2.ensemble.calibrate_base_models
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

from . import config, models_config


def load_model_predictions(model_probs_path: str) -> np.ndarray:
    """Load probability predictions from a CSV."""
    df = pd.read_csv(model_probs_path)
    if 'probability' not in df.columns:
        raise ValueError(f"No 'probability' column in {model_probs_path}")
    return df['probability'].values


def load_labels(csv_path: str) -> np.ndarray:
    """Load labels from training/dev CSV."""
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError(f"No 'label' column in {csv_path}")
    return df['label'].values


def calibrate_model(model_name: str, probs_path: str, labels: np.ndarray, indices: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression calibrator for a single model.
    
    Args:
        model_name: Name of model (for logging)
        probs_path: Path to CSV with raw predictions
        labels: Ground truth labels (full set)
        indices: Indices to use for fitting (calibration set)
    
    Returns:
        Fitted isotonic regressor
    """
    print(f"\n[calibrate] {model_name}:")
    
    # Load raw predictions
    raw_probs = load_model_predictions(probs_path)
    
    # Get calibration subset
    calib_probs = raw_probs[indices]
    calib_labels = labels[indices]
    
    # Compute calibration error on this set
    original_prob_true, original_prob_pred = calibration_curve(
        calib_labels, calib_probs, n_bins=10, strategy='uniform'
    )
    original_ece = np.mean(np.abs(original_prob_true - original_prob_pred))
    
    # Fit isotonic regressor on calibration set
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(calib_probs, calib_labels)
    
    # Apply calibration to calibration set (to show improvement)
    calibrated_calib_probs = iso_reg.predict(calib_probs)
    
    # Compute calibrated calibration error
    calibrated_prob_true, calibrated_prob_pred = calibration_curve(
        calib_labels, calibrated_calib_probs, n_bins=10, strategy='uniform'
    )
    calibrated_ece = np.mean(np.abs(calibrated_prob_true - calibrated_prob_pred))
    
    print(f"  Calibration set size: {len(calib_probs)}")
    print(f"  Original ECE: {original_ece:.4f}")
    print(f"  Calibrated ECE: {calibrated_ece:.4f}")
    print(f"  Improvement: {(original_ece - calibrated_ece) / original_ece * 100:.1f}%")
    
    return iso_reg


def main():
    """Calibrate all base models using train_calib split."""
    
    # Load training labels
    y_train = load_labels(models_config.TRAIN_LABELS_FILE)
    
    # Split training set: use first 20% for calibration, rest for training
    n_calib = int(len(y_train) * 0.2)
    calib_indices = np.arange(n_calib)
    train_indices = np.arange(n_calib, len(y_train))
    
    print("[calibrate] Calibrating base model predictions...")
    print(f"[calibrate] Training set size: {len(y_train)}")
    print(f"[calibrate] Calibration subset: {len(calib_indices)} samples")
    print(f"[calibrate] Training subset: {len(train_indices)} samples")
    
    # Create calibration directory
    calibration_dir = os.path.join(config.ENSEMBLE_MODEL_DIR, "calibrators")
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Calibrate each model (using calib subset for honest metrics)
    calibrators = {}
    for model_name, paths in models_config.MODELS.items():
        iso_reg = calibrate_model(
            model_name,
            paths['train_probs'],
            y_train,
            calib_indices
        )
        calibrators[model_name] = iso_reg
    
    # Save all calibrators
    calibrators_path = os.path.join(calibration_dir, "calibrators.pkl")
    with open(calibrators_path, 'wb') as f:
        pickle.dump(calibrators, f)
    print(f"\n[calibrate] All calibrators saved → {calibrators_path}")
    
    return calibrators


if __name__ == "__main__":
    main()

