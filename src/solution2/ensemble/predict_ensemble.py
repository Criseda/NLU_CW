"""
predict_ensemble.py — Make predictions with trained neural network meta-learner.

Loads predictions from all configured models and uses the trained neural network
to generate combined ensemble predictions.

Usage:
    # Predict on dev set:
    python -m src.solution2.ensemble.predict_ensemble --split dev

    # Predict with custom model predictions:
    python -m src.solution2.ensemble.predict_ensemble \\
        --split custom
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch

from . import config, models_config
from .train_ensemble import create_feature_matrix, load_model_predictions, MetaLearner


def apply_base_model_calibration(model_dfs: dict, split: str = "dev") -> dict:
    """
    Apply isotonic regression calibration to base model predictions.
    
    This improves calibration of individual models before they enter the ensemble.
    """
    calibrators_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "calibrators", "calibrators.pkl")
    
    if not os.path.exists(calibrators_path):
        print(f"[ensemble-predict] Calibrators not found at {calibrators_path}")
        print(f"[ensemble-predict] Using raw model predictions (not calibrated)")
        return model_dfs
    
    print(f"[ensemble-predict] Loading base model calibrators...")
    with open(calibrators_path, 'rb') as f:
        calibrators = pickle.load(f)
    
    # Apply calibrators to each model's predictions
    calibrated_dfs = {}
    for model_name, df in model_dfs.items():
        if model_name not in calibrators:
            print(f"[ensemble-predict] Warning: No calibrator for {model_name}, using raw predictions")
            calibrated_dfs[model_name] = df
            continue
        
        iso_reg = calibrators[model_name]
        
        # Apply calibration to probability column
        raw_probs = df[f'{model_name}_prob'].values
        calibrated_probs = iso_reg.predict(raw_probs)
        
        # Create new dataframe with calibrated probs but original preds
        calibrated_df = df.copy()
        calibrated_df[f'{model_name}_prob'] = calibrated_probs
        calibrated_dfs[model_name] = calibrated_df
    
    print(f"[ensemble-predict] Applied calibration to {len(calibrated_dfs)} models")
    return calibrated_dfs


def predict_probs(split: str = "val") -> pd.DataFrame:
    """Load model predictions and generate ensemble predictions."""
    print(f"[ensemble-predict] Loading model predictions for {split}...")
    
    # Load predictions from all configured models
    model_dfs = {}
    for model_name, paths in models_config.MODELS.items():
        if split == "dev":
            probs_path = paths['dev_probs']
        elif split == "train":
            probs_path = paths['train_probs']
        elif split == "test":
            probs_path = paths['test_probs']
        else:
            raise ValueError(f"Unknown split: {split}. Use 'dev', 'train', or 'test'")
        
        print(f"  - Loading {model_name} {split} predictions...")
        model_dfs[model_name] = load_model_predictions(probs_path, model_name)
    
    # Apply base model calibration
    model_dfs = apply_base_model_calibration(model_dfs, split=split)
    
    print(f"[ensemble-predict] Creating feature matrix...")
    X = create_feature_matrix(model_dfs)
    
    # Load trained model
    model_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "meta_learner.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model not found at {model_path}. "
                                f"Run train_ensemble.py first.")
    
    # Load feature names to ensure correct order
    feature_names_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "feature_names.pkl")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load optimal threshold
    threshold_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "threshold.pkl")
    if os.path.exists(threshold_path):
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
        print(f"[ensemble-predict] Using optimal threshold: {threshold:.2f}")
    else:
        threshold = 0.5
        print(f"[ensemble-predict] Threshold file not found, using default 0.5")
    
    # Load isotonic regressor for calibration
    iso_reg_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "iso_regressor.pkl")
    iso_reg = None
    if os.path.exists(iso_reg_path):
        with open(iso_reg_path, 'rb') as f:
            iso_reg = pickle.load(f)
        print(f"[ensemble-predict] Loaded isotonic regressor for calibration")
    else:
        print(f"[ensemble-predict] Isotonic regressor not found, predictions will not be calibrated")
    
    print(f"[ensemble-predict] Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MetaLearner(
        input_size=len(feature_names),
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Predict
    print(f"[ensemble-predict] Generating ensemble predictions...")
    X_tensor = torch.FloatTensor(X[feature_names].values).to(device)
    
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy().flatten()
    
    # Apply isotonic regression calibration if available
    if iso_reg is not None:
        print(f"[ensemble-predict] Applying isotonic regression calibration...")
        probs = iso_reg.predict(probs)
    
    preds = (probs > threshold).astype(int)
    
    results = pd.DataFrame({
        'probability': np.round(probs, 6),
        'prediction': preds,
    })
    
    # Save
    os.makedirs(config.ENSEMBLE_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.ENSEMBLE_OUTPUT_DIR, f"probs_{split}.csv")
    results.to_csv(out_path, index=False)
    print(f"[ensemble-predict] Saved → {out_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions with trained neural network ensemble"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["dev", "train", "test"],
        help="Which split to predict on"
    )
    args = parser.parse_args()
    
    predict_probs(split=args.split)


if __name__ == "__main__":
    main()
