"""
predict.py — Top-level inference script for Solution 1.

This script runs the full stacking pipeline on the provided feature array
and outputs the final predictions to a submission-ready CSV format
with a single column named `prediction` containing only 0s and 1s.

Usage:
    python -m src.solution1.predict \
        --features src/solution1/features/all_features_test.npy \
        --model_dir src/solution1/models/full \
        --output outputs/solution1/predictions_test.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Try to import predict_with_stack if available directly from evaluation code
try:
    from src.solution1.training.evaluate import predict_with_stack
except ImportError:
    import joblib
    def predict_with_stack(X_raw: np.ndarray, model_dir: Path) -> tuple:
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


def main():
    parser = argparse.ArgumentParser(description="Generate final formatted predictions for Solution 1")
    parser.add_argument("--features", required=False, 
                        default="src/solution1/features/all_features_test.npy",
                        help="Path to (N, 97) test features .npy")
    parser.add_argument("--model_dir", required=False, 
                        default="models/solution1",
                        help="Directory containing trained models")
    parser.add_argument("--output", required=False, 
                        default="outputs/solution1/predictions_test.csv",
                        help="Path to save submission-ready CSV")
    
    args = parser.parse_args()

    feature_path = Path(args.features)
    model_dir = Path(args.model_dir)
    out_path = Path(args.output)
    
    # Ensure dependencies exist
    if not feature_path.exists():
        print(f"ERROR: Could not find features file at {feature_path}.")
        return
        
    if not (model_dir / "meta_classifier.joblib").exists():
        print(f"ERROR: Could not find trained models in {model_dir}.")
        return

    print(f"\n[solution1] Loading features from {feature_path}...")
    X_raw = np.load(feature_path)
    
    print(f"[solution1] Running stacking inference models...")
    y_pred, _, _, _ = predict_with_stack(X_raw, model_dir)
    
    # Format the output required by the shared task spec
    # Keep strictly only the 'prediction' column
    submission_df = pd.DataFrame({'prediction': y_pred})
    
    # Save the submission-ready CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    
    print(f"\n[solution1] Success! Submission-ready CSV created:")
    print(f"  --> {out_path}")
    print(f"\nDataFrame preview:")
    print(submission_df.head())

if __name__ == "__main__":
    main()
