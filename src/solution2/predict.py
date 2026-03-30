"""
predict.py — Top-level inference script for Solution 2.

This script runs the neural meta-learner ensemble on the test set
and formats the final output to match the shared task submission specification:
A single CSV file with exactly one column named `prediction` containing only 0s and 1s.

Usage:
    python -m src.solution2.predict
    python -m src.solution2.predict --split test

Note: Ensure that you have already generated the base model predictions for the 
target split (e.g. outputs/solution2/deberta_model/probs_test.csv).
"""

import argparse
import os
import pandas as pd
from pathlib import Path

from src.solution2.ensemble.predict_ensemble import predict_probs
from src.solution2.ensemble import config as ensemble_config

def main():
    parser = argparse.ArgumentParser(description="Generate final formatted predictions for Solution 2")
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "train", "test"],
        help="Which split to predict on and format"
    )
    args = parser.parse_args()

    print(f"\n[solution2] Running ensemble inference on '{args.split}' split...")
    
    # Run the ensemble prediction logic (this creates output in ensemble dir)
    try:
        results_df = predict_probs(split=args.split)
    except FileNotFoundError as e:
        print(f"\nERROR: Missing base model predictions or ensemble model.")
        print(e)
        print("Please ensure you have generated predictions for all base models on the target split.")
        return

    # Format the output required by the shared task spec
    # Keep strictly only the 'prediction' column
    submission_df = results_df[['prediction']].copy()
    
    # Save the submission-ready CSV
    out_dir = Path(ensemble_config.ROOT_DIR) / "outputs" / "solution2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"predictions_{args.split}.csv"
    submission_df.to_csv(out_path, index=False)
    
    print(f"\n[solution2] Success! Submission-ready CSV created:")
    print(f"  --> {out_path}")
    print(f"\nDataFrame preview:")
    print(submission_df.head())

if __name__ == "__main__":
    main()
