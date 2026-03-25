import numpy as np
import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_stack", required=True, help="Path to oof_stack.npy — (N, 2) matrix")
    parser.add_argument("--labels",    required=True, help="Path to labels .npy")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOF probs (the (N, 2) stacking input) and labels
    X_meta = np.load(args.oof_stack)   # (N, 2): [P_A1, P_A2]
    y      = np.load(args.labels).astype(int)

    print(f"Meta-classifier input: {X_meta.shape}")

    # Simple logistic regression — intentionally kept simple
    # The base models do the heavy lifting; meta-clf just learns their blend
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # Evaluate via CV first
    cv_scores = cross_val_score(meta, X_meta, y, cv=5, scoring='accuracy')
    print(f"Meta-clf CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full OOF data
    meta.fit(X_meta, y)
    print(f"Meta-clf weights: A1={meta.coef_[0][0]:.4f}, A2={meta.coef_[0][1]:.4f}")
    print(f"  (higher weight → that base model is more trusted)")

    joblib.dump(meta, output_dir / "meta_classifier.joblib")
    print(f"Saved to {output_dir / 'meta_classifier.joblib'}")


if __name__ == "__main__":
    main()
