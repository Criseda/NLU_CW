import numpy as np
import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import joblib
import lightgbm as lgb
from tqdm import tqdm


def load_features(stylometric_path: str, unmasking_path: str, impostor_path: str) -> np.ndarray:
    """Load and concatenate all feature arrays into a single (N, 97) matrix."""
    X_style  = np.load(stylometric_path)   # (N, 84)
    X_unmask = np.load(unmasking_path)     # (N, 8)  — may contain NaN
    X_impost = np.load(impostor_path)      # (N, 5)
    assert X_style.shape[0] == X_unmask.shape[0] == X_impost.shape[0], \
        "Feature arrays have different number of rows!"
    return np.concatenate([X_style, X_unmask, X_impost], axis=1)  # (N, 97)


def train_and_oof(model_fn, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Stratified K-fold CV. Returns:
      - oof_probs: (N,) out-of-fold predicted probabilities (for stacking)
      - fold_scores: list of per-fold accuracy scores
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        preds = (probs >= 0.5).astype(int)
        acc = (preds == y_val).mean()
        fold_scores.append(acc)
        print(f"  Fold {fold+1}: accuracy = {acc:.4f}")

    return oof_probs, fold_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylometric", required=True)
    parser.add_argument("--unmasking",   required=True)
    parser.add_argument("--impostor",    required=True)
    parser.add_argument("--labels",      required=True, help="Path to labels .npy")
    parser.add_argument("--output_dir",  required=True, help="Where to save models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & stack all features
    print("Loading features ...")
    X_raw = load_features(args.stylometric, args.unmasking, args.impostor)
    y = np.load(args.labels).astype(int)
    print(f"Feature matrix: {X_raw.shape}, Labels: {y.shape}, Class balance: {y.mean():.2%} positive")

    # 2. Impute NaNs (median imputation — fit on training data only)
    print("Imputing NaNs ...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)
    joblib.dump(imputer, output_dir / "imputer.joblib")

    # 3. Scale for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    # 4. Train Model A1 — Logistic Regression (ElasticNet)
    print("\n--- Model A1: Logistic Regression (ElasticNet) ---")
    def make_lr():
        return LogisticRegression(solver='saga', l1_ratio=0.5, C=1.0, max_iter=5000, random_state=42)
    oof_a1, scores_a1 = train_and_oof(make_lr, X_scaled, y)
    print(f"A1 OOF accuracy: {np.mean(scores_a1):.4f} ± {np.std(scores_a1):.4f}")

    # Retrain A1 on full data
    model_a1 = make_lr()
    model_a1.fit(X_scaled, y)
    joblib.dump(model_a1, output_dir / "model_a1.joblib")
    np.save(output_dir / "oof_probs_a1.npy", oof_a1)

    # 5. Train Model A2 — LightGBM
    print("\n--- Model A2: LightGBM ---")
    def make_lgbm():
        return lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=63, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
    oof_a2, scores_a2 = train_and_oof(make_lgbm, X, y)  # LightGBM uses raw (not scaled)
    print(f"A2 OOF accuracy: {np.mean(scores_a2):.4f} ± {np.std(scores_a2):.4f}")

    # Retrain A2 on full data
    model_a2 = make_lgbm()
    model_a2.fit(X, y)
    joblib.dump(model_a2, output_dir / "model_a2.joblib")
    np.save(output_dir / "oof_probs_a2.npy", oof_a2)

    # 6. Save OOF predictions for meta-classifier input
    oof_stack = np.column_stack([oof_a1, oof_a2])
    np.save(output_dir / "oof_stack.npy", oof_stack)  # (N, 2) — input to meta-classifier
    print(f"\nOOF stacking matrix saved: {oof_stack.shape}")
    print("Done. Models saved to", output_dir)


if __name__ == "__main__":
    main()
