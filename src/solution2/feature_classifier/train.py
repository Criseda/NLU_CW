"""
train.py — Train feature-based classifiers (MLP / LightGBM / XGBoost) on 97 handcrafted features.

Uses:
  - all_features_train.npy  (pre-computed, shape [27643, 97])
  - all_features_dev.npy    (cached after first run, shape [5993, 97])

Three models are compared:
  A. PyTorch MLP (FeatureClassifier)
  B. LightGBM
  C. XGBoost

Usage:
    .venv/bin/python -m src.solution2.feature_classifier.train

Results saved to outputs/solution2/feature_classifier/feature_results.json
"""

import os
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
import lightgbm as lgb
import xgboost as xgb

from src.solution2.bi_encoder import config as base_config
from .model import FeatureClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR           = base_config.ROOT_DIR
TRAIN_FEATURES     = os.path.join(ROOT_DIR, "all_features_train.npy")
DEV_FEATURES       = os.path.join(ROOT_DIR, "all_features_dev.npy")
TRAIN_CSV          = base_config.TRAIN_FILE
DEV_CSV            = base_config.DEV_FILE

MODEL_SAVE_DIR     = os.path.join(ROOT_DIR, "models",  "solution2", "feature_classifier")
OUTPUT_DIR         = os.path.join(ROOT_DIR, "outputs", "solution2", "feature_classifier")

# ── Hyperparams ────────────────────────────────────────────────────────────────
SEED          = base_config.SEED
BATCH_SIZE    = 256
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10      # early stopping patience (by dev F1)
N_FEATURES    = 97
HIDDEN_DIM    = 256
DROPOUT       = 0.3

LGBM_PARAMS = {
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
XGB_PARAMS = {
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": SEED, "n_jobs": -1, "verbosity": 0,
}

# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    m = {
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_prob is not None:
        m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return m


def print_metrics(name: str, m: dict) -> None:
    auc = m.get("roc_auc", float("nan"))
    print(
        f"  [{name}] F1={m['f1']:.4f} | Acc={m['accuracy']:.4f} | "
        f"Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | AUC={auc:.4f}"
    )

# ── MLP training ───────────────────────────────────────────────────────────────

@torch.no_grad()
def mlp_evaluate(model, loader, device) -> tuple:
    model.eval()
    all_logits, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(y_batch)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    preds = (all_logits > 0).astype(int)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    return all_labels, preds, probs


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev:   np.ndarray, y_dev:   np.ndarray,
    device:  torch.device,
    save_path: str,
) -> dict:
    # Scale features (MLP benefits from normalisation; tree models don't need this)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_dev_s   = scaler.transform(X_dev).astype(np.float32)
    joblib.dump(scaler, save_path.replace(".pt", "_scaler.pkl"))

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    dev_ds = TensorDataset(
        torch.from_numpy(X_dev_s),
        torch.from_numpy(y_dev.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE * 2, shuffle=False)

    model     = FeatureClassifier(n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, patience_count = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch.to(device))
            loss   = criterion(logits, y_batch.to(device))
            loss.backward()
            optimizer.step()

        # Evaluate after every epoch
        y_true, preds, probs = mlp_evaluate(model, dev_loader, device)
        metrics = evaluate_preds(y_true, preds, probs)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | ", end="")
            print_metrics("MLP", metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best F1={best_f1:.4f})")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))
    y_true, preds, probs = mlp_evaluate(model, dev_loader, device)
    return evaluate_preds(y_true, preds, probs)

# ── Main ───────────────────────────────────────────────────────────────────────

def train() -> None:
    set_seed(SEED)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,     exist_ok=True)

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  feature_classifier — Feature-Based Classifier (97 handcrafted features)")
    print(f"  Device: {device}")
    print("=" * 70)

    # ── Load features ─────────────────────────────────────────────────────────
    if not os.path.exists(TRAIN_FEATURES):
        raise FileNotFoundError(
            f"Train features not found at {TRAIN_FEATURES}.\n"
            "Please provide all_features_train.npy in the project root."
        )
    if not os.path.exists(DEV_FEATURES):
        raise FileNotFoundError(
            f"Dev features not found at {DEV_FEATURES}.\n"
            "Run `python -m src.solution2.feature_classifier.train` after generating dev features,\n"
            "or ensure all_features_dev.npy exists in the project root."
        )

    X_train = np.load(TRAIN_FEATURES).astype(np.float32)
    X_dev   = np.load(DEV_FEATURES).astype(np.float32)

    y_train = pd.read_csv(TRAIN_CSV)["label"].astype(int).values
    y_dev   = pd.read_csv(DEV_CSV)["label"].astype(int).values

    # Clamp any inf/nan
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_dev   = np.nan_to_num(X_dev,   nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"\n  Train: {X_train.shape} | Dev: {X_dev.shape}")
    print(f"  Train labels — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")
    print(f"  Dev   labels — 0: {(y_dev==0).sum()} | 1: {(y_dev==1).sum()}\n")

    results = {}

    # ── A. PyTorch MLP ────────────────────────────────────────────────────────
    print("─" * 50)
    print("  [A] PyTorch MLP (FeatureClassifier)")
    t0 = time.time()
    mlp_ckpt = os.path.join(MODEL_SAVE_DIR, "feature_mlp.pt")
    mlp_metrics = train_mlp(X_train, y_train, X_dev, y_dev, device, mlp_ckpt)
    print_metrics("MLP  (best)", mlp_metrics)
    print(f"  Time: {time.time()-t0:.1f}s")
    results["mlp"] = mlp_metrics

    # ── B. LightGBM ──────────────────────────────────────────────────────────
    print("─" * 50)
    print("  [B] LightGBM")
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    lgbm_pred  = lgbm.predict(X_dev)
    lgbm_prob  = lgbm.predict_proba(X_dev)[:, 1]
    lgbm_metrics = evaluate_preds(y_dev, lgbm_pred, lgbm_prob)
    print_metrics("LGBM", lgbm_metrics)
    print(f"  Best iteration: {lgbm.best_iteration_} | Time: {time.time()-t0:.1f}s")
    results["lightgbm"] = lgbm_metrics
    lgbm.booster_.save_model(os.path.join(MODEL_SAVE_DIR, "lightgbm.txt"))

    # ── C. XGBoost ────────────────────────────────────────────────────────────
    print("─" * 50)
    print("  [C] XGBoost")
    t0 = time.time()
    xgbm = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=50, eval_metric="logloss")
    xgbm.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
    xgbm_pred  = xgbm.predict(X_dev)
    xgbm_prob  = xgbm.predict_proba(X_dev)[:, 1]
    xgbm_metrics = evaluate_preds(y_dev, xgbm_pred, xgbm_prob)
    print_metrics("XGB ", xgbm_metrics)
    print(f"  Best iteration: {xgbm.best_iteration} | Time: {time.time()-t0:.1f}s")
    results["xgboost"] = xgbm_metrics
    xgbm.save_model(os.path.join(MODEL_SAVE_DIR, "xgboost.json"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Dev Set (97 handcrafted features)")
    print("=" * 70)
    print(f"  {'Model':<20} {'F1':>8} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'AUC':>8}")
    print("  " + "-" * 68)
    for name, m in results.items():
        auc = m.get("roc_auc", float("nan"))
        print(
            f"  {name:<20} {m['f1']:>8.4f} {m['accuracy']:>10.4f} "
            f"{m['precision']:>11.4f} {m['recall']:>8.4f} {auc:>8.4f}"
        )
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  ★ Best model by F1: {best_name} (F1={results[best_name]['f1']:.4f})")
    print("=" * 70)

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "feature_results.json")
    with open(out_path, "w") as f:
        json.dump({"dev_metrics": results, "best_model": best_name}, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("  [feature_classifier] Done.")


if __name__ == "__main__":
    train()
