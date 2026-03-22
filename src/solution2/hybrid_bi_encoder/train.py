"""
train.py — Fine-tune the Hybrid Bi-Encoder (transformer + 97 handcrafted features).

Architecture:
  text pair → MiniLM (shared) → [u, v, |u-v|, hand_features_97] → classifier → logit

Requires:
  - all_features_train.npy  (27643, 97) in project root
  - all_features_dev.npy    (5993, 97)  in project root

Usage:
    .venv/bin/python -m src.solution2.hybrid_bi_encoder.train

Settings live in this file (at the top). Does NOT touch bi_encoder/ files.
"""

import os
import random

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.solution2.bi_encoder import config as base_config   # borrow paths + MODEL_NAME only
from .model import HybridBiEncoder
from .dataset import HybridAVDataset

# ── Config (edit here, not in config.py) ──────────────────────────────────────

MODEL_NAME   = base_config.MODEL_NAME          # sentence-transformers/all-MiniLM-L6-v2
MAX_LENGTH   = base_config.MAX_LENGTH          # 256 tokens per text
DEVICE       = torch.device(base_config.DEVICE)
SEED         = base_config.SEED

N_FEATURES   = 97
EPOCHS       = 5
BATCH_SIZE   = 16
GRAD_ACCUM   = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
EVAL_STEPS    = 200
DROPOUT       = 0.1

ROOT_DIR     = base_config.ROOT_DIR
TRAIN_CSV    = base_config.TRAIN_FILE
DEV_CSV      = base_config.DEV_FILE
TRAIN_FEATS  = os.path.join(ROOT_DIR, "all_features_train.npy")
DEV_FEATS    = os.path.join(ROOT_DIR, "all_features_dev.npy")

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models",  "solution2", "hybrid_model")
OUTPUT_DIR     = os.path.join(ROOT_DIR, "outputs", "solution2", "hybrid_model")

# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: HybridBiEncoder, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        logits = model(
            input_ids_1      = batch["input_ids_1"].to(device),
            attention_mask_1 = batch["attention_mask_1"].to(device),
            input_ids_2      = batch["input_ids_2"].to(device),
            attention_mask_2 = batch["attention_mask_2"].to(device),
            hand_features    = batch["hand_features"].to(device),
        )
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    preds      = (all_logits > 0).astype(int)
    probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()

    return {
        "f1":        f1_score(all_labels, preds, zero_division=0),
        "accuracy":  accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall":    recall_score(all_labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_labels, probs),
    }

# ── Training ───────────────────────────────────────────────────────────────────

def train() -> None:
    set_seed(SEED)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,     exist_ok=True)

    print("=" * 70)
    print("  Hybrid Bi-Encoder — Transformer + 97 Handcrafted Features")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # ── Sanity-check feature files ────────────────────────────────────────────
    for path, label in [(TRAIN_FEATS, "train"), (DEV_FEATS, "dev")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} features not found at {path}.\n"
                "Make sure all_features_train.npy and all_features_dev.npy\n"
                "are in the project root. Run train_features.py first if needed."
            )

    # ── Normalise features (fit on train, apply to both) ─────────────────────
    # Raw feature values can be very large (e.g. readability scores, compression
    # ratios) which would dwarf transformer embeddings (~unit-norm) and cause
    # loss explosion. StandardScaler brings everything to zero-mean, unit-var.
    print("\n[0/3] Normalising features with StandardScaler (fit on train) …")
    X_train_raw = np.load(TRAIN_FEATS).astype(np.float64)
    X_dev_raw   = np.load(DEV_FEATS).astype(np.float64)
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    X_dev_raw   = np.nan_to_num(X_dev_raw,   nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_dev_scaled   = scaler.transform(X_dev_raw).astype(np.float32)

    # Save scaled features to temp paths so HybridAVDataset can load them
    scaled_train_path = os.path.join(MODEL_SAVE_DIR, "features_train_scaled.npy")
    scaled_dev_path   = os.path.join(MODEL_SAVE_DIR, "features_dev_scaled.npy")
    np.save(scaled_train_path, X_train_scaled)
    np.save(scaled_dev_path,   X_dev_scaled)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "feature_scaler.pkl"))
    print(f"  Scaler saved. Feature stats — mean: {scaler.mean_[:5].round(3)} … std: {scaler.scale_[:5].round(3)} …")

    # ── Tokeniser & datasets ──────────────────────────────────────────────────
    print("\n[1/3] Loading datasets …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = HybridAVDataset(TRAIN_CSV, scaled_train_path, tokenizer, MAX_LENGTH)
    dev_ds   = HybridAVDataset(DEV_CSV,   scaled_dev_path,   tokenizer, MAX_LENGTH)

    # Use 0 workers on MPS to avoid multiprocessing issues
    num_workers = 0 if str(DEVICE) == "mps" else 2

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(str(DEVICE) != "mps"),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(str(DEVICE) != "mps"),
    )
    print(f"  Train: {len(train_ds)} samples | Dev: {len(dev_ds)} samples")
    print(f"  Feature dim: {train_ds.features.shape[1]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Building HybridBiEncoder …")
    model = HybridBiEncoder(
        model_name=MODEL_NAME,
        n_features=N_FEATURES,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Training for {EPOCHS} epoch(s) …\n")
    best_f1, global_step = 0.0, 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            input_ids_1      = batch["input_ids_1"].to(DEVICE)
            attention_mask_1 = batch["attention_mask_1"].to(DEVICE)
            input_ids_2      = batch["input_ids_2"].to(DEVICE)
            attention_mask_2 = batch["attention_mask_2"].to(DEVICE)
            hand_features    = batch["hand_features"].to(DEVICE)
            labels           = batch["labels"].to(DEVICE)

            logits = model(
                input_ids_1=input_ids_1,
                attention_mask_1=attention_mask_1,
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2,
                hand_features=hand_features,
            )
            loss = criterion(logits, labels) / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic evaluation
                if global_step % EVAL_STEPS == 0:
                    metrics = evaluate(model, dev_loader, DEVICE)
                    print(
                        f"\n  Step {global_step} | "
                        f"Loss {running_loss / EVAL_STEPS:.4f} | "
                        f"F1 {metrics['f1']:.4f} | "
                        f"Acc {metrics['accuracy']:.4f} | "
                        f"Prec {metrics['precision']:.4f} | "
                        f"Rec {metrics['recall']:.4f} | "
                        f"AUC {metrics['roc_auc']:.4f}"
                    )
                    running_loss = 0.0
                    model.train()

                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        ckpt = os.path.join(MODEL_SAVE_DIR, "best_hybrid.pt")
                        torch.save(model.state_dict(), ckpt)
                        print(f"  ✓ New best F1 {best_f1:.4f} — saved to {ckpt}")

        # End-of-epoch evaluation
        metrics = evaluate(model, dev_loader, DEVICE)
        print(
            f"\n[Epoch {epoch+1}] F1 {metrics['f1']:.4f} | "
            f"Acc {metrics['accuracy']:.4f} | "
            f"Prec {metrics['precision']:.4f} | "
            f"Rec {metrics['recall']:.4f} | "
            f"AUC {metrics['roc_auc']:.4f} | "
            f"Best F1 so far: {best_f1:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt = os.path.join(MODEL_SAVE_DIR, "best_hybrid.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ New best F1 {best_f1:.4f} — saved to {ckpt}")

    print(f"\n[train_hybrid] Done. Best dev F1: {best_f1:.4f}")
    print(f"  Checkpoint: {os.path.join(MODEL_SAVE_DIR, 'best_hybrid.pt')}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
