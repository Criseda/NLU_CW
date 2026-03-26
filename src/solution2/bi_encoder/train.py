"""
train.py — Train the local Bi-Encoder small model on the AV task.

Usage:
    python -m src.solution2.small_model.train
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from . import config
from .model import BiEncoder

from ..shared.utils import set_seed, get_device

# ── Dataset ────────────────────────────────────────────────────────────────────

class AVDataset(Dataset):
    """
    Loads an AV CSV (columns: text_1, text_2, label) and tokenises pairs
    INDEPENDENTLY for the Bi-Encoder.
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int):
        df = pd.read_csv(csv_path)
        self.labels = df["label"].astype(float).tolist()
        
        # Tokenize text_1
        self.encodings_1 = tokenizer(
            df["text_1"].fillna("").tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize text_2
        self.encodings_2 = tokenizer(
            df["text_2"].fillna("").tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids_1": self.encodings_1["input_ids"][idx],
            "attention_mask_1": self.encodings_1["attention_mask"][idx],
            "input_ids_2": self.encodings_2["input_ids"][idx],
            "attention_mask_2": self.encodings_2["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

# ── Evaluation helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: BiEncoder, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids_1      = batch["input_ids_1"].to(device)
        attention_mask_1 = batch["attention_mask_1"].to(device)
        input_ids_2      = batch["input_ids_2"].to(device)
        attention_mask_2 = batch["attention_mask_2"].to(device)

        logits = model(
            input_ids_1=input_ids_1, 
            attention_mask_1=attention_mask_1, 
            input_ids_2=input_ids_2, 
            attention_mask_2=attention_mask_2
        )
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    preds      = (all_logits > 0).astype(int)    # threshold at 0 (pre-sigmoid)

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

    return {
        "f1":        f1_score(all_labels, preds),
        "accuracy":  accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall":    recall_score(all_labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_labels, probs),
    }

# ── Training loop ──────────────────────────────────────────────────────────────

def train() -> None:
    set_seed(config.SEED)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f"[train] Device: {device}")

    # ── Tokeniser & datasets ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_ds = AVDataset(config.TRAIN_FILE, tokenizer, config.MAX_LENGTH)
    dev_ds   = AVDataset(config.DEV_FILE,   tokenizer, config.MAX_LENGTH)

    # Use 0 workers on Mac MPS backend to avoid multi-processing hanging bugs
    num_workers = 0 if device.type == "mps" else 2

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds,   batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"[train] Train: {len(train_ds)} samples | Dev: {len(dev_ds)} samples")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = BiEncoder(model_name=config.MODEL_NAME).to(device)

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    total_steps   = (len(train_loader) // config.GRAD_ACCUM) * config.EPOCHS
    warmup_steps  = int(total_steps * config.WARMUP_RATIO)

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # We do not use mixed precision (AMP) locally for simplicity, especially on MPS
    criterion = nn.BCEWithLogitsLoss()

    # ── Training ───────────────────────────────────────────────────────────────
    best_f1, global_step = 0.0, 0

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")):
            input_ids_1      = batch["input_ids_1"].to(device)
            attention_mask_1 = batch["attention_mask_1"].to(device)
            input_ids_2      = batch["input_ids_2"].to(device)
            attention_mask_2 = batch["attention_mask_2"].to(device)
            labels           = batch["labels"].to(device)

            logits = model(
                input_ids_1=input_ids_1, 
                attention_mask_1=attention_mask_1, 
                input_ids_2=input_ids_2, 
                attention_mask_2=attention_mask_2
            )
            loss = criterion(logits, labels) / config.GRAD_ACCUM
            loss.backward()
            
            running_loss += loss.item() * config.GRAD_ACCUM

            # Gradient accumulation
            if (step + 1) % config.GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Evaluate periodically ─────────────────────────────────────
                if global_step % config.EVAL_STEPS == 0:
                    metrics = evaluate(model, dev_loader, device)
                    print(
                        f"\n  Step {global_step} | "
                        f"Loss {running_loss / config.EVAL_STEPS:.4f} | "
                        f"F1 {metrics['f1']:.4f} | "
                        f"Acc {metrics['accuracy']:.4f} | "
                        f"Prec {metrics['precision']:.4f} | "
                        f"Rec {metrics['recall']:.4f} | "
                        f"AUC {metrics['roc_auc']:.4f}"
                    )
                    running_loss = 0.0
                    model.train()

                    # Save best checkpoint
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        ckpt_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pt")
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"  ✓ New best F1 {best_f1:.4f} — saved to {ckpt_path}")

        # End-of-epoch evaluation
        metrics = evaluate(model, dev_loader, device)
        print(
            f"\n[Epoch {epoch+1}] F1 {metrics['f1']:.4f} | "
            f"Acc {metrics['accuracy']:.4f} | "
            f"Prec {metrics['precision']:.4f} | "
            f"Rec {metrics['recall']:.4f} | "
            f"AUC {metrics['roc_auc']:.4f} | "
            f"Best F1 so far: {best_f1:.4f}"
        )

    print(f"\n[train] Done. Best dev F1: {best_f1:.4f}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
