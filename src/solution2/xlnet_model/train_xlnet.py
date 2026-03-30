"""
train_xlnet.py — Train XLNet-large cross-encoder for AV (NO stylistic features).

Architecture: Pure semantic learning via permutation LM
Hypothesis: XLNet's permutation language modeling alone captures sufficient stylistic 
            signal without explicit feature engineering.

Usage:
    python -m src.solution2.xlnet_model.train_xlnet
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch.nn as nn

from . import config
from .model import load_model


# ── Set random seeds ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ────────────────────────────────────────────────────────────────────

class AVDataset(Dataset):
    """Simple dataset for AV task."""

    def __init__(self, csv_path: str, tokenizer, max_length: int):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.encodings = tokenizer(
            self.df["text_1"].fillna("").tolist(),
            self.df["text_2"].fillna("").tolist(),
            max_length=max_length,
            truncation=config.TRUNCATION,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(self.df["label"].tolist(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["label"] = self.labels[idx]
        return item


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        labels = batch["label"].to(device)
        
        # Debug: print first batch info
        if batch_idx == 0:
            print(f"[train_epoch] Batch shape - input_ids: {input_ids.shape}, labels: {labels.shape}")
        
        # Forward pass
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate on a dataset, return metrics."""
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    all_logits, all_labels = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        labels = batch["label"].to(device)
        
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    from src.evaluation.evaluate import compute_metrics
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds_05 = (probs > 0.5).astype(int)
    
    comp_metrics = compute_metrics(all_labels, preds_05, probs)
    metrics = {
        "loss": total_loss / len(loader),
        "f1": comp_metrics["f1_macro"],
        "acc": comp_metrics["accuracy"],
        "precision": comp_metrics["precision"],
        "recall": comp_metrics["recall"],
    }
    
    return metrics


def main():
    """Main training loop."""
    set_seed(config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    
    # Load data
    print(f"[train] Loading data from {config.TRAIN_FILE}")
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_dev = pd.read_csv(config.DEV_FILE)
    print(f"[train] Train: {len(df_train)} | Dev: {len(df_dev)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    
    # Datasets & loaders
    train_dataset = AVDataset(config.TRAIN_FILE, tokenizer, config.MAX_LENGTH)
    dev_dataset = AVDataset(config.DEV_FILE, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Model
    model = load_model(config.MODEL_NAME, token=config.HF_TOKEN).to(device)
    print(f"[train] Model loaded: {config.MODEL_NAME}")
    
    # ── Multi-GPU support via DataParallel ─────────────────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"[train] Using {torch.cuda.device_count()} GPUs via nn.DataParallel!")
        model = nn.DataParallel(model)
    
    print(f"[train] Model ready. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & scheduler
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"[train] Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    
    # Create output dir
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Training loop
    best_f1 = -1.0
    best_checkpoint = None
    patience_counter = 0
    MAX_PATIENCE = 3
    
    for epoch in range(config.EPOCHS):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"[train] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
        # Evaluate
        dev_metrics = evaluate(model, dev_loader, device)
        print(
            f"[eval]  F1: {dev_metrics['f1']:.4f} | "
            f"Acc: {dev_metrics['acc']:.4f} | "
            f"Precision: {dev_metrics['precision']:.4f} | "
            f"Recall: {dev_metrics['recall']:.4f}"
        )
        
        # Save checkpoint if best F1
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_checkpoint = os.path.join(config.MODEL_SAVE_DIR, "best_model.pt")
            
            # Handle DataParallel: save unwrapped state dict
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), best_checkpoint)
            
            print(f"[train] Saved best model to {best_checkpoint}")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            print(f"[train] No improvement. Patience: {patience_counter}/{MAX_PATIENCE}")
            
            if patience_counter >= MAX_PATIENCE:
                print(f"[train] Early stopping triggered! No F1 improvement for {MAX_PATIENCE} epochs.")
                break
    
    print(f"\n[train] Training complete! Best F1: {best_f1:.4f}")
    print(f"[train] Best model saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()
