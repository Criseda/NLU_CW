"""
train.py — Fine-tune DeBERTa-v3-large cross-encoder on the AV task.

Usage:
    python -m src.solution2.big_model.train

All settings live in config.py — edit there, not here.
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
from .model import AVTripletEncoder
from .feature_loader import load_and_normalize_features, load_features_optional


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ────────────────────────────────────────────────────────────────────

class AVTripletDataset(Dataset):
    """
    Loads an AV CSV (columns: text_1, text_2, label) and creates triplets:
      - Anchor: text_1
      - Positive: text_2 (if label=1, same author) or a randomly sampled same-author text
      - Negative: a randomly sampled different-author text
    
    Tokenizes all three and loads pre-computed 97-dim features.
    """

    def __init__(self, csv_path: str, features_path: str, tokenizer, max_length: int):
        df = pd.read_csv(csv_path)
        self.texts_1 = df["text_1"].fillna("").tolist()
        self.texts_2 = df["text_2"].fillna("").tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load features
        self.features = load_and_normalize_features(features_path)
        assert len(self.features) == len(self.labels), \
            f"Feature count {len(self.features)} != Label count {len(self.labels)}"
        
        # Pre-compute positive and negative indices for efficiency
        self.positive_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(self.labels) if label == 0]
        
        if not self.positive_indices or not self.negative_indices:
            raise ValueError("Dataset must contain both positive and negative examples")
    
    def __len__(self) -> int:
        return len(self.positive_indices)
    
    def __getitem__(self, idx: int) -> dict:
        # Use positive examples as anchors (these are harder to learn)
        anchor_idx = self.positive_indices[idx]
        
        # Positive: the paired text for this anchor
        pos_idx = anchor_idx  # We'll use text_2 as positive
        
        # Negative: randomly sample from negative examples
        neg_idx = random.choice(self.negative_indices)
        
        # Encode anchor and positive (both directions of the same pair)
        anchor_text_1 = self.texts_1[anchor_idx]
        anchor_text_2 = self.texts_2[anchor_idx]
        
        # Anchor: text_1, Positive: text_2 (they should be same author, label=1)
        anchor_encoding = self.tokenizer(
            anchor_text_1,
            anchor_text_2,
            max_length=self.max_length,
            truncation=config.TRUNCATION,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Positive: use the pair in reverse (text_2 as "anchor", but we'll treat carefully)
        # Actually, for triplet loss, positive should be another instance of same author
        # Here we use the paired text_2 from the positive pair
        pos_text_1 = self.texts_2[pos_idx]  # text_2 from positive pair
        pos_text_2 = self.texts_1[pos_idx]  # text_1 from positive pair (reverse)
        
        pos_encoding = self.tokenizer(
            pos_text_1,
            pos_text_2,
            max_length=self.max_length,
            truncation=config.TRUNCATION,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Negative: text_1 from negative pair paired with random text
        neg_text_1 = self.texts_1[neg_idx]
        neg_text_2 = self.texts_2[neg_idx]
        
        neg_encoding = self.tokenizer(
            neg_text_1,
            neg_text_2,
            max_length=self.max_length,
            truncation=config.TRUNCATION,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "anchor_input_ids": anchor_encoding["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_encoding["attention_mask"].squeeze(0),
            "anchor_token_type_ids": anchor_encoding.get("token_type_ids", torch.zeros_like(anchor_encoding["input_ids"])).squeeze(0),
            "anchor_features": torch.tensor(self.features[anchor_idx], dtype=torch.float32),
            
            "pos_input_ids": pos_encoding["input_ids"].squeeze(0),
            "pos_attention_mask": pos_encoding["attention_mask"].squeeze(0),
            "pos_token_type_ids": pos_encoding.get("token_type_ids", torch.zeros_like(pos_encoding["input_ids"])).squeeze(0),
            "pos_features": torch.tensor(self.features[pos_idx], dtype=torch.float32),
            
            "neg_input_ids": neg_encoding["input_ids"].squeeze(0),
            "neg_attention_mask": neg_encoding["attention_mask"].squeeze(0),
            "neg_token_type_ids": neg_encoding.get("token_type_ids", torch.zeros_like(neg_encoding["input_ids"])).squeeze(0),
            "neg_features": torch.tensor(self.features[neg_idx], dtype=torch.float32),
        }



# ── Triplet Loss ───────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet loss: L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Pulls same-author embeddings closer and pushes different-author embeddings apart.
    Uses L2 distance in the embedding space.
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor   : (batch, embedding_dim) — anchor embeddings
            positive : (batch, embedding_dim) — positive (same-author) embeddings
            negative : (batch, embedding_dim) — negative (different-author) embeddings
        
        Returns:
            loss     : scalar tensor
        """
        # Compute pairwise L2 distances
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # (batch,)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # (batch,)
        
        # Triplet loss with hard margin
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ── Evaluation helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: "AVTripletEncoder", loader: DataLoader, device: torch.device) -> dict:
    """
    Evaluate triplet encoder on dev set.
    
    Computes embeddings for text pairs and uses cosine similarity threshold
    to classify as same/different author.
    """
    model.eval()
    all_sims, all_labels = [], []
    
    for batch in loader:
        # Anchor embeddings
        anchor_input_ids = batch["anchor_input_ids"].to(device)
        anchor_attention_mask = batch["anchor_attention_mask"].to(device)
        anchor_features = batch["anchor_features"].to(device)
        anchor_token_type_ids = batch.get("anchor_token_type_ids")
        if anchor_token_type_ids is not None:
            anchor_token_type_ids = anchor_token_type_ids.to(device)
        
        # Positive embeddings
        pos_input_ids = batch["pos_input_ids"].to(device)
        pos_attention_mask = batch["pos_attention_mask"].to(device)
        pos_features = batch["pos_features"].to(device)
        pos_token_type_ids = batch.get("pos_token_type_ids")
        if pos_token_type_ids is not None:
            pos_token_type_ids = pos_token_type_ids.to(device)
        
        # Get embeddings
        anchor_emb = model(anchor_input_ids, anchor_attention_mask, anchor_features, anchor_token_type_ids)
        pos_emb = model(pos_input_ids, pos_attention_mask, pos_features, pos_token_type_ids)
        
        # Compute cosine similarity
        sim = torch.nn.functional.cosine_similarity(anchor_emb, pos_emb, dim=1)  # (batch,)
        all_sims.append(sim.cpu())
        
        # Labels: all positives in this batch (triplet dataset only has positives)
        all_labels.append(torch.ones(sim.shape[0]))
    
    sims = torch.cat(all_sims).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    
    # Threshold at 0.5 for binary classification
    preds = (sims > 0.5).astype(int)
    
    return {
        "f1":        f1_score(labels, preds, zero_division=0),
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels, sims),
        "mean_sim":  sims.mean(),
    }



# ── Training loop ──────────────────────────────────────────────────────────────

def train() -> None:
    set_seed(config.SEED)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Tokeniser & datasets ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)

    train_ds = AVTripletDataset(config.TRAIN_FILE, config.FEATURES_TRAIN, tokenizer, config.MAX_LENGTH)
    dev_ds   = AVTripletDataset(config.DEV_FILE, config.FEATURES_TRAIN, tokenizer, config.MAX_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds,   batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"[train] Train: {len(train_ds)} triplets | Dev: {len(dev_ds)} triplets")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AVTripletEncoder(
        model_name=config.MODEL_NAME,
        feature_dim=config.FEATURE_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        token=config.HF_TOKEN,
    ).to(device)
    model = model.float()

    if torch.cuda.device_count() > 1:
        print(f"[train] Using {torch.cuda.device_count()} GPUs via nn.DataParallel!")
        model = nn.DataParallel(model)

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
    scaler    = torch.amp.GradScaler('cuda', enabled=config.FP16)
    criterion = TripletLoss(margin=config.TRIPLET_MARGIN)

    # ── Training ───────────────────────────────────────────────────────────────
    best_f1, global_step = 0.0, 0

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")):
            # Unpack triplet batch
            anchor_input_ids = batch["anchor_input_ids"].to(device)
            anchor_attention_mask = batch["anchor_attention_mask"].to(device)
            anchor_token_type_ids = batch.get("anchor_token_type_ids")
            if anchor_token_type_ids is not None:
                anchor_token_type_ids = anchor_token_type_ids.to(device)
            anchor_features = batch["anchor_features"].to(device)
            
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            pos_token_type_ids = batch.get("pos_token_type_ids")
            if pos_token_type_ids is not None:
                pos_token_type_ids = pos_token_type_ids.to(device)
            pos_features = batch["pos_features"].to(device)
            
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)
            neg_token_type_ids = batch.get("neg_token_type_ids")
            if neg_token_type_ids is not None:
                neg_token_type_ids = neg_token_type_ids.to(device)
            neg_features = batch["neg_features"].to(device)

            with torch.amp.autocast('cuda', enabled=config.FP16):
                # Forward pass
                anchor_emb = model(anchor_input_ids, anchor_attention_mask, anchor_features, anchor_token_type_ids)
                pos_emb = model(pos_input_ids, pos_attention_mask, pos_features, pos_token_type_ids)
                neg_emb = model(neg_input_ids, neg_attention_mask, neg_features, neg_token_type_ids)
                
                # Triplet loss
                loss = criterion(anchor_emb, pos_emb, neg_emb) / config.GRAD_ACCUM

            scaler.scale(loss).backward()
            running_loss += loss.item() * config.GRAD_ACCUM

            # Gradient accumulation
            if (step + 1) % config.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
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
                        f"AUC {metrics['roc_auc']:.4f} | "
                        f"Mean Sim {metrics['mean_sim']:.4f}"
                    )
                    running_loss = 0.0
                    model.train()

                    # Save best checkpoint
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        ckpt_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pt")
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), ckpt_path)
                        print(f"  ✓ New best F1 {best_f1:.4f} — saved to {ckpt_path}")

        # End-of-epoch evaluation
        metrics = evaluate(model, dev_loader, device)
        print(
            f"\n[Epoch {epoch+1}] F1 {metrics['f1']:.4f} | "
            f"Acc {metrics['accuracy']:.4f} | "
            f"Prec {metrics['precision']:.4f} | "
            f"Rec {metrics['recall']:.4f} | "
            f"AUC {metrics['roc_auc']:.4f} | "
            f"Mean Sim {metrics['mean_sim']:.4f} | "
            f"Best F1 so far: {best_f1:.4f}"
        )

    print(f"\n[train] Done. Best dev F1: {best_f1:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()