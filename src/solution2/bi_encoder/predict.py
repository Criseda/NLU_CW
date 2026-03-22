"""
predict.py — Run inference with the saved local Bi-Encoder small model.

Writes probability files that the ensemble will consume later:
    outputs/solution2/small_probs_val.csv
    outputs/solution2/small_probs_test.csv

Usage:
    # Predict on dev set (default):
    python -m src.solution2.small_model.predict

    # Predict on a specific CSV:
    python -m src.solution2.small_model.predict --input path/to/file.csv --split test
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from . import config
from .model import load_local_model


# ── Dataset (inference — no labels required) ───────────────────────────────────

class AVInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.pair_ids = df.index.tolist()
        
        # Bi-Encoder tokenizes text_1 and text_2 separately
        self.encodings_1 = tokenizer(
            df["text_1"].fillna("").tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.encodings_2 = tokenizer(
            df["text_2"].fillna("").tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Store gold labels if present (for quick sanity check)
        self.labels = df["label"].tolist() if "label" in df.columns else None

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids_1": self.encodings_1["input_ids"][idx],
            "attention_mask_1": self.encodings_1["attention_mask"][idx],
            "input_ids_2": self.encodings_2["input_ids"][idx],
            "attention_mask_2": self.encodings_2["attention_mask"][idx],
            "pair_id": self.pair_ids[idx]
        }
        return item


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_probs(
    csv_path: str,
    checkpoint_path: str,
    split: str = "val",
) -> pd.DataFrame:
    """
    Load a CSV, run the model, and return a DataFrame with columns:
        pair_id | prob | pred
    Also writes the result to outputs/solution2/small_model/small_probs_{split}.csv.
    """
    device = torch.device(config.DEVICE)
    print(f"[predict] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    df        = pd.read_csv(csv_path)

    dataset = AVInferenceDataset(df, tokenizer, config.MAX_LENGTH)
    
    num_workers = 0 if config.DEVICE == "mps" else 2
    loader  = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = load_local_model(config.MODEL_NAME, checkpoint_path=checkpoint_path).to(device)
    model.eval()

    all_ids, all_probs = [], []

    for batch in tqdm(loader, desc=f"Predicting ({split})"):
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
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_ids.extend(batch["pair_id"].tolist())
        all_probs.extend(probs.tolist())

    results = pd.DataFrame({
        "pair_id": all_ids,
        "prob":    np.round(all_probs, 6),
        "pred":    (np.array(all_probs) > 0.5).astype(int),
    })

    # ── Write output ───────────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, f"small_probs_{split}.csv")
    results.to_csv(out_path, index=False)
    print(f"[predict] Saved → {out_path}")

    # Quick F1 sanity check if labels available
    if dataset.labels is not None:
        from sklearn.metrics import f1_score, accuracy_score
        labels = np.array(dataset.labels)
        preds  = results["pred"].values
        print(
            f"[predict] F1: {f1_score(labels, preds):.4f} | "
            f"Acc: {accuracy_score(labels, preds):.4f}"
        )

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",  default=config.DEV_FILE,
        help="Path to input CSV (text_1, text_2[, label])"
    )
    parser.add_argument(
        "--checkpoint", default=os.path.join(config.MODEL_SAVE_DIR, "best_model.pt"),
        help="Path to saved model weights (.pt)"
    )
    parser.add_argument(
        "--split", default="val",
        help="Label for output file: small_probs_{split}.csv"
    )
    args = parser.parse_args()

    predict_probs(
        csv_path=args.input,
        checkpoint_path=args.checkpoint,
        split=args.split,
    )


if __name__ == "__main__":
    main()
