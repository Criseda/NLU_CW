"""
predict_roberta.py — Inference with RoBERTa-large cross-encoder.

Writes probability files for model evaluation:
    outputs/solution2/roberta_model/probs_{split}.csv

Usage:
    # Predict on dev set (default):
    python -m src.solution2.roberta_model.predict_roberta

    # Predict on a specific CSV:
    python -m src.solution2.roberta_model.predict_roberta --input path/to/file.csv --split test
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

from . import config
from .model import load_model


class AVInferenceDataset(Dataset):
    """Simple dataset for inference."""

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
        # Store gold labels if present (for quick sanity check)
        self.labels = self.df["label"].tolist() if "label" in self.df.columns else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item


@torch.no_grad()
def predict_probs(
    csv_path: str,
    checkpoint_path: str,
    split: str = "val",
):
    """Load a CSV, run the model, and save probabilities and predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    dataset = AVInferenceDataset(csv_path, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = load_model(config.MODEL_NAME, checkpoint_path=checkpoint_path, token=config.HF_TOKEN).to(device)
    model.eval()

    all_probs = []
    all_preds = []

    for batch in tqdm(loader, desc=f"Predicting ({split})"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs.flatten())
        all_preds.extend((probs > 0.5).astype(int).flatten())

    results = pd.DataFrame({
        "probability": np.round(all_probs, 6),
        "prediction": all_preds,
    })

    # ── Write output ───────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, f"probs_{split}.csv")
    results.to_csv(out_path, index=False)
    print(f"[predict] Saved → {out_path}")

    # Quick metrics sanity check if labels available
    if dataset.labels is not None:
        from sklearn.metrics import f1_score, accuracy_score
        labels = np.array(dataset.labels)
        preds = results["prediction"].values
        f1_binary = f1_score(labels, preds, average="binary")
        f1_macro = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        print(
            f"[predict] F1 (binary): {f1_binary:.4f} | "
            f"F1 (macro): {f1_macro:.4f} | "
            f"Acc: {acc:.4f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=config.DEV_FILE,
        help="Path to input CSV (text_1, text_2[, label])"
    )
    parser.add_argument(
        "--checkpoint", default=os.path.join(config.MODEL_SAVE_DIR, "best_model.pt"),
        help="Path to saved model weights (.pt)"
    )
    parser.add_argument(
        "--split", default="val",
        help="Label for output file: probs_{split}.csv"
    )
    args = parser.parse_args()

    predict_probs(
        csv_path=args.input,
        checkpoint_path=args.checkpoint,
        split=args.split,
    )


if __name__ == "__main__":
    main()
