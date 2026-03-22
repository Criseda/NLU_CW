"""
hybrid_dataset.py — Dataset for the Hybrid Bi-Encoder.

Loads a CSV (text_1, text_2, label) AND the corresponding .npy feature
matrix row-aligned. Each item returned contains:
  - input_ids_1 / attention_mask_1  (text_1 tokenised)
  - input_ids_2 / attention_mask_2  (text_2 tokenised)
  - hand_features                   (97-dim float tensor)
  - labels                          (scalar float)

The .npy file MUST have the same number of rows as the CSV.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HybridAVDataset(Dataset):
    """
    Parameters
    ----------
    csv_path      : Path to the AV CSV (columns: text_1, text_2, label)
    features_path : Path to the .npy feature matrix, shape (N, 97)
    tokenizer     : HuggingFace tokenizer
    max_length    : Max tokens per text (texts are encoded INDEPENDENTLY)
    """

    def __init__(
        self,
        csv_path: str,
        features_path: str,
        tokenizer,
        max_length: int,
    ):
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"Empty CSV: {csv_path}"

        # Load pre-computed features
        feats = np.load(features_path).astype(np.float32)
        assert feats.shape[0] == len(df), (
            f"Feature matrix row count ({feats.shape[0]}) does not match "
            f"CSV row count ({len(df)}) — files are misaligned!"
        )
        # Clamp any inf / nan before converting to tensors
        feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
        self.features = torch.from_numpy(feats)          # (N, 97)
        self.labels   = torch.tensor(
            df["label"].astype(float).tolist(), dtype=torch.float
        )

        # Tokenise text_1
        self.enc1 = tokenizer(
            df["text_1"].fillna("").tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenise text_2
        self.enc2 = tokenizer(
            df["text_2"].fillna("").tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids_1":      self.enc1["input_ids"][idx],
            "attention_mask_1": self.enc1["attention_mask"][idx],
            "input_ids_2":      self.enc2["input_ids"][idx],
            "attention_mask_2": self.enc2["attention_mask"][idx],
            "hand_features":    self.features[idx],       # (97,)
            "labels":           self.labels[idx],
        }
