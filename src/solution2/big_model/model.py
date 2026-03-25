"""
model.py — DeBERTa-v3-large cross-encoder for Authorship Verification.

Architecture:
  Input : [CLS] text_1 [SEP] text_2 [SEP]   (standard cross-encoder)
  Output: single logit → apply sigmoid → P(same author)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class AVCrossEncoder(nn.Module):
    """
    DeBERTa-v3-large cross-encoder that takes a concatenated text pair and
    outputs a single logit for binary classification (same-author / not).

    Use BCEWithLogitsLoss during training — sigmoid is NOT applied here so
    that the loss function stays numerically stable.
    """

    def __init__(self, model_name: str, dropout: float = 0.1, token: str | None = None):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, token=token)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout

        self.encoder = AutoModel.from_pretrained(model_name, config=config, token=token)
        hidden_size = config.hidden_size  # 1024 for deberta-v3-large

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),  # single logit
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids       : (batch, seq_len)
            attention_mask  : (batch, seq_len)
            token_type_ids  : (batch, seq_len) — optional, DeBERTa ignores it

        Returns:
            logits          : (batch,)   — raw (pre-sigmoid) scores
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use the [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        logits = self.classifier(cls_repr).squeeze(-1)  # (batch,)
        return logits


def load_model(
    model_name: str, checkpoint_path: str | None = None, token: str | None = None
) -> AVCrossEncoder:
    """
    Convenience loader.
      - If checkpoint_path is None  → returns freshly initialised model.
      - If checkpoint_path is given → loads saved state_dict on top.
    """
    model = AVCrossEncoder(model_name=model_name, token=token)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[model] Loaded weights from {checkpoint_path}")
    return model
