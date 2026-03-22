"""
hybrid_model.py — Hybrid Bi-Encoder that fuses transformer embeddings
                  with 97 handcrafted stylometric features.

Architecture:
  text_1 → shared MiniLM encoder → mean pool → u  ─┐
                                                      ├─ [u, v, |u-v|, hand_features]
  text_2 → shared MiniLM encoder → mean pool → v  ─┘         │
                                                      classifier head → logit

The classifier head input size = 3 * hidden_size + n_features
  = 3 * 384 + 97 = 1249  (for all-MiniLM-L6-v2)

This is a drop-in sibling of BiEncoder (model.py) — completely independent,
same interface but with an extra `hand_features` argument to forward().
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class HybridBiEncoder(nn.Module):
    """
    Siamese Bi-Encoder whose classifier head receives both transformer
    embeddings AND a vector of hand-crafted features.

    Parameters
    ----------
    model_name   : HuggingFace model ID (default: all-MiniLM-L6-v2)
    n_features   : Number of hand-crafted features (default: 97)
    dropout      : Dropout probability applied inside classifier
    """

    def __init__(
        self,
        model_name: str,
        n_features: int = 97,
        dropout: float = 0.1,
    ):
        super().__init__()

        cfg = AutoConfig.from_pretrained(model_name)
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob          = dropout

        self.encoder    = AutoModel.from_pretrained(model_name, config=cfg)
        hidden_size     = cfg.hidden_size  # 384 for MiniLM-L6

        # Input to classifier = [u, v, |u-v|] + hand_features
        classifier_input = hidden_size * 3 + n_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # single logit → BCEWithLogitsLoss
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings, masked by attention mask."""
        token_emb    = model_output[0]  # (batch, seq_len, hidden)
        mask_exp     = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids_1:      torch.Tensor,   # (batch, seq_len)
        attention_mask_1: torch.Tensor,
        input_ids_2:      torch.Tensor,
        attention_mask_2: torch.Tensor,
        hand_features:    torch.Tensor,   # (batch, n_features)  ← new!
    ) -> torch.Tensor:                    # (batch,) logit

        # Encode both texts with the shared transformer
        out_1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
        out_2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Mean-pooled sentence embeddings
        u = self.mean_pooling(out_1, attention_mask_1)  # (batch, hidden)
        v = self.mean_pooling(out_2, attention_mask_2)

        # Interaction features from the transformer side
        abs_diff = torch.abs(u - v)                     # (batch, hidden)

        # Concatenate everything: transformer side + hand-crafted side
        combined = torch.cat([u, v, abs_diff, hand_features], dim=1)  # (batch, 3*hidden + n_feat)

        return self.classifier(combined).squeeze(-1)    # (batch,)
