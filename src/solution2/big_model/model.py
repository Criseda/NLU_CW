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
        hidden_size = config.hidden_size              # 1024 for deberta-v3-large

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),          # single logit
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
        cls_repr = outputs.last_hidden_state[:, 0, :]   # (batch, hidden_size)
        logits = self.classifier(cls_repr).squeeze(-1)  # (batch,)
        return logits


def load_model(model_name: str, checkpoint_path: str | None = None, token: str | None = None) -> AVCrossEncoder:
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


class AVTripletEncoder(nn.Module):
    """
    DeBERTa-v3-large triplet encoder for Authorship Verification.
    
    Fuses DeBERTa [CLS] representation (1024 dims) with 97-dim stylometric features
    into a learned embedding space for triplet loss training.
    
    Architecture:
      [CLS] repr (1024) + Stylometric features (97)
        ↓
      Feature projector (97 → 256)
        ↓
      Concatenate: [1024 + 256] = 1280
        ↓
      Fusion MLP: 1280 → 512 → embedding_dim (256)
        ↓
      Normalized embedding for triplet loss
    """
    
    def __init__(
        self,
        model_name: str,
        feature_dim: int = 97,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        token: str | None = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, token=token)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        
        self.encoder = AutoModel.from_pretrained(model_name, config=config, token=token)
        hidden_size = config.hidden_size  # 1024 for deberta-v3-large
        self.embedding_dim = embedding_dim
        
        # Project stylometric features to a meaningful dimension
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
        )
        
        # Fuse text representation + projected features
        fusion_input_dim = hidden_size + 256  # 1024 + 256 = 1280
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
        
        # L2 normalization layer
        self.normalize = nn.functional.normalize
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_ids       : (batch, seq_len)
            attention_mask  : (batch, seq_len)
            features        : (batch, feature_dim) — stylometric features
            token_type_ids  : (batch, seq_len) — optional
            normalize       : whether to L2-normalize output embeddings
        
        Returns:
            embeddings      : (batch, embedding_dim) — learned embeddings
        """
        # Encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (batch, 1024)
        
        # Project features
        proj_features = self.feature_projector(features)  # (batch, 256)
        
        # Fuse representations
        fused = torch.cat([cls_repr, proj_features], dim=1)  # (batch, 1280)
        embedding = self.fusion(fused)  # (batch, embedding_dim)
        
        # Normalize for better triplet loss behavior
        if normalize:
            embedding = self.normalize(embedding, p=2, dim=1)
        
        return embedding


def load_triplet_model(
    model_name: str,
    feature_dim: int = 97,
    embedding_dim: int = 256,
    checkpoint_path: str | None = None,
    token: str | None = None,
) -> AVTripletEncoder:
    """
    Convenience loader for triplet encoder.
      - If checkpoint_path is None  → returns freshly initialised model.
      - If checkpoint_path is given → loads saved state_dict on top.
    """
    model = AVTripletEncoder(
        model_name=model_name,
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        token=token,
    )
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[model] Loaded triplet encoder from {checkpoint_path}")
    return model