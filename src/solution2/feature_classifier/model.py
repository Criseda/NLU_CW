"""
feature_model.py — Feature-based MLP classifier for the AV task.

Takes 97 hand-crafted stylometric/info-theoretic/impostor features as input
and classifies text pairs as same-author (1) or different-author (0).

Architecture: MLP with batch norm, dropout, and residual skip.

This is trained separately from the BiEncoder (train.py) via train_features.py.
"""

import torch
import torch.nn as nn


class FeatureClassifier(nn.Module):
    """
    Lightweight MLP that operates on a fixed feature vector.

    Input:  (batch, n_features)  — default 97
    Output: (batch,)             — single logit for BCEWithLogitsLoss
    """

    def __init__(self, n_features: int = 97, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)
