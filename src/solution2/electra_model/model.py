"""ELECTRA-large cross-encoder for authorship verification."""

import torch
import torch.nn as nn
from transformers import AutoModel


def load_model(model_name: str, checkpoint_path: str | None = None, token=None):
    """
    Convenience loader.
      - If checkpoint_path is None  → returns freshly initialised model.
      - If checkpoint_path is given → loads saved state_dict on top.
    """
    model = AVCrossEncoder(model_name, token=token)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[model] Loaded weights from {checkpoint_path}")
    return model


class AVCrossEncoder(nn.Module):
    """
    ELECTRA-large cross-encoder for Authorship Verification.
    
    Architecture:
    - Input: [CLS] text_1 [SEP] text_2 [SEP]
    - Encoder: ELECTRA-large discriminator (358M params)
    - Classifier: hidden_size → hidden_size//2 (ReLU) → 1
    - Output: logits (pass to BCEWithLogitsLoss for training)
    """
    
    def __init__(self, model_name: str = "google/electra-large-discriminator", token=None):
        super().__init__()
        self.model_name = model_name
        
        # Load ELECTRA encoder
        self.encoder = AutoModel.from_pretrained(model_name, token=token)
        self.hidden_size = self.encoder.config.hidden_size  # 1024
        
        # Classifier: hidden → hidden//2 → 1
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1),
        )
        
        print(f"[AVCrossEncoder] Loaded {model_name}")
        print(f"[AVCrossEncoder] Hidden size: {self.hidden_size}")
        print(f"[AVCrossEncoder] Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) - token IDs
            attention_mask: (batch_size, seq_len) - 1 for real, 0 for padding
            token_type_ids: (batch_size, seq_len) - segment IDs (ELECTRA uses these)
        
        Returns:
            logits: (batch_size, 1) - raw scores
        """
        # Encode the concatenated pair
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Get [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Classify
        logits = self.classifier(cls_repr).squeeze(-1)  # (batch_size,)
        
        return logits
