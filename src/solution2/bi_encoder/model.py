"""
model.py — Siamese Bi-Encoder for Authorship Verification (Track C)

Architecture:
  - Passes text_1 and text_2 through the SAME transformer independently.
  - Extracts the mean-pooled embedding for each text.
  - Computes the absolute difference and (optionally) element-wise product.
  - Passes the combined features through a linear classifier.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BiEncoder(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size # Embedding size (324 for MiniLM)

        # Classifier head: takes [u, v, |u-v| ] as input (3 x hidden_size)
        # We pass into u, v and |u-v| so that the model can learn about the interaction between the two texts 
        # https://arxiv.org/abs/1908.10084 (reference for this architecture)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) # Single logit for BCEWithLogitsLoss
        )

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Batching may return false tokens, so we need to mask them out using an attention mask
        This means we average over the true tokens (with mask 1) and ignore the padding tokens (with mask 0)
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor
    ) -> torch.Tensor:
        
        # Pass both texts through the shared encoder independently (known as u, v)
        out_1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
        out_2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Extract sentence embeddings (mean pooling)
        u = self.mean_pooling(out_1, attention_mask_1) # (batch, hidden_size)
        v = self.mean_pooling(out_2, attention_mask_2) # (batch, hidden_size)

        # Compute interaction features
        # Common heuristic: concatenate u, v, and absolute difference |u - v|
        abs_diff = torch.abs(u - v)
        features = torch.cat([u, v, abs_diff], dim=1) # (batch, 3 * hidden_size)

        #  Classification
        logits = self.classifier(features).squeeze(-1) # (batch,)
        
        return logits

def load_local_model(model_name: str, checkpoint_path: str = None) -> BiEncoder:
    """Convenience loader for the Bi-Encoder."""
    model = BiEncoder(model_name=model_name)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[model] Loaded Bi-Encoder weights from {checkpoint_path}")
    return model
