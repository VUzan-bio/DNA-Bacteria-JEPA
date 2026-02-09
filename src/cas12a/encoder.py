"""
SparseTransformerEncoder used by pretraining and downstream tasks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class SparseTransformerEncoder(nn.Module):
    """
    Lightweight transformer encoder that returns pooled sequence embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def get_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens != self.pad_token_id

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_token_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len or truncate inputs before encoding."
            )

        if attention_mask is None:
            attention_mask = self.get_attention_mask(tokens)

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        x = self.dropout(x)

        key_padding_mask = ~attention_mask.bool()
        token_embeddings = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        info: Dict[str, torch.Tensor] = {
            "attention_mask": attention_mask,
        }
        if return_token_embeddings:
            info["token_embeddings"] = token_embeddings

        return pooled, info
