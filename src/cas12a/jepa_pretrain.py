"""
Minimal JEPA-style pretraining wrapper.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class MaskingConfig:
    """Configuration for random masking."""

    min_mask_ratio: float = 0.10
    max_mask_ratio: float = 0.50


class Cas12aJEPA(nn.Module):
    """
    Context/target encoder pair with EMA target updates.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor_dim: int = 256,
        ema_decay: float = 0.996,
        masking_config: MaskingConfig = MaskingConfig(),
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.ema_decay = ema_decay
        self.masking_config = masking_config

        embed_dim = getattr(self.context_encoder, "embed_dim", 384)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, embed_dim),
        )

        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _mask_tokens(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_ratio = max(self.masking_config.min_mask_ratio, min(mask_ratio, self.masking_config.max_mask_ratio))
        eligible = tokens != pad_token_id
        random_vals = torch.rand(tokens.shape, device=tokens.device)
        mask = (random_vals < mask_ratio) & eligible

        masked_tokens = tokens.clone()
        masked_tokens[mask] = pad_token_id
        return masked_tokens, mask

    def update_ema(self) -> None:
        with torch.no_grad():
            for target_param, context_param in zip(
                self.target_encoder.parameters(),
                self.context_encoder.parameters(),
            ):
                target_param.data.mul_(self.ema_decay).add_(context_param.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mask_ratio: float = 0.30,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        masked_tokens, token_mask = self._mask_tokens(tokens, mask_ratio=mask_ratio, pad_token_id=pad_token_id)

        pred_context, _ = self.context_encoder(masked_tokens, attention_mask=attention_mask)
        pred_emb = self.predictor(pred_context)

        with torch.no_grad():
            target_emb, _ = self.target_encoder(tokens, attention_mask=attention_mask)

        info = {
            "mask": token_mask,
        }
        return pred_emb, target_emb, info
