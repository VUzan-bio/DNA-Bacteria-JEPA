"""
Loss functions for JEPA pretraining and sparse-aware regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseLossConfig:
    """Configuration for sparse regularization losses."""

    sparsity_weight: float = 1e-4
    reg_l1_weight: float = 1e-4
    active_threshold: float = 1e-3


class JEPASparseLoss(nn.Module):
    """Wraps JEPA loss with a lightweight sparsity penalty."""

    def __init__(self, config: SparseLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        jepa_loss: torch.Tensor,
        pred_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_penalty = pred_emb.abs().mean()
        active_ratio = (pred_emb.abs() > self.config.active_threshold).float().mean()

        total_loss = jepa_loss + self.config.sparsity_weight * l1_penalty
        metrics = {
            "loss": float(total_loss.detach().item()),
            "jepa_loss": float(jepa_loss.detach().item()),
            "l1_penalty": float(l1_penalty.detach().item()),
            "active_ratio": float(active_ratio.detach().item()),
        }
        return total_loss, metrics


class SparseRegressionLoss(nn.Module):
    """Regression loss with embedding sparsity regularization."""

    def __init__(self, config: SparseLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        targets = targets.view_as(preds)
        mse = F.mse_loss(preds, targets)
        l1_penalty = embeddings.abs().mean()

        loss = mse + self.config.reg_l1_weight * l1_penalty
        metrics = {
            "loss": float(loss.detach().item()),
            "mse": float(mse.detach().item()),
            "l1_penalty": float(l1_penalty.detach().item()),
        }
        return loss, metrics
