"""
Task-specific heads for multi-task learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class TaskHeadConfig:
    """Configuration for task heads."""

    embed_dim: int = 384
    hidden_dim: int = 128
    dropout: float = 0.2


class RPAPrimerHead(nn.Module):
    """
    Binary classifier for RPA primer pairs.
    Predicts: will this primer pair amplify successfully?
    """

    def __init__(self, config: TaskHeadConfig = TaskHeadConfig()):
        super().__init__()

        self.forward_proj = nn.Linear(config.embed_dim, config.hidden_dim)
        self.reverse_proj = nn.Linear(config.embed_dim, config.hidden_dim)
        self.target_proj = nn.Linear(config.embed_dim, config.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fwd_tokens: torch.Tensor,
        rev_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            fwd_tokens: (batch, seq_len) forward primer tokens.
            rev_tokens: (batch, seq_len) reverse primer tokens.
            tgt_tokens: (batch, seq_len) target sequence tokens.
            encoder: pretrained encoder.

        Returns:
            predictions: (batch, 1) probability of successful amplification.
        """
        fwd_emb, _ = encoder(fwd_tokens)
        rev_emb, _ = encoder(rev_tokens)
        tgt_emb, _ = encoder(tgt_tokens)

        combined = torch.cat(
            [
                self.forward_proj(fwd_emb),
                self.reverse_proj(rev_emb),
                self.target_proj(tgt_emb),
            ],
            dim=1,
        )

        return self.classifier(combined)


class Cas12aEfficiencyHead(nn.Module):
    """
    Regression head for Cas12a trans-cleavage efficiency.
    Predicts: trans-cleavage rate (continuous value).
    """

    def __init__(self, config: TaskHeadConfig = TaskHeadConfig()):
        super().__init__()

        self.guide_proj = nn.Linear(config.embed_dim, config.hidden_dim)
        self.target_proj = nn.Linear(config.embed_dim, config.hidden_dim)

        # PAM-aware attention (placeholder; see note in forward).
        self.pam_attention = nn.MultiheadAttention(
            config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        encoder: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (batch, seq_len) full Cas12a sequence (guide + target + PAM).
            encoder: pretrained encoder.

        Returns:
            predictions: (batch, 1) predicted log(efficiency).
            embeddings: (batch, embed_dim) for sparse regularization.
        """
        # Get pooled embeddings and optional token-level outputs.
        embeddings, info = encoder(tokens, return_token_embeddings=True)
        _ = info["token_embeddings"]

        # Placeholder region handling: this currently uses pooled embeddings
        # instead of explicit guide/target/PAM masks.
        guide_emb = self.guide_proj(embeddings)
        target_emb = self.target_proj(embeddings)

        target_attended, _ = self.pam_attention(
            target_emb.unsqueeze(1),
            target_emb.unsqueeze(1),
            target_emb.unsqueeze(1),
        )

        combined = torch.cat([guide_emb, target_attended.squeeze(1)], dim=1)
        predictions = self.regressor(combined)

        return predictions, embeddings


class MultiplexOptimizerHead(nn.Module):
    """
    Set-level predictor for multiplex panel optimization.
    Predicts panel quality score and cross-reactivity matrix.
    """

    def __init__(self, config: TaskHeadConfig = TaskHeadConfig(), max_guides: int = 14):
        super().__init__()
        self.max_guides = max_guides

        self.cross_attn = nn.MultiheadAttention(
            config.embed_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        self.interference = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.panel_scorer = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        guide_tokens_list: list[torch.Tensor],
        encoder: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            guide_tokens_list: list of K guide token sequences.
            encoder: pretrained encoder.

        Returns:
            panel_quality: (batch, 1) overall panel score.
            interference_matrix: (batch, K, K) pairwise interference scores.
        """
        guide_embs = []
        for tokens in guide_tokens_list:
            emb, _ = encoder(tokens)
            guide_embs.append(emb)

        guide_embs = torch.stack(guide_embs, dim=1)  # (batch, K, embed_dim)

        attended, _ = self.cross_attn(guide_embs, guide_embs, guide_embs)

        k_guides = len(guide_tokens_list)
        batch_size = guide_embs.shape[0]
        interference_matrix = torch.zeros(
            batch_size,
            k_guides,
            k_guides,
            device=guide_embs.device,
        )

        for i in range(k_guides):
            for j in range(i + 1, k_guides):
                pair = torch.cat([guide_embs[:, i], guide_embs[:, j]], dim=1)
                score = self.interference(pair).squeeze(1)
                interference_matrix[:, i, j] = score
                interference_matrix[:, j, i] = score

        pooled = attended.mean(dim=1)
        panel_quality = self.panel_scorer(pooled)

        return panel_quality, interference_matrix


class Cas12aEfficiencyModel(nn.Module):
    """Complete model wrapper: encoder + task head."""

    def __init__(self, encoder: nn.Module, task_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.task_head = task_head

    def forward(self, *args, **kwargs):
        return self.task_head(*args, encoder=self.encoder, **kwargs)


def inject_lora_adapters(encoder: nn.Module, rank: int = 16, alpha: int = 32):
    """
    Inject LoRA adapters for parameter-efficient fine-tuning.

    Reference: Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models".

    Note: This is a simplified placeholder. Use `peft` for production.
    """
    print(f"Would inject LoRA with rank={rank}, alpha={alpha}")
    print("Use: from peft import get_peft_model, LoraConfig")
    return encoder
