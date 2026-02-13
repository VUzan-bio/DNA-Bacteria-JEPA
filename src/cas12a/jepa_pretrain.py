"""
Joint-Embedding Predictive Architecture (JEPA) for bacterial DNA sequences.

v3 — SOTA improvements over v2
-------------------------------
  - **Multi-block masking (I-JEPA style)** — predict M large target blocks
    from remaining context, forcing semantic-level representation learning
    (Assran et al., CVPR 2023)
  - **Transformer predictor with mask tokens** — narrow ViT predictor (384-dim
    bottleneck) with learnable mask tokens + positional embeddings, following
    I-JEPA best practices (predictor width ≤ encoder width)
  - **Curriculum masking** — progressive increase in mask ratio and block size
    from easy (small/nearby) to hard (large/distant), exploiting DNA's strong
    local correlations early then forcing long-range learning (A-JEPA, 2024)
  - **LDReg (Local Dimensionality Regularization)** — maximizes local intrinsic
    dimensionality via k-NN distances, catching a collapse mode VICReg misses
    where representations span high-dimensional space globally but collapse
    locally (Huang et al., ICLR 2024)

Retained from v2
-----------------
  - **VICReg regularisation** with cov_weight=1.0 (C-JEPA, NeurIPS 2024)
  - **Reverse complement consistency loss** (Caduceus, ICML 2024)
  - **Supervised contrastive loss** (DNABERT-S, ISMB 2025)
  - **Adversarial GC debiasing** via gradient reversal (NOVEL)
  - **Cosine EMA schedule** for target encoder

Encoder contract
----------------
``SparseTransformerEncoder.forward(tokens, attention_mask, return_token_embeddings)``
returns ``(pooled, info_dict)`` where:
  - ``pooled``:  (B, D) mean-pooled sequence embeddings
  - ``info_dict["token_embeddings"]``:  (B, L, D) per-token embeddings

References
----------
- Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture." CVPR 2023.
- Bardes et al. "VICReg." ICLR 2022.
- Mo & Tong. "C-JEPA." NeurIPS 2024.
- Huang et al. "LDReg: Local Dimensionality Regularization." ICLR 2024.
- Garrido et al. "RankMe." ICML 2023.
- Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.
- Schiff et al. "Caduceus." ICML 2024.
- Ganin & Lempitsky. "Domain-Adversarial Training." JMLR 2016.

Author : Valentin Uzan
Project: DNA-Bacteria-JEPA
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MaskingConfig:
    """Multi-block masking hyper-parameters (I-JEPA style).

    Generates M non-overlapping *target* blocks.  The context is everything
    outside the target blocks.  The predictor must reconstruct target-block
    representations from context — forcing long-range semantic prediction.

    Curriculum masking linearly ramps mask_ratio and block scale from
    start→end values over training (A-JEPA, 2024).

    Parameters
    ----------
    mask_ratio_start / mask_ratio_end : float
        Curriculum: fraction of non-pad tokens in target blocks.
    num_target_blocks : int
        Number of target blocks per sequence (I-JEPA uses 4).
    min_block_len : int
        Minimum tokens per target block.
    context_ratio_floor : float
        Minimum fraction of tokens reserved for context (prevents
        degenerate cases where almost everything is masked).
    """
    # Curriculum endpoints
    mask_ratio_start: float = 0.15
    mask_ratio_end: float = 0.50
    # Block structure
    num_target_blocks: int = 4
    min_block_len_start: int = 3
    min_block_len_end: int = 10
    # Safety
    context_ratio_floor: float = 0.30
    # Legacy compat
    mask_ratio: float = 0.30
    min_mask_ratio: float = 0.10
    max_mask_ratio: float = 0.60
    num_blocks: int = 4
    min_block_len: int = 3


@dataclass
class VICRegConfig:
    """Coefficients for VICReg regularisation.

    cov_weight=1.0 is critical (C-JEPA, NeurIPS 2024).
    """
    sim_weight: float = 1.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    var_gamma: float = 1.0


@dataclass
class LDRegConfig:
    """Local Dimensionality Regularization (Huang et al., ICLR 2024).

    Maximizes local intrinsic dimensionality via k-NN distance ratios,
    catching collapse modes VICReg misses (representations globally spread
    but locally low-dimensional).

    Parameters
    ----------
    k : int
        Number of nearest neighbours for LID estimation.
    weight : float
        Loss weight (λ_ldreg).
    """
    k: int = 5
    weight: float = 0.1


@dataclass
class JEPAConfig:
    """Top-level JEPA configuration (v3)."""
    predictor_dim: int = 384       # ← I-JEPA: always 384 (bottleneck)
    predictor_depth: int = 4       # ← half encoder depth (survey guideline)
    predictor_num_heads: int = 6
    max_seq_len: int = 1024        # for predictor positional embeddings
    ema_decay_start: float = 0.996
    ema_decay_end: float = 1.0
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    ldreg: LDRegConfig = field(default_factory=LDRegConfig)


# ═══════════════════════════════════════════════════════════════════════════
# VICReg loss components
# ═══════════════════════════════════════════════════════════════════════════

def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Hinge loss pushing per-dimension std above *gamma*."""
    std = z.float().std(dim=0)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalise off-diagonal covariance entries."""
    z = z.float()
    N, D = z.shape
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(N - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


def compute_vicreg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: VICRegConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Full VICReg objective."""
    inv = F.mse_loss(pred, target)
    var = variance_loss(pred, gamma=cfg.var_gamma)
    cov = covariance_loss(pred)
    total = cfg.sim_weight * inv + cfg.var_weight * var + cfg.cov_weight * cov
    metrics = {
        "inv_loss": inv.item(),
        "var_loss": var.item(),
        "cov_loss": cov.item(),
        "vicreg_total": total.item(),
    }
    return total, metrics


# ═══════════════════════════════════════════════════════════════════════════
# LDReg — Local Dimensionality Regularization (Huang et al., ICLR 2024)
# ═══════════════════════════════════════════════════════════════════════════

def compute_ldreg_loss(
    embeddings: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Local intrinsic dimensionality regularization.

    Maximises the MLE estimate of local intrinsic dimensionality (LID)
    using k-nearest-neighbour distances.  Low LID means representations
    are locally collapsed onto low-dimensional manifolds even if globally
    they appear spread — a failure mode VICReg cannot detect.

    Parameters
    ----------
    embeddings : (N, D) — input representations (e.g. pooled or token-level)
    k : number of nearest neighbours

    Returns
    -------
    loss : scalar — *negative* mean LID (minimise to maximise LID)

    Reference: Huang et al., "Understanding and Improving the Role of
    Projection Head in Self-Supervised Learning", ICLR 2024.
    """
    N = embeddings.shape[0]
    if N <= k + 1:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    emb = embeddings.float()
    # Pairwise distances — (N, N)
    dists = torch.cdist(emb, emb)

    # k+1 smallest (includes self at 0)
    knn_dists, _ = dists.topk(k + 1, largest=False, dim=1)
    knn_dists = knn_dists[:, 1:]  # exclude self → (N, k)

    # Clamp for numerical stability
    knn_dists = knn_dists.clamp(min=1e-8)

    # MLE estimator for local intrinsic dimensionality
    # LID_i = -k / Σ_j log(d_ij / d_ik)   where d_ik is k-th neighbour
    rk = knn_dists[:, -1:]  # (N, 1) — k-th neighbour distance
    log_ratios = torch.log(knn_dists[:, :-1] / rk)  # (N, k-1)
    lid = -(k - 1) / log_ratios.sum(dim=1).clamp(max=-1e-6)  # (N,)

    # Maximise LID → minimise negative LID
    return -lid.mean()


# ═══════════════════════════════════════════════════════════════════════════
# RankMe — effective dimensionality metric
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rankme(embeddings: torch.Tensor) -> float:
    """RankMe score (entropy-based effective rank).

    Reference: Garrido et al., ICML 2023.
    """
    z = embeddings.float()
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + 1e-12)
    H = -(p * torch.log(p + 1e-12)).sum()
    return torch.exp(H).item()


# ═══════════════════════════════════════════════════════════════════════════
# GC-content utilities
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gc_correlation(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    pad_token_id: int = 0,
    gc_token_ids: Optional[Set[int]] = None,
) -> Tuple[float, float]:
    """Pearson |r| between GC-content and first embedding PC."""
    B = tokens.shape[0]
    if B < 4:
        return 0.0, 0.0

    non_pad = tokens != pad_token_id
    lengths = non_pad.sum(dim=1).float().clamp(min=1)

    if gc_token_ids is not None:
        gc_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for tid in gc_token_ids:
            gc_mask |= (tokens == tid)
        gc_frac = (gc_mask & non_pad).sum(dim=1).float() / lengths
    else:
        vocab_size = tokens.max().item() + 1
        upper = tokens >= (vocab_size // 2)
        gc_frac = (upper & non_pad).sum(dim=1).float() / lengths

    emb = embeddings.float()
    emb = emb - emb.mean(dim=0, keepdim=True)
    v = torch.randn(emb.shape[1], 1, device=emb.device)
    for _ in range(5):
        v = emb.T @ (emb @ v)
        v = v / (v.norm() + 1e-12)
    pc1 = (emb @ v).squeeze(-1)

    gc_c = gc_frac - gc_frac.mean()
    pc_c = pc1 - pc1.mean()
    denom = (gc_c.norm() * pc_c.norm()).clamp(min=1e-12)
    r = (gc_c @ pc_c) / denom
    return abs(r.item()), r.item()


def compute_gc_content(
    tokens: torch.Tensor,
    pad_token_id: int = 0,
    gc_token_ids: Optional[Set[int]] = None,
) -> torch.Tensor:
    """Compute GC fraction per sequence from token IDs → (B,)."""
    non_pad = tokens != pad_token_id
    lengths = non_pad.sum(dim=1).float().clamp(min=1)

    if gc_token_ids is not None:
        gc_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for tid in gc_token_ids:
            gc_mask |= (tokens == tid)
        gc_frac = (gc_mask & non_pad).sum(dim=1).float() / lengths
    else:
        vocab_size = tokens.max().item() + 1
        upper = tokens >= (vocab_size // 2)
        gc_frac = (upper & non_pad).sum(dim=1).float() / lengths
    return gc_frac


# ═══════════════════════════════════════════════════════════════════════════
# Reverse Complement
# ═══════════════════════════════════════════════════════════════════════════

def reverse_complement_tokens(
    tokens: torch.Tensor,
    complement_map: Dict[int, int],
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Compute reverse complement of tokenized DNA sequences."""
    rc = tokens.clone()
    for orig_id, comp_id in complement_map.items():
        rc[tokens == orig_id] = comp_id

    non_pad = tokens != pad_token_id
    lengths = non_pad.sum(dim=1)

    rc_result = torch.full_like(tokens, pad_token_id)
    for b in range(tokens.shape[0]):
        L = lengths[b].item()
        if L > 0:
            rc_result[b, :L] = rc[b, :L].flip(0)
    return rc_result


# ═══════════════════════════════════════════════════════════════════════════
# Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
# ═══════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised contrastive loss for genome-level clustering."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        features = F.normalize(features.float(), dim=1)
        sim = features @ features.T / self.temperature

        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        label_mask.fill_diagonal_(0)

        pos_count = label_mask.sum(dim=1)
        has_positives = pos_count > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        self_mask = 1.0 - torch.eye(B, device=features.device)
        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        loss_per_sample = -(label_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        return loss_per_sample[has_positives].mean()


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer + GC Adversary
# ═══════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GCAdversary(nn.Module):
    """Adversarial head predicting GC-content with gradient reversal."""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        reversed_emb = GradientReversalFunction.apply(embeddings, lambda_)
        return self.net(reversed_emb).squeeze(-1)

    @staticmethod
    def ganin_lambda(epoch: int, total_epochs: int) -> float:
        p = epoch / max(total_epochs - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Predictor with Mask Tokens (I-JEPA style)
# ═══════════════════════════════════════════════════════════════════════════

class _PredictorAttentionBlock(nn.Module):
    """Pre-norm transformer block: LN → MultiheadAttention → skip → LN → FFN → skip.

    Standard ViT block matching I-JEPA predictor design.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class JEPAPredictor(nn.Module):
    """I-JEPA-style transformer predictor with mask tokens.

    Architecture (following Assran et al., CVPR 2023)::

        context_emb (B, L, D_enc) → Proj(D_enc → D_pred) → replace target
        positions with learnable mask_token + positional_embed →
        [TransformerBlock × depth] → LN → Proj(D_pred → D_enc)

    Design principles from the survey:
    - **Width = bottleneck**: D_pred = 384 regardless of encoder size
    - **Depth ≈ half encoder**: 4 layers for 6-layer encoder
    - **Standard self-attention** over context + mask tokens (no cross-attn)
    - **Mask tokens**: shared learnable vector + per-position embeddings

    Parameters
    ----------
    embed_dim : int
        Encoder output dimension.
    predictor_dim : int
        Internal predictor dimension (384 = I-JEPA standard bottleneck).
    depth : int
        Number of transformer layers.
    num_heads : int
        Attention heads.
    max_seq_len : int
        Maximum sequence length (for positional embeddings).
    """

    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim

        # Input projection: encoder dim → predictor dim (bottleneck)
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
        )

        # Learnable mask token (I-JEPA: shared across all target positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embeddings for predictor (separate from encoder's)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _PredictorAttentionBlock(predictor_dim, num_heads, mlp_ratio=2.0)
            for _ in range(depth)
        ])

        # Output: LN → project back to encoder dim
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_token_emb: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_token_emb : (B, L, D_enc)
            Context encoder output.  Values at target positions are
            irrelevant (will be replaced by mask tokens).
        target_mask : (B, L) bool
            True at target positions (to be predicted).

        Returns
        -------
        predictions : (B, L, D_enc)
            Full-sequence predictions.  Only values at target_mask
            positions are meaningful.
        """
        B, L, _ = context_token_emb.shape

        # Project to predictor dimension
        x = self.input_proj(context_token_emb)  # (B, L, D_pred)

        # Replace target positions with mask tokens
        mask_tokens = self.mask_token.expand(B, L, -1)  # (B, L, D_pred)
        target_float = target_mask.unsqueeze(-1).float()  # (B, L, 1)
        x = x * (1.0 - target_float) + mask_tokens * target_float

        # Add positional embeddings (critical: tells predictor WHERE targets are)
        x = x + self.pos_embed[:, :L, :]

        # Transformer blocks — self-attention over context + mask tokens
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.output_proj(x)  # (B, L, D_enc)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy MLP Predictor (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

class _PredictorBlock(nn.Module):
    """Pre-norm residual MLP block (v1/v2 legacy)."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class JEPAPredictorLegacy(nn.Module):
    """v1/v2 MLP predictor (no mask tokens). Kept for checkpoint compat."""
    def __init__(self, embed_dim: int, predictor_dim: int, depth: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
        )
        self.blocks = nn.ModuleList(
            [_PredictorBlock(predictor_dim, predictor_dim * 2) for _ in range(depth)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Block Masking (I-JEPA style)
# ═══════════════════════════════════════════════════════════════════════════

def multi_block_mask_1d(
    seq_len: int,
    mask_ratio: float,
    num_target_blocks: int,
    min_block_len: int,
    eligible: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate I-JEPA-style multi-block target masks.

    Creates ``num_target_blocks`` non-overlapping contiguous spans as
    target regions.  Context = everything outside targets.

    Unlike random masking, contiguous blocks force the predictor to
    reconstruct entire genomic regions rather than interpolate between
    neighbours — the key I-JEPA insight (Assran et al., CVPR 2023).

    Parameters
    ----------
    seq_len : L
    mask_ratio : target fraction of eligible tokens to mask
    num_target_blocks : M target blocks
    min_block_len : minimum tokens per block
    eligible : (B, L) bool — True for non-pad tokens
    device : torch device

    Returns
    -------
    target_mask : (B, L) bool — True at TARGET positions (to predict)
    """
    B = eligible.shape[0]
    target_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    elig_lens = eligible.sum(dim=1)

    for b in range(B):
        L_elig = elig_lens[b].item()
        n_target = int(L_elig * mask_ratio)

        if L_elig < min_block_len * 2 or n_target < min_block_len:
            # Sequence too short → mask a single small block
            blen = max(1, min(min_block_len, int(L_elig) - 1))
            if blen > 0 and int(L_elig) > blen:
                start = torch.randint(0, int(L_elig) - blen + 1, (1,)).item()
                target_mask[b, start:start + blen] = True
            continue

        # Distribute tokens across blocks
        per_block_target = max(min_block_len, n_target // num_target_blocks)
        remaining = n_target
        occupied = torch.zeros(int(L_elig), dtype=torch.bool)

        for block_idx in range(num_target_blocks):
            if remaining < min_block_len:
                break

            blen = min(per_block_target, remaining)
            blen = max(min_block_len, blen)
            blen = min(blen, int(L_elig))

            # Find valid start positions (non-overlapping with existing blocks)
            # Try up to 50 random placements
            placed = False
            for _attempt in range(50):
                max_start = int(L_elig) - blen
                if max_start < 0:
                    break
                start = torch.randint(0, max_start + 1, (1,)).item()
                # Check overlap
                if not occupied[start:start + blen].any():
                    occupied[start:start + blen] = True
                    target_mask[b, start:start + blen] = True
                    remaining -= blen
                    placed = True
                    break

            if not placed:
                # Fallback: find first available gap
                free_positions = (~occupied).nonzero(as_tuple=True)[0]
                if len(free_positions) >= min_block_len:
                    start = free_positions[0].item()
                    actual_len = min(blen, len(free_positions), int(L_elig) - start)
                    actual_len = max(1, actual_len)
                    occupied[start:start + actual_len] = True
                    target_mask[b, start:start + actual_len] = True
                    remaining -= actual_len

    # Ensure we only mask eligible (non-pad) positions
    target_mask &= eligible
    return target_mask


# Legacy function kept for backward compatibility
def block_mask_1d(
    seq_len: int,
    mask_ratio: float,
    num_blocks: int,
    min_block_len: int,
    eligible: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Legacy block masking — redirects to multi_block_mask_1d."""
    return multi_block_mask_1d(
        seq_len, mask_ratio, num_blocks, min_block_len, eligible, device,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Curriculum Masking Schedule
# ═══════════════════════════════════════════════════════════════════════════

def curriculum_masking_params(
    epoch: int,
    total_epochs: int,
    cfg: MaskingConfig,
) -> Tuple[float, int]:
    """Compute curriculum-scheduled masking parameters.

    Linear ramp from easy (small ratio, small blocks) to hard
    (large ratio, large blocks) over training.

    Inspired by A-JEPA's curriculum masking for audio spectrograms
    and the survey's recommendation for DNA: "early training should
    exploit local correlations (codons), then force longer-range learning."

    Returns
    -------
    mask_ratio : float — current epoch's mask ratio
    min_block_len : int — current epoch's minimum block length
    """
    progress = epoch / max(total_epochs - 1, 1)
    # Cosine schedule (smoother than linear)
    t = 0.5 * (1.0 - math.cos(math.pi * progress))

    mask_ratio = cfg.mask_ratio_start + t * (cfg.mask_ratio_end - cfg.mask_ratio_start)
    min_block_len = int(
        cfg.min_block_len_start + t * (cfg.min_block_len_end - cfg.min_block_len_start)
    )
    min_block_len = max(1, min_block_len)

    return mask_ratio, min_block_len


# ═══════════════════════════════════════════════════════════════════════════
# Main JEPA Module (v3)
# ═══════════════════════════════════════════════════════════════════════════

class Cas12aJEPA(nn.Module):
    """Joint-Embedding Predictive Architecture for DNA sequences (v3).

    v3 workflow (I-JEPA style)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Generate M non-overlapping target blocks via multi-block masking.
    2. Context encoder sees input with target positions replaced by pad
       → per-token context embeddings (B, L, D).
    3. Transformer predictor receives context embeddings + mask tokens at
       target positions (with positional embeddings) → predicted embeddings.
    4. Target encoder (EMA, no grad) encodes full input → target embeddings.
    5. VICReg + LDReg loss on (predicted, target) at target positions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[JEPAConfig] = None,
        # Legacy kwargs
        predictor_dim: int = 384,
        ema_decay: float = 0.996,
        masking_config: Optional[MaskingConfig] = None,
    ):
        super().__init__()

        if config is not None:
            self.config = config
        else:
            mcfg = masking_config if masking_config is not None else MaskingConfig()
            self.config = JEPAConfig(
                predictor_dim=predictor_dim,
                ema_decay_start=ema_decay,
                masking=mcfg,
            )

        # ── Encoders ──
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # ── Predictor (v3: transformer with mask tokens) ──
        embed_dim = getattr(self.context_encoder, "embed_dim", 384)
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=self.config.predictor_dim,
            depth=self.config.predictor_depth,
            num_heads=self.config.predictor_num_heads,
            max_seq_len=self.config.max_seq_len,
        )

        # ── EMA state ──
        self._ema_decay = self.config.ema_decay_start

    # ------------------------------------------------------------------
    # EMA schedule
    # ------------------------------------------------------------------

    def set_ema_decay(self, progress: float) -> float:
        tau_0 = self.config.ema_decay_start
        tau_1 = self.config.ema_decay_end
        self._ema_decay = tau_1 - (tau_1 - tau_0) * (
            1 + math.cos(math.pi * progress)
        ) / 2
        return self._ema_decay

    @torch.no_grad()
    def update_ema(self) -> None:
        tau = self._ema_decay
        for tp, cp in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            tp.data.mul_(tau).add_(cp.data, alpha=1.0 - tau)

    # ------------------------------------------------------------------
    # Internal encoder call
    # ------------------------------------------------------------------

    def _encode_tokens(
        self,
        encoder: nn.Module,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _pooled, info = encoder(
            tokens,
            attention_mask=attention_mask,
            return_token_embeddings=True,
        )
        return info["token_embeddings"]  # (B, L, D)

    # ------------------------------------------------------------------
    # Forward (v3: I-JEPA multi-block)
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.30,
        min_block_len: int = 3,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        tokens : (B, L) long
        attention_mask : optional (B, L)
        mask_ratio : fraction of eligible tokens in target blocks
        min_block_len : minimum tokens per target block (curriculum)
        pad_token_id : padding token ID

        Returns
        -------
        pred_emb : (M, D) predicted embeddings at target positions
        target_emb : (M, D) target embeddings at target positions (detached)
        info : dict with diagnostics
        """
        B, L = tokens.shape
        device = tokens.device
        mcfg = self.config.masking

        effective_ratio = max(
            mcfg.min_mask_ratio, min(mask_ratio, mcfg.max_mask_ratio)
        )

        eligible = tokens != pad_token_id  # (B, L)

        # ── Multi-block target mask (I-JEPA style) ──
        target_mask = multi_block_mask_1d(
            seq_len=L,
            mask_ratio=effective_ratio,
            num_target_blocks=mcfg.num_target_blocks,
            min_block_len=min_block_len,
            eligible=eligible,
            device=device,
        )

        # Replace target positions with pad for context encoder
        masked_tokens = tokens.clone()
        masked_tokens[target_mask] = pad_token_id

        # ── Context encoder: masked input → (B, L, D) ──
        ctx_token_emb = self._encode_tokens(
            self.context_encoder, masked_tokens, attention_mask,
        )

        # ── Predictor: context + mask tokens → predictions (B, L, D) ──
        pred_all = self.predictor(ctx_token_emb, target_mask)

        # ── Target encoder: full input → (B, L, D) ──
        with torch.no_grad():
            tgt_token_emb = self._encode_tokens(
                self.target_encoder, tokens, attention_mask,
            )

        # ── Extract target positions only ──
        pred_emb = pred_all[target_mask]       # (M, D)
        target_emb = tgt_token_emb[target_mask]  # (M, D)

        # ── Context-pooled embedding for auxiliary losses ──
        visible = eligible & ~target_mask
        visible_float = visible.unsqueeze(-1).float()
        vis_count = visible_float.sum(dim=1).clamp(min=1)
        context_pooled = (ctx_token_emb * visible_float).sum(dim=1) / vis_count  # (B, D)

        # ── Target-pooled embedding for sequence-level VICReg ──
        # Pool target encoder over ALL eligible (non-pad) positions
        eligible_float = eligible.unsqueeze(-1).float()
        elig_count = eligible_float.sum(dim=1).clamp(min=1)
        target_pooled = (tgt_token_emb * eligible_float).sum(dim=1) / elig_count  # (B, D)

        info = {
            "mask": target_mask,
            "n_masked": target_mask.sum().item(),
            "n_eligible": eligible.sum().item(),
            "ema_decay": self._ema_decay,
            "context_pooled": context_pooled,
            "target_pooled": target_pooled.detach(),  # (B, D) — detached (EMA target)
            "effective_mask_ratio": effective_ratio,
            "min_block_len": min_block_len,
        }
        return pred_emb, target_emb, info

    # ------------------------------------------------------------------
    # Encode utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_target: bool = True,
    ) -> torch.Tensor:
        """(B, D) mean-pooled embeddings (no masking)."""
        encoder = self.target_encoder if use_target else self.context_encoder
        pooled, _info = encoder(tokens, attention_mask=attention_mask)
        return pooled