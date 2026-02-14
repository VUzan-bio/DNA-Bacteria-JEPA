#!/usr/bin/env python3
"""
11_jepa_enhanced_scoring.py — Enhanced JEPA crRNA scoring & visualization
═══════════════════════════════════════════════════════════════════════════

Replaces the L2-norm proxy with three real JEPA-powered enhancements:

  1. COSINE SIMILARITY SCORING
     - Embed reverse complement of each candidate
     - RC consistency = cosine(fwd, rc_embed) → uses JEPA's RC invariance loss
     - Locus centroid proximity = cosine(embed, locus_centroid)
     - Combined: score = 0.5 * rc_consistency + 0.5 * centroid_proximity

  2. ATTENTION MAP VISUALIZATION
     - Hook into MultiheadAttention layers (6 layers × 6 heads)
     - Extract per-position attention for each crRNA
     - Highlight PAM, seed, SNP, SM positions
     - Shows what the encoder "looks at" → biologically interpretable

  3. EMBEDDING DIVERSITY FOR MULTIPLEX PANEL
     - Pairwise cosine distance matrix between selected crRNAs
     - Diversity score = min pairwise distance in final panel
     - Ensures 5-plex panel is maximally distinct in embedding space

Usage:
    python3 scripts/11_jepa_enhanced_scoring.py \
        --genome data/H37Rv.fna.gz \
        --checkpoint checkpoints/checkpoint_epoch200.pt \
        --output-dir results/mdrtb_enhanced

Author: Valentin Vézina | DNA-Bacteria-JEPA / ETH Zürich PhD Application
"""

import sys
import os
import argparse
import json
import logging
import gzip
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES (from v5)
# ═══════════════════════════════════════════════════════════════════════════
import re

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def get_complement(base: str) -> str:
    return {"A":"T","T":"A","G":"C","C":"G"}.get(base.upper(), base)

def load_genome(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    seq_parts = []
    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                seq_parts.append(line.strip().upper())
    return "".join(seq_parts)


@dataclass
class ResistanceSNP:
    gene: str
    mutation_ecoli: str
    genome_pos: int
    ref_allele: str
    alt_allele: str
    drug: str
    clinical_frequency: str = ""
    description: str = ""

MDR_TB_TARGETS = {
    "rpoB_RRDR": {
        "description": "Rifampicin Resistance Determining Region",
        "center": 761155,
        "snps": [
            ResistanceSNP("rpoB", "S531L", 761155, "C", "T", "Rifampicin", "~70%"),
            ResistanceSNP("rpoB", "H526Y", 761139, "C", "T", "Rifampicin", "~10%"),
            ResistanceSNP("rpoB", "H526D", 761139, "C", "G", "Rifampicin", "~7%"),
            ResistanceSNP("rpoB", "D516V", 761110, "A", "T", "Rifampicin", "~5%"),
            ResistanceSNP("rpoB", "L533P", 761161, "T", "C", "Rifampicin", "~2%"),
        ],
    },
    "katG_315": {
        "description": "katG codon 315",
        "center": 2155168,
        "snps": [
            ResistanceSNP("katG", "S315T", 2155168, "C", "G", "Isoniazid", "~90%"),
            ResistanceSNP("katG", "S315N", 2155168, "C", "T", "Isoniazid", "~2%"),
        ],
    },
    "inhA_promoter": {
        "description": "fabG1-inhA promoter",
        "center": 1673425,
        "snps": [
            ResistanceSNP("inhA", "c.-15C>T", 1673425, "C", "T", "INH/Ethionamide", "~20%"),
            ResistanceSNP("inhA", "c.-8T>C", 1673432, "T", "C", "INH/Ethionamide", "~5%"),
        ],
    },
}

PAM_PATTERNS = {
    "strict":  {"plus": r"TTT[ACG]", "minus": r"[CGT]AAA"},
    "relaxed": {"plus": r"TT[ACGT][ACGT]", "minus": r"[ACGT][ACGT]AA"},
}

@dataclass
class PAMSite:
    position: int
    strand: str
    pam_seq: str
    spacer_start: int
    spacer_end: int
    snp_position_in_spacer: int
    distance_to_snp: int
    pam_type: str

@dataclass
class CrRNADesign:
    name: str
    target_snp: ResistanceSNP
    pam_site: PAMSite
    spacer_wt: str
    spacer_mut: str
    spacer_designed: str
    spacer_length: int
    snp_pos_in_spacer: int
    synthetic_mm_pos: int
    synthetic_mm_type: str
    n_mm_vs_mutant: int
    n_mm_vs_wildtype: int
    gc_content: float
    pam_seq: str
    strand: str
    pam_type: str
    design_type: str
    score: float = 0.0
    jepa_activity: float = 0.0
    combined_score: float = 0.0
    context_seq: str = ""
    # Enhanced JEPA scores
    rc_consistency: float = 0.0
    centroid_proximity: float = 0.0
    jepa_enhanced_score: float = 0.0


def find_pam_sites(genome, snp, pam_mode, spacer_len=25, max_distance=150):
    snp_pos = snp.genome_pos
    search_start = max(0, snp_pos - 1 - max_distance - spacer_len - 4)
    search_end = min(len(genome), snp_pos - 1 + max_distance + spacer_len + 4)
    region = genome[search_start:search_end]
    results = []
    pat = PAM_PATTERNS[pam_mode]
    for m in re.finditer(pat["plus"], region, re.IGNORECASE):
        pam_gpos = search_start + m.start() + 1
        pam_seq = region[m.start():m.start() + 4]
        sp_start = pam_gpos + 4
        sp_end = sp_start + spacer_len - 1
        if sp_start <= snp_pos <= sp_end:
            snp_in_sp = snp_pos - sp_start + 1
            results.append(PAMSite(pam_gpos, "+", pam_seq, sp_start, sp_end,
                                   snp_in_sp, abs(snp_pos - pam_gpos), pam_mode))
    for m in re.finditer(pat["minus"], region, re.IGNORECASE):
        vaaa_gpos = search_start + m.start() + 1
        sp_end_plus = vaaa_gpos - 1
        sp_start_plus = sp_end_plus - spacer_len + 1
        if sp_start_plus <= snp_pos <= sp_end_plus:
            snp_in_sp = sp_end_plus - snp_pos + 1
            pam_seq_minus = reverse_complement(region[m.start():m.start() + 4])
            results.append(PAMSite(vaaa_gpos, "-", pam_seq_minus, sp_start_plus,
                                   sp_end_plus, snp_in_sp,
                                   abs(snp_pos - vaaa_gpos), pam_mode))
    return results


def design_candidates(genome, snp, pam, spacer_lengths=None):
    if spacer_lengths is None:
        spacer_lengths = [17, 18, 20, 23, 25]
    designs = []
    snp_pos_0 = snp.genome_pos - 1
    for sp_len in spacer_lengths:
        if pam.strand == "+":
            sp_start_0 = pam.position - 1 + 4
            if sp_start_0 + sp_len > len(genome): continue
            spacer_region = genome[sp_start_0:sp_start_0 + sp_len]
            snp_idx = snp_pos_0 - sp_start_0
            ctx_start = pam.position - 1
            ctx_end = min(sp_start_0 + sp_len + 5, len(genome))
            context_seq = genome[ctx_start:ctx_end]
        else:
            sp_end_0 = pam.position - 2
            sp_start_0 = sp_end_0 - sp_len + 1
            if sp_start_0 < 0: continue
            spacer_region = reverse_complement(genome[sp_start_0:sp_end_0 + 1])
            snp_idx = sp_end_0 - snp_pos_0
            ctx_start = max(sp_start_0 - 5, 0)
            ctx_end = pam.position - 1 + 4
            context_seq = reverse_complement(genome[ctx_start:min(ctx_end, len(genome))])
        if snp_idx < 0 or snp_idx >= sp_len: continue
        snp_pos_in_spacer = snp_idx + 1
        spacer_wt = list(spacer_region)
        spacer_mut = list(spacer_region)
        if pam.strand == "+":
            spacer_mut[snp_idx] = snp.alt_allele
        else:
            spacer_mut[snp_idx] = get_complement(snp.alt_allele)
        wt_str = "".join(spacer_wt)
        mut_str = "".join(spacer_mut)
        gc = sum(1 for b in mut_str if b in "GC") / len(mut_str)

        designs.append(CrRNADesign(
            name=f"{snp.gene}_{snp.mutation_ecoli}_sp{sp_len}_noSM",
            target_snp=snp, pam_site=pam,
            spacer_wt=wt_str, spacer_mut=mut_str, spacer_designed=mut_str,
            spacer_length=sp_len, snp_pos_in_spacer=snp_pos_in_spacer,
            synthetic_mm_pos=0, synthetic_mm_type="none",
            n_mm_vs_mutant=0, n_mm_vs_wildtype=1,
            gc_content=gc, pam_seq=pam.pam_seq, strand=pam.strand,
            pam_type=pam.pam_type, design_type="detect_only",
            context_seq=context_seq))

        for sm_offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            sm_idx = snp_idx + sm_offset
            if sm_idx < 0 or sm_idx >= sp_len: continue
            sm_pos = sm_idx + 1
            original_base = mut_str[sm_idx]
            for new_base in "ACGT":
                if new_base == original_base: continue
                designed = list(mut_str)
                designed[sm_idx] = new_base
                designed_str = "".join(designed)
                n_mm_mut = sum(1 for a, b in zip(designed_str, mut_str) if a != b)
                n_mm_wt = sum(1 for a, b in zip(designed_str, wt_str) if a != b)
                if n_mm_mut < n_mm_wt:
                    sm_type = f"{original_base}>{new_base} at pos {sm_pos}"
                    gc_d = sum(1 for b in designed_str if b in "GC") / len(designed_str)
                    designs.append(CrRNADesign(
                        name=f"{snp.gene}_{snp.mutation_ecoli}_sp{sp_len}_SM{sm_pos}",
                        target_snp=snp, pam_site=pam,
                        spacer_wt=wt_str, spacer_mut=mut_str,
                        spacer_designed=designed_str,
                        spacer_length=sp_len, snp_pos_in_spacer=snp_pos_in_spacer,
                        synthetic_mm_pos=sm_pos, synthetic_mm_type=sm_type,
                        n_mm_vs_mutant=n_mm_mut, n_mm_vs_wildtype=n_mm_wt,
                        gc_content=gc_d, pam_seq=pam.pam_seq, strand=pam.strand,
                        pam_type=pam.pam_type, design_type="SM",
                        context_seq=context_seq))
    return designs


def score_crrna(d: CrRNADesign) -> float:
    s = 0.0
    sp = d.snp_pos_in_spacer
    if 1 <= sp <= 8: s += 10.0
    elif 1 <= sp <= 12: s += 5.0
    else: s += 1.0
    s += (d.n_mm_vs_wildtype - d.n_mm_vs_mutant) * 3.0
    if d.synthetic_mm_pos > 0:
        dist = abs(d.synthetic_mm_pos - sp)
        s += max(0, 3.0 - dist * 0.5)
    if d.gc_content > 0.75 or d.gc_content < 0.3: s -= 2.0
    if d.design_type == "SM" and 1 <= d.synthetic_mm_pos <= 8: s += 2.0
    return s


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED JEPA EMBEDDER WITH ATTENTION CAPTURE
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedJEPAScorer:
    """
    Enhanced JEPA scoring using embedding direction, not magnitude.

    Three scoring modes:
      1. RC consistency: cosine(embed(seq), embed(RC(seq)))
         → exploits JEPA's reverse complement invariance training
      2. Centroid proximity: cosine(embed, locus_centroid)
         → sequences near locus centroid are well-represented
      3. Attention maps: per-position attention from all 6 layers × 6 heads
         → shows what the encoder "looks at" per nucleotide
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        import torch
        import torch.nn as nn

        self.available = False
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.embed_dim = 384

        try:
            from src.cas12a.encoder import SparseTransformerEncoder
            from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig
        except ImportError as e:
            log.warning("Cannot import JEPA modules: %s", e)
            return

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        embed_dim = args.get("embed_dim", 384)
        num_layers = args.get("num_layers", 6)
        num_heads = args.get("num_heads", 6)
        ff_dim = args.get("ff_dim", 1024)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.tokenizer = Cas12aTokenizer(TokenizerConfig())
        self.encoder = SparseTransformerEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=embed_dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim,
        )

        if "target_encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["target_encoder_state_dict"])
        elif "encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])

        self.encoder.to(self.device).eval()
        self.available = True

        epoch = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        log.info("Enhanced JEPA scorer loaded: %dD × %dL × %dH, epoch %s",
                 embed_dim, num_layers, num_heads, epoch)
        log.info("  Scoring: RC consistency + centroid proximity + attention maps")

    def _tokenize(self, sequences: List[str]):
        """Tokenize sequences, return (tokens, attention_mask) tensors."""
        import torch
        token_tensors = []
        for seq in sequences:
            tokens = self.tokenizer.encode_generic_sequence(seq.upper())
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze()
                if tokens.dim() == 0:
                    tokens = tokens.unsqueeze(0)
            token_tensors.append(tokens)

        max_len = max(len(t) for t in token_tensors)
        pad_id = self.tokenizer.pad_id
        padded = []
        for toks in token_tensors:
            pad_len = max_len - len(toks)
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_id, dtype=torch.long)
                toks = torch.cat([toks, padding])
            padded.append(toks)

        tokens_batch = torch.stack(padded).to(self.device)
        attention_mask = self.tokenizer.get_attention_mask(tokens_batch)
        return tokens_batch, attention_mask

    def embed(self, sequences: List[str]) -> np.ndarray:
        """Embed sequences → (N, D) numpy array."""
        import torch
        tokens, mask = self._tokenize(sequences)
        with torch.no_grad():
            pooled, _info = self.encoder(tokens, attention_mask=mask)
        return pooled.cpu().numpy()

    def embed_with_attention(self, sequences: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Embed sequences AND capture attention weights from all layers.

        Returns:
            embeddings: (N, D) numpy
            attention_weights: list of (N, num_heads, L, L) numpy per layer
        """
        import torch
        import torch.nn as nn

        tokens, mask = self._tokenize(sequences)

        # Register hooks to capture attention weights
        captured_attns = []
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                # MultiheadAttention output: (attn_output, attn_weights)
                # But we need to force need_weights=True
                pass
            return hook_fn

        # Monkey-patch MultiheadAttention to capture weights
        original_forwards = {}
        layer_attns = {}

        for i in range(self.num_layers):
            attn_module = self.encoder.encoder.layers[i].self_attn
            original_forwards[i] = attn_module.forward

            def patched_forward(self_attn, *args, _layer_idx=i, _orig=attn_module.forward, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False  # (B, H, L, L)
                out = _orig(*args, **kwargs)
                if isinstance(out, tuple) and len(out) >= 2:
                    layer_attns[_layer_idx] = out[1].detach().cpu().numpy()
                return out

            attn_module.forward = lambda *a, _pf=patched_forward, _am=attn_module, **kw: _pf(_am, *a, **kw)

        # Forward pass
        with torch.no_grad():
            pooled, _info = self.encoder(tokens, attention_mask=mask)

        # Restore original forwards
        for i in range(self.num_layers):
            self.encoder.encoder.layers[i].self_attn.forward = original_forwards[i]

        # Collect attention weights
        attn_list = [layer_attns.get(i, np.zeros((len(sequences), self.num_heads, 1, 1)))
                     for i in range(self.num_layers)]

        return pooled.cpu().numpy(), attn_list

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCEMENT 1: RC CONSISTENCY SCORING
    # ═══════════════════════════════════════════════════════════════════

    def compute_rc_consistency(self, designs: List[CrRNADesign]) -> np.ndarray:
        """
        Score each candidate by cosine similarity between its embedding
        and its reverse complement's embedding.

        Rationale: JEPA was trained with RC consistency loss.
        A well-encoded sequence should produce similar embeddings
        regardless of strand orientation. Higher RC consistency →
        the encoder has a robust representation of this genomic context.

        Returns: (N,) array of RC consistency scores ∈ [-1, 1].
        """
        fwd_seqs = [d.context_seq for d in designs]
        rc_seqs = [reverse_complement(s) for s in fwd_seqs]

        fwd_emb = self.embed(fwd_seqs)  # (N, D)
        rc_emb = self.embed(rc_seqs)    # (N, D)

        # Cosine similarity per sample
        dot = np.sum(fwd_emb * rc_emb, axis=1)
        norms_fwd = np.linalg.norm(fwd_emb, axis=1)
        norms_rc = np.linalg.norm(rc_emb, axis=1)
        cos_sim = dot / (norms_fwd * norms_rc + 1e-8)

        return cos_sim

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCEMENT 2: CENTROID PROXIMITY SCORING
    # ═══════════════════════════════════════════════════════════════════

    def compute_centroid_proximity(self, designs: List[CrRNADesign],
                                   embeddings: np.ndarray) -> np.ndarray:
        """
        Score each candidate by cosine similarity to its locus centroid.

        Rationale: candidates near the centroid of their genomic region
        are in well-represented embedding space. Outliers may be in
        poorly-learned regions (e.g., extreme GC, repetitive elements).

        Returns: (N,) array of centroid proximity scores ∈ [-1, 1].
        """
        # Compute centroids per locus
        loci = [d.target_snp.gene for d in designs]
        unique_loci = sorted(set(loci))
        centroids = {}
        for locus in unique_loci:
            mask = np.array([l == locus for l in loci])
            centroids[locus] = embeddings[mask].mean(axis=0)

        # Cosine similarity to own centroid
        scores = np.zeros(len(designs))
        for i, d in enumerate(designs):
            centroid = centroids[d.target_snp.gene]
            dot = np.dot(embeddings[i], centroid)
            norm_e = np.linalg.norm(embeddings[i])
            norm_c = np.linalg.norm(centroid)
            scores[i] = dot / (norm_e * norm_c + 1e-8)

        return scores

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED COMBINED SCORING
    # ═══════════════════════════════════════════════════════════════════

    def score_designs(self, designs: List[CrRNADesign]) -> List[CrRNADesign]:
        """
        Score all designs using enhanced JEPA metrics.

        Combined score = 0.5 * rc_consistency_norm + 0.5 * centroid_proximity_norm

        Then: final_score = jepa_enhanced × (1 + rule_score / max_rule)
        """
        if not self.available or not designs:
            return designs

        log.info("Computing enhanced JEPA scores for %d candidates...", len(designs))

        # Get embeddings
        sequences = [d.context_seq for d in designs]
        batch_size = 128
        all_emb = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            all_emb.append(self.embed(batch))
        embeddings = np.concatenate(all_emb, axis=0)

        # Enhancement 1: RC consistency
        log.info("  Computing RC consistency (cosine fwd vs RC)...")
        all_rc = []
        for i in range(0, len(designs), batch_size):
            batch = designs[i:i+batch_size]
            all_rc.append(self.compute_rc_consistency(batch))
        rc_scores = np.concatenate(all_rc, axis=0)

        # Enhancement 2: Centroid proximity
        log.info("  Computing centroid proximity...")
        centroid_scores = self.compute_centroid_proximity(designs, embeddings)

        # Normalise both to [0, 1]
        def minmax(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8) if mx - mn > 1e-8 else np.full_like(x, 0.5)

        rc_norm = minmax(rc_scores)
        cent_norm = minmax(centroid_scores)

        # Combined JEPA score
        jepa_enhanced = 0.5 * rc_norm + 0.5 * cent_norm

        # Rule-based scores
        for d in designs:
            d.score = score_crrna(d)
        max_rule = max(d.score for d in designs)
        max_rule = max(max_rule, 1.0)

        # Assign to designs
        for i, d in enumerate(designs):
            d.rc_consistency = float(rc_scores[i])
            d.centroid_proximity = float(centroid_scores[i])
            d.jepa_enhanced_score = float(jepa_enhanced[i])
            d.jepa_activity = float(jepa_enhanced[i])
            d.combined_score = float(jepa_enhanced[i]) * (1.0 + d.score / max_rule)

        # Stats
        log.info("  RC consistency:    min=%.3f, max=%.3f, mean=%.3f",
                 rc_scores.min(), rc_scores.max(), rc_scores.mean())
        log.info("  Centroid proximity: min=%.3f, max=%.3f, mean=%.3f",
                 centroid_scores.min(), centroid_scores.max(), centroid_scores.mean())
        log.info("  Enhanced JEPA:     min=%.3f, max=%.3f, mean=%.3f",
                 jepa_enhanced.min(), jepa_enhanced.max(), jepa_enhanced.mean())

        return designs, embeddings

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCEMENT 3: MULTIPLEX DIVERSITY
    # ═══════════════════════════════════════════════════════════════════

    def compute_panel_diversity(self, selected_designs: List[CrRNADesign],
                                embeddings_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Compute pairwise cosine distance matrix for the selected panel.
        Higher minimum pairwise distance → more diverse panel → lower
        cross-reactivity risk.

        Returns dict with distance matrix, diversity score, and names.
        """
        names = [f"{d.target_snp.gene}\n{d.target_snp.mutation_ecoli}" for d in selected_designs]
        embs = np.array([embeddings_dict[d.name] for d in selected_designs])

        # Cosine distance matrix
        n = len(embs)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cos_sim = np.dot(embs[i], embs[j]) / (
                    np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8)
                dist_matrix[i, j] = 1.0 - cos_sim

        # Diversity = minimum off-diagonal distance
        off_diag = dist_matrix[np.triu_indices(n, k=1)]
        diversity_score = off_diag.min() if len(off_diag) > 0 else 0.0
        mean_diversity = off_diag.mean() if len(off_diag) > 0 else 0.0

        return {
            "names": names,
            "distance_matrix": dist_matrix,
            "diversity_min": diversity_score,
            "diversity_mean": mean_diversity,
        }


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)

LOCUS_COLORS = {"rpoB": "#2171b5", "katG": "#d94801", "inhA": "#6a3d9a"}


def fig_rc_consistency(designs, out):
    """Fig A: RC consistency scores per locus — shows JEPA's learned RC invariance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Distribution by locus
    ax = axes[0]
    loci = sorted(set(d.target_snp.gene for d in designs))
    for i, locus in enumerate(loci):
        vals = [d.rc_consistency for d in designs if d.target_snp.gene == locus]
        jitter = np.random.normal(0, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=LOCUS_COLORS.get(locus, "#999"), s=15, alpha=0.5, edgecolors="none")
        ax.boxplot([vals], positions=[i], widths=0.4, showfliers=False,
                   boxprops=dict(color=LOCUS_COLORS.get(locus, "#999")),
                   medianprops=dict(color="red", linewidth=2))
    ax.set_xticks(range(len(loci)))
    ax.set_xticklabels(loci, fontsize=12, fontweight="bold")
    ax.set_ylabel("RC Consistency (cosine sim)", fontsize=11)
    ax.set_title("A. Reverse Complement Consistency by Locus\n"
                 "(JEPA trained with RC invariance loss)", fontsize=12, fontweight="bold")

    # Panel B: RC consistency vs rule score, colored by combined
    ax = axes[1]
    rule_scores = [d.score for d in designs if d.design_type == "SM"]
    rc_scores = [d.rc_consistency for d in designs if d.design_type == "SM"]
    combined = [d.combined_score for d in designs if d.design_type == "SM"]
    loci_sm = [d.target_snp.gene for d in designs if d.design_type == "SM"]

    for locus in sorted(set(loci_sm)):
        mask = [l == locus for l in loci_sm]
        rs = [rule_scores[i] for i in range(len(mask)) if mask[i]]
        rcs = [rc_scores[i] for i in range(len(mask)) if mask[i]]
        ax.scatter(rs, rcs, c=LOCUS_COLORS.get(locus, "#999"), s=20, alpha=0.5,
                   label=locus, edgecolors="white", linewidth=0.3)

    ax.set_xlabel("Rule-Based Score", fontsize=11)
    ax.set_ylabel("RC Consistency", fontsize=11)
    ax.set_title("B. RC Consistency vs Rule Score (SM designs)\n"
                 "(high in both → best candidates)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("DNA-JEPA Enhancement 1: Reverse Complement Consistency Scoring",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "enhanced_fig1_rc_consistency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → enhanced_fig1_rc_consistency.png")


def fig_centroid_proximity(designs, embeddings, out):
    """Fig B: Centroid proximity with t-SNE showing centroids."""
    from sklearn.manifold import TSNE

    N = len(designs)
    if N < 5: return

    perp = min(30, max(5, N // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                learning_rate="auto", init="pca")
    proj = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel A: t-SNE colored by centroid proximity
    ax = axes[0]
    cent_scores = [d.centroid_proximity for d in designs]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=cent_scores, cmap="viridis",
                    s=20, alpha=0.7, edgecolors="white", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="Centroid Proximity (cosine)", shrink=0.8)

    # Plot centroids
    loci = [d.target_snp.gene for d in designs]
    for locus in sorted(set(loci)):
        mask = [l == locus for l in loci]
        idx = [i for i, m in enumerate(mask) if m]
        cx, cy = proj[idx, 0].mean(), proj[idx, 1].mean()
        ax.scatter([cx], [cy], c=LOCUS_COLORS.get(locus), s=200, marker="*",
                   edgecolors="black", linewidth=1.5, zorder=10)
        ax.annotate(locus, (cx, cy), fontsize=10, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points",
                    color=LOCUS_COLORS.get(locus))

    ax.set_title("A. t-SNE colored by Centroid Proximity\n"
                 "(★ = locus centroids)", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    # Panel B: t-SNE colored by enhanced JEPA score
    ax = axes[1]
    enhanced = [d.jepa_enhanced_score for d in designs]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=enhanced, cmap="RdYlGn",
                    s=20, alpha=0.7, edgecolors="white", linewidth=0.3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Enhanced JEPA Score", shrink=0.8)

    ax.set_title("B. t-SNE colored by Enhanced JEPA Score\n"
                 "(0.5 × RC + 0.5 × centroid)", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    fig.suptitle("DNA-JEPA Enhancement 2: Locus Centroid Proximity in Embedding Space",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "enhanced_fig2_centroid_proximity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → enhanced_fig2_centroid_proximity.png")


def fig_attention_maps(scorer, designs, out):
    """Fig C: Attention heatmaps for the top crRNA per target."""
    # Select top SM design per SNP
    best_per_snp = {}
    for d in designs:
        if d.design_type != "SM": continue
        key = d.target_snp.mutation_ecoli
        if key not in best_per_snp or d.combined_score > best_per_snp[key].combined_score:
            best_per_snp[key] = d

    if not best_per_snp:
        log.warning("No SM designs for attention visualization")
        return

    selected = list(best_per_snp.values())[:5]  # top 5
    sequences = [d.context_seq for d in selected]

    log.info("  Extracting attention maps for %d selected crRNAs...", len(selected))
    embeddings, attn_layers = scorer.embed_with_attention(sequences)

    n_designs = len(selected)
    n_layers = len(attn_layers)

    # Average attention across heads for each layer, focus on CLS token (idx 0)
    # or average across all query positions
    fig, axes = plt.subplots(n_designs, 1, figsize=(16, 4 * n_designs))
    if n_designs == 1:
        axes = [axes]

    for idx, (d, ax) in enumerate(zip(selected, axes)):
        seq = d.context_seq
        seq_len = len(seq)

        # Average attention over all layers and heads, query=all positions, key=all positions
        # Shape per layer: (B, H, L, L) → average over H → (B, L, L) → average over query → (B, L)
        attn_per_pos = np.zeros(attn_layers[0].shape[-1])  # L
        for layer_attn in attn_layers:
            # layer_attn: (B, H, L, L)
            a = layer_attn[idx]  # (H, L, L)
            a_mean = a.mean(axis=0)  # (L, L) — average over heads
            a_col = a_mean.mean(axis=0)  # (L,) — average attention received per position
            attn_per_pos += a_col
        attn_per_pos /= n_layers

        # Trim to actual sequence length (tokens include CLS, SEP, padding)
        # Token layout: [CLS, A, C, G, T, ..., SEP, PAD, PAD, ...]
        actual_len = min(seq_len + 2, len(attn_per_pos))  # +2 for CLS, SEP
        attn_trimmed = attn_per_pos[1:seq_len + 1]  # skip CLS, take seq_len tokens

        if len(attn_trimmed) < seq_len:
            attn_trimmed = np.pad(attn_trimmed, (0, seq_len - len(attn_trimmed)))

        # Normalize
        attn_trimmed = attn_trimmed[:seq_len]
        if attn_trimmed.max() > 0:
            attn_norm = attn_trimmed / attn_trimmed.max()
        else:
            attn_norm = attn_trimmed

        # Plot as bar chart with nucleotide labels
        colors = []
        for pos in range(seq_len):
            nuc_pos = pos + 1  # 1-indexed position in context
            if pos < 4:
                colors.append("#e6550d")  # PAM (first 4 nt)
            elif d.synthetic_mm_pos > 0 and (pos - 4 + 1) == d.synthetic_mm_pos:
                colors.append("#ff7f00")  # SM position
            elif (pos - 4 + 1) == d.snp_pos_in_spacer:
                colors.append("#e41a1c")  # SNP position
            elif 1 <= (pos - 4 + 1) <= 8:
                colors.append("#a1d99b")  # Seed region
            else:
                colors.append("#6baed6")  # Rest of spacer

        bars = ax.bar(range(seq_len), attn_norm[:seq_len], color=colors, alpha=0.8,
                       edgecolor="white", linewidth=0.5)

        # Add nucleotide labels
        for pos in range(seq_len):
            if pos < len(seq):
                ax.text(pos, -0.08, seq[pos], ha="center", va="top",
                       fontsize=6, fontfamily="monospace", fontweight="bold")

        gene = d.target_snp.gene
        snp_name = d.target_snp.mutation_ecoli
        ax.set_ylabel("Attention\n(avg all layers)", fontsize=9)
        ax.set_title(f"{gene} {snp_name} — PAM={d.pam_seq} | SM@{d.synthetic_mm_pos} | "
                     f"SNP@{d.snp_pos_in_spacer} | RC={d.rc_consistency:.3f} | "
                     f"JEPA={d.jepa_enhanced_score:.3f}",
                     fontsize=10, fontweight="bold", color=LOCUS_COLORS.get(gene, "#333"))
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(-0.15, 1.1)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#e6550d", lw=8, label="PAM"),
        Line2D([0], [0], color="#a1d99b", lw=8, label="Seed (1-8)"),
        Line2D([0], [0], color="#e41a1c", lw=8, label="SNP"),
        Line2D([0], [0], color="#ff7f00", lw=8, label="Synth. MM"),
        Line2D([0], [0], color="#6baed6", lw=8, label="Spacer"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right", ncol=5, fontsize=8)

    fig.suptitle("DNA-JEPA Attention Maps — Per-Position Attention Weight\n"
                 "(averaged across 6 layers × 6 heads, 384D encoder)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "enhanced_fig3_attention_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → enhanced_fig3_attention_maps.png")


def fig_panel_diversity(diversity_info, out):
    """Fig D: Cosine distance heatmap for the selected multiplex panel."""
    names = diversity_info["names"]
    dist = diversity_info["distance_matrix"]
    n = len(names)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(dist, cmap="YlOrRd", vmin=0, vmax=dist.max() * 1.1)
    plt.colorbar(im, ax=ax, label="Cosine Distance (1 - cosine sim)", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if dist[i, j] > dist.max() * 0.6 else "black"
            ax.text(j, i, f"{dist[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    div_min = diversity_info["diversity_min"]
    div_mean = diversity_info["diversity_mean"]

    ax.set_title(f"Multiplex Panel Embedding Diversity\n"
                 f"Min pairwise distance: {div_min:.4f} | "
                 f"Mean: {div_mean:.4f}",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "enhanced_fig4_panel_diversity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → enhanced_fig4_panel_diversity.png")


def fig_enhanced_comparison(designs, out):
    """Fig E: Old (L2 norm) vs New (RC + centroid) scoring comparison."""
    sm_designs = [d for d in designs if d.design_type == "SM"]
    if len(sm_designs) < 3: return

    # Compute L2 norm-based score for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: RC consistency vs centroid proximity
    ax = axes[0]
    rc = [d.rc_consistency for d in sm_designs]
    cent = [d.centroid_proximity for d in sm_designs]
    loci = [d.target_snp.gene for d in sm_designs]
    for locus in sorted(set(loci)):
        mask = [l == locus for l in loci]
        ax.scatter([rc[i] for i in range(len(mask)) if mask[i]],
                   [cent[i] for i in range(len(mask)) if mask[i]],
                   c=LOCUS_COLORS.get(locus), s=20, alpha=0.5, label=locus,
                   edgecolors="white", linewidth=0.3)
    ax.set_xlabel("RC Consistency", fontsize=11)
    ax.set_ylabel("Centroid Proximity", fontsize=11)
    ax.set_title("A. Two JEPA Scoring Axes\n(both use direction, not magnitude)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Enhanced score vs rule score
    ax = axes[1]
    enhanced = [d.jepa_enhanced_score for d in sm_designs]
    rule = [d.score for d in sm_designs]
    for locus in sorted(set(loci)):
        mask = [l == locus for l in loci]
        ax.scatter([rule[i] for i in range(len(mask)) if mask[i]],
                   [enhanced[i] for i in range(len(mask)) if mask[i]],
                   c=LOCUS_COLORS.get(locus), s=20, alpha=0.5, label=locus,
                   edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Rule-Based Score", fontsize=11)
    ax.set_ylabel("Enhanced JEPA Score", fontsize=11)
    ax.set_title("B. Enhanced JEPA vs Rule-Based\n(JEPA uses RC + centroid, not L2 norm)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Distribution of enhanced scores
    ax = axes[2]
    for locus in sorted(set(loci)):
        vals = [d.jepa_enhanced_score for d in sm_designs if d.target_snp.gene == locus]
        ax.hist(vals, bins=20, alpha=0.5, color=LOCUS_COLORS.get(locus), label=locus,
                edgecolor="white")
    ax.set_xlabel("Enhanced JEPA Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("C. Score Distribution by Locus", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Enhanced JEPA Scoring: Direction-Based vs Magnitude-Based",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "enhanced_fig5_scoring_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → enhanced_fig5_scoring_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced JEPA Scoring & Visualization")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/mdrtb_enhanced")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load genome
    log.info("Loading genome: %s", args.genome)
    genome = load_genome(args.genome)
    log.info("Loaded: %s bp, GC=%.1f%%",
             f"{len(genome):,}", 100 * sum(1 for b in genome if b in "GC") / len(genome))

    # Generate candidates
    log.info("Generating crRNA candidates...")
    all_designs = []
    for group_name, group in MDR_TB_TARGETS.items():
        for snp in group["snps"]:
            for pam_mode in ["strict", "relaxed"]:
                pams = find_pam_sites(genome, snp, pam_mode)
                for pam in pams:
                    candidates = design_candidates(genome, snp, pam)
                    all_designs.extend(candidates)
    log.info("Total candidates: %d", len(all_designs))

    # Load enhanced JEPA scorer
    scorer = EnhancedJEPAScorer(args.checkpoint)
    if not scorer.available:
        log.error("JEPA not available — cannot run enhanced scoring")
        return

    # Score with enhanced method
    scored_designs, embeddings = scorer.score_designs(all_designs)

    # Select best SM design per SNP
    best_per_snp = {}
    for d in scored_designs:
        if d.design_type != "SM": continue
        key = d.target_snp.mutation_ecoli
        if key not in best_per_snp or d.combined_score > best_per_snp[key].combined_score:
            best_per_snp[key] = d

    selected_panel = list(best_per_snp.values())

    log.info("\n" + "=" * 60)
    log.info("  ENHANCED JEPA SCORING RESULTS")
    log.info("=" * 60)
    for d in sorted(selected_panel, key=lambda x: x.combined_score, reverse=True):
        log.info("  %s %s: RC=%.3f | Centroid=%.3f | JEPA=%.3f | Rule=%.1f | Combined=%.3f",
                 d.target_snp.gene, d.target_snp.mutation_ecoli,
                 d.rc_consistency, d.centroid_proximity,
                 d.jepa_enhanced_score, d.score, d.combined_score)

    # Build embedding lookup for diversity computation
    emb_dict = {}
    for d, emb in zip(scored_designs, embeddings):
        emb_dict[d.name] = emb

    # Enhancement 3: Panel diversity
    if len(selected_panel) >= 2:
        diversity = scorer.compute_panel_diversity(selected_panel, emb_dict)
        log.info("\n  Panel diversity: min=%.4f, mean=%.4f",
                 diversity["diversity_min"], diversity["diversity_mean"])

    # ── Generate figures ──
    log.info("\nGenerating enhanced figures...")

    fig_rc_consistency(scored_designs, out)
    fig_centroid_proximity(scored_designs, embeddings, out)
    fig_attention_maps(scorer, scored_designs, out)
    if len(selected_panel) >= 2:
        fig_panel_diversity(diversity, out)
    fig_enhanced_comparison(scored_designs, out)

    # ── Save summary ──
    summary = {
        "n_candidates": len(scored_designs),
        "n_selected": len(selected_panel),
        "scoring_method": "0.5 * RC_consistency + 0.5 * centroid_proximity",
        "selected_panel": [
            {
                "gene": d.target_snp.gene,
                "snp": d.target_snp.mutation_ecoli,
                "spacer": d.spacer_designed,
                "pam": d.pam_seq,
                "rc_consistency": round(d.rc_consistency, 4),
                "centroid_proximity": round(d.centroid_proximity, 4),
                "enhanced_jepa": round(d.jepa_enhanced_score, 4),
                "rule_score": round(d.score, 1),
                "combined_score": round(d.combined_score, 4),
            }
            for d in selected_panel
        ],
    }
    if len(selected_panel) >= 2:
        summary["panel_diversity_min"] = round(diversity["diversity_min"], 4)
        summary["panel_diversity_mean"] = round(diversity["diversity_mean"], 4)

    with open(out / "enhanced_scoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  → enhanced_scoring_summary.json")

    log.info("\n" + "=" * 60)
    log.info("  ENHANCED FIGURES COMPLETE")
    log.info("=" * 60)
    for f in sorted(out.glob("enhanced_*.png")):
        log.info("  %s (%.0f KB)", f.name, f.stat().st_size / 1024)


if __name__ == "__main__":
    main()
