#!/usr/bin/env python3
"""
09_mdrtb_assay_design.py v5 — CRISPR-Cas12a assay design for MDR-TB diagnostics
                              with DNA-JEPA guided crRNA ranking.

WHAT'S NEW in v5 (over v4):
  - JEPA INTEGRATION: loads actual DNA-Bacteria-JEPA checkpoint
    (SparseTransformerEncoder + Cas12aTokenizer from src.cas12a),
    embeds candidate 34nt context windows, scores via activity head
    (fine-tuned) or embedding L2 norm (pretrained only).
    Final ranking = JEPA_score × rule_based_discrimination_score.
  - Fallback: if no checkpoint provided, uses rule-based scoring only (v4 behaviour).
  - Bug fixes: n_relaxed undefined, generate_summary_table loop, export_designs
    missing iterator, score_crrna SM@pos1 penalty sign.
  - Cleaner CLI, proper logging, deterministic output.

FIXES for M. tuberculosis (65.6% GC → extremely sparse TTTV PAMs):
  v2-v3: Only 1/9 targets got designs (0.40 expected TTTV PAMs per target)
  v4-v5 solution:
    - Dual PAM mode: TTTV (standard LbCas12a) → auto-fallback to TTYN (enAsCas12a)
    - TTYN PAMs: P(TTY) = 0.010/pos → ~1.16 expected PAMs/target (3× improvement)
    - Spacer scan at 25nt (was 20nt) → 21% wider useful window
    - max_distance=150bp → broader search in GC-rich regions
    - Amplicons up to 260bp (validated for RPA at 37-42°C)
    - Mtb-aware scoring: tolerates 70-75% GC spacers
    - Extended seed: positions 1-12 scored (Cas12a seed extends beyond pos 8)

PAM density math (Mtb, 65.6% GC, both strands, spacer_len=25):
  TTTV: P=0.0042/pos → 0.49 expected PAMs/target → ~40% targets covered
  TTTN: P=0.0051/pos → 0.59 expected PAMs/target → ~45% targets covered
  TTYN: P=0.010/pos  → 1.16 expected PAMs/target → ~69% targets covered

Design strategy (Kohabir et al. 2024, Cell Reports Methods):
  1. Find TTTV/TTYN PAMs placing SNP within Cas12a spacer
  2. Design crRNA matching MUTANT allele
  3. Add synthetic mismatch (SM) 1-5 positions from SNP
  4. vs MUTANT: 1 MM (SM only) → Cas12a ACTIVE → signal
  5. vs WILDTYPE: 2 MM (SM + SNP) → Cas12a INACTIVE → no signal

JEPA integration (v5):
  Model: DNA-Bacteria-JEPA (src.cas12a.encoder.SparseTransformerEncoder)
  Checkpoint: checkpoints/checkpoint_epoch200.pt (HuggingFace: orgava/dna-bacteria-jepa)
    Keys: target_encoder_state_dict (EMA), encoder_state_dict, args, metrics
  Tokenizer: Cas12aTokenizer (character-level: A→1, C→2, G→3, T→4)
  Pipeline:
    1. Tokenize 34nt context window (PAM+spacer+flank) via Cas12aTokenizer
    2. Embed through frozen SparseTransformerEncoder → (B, 384)
    3. Score: activity_head(emb) if fine-tuned, else normalised L2 norm
    4. Rank: combined = JEPA_score × (1 + rule_score / max_rule_score)
  Mode A — pretrained only: embedding-based confidence scoring
  Mode B — fine-tuned (future): direct Cas12a activity prediction

Targets: rpoB RRDR (S531L, H526Y/D, D516V, L533P), katG S315T/N,
         inhA promoter c.-15C>T, c.-8T>C
Reference: M. tuberculosis H37Rv (NC_000962.3, 4,411,532 bp)

Usage:
  # Rule-based only (no checkpoint):
  python3 scripts/09_mdrtb_assay_design.py \\
      --genome data/H37Rv.fna --output-dir results/mdrtb_assay

  # With JEPA-guided ranking (pretrained encoder):
  python3 scripts/09_mdrtb_assay_design.py \\
      --genome data/H37Rv.fna --checkpoint checkpoints/checkpoint_epoch200.pt

  # Force enAsCas12a PAMs for all targets:
  python3 scripts/09_mdrtb_assay_design.py \\
      --genome data/H37Rv.fna --pam-mode relaxed

Author: Valentin Uzan | DNA-Bacteria-JEPA / ETH Zürich PhD Application
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Matplotlib (non-interactive)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mdrtb_v5")

# ---------------------------------------------------------------------------
# Project root (parent of scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# 1. VERIFIED SNP TARGET DEFINITIONS  (NC_000962.3)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResistanceSNP:
    gene: str
    mutation_ecoli: str       # E. coli numbering (standard in literature)
    mutation_mtb: str         # Mtb numbering
    genome_pos: int           # 1-indexed on + strand of NC_000962.3
    ref_allele: str           # Reference allele (+ strand)
    alt_allele: str           # Resistance allele (+ strand)
    codon_wt: str
    codon_mut: str
    drug: str
    frequency: str
    gene_strand: str
    gene_start: int
    gene_end: int


TARGETS: List[ResistanceSNP] = [
    # ── rpoB RRDR (Rifampicin) ─────────────────────────────────────
    ResistanceSNP("rpoB", "S531L", "S450L", 761155, "C", "T",
                  "TCG", "TTG", "Rifampicin", "~50% RIF-R",
                  "+", 759807, 763325),
    ResistanceSNP("rpoB", "H526Y", "H445Y", 761139, "C", "T",
                  "CAC", "TAC", "Rifampicin", "~14% RIF-R",
                  "+", 759807, 763325),
    ResistanceSNP("rpoB", "H526D", "H445D", 761139, "C", "G",
                  "CAC", "GAC", "Rifampicin", "~5% RIF-R",
                  "+", 759807, 763325),
    ResistanceSNP("rpoB", "D516V", "D435V", 761110, "A", "T",
                  "GAC", "GTC", "Rifampicin", "~8% RIF-R",
                  "+", 759807, 763325),
    ResistanceSNP("rpoB", "L533P", "L452P", 761161, "T", "C",
                  "CTG", "CCG", "Rifampicin", "~2% RIF-R",
                  "+", 759807, 763325),
    # ── katG (Isoniazid) ──────────────────────────────────────────
    ResistanceSNP("katG", "S315T", "S315T", 2155168, "C", "G",
                  "AGC", "ACC", "Isoniazid", "50-94% INH-R",
                  "-", 2153889, 2156111),
    ResistanceSNP("katG", "S315N", "S315N", 2155168, "C", "T",
                  "AGC", "AAC", "Isoniazid", "~3% INH-R",
                  "-", 2153889, 2156111),
    # ── inhA promoter (Isoniazid / Ethionamide) ───────────────────
    ResistanceSNP("inhA_promoter", "c.-15C>T", "c.-15C>T", 1673425, "C", "T",
                  "C", "T", "Isoniazid/Ethionamide", "29-35% INH-R",
                  "+", 1673440, 1674183),
    ResistanceSNP("inhA_promoter", "c.-8T>C", "c.-8T>C", 1673432, "T", "C",
                  "T", "C", "Isoniazid", "~4% INH-R",
                  "+", 1673440, 1674183),
]

AMPLICON_GROUPS: Dict[str, dict] = {
    "rpoB_RRDR": {
        "description": "Rifampicin Resistance Determining Region",
        "center": 761139,
        "targets": ["S531L", "H526Y", "H526D", "D516V", "L533P"],
    },
    "katG_315": {
        "description": "katG codon 315 (catalase-peroxidase)",
        "center": 2155168,
        "targets": ["S315T", "S315N"],
    },
    "inhA_promoter": {
        "description": "fabG1-inhA promoter region",
        "center": 1673428,
        "targets": ["c.-15C>T", "c.-8T>C"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. GENOME HANDLING
# ═══════════════════════════════════════════════════════════════════════════

COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(COMP)[::-1]


def get_complement(base: str) -> str:
    """Return complement of a single base."""
    return {"A": "T", "T": "A", "C": "G", "G": "C"}.get(base.upper(), "N")


def load_genome(fasta_path: Path) -> str:
    """Load first record from FASTA (plain or gzipped)."""
    import gzip
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    parts: List[str] = []
    with opener(fasta_path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if parts:
                    break          # only first record
                continue
            parts.append(line.strip())
    genome = "".join(parts).upper()
    gc = sum(1 for b in genome if b in "GC") / len(genome)
    log.info("Loaded genome: %s bp, GC=%.1f%%", f"{len(genome):,}", gc * 100)
    return genome


def verify_snp(genome: str, snp: ResistanceSNP) -> bool:
    """Verify that the reference allele matches at the expected position."""
    pos_0 = snp.genome_pos - 1
    if pos_0 >= len(genome):
        log.warning("%s %s pos %d out of range", snp.gene, snp.mutation_ecoli, snp.genome_pos)
        return False
    actual = genome[pos_0]
    if actual != snp.ref_allele:
        log.warning("%s %s at %d: expected %s, found %s",
                    snp.gene, snp.mutation_ecoli, snp.genome_pos,
                    snp.ref_allele, actual)
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 3. PAM SCANNER — DUAL MODE (TTTV / TTYN)
# ═══════════════════════════════════════════════════════════════════════════

PAM_PATTERNS = {
    "strict": {
        "plus":  r"TTT[ACG]",
        "minus": r"[CGT]AAA",
        "label": "TTTV (LbCas12a)",
    },
    "relaxed": {
        "plus":  r"TT[CT][ACGT]",
        "minus": r"[ACGT][AG]AA",
        "label": "TTYN (enAsCas12a)",
    },
}


@dataclass
class PAMSite:
    position: int                 # 1-indexed genome pos of first PAM base
    strand: str                   # "+" or "-"
    pam_seq: str                  # 4-nt PAM on non-target strand
    spacer_start: int             # 1-indexed on + strand
    spacer_end: int
    snp_position_in_spacer: int   # 1 = PAM-proximal
    distance_to_snp: int
    pam_type: str                 # "strict" or "relaxed"


def scan_pam_sites(
    genome: str,
    snp: ResistanceSNP,
    spacer_len: int = 25,
    max_distance: int = 150,
    pam_mode: str = "strict",
) -> List[PAMSite]:
    """
    Scan for PAM sites near a SNP on both strands.

    pam_mode:
      "strict"  → TTTV only (standard LbCas12a)
      "relaxed" → TTYN (enAsCas12a, ~2.4× more PAMs in Mtb)
    """
    snp_pos = snp.genome_pos
    search_start = max(0, snp_pos - 1 - max_distance - spacer_len - 4)
    search_end = min(len(genome), snp_pos - 1 + max_distance + spacer_len + 4)
    region = genome[search_start:search_end]

    results: List[PAMSite] = []
    pat = PAM_PATTERNS[pam_mode]

    # ── + strand: PAM → spacer on + strand ──
    for m in re.finditer(pat["plus"], region, re.IGNORECASE):
        pam_gpos = search_start + m.start() + 1
        pam_seq = region[m.start():m.start() + 4]
        sp_start = pam_gpos + 4
        sp_end = sp_start + spacer_len - 1
        if sp_start <= snp_pos <= sp_end:
            snp_in_sp = snp_pos - sp_start + 1
            results.append(PAMSite(
                position=pam_gpos, strand="+", pam_seq=pam_seq,
                spacer_start=sp_start, spacer_end=sp_end,
                snp_position_in_spacer=snp_in_sp,
                distance_to_snp=abs(snp_pos - pam_gpos),
                pam_type=pam_mode,
            ))

    # ── − strand: reverse complement PAM on + strand ──
    for m in re.finditer(pat["minus"], region, re.IGNORECASE):
        vaaa_gpos = search_start + m.start() + 1
        sp_end_plus = vaaa_gpos - 1
        sp_start_plus = sp_end_plus - spacer_len + 1
        if sp_start_plus <= snp_pos <= sp_end_plus:
            snp_in_sp = sp_end_plus - snp_pos + 1
            pam_seq_minus = reverse_complement(region[m.start():m.start() + 4])
            results.append(PAMSite(
                position=vaaa_gpos, strand="-", pam_seq=pam_seq_minus,
                spacer_start=sp_start_plus, spacer_end=sp_end_plus,
                snp_position_in_spacer=snp_in_sp,
                distance_to_snp=abs(snp_pos - vaaa_gpos),
                pam_type=pam_mode,
            ))

    # Sort: SNP in seed first, then by position in spacer
    results.sort(key=lambda x: (
        0 if 1 <= x.snp_position_in_spacer <= 8 else
        1 if 1 <= x.snp_position_in_spacer <= 12 else 2,
        x.snp_position_in_spacer,
    ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. crRNA DESIGNER WITH SYNTHETIC MISMATCH
# ═══════════════════════════════════════════════════════════════════════════

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
    synthetic_mm_pos: int       # 0 = no SM
    synthetic_mm_type: str
    n_mm_vs_mutant: int
    n_mm_vs_wildtype: int
    gc_content: float
    pam_seq: str
    strand: str
    pam_type: str               # "strict" or "relaxed"
    design_type: str            # "SM" or "detect_only"
    score: float = 0.0
    jepa_activity: float = 0.0  # v5: JEPA-predicted trans-cleavage
    combined_score: float = 0.0 # v5: jepa_activity × normalised rule score
    # v5: context window for JEPA embedding (PAM + spacer region)
    context_seq: str = ""


def design_crrna_candidates(
    genome: str,
    snp: ResistanceSNP,
    pam: PAMSite,
    spacer_lengths: Optional[List[int]] = None,
) -> List[CrRNADesign]:
    """
    Design crRNA candidates with synthetic mismatch strategy.

    Spacer lengths: [17, 18, 20, 23, 25] — broader range for GC-rich genomes.
    SM offsets: [-5..5] excluding 0.
    """
    if spacer_lengths is None:
        spacer_lengths = [17, 18, 20, 23, 25]

    designs: List[CrRNADesign] = []
    snp_pos_0 = snp.genome_pos - 1

    for sp_len in spacer_lengths:
        # ── Extract spacer region ──
        if pam.strand == "+":
            sp_start_0 = pam.position - 1 + 4
            if sp_start_0 + sp_len > len(genome):
                continue
            spacer_region = genome[sp_start_0:sp_start_0 + sp_len]
            snp_idx = snp_pos_0 - sp_start_0
            # v5: context window for JEPA (4nt PAM + spacer + 5nt downstream)
            ctx_start = pam.position - 1
            ctx_end = min(sp_start_0 + sp_len + 5, len(genome))
            context_seq = genome[ctx_start:ctx_end]
        else:
            sp_end_0 = pam.position - 2
            sp_start_0 = sp_end_0 - sp_len + 1
            if sp_start_0 < 0:
                continue
            spacer_region = reverse_complement(genome[sp_start_0:sp_end_0 + 1])
            snp_idx = sp_end_0 - snp_pos_0
            # v5: context on - strand
            ctx_start = max(sp_start_0 - 5, 0)
            ctx_end = pam.position - 1 + 4
            context_seq = reverse_complement(genome[ctx_start:min(ctx_end, len(genome))])

        if snp_idx < 0 or snp_idx >= sp_len:
            continue

        snp_pos_in_spacer = snp_idx + 1

        # ── Build WT / mutant spacers ──
        spacer_wt = list(spacer_region)
        spacer_mut = list(spacer_region)
        if pam.strand == "+":
            spacer_mut[snp_idx] = snp.alt_allele
        else:
            spacer_mut[snp_idx] = get_complement(snp.alt_allele)

        wt_str = "".join(spacer_wt)
        mut_str = "".join(spacer_mut)
        gc = sum(1 for b in mut_str if b in "GC") / len(mut_str)

        # === detect_only (no SM): matches mutant, 0 MM vs MUT, 1 MM vs WT ===
        designs.append(CrRNADesign(
            name=f"{snp.gene}_{snp.mutation_ecoli}_sp{sp_len}_noSM",
            target_snp=snp, pam_site=pam,
            spacer_wt=wt_str, spacer_mut=mut_str,
            spacer_designed=mut_str,
            spacer_length=sp_len,
            snp_pos_in_spacer=snp_pos_in_spacer,
            synthetic_mm_pos=0, synthetic_mm_type="none",
            n_mm_vs_mutant=0, n_mm_vs_wildtype=1,
            gc_content=gc,
            pam_seq=pam.pam_seq, strand=pam.strand,
            pam_type=pam.pam_type,
            design_type="detect_only",
            context_seq=context_seq,
        ))

        # === SM designs: try offsets [-5..5] excluding 0 ===
        for sm_offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            sm_idx = snp_idx + sm_offset
            if sm_idx < 0 or sm_idx >= sp_len:
                continue
            sm_pos = sm_idx + 1
            original_base = mut_str[sm_idx]

            for new_base in "ACGT":
                if new_base == original_base:
                    continue
                designed = list(mut_str)
                designed[sm_idx] = new_base
                designed_str = "".join(designed)

                n_mm_mut = sum(a != b for a, b in zip(designed_str, mut_str))
                n_mm_wt = sum(a != b for a, b in zip(designed_str, wt_str))

                # Core constraint: 1 MM vs mutant, 2 MM vs wildtype
                if n_mm_mut != 1 or n_mm_wt != 2:
                    continue

                gc_d = sum(1 for b in designed_str if b in "GC") / len(designed_str)
                designs.append(CrRNADesign(
                    name=f"{snp.gene}_{snp.mutation_ecoli}_sp{sp_len}_SM{sm_pos}{new_base}",
                    target_snp=snp, pam_site=pam,
                    spacer_wt=wt_str, spacer_mut=mut_str,
                    spacer_designed=designed_str,
                    spacer_length=sp_len,
                    snp_pos_in_spacer=snp_pos_in_spacer,
                    synthetic_mm_pos=sm_pos,
                    synthetic_mm_type=f"{original_base}>{new_base} at pos {sm_pos}",
                    n_mm_vs_mutant=n_mm_mut,
                    n_mm_vs_wildtype=n_mm_wt,
                    gc_content=gc_d,
                    pam_seq=pam.pam_seq, strand=pam.strand,
                    pam_type=pam.pam_type,
                    design_type="SM",
                    context_seq=context_seq,
                ))

    return designs


# ═══════════════════════════════════════════════════════════════════════════
# 4b. RULE-BASED SCORING (Mtb-aware)
# ═══════════════════════════════════════════════════════════════════════════

def score_crrna(design: CrRNADesign) -> float:
    """
    Rule-based crRNA score. Mtb-aware: tolerates high GC, extended seed.

    Scoring (higher = better):
      SM present:       +3.0 (essential for discrimination)
      SNP pos 1-6:      +3.0 | pos 7-8: +2.0 | pos 9-12: +1.0 | 13-15: +0.3
      SM in seed (1-8): +1.5
      SM close to SNP:  +1.0  (|SM−SNP| ≤ 2)
      Both in seed:     +1.0
      GC 35-65%:        +1.0 | 65-75%: +0.5 | >80%: −0.5
      Spacer ≤17nt:     +1.5 | ≤18nt: +0.5 | ≥25nt: −0.5
      No homopolymer≥4: +0.5 | ≥5: −1.0
      SM at pos 1:      −1.0 (kills trans-cleavage activity)
      TTTV PAM:         +0.5 (prefer canonical over relaxed)
    """
    s = 0.0
    sp = design.snp_pos_in_spacer
    sm = design.synthetic_mm_pos

    # SNP in seed / extended seed
    if 1 <= sp <= 6:
        s += 3.0
    elif 1 <= sp <= 8:
        s += 2.0
    elif 1 <= sp <= 12:
        s += 1.0
    elif 1 <= sp <= 15:
        s += 0.3

    # Synthetic mismatch bonuses/penalties
    if sm > 0:
        s += 3.0                        # SM present
        if 1 <= sm <= 8:
            s += 1.5                    # SM in seed
        if abs(sm - sp) <= 2:
            s += 1.0                    # SM close to SNP
        if sm == 1:
            s -= 1.0                    # pos 1 kills activity (FIXED from v4 doc1)
        if 1 <= sp <= 8 and 1 <= sm <= 8:
            s += 1.0                    # both SNP and SM in seed

    # GC — Mtb-aware (Mtb spacers often 65-75%)
    gc = design.gc_content
    if 0.35 <= gc <= 0.65:
        s += 1.0
    elif gc <= 0.75:
        s += 0.5
    elif gc > 0.80:
        s -= 0.5

    # Spacer length
    if design.spacer_length <= 17:
        s += 1.5
    elif design.spacer_length <= 18:
        s += 0.5
    elif design.spacer_length >= 25:
        s -= 0.5

    # Homopolymer run
    seq = design.spacer_designed
    max_run = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        max_run = max(max_run, run)
    if max_run < 4:
        s += 0.5
    elif max_run >= 5:
        s -= 1.0

    # Prefer canonical PAM
    if design.pam_type == "strict":
        s += 0.5

    return s


# ═══════════════════════════════════════════════════════════════════════════
# 4c. JEPA-GUIDED ACTIVITY PREDICTION  (v5 NEW)
# ═══════════════════════════════════════════════════════════════════════════
#
# Uses the actual DNA-Bacteria-JEPA pretrained encoder:
#   - SparseTransformerEncoder from src.cas12a.encoder
#   - Cas12aTokenizer from src.cas12a.tokenizer
#   - Checkpoint: target_encoder_state_dict (EMA, best for inference)
#
# Two modes:
#   a) Fine-tuned checkpoint (has "activity_head" key):
#      → direct Cas12a activity prediction ∈ [0, 1]
#   b) Pretrained-only checkpoint (epoch 200):
#      → embedding-based confidence scoring (L2 norm, normalised to [0, 1])
#      → captures sequence context quality from 20-genome pretraining
# ═══════════════════════════════════════════════════════════════════════════

CONTEXT_WINDOW = 34  # 4nt PAM + 25nt spacer + 5nt flanking


class JEPAActivityPredictor:
    """
    Wrapper around DNA-Bacteria-JEPA encoder for crRNA ranking.

    Loads the actual pretrained SparseTransformerEncoder from the
    DNA-Bacteria-JEPA project checkpoint (checkpoint_epoch200.pt).

    Checkpoint structure (from scripts/01_pretrain_jepa.py)::

        ckpt["target_encoder_state_dict"]  ← EMA encoder (best for inference)
        ckpt["encoder_state_dict"]         ← context encoder
        ckpt["predictor_state_dict"]       ← JEPA predictor (not needed here)
        ckpt["args"]                       ← {embed_dim, num_layers, num_heads, ff_dim, ...}
        ckpt["metrics"]                    ← {rankme, gc_abs_r, pred_std, ...}

    Optionally, a fine-tuned checkpoint may also contain:
        ckpt["activity_head"]              ← Cas12a regression head state dict
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.encoder = None
        self.tokenizer = None
        self.head = None           # only if fine-tuned checkpoint
        self.device = "cpu"
        self.available = False
        self.has_activity_head = False
        self.embed_dim = 384

        if checkpoint_path is None or not checkpoint_path.exists():
            log.info("JEPA checkpoint not provided — using rule-based scoring only")
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            log.warning("PyTorch not installed — JEPA disabled")
            return

        try:
            from src.cas12a.encoder import SparseTransformerEncoder
            from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig
        except ImportError as e:
            log.warning("Cannot import DNA-Bacteria-JEPA source (src.cas12a): %s", e)
            log.warning("  Ensure the project root is in sys.path or PYTHONPATH")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading DNA-Bacteria-JEPA checkpoint: %s (device=%s)",
                     checkpoint_path, self.device)

            ckpt = torch.load(checkpoint_path, map_location=self.device,
                              weights_only=False)

            # ── Extract model hyperparameters from checkpoint args ──
            args = ckpt.get("args", {})
            embed_dim = args.get("embed_dim", 384)
            num_layers = args.get("num_layers", 6)
            num_heads = args.get("num_heads", 6)
            ff_dim = args.get("ff_dim", 1024)
            self.embed_dim = embed_dim

            # ── Initialise tokenizer ──
            self.tokenizer = Cas12aTokenizer(TokenizerConfig())
            vocab_size = self.tokenizer.vocab_size

            # ── Reconstruct encoder architecture ──
            self.encoder = SparseTransformerEncoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
            )

            # ── Load weights (prefer EMA target encoder) ──
            if "target_encoder_state_dict" in ckpt:
                self.encoder.load_state_dict(ckpt["target_encoder_state_dict"])
                log.info("  Loaded target encoder (EMA): %dD × %dL × %dH",
                         embed_dim, num_layers, num_heads)
            elif "encoder_state_dict" in ckpt:
                self.encoder.load_state_dict(ckpt["encoder_state_dict"])
                log.info("  Loaded context encoder: %dD × %dL × %dH",
                         embed_dim, num_layers, num_heads)
            else:
                log.warning("  No encoder state dict found in checkpoint")
                return

            self.encoder.to(self.device).eval()

            # ── Report checkpoint metrics ──
            metrics = ckpt.get("metrics", {})
            epoch = ckpt.get("epoch", "?")
            rankme = metrics.get("rankme", "?")
            gc_r = metrics.get("gc_abs_r", "?")
            log.info("  Checkpoint epoch %s: RankMe=%s/%d, GC|r|=%s",
                     epoch, rankme, embed_dim, gc_r)

            # ── Optional: load fine-tuned activity head ──
            if "activity_head" in ckpt:
                self.head = nn.Sequential(
                    nn.Linear(embed_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1),
                    nn.Sigmoid(),
                )
                self.head.load_state_dict(ckpt["activity_head"])
                self.head.to(self.device).eval()
                self.has_activity_head = True
                log.info("  Activity head loaded (fine-tuned checkpoint)")
            else:
                log.info("  No activity head — using embedding-based scoring")
                log.info("  (Fine-tune with scripts/06_cas12a_finetune.py for "
                         "direct activity prediction)")

            self.available = True
            log.info("  DNA-Bacteria-JEPA encoder ready ✓")

        except Exception as exc:
            log.warning("Failed to load JEPA checkpoint: %s — rule-based fallback", exc)
            import traceback
            traceback.print_exc()
            self.encoder = None
            self.tokenizer = None
            self.available = False

    def _tokenize_sequences(self, sequences: List[str]) -> "torch.Tensor":
        """Tokenize DNA sequences using Cas12aTokenizer.

        Each sequence is tokenized character-by-character (A→1, C→2, G→3, T→4),
        padded to uniform length.

        Returns: (B, L) long tensor of token IDs.
        """
        import torch

        token_lists = []
        max_len = 0
        for seq in sequences:
            tokens = self.tokenizer.encode(seq.upper())
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
            token_lists.append(tokens)
            max_len = max(max_len, len(tokens))

        # Pad to uniform length
        pad_id = self.tokenizer.pad_id
        padded = []
        for toks in token_lists:
            padded.append(toks + [pad_id] * (max_len - len(toks)))

        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def embed_batch(self, designs: List[CrRNADesign]) -> "torch.Tensor":
        """
        Embed crRNA context sequences through the frozen JEPA encoder.

        Returns: (B, embed_dim) mean-pooled embeddings.
        """
        import torch

        sequences = [d.context_seq for d in designs]
        tokens = self._tokenize_sequences(sequences)  # (B, L)
        attention_mask = self.tokenizer.get_attention_mask(tokens)

        with torch.no_grad():
            pooled, _info = self.encoder(tokens, attention_mask=attention_mask)

        return pooled  # (B, D)

    def predict_batch(self, designs: List[CrRNADesign]) -> List[float]:
        """
        Predict activity scores for a batch of crRNA designs.

        Mode A — fine-tuned head available:
            activity = head(embedding) ∈ [0, 1]

        Mode B — pretrained encoder only:
            score = normalised embedding L2 norm ∈ [0, 1]
            (captures how well the encoder represents this genomic context;
             sequences in well-learned regions produce higher-norm embeddings)

        Returns: list of scores ∈ [0, 1].
        """
        if not self.available or not designs:
            return [0.0] * len(designs)

        import torch

        embeddings = self.embed_batch(designs)  # (B, D)

        with torch.no_grad():
            if self.has_activity_head:
                # Mode A: direct activity prediction
                activities = self.head(embeddings).squeeze(-1)  # (B,)
            else:
                # Mode B: embedding norm as confidence proxy
                norms = embeddings.norm(dim=1)  # (B,)
                # Normalise to [0, 1] via min-max within batch
                n_min, n_max = norms.min(), norms.max()
                if n_max - n_min > 1e-8:
                    activities = (norms - n_min) / (n_max - n_min)
                else:
                    activities = torch.ones_like(norms) * 0.5

        return activities.cpu().tolist()

    def rank_designs(self, designs: List[CrRNADesign]) -> List[CrRNADesign]:
        """
        Combine JEPA activity prediction with rule-based discrimination score.

        combined = jepa_activity × (1 + rule_score / max_rule_score)

        This ensures:
          - High JEPA activity + high rule score → top rank
          - Good activity but poor discrimination → moderate rank
          - Low activity → penalised regardless of rules
        """
        if not designs:
            return designs

        # Compute rule-based discrimination scores
        for d in designs:
            d.score = score_crrna(d)

        if not self.available:
            # No JEPA: combined_score = rule score only
            for d in designs:
                d.combined_score = d.score
            return sorted(designs, key=lambda x: x.combined_score, reverse=True)

        # JEPA-guided scoring
        activities = self.predict_batch(designs)
        max_rule = max(d.score for d in designs) if designs else 1.0
        max_rule = max(max_rule, 1.0)  # avoid division by zero

        for d, act in zip(designs, activities):
            d.jepa_activity = act
            d.combined_score = act * (1.0 + d.score / max_rule)

        return sorted(designs, key=lambda x: x.combined_score, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. RPA PRIMER DESIGNER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RPAPrimer:
    name: str
    sequence: str
    position: int
    length: int
    strand: str
    gc_content: float
    tm_rough: float
    max_homopolymer: int
    score: float


@dataclass
class RPAAmplicon:
    name: str
    forward: RPAPrimer
    reverse: RPAPrimer
    amplicon_length: int
    targets_covered: List[str]
    score: float


def compute_gc(seq: str) -> float:
    s = seq.upper()
    return sum(1 for b in s if b in "GC") / max(len(s), 1)


def compute_tm_basic(seq: str) -> float:
    """Wallace rule (<14 nt) or simple salt-adjusted Tm."""
    s = seq.upper()
    at = s.count("A") + s.count("T")
    gc = s.count("G") + s.count("C")
    if len(s) < 14:
        return 2 * at + 4 * gc
    return 64.9 + 41 * (gc - 16.4) / max(at + gc, 1)


def max_hp(seq: str) -> int:
    """Maximum homopolymer run length."""
    if not seq:
        return 0
    mx = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        mx = max(mx, run)
    return mx


def score_rpa_primer(seq: str) -> float:
    """Score an RPA primer candidate (higher = better)."""
    s = 0.0
    gc = compute_gc(seq)
    hp = max_hp(seq)
    # Length
    if 30 <= len(seq) <= 35:
        s += 2.0
    elif 28 <= len(seq) <= 37:
        s += 1.0
    # GC
    if 0.35 <= gc <= 0.65:
        s += 2.0
    elif 0.30 <= gc <= 0.70:
        s += 1.0
    else:
        s -= 0.5
    # Homopolymer
    if hp <= 3:
        s += 1.0
    elif hp >= 5:
        s -= 1.0
    # 3' end GC clamp
    if any(b in "GC" for b in seq[-2:].upper()):
        s += 0.5
    # Avoid 3' T (mispriming risk)
    if seq[-1].upper() != "T":
        s += 0.3
    return s


def design_rpa_primers(
    genome: str,
    center: int,
    target_names: List[str],
    amplicon_range: Tuple[int, int] = (120, 260),
    primer_len_range: Tuple[int, int] = (30, 35),
    n_candidates: int = 5,
) -> List[RPAAmplicon]:
    """Design RPA primer pairs around a genomic center position."""
    amplicons: List[RPAAmplicon] = []
    min_amp, max_amp = amplicon_range
    min_pl, max_pl = primer_len_range

    for amp_size in range(min_amp, max_amp + 1, 10):
        half = amp_size // 2
        for offset in range(-30, 31, 5):
            amp_start = center - half + offset
            amp_end = amp_start + amp_size - 1
            for pl in range(min_pl, max_pl + 1):
                # Forward primer
                fs0 = amp_start - 1
                if fs0 < 0 or fs0 + pl > len(genome):
                    continue
                fwd_seq = genome[fs0:fs0 + pl]
                # Reverse primer
                re0 = amp_end - 1
                rs0 = re0 - pl + 1
                if rs0 < 0 or re0 >= len(genome):
                    continue
                rev_seq = reverse_complement(genome[rs0:re0 + 1])

                fs = score_rpa_primer(fwd_seq)
                rs = score_rpa_primer(rev_seq)

                fwd = RPAPrimer(
                    f"F_{amp_start}_{pl}nt", fwd_seq, amp_start, pl, "F",
                    compute_gc(fwd_seq), compute_tm_basic(fwd_seq),
                    max_hp(fwd_seq), fs,
                )
                rev = RPAPrimer(
                    f"R_{amp_end}_{pl}nt", rev_seq, amp_end, pl, "R",
                    compute_gc(rev_seq), compute_tm_basic(rev_seq),
                    max_hp(rev_seq), rs,
                )
                amplicons.append(RPAAmplicon(
                    f"amp_{amp_start}-{amp_end}", fwd, rev,
                    amp_end - amp_start + 1, target_names, fs + rs,
                ))

    amplicons.sort(key=lambda x: x.score, reverse=True)
    return amplicons[:n_candidates]


# ═══════════════════════════════════════════════════════════════════════════
# 6. FULL ASSAY PIPELINE — DUAL-PASS PAM + JEPA RANKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssayDesign:
    group_name: str
    description: str
    amplicon: Optional[RPAAmplicon]
    crrna_designs: List[CrRNADesign]
    all_pam_sites: Dict[str, List[PAMSite]]
    all_candidates: Dict[str, int]
    verification: Dict[str, bool]


def design_for_target(
    genome: str,
    snp: ResistanceSNP,
    pam_mode: str,
    jepa: JEPAActivityPredictor,
) -> Tuple[Optional[CrRNADesign], List[PAMSite], int, str]:
    """
    Design best crRNA for one target SNP with JEPA-guided ranking.

    Returns: (best_design, pam_sites, n_candidates, status)

    pam_mode = "auto"    → try TTTV first, fall back to TTYN
    pam_mode = "strict"  → TTTV only
    pam_mode = "relaxed" → TTYN only
    """
    modes_to_try: List[str] = []
    if pam_mode == "auto":
        modes_to_try = ["strict", "relaxed"]
    elif pam_mode == "strict":
        modes_to_try = ["strict"]
    elif pam_mode == "relaxed":
        modes_to_try = ["relaxed"]

    all_pams: List[PAMSite] = []
    all_designs: List[CrRNADesign] = []

    for mode in modes_to_try:
        pams = scan_pam_sites(genome, snp, spacer_len=25,
                              max_distance=150, pam_mode=mode)
        if not pams:
            continue
        all_pams.extend(pams)

        for pam in pams[:10]:       # top 10 PAMs per mode
            candidates = design_crrna_candidates(genome, snp, pam)
            all_designs.extend(candidates)

        # If strict mode found SM designs, skip relaxed
        sm_found = any(d.design_type == "SM" for d in all_designs)
        if sm_found and mode == "strict":
            break

    if not all_designs:
        return None, all_pams, 0, "NO_PAM"

    # ── v5: JEPA-guided ranking (scores + predicts + sorts) ──
    ranked = jepa.rank_designs(all_designs)

    # Selection priority: SM > detect_only
    sm = [d for d in ranked if d.design_type == "SM"]
    nosm = [d for d in ranked if d.design_type == "detect_only"]

    if sm:
        return sm[0], all_pams, len(all_designs), "SM"
    elif nosm:
        return nosm[0], all_pams, len(all_designs), "detect_only"
    else:
        return None, all_pams, len(all_designs), "NO_VIABLE"


def design_full_assay(
    genome: str,
    pam_mode: str = "auto",
    jepa: Optional[JEPAActivityPredictor] = None,
) -> Dict[str, AssayDesign]:
    """Run complete assay design with dual-pass PAM + JEPA ranking."""
    if jepa is None:
        jepa = JEPAActivityPredictor(None)

    assay_designs: Dict[str, AssayDesign] = {}

    for group_name, group in AMPLICON_GROUPS.items():
        log.info("=" * 60)
        log.info("  %s: %s", group_name, group["description"])
        log.info("=" * 60)

        group_snps = [
            s for s in TARGETS
            if s.mutation_ecoli in group["targets"]
            or s.mutation_mtb in group["targets"]
        ]

        # ── Verify SNPs ──
        verification: Dict[str, bool] = {}
        for snp in group_snps:
            ok = verify_snp(genome, snp)
            verification[snp.mutation_ecoli] = ok
            log.info("  Verify %s (%d %s>%s): %s",
                     snp.mutation_ecoli, snp.genome_pos,
                     snp.ref_allele, snp.alt_allele,
                     "OK" if ok else "MISMATCH")

        # ── Design crRNAs per SNP ──
        all_pam_sites: Dict[str, List[PAMSite]] = {}
        all_candidates: Dict[str, int] = {}
        best_crrnas: List[CrRNADesign] = []

        for snp in group_snps:
            best, pams, n_cand, status = design_for_target(
                genome, snp, pam_mode, jepa)
            all_pam_sites[snp.mutation_ecoli] = pams
            all_candidates[snp.mutation_ecoli] = n_cand

            n_strict = sum(1 for p in pams if p.pam_type == "strict")
            n_relaxed = sum(1 for p in pams if p.pam_type == "relaxed")

            if best:
                pam_label = "TTTV" if best.pam_type == "strict" else "TTYN"
                sm_info = (
                    f"SM@{best.synthetic_mm_pos} ({best.synthetic_mm_type})"
                    if best.design_type == "SM" else "detect_only"
                )
                jepa_str = (f" JEPA={best.jepa_activity:.3f}"
                            if jepa.available else "")
                log.info(
                    "  %s %s: %s | SNP@pos%d | %dnt | PAM=%s [%s] | "
                    "score=%.1f%s | PAMs: %dstrict+%drelaxed | %d cands",
                    "*" if status == "SM" else "~",
                    snp.mutation_ecoli, sm_info,
                    best.snp_pos_in_spacer, best.spacer_length,
                    best.pam_seq, pam_label, best.score, jepa_str,
                    n_strict, n_relaxed, n_cand,
                )
                log.info("    5'-%s-3' (%dMM MUT, %dMM WT)",
                         best.spacer_designed,
                         best.n_mm_vs_mutant, best.n_mm_vs_wildtype)
                best_crrnas.append(best)
            else:
                log.info("  X %s: %s (PAMs: %dstrict+%drelaxed)",
                         snp.mutation_ecoli, status, n_strict, n_relaxed)

        # ── RPA primers ──
        target_names = [s.mutation_ecoli for s in group_snps]
        amplicons = design_rpa_primers(genome, group["center"], target_names)
        best_amp = amplicons[0] if amplicons else None
        if best_amp:
            log.info("  RPA: %dbp", best_amp.amplicon_length)
            log.info("    F: %s (GC=%.0f%%)",
                     best_amp.forward.sequence,
                     best_amp.forward.gc_content * 100)
            log.info("    R: %s (GC=%.0f%%)",
                     best_amp.reverse.sequence,
                     best_amp.reverse.gc_content * 100)

        assay_designs[group_name] = AssayDesign(
            group_name, group["description"], best_amp,
            best_crrnas, all_pam_sites, all_candidates, verification,
        )

    return assay_designs


# ═══════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

COL = {
    "pam": "#e6550d", "snp": "#e41a1c", "sm": "#ff7f00",
    "seed": "#d4eac7", "nonseed": "#e8e8e8",
    "rpoB": "#2171b5", "katG": "#08519c", "inhA_promoter": "#6a3d9a",
}


def plot_assay_architecture(designs: Dict[str, AssayDesign], out: Path):
    """Generate the main assay architecture figure with PAM + spacer diagrams."""
    plt.rcParams.update(STYLE)
    all_d = [c for d in designs.values() for c in d.crrna_designs]
    if not all_d:
        log.warning("No designs for architecture figure")
        return

    n = len(all_d)
    fig = plt.figure(figsize=(20, max(6, 2.5 * n + 3)))
    gs = gridspec.GridSpec(n + 1, 1, height_ratios=[2.5] * n + [3], hspace=0.5)

    row = 0
    for gn, design in designs.items():
        for cr in design.crrna_designs:
            ax = fig.add_subplot(gs[row])
            row += 1
            gene_c = COL.get(cr.target_snp.gene, "#333")
            sp = cr.spacer_length
            ax.set_xlim(-0.5, 4 + 0.5 + sp * 0.8 + 14)
            ax.set_ylim(-0.5, 2.5)

            pam_label = "TTTV" if cr.pam_type == "strict" else "TTYN"
            jepa_tag = (f" JEPA={cr.jepa_activity:.2f}"
                        if cr.jepa_activity > 0 else "")
            ax.text(
                -0.3, 2.2,
                f"{cr.target_snp.gene} {cr.target_snp.mutation_ecoli} "
                f"({cr.target_snp.drug}) [{cr.design_type}|{pam_label}]{jepa_tag}",
                fontsize=11, fontweight="bold", color=gene_c, va="center",
            )

            # PAM box
            ax.add_patch(FancyBboxPatch(
                (0, 0.8), 4, 0.9,
                boxstyle="round,pad=0.1", facecolor=COL["pam"],
                edgecolor="black", linewidth=1.5, alpha=0.9,
            ))
            ax.text(2, 1.25, f"PAM\n{cr.pam_seq}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

            # Spacer nucleotides
            x0 = 4.5
            for j, base in enumerate(cr.spacer_designed):
                pos = j + 1
                x = x0 + j * 0.8
                if pos == cr.snp_pos_in_spacer:
                    fc, ec, lw = COL["snp"], "black", 2
                elif pos == cr.synthetic_mm_pos:
                    fc, ec, lw = COL["sm"], "black", 2
                elif pos <= 8:
                    fc, ec, lw = COL["seed"], "#999", 0.5
                else:
                    fc, ec, lw = COL["nonseed"], "#999", 0.5
                ax.add_patch(plt.Rectangle(
                    (x, 0.8), 0.7, 0.9,
                    facecolor=fc, edgecolor=ec, linewidth=lw,
                ))
                fw = ("bold" if pos in (cr.snp_pos_in_spacer,
                                        cr.synthetic_mm_pos) else "normal")
                ax.text(x + 0.35, 1.25, base, ha="center", va="center",
                        fontsize=7, fontweight=fw)
                if pos <= 10 or pos == sp or pos == cr.snp_pos_in_spacer:
                    ax.text(x + 0.35, 0.55, str(pos), ha="center", va="center",
                            fontsize=6, color="#666")

            # Seed bracket
            se = x0 + min(8, sp) * 0.8
            ax.annotate("", xy=(x0, 0.3), xytext=(se, 0.3),
                        arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.5))
            ax.text((x0 + se) / 2, 0.05, "SEED(1-8)", ha="center", fontsize=7,
                    color="#2ca02c", fontweight="bold")

            # Info panel
            ix = x0 + sp * 0.8 + 1
            info = [
                f"{sp}nt GC={cr.gc_content:.0%} score={cr.score:.1f}",
                f"SNP@{cr.snp_pos_in_spacer} strand:{cr.strand}",
            ]
            if cr.synthetic_mm_pos > 0:
                info += [
                    f"SM: {cr.synthetic_mm_type}",
                    f"MUT:{cr.n_mm_vs_mutant}MM→ON  WT:{cr.n_mm_vs_wildtype}MM→OFF",
                ]
            else:
                info += [
                    "No SM (detect only)",
                    f"MUT:0MM→ON  WT:1MM→reduced",
                ]
            if cr.jepa_activity > 0:
                info.append(f"JEPA activity: {cr.jepa_activity:.3f}")
            ax.text(ix, 1.25, "\n".join(info), fontsize=7, va="center",
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="#f9f9f9", alpha=0.8))
            ax.set_yticks([])
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)

    # Legend
    if row > 0:
        fig.axes[row - 1].legend(handles=[
            mpatches.Patch(facecolor=COL["pam"], label="PAM"),
            mpatches.Patch(facecolor=COL["seed"], label="Seed(1-8)"),
            mpatches.Patch(facecolor=COL["snp"], label="SNP"),
            mpatches.Patch(facecolor=COL["sm"], label="Synth.MM"),
        ], loc="upper right", fontsize=8, framealpha=0.9)

    # Workflow diagram
    axw = fig.add_subplot(gs[-1])
    axw.set_xlim(0, 10)
    axw.set_ylim(0, 3)
    axw.axis("off")
    steps = [
        ("1.Sample", "#bdbdbd", 0.5),
        ("2.DNA ext", "#969696", 2.0),
        ("3.RPA\n39°C 20min", "#984ea3", 3.5),
        ("4.Cas12a+crRNA\n+reporter 37°C", "#2171b5", 5.5),
        ("5.Electrochem\nSWV readout", "#e6550d", 7.5),
        ("6.MDR-TB\nresult", "#e41a1c", 9.2),
    ]
    for lb, c, x in steps:
        axw.add_patch(FancyBboxPatch(
            (x - 0.6, 0.5), 1.3, 1.8,
            boxstyle="round,pad=0.15", facecolor=c, alpha=0.2,
            edgecolor=c, linewidth=2,
        ))
        axw.text(x + 0.05, 1.4, lb, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=c)
    for i in range(len(steps) - 1):
        axw.annotate(
            "", xy=(steps[i + 1][2] - 0.6, 1.4),
            xytext=(steps[i][2] + 0.7, 1.4),
            arrowprops=dict(arrowstyle="->", color="#333", lw=2),
        )
    axw.text(5, 0.1,
             "<60min | heating block + potentiostat | "
             "4 electrodes (rpoB+katG+inhA+IS6110)",
             ha="center", fontsize=10, fontstyle="italic", color="#555")
    axw.set_title("Point-of-Care MDR-TB Workflow",
                  fontsize=13, fontweight="bold")

    plt.savefig(out / "mdrtb_assay_architecture.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  → mdrtb_assay_architecture.png")


def plot_discrimination_diagram(designs: Dict[str, AssayDesign], out: Path):
    """Show mismatch discrimination for the best SM design."""
    plt.rcParams.update(STYLE)

    best: Optional[CrRNADesign] = None
    for d in designs.values():
        for c in d.crrna_designs:
            if c.design_type == "SM" and (best is None or c.score > best.score):
                best = c
    if best is None:
        for d in designs.values():
            if d.crrna_designs:
                best = d.crrna_designs[0]
                break
    if best is None:
        log.warning("No designs for discrimination diagram")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (ax, label) in enumerate(
        zip(axes, ["vs MUTANT target", "vs WILDTYPE target"])
    ):
        ax.set_xlim(-1, 24)
        ax.set_ylim(-2, 6)
        ax.set_title(label, fontsize=14, fontweight="bold",
                     color="#2ca02c" if idx == 0 else "#e41a1c")
        sp = best.spacer_length

        # crRNA row
        ax.text(-0.5, 5, "crRNA:", fontsize=10, fontweight="bold", va="center")
        for j, base in enumerate(best.spacer_designed):
            pos = j + 1
            x = j * 0.85
            fc = (
                "#ff9999" if pos == best.snp_pos_in_spacer else
                "#ffcc66" if pos == best.synthetic_mm_pos else
                "#d4eac7" if pos <= 8 else "#e8e8e8"
            )
            ax.add_patch(plt.Rectangle((x, 4.2), 0.75, 0.8, facecolor=fc,
                                       edgecolor="#333", linewidth=0.8))
            ax.text(x + 0.375, 4.6, base, ha="center", va="center",
                    fontsize=7, fontweight="bold")

        # Target row
        ax.text(-0.5, 3, "Target:", fontsize=10, fontweight="bold", va="center")
        tgt = best.spacer_mut if idx == 0 else best.spacer_wt
        n_mm = 0
        for j in range(min(sp, len(tgt))):
            x = j * 0.85
            is_mm = best.spacer_designed[j] != tgt[j]
            if is_mm:
                n_mm += 1
            fc = "#ffcccc" if is_mm else "#e8e8e8"
            ax.add_patch(plt.Rectangle((x, 2.2), 0.75, 0.8, facecolor=fc,
                                       edgecolor="#333", linewidth=0.8))
            ax.text(x + 0.375, 2.6, tgt[j], ha="center", va="center",
                    fontsize=7, fontweight="bold")
            if is_mm:
                ax.text(x + 0.375, 3.6, "X", ha="center", va="center",
                        fontsize=14, color="red", fontweight="bold")
            else:
                ax.plot([x + 0.375] * 2, [3.0, 4.2],
                        color="#ccc", lw=0.5, ls=":")

        # Result annotation
        if idx == 0:
            rc = "#2ca02c"
            rt = f"{n_mm}MM → TOLERATED\nCas12a ON → SIGNAL"
            oc = "DETECTED"
        else:
            rc = "#e41a1c"
            rt = f"{n_mm}MM → NOT TOLERATED\nCas12a OFF → NO SIGNAL"
            oc = "NO SIGNAL"
        ax.add_patch(FancyBboxPatch(
            (2, -1.5), 14, 1.3, boxstyle="round,pad=0.2",
            facecolor=rc, alpha=0.15, edgecolor=rc, linewidth=2,
        ))
        ax.text(9, -0.85, rt, ha="center", va="center",
                fontsize=10, color=rc, fontweight="bold")
        ax.text(9, 1.5, oc, ha="center", va="center",
                fontsize=16, color=rc, fontweight="bold")
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    snp = best.target_snp
    sm_l = f"SM@{best.synthetic_mm_pos}" if best.synthetic_mm_pos > 0 else "no SM"
    pam_l = "TTTV" if best.pam_type == "strict" else "TTYN"
    fig.suptitle(
        f"Discrimination: {snp.gene} {snp.mutation_ecoli} ({snp.drug})\n"
        f"{best.design_type} | {best.spacer_length}nt | "
        f"SNP@{best.snp_pos_in_spacer} | {sm_l} | PAM={best.pam_seq} [{pam_l}]",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out / "mdrtb_discrimination_diagram.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  → mdrtb_discrimination_diagram.png")


def plot_pam_landscape(designs: Dict[str, AssayDesign], out: Path):
    """Scatter plot of PAM sites by SNP position in spacer."""
    plt.rcParams.update(STYLE)
    n = len(designs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (gn, d) in zip(axes, designs.items()):
        first = True
        for sn, pams in d.all_pam_sites.items():
            strict_pos = [p.snp_position_in_spacer for p in pams
                          if p.pam_type == "strict"]
            relax_pos = [p.snp_position_in_spacer for p in pams
                         if p.pam_type == "relaxed"]
            ax.scatter(strict_pos, [sn] * len(strict_pos), c="#2171b5",
                       s=60, marker="^", label="TTTV" if first else "",
                       alpha=0.7, edgecolors="black", linewidths=0.5)
            ax.scatter(relax_pos, [sn] * len(relax_pos), c="#e6550d",
                       s=60, marker="v", label="TTYN" if first else "",
                       alpha=0.7, edgecolors="black", linewidths=0.5)
            first = False
        ax.axvspan(1, 8, alpha=0.12, color="green", label="Seed(1-8)")
        ax.axvspan(9, 12, alpha=0.06, color="blue", label="Ext.seed(9-12)")
        tot = sum(len(p) for p in d.all_pam_sites.values())
        ax.set_xlabel("SNP pos in spacer")
        ax.set_xlim(0, 26)
        ax.set_title(f"{gn} ({tot} PAMs)", fontweight="bold")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out / "mdrtb_pam_landscape.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  → mdrtb_pam_landscape.png")


def generate_summary_table(designs: Dict[str, AssayDesign], out: Path):
    """Generate summary table as a PNG image."""
    plt.rcParams.update(STYLE)
    all_d = [c for d in designs.values() for c in d.crrna_designs]
    if not all_d:
        log.warning("No designs for summary table")
        return

    fig, ax = plt.subplots(figsize=(26, max(4, 1.2 * len(all_d) + 3)))
    ax.axis("off")

    hdr = [
        "Gene", "Mutation", "Drug", "Type", "PAM", "PAM\nType", "Strand",
        "crRNA Spacer 5'→3'", "Len", "SNP\nPos", "SM\nPos",
        "MM\nMUT", "MM\nWT", "GC%", "Score", "JEPA",
    ]
    rows = []
    for c in all_d:
        s = c.target_snp
        rows.append([
            s.gene, s.mutation_ecoli, s.drug[:3], c.design_type,
            c.pam_seq, "TTTV" if c.pam_type == "strict" else "TTYN",
            c.strand, c.spacer_designed, str(c.spacer_length),
            str(c.snp_pos_in_spacer),
            str(c.synthetic_mm_pos) if c.synthetic_mm_pos > 0 else "-",
            str(c.n_mm_vs_mutant), str(c.n_mm_vs_wildtype),
            f"{c.gc_content:.0%}", f"{c.score:.1f}",
            f"{c.jepa_activity:.3f}" if c.jepa_activity > 0 else "-",
        ])

    t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.scale(1.0, 1.5)

    # Header styling
    for j in range(len(hdr)):
        t[0, j].set_facecolor("#2171b5")
        t[0, j].set_text_props(color="white", fontweight="bold")

    # Row colouring by gene and design type
    gbg = {"rpoB": "#e3f2fd", "katG": "#fce4ec", "inhA_promoter": "#f3e5f5"}
    tbg = {"SM": "#e8f5e9", "detect_only": "#fff3e0"}
    for i, r in enumerate(rows):
        for j in range(len(hdr)):
            t[i + 1, j].set_facecolor(
                tbg.get(r[3], "#fff") if j == 3 else gbg.get(r[0], "#fff")
            )

    nsm = sum(1 for c in all_d if c.design_type == "SM")
    ax.set_title(
        f"MDR-TB crRNA Summary: {len(all_d)} designs "
        f"({nsm} SM + {len(all_d) - nsm} detect_only)",
        fontsize=14, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    fig.savefig(out / "mdrtb_crrna_summary_table.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  → mdrtb_crrna_summary_table.png")


def generate_rpa_table(designs: Dict[str, AssayDesign], out: Path):
    """Generate RPA primer table as a PNG image."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")

    hdr = ["Amplicon", "Targets", "Primer", "Sequence 5'→3'", "Len",
           "GC%", "HP", "Amp", "Score"]
    rows = []
    for gn, d in designs.items():
        a = d.amplicon
        if not a:
            continue
        rows.append([
            gn, ", ".join(a.targets_covered), "Fwd",
            a.forward.sequence, str(a.forward.length),
            f"{a.forward.gc_content:.0%}",
            str(a.forward.max_homopolymer),
            f"{a.amplicon_length}bp", f"{a.score:.1f}",
        ])
        rows.append([
            "", "", "Rev", a.reverse.sequence, str(a.reverse.length),
            f"{a.reverse.gc_content:.0%}",
            str(a.reverse.max_homopolymer), "", "",
        ])

    if rows:
        t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1.0, 1.6)
        for j in range(len(hdr)):
            t[0, j].set_facecolor("#984ea3")
            t[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("RPA Primers (30-35nt, 120-260bp amplicons, 37-39°C)",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(out / "mdrtb_rpa_primers_table.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  → mdrtb_rpa_primers_table.png")


def plot_design_statistics(designs: Dict[str, AssayDesign], out: Path):
    """Bar charts of PAM availability and candidate counts."""
    plt.rcParams.update(STYLE)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    # PAM availability
    tgts, np_, ns_ = [], [], []
    for d in designs.values():
        for sn, pams in d.all_pam_sites.items():
            tgts.append(sn)
            np_.append(len(pams))
            ns_.append(sum(1 for p in pams if 1 <= p.snp_position_in_spacer <= 12))
    y = range(len(tgts))
    a1.barh(list(y), np_, color="#74c476", alpha=0.8, label="Total PAMs")
    a1.barh(list(y), ns_, color="#2171b5", alpha=0.8, label="SNP in ext.seed")
    a1.set_yticks(list(y))
    a1.set_yticklabels(tgts, fontsize=9)
    a1.set_xlabel("PAM sites")
    a1.legend(fontsize=8)
    a1.set_title("PAM Availability (TTTV+TTYN)\nMtb GC=65.6%", fontweight="bold")

    # Candidates per target
    t2, nc, dt, sc = [], [], [], []
    for d in designs.values():
        for c in d.crrna_designs:
            nm = c.target_snp.mutation_ecoli
            t2.append(nm)
            nc.append(d.all_candidates.get(nm, 0))
            dt.append(c.design_type)
            sc.append(c.score)
    if t2:
        bc = ["#2ca02c" if d == "SM" else "#ff7f00" for d in dt]
        y2 = range(len(t2))
        a2.barh(list(y2), nc, color=bc, alpha=0.8)
        a2.set_yticks(list(y2))
        a2.set_yticklabels(t2, fontsize=9)
        a2.set_xlabel("Candidates")
        a2.set_title("Candidates per Target\n(green=SM, orange=detect_only)",
                     fontweight="bold")
        for i in range(len(t2)):
            a2.text(nc[i] + 0.5, i, f"s={sc[i]:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out / "mdrtb_design_statistics.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  → mdrtb_design_statistics.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. JEPA INTEGRATION SUMMARY + EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def jepa_integration_summary(
    designs: Dict[str, AssayDesign],
    out: Path,
    jepa_available: bool = False,
):
    """Write human-readable integration summary."""
    tot = sum(len(d.crrna_designs) for d in designs.values())
    nsm = sum(1 for d in designs.values()
              for c in d.crrna_designs if c.design_type == "SM")
    tcand = sum(v for d in designs.values()
                for v in d.all_candidates.values())
    n_strict = sum(1 for d in designs.values()
                   for c in d.crrna_designs if c.pam_type == "strict")
    n_relax = tot - n_strict

    jepa_status = "ACTIVE (checkpoint loaded)" if jepa_available else "INACTIVE (rule-based only)"

    lines = [
        "=" * 70,
        "  DNA-Bacteria-JEPA → MDR-TB Diagnostic: Integration Summary (v5)",
        "=" * 70, "",
        "1. PRE-TRAINING (completed)",
        "   Model: SparseTransformerEncoder (384D × 6L × 6H, 1024 FFN)",
        "   Training: 200 epochs, 20 bacterial genomes, RankMe 380/384",
        "   Architecture: I-JEPA multi-block masking + transformer predictor",
        "   Losses: VICReg + LDReg + RC consistency + SupCon + GC adversary",
        "   Checkpoint: target_encoder_state_dict (EMA, best for inference)",
        "   Tokenizer: Cas12aTokenizer (character-level: A=1,C=2,G=3,T=4)",
        "   HuggingFace: orgava/dna-bacteria-jepa", "",
        "2. Cas12a FINE-TUNING (planned — Y1 milestone)",
        "   Dataset: EasyDesign 10,634 LbCas12a pairs",
        "   Expected: JEPA frozen → ρ≈0.48 vs random init ρ≈0.25",
        "   Script: scripts/06_cas12a_finetune.py", "",
        f"3. JEPA RANKING STATUS: {jepa_status}", "",
        f"4. MDR-TB ASSAY DESIGN v5",
        f"   {tot} crRNAs: {nsm} SM + {tot - nsm} detect_only",
        f"   PAM types: {n_strict} TTTV + {n_relax} TTYN",
        f"   From {tcand} candidates across {len(designs)} regions",
        "   Targets: rpoB(RIF) + katG(INH) + inhA(INH/ETH)",
        "   Challenge: Mtb GC=65.6% → dual PAM strategy", "",
        "5. JEPA-GUIDED RANKING PIPELINE",
        "   a) Extract 34nt context window (PAM+spacer+flank) per candidate",
        "   b) Tokenize via Cas12aTokenizer → (B, L) token IDs",
        "   c) Embed through frozen SparseTransformerEncoder → (B, 384)",
        "   d) Score: activity_head(embedding) if fine-tuned,",
        "      or normalised embedding L2 norm if pretrained only",
        "   e) Rank: combined = JEPA_score × (1 + rule_score / max_score)",
        "   f) Select top SM design per target for cartridge", "",
        "6. PhD ROADMAP (ETH Zürich, deMello/Richards)",
        "   Y1: Scale to 1000+ genomes, fine-tune Cas12a activity head,",
        "       validate vs DeepCpf1/CRISPR-ML, integrate enAsCas12a data",
        "   Y2: Optimise 6-8 target MDR-TB panel with CSEM cartridge",
        "   Y3: Clinical validation Swiss TPH + NCTLD Georgia",
        "=" * 70,
    ]
    text = "\n".join(lines)
    log.info("\n%s", text)
    (out / "jepa_mdrtb_integration.txt").write_text(text)
    log.info("  → jepa_mdrtb_integration.txt")


def export_designs(designs: Dict[str, AssayDesign], out: Path):
    """Export all designs as JSON."""
    export: Dict[str, dict] = {}
    for gn, d in designs.items():
        ge: dict = {
            "description": d.description,
            "verification": d.verification,
            "candidates_evaluated": d.all_candidates,
            "crrnas": [],
            "rpa_primers": None,
        }
        for c in d.crrna_designs:
            s = c.target_snp
            ge["crrnas"].append({
                "name": c.name,
                "gene": s.gene,
                "mutation": s.mutation_ecoli,
                "drug": s.drug,
                "genome_pos": s.genome_pos,
                "ref_allele": s.ref_allele,
                "alt_allele": s.alt_allele,
                "pam_seq": c.pam_seq,
                "pam_strand": c.strand,
                "pam_type": c.pam_type,
                "spacer_designed": c.spacer_designed,
                "spacer_wt": c.spacer_wt,
                "spacer_mut": c.spacer_mut,
                "spacer_length": c.spacer_length,
                "snp_pos_in_spacer": c.snp_pos_in_spacer,
                "synthetic_mm_pos": c.synthetic_mm_pos,
                "synthetic_mm_type": c.synthetic_mm_type,
                "n_mm_vs_mutant": c.n_mm_vs_mutant,
                "n_mm_vs_wildtype": c.n_mm_vs_wildtype,
                "gc_content": round(c.gc_content, 3),
                "design_type": c.design_type,
                "score": round(c.score, 2),
                "jepa_activity": round(c.jepa_activity, 4),
                "combined_score": round(c.combined_score, 4),
                "context_seq": c.context_seq,
            })
        if d.amplicon:
            a = d.amplicon
            ge["rpa_primers"] = {
                "amplicon_size": a.amplicon_length,
                "forward": {
                    "sequence": a.forward.sequence,
                    "position": a.forward.position,
                    "length": a.forward.length,
                    "gc_content": round(a.forward.gc_content, 3),
                    "tm": round(a.forward.tm_rough, 1),
                },
                "reverse": {
                    "sequence": a.reverse.sequence,
                    "position": a.reverse.position,
                    "length": a.reverse.length,
                    "gc_content": round(a.reverse.gc_content, 3),
                    "tm": round(a.reverse.tm_rough, 1),
                },
            }
        export[gn] = ge

    out_json = out / "mdrtb_assay_designs.json"
    out_json.write_text(json.dumps(export, indent=2))
    log.info("  → mdrtb_assay_designs.json")


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="MDR-TB CRISPR-Cas12a assay design v5 "
                    "(DNA-JEPA guided crRNA ranking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--genome", required=True,
                   help="Path to H37Rv FASTA (plain or .gz)")
    p.add_argument("--output-dir", default="results/mdrtb_assay",
                   help="Output directory (default: results/mdrtb_assay)")
    p.add_argument("--pam-mode", choices=["strict", "relaxed", "auto"],
                   default="auto",
                   help="PAM strategy: strict=TTTV, relaxed=TTYN, "
                        "auto=TTTV→TTYN fallback (default: auto)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to JEPA Cas12a fine-tuned checkpoint (.pt). "
                        "If omitted, uses rule-based scoring only.")
    p.add_argument("--spacer-lengths", default="17,18,20,23,25",
                   help="Comma-separated spacer lengths to try "
                        "(default: 17,18,20,23,25)")
    args = p.parse_args()

    # ── Resolve paths ──
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    gp = Path(args.genome)
    if not gp.is_absolute():
        gp = (PROJECT_ROOT / gp).resolve()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path and not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()

    # ── Banner ──
    log.info("=" * 60)
    log.info("  MDR-TB CRISPR-Cas12a Assay Design v5")
    log.info("  DNA-Bacteria-JEPA guided ranking")
    log.info("=" * 60)
    log.info("  Genome:     %s", gp)
    log.info("  Output:     %s", out)
    log.info("  PAM mode:   %s", args.pam_mode)
    log.info("  Checkpoint: %s", ckpt_path or "NONE (rule-based)")
    log.info("  Targets:    %d SNPs, 3 loci", len(TARGETS))
    log.info("  Search:     spacer_len=25, max_dist=150, amp=120-260bp")
    log.info("  Spacers:    %s", args.spacer_lengths)

    # ── Load genome ──
    log.info("Loading genome...")
    genome = load_genome(gp)
    assert len(genome) > 4_000_000, (
        f"Genome too short ({len(genome)} bp) — expected H37Rv ~4.4Mbp"
    )

    # ── Verify SNPs ──
    log.info("Verifying %d SNP targets...", len(TARGETS))
    all_ok = all(verify_snp(genome, s) for s in TARGETS)
    log.info("  %s %d SNPs %s",
             "✓" if all_ok else "⚠",
             len(TARGETS),
             "verified" if all_ok else "had mismatches")

    # ── Initialise JEPA predictor ──
    jepa = JEPAActivityPredictor(ckpt_path)

    # ── Design assays ──
    log.info("Designing assays (PAM mode: %s)...", args.pam_mode)
    designs = design_full_assay(genome, args.pam_mode, jepa)

    # ── Generate all outputs ──
    log.info("=" * 60)
    log.info("  Generating outputs...")
    log.info("=" * 60)

    plot_assay_architecture(designs, out)
    plot_discrimination_diagram(designs, out)
    plot_pam_landscape(designs, out)
    generate_summary_table(designs, out)
    generate_rpa_table(designs, out)
    plot_design_statistics(designs, out)
    export_designs(designs, out)
    jepa_integration_summary(designs, out, jepa_available=jepa.available)

    # ── Final summary ──
    tot = sum(len(d.crrna_designs) for d in designs.values())
    nsm = sum(1 for d in designs.values()
              for c in d.crrna_designs if c.design_type == "SM")
    gaps = sum(1 for d in designs.values()
               for nm, ps in d.all_pam_sites.items() if len(ps) == 0)
    tcand = sum(v for d in designs.values()
                for v in d.all_candidates.values())

    log.info("=" * 60)
    log.info("  DESIGN COMPLETE")
    log.info("=" * 60)
    log.info("  crRNAs:     %d (%d SM + %d detect_only)", tot, nsm, tot - nsm)
    log.info("  Candidates: %d evaluated", tcand)
    log.info("  No PAM:     %d targets", gaps)
    log.info("  JEPA:       %s", "active" if jepa.available else "rule-based")
    log.info("  PAM mode:   %s", args.pam_mode)
    log.info("  Outputs:    %s", out)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
