#!/usr/bin/env python3
"""
09_mdrtb_assay_design.py v4 — CRISPR-Cas12a assay design for MDR-TB diagnostics.

FIXES for M. tuberculosis (65.6% GC → extremely sparse TTTV PAMs):
  v2-v3: Only 1/9 targets got designs (0.40 expected TTTV PAMs per target)
  v4 solution:
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

Targets: rpoB RRDR (S531L, H526Y/D, D516V, L533P), katG S315T/N,
         inhA promoter c.-15C>T, c.-8T>C
Reference: M. tuberculosis H37Rv (NC_000962.3, 4,411,532 bp)

Usage:
  python3 scripts/09_mdrtb_assay_design.py \\
      --genome data/H37Rv.fna --output-dir results/mdrtb_assay

  # Force enAsCas12a PAMs for all targets:
  python3 scripts/09_mdrtb_assay_design.py \\
      --genome data/H37Rv.fna --pam-mode relaxed

Author: Valentin Vézina | DNA-Bacteria-JEPA / ETH Zürich PhD Application
"""
from __future__ import annotations
import argparse, sys, os, json, warnings, re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# 1. VERIFIED SNP TARGET DEFINITIONS (NC_000962.3)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResistanceSNP:
    gene: str
    mutation_ecoli: str
    mutation_mtb: str
    genome_pos: int          # 1-indexed on + strand of NC_000962.3
    ref_allele: str          # Reference allele (+ strand)
    alt_allele: str          # Resistance allele (+ strand)
    codon_wt: str
    codon_mut: str
    drug: str
    frequency: str
    gene_strand: str
    gene_start: int
    gene_end: int

TARGETS = [
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
    ResistanceSNP("katG", "S315T", "S315T", 2155168, "C", "G",
                  "AGC", "ACC", "Isoniazid", "50-94% INH-R",
                  "-", 2153889, 2156111),
    ResistanceSNP("katG", "S315N", "S315N", 2155168, "C", "T",
                  "AGC", "AAC", "Isoniazid", "~3% INH-R",
                  "-", 2153889, 2156111),
    ResistanceSNP("inhA_promoter", "c.-15C>T", "c.-15C>T", 1673425, "C", "T",
                  "C", "T", "Isoniazid/Ethionamide", "29-35% INH-R",
                  "+", 1673440, 1674183),
    ResistanceSNP("inhA_promoter", "c.-8T>C", "c.-8T>C", 1673432, "T", "C",
                  "T", "C", "Isoniazid", "~4% INH-R",
                  "+", 1673440, 1674183),
]

AMPLICON_GROUPS = {
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


# ═══════════════════════════════════════════════════════════════════
# 2. GENOME HANDLING
# ═══════════════════════════════════════════════════════════════════

COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def reverse_complement(seq: str) -> str:
    return seq.translate(COMP)[::-1]

def load_genome(fasta_path: Path) -> str:
    import gzip
    opener = gzip.open if str(fasta_path).endswith('.gz') else open
    parts = []
    with opener(fasta_path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                if parts: break
                continue
            parts.append(line.strip())
    genome = ''.join(parts).upper()
    gc = sum(1 for b in genome if b in 'GC') / len(genome)
    print(f"  Loaded: {len(genome):,} bp, GC={gc:.1%}")
    return genome

def verify_snp(genome: str, snp: ResistanceSNP) -> bool:
    pos_0 = snp.genome_pos - 1
    if pos_0 >= len(genome):
        return False
    actual = genome[pos_0]
    if actual != snp.ref_allele:
        print(f"  WARNING: {snp.gene} {snp.mutation_ecoli} at {snp.genome_pos}: "
              f"expected {snp.ref_allele}, found {actual}")
        return False
    return True

def get_complement(base: str) -> str:
    return {"A": "T", "T": "A", "C": "G", "G": "C"}.get(base.upper(), "N")


# ═══════════════════════════════════════════════════════════════════
# 3. PAM SCANNER — DUAL MODE (TTTV / TTYN)
# ═══════════════════════════════════════════════════════════════════

PAM_PATTERNS = {
    "strict":  {"plus": r"TTT[ACG]",    "minus": r"[CGT]AAA",    "label": "TTTV (LbCas12a)"},
    "relaxed": {"plus": r"TT[CT][ACGT]","minus": r"[ACGT][AG]AA","label": "TTYN (enAsCas12a)"},
}

@dataclass
class PAMSite:
    position: int          # 1-indexed genome pos of first base of PAM motif
    strand: str            # '+' or '-'
    pam_seq: str           # 4nt PAM on non-target strand
    spacer_start: int      # 1-indexed on + strand
    spacer_end: int
    snp_position_in_spacer: int  # 1-indexed, pos 1 = PAM-proximal
    distance_to_snp: int
    pam_type: str          # "strict" or "relaxed"


def scan_pam_sites(genome: str, snp: ResistanceSNP,
                   spacer_len: int = 25,
                   max_distance: int = 150,
                   pam_mode: str = "strict") -> List[PAMSite]:
    """
    Scan for PAM sites near a SNP on both strands.

    pam_mode:
      "strict"  = TTTV only (standard LbCas12a)
      "relaxed" = TTYN (enAsCas12a, ~2.4x more PAMs in Mtb)
      "auto"    = try strict first, relaxed if strict yields 0
    """
    snp_pos = snp.genome_pos
    search_start = max(0, snp_pos - 1 - max_distance - spacer_len - 4)
    search_end = min(len(genome), snp_pos - 1 + max_distance + spacer_len + 4)
    region = genome[search_start:search_end]

    results = []
    pat = PAM_PATTERNS[pam_mode]

    # + strand: PAM-spacer→ on + strand
    for m in re.finditer(pat["plus"], region, re.IGNORECASE):
        pam_gpos = search_start + m.start() + 1
        pam_seq = region[m.start():m.start()+4]
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

    # - strand: reverse complement PAM on + strand
    for m in re.finditer(pat["minus"], region, re.IGNORECASE):
        vaaa_gpos = search_start + m.start() + 1
        sp_end_plus = vaaa_gpos - 1
        sp_start_plus = sp_end_plus - spacer_len + 1
        if sp_start_plus <= snp_pos <= sp_end_plus:
            snp_in_sp = sp_end_plus - snp_pos + 1
            pam_seq_minus = reverse_complement(region[m.start():m.start()+4])
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
        x.snp_position_in_spacer
    ))
    return results


# ═══════════════════════════════════════════════════════════════════
# 4. crRNA DESIGNER WITH SYNTHETIC MISMATCH
# ═══════════════════════════════════════════════════════════════════

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


def design_crrna_candidates(genome: str, snp: ResistanceSNP, pam: PAMSite,
                            spacer_lengths: List[int] = None,
                            ) -> List[CrRNADesign]:
    """
    Design crRNA candidates with synthetic mismatch strategy.
    Spacer lengths: [17, 18, 20, 23, 25] — broader range for GC-rich genomes.
    SM offsets: [-5..5] excluding 0.
    """
    if spacer_lengths is None:
        spacer_lengths = [17, 18, 20, 23, 25]

    designs = []
    snp_pos_0 = snp.genome_pos - 1

    for sp_len in spacer_lengths:
        if pam.strand == "+":
            sp_start_0 = pam.position - 1 + 4
            if sp_start_0 + sp_len > len(genome):
                continue
            spacer_region = genome[sp_start_0:sp_start_0 + sp_len]
            snp_idx = snp_pos_0 - sp_start_0
        else:
            sp_end_0 = pam.position - 2
            sp_start_0 = sp_end_0 - sp_len + 1
            if sp_start_0 < 0:
                continue
            spacer_region = reverse_complement(genome[sp_start_0:sp_end_0 + 1])
            snp_idx = sp_end_0 - snp_pos_0

        if snp_idx < 0 or snp_idx >= sp_len:
            continue

        snp_pos_in_spacer = snp_idx + 1

        # Build WT / mutant spacers
        spacer_wt = list(spacer_region)
        spacer_mut = list(spacer_region)
        if pam.strand == "+":
            spacer_mut[snp_idx] = snp.alt_allele
        else:
            spacer_mut[snp_idx] = get_complement(snp.alt_allele)

        wt_str = ''.join(spacer_wt)
        mut_str = ''.join(spacer_mut)
        gc = sum(1 for b in mut_str if b in 'GC') / len(mut_str)

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
        ))

        # === SM designs: try offsets [-5..5] ===
        for sm_offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            sm_idx = snp_idx + sm_offset
            if sm_idx < 0 or sm_idx >= sp_len:
                continue
            sm_pos = sm_idx + 1
            original_base = mut_str[sm_idx]

            for new_base in 'ACGT':
                if new_base == original_base:
                    continue
                designed = list(mut_str)
                designed[sm_idx] = new_base
                designed_str = ''.join(designed)

                n_mm_mut = sum(a != b for a, b in zip(designed_str, mut_str))
                n_mm_wt = sum(a != b for a, b in zip(designed_str, wt_str))

                if n_mm_mut != 1 or n_mm_wt != 2:
                    continue

                gc_d = sum(1 for b in designed_str if b in 'GC') / len(designed_str)
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
                ))

    return designs


def score_crrna(design: CrRNADesign) -> float:
    """
    Score crRNA design. Mtb-aware: tolerates high GC, extended seed.

    Scoring (higher = better):
      SM present:       +3.0 (essential for discrimination)
      SNP pos 1-6:      +3.0 | pos 7-8: +2.0 | pos 9-12: +1.0
      SM in seed (1-8): +1.5
      SM close to SNP:  +1.0
      Both in seed:     +1.0
      GC 35-75%:        +1.0 (Mtb-aware: 75% OK, >80% soft penalty)
      Spacer ≤17nt:     +1.5 | ≤18nt: +0.5
      No homopolymer:   +0.5
      SM at pos 1:      -1.0 (kills activity)
      Spacer ≥25nt:     -0.5 (tolerates mismatches)
      TTTV PAM:         +0.5 (prefer canonical over relaxed)
    """
    score = 0.0
    sp = design.snp_pos_in_spacer
    sm = design.synthetic_mm_pos

    # SNP in seed/extended seed
    if 1 <= sp <= 6:
        score += 3.0
    elif 1 <= sp <= 8:
        score += 2.0
    elif 1 <= sp <= 12:
        score += 1.0
    elif 1 <= sp <= 15:
        score += 0.3

    # Synthetic mismatch
    if sm > 0:
        score += 3.0
        if 1 <= sm <= 8:
            score += 1.5
        if abs(sm - sp) <= 2:
            score += 1.0
        if sm == 1:
            score -= 1.0
        if 1 <= sp <= 8 and 1 <= sm <= 8:
            score += 1.0

    # GC — Mtb-aware
    gc = design.gc_content
    if 0.35 <= gc <= 0.65:
        score += 1.0
    elif gc <= 0.75:
        score += 0.5     # Mtb commonly has 70-75% GC spacers
    elif gc > 0.80:
        score -= 0.5     # Soft penalty only

    # Spacer length
    if design.spacer_length <= 17:
        score += 1.5
    elif design.spacer_length <= 18:
        score += 0.5
    elif design.spacer_length >= 25:
        score -= 0.5

    # Homopolymer
    seq = design.spacer_designed
    max_run = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i-1] else 1
        max_run = max(max_run, run)
    if max_run < 4:
        score += 0.5
    elif max_run >= 5:
        score -= 1.0

    # Prefer canonical PAM
    if design.pam_type == "strict":
        score += 0.5

    return score


# ═══════════════════════════════════════════════════════════════════
# 5. RPA PRIMER DESIGNER
# ═══════════════════════════════════════════════════════════════════

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
    return sum(1 for b in s if b in 'GC') / max(len(s), 1)

def compute_tm_basic(seq: str) -> float:
    s = seq.upper()
    at = s.count('A') + s.count('T')
    gc = s.count('G') + s.count('C')
    if len(s) < 14:
        return 2 * at + 4 * gc
    return 64.9 + 41 * (gc - 16.4) / max(at + gc, 1)

def max_hp(seq: str) -> int:
    if not seq: return 0
    mx = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i-1] else 1
        mx = max(mx, run)
    return mx

def score_rpa_primer(seq: str) -> float:
    score = 0.0
    gc = compute_gc(seq)
    hp = max_hp(seq)
    if 30 <= len(seq) <= 35: score += 2.0
    elif 28 <= len(seq) <= 37: score += 1.0
    if 0.35 <= gc <= 0.65: score += 2.0
    elif 0.30 <= gc <= 0.70: score += 1.0
    else: score -= 0.5
    if hp <= 3: score += 1.0
    elif hp >= 5: score -= 1.0
    if any(b in 'GC' for b in seq[-2:].upper()): score += 0.5
    if seq[-1].upper() != 'T': score += 0.3
    return score

def design_rpa_primers(genome: str, center: int, target_names: List[str],
                       amplicon_range: Tuple[int, int] = (120, 260),
                       primer_len_range: Tuple[int, int] = (30, 35),
                       n_candidates: int = 5) -> List[RPAAmplicon]:
    amplicons = []
    min_amp, max_amp = amplicon_range
    min_pl, max_pl = primer_len_range
    for amp_size in range(min_amp, max_amp + 1, 10):
        half = amp_size // 2
        for offset in range(-30, 31, 5):
            amp_start = center - half + offset
            amp_end = amp_start + amp_size - 1
            for pl in range(min_pl, max_pl + 1):
                fs0 = amp_start - 1
                if fs0 < 0 or fs0 + pl > len(genome): continue
                fwd_seq = genome[fs0:fs0 + pl]
                re0 = amp_end - 1
                rs0 = re0 - pl + 1
                if rs0 < 0 or re0 >= len(genome): continue
                rev_seq = reverse_complement(genome[rs0:re0 + 1])
                fs = score_rpa_primer(fwd_seq)
                rs = score_rpa_primer(rev_seq)
                fwd = RPAPrimer(f"F_{amp_start}_{pl}nt", fwd_seq, amp_start,
                                pl, 'F', compute_gc(fwd_seq),
                                compute_tm_basic(fwd_seq), max_hp(fwd_seq), fs)
                rev = RPAPrimer(f"R_{amp_end}_{pl}nt", rev_seq, amp_end,
                                pl, 'R', compute_gc(rev_seq),
                                compute_tm_basic(rev_seq), max_hp(rev_seq), rs)
                amplicons.append(RPAAmplicon(
                    f"amp_{amp_start}-{amp_end}", fwd, rev,
                    amp_end - amp_start + 1, target_names, fs + rs))
    amplicons.sort(key=lambda x: x.score, reverse=True)
    return amplicons[:n_candidates]


# ═══════════════════════════════════════════════════════════════════
# 6. FULL ASSAY PIPELINE — DUAL-PASS PAM STRATEGY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AssayDesign:
    group_name: str
    description: str
    amplicon: Optional[RPAAmplicon]
    crrna_designs: List[CrRNADesign]
    all_pam_sites: Dict[str, List[PAMSite]]
    all_candidates: Dict[str, int]
    verification: Dict[str, bool]


def design_for_target(genome: str, snp: ResistanceSNP,
                      pam_mode: str) -> Tuple[Optional[CrRNADesign],
                                               List[PAMSite], int, str]:
    """
    Design best crRNA for one target SNP.
    Returns: (best_design, pam_sites, n_candidates, status)

    pam_mode = "auto": try TTTV first, fall back to TTYN
    pam_mode = "strict": TTTV only
    pam_mode = "relaxed": TTYN only
    """
    modes_to_try = []
    if pam_mode == "auto":
        modes_to_try = ["strict", "relaxed"]
    elif pam_mode == "strict":
        modes_to_try = ["strict"]
    elif pam_mode == "relaxed":
        modes_to_try = ["relaxed"]

    all_pams = []
    all_designs = []

    for mode in modes_to_try:
        pams = scan_pam_sites(genome, snp, spacer_len=25,
                              max_distance=150, pam_mode=mode)
        if not pams:
            continue

        all_pams.extend(pams)

        for pam in pams[:10]:  # Try top 10 PAMs
            candidates = design_crrna_candidates(genome, snp, pam)
            for d in candidates:
                d.score = score_crrna(d)
            all_designs.extend(candidates)

        # If strict mode found SM designs, don't need relaxed
        sm_found = any(d.design_type == "SM" for d in all_designs)
        if sm_found and mode == "strict":
            break

    if not all_designs:
        return None, all_pams, 0, "NO_PAM"

    # Selection priority: SM > detect_only
    sm = sorted([d for d in all_designs if d.design_type == "SM"],
                key=lambda x: x.score, reverse=True)
    nosm = sorted([d for d in all_designs if d.design_type == "detect_only"],
                  key=lambda x: x.score, reverse=True)

    if sm:
        return sm[0], all_pams, len(all_designs), "SM"
    elif nosm:
        return nosm[0], all_pams, len(all_designs), "detect_only"
    else:
        return None, all_pams, len(all_designs), "NO_VIABLE"


def design_full_assay(genome: str, pam_mode: str = "auto"
                      ) -> Dict[str, AssayDesign]:
    """Run complete assay design with dual-pass PAM strategy."""
    assay_designs = {}

    for group_name, group in AMPLICON_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"  {group_name}: {group['description']}")
        print(f"{'='*60}")

        group_snps = [s for s in TARGETS
                      if s.mutation_ecoli in group['targets']
                      or s.mutation_mtb in group['targets']]

        verification = {}
        for snp in group_snps:
            ok = verify_snp(genome, snp)
            verification[snp.mutation_ecoli] = ok
            print(f"  Verify {snp.mutation_ecoli} "
                  f"({snp.genome_pos} {snp.ref_allele}>{snp.alt_allele}): "
                  f"{'OK' if ok else 'MISMATCH'}")

        all_pam_sites = {}
        all_candidates = {}
        best_crrnas = []

        for snp in group_snps:
            best, pams, n_cand, status = design_for_target(
                genome, snp, pam_mode)
            all_pam_sites[snp.mutation_ecoli] = pams
            all_candidates[snp.mutation_ecoli] = n_cand

            seed_pams = sum(1 for p in pams if 1 <= p.snp_position_in_spacer <= 12)
            n_strict = sum(1 for p in pams if p.pam_type == "strict")
            n_relaxed = sum(1 for p in pams if p.pam_type == "relaxed")

            if best:
                pam_label = "TTTV" if best.pam_type == "strict" else "TTYN"
                sm_info = (f"SM@{best.synthetic_mm_pos} "
                           f"({best.synthetic_mm_type})"
                           if best.design_type == "SM" else "detect_only")
                print(f"  {'*' if status=='SM' else '~'} {snp.mutation_ecoli}: "
                      f"{sm_info} | SNP@pos{best.snp_pos_in_spacer} | "
                      f"{best.spacer_length}nt | PAM={best.pam_seq} [{pam_label}] | "
                      f"score={best.score:.1f} | "
                      f"PAMs: {n_strict}strict+{n_relaxed}relaxed | "
                      f"{n_cand} candidates")
                print(f"    5'-{best.spacer_designed}-3' "
                      f"({best.n_mm_vs_mutant}MM MUT, "
                      f"{best.n_mm_vs_wildtype}MM WT)")
                best_crrnas.append(best)
            else:
                print(f"  X {snp.mutation_ecoli}: {status} "
                      f"(PAMs: {n_strict}strict+{n_relaxed}relaxed)")

        # RPA primers
        target_names = [s.mutation_ecoli for s in group_snps]
        amplicons = design_rpa_primers(genome, group['center'], target_names)
        best_amp = amplicons[0] if amplicons else None
        if best_amp:
            print(f"\n  RPA: {best_amp.amplicon_length}bp")
            print(f"    F: {best_amp.forward.sequence} "
                  f"(GC={best_amp.forward.gc_content:.0%})")
            print(f"    R: {best_amp.reverse.sequence} "
                  f"(GC={best_amp.reverse.gc_content:.0%})")

        assay_designs[group_name] = AssayDesign(
            group_name, group['description'], best_amp,
            best_crrnas, all_pam_sites, all_candidates, verification)

    return assay_designs


# ═══════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
}

COL = {
    "pam": "#e6550d", "snp": "#e41a1c", "sm": "#ff7f00",
    "seed": "#d4eac7", "nonseed": "#e8e8e8",
    "rpoB": "#2171b5", "katG": "#08519c", "inhA_promoter": "#6a3d9a",
}


def plot_assay_architecture(designs: Dict[str, AssayDesign], out: Path):
    plt.rcParams.update(STYLE)
    all_d = [c for d in designs.values() for c in d.crrna_designs]
    if not all_d:
        print("  WARN: no designs for architecture figure"); return

    n = len(all_d)
    fig = plt.figure(figsize=(20, max(6, 2.5*n + 3)))
    gs = gridspec.GridSpec(n + 1, 1, height_ratios=[2.5]*n + [3], hspace=0.5)

    row = 0
    for gn, design in designs.items():
        for cr in design.crrna_designs:
            ax = fig.add_subplot(gs[row]); row += 1
            gene_c = COL.get(cr.target_snp.gene, "#333")
            sp = cr.spacer_length
            ax.set_xlim(-0.5, 4 + 0.5 + sp*0.8 + 14)
            ax.set_ylim(-0.5, 2.5)

            pam_label = "TTTV" if cr.pam_type == "strict" else "TTYN"
            ax.text(-0.3, 2.2,
                    f"{cr.target_snp.gene} {cr.target_snp.mutation_ecoli} "
                    f"({cr.target_snp.drug}) [{cr.design_type}|{pam_label}]",
                    fontsize=11, fontweight='bold', color=gene_c, va='center')

            # PAM
            ax.add_patch(FancyBboxPatch((0, 0.8), 4, 0.9,
                boxstyle="round,pad=0.1", facecolor=COL["pam"],
                edgecolor="black", linewidth=1.5, alpha=0.9))
            ax.text(2, 1.25, f"PAM\n{cr.pam_seq}", ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

            # Spacer
            x0 = 4.5
            for j, base in enumerate(cr.spacer_designed):
                pos = j + 1; x = x0 + j*0.8
                if pos == cr.snp_pos_in_spacer:
                    fc, ec, lw = COL["snp"], 'black', 2
                elif pos == cr.synthetic_mm_pos:
                    fc, ec, lw = COL["sm"], 'black', 2
                elif pos <= 8:
                    fc, ec, lw = COL["seed"], '#999', 0.5
                else:
                    fc, ec, lw = COL["nonseed"], '#999', 0.5
                ax.add_patch(plt.Rectangle((x, 0.8), 0.7, 0.9,
                    facecolor=fc, edgecolor=ec, linewidth=lw))
                fw = 'bold' if pos in (cr.snp_pos_in_spacer,
                                        cr.synthetic_mm_pos) else 'normal'
                ax.text(x+0.35, 1.25, base, ha='center', va='center',
                        fontsize=7, fontweight=fw)
                if pos <= 10 or pos == sp or pos == cr.snp_pos_in_spacer:
                    ax.text(x+0.35, 0.55, str(pos), ha='center', va='center',
                            fontsize=6, color='#666')

            # Seed bracket
            se = x0 + min(8, sp)*0.8
            ax.annotate('', xy=(x0, 0.3), xytext=(se, 0.3),
                arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=1.5))
            ax.text((x0+se)/2, 0.05, 'SEED(1-8)', ha='center', fontsize=7,
                    color='#2ca02c', fontweight='bold')

            # Info
            ix = x0 + sp*0.8 + 1
            info = [f"{sp}nt GC={cr.gc_content:.0%} score={cr.score:.1f}",
                    f"SNP@{cr.snp_pos_in_spacer} strand:{cr.strand}"]
            if cr.synthetic_mm_pos > 0:
                info += [f"SM: {cr.synthetic_mm_type}",
                         f"MUT:{cr.n_mm_vs_mutant}MM->ON  WT:{cr.n_mm_vs_wildtype}MM->OFF"]
            else:
                info += ["No SM (detect only)",
                         f"MUT:0MM->ON  WT:1MM->reduced"]
            ax.text(ix, 1.25, '\n'.join(info), fontsize=7, va='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f9f9f9', alpha=0.8))
            ax.set_yticks([])
            for s in ['top','right','left']: ax.spines[s].set_visible(False)

    if row > 0:
        fig.axes[row-1].legend(handles=[
            mpatches.Patch(facecolor=COL["pam"], label='PAM'),
            mpatches.Patch(facecolor=COL["seed"], label='Seed(1-8)'),
            mpatches.Patch(facecolor=COL["snp"], label='SNP'),
            mpatches.Patch(facecolor=COL["sm"], label='Synth.MM'),
        ], loc='upper right', fontsize=8, framealpha=0.9)

    # Workflow
    axw = fig.add_subplot(gs[-1]); axw.set_xlim(0,10); axw.set_ylim(0,3); axw.axis('off')
    steps = [("1.Sample", "#bdbdbd", 0.5), ("2.DNA ext", "#969696", 2.0),
             ("3.RPA\n39C 20min", "#984ea3", 3.5),
             ("4.Cas12a+crRNA\n+reporter 37C", "#2171b5", 5.5),
             ("5.Electrochem\nSWV readout", "#e6550d", 7.5),
             ("6.MDR-TB\nresult", "#e41a1c", 9.2)]
    for lb, c, x in steps:
        axw.add_patch(FancyBboxPatch((x-0.6,0.5),1.3,1.8,
            boxstyle="round,pad=0.15", facecolor=c, alpha=0.2,
            edgecolor=c, linewidth=2))
        axw.text(x+0.05, 1.4, lb, ha='center', va='center',
                 fontsize=9, fontweight='bold', color=c)
    for i in range(len(steps)-1):
        axw.annotate('', xy=(steps[i+1][2]-0.6,1.4),
                     xytext=(steps[i][2]+0.7,1.4),
                     arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    axw.text(5, 0.1, "<60min | heating block + potentiostat | "
             "4 electrodes (rpoB+katG+inhA+IS6110)", ha='center',
             fontsize=10, fontstyle='italic', color='#555')
    axw.set_title("Point-of-Care MDR-TB Workflow", fontsize=13, fontweight='bold')
    plt.savefig(out/"mdrtb_assay_architecture.png", dpi=300,
                bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> mdrtb_assay_architecture.png")


def plot_discrimination_diagram(designs: Dict[str, AssayDesign], out: Path):
    plt.rcParams.update(STYLE)
    best = None
    for d in designs.values():
        for c in d.crrna_designs:
            if c.design_type == "SM" and (best is None or c.score > best.score):
                best = c
    if best is None:
        for d in designs.values():
            if d.crrna_designs:
                best = d.crrna_designs[0]; break
    if best is None:
        print("  WARN: no designs for discrimination diagram"); return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (ax, label) in enumerate(zip(axes,
            ["vs MUTANT target", "vs WILDTYPE target"])):
        ax.set_xlim(-1, 24); ax.set_ylim(-2, 6)
        ax.set_title(label, fontsize=14, fontweight='bold',
                     color='#2ca02c' if idx==0 else '#e41a1c')
        sp = best.spacer_length
        # crRNA row
        ax.text(-0.5, 5, "crRNA:", fontsize=10, fontweight='bold', va='center')
        for j, base in enumerate(best.spacer_designed):
            pos = j+1; x = j*0.85
            fc = ('#ff9999' if pos==best.snp_pos_in_spacer else
                  '#ffcc66' if pos==best.synthetic_mm_pos else
                  '#d4eac7' if pos<=8 else '#e8e8e8')
            ax.add_patch(plt.Rectangle((x,4.2),0.75,0.8,facecolor=fc,
                         edgecolor='#333',linewidth=0.8))
            ax.text(x+0.375,4.6,base,ha='center',va='center',
                    fontsize=7,fontweight='bold')

        # Target row
        ax.text(-0.5, 3, "Target:", fontsize=10, fontweight='bold', va='center')
        tgt = best.spacer_mut if idx==0 else best.spacer_wt
        n_mm = 0
        for j in range(min(sp, len(tgt))):
            x = j*0.85
            is_mm = best.spacer_designed[j] != tgt[j]
            if is_mm: n_mm += 1
            fc = '#ffcccc' if is_mm else '#e8e8e8'
            ax.add_patch(plt.Rectangle((x,2.2),0.75,0.8,facecolor=fc,
                         edgecolor='#333',linewidth=0.8))
            ax.text(x+0.375,2.6,tgt[j],ha='center',va='center',
                    fontsize=7,fontweight='bold')
            if is_mm:
                ax.text(x+0.375,3.6,'X',ha='center',va='center',
                        fontsize=14,color='red',fontweight='bold')
            else:
                ax.plot([x+0.375]*2,[3.0,4.2],color='#ccc',lw=0.5,ls=':')

        # Result
        if idx==0:
            rc='#2ca02c'; rt=f"{n_mm}MM -> TOLERATED\nCas12a ON -> SIGNAL"
            oc = "DETECTED"
        else:
            rc='#e41a1c'; rt=f"{n_mm}MM -> NOT TOLERATED\nCas12a OFF -> NO SIGNAL"
            oc = "NO SIGNAL"
        ax.add_patch(FancyBboxPatch((2,-1.5),14,1.3,boxstyle="round,pad=0.2",
            facecolor=rc,alpha=0.15,edgecolor=rc,linewidth=2))
        ax.text(9,-0.85,rt,ha='center',va='center',fontsize=10,
                color=rc,fontweight='bold')
        ax.text(9,1.5,oc,ha='center',va='center',fontsize=16,
                color=rc,fontweight='bold')
        ax.set_yticks([]); ax.set_xticks([])
        for s in ax.spines.values(): s.set_visible(False)

    snp = best.target_snp
    sm_l = f"SM@{best.synthetic_mm_pos}" if best.synthetic_mm_pos>0 else "no SM"
    pam_l = "TTTV" if best.pam_type=="strict" else "TTYN"
    fig.suptitle(f"Discrimination: {snp.gene} {snp.mutation_ecoli} ({snp.drug})\n"
                 f"{best.design_type} | {best.spacer_length}nt | "
                 f"SNP@{best.snp_pos_in_spacer} | {sm_l} | PAM={best.pam_seq} [{pam_l}]",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out/"mdrtb_discrimination_diagram.png", dpi=300,
                bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> mdrtb_discrimination_diagram.png")


def plot_pam_landscape(designs: Dict[str, AssayDesign], out: Path):
    plt.rcParams.update(STYLE)
    n = len(designs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    for ax, (gn, d) in zip(axes, designs.items()):
        first = True
        for sn, pams in d.all_pam_sites.items():
            strict_seed = [p.snp_position_in_spacer for p in pams
                          if p.pam_type=="strict"]
            relax_seed = [p.snp_position_in_spacer for p in pams
                         if p.pam_type=="relaxed"]
            ax.scatter(strict_seed, [sn]*len(strict_seed), c='#2171b5',
                      s=60, marker='^', label='TTTV' if first else '',
                      alpha=0.7, edgecolors='black', linewidths=0.5)
            ax.scatter(relax_seed, [sn]*len(relax_seed), c='#e6550d',
                      s=60, marker='v', label='TTYN' if first else '',
                      alpha=0.7, edgecolors='black', linewidths=0.5)
            first = False
        ax.axvspan(1, 8, alpha=0.12, color='green', label='Seed(1-8)')
        ax.axvspan(9, 12, alpha=0.06, color='blue', label='Ext.seed(9-12)')
        tot = sum(len(p) for p in d.all_pam_sites.values())
        ax.set_xlabel("SNP pos in spacer"); ax.set_xlim(0,26)
        ax.set_title(f"{gn} ({tot} PAMs)", fontweight='bold')
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out/"mdrtb_pam_landscape.png", dpi=300,
                bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> mdrtb_pam_landscape.png")


def generate_summary_table(designs: Dict[str, AssayDesign], out: Path):
    plt.rcParams.update(STYLE)
    all_d = [c for d in designs.values() for c in d.crrna_designs]
    if not all_d:
        print("  WARN: no designs for summary table"); return
    fig, ax = plt.subplots(figsize=(24, max(4, 1.2*len(all_d)+3))); ax.axis('off')
    hdr = ["Gene","Mutation","Drug","Type","PAM","PAM\nType","Strand",
           "crRNA Spacer 5'->3'","Len","SNP\nPos","SM\nPos",
           "MM\nMUT","MM\nWT","GC%","Score"]
    rows = []
    for c in all_d:
        s = c.target_snp
        rows.append([s.gene, s.mutation_ecoli, s.drug[:3], c.design_type,
                     c.pam_seq, "TTTV" if c.pam_type=="strict" else "TTYN",
                     c.strand, c.spacer_designed, str(c.spacer_length),
                     str(c.snp_pos_in_spacer),
                     str(c.synthetic_mm_pos) if c.synthetic_mm_pos>0 else '-',
                     str(c.n_mm_vs_mutant), str(c.n_mm_vs_wildtype),
                     f"{c.gc_content:.0%}", f"{c.score:.1f}"])
    t = ax.table(cellText=rows, colLabels=hdr, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1.0, 1.5)
    for j in range(len(hdr)):
        t[0,j].set_facecolor('#2171b5')
        t[0,j].set_text_props(color='white', fontweight='bold')
    gbg = {"rpoB":"#e3f2fd","katG":"#fce4ec","inhA_promoter":"#f3e5f5"}
    tbg = {"SM":"#e8f5e9","detect_only":"#fff3e0"}
    for i, r in enumerate(rows):
        for j in range(len(hdr)):
            t[i+1,j].set_facecolor(tbg.get(r[3],"#fff") if j==3 else
                                   gbg.get(r[0],"#fff"))
    nsm = sum(1 for c in all_d if c.design_type=="SM")
    ax.set_title(f"MDR-TB crRNA Summary: {len(all_d)} designs "
                 f"({nsm} SM + {len(all_d)-nsm} detect_only)",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(out/"mdrtb_crrna_summary_table.png", dpi=300,
                bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> mdrtb_crrna_summary_table.png")


def generate_rpa_table(designs: Dict[str, AssayDesign], out: Path):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(18, 6)); ax.axis('off')
    hdr = ["Amplicon","Targets","Primer","Sequence 5'->3'","Len",
           "GC%","HP","Amp","Score"]
    rows = []
    for gn, d in designs.items():
        a = d.amplicon
        if not a: continue
        rows.append([gn, ', '.join(a.targets_covered), "Fwd",
                     a.forward.sequence, str(a.forward.length),
                     f"{a.forward.gc_content:.0%}",
                     str(a.forward.max_homopolymer),
                     f"{a.amplicon_length}bp", f"{a.score:.1f}"])
        rows.append(["","","Rev", a.reverse.sequence, str(a.reverse.length),
                     f"{a.reverse.gc_content:.0%}",
                     str(a.reverse.max_homopolymer), "", ""])
    if rows:
        t = ax.table(cellText=rows, colLabels=hdr, loc='center', cellLoc='center')
        t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1.0, 1.6)
        for j in range(len(hdr)):
            t[0,j].set_facecolor('#984ea3')
            t[0,j].set_text_props(color='white', fontweight='bold')
    ax.set_title("RPA Primers (30-35nt, 120-260bp amplicons, 37-39C)",
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(out/"mdrtb_rpa_primers_table.png", dpi=300,
                bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> mdrtb_rpa_primers_table.png")


def plot_design_statistics(designs: Dict[str, AssayDesign], out: Path):
    plt.rcParams.update(STYLE)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    tgts, np_, ns_ = [], [], []
    for d in designs.values():
        for sn, pams in d.all_pam_sites.items():
            tgts.append(sn); np_.append(len(pams))
            ns_.append(sum(1 for p in pams if 1<=p.snp_position_in_spacer<=12))
    y = range(len(tgts))
    a1.barh(list(y), np_, color='#74c476', alpha=0.8, label='Total PAMs')
    a1.barh(list(y), ns_, color='#2171b5', alpha=0.8, label='SNP in ext.seed')
    a1.set_yticks(list(y)); a1.set_yticklabels(tgts, fontsize=9)
    a1.set_xlabel("PAM sites"); a1.legend(fontsize=8)
    a1.set_title("PAM Availability (TTTV+TTYN)\nMtb GC=65.6%", fontweight='bold')

    t2, nc, dt, sc = [], [], [], []
    for d in designs.values():
        for c in d.crrna_designs:
            nm = c.target_snp.mutation_ecoli
            t2.append(nm); nc.append(d.all_candidates.get(nm,0))
            dt.append(c.design_type); sc.append(c.score)
    if t2:
        bc = ['#2ca02c' if d=='SM' else '#ff7f00' for d in dt]
        y2 = range(len(t2))
        a2.barh(list(y2), nc, color=bc, alpha=0.8)
        a2.set_yticks(list(y2)); a2.set_yticklabels(t2, fontsize=9)
        a2.set_xlabel("Candidates"); a2.set_title(
            "Candidates per Target\n(green=SM, orange=detect_only)", fontweight='bold')
        for i in range(len(t2)):
            a2.text(nc[i]+0.5, i, f"s={sc[i]:.1f}", va='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(out/"mdrtb_design_statistics.png", dpi=300,
                bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> mdrtb_design_statistics.png")


# ═══════════════════════════════════════════════════════════════════
# 8. JEPA INTEGRATION + EXPORT
# ═══════════════════════════════════════════════════════════════════

def jepa_integration_summary(designs: Dict[str, AssayDesign], out: Path):
    tot = sum(len(d.crrna_designs) for d in designs.values())
    nsm = sum(1 for d in designs.values()
              for c in d.crrna_designs if c.design_type=="SM")
    tcand = sum(v for d in designs.values()
                for v in d.all_candidates.values())
    n_strict = sum(1 for d in designs.values()
                   for c in d.crrna_designs if c.pam_type=="strict")
    n_relax = tot - n_strict

    lines = [
        "="*70,
        "  DNA-JEPA -> MDR-TB Diagnostic: Integration Summary",
        "="*70, "",
        "1. PRE-TRAINING (completed)",
        "   200 epochs, 20 bacterial genomes, RankMe 380/384",
        "   Phylum classification: 99% from embeddings", "",
        "2. Cas12a FINE-TUNING (completed)",
        "   EasyDesign 10,634 LbCas12a pairs",
        "   JEPA frozen: rho=0.484+/-0.016",
        "   Random init: rho=0.254+/-0.021",
        "   Delta: +0.229 (validates pre-training)", "",
        f"3. MDR-TB ASSAY DESIGN v4",
        f"   {tot} crRNAs: {nsm} SM + {tot-nsm} detect_only",
        f"   PAM types: {n_strict} TTTV + {n_relax} TTYN",
        f"   From {tcand} candidates across {len(designs)} regions",
        "   Targets: rpoB(RIF) + katG(INH) + inhA(INH/ETH)",
        "   Challenge: Mtb GC=65.6% -> dual PAM strategy", "",
        "4. NEXT: JEPA-GUIDED RANKING",
        "   Feed candidates into fine-tuned JEPA",
        "   Predict trans-cleavage per design",
        "   Rank: predicted_activity x discrimination_score",
        "   Select top per target for multiplexed cartridge", "",
        "5. PhD ROADMAP (ETH Zurich, deMello/Richards)",
        "   Y1: Scale to 1000+ genomes, validate DeepCpf1/CRISPR-ML",
        "       Integrate enAsCas12a activity data",
        "   Y2: Optimize 6-8 target MDR-TB panel with CSEM cartridge",
        "   Y3: Clinical validation Swiss TPH + NCTLD Georgia",
        "="*70,
    ]
    text = '\n'.join(lines)
    print(text)
    (out/"jepa_mdrtb_integration.txt").write_text(text)
    print(f"\n  -> jepa_mdrtb_integration.txt")


def export_designs(designs: Dict[str, AssayDesign], out: Path):
    export = {}
    for gn, d in designs.items():
        ge = {"description": d.description,
              "verification": d.verification,
              "candidates_evaluated": d.all_candidates,
              "crrnas": [], "rpa_primers": None}
        for c in d.crrna_designs:
            s = c.target_snp
            ge["crrnas"].append({
                "name": c.name, "gene": s.gene,
                "mutation": s.mutation_ecoli, "drug": s.drug,
                "genome_pos": s.genome_pos,
                "ref_allele": s.ref_allele, "alt_allele": s.alt_allele,
                "pam_seq": c.pam_seq, "pam_strand": c.strand,
                "pam_type": c.pam_type,
                "spacer_designed": c.spacer_designed,
                "spacer_wt": c.spacer_wt, "spacer_mut": c.spacer_mut,
                "spacer_length": c.spacer_length,
                "snp_pos_in_spacer": c.snp_pos_in_spacer,
                "synthetic_mm_pos": c.synthetic_mm_pos,
                "synthetic_mm_type": c.synthetic_mm_type,
                "n_mm_vs_mutant": c.n_mm_vs_mutant,
                "n_mm_vs_wildtype": c.n_mm_vs_wildtype,
                "gc_content": round(c.gc_content, 3),
                "design_type": c.design_type,
                "score": round(c.score, 2),
            })
        if d.amplicon:
            a = d.amplicon
            ge["rpa_primers"] = {
                "amplicon_size": a.amplicon_length,
                "forward": {"sequence": a.forward.sequence,
                            "position": a.forward.position,
                            "length": a.forward.length,
                            "gc_content": round(a.forward.gc_content, 3)},
                "reverse": {"sequence": a.reverse.sequence,
                            "position": a.reverse.position,
                            "length": a.reverse.length,
                            "gc_content": round(a.reverse.gc_content, 3)},
            }
        export[gn] = ge
    (out/"mdrtb_assay_designs.json").write_text(json.dumps(export, indent=2))
    print(f"  -> mdrtb_assay_designs.json")


# ═══════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="MDR-TB Cas12a assay design v4")
    p.add_argument("--genome", required=True, help="H37Rv FASTA")
    p.add_argument("--output-dir", default="results/mdrtb_assay")
    p.add_argument("--pam-mode", choices=["strict","relaxed","auto"],
                   default="auto",
                   help="PAM strategy: strict=TTTV, relaxed=TTYN, "
                        "auto=TTTV then TTYN for gaps (default: auto)")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    gp = Path(args.genome)
    if not gp.is_absolute():
        gp = (PROJECT_ROOT / gp).resolve()

    print(f"{'='*60}")
    print(f"  MDR-TB CRISPR-Cas12a Assay Design v4")
    print(f"  DNA-Bacteria-JEPA")
    print(f"{'='*60}")
    print(f"  Genome: {gp}")
    print(f"  Output: {out}")
    print(f"  PAM mode: {args.pam_mode}")
    print(f"  Targets: {len(TARGETS)} SNPs, 3 loci")
    print(f"  Search: spacer_len=25, max_dist=150, amp=120-260bp")

    print(f"\nLoading genome...")
    genome = load_genome(gp)
    assert len(genome) > 4_000_000

    print(f"\nVerifying SNPs...")
    ok = all(verify_snp(genome, s) for s in TARGETS)
    print(f"  {'All' if ok else 'Some'} {len(TARGETS)} SNPs "
          f"{'verified' if ok else 'had issues'}")

    print(f"\nDesigning assays (PAM mode: {args.pam_mode})...")
    designs = design_full_assay(genome, args.pam_mode)

    print(f"\n{'='*60}")
    print(f"  Generating outputs...")
    print(f"{'='*60}")
    plot_assay_architecture(designs, out)
    plot_discrimination_diagram(designs, out)
    plot_pam_landscape(designs, out)
    generate_summary_table(designs, out)
    generate_rpa_table(designs, out)
    plot_design_statistics(designs, out)
    export_designs(designs, out)
    jepa_integration_summary(designs, out)

    tot = sum(len(d.crrna_designs) for d in designs.values())
    nsm = sum(1 for d in designs.values()
              for c in d.crrna_designs if c.design_type=="SM")
    gaps = sum(1 for d in designs.values()
               for nm, ps in d.all_pam_sites.items() if len(ps)==0)
    tcand = sum(v for d in designs.values()
                for v in d.all_candidates.values())

    print(f"\n{'='*60}")
    print(f"  DESIGN COMPLETE")
    print(f"{'='*60}")
    print(f"  crRNAs: {tot} ({nsm} SM + {tot-nsm} detect_only)")
    print(f"  Candidates evaluated: {tcand}")
    print(f"  Targets with no PAM: {gaps}")
    print(f"  PAM mode: {args.pam_mode}")
    print(f"  Outputs: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
