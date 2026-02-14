#!/usr/bin/env python3
"""
12_interview_figures.py — 4 publication-quality figures for ETH PhD interview
═══════════════════════════════════════════════════════════════════════════════

  Fig 1: t-SNE of crRNA embeddings colored by MDR-TB locus
  Fig 2: UMAP of crRNA embeddings colored by MDR-TB locus
  Fig 3: Within-locus vs between-locus cosine distance boxplot
  Fig 4: Cosine distance vs genomic distance scatter

Usage:
    PYTHONPATH=/workspace/DNA-Bacteria-JEPA python3 scripts/12_interview_figures.py \
        --genome data/H37Rv.fna.gz \
        --checkpoint checkpoints/checkpoint_epoch200.pt \
        --output-dir results/interview_figures
"""

import sys, os, argparse, gzip, re, warnings, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine as cosine_dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# GENOME + DESIGN UTILITIES (minimal, from v5/v11)
# ═══════════════════════════════════════════════════════════════════════════

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def load_genome(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    parts = []
    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    return "".join(parts)

@dataclass
class ResistanceSNP:
    gene: str
    mutation: str
    genome_pos: int
    ref: str
    alt: str
    drug: str
    freq: str = ""

MDR_TB_TARGETS = {
    "rpoB_RRDR": {
        "center": 761155,
        "snps": [
            ResistanceSNP("rpoB", "S531L", 761155, "C", "T", "RIF", "~70%"),
            ResistanceSNP("rpoB", "H526Y", 761139, "C", "T", "RIF", "~10%"),
            ResistanceSNP("rpoB", "H526D", 761139, "C", "G", "RIF", "~7%"),
            ResistanceSNP("rpoB", "D516V", 761110, "A", "T", "RIF", "~5%"),
            ResistanceSNP("rpoB", "L533P", 761161, "T", "C", "RIF", "~2%"),
        ],
    },
    "katG_315": {
        "center": 2155168,
        "snps": [
            ResistanceSNP("katG", "S315T", 2155168, "C", "G", "INH", "~90%"),
            ResistanceSNP("katG", "S315N", 2155168, "C", "T", "INH", "~2%"),
        ],
    },
    "inhA_promoter": {
        "center": 1673425,
        "snps": [
            ResistanceSNP("inhA", "c.-15C>T", 1673425, "C", "T", "INH/ETH", "~20%"),
            ResistanceSNP("inhA", "c.-8T>C", 1673432, "T", "C", "INH/ETH", "~5%"),
        ],
    },
}

PAM_PATTERNS = {
    "strict":  {"plus": r"TTT[ACG]", "minus": r"[CGT]AAA"},
    "relaxed": {"plus": r"TT[ACGT][ACGT]", "minus": r"[ACGT][ACGT]AA"},
}

@dataclass
class CrRNA:
    name: str
    gene: str
    snp_mutation: str
    genome_pos: int
    context_seq: str
    spacer: str
    strand: str
    pam_seq: str
    spacer_len: int


def find_crrna_candidates(genome, targets, spacer_lengths=None, max_distance=150, context_window=500):
    """Find all crRNA candidates near each SNP, with 500bp context windows."""
    if spacer_lengths is None:
        spacer_lengths = [17, 18, 20, 23, 25]

    candidates = []
    for group_name, group in targets.items():
        for snp in group["snps"]:
            snp_pos = snp.genome_pos
            # 500bp window centered on SNP
            half = context_window // 2
            ctx_start = max(0, snp_pos - 1 - half)
            ctx_end = min(len(genome), snp_pos - 1 + half)
            context_500bp = genome[ctx_start:ctx_end]

            for pam_mode in ["strict", "relaxed"]:
                pat = PAM_PATTERNS[pam_mode]
                search_start = max(0, snp_pos - 1 - max_distance - 30)
                search_end = min(len(genome), snp_pos - 1 + max_distance + 30)
                region = genome[search_start:search_end]

                for m in re.finditer(pat["plus"], region, re.IGNORECASE):
                    pam_gpos = search_start + m.start() + 1
                    pam_seq = region[m.start():m.start() + 4]
                    for sp_len in spacer_lengths:
                        sp_start = pam_gpos + 3
                        sp_end = sp_start + sp_len
                        if sp_end > len(genome): continue
                        spacer = genome[sp_start:sp_end]
                        candidates.append(CrRNA(
                            name=f"{snp.gene}_{snp.mutation}_sp{sp_len}_{pam_mode}_{pam_gpos}",
                            gene=snp.gene, snp_mutation=snp.mutation,
                            genome_pos=snp.genome_pos,
                            context_seq=context_500bp,  # 500bp window
                            spacer=spacer,
                            strand="+", pam_seq=pam_seq, spacer_len=sp_len,
                        ))

                for m in re.finditer(pat["minus"], region, re.IGNORECASE):
                    vaaa_gpos = search_start + m.start() + 1
                    for sp_len in spacer_lengths:
                        sp_end_0 = vaaa_gpos - 2
                        sp_start_0 = sp_end_0 - sp_len + 1
                        if sp_start_0 < 0: continue
                        pam_seq_rc = reverse_complement(region[m.start():m.start() + 4])
                        spacer = reverse_complement(genome[sp_start_0:sp_end_0 + 1])
                        candidates.append(CrRNA(
                            name=f"{snp.gene}_{snp.mutation}_sp{sp_len}_{pam_mode}_{vaaa_gpos}_m",
                            gene=snp.gene, snp_mutation=snp.mutation,
                            genome_pos=snp.genome_pos,
                            context_seq=context_500bp,  # 500bp window
                            spacer=spacer,
                            strand="-", pam_seq=pam_seq_rc, spacer_len=sp_len,
                        ))
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# JEPA EMBEDDER
# ═══════════════════════════════════════════════════════════════════════════

class JEPAEmbedder:
    def __init__(self, checkpoint_path, device="auto"):
        import torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        from src.cas12a.encoder import SparseTransformerEncoder
        from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        self.tokenizer = Cas12aTokenizer(TokenizerConfig())
        self.encoder = SparseTransformerEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=args.get("embed_dim", 384),
            num_layers=args.get("num_layers", 6),
            num_heads=args.get("num_heads", 6),
            ff_dim=args.get("ff_dim", 1024),
        )
        if "target_encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["target_encoder_state_dict"])
        elif "encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.encoder.to(self.device).eval()
        log.info("JEPA loaded: %dD × %dL × %dH, epoch %s",
                 args.get("embed_dim", 384), args.get("num_layers", 6),
                 args.get("num_heads", 6), ckpt.get("epoch", "?"))

    def embed(self, sequences, batch_size=128):
        import torch
        all_emb = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            tokens = []
            for seq in batch:
                t = self.tokenizer.encode_generic_sequence(seq.upper())
                if isinstance(t, torch.Tensor):
                    t = t.squeeze()
                    if t.dim() == 0: t = t.unsqueeze(0)
                tokens.append(t)
            max_len = max(len(t) for t in tokens)
            padded = []
            for t in tokens:
                if len(t) < max_len:
                    t = torch.cat([t, torch.full((max_len - len(t),), self.tokenizer.pad_id, dtype=torch.long)])
                padded.append(t)
            tok_batch = torch.stack(padded).to(self.device)
            mask = self.tokenizer.get_attention_mask(tok_batch)
            with torch.no_grad():
                pooled, _ = self.encoder(tok_batch, attention_mask=mask)
            all_emb.append(pooled.cpu().numpy())
        return np.concatenate(all_emb, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

LOCUS_COLORS = {"rpoB": "#2171b5", "katG": "#d94801", "inhA": "#6a3d9a"}
LOCUS_LABELS = {"rpoB": "rpoB (Rifampicin)", "katG": "katG (Isoniazid)", "inhA": "inhA (INH/Ethionamide)"}

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
    })


def fig1_tsne(embeddings, genes, out):
    """t-SNE of crRNA embeddings colored by MDR-TB resistance locus."""
    from sklearn.manifold import TSNE

    N = len(embeddings)
    perp = min(30, max(5, N // 5))
    proj = TSNE(n_components=2, perplexity=perp, random_state=42,
                learning_rate="auto", init="pca").fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))

    for locus in ["rpoB", "katG", "inhA"]:
        mask = np.array([g == locus for g in genes])
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=LOCUS_COLORS[locus], s=40, alpha=0.7,
                   edgecolors="white", linewidth=0.5,
                   label=LOCUS_LABELS[locus], zorder=3)
        # Centroid label
        cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
        ax.annotate(locus, (cx, cy), fontsize=13, fontweight="bold",
                    color=LOCUS_COLORS[locus], ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=LOCUS_COLORS[locus], alpha=0.9),
                    zorder=5)

    ax.set_xlabel("t-SNE 1", fontsize=13)
    ax.set_ylabel("t-SNE 2", fontsize=13)
    ax.set_title("DNA-JEPA Embeddings of MDR-TB crRNA Candidates\n"
                 "(pretrained encoder, no CRISPR-specific training)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    # Remove tick labels (t-SNE axes are arbitrary)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out / "fig1_tsne_locus.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → fig1_tsne_locus.png")


def fig2_umap(embeddings, genes, out):
    """UMAP of crRNA embeddings colored by MDR-TB resistance locus."""
    from umap import UMAP

    proj = UMAP(n_neighbors=15, min_dist=0.3, metric="cosine",
                random_state=42).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))

    for locus in ["rpoB", "katG", "inhA"]:
        mask = np.array([g == locus for g in genes])
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=LOCUS_COLORS[locus], s=40, alpha=0.7,
                   edgecolors="white", linewidth=0.5,
                   label=LOCUS_LABELS[locus], zorder=3)
        cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
        ax.annotate(locus, (cx, cy), fontsize=13, fontweight="bold",
                    color=LOCUS_COLORS[locus], ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=LOCUS_COLORS[locus], alpha=0.9),
                    zorder=5)

    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.set_title("UMAP of DNA-JEPA crRNA Embeddings\n"
                 "(cosine metric, pretrained encoder)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out / "fig2_umap_locus.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → fig2_umap_locus.png")


def fig3_within_between_boxplot(embeddings, genes, out):
    """Within-locus vs between-locus cosine distance boxplot."""
    from scipy.spatial.distance import cosine as cos_dist

    N = len(embeddings)

    # Exhaustive within-locus
    within_dists = []
    unique_loci = sorted(set(genes))
    within_by_locus = {l: [] for l in unique_loci}
    for locus in unique_loci:
        idx = [i for i, g in enumerate(genes) if g == locus]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d = cos_dist(embeddings[idx[i]], embeddings[idx[j]])
                within_dists.append(d)
                within_by_locus[locus].append(d)

    # Exhaustive between-locus (subsample if huge)
    between_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if genes[i] != genes[j]]
    np.random.seed(42)
    if len(between_pairs) > 5000:
        between_pairs = [between_pairs[k] for k in np.random.choice(len(between_pairs), 5000, replace=False)]
    between_dists = [cos_dist(embeddings[i], embeddings[j]) for i, j in between_pairs]

    if not within_dists or not between_dists:
        log.warning("  Not enough pairs for boxplot")
        return 0, 0, 0

    fig, ax = plt.subplots(figsize=(7, 7))

    bp = ax.boxplot([within_dists, between_dists],
                    positions=[0, 1], widths=0.5, showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color="red", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    bp["boxes"][0].set_facecolor("#a1d99b")
    bp["boxes"][0].set_edgecolor("#31a354")
    bp["boxes"][1].set_facecolor("#fdae6b")
    bp["boxes"][1].set_edgecolor("#e6550d")

    # Jittered points
    for data, color, xpos in [
        (within_dists, "#31a354", 0),
        (between_dists, "#e6550d", 1),
    ]:
        n_pts = len(data)
        alpha = max(0.08, min(0.5, 200 / max(n_pts, 1)))
        jitter = np.random.normal(0, 0.06, n_pts)
        ax.scatter(np.full(n_pts, xpos) + jitter, data,
                   c=color, s=10, alpha=alpha, edgecolors="none", zorder=2)

    w_med = np.median(within_dists)
    b_med = np.median(between_dists)
    ratio = b_med / max(w_med, 1e-8)

    ax.text(0, max(within_dists) * 1.02, f"median = {w_med:.4f}\nn = {len(within_dists)}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#31a354")
    ax.text(1, max(between_dists) * 1.02, f"median = {b_med:.4f}\nn = {len(between_dists)}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#e6550d")

    # Statistical test
    from scipy.stats import mannwhitneyu
    try:
        stat, pval = mannwhitneyu(within_dists, between_dists, alternative="less")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        ax.text(0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.35,
                f"Between/Within: {ratio:.2f}×  |  Mann-Whitney p = {pval:.2e} {sig}",
                ha="center", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                          edgecolor="#ffc107", alpha=0.9))
    except Exception:
        ax.text(0.5, 0.35, f"Between/Within ratio: {ratio:.2f}×",
                ha="center", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                          edgecolor="#ffc107", alpha=0.9))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-locus\n(same gene)", "Between-locus\n(different genes)"],
                       fontsize=12, fontweight="bold")
    ax.set_ylabel("Cosine Distance", fontsize=13)
    ax.set_title("Embedding Separation by Resistance Locus\n"
                 "(pretrained DNA-JEPA, no CRISPR training)",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "fig3_within_between_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → fig3_within_between_distance.png")

    return w_med, b_med, ratio


def fig4_cosine_vs_genomic(embeddings, candidates, out):
    """Cosine distance vs genomic distance scatter."""
    N = len(candidates)

    # Compute pairwise cosine and genomic distances
    # Subsample for performance
    np.random.seed(42)
    max_pairs = 5000
    if N * (N - 1) // 2 > max_pairs:
        pairs = []
        while len(pairs) < max_pairs:
            i, j = np.random.randint(0, N, 2)
            if i != j:
                pairs.append((min(i, j), max(i, j)))
        pairs = list(set(pairs))[:max_pairs]
    else:
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    cos_dists = []
    gen_dists = []
    pair_genes = []  # same or different locus

    for i, j in pairs:
        cd = cosine_dist(embeddings[i], embeddings[j])
        gd = abs(candidates[i].genome_pos - candidates[j].genome_pos)
        cos_dists.append(cd)
        gen_dists.append(gd)
        if candidates[i].gene == candidates[j].gene:
            pair_genes.append("same")
        else:
            pair_genes.append("diff")

    cos_dists = np.array(cos_dists)
    gen_dists = np.array(gen_dists)
    pair_genes = np.array(pair_genes)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot same-locus and different-locus separately
    same_mask = pair_genes == "same"
    diff_mask = pair_genes == "diff"

    ax.scatter(gen_dists[same_mask] / 1000, cos_dists[same_mask],
               c="#31a354", s=15, alpha=0.4, edgecolors="none",
               label="Same locus", zorder=2)
    ax.scatter(gen_dists[diff_mask] / 1000, cos_dists[diff_mask],
               c="#e6550d", s=15, alpha=0.3, edgecolors="none",
               label="Different loci", zorder=2)

    # Add trend line for same-locus
    if same_mask.sum() > 10:
        z = np.polyfit(gen_dists[same_mask] / 1000, cos_dists[same_mask], 1)
        xl = np.linspace(0, gen_dists[same_mask].max() / 1000, 100)
        ax.plot(xl, np.polyval(z, xl), "--", color="#31a354", linewidth=2, alpha=0.8)

    # Correlation
    from scipy.stats import spearmanr
    rho_same, p_same = spearmanr(gen_dists[same_mask], cos_dists[same_mask])
    ax.text(0.05, 0.95,
            f"Within-locus: ρ = {rho_same:.3f} (p = {p_same:.1e})",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="#31a354", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax.set_xlabel("Genomic Distance (kb)", fontsize=13)
    ax.set_ylabel("Cosine Distance in Embedding Space", fontsize=13)
    ax.set_title("Embedding Distance Reflects Genomic Proximity\n"
                 "(pretrained DNA-JEPA captures spatial genome structure)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out / "fig4_cosine_vs_genomic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  → fig4_cosine_vs_genomic.png")

    return rho_same


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def deduplicate_by_context(candidates):
    """Deduplicate candidates sharing identical context sequences, keep one per unique context per gene."""
    seen = {}
    deduped = []
    for c in candidates:
        key = (c.gene, c.context_seq[:50])  # first 50 chars enough to distinguish 500bp windows
        if key not in seen:
            seen[key] = c
            deduped.append(c)
    return deduped


def best_per_pam_per_snp(candidates):
    """Keep one candidate per unique PAM site per SNP — removes spacer-length redundancy."""
    seen = {}
    result = []
    for c in candidates:
        # With 500bp context, group by SNP + strand + PAM position
        key = (c.gene, c.snp_mutation, c.strand, c.pam_seq)
        if key not in seen:
            seen[key] = c
            result.append(c)
    return result


def best_per_snp(candidates):
    """Keep single best candidate per SNP target (the actual assay panel)."""
    best = {}
    for c in candidates:
        key = (c.gene, c.snp_mutation)
        if key not in best or len(c.spacer) > len(best[key].spacer):
            best[key] = c
    return list(best.values())


def run_figures_for_set(name, candidates_subset, embedder, out_subdir):
    """Generate all 4 figures for a given candidate set."""
    out_subdir.mkdir(parents=True, exist_ok=True)

    genes = [c.gene for c in candidates_subset]
    sequences = [c.context_seq for c in candidates_subset]
    embeddings = embedder.embed(sequences)

    n_rpoB = sum(1 for g in genes if g == "rpoB")
    n_katG = sum(1 for g in genes if g == "katG")
    n_inhA = sum(1 for g in genes if g == "inhA")
    log.info("  [%s] %d candidates: rpoB=%d, katG=%d, inhA=%d",
             name, len(candidates_subset), n_rpoB, n_katG, n_inhA)

    # Only generate dimensionality reduction if enough points
    if len(candidates_subset) >= 10:
        fig1_tsne(embeddings, genes, out_subdir)
        fig2_umap(embeddings, genes, out_subdir)
    elif len(candidates_subset) >= 3:
        fig1_tsne(embeddings, genes, out_subdir)
        log.info("  Skipping UMAP (too few points for meaningful UMAP)")

    if len(candidates_subset) >= 6:
        w_med, b_med, ratio = fig3_within_between_boxplot(embeddings, genes, out_subdir)
        log.info("    Within: %.4f, Between: %.4f, Ratio: %.1f×", w_med, b_med, ratio)
    else:
        log.info("  Skipping boxplot (too few points)")
        w_med, b_med, ratio = 0, 0, 0

    if len(candidates_subset) >= 10:
        rho = fig4_cosine_vs_genomic(embeddings, candidates_subset, out_subdir)
        log.info("    Genomic-embedding ρ: %.3f", rho)
    else:
        rho = 0

    return {"n": len(candidates_subset), "w_med": w_med, "b_med": b_med,
            "ratio": ratio, "rho": rho}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/interview_figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Load genome
    log.info("Loading genome: %s", args.genome)
    genome = load_genome(args.genome)
    log.info("  %s bp, GC=%.1f%%", f"{len(genome):,}",
             100 * sum(1 for b in genome if b in "GC") / len(genome))

    # Generate ALL candidates
    log.info("Generating crRNA candidates...")
    all_candidates = find_crrna_candidates(genome, MDR_TB_TARGETS)
    log.info("  Total raw: %d", len(all_candidates))

    # Load embedder once
    log.info("Loading JEPA...")
    embedder = JEPAEmbedder(args.checkpoint)

    # === Three filtering levels ===
    levels = {
        "A_best_per_snp": best_per_snp(all_candidates),
        "B_best_per_pam": best_per_pam_per_snp(all_candidates),
        "C_unique_context": deduplicate_by_context(all_candidates),
    }

    results = {}
    for level_name, subset in levels.items():
        log.info("\n" + "=" * 60)
        log.info("  Level: %s", level_name)
        log.info("=" * 60)
        r = run_figures_for_set(level_name, subset, embedder, out / level_name)
        results[level_name] = r

    # Summary
    log.info("\n" + "=" * 60)
    log.info("  COMPARISON ACROSS FILTERING LEVELS")
    log.info("=" * 60)
    log.info("  %-25s %6s %8s %8s %6s %8s",
             "Level", "N", "Within", "Between", "Ratio", "Geo ρ")
    for name, r in results.items():
        log.info("  %-25s %6d %8.4f %8.4f %5.1f× %8.3f",
                 name, r["n"], r["w_med"], r["b_med"], r["ratio"], r["rho"])
    log.info("")
    for f in sorted(out.rglob("fig*.png")):
        log.info("  %s (%.0f KB)", f.relative_to(out), f.stat().st_size / 1024)


if __name__ == "__main__":
    main()
