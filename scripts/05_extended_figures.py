"""Extended publication figures for DNA-Bacteria-JEPA."""
from __future__ import annotations
import argparse, sys, time, copy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cas12a.dataset import BacterialGenomeDataset
from src.cas12a.encoder import SparseTransformerEncoder
from src.cas12a.jepa_pretrain import Cas12aJEPA, JEPAConfig
from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig

NC_TO_SPECIES = {
    "NC_000913.3": "Escherichia coli K-12",
    "NC_000964.3": "Bacillus subtilis",
    "NC_002947.4": "Pseudomonas putida",
    "NC_003888.3": "Streptomyces coelicolor",
    "NC_006814.1": "Lactobacillus acidophilus",
    "NC_007795.1": "Staphylococcus aureus",
    "NC_008253.1": "Escherichia coli 536",
    "NC_009085.1": "Acinetobacter baumannii",
    "NC_000962.3": "Mycobacterium tuberculosis",
    "NC_002163.1": "Campylobacter jejuni",
    "NC_000915.1": "Helicobacter pylori",
    "NC_003210.1": "Listeria monocytogenes",
    "NC_001263.1": "Deinococcus radiodurans",
    "NC_000908.2": "Mycoplasma genitalium",
    "NC_003030.1": "Clostridium acetobutylicum",
    "NC_004668.1": "Enterococcus faecalis",
    "NC_002505.1": "Vibrio cholerae",
    "NC_001318.1": "Borrelia burgdorferi",
    "NC_005835.1": "Thermus thermophilus",
    "NC_007643.1": "Rhodopseudomonas palustris",
    "NC_007493.2": "Rhodobacter sphaeroides",
}
for k, v in list(NC_TO_SPECIES.items()):
    base = k.rsplit(".", 1)[0]
    if base not in NC_TO_SPECIES:
        NC_TO_SPECIES[base] = v

SPECIES_TO_PHYLUM = {
    "Escherichia coli K-12": "Proteobacteria",
    "Escherichia coli 536": "Proteobacteria",
    "Pseudomonas putida": "Proteobacteria",
    "Acinetobacter baumannii": "Proteobacteria",
    "Vibrio cholerae": "Proteobacteria",
    "Rhodobacter sphaeroides": "Proteobacteria",
    "Rhodopseudomonas palustris": "Proteobacteria",
    "Campylobacter jejuni": "Proteobacteria",
    "Helicobacter pylori": "Proteobacteria",
    "Bacillus subtilis": "Firmicutes",
    "Staphylococcus aureus": "Firmicutes",
    "Lactobacillus acidophilus": "Firmicutes",
    "Listeria monocytogenes": "Firmicutes",
    "Enterococcus faecalis": "Firmicutes",
    "Clostridium acetobutylicum": "Firmicutes",
    "Mycobacterium tuberculosis": "Actinobacteria",
    "Streptomyces coelicolor": "Actinobacteria",
    "Deinococcus radiodurans": "Deinococcota",
    "Thermus thermophilus": "Deinococcota",
    "Mycoplasma genitalium": "Tenericutes",
    "Borrelia burgdorferi": "Spirochaetes",
}

SPECIES_GC = {
    "Escherichia coli K-12": 50.8,
    "Escherichia coli 536": 50.5,
    "Pseudomonas putida": 61.5,
    "Acinetobacter baumannii": 39.0,
    "Vibrio cholerae": 47.5,
    "Rhodobacter sphaeroides": 68.8,
    "Rhodopseudomonas palustris": 65.0,
    "Campylobacter jejuni": 30.5,
    "Helicobacter pylori": 38.9,
    "Bacillus subtilis": 43.5,
    "Staphylococcus aureus": 32.9,
    "Lactobacillus acidophilus": 34.7,
    "Listeria monocytogenes": 38.0,
    "Enterococcus faecalis": 37.4,
    "Clostridium acetobutylicum": 30.9,
    "Mycobacterium tuberculosis": 65.6,
    "Streptomyces coelicolor": 72.1,
    "Deinococcus radiodurans": 67.0,
    "Thermus thermophilus": 69.4,
    "Mycoplasma genitalium": 31.7,
    "Borrelia burgdorferi": 28.6,
}

PHYLUM_COLORS = {
    "Proteobacteria": "#2171b5",
    "Firmicutes": "#cb181d",
    "Actinobacteria": "#7b4fbe",
    "Deinococcota": "#b5a400",
    "Tenericutes": "#636363",
    "Spirochaetes": "#8c564b",
}

PHYLUM_ORDER = ["Proteobacteria", "Firmicutes", "Actinobacteria",
                "Deinococcota", "Tenericutes", "Spirochaetes"]

SPECIES_COLORS = {
    "Escherichia coli K-12": "#2171b5",
    "Escherichia coli 536": "#4a9fd4",
    "Pseudomonas putida": "#08519c",
    "Acinetobacter baumannii": "#6baed6",
    "Vibrio cholerae": "#084594",
    "Rhodobacter sphaeroides": "#17becf",
    "Rhodopseudomonas palustris": "#0d9faf",
    "Campylobacter jejuni": "#238b45",
    "Helicobacter pylori": "#74c476",
    "Bacillus subtilis": "#cb181d",
    "Staphylococcus aureus": "#e6550d",
    "Lactobacillus acidophilus": "#fd8d3c",
    "Listeria monocytogenes": "#de2d73",
    "Enterococcus faecalis": "#f768a1",
    "Clostridium acetobutylicum": "#fc4e2a",
    "Mycobacterium tuberculosis": "#7b4fbe",
    "Streptomyces coelicolor": "#54278f",
    "Deinococcus radiodurans": "#b5a400",
    "Thermus thermophilus": "#ffd700",
    "Mycoplasma genitalium": "#636363",
    "Borrelia burgdorferi": "#8c564b",
}

def resolve_name(raw):
    if raw in SPECIES_TO_PHYLUM: return raw
    clean = raw.strip()
    if clean in NC_TO_SPECIES: return NC_TO_SPECIES[clean]
    base = clean.rsplit(".", 1)[0] if "." in clean else clean
    if base in NC_TO_SPECIES: return NC_TO_SPECIES[base]
    for key, val in NC_TO_SPECIES.items():
        if key.startswith(base) or base.startswith(key.split(".")[0]): return val
    return clean.replace("_", " ")

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

@torch.no_grad()
def extract_embeddings(model, dl, tokenizer, device, max_samples=10000):
    model.eval()
    all_emb, all_ids = [], []
    n = 0
    for batch in dl:
        if n >= max_samples: break
        tokens, gids = batch[0], batch[1]
        tokens = tokens.to(device, non_blocking=True)
        attn = tokenizer.get_attention_mask(tokens)
        emb = model.encode(tokens, attention_mask=attn, use_target=True)
        all_emb.append(emb.cpu().numpy())
        all_ids.append(gids.numpy())
        n += tokens.shape[0]
    return np.concatenate(all_emb)[:max_samples], np.concatenate(all_ids)[:max_samples]

def mc_uncertainty(model, dl, tokenizer, device, max_samples=4000, n_passes=10, mask_frac=0.15):
    """Monte Carlo masking uncertainty: multiple forward passes with random token masking."""
    model.eval()
    all_var, all_mean, all_ids = [], [], []
    n = 0
    for batch in dl:
        if n >= max_samples: break
        tokens, gids = batch[0], batch[1]
        tokens = tokens.to(device, non_blocking=True)
        attn = tokenizer.get_attention_mask(tokens)
        B, L = tokens.shape
        pass_embs = []
        for _ in range(n_passes):
            masked = tokens.clone()
            mask = torch.rand(B, L, device=device) < mask_frac
            mask[:, 0] = False  # keep CLS/start
            masked[mask] = 0  # mask token = 0
            with torch.no_grad():
                emb = model.encode(masked, attention_mask=attn, use_target=True)
            pass_embs.append(emb.cpu().numpy())
        stacked = np.stack(pass_embs, axis=0)  # (n_passes, B, D)
        variance = np.mean(np.var(stacked, axis=0), axis=1)  # (B,) mean var across dims
        mean_emb = np.mean(stacked, axis=0)  # (B, D)
        all_var.append(variance)
        all_mean.append(mean_emb)
        all_ids.append(gids.numpy())
        n += B
    return (np.concatenate(all_mean)[:max_samples],
            np.concatenate(all_var)[:max_samples],
            np.concatenate(all_ids)[:max_samples])

def load_model(ckpt_path, tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt.get("args", {})
    encoder = SparseTransformerEncoder(vocab_size=tokenizer.vocab_size,
        embed_dim=sa.get("embed_dim",384), num_layers=sa.get("num_layers",6),
        num_heads=sa.get("num_heads",6), ff_dim=sa.get("ff_dim",1024))
    cfg = JEPAConfig(predictor_dim=sa.get("predictor_dim",384),
        predictor_depth=sa.get("predictor_depth",4),
        predictor_num_heads=sa.get("predictor_num_heads",6),
        max_seq_len=sa.get("max_seq_len",1024))
    model = Cas12aJEPA(encoder, config=cfg)
    model.context_encoder.load_state_dict(ckpt["encoder_state_dict"])
    if "target_encoder_state_dict" in ckpt:
        model.target_encoder.load_state_dict(ckpt["target_encoder_state_dict"])
    else:
        model.target_encoder = copy.deepcopy(model.context_encoder)
    if "predictor_state_dict" in ckpt:
        try: model.predictor.load_state_dict(ckpt["predictor_state_dict"])
        except RuntimeError: pass
    model.to(device).eval()
    ep = ckpt.get("epoch",-1)+1
    print(f"  Loaded epoch {ep}")
    return model, ep

# ═══════════════════════════════════════════════════════════════
# Figure 1: t-SNE / UMAP colored by continuous GC%
# ═══════════════════════════════════════════════════════════════

def plot_gc_continuous(coords, ids, resolved, method, path):
    gc_vals = np.array([SPECIES_GC.get(resolved[gid], 50.0) for gid in ids])
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:,0], coords[:,1], c=gc_vals, cmap="viridis",
                    s=10, alpha=0.6, edgecolors="white", linewidths=0.1, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("GC Content (%)", fontsize=11)
    ax.set_title(f"DNA-Bacteria-JEPA - GC Content Gradient ({method})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ═══════════════════════════════════════════════════════════════
# Figure 2: t-SNE / UMAP colored by phylum (6 clean colors)
# ═══════════════════════════════════════════════════════════════

def plot_phylum(coords, ids, resolved, method, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for phy in PHYLUM_ORDER:
        mask = np.array([SPECIES_TO_PHYLUM.get(resolved[gid], "Other") == phy for gid in ids])
        if not mask.any(): continue
        color = PHYLUM_COLORS.get(phy, "#999999")
        count = mask.sum()
        ax.scatter(coords[mask,0], coords[mask,1], c=color, s=10, alpha=0.55,
                   label=f"{phy} (n={count})", edgecolors="white", linewidths=0.15, rasterized=True)
    ax.set_title(f"DNA-Bacteria-JEPA - Phylum Clustering ({method})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9,
              markerscale=2, framealpha=0.95, edgecolor="#ccc")
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ═══════════════════════════════════════════════════════════════
# Figure 3: Species x Species cosine similarity heatmap
# ═══════════════════════════════════════════════════════════════

def plot_similarity_heatmap(emb, ids, resolved, path):
    species_names = sorted(set(resolved[gid] for gid in ids),
        key=lambda s: (PHYLUM_ORDER.index(SPECIES_TO_PHYLUM.get(s,"Other"))
                       if SPECIES_TO_PHYLUM.get(s,"Other") in PHYLUM_ORDER else 99, s))
    n_sp = len(species_names)
    centroids = []
    for sp in species_names:
        mask = np.array([resolved[gid] == sp for gid in ids])
        centroid = emb[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    sim = centroids @ centroids.T

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(n_sp))
    ax.set_yticks(np.arange(n_sp))
    short = []
    for sp in species_names:
        parts = sp.split()
        short.append(f"{parts[0][0]}. {parts[1]}" if len(parts)>=2 else sp[:15])
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(short, fontsize=7.5)

    # Annotate values
    for i in range(n_sp):
        for j in range(n_sp):
            v = sim[i,j]
            c = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color=c)

    # Phylum group lines
    current_phy = None
    for i, sp in enumerate(species_names):
        phy = SPECIES_TO_PHYLUM.get(sp, "Other")
        if phy != current_phy and current_phy is not None:
            ax.axhline(i-0.5, color="black", linewidth=1.5)
            ax.axvline(i-0.5, color="black", linewidth=1.5)
        current_phy = phy

    # Phylum color patches on left
    for i, sp in enumerate(species_names):
        phy = SPECIES_TO_PHYLUM.get(sp, "Other")
        color = PHYLUM_COLORS.get(phy, "#999")
        ax.plot(-1.2, i, marker="s", markersize=6, color=color, clip_on=False, transform=ax.transData)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Cosine Similarity", fontsize=10)
    ax.set_title("Species Embedding Similarity (Epoch 200)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ═══════════════════════════════════════════════════════════════
# Figure 4: MC-masking uncertainty t-SNE
# ═══════════════════════════════════════════════════════════════

def plot_uncertainty_tsne(coords, variance, ids, resolved, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    log_var = np.log10(variance + 1e-10)
    sc = ax.scatter(coords[:,0], coords[:,1], c=log_var, cmap="magma_r",
                    s=10, alpha=0.6, edgecolors="white", linewidths=0.1, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("log10(MC Variance)", fontsize=11)
    ax.set_title("DNA-Bacteria-JEPA - Embedding Uncertainty (MC Masking)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ═══════════════════════════════════════════════════════════════
# Figure 5: Phylum-level confusion matrix
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(emb, ids, resolved, path):
    phy_labels = np.array([SPECIES_TO_PHYLUM.get(resolved[gid], "Other") for gid in ids])
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    emb_n = emb / norms
    knn = KNeighborsClassifier(n_neighbors=10, metric="euclidean", weights="distance", n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(knn, emb_n, phy_labels, cv=cv)
    labels_order = [p for p in PHYLUM_ORDER if p in set(phy_labels)]
    cm = confusion_matrix(phy_labels, preds, labels=labels_order, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    n = len(labels_order)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels_order, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = cm[i,j]
            c = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10, color=c, fontweight="bold")
    ax.set_xlabel("Predicted Phylum", fontsize=11)
    ax.set_ylabel("True Phylum", fontsize=11)
    acc = np.trace(cm * np.array([np.sum(phy_labels==l) for l in labels_order]).reshape(-1,1)) / len(phy_labels)
    ax.set_title(f"Phylum Classification Confusion Matrix (10-NN, 5-CV)", fontsize=13, fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Proportion", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ═══════════════════════════════════════════════════════════════
# Figure 6: Genome size / sample count distribution
# ═══════════════════════════════════════════════════════════════

def plot_sample_distribution(ids, resolved, path):
    species_counts = {}
    for gid in ids:
        sp = resolved[gid]
        species_counts[sp] = species_counts.get(sp, 0) + 1
    sp_sorted = sorted(species_counts.keys(),
        key=lambda s: (PHYLUM_ORDER.index(SPECIES_TO_PHYLUM.get(s,"Other"))
                       if SPECIES_TO_PHYLUM.get(s,"Other") in PHYLUM_ORDER else 99, s))
    counts = [species_counts[s] for s in sp_sorted]
    colors = [SPECIES_COLORS.get(s, "#999") for s in sp_sorted]
    short = []
    for sp in sp_sorted:
        parts = sp.split()
        short.append(f"{parts[0][0]}. {parts[1]}" if len(parts)>=2 else sp[:15])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(np.arange(len(sp_sorted)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(sp_sorted)))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Number of 500bp Fragments", fontsize=11)
    ax.set_title("Sample Distribution per Species", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2, str(c),
                va="center", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Single checkpoint (epoch 200)")
    p.add_argument("--data-path", default="data/processed/pretrain_sequences_expanded.csv")
    p.add_argument("--output-dir", default="figures/embeddings")
    p.add_argument("--max-samples", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--mc-passes", type=int, default=10)
    args = p.parse_args()

    setup_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = Cas12aTokenizer(TokenizerConfig())
    dp = Path(args.data_path)
    if not dp.is_absolute(): dp = (PROJECT_ROOT / dp).resolve()
    base_ds = BacterialGenomeDataset(str(dp), tokenizer)
    print(f"Dataset: {len(base_ds):,} sequences")

    genomes = base_ds.df["genome"].values
    ug = sorted(set(genomes))
    g2id = {g:i for i,g in enumerate(ug)}
    gids_list = [g2id[g] for g in genomes]
    resolved = [resolve_name(g) for g in ug]

    class LDS(torch.utils.data.Dataset):
        def __len__(self): return len(base_ds)
        def __getitem__(self, idx): return base_ds[idx], gids_list[idx]
    ds = LDS()
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                    pin_memory=device.type=="cuda", drop_last=False)

    od = Path(args.output_dir)
    if not od.is_absolute(): od = (PROJECT_ROOT / od).resolve()
    od.mkdir(parents=True, exist_ok=True)

    cp = Path(args.checkpoint)
    if not cp.is_absolute(): cp = (PROJECT_ROOT / cp).resolve()

    print(f"\nLoading {cp.name}...")
    model, ep = load_model(cp, tokenizer, device)

    # Standard embeddings
    print(f"Extracting embeddings...")
    t0 = time.time()
    emb, ids = extract_embeddings(model, dl, tokenizer, device, args.max_samples)
    print(f"  {emb.shape[0]:,} x {emb.shape[1]}D in {time.time()-t0:.1f}s")

    # MC uncertainty embeddings (smaller sample for speed)
    print(f"\nMC uncertainty ({args.mc_passes} passes)...")
    t0 = time.time()
    mc_emb, mc_var, mc_ids = mc_uncertainty(model, dl, tokenizer, device,
        max_samples=min(4000, args.max_samples), n_passes=args.mc_passes)
    print(f"  {mc_emb.shape[0]:,} points in {time.time()-t0:.1f}s")

    del model
    if device.type=="cuda": torch.cuda.empty_cache()

    # t-SNE for main embeddings
    print("\nRunning t-SNE (main)...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                init="pca", random_state=42, max_iter=1500)
    coords_tsne = tsne.fit_transform(emb)

    # t-SNE for MC embeddings
    print("Running t-SNE (uncertainty)...")
    tsne_mc = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                   init="pca", random_state=42, max_iter=1500)
    coords_mc = tsne_mc.fit_transform(mc_emb)

    # UMAP
    try:
        import umap
        print("Running UMAP...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            metric="cosine", random_state=42)
        coords_umap = reducer.fit_transform(emb)
        has_umap = True
    except ImportError:
        print("  UMAP not installed, skipping UMAP figures")
        has_umap = False

    # Generate all figures
    print("\n" + "="*60)
    print("Generating figures...")

    # GC continuous
    plot_gc_continuous(coords_tsne, ids, resolved, "t-SNE", od / "tsne_gc_gradient.png")
    if has_umap:
        plot_gc_continuous(coords_umap, ids, resolved, "UMAP", od / "umap_gc_gradient.png")

    # Phylum
    plot_phylum(coords_tsne, ids, resolved, "t-SNE", od / "tsne_phylum.png")
    if has_umap:
        plot_phylum(coords_umap, ids, resolved, "UMAP", od / "umap_phylum.png")

    # Similarity heatmap
    plot_similarity_heatmap(emb, ids, resolved, od / "species_similarity_heatmap.png")

    # Uncertainty
    plot_uncertainty_tsne(coords_mc, mc_var, mc_ids, resolved, od / "tsne_uncertainty_mc.png")

    # Confusion matrix
    plot_confusion_matrix(emb, ids, resolved, od / "phylum_confusion_matrix.png")

    # Sample distribution
    plot_sample_distribution(ids, resolved, od / "sample_distribution.png")

    print(f"\n{'='*60}")
    print(f"  All extended figures saved to: {od}")
    print(f"{'='*60}")

if __name__ == "__main__": main()
