"""Publication-quality embedding visualization for DNA-Bacteria-JEPA."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

SPECIES_METADATA = {
    "Escherichia coli K-12":       ("g-Proteobacteria",   50.8),
    "Escherichia coli 536":        ("g-Proteobacteria",   50.5),
    "Pseudomonas putida":          ("g-Proteobacteria",   61.5),
    "Acinetobacter baumannii":     ("g-Proteobacteria",   39.0),
    "Vibrio cholerae":             ("g-Proteobacteria",   47.5),
    "Rhodobacter sphaeroides":     ("a-Proteobacteria",   68.8),
    "Rhodopseudomonas palustris":  ("a-Proteobacteria",   65.0),
    "Campylobacter jejuni":        ("e-Proteobacteria",   30.5),
    "Helicobacter pylori":         ("e-Proteobacteria",   38.9),
    "Bacillus subtilis":           ("Firmicutes",          43.5),
    "Staphylococcus aureus":       ("Firmicutes",          32.9),
    "Lactobacillus acidophilus":    ("Firmicutes",          34.7),
    "Listeria monocytogenes":      ("Firmicutes",          38.0),
    "Enterococcus faecalis":       ("Firmicutes",          37.4),
    "Clostridium acetobutylicum":  ("Firmicutes",          30.9),
    "Mycobacterium tuberculosis":  ("Actinobacteria",      65.6),
    "Streptomyces coelicolor":     ("Actinobacteria",      72.1),
    "Deinococcus radiodurans":     ("Deinococcota",        67.0),
    "Thermus thermophilus":        ("Deinococcota",        69.4),
    "Mycoplasma genitalium":       ("Tenericutes",         31.7),
    "Borrelia burgdorferi":        ("Spirochaetes",        28.6),
}

PHYLUM_COLORS = {
    "Escherichia coli K-12":       "#2171b5",
    "Escherichia coli 536":        "#4a9fd4",
    "Pseudomonas putida":          "#08519c",
    "Acinetobacter baumannii":     "#6baed6",
    "Vibrio cholerae":             "#084594",
    "Rhodobacter sphaeroides":     "#17becf",
    "Rhodopseudomonas palustris":  "#0d9faf",
    "Campylobacter jejuni":        "#238b45",
    "Helicobacter pylori":         "#74c476",
    "Bacillus subtilis":           "#cb181d",
    "Staphylococcus aureus":       "#e6550d",
    "Lactobacillus acidophilus":    "#fd8d3c",
    "Listeria monocytogenes":      "#de2d73",
    "Enterococcus faecalis":       "#f768a1",
    "Clostridium acetobutylicum":  "#fc4e2a",
    "Mycobacterium tuberculosis":  "#7b4fbe",
    "Streptomyces coelicolor":     "#54278f",
    "Deinococcus radiodurans":     "#b5a400",
    "Thermus thermophilus":        "#ffd700",
    "Mycoplasma genitalium":       "#636363",
    "Borrelia burgdorferi":        "#8c564b",
}

PHYLUM_ORDER = [
    "g-Proteobacteria", "a-Proteobacteria", "e-Proteobacteria",
    "Firmicutes", "Actinobacteria", "Deinococcota",
    "Tenericutes", "Spirochaetes", "Other",
]

def resolve_genome_name(raw):
    if raw in PHYLUM_COLORS: return raw
    clean = raw.strip()
    if clean in NC_TO_SPECIES: return NC_TO_SPECIES[clean]
    base = clean.rsplit(".", 1)[0] if "." in clean else clean
    if base in NC_TO_SPECIES: return NC_TO_SPECIES[base]
    for key, val in NC_TO_SPECIES.items():
        if key.startswith(base) or base.startswith(key.split(".")[0]): return val
    known = ["Escherichia","Bacillus","Pseudomonas","Streptomyces",
             "Lactobacillus","Staphylococcus","Acinetobacter","Vibrio",
             "Mycobacterium","Campylobacter","Helicobacter","Listeria",
             "Deinococcus","Mycoplasma","Clostridium","Enterococcus",
             "Borrelia","Thermus","Rhodobacter","Rhodopseudomonas"]
    for g in known:
        if g.lower() in raw.lower(): return raw.replace("_", " ")
    return clean.replace("_", " ")

def get_color(name):
    if name in PHYLUM_COLORS: return PHYLUM_COLORS[name]
    genus = name.split()[0] if " " in name else name
    for k, c in PHYLUM_COLORS.items():
        if k.startswith(genus): return c
    return "#999999"

def get_phylum(name):
    if name in SPECIES_METADATA: return SPECIES_METADATA[name][0]
    genus = name.split()[0] if " " in name else name
    for k, (p, _) in SPECIES_METADATA.items():
        if k.startswith(genus): return p
    return "Other"

def sort_by_phylum(unique_ids, resolved):
    items = []
    for gid in unique_ids:
        nm = resolved[gid] if gid < len(resolved) else f"Genome {gid}"
        items.append((gid, nm, get_phylum(nm)))
    items.sort(key=lambda x: (
        PHYLUM_ORDER.index(x[2]) if x[2] in PHYLUM_ORDER else len(PHYLUM_ORDER), x[1]))
    return items

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
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

def load_model(ckpt_path, tokenizer, device):
    import copy
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
    m = ckpt.get("metrics",{})
    rm = m.get("rankme",0); gc = m.get("gc_abs_r",0)
    print(f"  Loaded ep {ep}: RankMe={rm:.1f}, GC|r|={gc:.3f}")
    return model, ep

def compute_metrics(emb, labels, species_names, n_neighbors=10, n_folds=5):
    results = {}
    n_uniq = len(set(labels))
    if n_uniq > 1:
        results["silhouette"] = silhouette_score(emb, labels, metric="cosine",
            sample_size=min(5000, len(labels)), random_state=42)
    else:
        results["silhouette"] = 0.0
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    emb_n = emb / norms
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean",
                                weights="distance", n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(knn, emb_n, labels, cv=cv, scoring="accuracy")
    results["knn_mean"] = scores.mean()
    results["knn_std"] = scores.std()
    knn2 = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean",
                                 weights="distance", n_jobs=-1)
    knn2.fit(emb_n, labels)
    preds = knn2.predict(emb_n)
    per_sp = {}
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        nm = species_names[lbl] if lbl < len(species_names) else f"Species {lbl}"
        per_sp[nm] = {"accuracy": float((preds[mask]==labels[mask]).mean()), "count": int(mask.sum())}
    results["per_species"] = per_sp
    return results

def print_metrics(met, label):
    print(f"\n  {label} Metrics:")
    print(f"    Silhouette (cosine):   {met['silhouette']:.4f}")
    print(f"    10-NN Accuracy (5-CV): {met['knn_mean']:.4f} +/- {met['knn_std']:.4f}")
    print(f"    {'Species':<35s} {'Acc':>6s} {'N':>5s}")
    print(f"    {'-'*35} {'-'*6} {'-'*5}")
    for nm, info in sorted(met["per_species"].items()):
        print(f"    {nm:<35s} {info['accuracy']:>5.1%} {info['count']:>5d}")

def plot_single(coords, ids, resolved, title, method, metrics, path):
    uids = sorted(set(ids))
    species = sort_by_phylum(uids, resolved)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for gid, name, phy in species:
        mask = ids == gid
        color = get_color(name)
        parts = name.split()
        if len(parts) >= 2 and parts[0][0].isupper():
            lab = f"$\\it{{{parts[0][0]}.}}$ $\\it{{{parts[1]}}}$"
            if len(parts) > 2: lab += " " + " ".join(parts[2:])
        else:
            lab = name
        ax.scatter(coords[mask,0], coords[mask,1], c=color, s=10, alpha=0.55,
                   label=lab, edgecolors="white", linewidths=0.15, rasterized=True)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"{method} 1", fontsize=11)
    ax.set_ylabel(f"{method} 2", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7,
              markerscale=2.2, framealpha=0.95, ncol=1, handletextpad=0.3,
              borderpad=0.6, labelspacing=0.4, edgecolor="#cccccc")
    if metrics:
        sil = metrics.get("silhouette", 0)
        knn = metrics.get("knn_mean", 0)
        knn_s = metrics.get("knn_std", 0)
        txt = f"Silhouette (cos): {sil:.3f}\n10-NN Acc (5-CV): {knn:.1%} +/- {knn_s:.1%}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5,
                verticalalignment="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#bbbbbb", alpha=0.92))
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {path}")

def plot_multi(all_c, all_i, labels, resolved, all_met, method, path):
    n_p = len(all_c)
    fig, axes = plt.subplots(1, n_p, figsize=(6.5*n_p, 6))
    if n_p == 1: axes = [axes]
    all_uid = sorted(set(np.concatenate(all_i)))
    species = sort_by_phylum(all_uid, resolved)
    for ax, co, ids, lab, met in zip(axes, all_c, all_i, labels, all_met):
        for gid, name, phy in species:
            mask = ids == gid
            if not mask.any(): continue
            color = get_color(name)
            parts = name.split()
            short = f"{parts[0][0]}. {parts[1]}" if len(parts) >= 2 else name[:20]
            ax.scatter(co[mask,0], co[mask,1], c=color, s=5, alpha=0.45,
                       label=short, edgecolors="none", rasterized=True)
        ax.set_title(lab, fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if met:
            sil = met.get("silhouette", 0)
            knn = met.get("knn_mean", 0)
            ax.text(0.03, 0.03, f"Sil: {sil:.3f} | kNN: {knn:.1%}",
                    transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#ccc", alpha=0.9))
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=6.5, markerscale=2.5,
               bbox_to_anchor=(1.0, 0.5), ncol=1, handletextpad=0.3,
               borderpad=0.5, labelspacing=0.35, edgecolor="#cccccc")
    fig.suptitle(f"DNA-Bacteria-JEPA - Representation Evolution ({method})",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {path}")

def plot_metrics_bars(labels, all_met, path):
    n = len(labels)
    sils = [m["silhouette"] for m in all_met]
    knns = [m["knn_mean"] for m in all_met]
    kstds = [m["knn_std"] for m in all_met]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(n); w = 0.5
    b1 = a1.bar(x, sils, w, color="#4a90d9", edgecolor="white", linewidth=0.5)
    a1.set_ylabel("Silhouette Score (cosine)", fontsize=10)
    a1.set_title("Cluster Separability", fontsize=12, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=10)
    a1.set_ylim(0, max(sils)*1.3 if max(sils)>0 else 1)
    for bar, val in zip(b1, sils):
        a1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    b2 = a2.bar(x, knns, w, yerr=kstds, capsize=4, color="#e67e22",
                edgecolor="white", linewidth=0.5, error_kw={"linewidth":1.2})
    a2.set_ylabel("10-NN Accuracy (5-fold CV)", fontsize=10)
    a2.set_title("Species Classification", fontsize=12, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=10)
    a2.set_ylim(0, min(max(knns)*1.2, 1.0) if max(knns)>0 else 1)
    for bar, val in zip(b2, knns):
        a2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle("DNA-Bacteria-JEPA - Quantitative Evaluation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {path}")

def plot_heatmap(labels, all_met, path):
    all_sp = set()
    for m in all_met: all_sp.update(m["per_species"].keys())
    sp_sorted = sorted(all_sp, key=lambda s: (
        PHYLUM_ORDER.index(get_phylum(s)) if get_phylum(s) in PHYLUM_ORDER else 99, s))
    ns = len(sp_sorted); ne = len(labels)
    fig, ax = plt.subplots(figsize=(3+1.2*ne, 0.35*ns+1.5))
    data = np.zeros((ns, ne))
    for j, m in enumerate(all_met):
        for i, sp in enumerate(sp_sorted):
            if sp in m["per_species"]: data[i,j] = m["per_species"][sp]["accuracy"]
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(ne)); ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(ns))
    ylabs = []
    for sp in sp_sorted:
        parts = sp.split()
        ylabs.append(f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts)>=2 else sp)
    ax.set_yticklabels(ylabs, fontsize=8)
    for i in range(ns):
        for j in range(ne):
            v = data[i,j]; c = "white" if v < 0.7 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=c, fontweight="bold")
    ax.tick_params(axis="y", which="both", left=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("kNN Accuracy (train)", fontsize=9)
    ax.set_title("Per-Species Classification Accuracy", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--data-path", default="data/processed/pretrain_sequences_expanded.csv")
    p.add_argument("--output-dir", default="figures/embeddings")
    p.add_argument("--max-samples", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-neighbors", type=int, default=10)
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

    resolved = [resolve_genome_name(g) for g in ug]
    print(f"\n  Genome Mapping:")
    for raw, res in zip(ug, resolved):
        st = "OK" if raw != res else "  "
        print(f"    {st} {raw:<25s} -> {res:<30s} [{get_phylum(res)}]")

    class LDS(torch.utils.data.Dataset):
        def __len__(self): return len(base_ds)
        def __getitem__(self, idx): return base_ds[idx], gids_list[idx]
    ds = LDS()

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                    pin_memory=device.type=="cuda", drop_last=False)
    od = Path(args.output_dir)
    if not od.is_absolute(): od = (PROJECT_ROOT / od).resolve()
    od.mkdir(parents=True, exist_ok=True)
    ckpts = [Path(c) if Path(c).is_absolute() else (PROJECT_ROOT/c).resolve() for c in args.checkpoint]

    all_emb, all_ids, ep_labels = [], [], []
    for cp in ckpts:
        print(f"\nLoading {cp.name}...")
        model, ep = load_model(cp, tokenizer, device)
        ep_labels.append(f"Epoch {ep}")
        t0 = time.time()
        emb, ids = extract_embeddings(model, dl, tokenizer, device, args.max_samples)
        print(f"  {emb.shape[0]:,} x {emb.shape[1]}D in {time.time()-t0:.1f}s")
        all_emb.append(emb); all_ids.append(ids)
        del model
        if device.type=="cuda": torch.cuda.empty_cache()

    print("\nComputing metrics...")
    all_met = []
    for emb, ids, lab in zip(all_emb, all_ids, ep_labels):
        met = compute_metrics(emb, ids, resolved, n_neighbors=args.n_neighbors)
        print_metrics(met, lab)
        all_met.append(met)

    print("\nRunning t-SNE...")
    all_tsne = []
    for emb, ids, lab, met in zip(all_emb, all_ids, ep_labels, all_met):
        t0 = time.time()
        ts = TSNE(n_components=2, perplexity=min(30, emb.shape[0]//4),
                  learning_rate="auto", init="pca", random_state=42, max_iter=1500)
        coords = ts.fit_transform(emb)
        all_tsne.append(coords)
        print(f"  {lab}: {time.time()-t0:.1f}s")
        plot_single(coords, ids, resolved, f"DNA-Bacteria-JEPA - {lab}",
                    "t-SNE", met, od / f"tsne_{lab.lower().replace(' ','_')}.png")
    if len(ckpts) > 1:
        plot_multi(all_tsne, all_ids, ep_labels, resolved, all_met,
                   "t-SNE", od / "tsne_evolution.png")

    try:
        import umap
        print("\nRunning UMAP...")
        all_um = []
        for emb, ids, lab, met in zip(all_emb, all_ids, ep_labels, all_met):
            t0 = time.time()
            r = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          metric="cosine", random_state=42)
            coords = r.fit_transform(emb)
            all_um.append(coords)
            print(f"  {lab}: {time.time()-t0:.1f}s")
            plot_single(coords, ids, resolved, f"DNA-Bacteria-JEPA - {lab}",
                        "UMAP", met, od / f"umap_{lab.lower().replace(' ','_')}.png")
        if len(ckpts) > 1:
            plot_multi(all_um, all_ids, ep_labels, resolved, all_met,
                       "UMAP", od / "umap_evolution.png")
    except ImportError:
        print("\n  UMAP not installed. pip install umap-learn")

    if len(ckpts) > 1:
        plot_metrics_bars(ep_labels, all_met, od / "metrics_comparison.png")
        plot_heatmap(ep_labels, all_met, od / "per_species_accuracy.png")

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {od}")
    for lab, met in zip(ep_labels, all_met):
        print(f"    {lab}: Sil={met['silhouette']:.4f}, kNN={met['knn_mean']:.1%} +/- {met['knn_std']:.1%}")
    print(f"{'='*60}")

if __name__ == "__main__": main()
