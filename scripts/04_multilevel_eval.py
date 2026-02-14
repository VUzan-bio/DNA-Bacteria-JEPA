"""Phylum-level evaluation of DNA-Bacteria-JEPA embeddings."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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

SPECIES_TO_PHYLUM = {
    "Escherichia coli K-12":       "Proteobacteria",
    "Escherichia coli 536":        "Proteobacteria",
    "Pseudomonas putida":          "Proteobacteria",
    "Acinetobacter baumannii":     "Proteobacteria",
    "Vibrio cholerae":             "Proteobacteria",
    "Rhodobacter sphaeroides":     "Proteobacteria",
    "Rhodopseudomonas palustris":  "Proteobacteria",
    "Campylobacter jejuni":        "Proteobacteria",
    "Helicobacter pylori":         "Proteobacteria",
    "Bacillus subtilis":           "Firmicutes",
    "Staphylococcus aureus":       "Firmicutes",
    "Lactobacillus acidophilus":    "Firmicutes",
    "Listeria monocytogenes":      "Firmicutes",
    "Enterococcus faecalis":       "Firmicutes",
    "Clostridium acetobutylicum":  "Firmicutes",
    "Mycobacterium tuberculosis":  "Actinobacteria",
    "Streptomyces coelicolor":     "Actinobacteria",
    "Deinococcus radiodurans":     "Deinococcota",
    "Thermus thermophilus":        "Deinococcota",
    "Mycoplasma genitalium":       "Tenericutes",
    "Borrelia burgdorferi":        "Spirochaetes",
}

SPECIES_TO_CLASS = {
    "Escherichia coli K-12":       "Gammaproteobacteria",
    "Escherichia coli 536":        "Gammaproteobacteria",
    "Pseudomonas putida":          "Gammaproteobacteria",
    "Acinetobacter baumannii":     "Gammaproteobacteria",
    "Vibrio cholerae":             "Gammaproteobacteria",
    "Rhodobacter sphaeroides":     "Alphaproteobacteria",
    "Rhodopseudomonas palustris":  "Alphaproteobacteria",
    "Campylobacter jejuni":        "Epsilonproteobacteria",
    "Helicobacter pylori":         "Epsilonproteobacteria",
    "Bacillus subtilis":           "Bacilli",
    "Staphylococcus aureus":       "Bacilli",
    "Lactobacillus acidophilus":    "Bacilli",
    "Listeria monocytogenes":      "Bacilli",
    "Enterococcus faecalis":       "Bacilli",
    "Clostridium acetobutylicum":  "Clostridia",
    "Mycobacterium tuberculosis":  "Actinomycetia",
    "Streptomyces coelicolor":     "Actinomycetia",
    "Deinococcus radiodurans":     "Deinococci",
    "Thermus thermophilus":        "Deinococci",
    "Mycoplasma genitalium":       "Mollicutes",
    "Borrelia burgdorferi":        "Spirochaetia",
}

SPECIES_TO_GC_BIN = {
    "Escherichia coli K-12":       "Medium GC (40-55%)",
    "Escherichia coli 536":        "Medium GC (40-55%)",
    "Pseudomonas putida":          "High GC (>55%)",
    "Acinetobacter baumannii":     "Low GC (<40%)",
    "Vibrio cholerae":             "Medium GC (40-55%)",
    "Rhodobacter sphaeroides":     "High GC (>55%)",
    "Rhodopseudomonas palustris":  "High GC (>55%)",
    "Campylobacter jejuni":        "Low GC (<40%)",
    "Helicobacter pylori":         "Low GC (<40%)",
    "Bacillus subtilis":           "Medium GC (40-55%)",
    "Staphylococcus aureus":       "Low GC (<40%)",
    "Lactobacillus acidophilus":    "Low GC (<40%)",
    "Listeria monocytogenes":      "Low GC (<40%)",
    "Enterococcus faecalis":       "Low GC (<40%)",
    "Clostridium acetobutylicum":  "Low GC (<40%)",
    "Mycobacterium tuberculosis":  "High GC (>55%)",
    "Streptomyces coelicolor":     "High GC (>55%)",
    "Deinococcus radiodurans":     "High GC (>55%)",
    "Thermus thermophilus":        "High GC (>55%)",
    "Mycoplasma genitalium":       "Low GC (<40%)",
    "Borrelia burgdorferi":        "Low GC (<40%)",
}

PHYLUM_COLORS = {
    "Proteobacteria": "#2171b5",
    "Firmicutes":     "#cb181d",
    "Actinobacteria": "#7b4fbe",
    "Deinococcota":   "#b5a400",
    "Tenericutes":    "#636363",
    "Spirochaetes":   "#8c564b",
}

CLASS_COLORS = {
    "Gammaproteobacteria":   "#2171b5",
    "Alphaproteobacteria":   "#17becf",
    "Epsilonproteobacteria": "#238b45",
    "Bacilli":               "#cb181d",
    "Clostridia":            "#fc4e2a",
    "Actinomycetia":         "#7b4fbe",
    "Deinococci":            "#b5a400",
    "Mollicutes":            "#636363",
    "Spirochaetia":          "#8c564b",
}

GC_COLORS = {
    "Low GC (<40%)":     "#e6550d",
    "Medium GC (40-55%)":"#2171b5",
    "High GC (>55%)":    "#54278f",
}

def resolve_genome_name(raw):
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
    print(f"  Loaded ep {ep}: RankMe={m.get('rankme',0):.1f}")
    return model, ep

def eval_grouping(emb, labels, group_name, n_neighbors=10, n_folds=5):
    unique = sorted(set(labels))
    n_classes = len(unique)
    random_baseline = 1.0 / n_classes
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    emb_n = emb / norms
    sil = silhouette_score(emb, labels, metric="cosine",
        sample_size=min(5000, len(labels)), random_state=42) if n_classes > 1 else 0.0
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean",
                                weights="distance", n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(knn, emb_n, labels, cv=cv, scoring="accuracy")
    # Per-group breakdown
    knn2 = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean",
                                 weights="distance", n_jobs=-1)
    knn2.fit(emb_n, labels)
    preds = knn2.predict(emb_n)
    per_group = {}
    label_names = sorted(set(labels))
    for lbl in label_names:
        mask = labels == lbl
        per_group[lbl] = {"accuracy": float((preds[mask]==labels[mask]).mean()), "count": int(mask.sum())}
    return {
        "group_name": group_name,
        "n_classes": n_classes,
        "random_baseline": random_baseline,
        "silhouette": sil,
        "knn_mean": scores.mean(),
        "knn_std": scores.std(),
        "per_group": per_group,
    }

def print_eval(res):
    print(f"\n  === {res['group_name']} ({res['n_classes']} classes, random={res['random_baseline']:.1%}) ===")
    print(f"    Silhouette (cosine):   {res['silhouette']:.4f}")
    print(f"    10-NN Accuracy (5-CV): {res['knn_mean']:.4f} +/- {res['knn_std']:.4f}")
    print(f"    Lift over random:      {res['knn_mean']/res['random_baseline']:.1f}x")
    print(f"    {'Group':<30s} {'Acc':>6s} {'N':>5s}")
    print(f"    {'-'*30} {'-'*6} {'-'*5}")
    for nm, info in sorted(res["per_group"].items()):
        print(f"    {nm:<30s} {info['accuracy']:>5.1%} {info['count']:>5d}")

def plot_multilevel_bars(ep_labels, all_results, path):
    levels = ["Species", "Class", "Phylum", "GC Content"]
    n_ep = len(ep_labels)
    n_lv = len(levels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n_lv)
    w = 0.8 / n_ep
    epoch_colors = ["#a6cee3", "#1f78b4", "#08306b"][:n_ep]
    # Silhouette
    ax = axes[0]
    for j, (lab, res_dict) in enumerate(zip(ep_labels, all_results)):
        vals = [res_dict[lv]["silhouette"] for lv in levels]
        ax.bar(x + j*w - (n_ep-1)*w/2, vals, w, label=lab, color=epoch_colors[j],
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Silhouette Score (cosine)", fontsize=10)
    ax.set_title("Cluster Separability by Taxonomic Level", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    # kNN
    ax = axes[1]
    for j, (lab, res_dict) in enumerate(zip(ep_labels, all_results)):
        vals = [res_dict[lv]["knn_mean"] for lv in levels]
        errs = [res_dict[lv]["knn_std"] for lv in levels]
        bars = ax.bar(x + j*w - (n_ep-1)*w/2, vals, w, yerr=errs, capsize=3,
               label=lab, color=epoch_colors[j], edgecolor="white", linewidth=0.5,
               error_kw={"linewidth":1})
    # Random baselines
    baselines = [all_results[0][lv]["random_baseline"] for lv in levels]
    for i, bl in enumerate(baselines):
        ax.plot([i-0.5, i+0.5], [bl, bl], color="#999", linewidth=1, linestyle=":", zorder=0)
    ax.text(n_lv-0.5, baselines[-1]+0.01, "random", fontsize=7, color="#999", ha="right")
    ax.set_ylabel("10-NN Accuracy (5-fold CV)", fontsize=10)
    ax.set_title("Classification Accuracy by Taxonomic Level", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    fig.suptitle("DNA-Bacteria-JEPA - Multi-Level Taxonomic Evaluation",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {path}")

def plot_per_group_heatmap(ep_labels, all_results, level, color_map, path):
    groups = sorted(set(g for res in all_results for g in res[level]["per_group"].keys()))
    ng = len(groups); ne = len(ep_labels)
    fig, ax = plt.subplots(figsize=(3+1.2*ne, 0.4*ng+1.5))
    data = np.zeros((ng, ne))
    for j, res in enumerate(all_results):
        for i, g in enumerate(groups):
            if g in res[level]["per_group"]:
                data[i,j] = res[level]["per_group"][g]["accuracy"]
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(ne)); ax.set_xticklabels(ep_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(ng)); ax.set_yticklabels(groups, fontsize=9)
    for i in range(ng):
        for j in range(ne):
            v = data[i,j]; c = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color=c, fontweight="bold")
    # Color patches for phylum identity
    for i, g in enumerate(groups):
        if g in color_map:
            ax.plot(-0.45, i, marker="s", markersize=8, color=color_map[g], clip_on=False)
    ax.tick_params(axis="y", which="both", left=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("kNN Accuracy (train)", fontsize=9)
    bl = all_results[0][level]["random_baseline"]
    ax.set_title(f"{level}-Level Classification (random baseline: {bl:.1%})",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved: {path}")

def plot_summary_table(ep_labels, all_results, path):
    levels = ["Species", "Class", "Phylum", "GC Content"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    headers = ["Level", "# Classes", "Random"] + [f"{l} kNN" for l in ep_labels] + [f"{l} Sil" for l in ep_labels]
    rows = []
    for lv in levels:
        row = [lv, str(all_results[0][lv]["n_classes"]), f"{all_results[0][lv]['random_baseline']:.1%}"]
        for res in all_results:
            row.append(f"{res[lv]['knn_mean']:.1%} +/- {res[lv]['knn_std']:.1%}")
        for res in all_results:
            row.append(f"{res[lv]['silhouette']:.3f}")
        rows.append(row)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2171b5")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#dddddd")
    ax.set_title("DNA-Bacteria-JEPA - Multi-Level Evaluation Summary",
                 fontsize=13, fontweight="bold", pad=20)
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

    # Build label arrays for each taxonomic level
    species_labels = np.array([resolved[g2id[g]] for g in genomes])
    phylum_labels = np.array([SPECIES_TO_PHYLUM.get(resolve_genome_name(g), "Unknown") for g in genomes])
    class_labels = np.array([SPECIES_TO_CLASS.get(resolve_genome_name(g), "Unknown") for g in genomes])
    gc_labels = np.array([SPECIES_TO_GC_BIN.get(resolve_genome_name(g), "Unknown") for g in genomes])

    print(f"\n  Taxonomic levels:")
    print(f"    Species: {len(set(species_labels))} classes")
    print(f"    Class:   {len(set(class_labels))} classes")
    print(f"    Phylum:  {len(set(phylum_labels))} classes")
    print(f"    GC bin:  {len(set(gc_labels))} classes")

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

    all_results = []
    ep_labels = []

    for cp in ckpts:
        print(f"\n{'='*60}")
        print(f"Loading {cp.name}...")
        model, ep = load_model(cp, tokenizer, device)
        ep_labels.append(f"Epoch {ep}")
        t0 = time.time()
        emb, ids = extract_embeddings(model, dl, tokenizer, device, args.max_samples)
        print(f"  {emb.shape[0]:,} x {emb.shape[1]}D in {time.time()-t0:.1f}s")
        del model
        if device.type=="cuda": torch.cuda.empty_cache()

        # Map genome ids back to labels for this subsample
        idx_to_genome = {i: ug[i] for i in range(len(ug))}
        sp_sub = np.array([resolved[gid] for gid in ids])
        ph_sub = np.array([SPECIES_TO_PHYLUM.get(resolved[gid], "Unknown") for gid in ids])
        cl_sub = np.array([SPECIES_TO_CLASS.get(resolved[gid], "Unknown") for gid in ids])
        gc_sub = np.array([SPECIES_TO_GC_BIN.get(resolved[gid], "Unknown") for gid in ids])

        results = {}
        print(f"\n  Evaluating {ep_labels[-1]}...")
        results["Species"] = eval_grouping(emb, sp_sub, "Species", args.n_neighbors)
        print_eval(results["Species"])
        results["Class"] = eval_grouping(emb, cl_sub, "Class", args.n_neighbors)
        print_eval(results["Class"])
        results["Phylum"] = eval_grouping(emb, ph_sub, "Phylum", args.n_neighbors)
        print_eval(results["Phylum"])
        results["GC Content"] = eval_grouping(emb, gc_sub, "GC Content", args.n_neighbors)
        print_eval(results["GC Content"])
        all_results.append(results)

    # Generate figures
    print(f"\n{'='*60}")
    print("Generating figures...")
    plot_multilevel_bars(ep_labels, all_results, od / "multilevel_metrics.png")
    plot_per_group_heatmap(ep_labels, all_results, "Phylum", PHYLUM_COLORS, od / "phylum_accuracy.png")
    plot_per_group_heatmap(ep_labels, all_results, "Class", CLASS_COLORS, od / "class_accuracy.png")
    plot_per_group_heatmap(ep_labels, all_results, "GC Content", GC_COLORS, od / "gc_accuracy.png")
    plot_summary_table(ep_labels, all_results, od / "multilevel_summary.png")

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {od}")
    print(f"\n  Final Summary:")
    print(f"  {'Level':<15s} {'Random':>8s}", end="")
    for lab in ep_labels: print(f" {lab:>15s}", end="")
    print()
    for lv in ["Species", "Class", "Phylum", "GC Content"]:
        bl = all_results[0][lv]["random_baseline"]
        print(f"  {lv:<15s} {bl:>7.1%}", end="")
        for res in all_results:
            knn = res[lv]["knn_mean"]
            lift = knn / bl
            print(f"  {knn:>5.1%} ({lift:.1f}x)", end="")
        print()
    print(f"{'='*60}")

if __name__ == "__main__": main()
