
"""Embedding visualization for DNA-Bacteria-JEPA."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cas12a.dataset import BacterialGenomeDataset
from src.cas12a.encoder import SparseTransformerEncoder
from src.cas12a.jepa_pretrain import Cas12aJEPA, JEPAConfig

@torch.no_grad()
def extract_embeddings(model, dataloader, tokenizer, device, max_samples=10000):
    model.eval()
    all_emb, all_ids = [], []
    n = 0
    for batch in dataloader:
        if n >= max_samples: break
        tokens, genome_ids = batch[0], batch[1]
        tokens = tokens.to(device, non_blocking=True)
        attn = tokenizer.get_attention_mask(tokens)
        emb = model.encode(tokens, attention_mask=attn, use_target=True)
        all_emb.append(emb.cpu().numpy())
        all_ids.append(genome_ids.numpy())
        n += tokens.shape[0]
    return np.concatenate(all_emb)[:max_samples], np.concatenate(all_ids)[:max_samples]

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
    import copy
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
    print(f"  Loaded epoch {ep}: RankMe={m.get('rankme',0):.1f}, GC|r|={m.get('gc_abs_r',0):.3f}")
    return model, ep

def plot_single(coords, ids, names, title, method, path):
    uids = sorted(set(ids))
    n = len(uids)
    cmap = plt.cm.tab20 if n <= 20 else plt.cm.gist_ncar
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    for i, gid in enumerate(uids):
        mask = ids == gid
        c = cmap(i/max(n-1,1))
        nm = names[gid].replace("_"," ")[:30] if gid < len(names) else f"G{gid}"
        ax.scatter(coords[mask,0], coords[mask,1], c=[c], s=8, alpha=0.6, label=nm, edgecolors="none")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=7, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

def plot_multi(all_c, all_i, labels, names, method, path):
    n_p = len(all_c)
    fig, axes = plt.subplots(1, n_p, figsize=(7*n_p, 6))
    if n_p == 1: axes = [axes]
    uids = sorted(set(np.concatenate(all_i)))
    n = len(uids)
    cmap = plt.cm.tab20 if n <= 20 else plt.cm.gist_ncar
    for ax, co, ids, lab in zip(axes, all_c, all_i, labels):
        for i, gid in enumerate(uids):
            mask = ids == gid
            if not mask.any(): continue
            c = cmap(i/max(n-1,1))
            nm = names[gid][:20].replace("_"," ") if gid < len(names) else f"G{gid}"
            ax.scatter(co[mask,0], co[mask,1], c=[c], s=6, alpha=0.5, label=nm, edgecolors="none")
        ax.set_title(lab, fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=6, markerscale=2, bbox_to_anchor=(1.0,0.5))
    fig.suptitle(f"DNA-Bacteria-JEPA Representation Evolution ({method})", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--data-path", default="data/processed/pretrain_sequences_expanded.csv")
    p.add_argument("--output-dir", default="figures/embeddings")
    p.add_argument("--max-samples", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig
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
    gids = [g2id[g] for g in genomes]
    class LDS(torch.utils.data.Dataset):
        def __len__(self): return len(base_ds)
        def __getitem__(self, idx): return base_ds[idx], gids[idx]
    ds = LDS()
    print(f"  {len(ug)} genomes")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                    pin_memory=device.type=="cuda", drop_last=False)
    od = Path(args.output_dir)
    if not od.is_absolute(): od = (PROJECT_ROOT / od).resolve()
    od.mkdir(parents=True, exist_ok=True)
    ckpts = [Path(c) if Path(c).is_absolute() else (PROJECT_ROOT/c).resolve() for c in args.checkpoint]

    all_emb, all_ids, labels = [], [], []
    for cp in ckpts:
        print(f"\nLoading {cp.name}...")
        model, ep = load_model(cp, tokenizer, device)
        labels.append(f"Epoch {ep}")
        t0 = time.time()
        emb, ids = extract_embeddings(model, dl, tokenizer, device, args.max_samples)
        print(f"  {emb.shape[0]} x {emb.shape[1]}D in {time.time()-t0:.1f}s")
        all_emb.append(emb); all_ids.append(ids)
        del model
        if device.type=="cuda": torch.cuda.empty_cache()

    print("\nRunning t-SNE...")
    all_tsne = []
    for emb, lab in zip(all_emb, labels):
        t0 = time.time()
        ts = TSNE(n_components=2, perplexity=min(30,emb.shape[0]//4),
                  learning_rate="auto", init="pca", random_state=42, max_iter=1000)
        coords = ts.fit_transform(emb)
        print(f"  {lab}: {time.time()-t0:.1f}s")
        all_tsne.append(coords)

    for coords, ids, lab in zip(all_tsne, all_ids, labels):
        plot_single(coords, ids, ug, f"DNA-Bacteria-JEPA {lab}", "t-SNE",
                    od / f"tsne_{lab.lower().replace(' ','_')}.png")

    if len(ckpts) > 1:
        plot_multi(all_tsne, all_ids, labels, ug, "t-SNE", od / "tsne_evolution.png")

    try:
        import umap
        print("\nRunning UMAP...")
        all_um = []
        for emb, lab in zip(all_emb, labels):
            t0 = time.time()
            r = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
            coords = r.fit_transform(emb)
            print(f"  {lab}: {time.time()-t0:.1f}s")
            all_um.append(coords)
        for coords, ids, lab in zip(all_um, all_ids, labels):
            plot_single(coords, ids, ug, f"DNA-Bacteria-JEPA {lab}", "UMAP",
                        od / f"umap_{lab.lower().replace(' ','_')}.png")
        if len(ckpts) > 1:
            plot_multi(all_um, all_ids, labels, ug, "UMAP", od / "umap_evolution.png")
    except ImportError:
        print("\n  UMAP not installed. pip install umap-learn")

    print(f"\nDone! Figures in: {od}")

if __name__ == "__main__": main()
