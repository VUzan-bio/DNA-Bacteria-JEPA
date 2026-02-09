"""
Fine-tuning script for Cas12a efficiency prediction.

BEFORE RUNNING:
1. Complete pretraining (01_pretrain_jepa.py)
2. Place your data in: data/processed/cas12a_efficiency.csv
3. Format: columns [crRNA_seq, target_seq, PAM, efficiency_normalized, source]
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# Ensure `src` imports resolve when running as `python scripts/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cas12a.data_manager import Cas12aDataManager
from src.cas12a.dataset import Cas12aEfficiencyDataset, cas12a_collate
from src.cas12a.encoder import SparseTransformerEncoder
from src.cas12a.model import Cas12aEfficiencyHead, Cas12aEfficiencyModel
from src.cas12a.sparse_loss import SparseLossConfig, SparseRegressionLoss
from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_amp_dtype(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return None

    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("bf16 is not supported on this GPU; falling back to fp16")
        return torch.float16

    if precision == "fp16":
        return torch.float16

    raise ValueError(f"Unsupported precision mode: {precision}")


def evaluate(model, dataloader, device, use_amp: bool, amp_dtype):
    """Compute evaluation metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device, non_blocking=(device.type == "cuda"))
            targets = batch["efficiency"].to(device, non_blocking=(device.type == "cuda"))

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                preds, _ = model(tokens)

            all_preds.extend(preds.detach().cpu().numpy().flatten())
            all_targets.extend(targets.detach().cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(all_preds) < 2:
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "mae": float(np.abs(all_preds - all_targets).mean()) if len(all_preds) else 0.0,
        }

    pearson_r, _ = pearsonr(all_preds, all_targets)
    spearman_rho, _ = spearmanr(all_preds, all_targets)
    mae = np.abs(all_preds - all_targets).mean()

    return {
        "pearson": float(pearson_r),
        "spearman": float(spearman_rho),
        "mae": float(mae),
    }


def finetune(args):
    set_seed(args.seed)

    data_path = resolve_path(args.data_path)
    pretrained_path = resolve_path(args.pretrained_path)
    output_dir = resolve_path(args.output_dir)

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("\nPlease create this file with columns:")
        print("  - crRNA_seq, target_seq, PAM, efficiency_normalized, source")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tokenizer = Cas12aTokenizer(TokenizerConfig())

    print("Loading data...")
    df = pd.read_csv(data_path)
    splits = Cas12aDataManager.split_data(df, test_size=0.15, val_size=0.15, seed=args.seed)

    if args.max_train_samples > 0 and args.max_train_samples < len(splits.train):
        splits.train = splits.train.sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)

    if args.max_val_samples > 0 and args.max_val_samples < len(splits.val):
        splits.val = splits.val.sample(n=args.max_val_samples, random_state=args.seed).reset_index(drop=True)

    print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")

    train_dataset = Cas12aEfficiencyDataset(str(data_path), tokenizer)
    train_dataset.df = splits.train

    val_dataset = Cas12aEfficiencyDataset(str(data_path), tokenizer)
    val_dataset.df = splits.val

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": cas12a_collate,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    print(f"Loading pretrained encoder from {pretrained_path}")
    encoder = SparseTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
    )

    if pretrained_path.exists():
        checkpoint = torch.load(pretrained_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
        print("Loaded pretrained weights")
    else:
        print("No pretrained checkpoint found, training from scratch")

    task_head = Cas12aEfficiencyHead()
    model = Cas12aEfficiencyModel(encoder, task_head)
    model.to(device)

    criterion = SparseRegressionLoss(SparseLossConfig(reg_l1_weight=args.reg_l1_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_dtype = select_amp_dtype(args.precision, device)
    use_amp = amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    grad_accum_steps = max(1, args.grad_accum_steps)

    precision_label = str(amp_dtype).replace("torch.", "") if use_amp else "fp32"
    print(
        "Config: "
        f"batch_size={args.batch_size}, grad_accum_steps={grad_accum_steps}, "
        f"effective_batch_size={args.batch_size * grad_accum_steps}, "
        f"precision={precision_label}, num_workers={args.num_workers}"
    )
    print(f"\nFine-tuning for {args.epochs} epochs")

    best_val_pearson = -1.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            tokens = batch["tokens"].to(device, non_blocking=(device.type == "cuda"))
            targets = batch["efficiency"].to(device, non_blocking=(device.type == "cuda"))

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )

            with autocast_ctx:
                preds, embeddings = model(tokens)
                loss, metrics = criterion(preds, targets, embeddings)
                scaled_loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (step % grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            if (step % args.log_every == 0) or (step == len(train_loader)):
                pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})

        if (epoch + 1) % args.val_every == 0:
            val_metrics = evaluate(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
            print(
                "Val - "
                f"Pearson: {val_metrics['pearson']:.3f}, "
                f"Spearman: {val_metrics['spearman']:.3f}, "
                f"MAE: {val_metrics['mae']:.3f}"
            )

            if val_metrics["pearson"] > best_val_pearson:
                best_val_pearson = val_metrics["pearson"]
                checkpoint_path = output_dir / "best_model.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
                print(f"Saved best model (Pearson: {best_val_pearson:.3f})")

    print("\nFine-tuning complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/cas12a_efficiency.csv")
    parser.add_argument("--pretrained-path", default="checkpoints/pretrain/checkpoint_epoch100.pt")
    parser.add_argument("--output-dir", default="checkpoints/finetune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=1024)

    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-l1-weight", type=float, default=1e-4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=1)

    finetune(parser.parse_args())
