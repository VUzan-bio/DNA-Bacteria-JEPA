"""
Pretraining script for bacterial genome JEPA.

BEFORE RUNNING:
1. Place your sequences in: data/processed/pretrain_sequences.csv
2. Format: columns [sequence, genome, position]
3. Sequences should be 512 bp DNA strings (ACGT only)
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import random
import sys

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

from src.cas12a.dataset import BacterialGenomeDataset
from src.cas12a.encoder import SparseTransformerEncoder
from src.cas12a.jepa_pretrain import Cas12aJEPA, MaskingConfig
from src.cas12a.sparse_loss import JEPASparseLoss, SparseLossConfig
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


def pretrain(args):
    set_seed(args.seed)

    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("\nPlease create this file with columns:")
        print("  - sequence: DNA string (512 bp)")
        print("  - genome: genome identifier")
        print("  - position: start position")
        print("\nExample:")
        print("  sequence,genome,position")
        print("  ATCGATCG...,E_coli_K12,0")
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
    print(f"Vocab size: {tokenizer.vocab_size}")

    dataset = BacterialGenomeDataset(str(data_path), tokenizer)
    if args.max_samples > 0 and args.max_samples < len(dataset):
        dataset.df = dataset.df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Using sampled subset: {len(dataset):,} sequences")

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    encoder = SparseTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
    )

    jepa_model = Cas12aJEPA(
        encoder,
        predictor_dim=args.predictor_dim,
        ema_decay=args.ema_decay,
        masking_config=MaskingConfig(),
    )
    jepa_model.to(device)

    criterion = JEPASparseLoss(SparseLossConfig(sparsity_weight=args.sparsity_weight))

    optimizer = torch.optim.AdamW(
        jepa_model.context_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
    print(f"\nStarting pretraining for {args.epochs} epochs")

    for epoch in range(args.epochs):
        jepa_model.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch_tokens in enumerate(pbar, start=1):
            tokens = batch_tokens.to(device, non_blocking=(device.type == "cuda"))
            attention_mask = tokenizer.get_attention_mask(tokens)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )

            with autocast_ctx:
                pred_emb, target_emb, _ = jepa_model(
                    tokens,
                    attention_mask=attention_mask,
                    mask_ratio=args.mask_ratio,
                    pad_token_id=tokenizer.pad_id,
                )
                jepa_loss = torch.nn.functional.mse_loss(pred_emb, target_emb)
                total_loss, metrics = criterion(jepa_loss, pred_emb)
                scaled_loss = total_loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                jepa_model.update_ema()

            epoch_losses.append(metrics["loss"])
            if (step % args.log_every == 0) or (step == len(dataloader)):
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "sparsity": f"{metrics['active_ratio']:.2%}",
                    }
                )

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": jepa_model.context_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nPretraining complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/pretrain_sequences.csv")
    parser.add_argument("--output-dir", default="checkpoints/pretrain")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument("--max-samples", type=int, default=0, help="Use only this many rows (0 = all)")
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--predictor-dim", type=int, default=256)
    parser.add_argument("--ema-decay", type=float, default=0.996)

    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)

    pretrain(parser.parse_args())
