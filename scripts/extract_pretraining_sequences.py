"""
Extract fixed-size windows from downloaded bacterial FASTA files.

Creates:
    data/processed/pretrain_sequences.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_fasta(fasta_path: Path) -> str:
    """Read FASTA and return concatenated sequence."""
    parts: List[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts.append(line.upper())
    return "".join(parts)


def extract_windows(
    sequence: str,
    genome_name: str,
    window_size: int = 512,
    stride: int = 256,
) -> List[Dict[str, object]]:
    """Extract clean A/C/G/T windows with overlap."""
    windows: List[Dict[str, object]] = []
    valid_bases = set("ACGT")

    if len(sequence) < window_size:
        return windows

    for i in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[i : i + window_size]
        if set(window).issubset(valid_bases):
            windows.append(
                {
                    "sequence": window,
                    "genome": genome_name,
                    "position": i,
                }
            )
    return windows


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--genome-dir",
        default="data/raw/bacterial_genomes",
        help="Directory containing downloaded FASTA files",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/pretrain_sequences.csv",
        help="Output CSV path",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genome_dir = resolve_path(args.genome_dir)
    output_csv = resolve_path(args.output_csv)

    if not genome_dir.exists():
        print(f"[error] Genome directory not found: {genome_dir}")
        print("Run: python scripts/download_bacterial_genomes.py")
        return

    fasta_files = sorted(genome_dir.glob("*.fasta"))
    if not fasta_files:
        print(f"[error] No FASTA files found in: {genome_dir}")
        return

    print("=" * 60)
    print("Extracting Pretraining Sequences")
    print("=" * 60)
    print(f"Genome files: {len(fasta_files)}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print()

    all_windows: List[Dict[str, object]] = []
    for fasta_file in fasta_files:
        print(f"[processing] {fasta_file.name}")
        genome_name = fasta_file.stem.split("_", 1)[1] if "_" in fasta_file.stem else fasta_file.stem

        sequence = read_fasta(fasta_file)
        print(f"  length: {len(sequence):,} bases")

        windows = extract_windows(
            sequence=sequence,
            genome_name=genome_name,
            window_size=args.window_size,
            stride=args.stride,
        )
        all_windows.extend(windows)
        print(f"  windows: {len(windows):,}")
        print()

    df = pd.DataFrame(all_windows, columns=["sequence", "genome", "position"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("=" * 60)
    print(f"Saved {len(df):,} windows")
    if not df.empty:
        total_bases = len(df) * args.window_size
        print(f"Total bases: {total_bases:,}")
        print(f"Unique genomes: {df['genome'].nunique()}")
    print(f"Output: {output_csv.resolve()}")
    print("Next: python scripts/01_pretrain_jepa.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
