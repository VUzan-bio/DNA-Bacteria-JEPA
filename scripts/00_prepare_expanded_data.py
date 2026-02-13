#!/usr/bin/env python3
"""
Download and fragment bacterial genomes from NCBI RefSeq.

This script expands the training dataset from 8 genomes (~146K fragments)
to 100+ genomes (~2M fragments) for DNA-Bacteria-JEPA pretraining.

Strategy
--------
Uses NCBI RefSeq representative genomes, which are curated reference
assemblies spanning all major bacterial phyla. This mirrors the data
diversity of DNABERT-S, HyenaDNA, and Evo at a computationally
tractable scale.

Usage
-----
Step 1 — Download genomes (requires ncbi-genome-download)::

    pip install ncbi-genome-download biopython

    ncbi-genome-download bacteria \
        --assembly-levels complete \
        --refseq-categories representative \
        --formats fasta \
        --output-folder data/refseq_genomes/ \
        --parallel 4

Step 2 — Fragment into training sequences::

    python scripts/00_prepare_expanded_data.py \
        --input-dir data/refseq_genomes/ \
        --output-csv data/processed/pretrain_sequences_expanded.csv \
        --fragment-len 500 \
        --stride 250 \
        --max-per-genome 20000

Step 3 — Train with expanded data::

    python scripts/01_pretrain_jepa.py \
        --data-path data/processed/pretrain_sequences_expanded.csv \
        --epochs 200 --batch-size 512 --lr 6e-4

Author : Valentin Uzan
Project: DNA-Bacteria-JEPA
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from Bio import SeqIO
except ImportError:
    print("Error: biopython not installed. Run: pip install biopython")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def extract_genome_name(fasta_path: Path) -> str:
    """Extract a human-readable genome name from NCBI filename.

    NCBI files are typically named like:
        GCF_000005845.2_ASM584v2_genomic.fna.gz
    We extract: GCF_000005845.2
    """
    name = fasta_path.name
    # Remove extensions
    for ext in ['.fna.gz', '.fna', '.fa.gz', '.fa', '.fasta.gz', '.fasta']:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    # Take first two underscore-separated parts (GCF_XXXXXXX)
    parts = name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return name


def fragment_genome(
    seq: str,
    fragment_len: int = 500,
    stride: int = 250,
    max_fragments: int = 20000,
    min_gc: float = 0.0,
    max_gc: float = 1.0,
) -> list:
    """Fragment a genome sequence into overlapping windows.

    Parameters
    ----------
    seq : str
        Full genome sequence (uppercase ACGT)
    fragment_len : int
        Length of each fragment
    stride : int
        Step between fragment starts
    max_fragments : int
        Maximum fragments per genome (prevents over-representation)
    min_gc, max_gc : float
        GC content bounds for filtering

    Returns
    -------
    fragments : list of (sequence_str, gc_content, position)
    """
    fragments = []
    positions = list(range(0, len(seq) - fragment_len + 1, stride))

    # Shuffle positions to get diverse sampling if max_fragments < available
    random.shuffle(positions)

    for pos in positions:
        if len(fragments) >= max_fragments:
            break

        frag = seq[pos:pos + fragment_len]

        # Skip fragments with N/ambiguous bases
        if any(c not in 'ACGT' for c in frag):
            continue

        # Compute GC content
        gc = (frag.count('G') + frag.count('C')) / len(frag)

        # Filter by GC range
        if gc < min_gc or gc > max_gc:
            continue

        fragments.append((frag, gc, pos))

    return fragments


def process_all_genomes(args: argparse.Namespace) -> pd.DataFrame:
    """Process all FASTA files in input directory."""

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("\nTo download genomes, run:")
        print("  pip install ncbi-genome-download")
        print("  ncbi-genome-download bacteria \\")
        print("      --assembly-levels complete \\")
        print("      --refseq-categories representative \\")
        print("      --formats fasta \\")
        print(f"      --output-folder {input_dir}/ \\")
        print("      --parallel 4")
        sys.exit(1)

    # Find all FASTA files (handle NCBI directory structure)
    fasta_files = []
    for ext in ['*.fna.gz', '*.fna', '*.fa.gz', '*.fa', '*.fasta.gz', '*.fasta']:
        fasta_files.extend(input_dir.rglob(ext))

    if not fasta_files:
        print(f"Error: No FASTA files found in {input_dir}")
        print("Expected .fna.gz files from ncbi-genome-download")
        sys.exit(1)

    print(f"Found {len(fasta_files)} FASTA files")
    if args.max_genomes > 0 and len(fasta_files) > args.max_genomes:
        random.shuffle(fasta_files)
        fasta_files = fasta_files[:args.max_genomes]
        print(f"  Subsampled to {args.max_genomes} genomes")

    all_records = []
    genome_stats = {}

    for fasta_path in tqdm(fasta_files, desc="Processing genomes"):
        genome_name = extract_genome_name(fasta_path)

        # Read FASTA (handle gzipped)
        try:
            if str(fasta_path).endswith('.gz'):
                handle = gzip.open(fasta_path, 'rt')
            else:
                handle = open(fasta_path, 'r')

            # Concatenate all contigs/chromosomes
            full_seq = ""
            for record in SeqIO.parse(handle, "fasta"):
                full_seq += str(record.seq).upper()
            handle.close()
        except Exception as e:
            print(f"  Warning: Could not read {fasta_path}: {e}")
            continue

        if len(full_seq) < args.fragment_len:
            continue

        # Fragment the genome
        fragments = fragment_genome(
            full_seq,
            fragment_len=args.fragment_len,
            stride=args.stride,
            max_fragments=args.max_per_genome,
            min_gc=args.min_gc,
            max_gc=args.max_gc,
        )

        # Store fragment records
        for seq_str, gc, pos in fragments:
            all_records.append({
                "sequence": seq_str,
                "genome": genome_name,
                "position": pos,
                "gc_content": round(gc, 4),
            })

        genome_stats[genome_name] = {
            "total_len": len(full_seq),
            "n_fragments": len(fragments),
            "mean_gc": sum(f[1] for f in fragments) / max(len(fragments), 1),
        }

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Total genomes:    {len(genome_stats)}")
    print(f"Total fragments:  {len(df):,}")
    print(f"Fragment length:  {args.fragment_len} bp")
    print(f"Stride:           {args.stride} bp")
    print(f"Max per genome:   {args.max_per_genome}")

    if len(df) > 0:
        gc_vals = df['gc_content']
        print(f"\nGC content distribution:")
        print(f"  Mean:   {gc_vals.mean():.3f}")
        print(f"  Std:    {gc_vals.std():.3f}")
        print(f"  Min:    {gc_vals.min():.3f}")
        print(f"  Max:    {gc_vals.max():.3f}")
        print(f"  25%:    {gc_vals.quantile(0.25):.3f}")
        print(f"  75%:    {gc_vals.quantile(0.75):.3f}")

        genome_counts = df['genome'].value_counts()
        print(f"\nFragments per genome:")
        print(f"  Mean:   {genome_counts.mean():.0f}")
        print(f"  Std:    {genome_counts.std():.0f}")
        print(f"  Min:    {genome_counts.min()}")
        print(f"  Max:    {genome_counts.max()}")

        # GC distribution by phylum (rough estimate from GC ranges)
        low_gc = (gc_vals < 0.35).sum()
        mid_gc = ((gc_vals >= 0.35) & (gc_vals <= 0.65)).sum()
        high_gc = (gc_vals > 0.65).sum()
        print(f"\nGC-content groups:")
        print(f"  Low  (<35%):  {low_gc:,} ({100*low_gc/len(df):.1f}%)")
        print(f"  Mid  (35-65%): {mid_gc:,} ({100*mid_gc/len(df):.1f}%)")
        print(f"  High (>65%):  {high_gc:,} ({100*high_gc/len(df):.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare expanded bacterial genome dataset for JEPA pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="data/refseq_genomes/",
                        help="Directory containing FASTA files (from ncbi-genome-download)")
    parser.add_argument("--output-csv", default="data/processed/pretrain_sequences_expanded.csv",
                        help="Output CSV path")
    parser.add_argument("--fragment-len", type=int, default=500,
                        help="Fragment length in base pairs")
    parser.add_argument("--stride", type=int, default=250,
                        help="Stride between fragment starts")
    parser.add_argument("--max-per-genome", type=int, default=20000,
                        help="Max fragments per genome (prevents over-representation)")
    parser.add_argument("--max-genomes", type=int, default=0,
                        help="Max genomes to process (0 = all)")
    parser.add_argument("--min-gc", type=float, default=0.0,
                        help="Minimum GC content filter")
    parser.add_argument("--max-gc", type=float, default=1.0,
                        help="Maximum GC content filter")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    df = process_all_genomes(args)

    if len(df) == 0:
        print("Error: No fragments generated!")
        sys.exit(1)

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(df):,} rows)")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nNext: train with")
    print(f"  python scripts/01_pretrain_jepa.py --data-path {output_path}")


if __name__ == "__main__":
    main()