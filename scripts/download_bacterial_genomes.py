"""
Download bacterial reference genomes from NCBI E-utilities.

Usage:
    python scripts/download_bacterial_genomes.py
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Edit this mapping if you want to include more accessions.
GENOMES = {
    "NC_000913.3": "Escherichia_coli_K12_MG1655",
    "NC_000964.3": "Bacillus_subtilis_168",
    "NC_002947.4": "Pseudomonas_putida_KT2440",
    "NC_003888.3": "Streptomyces_coelicolor_A3",
    "NC_006814.1": "Lactobacillus_acidophilus_NCFM",
    "NC_007795.1": "Staphylococcus_aureus_NCTC8325",
    "NC_008253.1": "Escherichia_coli_536",
    "NC_009085.1": "Acinetobacter_baumannii_AYE",
}


def build_ncbi_url(accession: str) -> str:
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    )


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def download_genome(
    accession: str,
    name: str,
    output_dir: Path,
    overwrite: bool = False,
    timeout: int = 120,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{accession}_{name}.fasta"

    if output_file.exists() and not overwrite:
        print(f"[skip] {output_file.name} already exists")
        return True

    url = build_ncbi_url(accession)
    req = urllib.request.Request(url, headers={"User-Agent": "DNA-Bacteria-JEPA/1.0"})

    print(f"[downloading] {name} ({accession})")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError("received empty response from NCBI")

        output_file.write_bytes(data)
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"[ok] {output_file.name} ({size_mb:.2f} MB)")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        print(f"[error] {name} ({accession}): {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/raw/bacterial_genomes",
        help="Directory where FASTA files are saved",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between downloads to avoid hammering NCBI",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)

    print("=" * 60)
    print("NCBI Bacterial Genome Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Total accessions: {len(GENOMES)}")
    print()

    success_count = 0
    for idx, (accession, name) in enumerate(GENOMES.items(), start=1):
        print(f"[{idx}/{len(GENOMES)}]")
        if download_genome(
            accession=accession,
            name=name,
            output_dir=output_dir,
            overwrite=args.overwrite,
        ):
            success_count += 1
        print()
        if idx < len(GENOMES):
            time.sleep(max(0.0, args.delay_seconds))

    print("=" * 60)
    print(f"Completed: {success_count}/{len(GENOMES)} genomes downloaded")
    print(f"Saved under: {output_dir.resolve()}")
    print("Next: python scripts/extract_pretraining_sequences.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
