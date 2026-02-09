"""
PyTorch datasets for Cas12a training.
YOU PROVIDE: CSV files with your data in data/processed/
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class BacterialGenomeDataset(Dataset):
    """
    Pretraining dataset for bacterial genome sequences.

    YOU PROVIDE: data/processed/pretrain_sequences.csv with columns:
    - sequence: DNA string (512 bp windows)
    - genome: genome identifier
    - position: start position in genome
    """

    def __init__(self, csv_path: str, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

        print(f"Loaded {len(self.df)} sequences from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sequence = self.df.iloc[idx]["sequence"]
        tokens = self.tokenizer.encode_generic_sequence(sequence, max_tokens=128)
        return tokens


class Cas12aEfficiencyDataset(Dataset):
    """
    Fine-tuning dataset for Cas12a efficiency prediction.

    YOU PROVIDE: data/processed/cas12a_efficiency.csv with columns:
    - crRNA_seq: guide RNA sequence
    - target_seq: target DNA sequence
    - PAM: PAM sequence (TTTV)
    - efficiency_normalized: normalized efficiency score
    - source: dataset source identifier
    """

    def __init__(self, csv_path: str, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

        print(f"Loaded {len(self.df)} guides from {csv_path}")
        print(f"Sources: {self.df['source'].unique()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        tokens = self.tokenizer.encode_cas12a_sample(
            guide=row["crRNA_seq"],
            target=row["target_seq"],
            pam=row["PAM"],
        )

        efficiency = torch.tensor([row["efficiency_normalized"]], dtype=torch.float32)

        return {
            "tokens": tokens,
            "efficiency": efficiency,
            "source": row["source"],
        }


class RPAPrimerDataset(Dataset):
    """
    Dataset for RPA primer design.

    YOU PROVIDE: data/processed/rpa_primers.csv with columns:
    - forward_primer: forward primer sequence
    - reverse_primer: reverse primer sequence
    - target_seq: target amplicon sequence
    - amplifies: binary label (0 or 1)
    - time_to_positive: time in minutes (optional)
    """

    def __init__(self, csv_path: str, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

        print(f"Loaded {len(self.df)} primer pairs from {csv_path}")
        print(f"Positive rate: {self.df['amplifies'].mean():.2%}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        fwd_tokens = self.tokenizer.encode_generic_sequence(row["forward_primer"])
        rev_tokens = self.tokenizer.encode_generic_sequence(row["reverse_primer"])
        tgt_tokens = self.tokenizer.encode_generic_sequence(row["target_seq"])

        label = torch.tensor([row["amplifies"]], dtype=torch.float32)

        return {
            "forward": fwd_tokens,
            "reverse": rev_tokens,
            "target": tgt_tokens,
            "label": label,
        }


def cas12a_collate(batch):
    """Collate function for Cas12a datasets."""
    tokens = torch.stack([item["tokens"] for item in batch])
    efficiency = torch.stack([item["efficiency"] for item in batch])

    return {
        "tokens": tokens,
        "efficiency": efficiency,
    }


def rpa_collate(batch):
    """Collate function for RPA datasets."""
    return {
        "forward": torch.stack([item["forward"] for item in batch]),
        "reverse": torch.stack([item["reverse"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
