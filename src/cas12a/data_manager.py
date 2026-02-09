"""
Data splitting and normalization helpers for Cas12a datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """Container for train/validation/test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class Cas12aDataManager:
    """Utility methods for dataset preprocessing and leakage-safe splits."""

    @staticmethod
    def robust_normalize(values: pd.Series) -> pd.Series:
        values = pd.to_numeric(values, errors="coerce")
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            return (values - median).fillna(0.0)
        return ((values - median) / iqr).fillna(0.0)

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
        group_col: str = "crRNA_seq",
    ) -> SplitResult:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in (0, 1).")
        if not 0 < val_size < 1:
            raise ValueError("val_size must be in (0, 1).")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1.")
        if df.empty:
            raise ValueError("Cannot split an empty DataFrame.")

        rng = np.random.default_rng(seed)

        if group_col in df.columns:
            groups = df[group_col].astype(str).unique().tolist()
            rng.shuffle(groups)
            total = len(groups)

            n_test = int(round(total * test_size))
            n_val = int(round(total * val_size))

            if total >= 3:
                n_test = max(1, n_test)
                n_val = max(1, n_val)

            if n_test + n_val >= total:
                n_test = max(1, total // 3)
                n_val = max(1, total // 3)
                if n_test + n_val >= total:
                    n_val = max(1, total - n_test - 1)

            test_groups = set(groups[:n_test])
            val_groups = set(groups[n_test : n_test + n_val])

            test_mask = df[group_col].astype(str).isin(test_groups)
            val_mask = df[group_col].astype(str).isin(val_groups)
            train_mask = ~(test_mask | val_mask)

            train_df = df.loc[train_mask]
            val_df = df.loc[val_mask]
            test_df = df.loc[test_mask]
        else:
            indices = np.arange(len(df))
            rng.shuffle(indices)

            n_test = int(round(len(df) * test_size))
            n_val = int(round(len(df) * val_size))

            test_idx = indices[:n_test]
            val_idx = indices[n_test : n_test + n_val]
            train_idx = indices[n_test + n_val :]

            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            test_df = df.iloc[test_idx]

        return SplitResult(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )
