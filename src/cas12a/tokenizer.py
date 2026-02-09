"""
Simple tokenizer utilities for Cas12a sequence tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenizerConfig:
    """Configuration for sequence tokenization."""

    max_tokens: int = 128
    add_special_tokens: bool = True


class Cas12aTokenizer:
    """Character-level DNA tokenizer with simple task-specific helpers."""

    _TOKEN_ORDER = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "A", "C", "G", "T", "N"]

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.token_to_id = {tok: idx for idx, tok in enumerate(self._TOKEN_ORDER)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.cls_id = self.token_to_id["<CLS>"]
        self.sep_id = self.token_to_id["<SEP>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _sanitize(self, sequence: str) -> str:
        return str(sequence).upper().replace("U", "T")

    def _encode_raw(self, sequence: str) -> list[int]:
        seq = self._sanitize(sequence)
        ids: list[int] = []
        for base in seq:
            if base in {"A", "C", "G", "T", "N"}:
                ids.append(self.token_to_id[base])
            else:
                ids.append(self.unk_id)
        return ids

    def _pad_or_truncate(self, ids: list[int], max_tokens: int) -> torch.Tensor:
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        else:
            ids = ids + [self.pad_id] * (max_tokens - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def encode_generic_sequence(
        self,
        sequence: str,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        max_tokens = max_tokens or self.config.max_tokens
        ids = self._encode_raw(sequence)

        if self.config.add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]

        return self._pad_or_truncate(ids, max_tokens=max_tokens)

    def encode_cas12a_sample(
        self,
        guide: str,
        target: str,
        pam: str,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        max_tokens = max_tokens or self.config.max_tokens

        ids = [self.cls_id]
        ids.extend(self._encode_raw(guide))
        ids.append(self.sep_id)
        ids.extend(self._encode_raw(target))
        ids.append(self.sep_id)
        ids.extend(self._encode_raw(pam))
        ids.append(self.sep_id)

        return self._pad_or_truncate(ids, max_tokens=max_tokens)

    def get_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens != self.pad_id
