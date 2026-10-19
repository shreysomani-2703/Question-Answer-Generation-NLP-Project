"""
T5QGDataset — pre-tokenised PyTorch Dataset for question generation.

All tokenisation happens **once** in ``__init__``; ``__getitem__`` is a
pure index-slice with zero tokenisation overhead.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class T5QGDataset(Dataset):
    """Map-style dataset that wraps Phase 1 T5-ready dicts.

    Parameters
    ----------
    data : list of dict
        Each dict must contain ``input_text``, ``target_text``, and ``id``.
    tokenizer : PreTrainedTokenizerBase
        A HuggingFace tokenizer (e.g. ``T5Tokenizer``).
    max_input_length : int
        Maximum number of tokens for the encoder input.
    max_target_length : int
        Maximum number of tokens for the decoder target.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int = 512,
        max_target_length: int = 64,
    ) -> None:
        # ── Tokenize all inputs in a single batch call ───────────────
        model_inputs = tokenizer(
            [d["input_text"] for d in data],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # ── Tokenize all targets in a single batch call ─────────────
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                [d["target_text"] for d in data],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        # Replace pad positions with -100 so the loss ignores them
        label_ids = labels.input_ids.clone()
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        # ── Store pre-computed tensors & metadata ────────────────────
        self.input_ids: torch.Tensor     = model_inputs.input_ids        # (N, max_input_length)
        self.attention_mask: torch.Tensor = model_inputs.attention_mask   # (N, max_input_length)
        self.labels: torch.Tensor        = label_ids                     # (N, max_target_length)
        self.ids: List[str]              = [d["id"] for d in data]
        self.target_texts: List[str]     = [d["target_text"] for d in data]
        self.input_texts: List[str]      = [d["input_text"]  for d in data]

    # ── Dataset protocol ─────────────────────────────────────────────
    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids":      self.input_ids[idx],       # LongTensor
            "attention_mask": self.attention_mask[idx],   # LongTensor
            "labels":         self.labels[idx],           # LongTensor
            "id":             self.ids[idx],              # str
            "target_text":    self.target_texts[idx],     # str
            "input_text":     self.input_texts[idx],      # str
        }
