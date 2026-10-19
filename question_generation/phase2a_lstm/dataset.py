"""
LSTM dataset and collation utilities.

Converts raw Phase 1 records (dicts with *tokens*, *bio_tags*, *question*, etc.)
into integer‑index sequences suitable for PyTorch DataLoaders.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """Map‑style dataset that converts Phase 1 records to index sequences.

    Parameters
    ----------
    data : list of dict
        Phase 1 records with keys: *tokens*, *bio_tags*, *answer_text*,
        *question*, *id*.
    word2idx : dict
        Word → integer index mapping (with special tokens <PAD>=0,
        <UNK>=1, <SOS>=2, <EOS>=3).
    max_src_len : int
        Maximum source (context) sequence length after truncation.
    max_tgt_len : int
        Maximum target (question) sequence length after truncation.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        word2idx: Dict[str, int],
        max_src_len: int = 512,
        max_tgt_len: int = 64,
    ) -> None:
        super().__init__()
        self.word2idx = word2idx
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # BIO tag vocabulary (built internally)
        self.bio2idx: Dict[str, int] = {
            "O": 0,
            "B-ANS": 1,
            "I-ANS": 2,
            "<PAD>": 3,
        }

        unk_idx = word2idx.get("<UNK>", 1)
        sos_idx = word2idx.get("<SOS>", 2)
        eos_idx = word2idx.get("<EOS>", 3)

        self.src_ids_list: List[List[int]] = []
        self.bio_ids_list: List[List[int]] = []
        self.tgt_ids_list: List[List[int]] = []
        self.ids: List[str] = []

        for rec in data:
            # ── Source (context tokens + BIO tags) ──
            src_ids = [
                word2idx.get(tok.lower(), unk_idx) for tok in rec["tokens"]
            ]
            bio_ids = [
                self.bio2idx.get(tag, 0) for tag in rec["bio_tags"]
            ]

            # Truncate
            src_ids = src_ids[: self.max_src_len]
            bio_ids = bio_ids[: self.max_src_len]

            # ── Target (question) ──
            q_tokens = rec["question"].split()
            tgt_ids = (
                [sos_idx]
                + [word2idx.get(tok.lower(), unk_idx) for tok in q_tokens]
                + [eos_idx]
            )
            tgt_ids = tgt_ids[: self.max_tgt_len]

            self.src_ids_list.append(src_ids)
            self.bio_ids_list.append(bio_ids)
            self.tgt_ids_list.append(tgt_ids)
            self.ids.append(rec["id"])

    # ── Dataset interface ────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.src_ids_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "src_ids": self.src_ids_list[idx],
            "bio_ids": self.bio_ids_list[idx],
            "tgt_ids": self.tgt_ids_list[idx],
            "src_len": len(self.src_ids_list[idx]),
            "tgt_len": len(self.tgt_ids_list[idx]),
            "id": self.ids[idx],
        }


# ── Collation ────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pad a list of dataset items into batched tensors.

    Padding index is **0** for all sequences (``<PAD>``).
    """
    max_src = max(item["src_len"] for item in batch)
    max_tgt = max(item["tgt_len"] for item in batch)

    src_ids_batch: List[List[int]] = []
    bio_ids_batch: List[List[int]] = []
    tgt_ids_batch: List[List[int]] = []
    src_lens: List[int] = []
    tgt_lens: List[int] = []

    for item in batch:
        pad_src = max_src - item["src_len"]
        pad_tgt = max_tgt - item["tgt_len"]

        src_ids_batch.append(item["src_ids"] + [0] * pad_src)
        bio_ids_batch.append(item["bio_ids"] + [0] * pad_src)
        tgt_ids_batch.append(item["tgt_ids"] + [0] * pad_tgt)
        src_lens.append(item["src_len"])
        tgt_lens.append(item["tgt_len"])

    return {
        "src_ids": torch.LongTensor(src_ids_batch),
        "bio_ids": torch.LongTensor(bio_ids_batch),
        "tgt_ids": torch.LongTensor(tgt_ids_batch),
        "src_lens": torch.LongTensor(src_lens),
        "tgt_lens": torch.LongTensor(tgt_lens),
    }
