"""
SQuAD dataset loader.

Downloads SQuAD 1.1 from HuggingFace and normalises every record into a
flat dict that downstream stages can consume without knowing anything
about the original HuggingFace schema.
"""

from __future__ import annotations

from typing import Any, Dict, List

from datasets import load_dataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SQUAD_VERSION, DATA_DIR


class SquadLoader:
    """Load and normalise the SQuAD 1.1 dataset.

    Attributes
    ----------
    dataset_name : str
        HuggingFace dataset identifier (default ``squad``).
    cache_dir : str
        Local directory used by the ``datasets`` library for caching.
    """

    def __init__(
        self,
        dataset_name: str = SQUAD_VERSION,
        cache_dir: str = DATA_DIR,
    ) -> None:
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    # ── public API ───────────────────────────────────────────────────
    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        """Download / load SQuAD and return normalised records.

        Returns
        -------
        dict
            ``{"train": [...], "validation": [...]}`` where each element
            is a dict with keys *id*, *context*, *question*,
            *answer_text*, *answer_start*.
        """
        raw = load_dataset(self.dataset_name, cache_dir=self.cache_dir)

        result: Dict[str, List[Dict[str, Any]]] = {}
        for split_name in ("train", "validation"):
            records = self._normalise_split(raw[split_name])
            result[split_name] = records
            print(f"[SquadLoader] {split_name}: {len(records):,} records loaded")

        return result

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _normalise_split(split) -> List[Dict[str, Any]]:
        """Convert one HF split into a list of flat dicts."""
        records: List[Dict[str, Any]] = []
        for row in split:
            answers = row["answers"]
            # SQuAD stores answers as {"text": [...], "answer_start": [...]}
            answer_text: str = answers["text"][0]
            answer_start: int = answers["answer_start"][0]  # already char offset

            records.append(
                {
                    "id": row["id"],
                    "context": row["context"],
                    "question": row["question"],
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                }
            )
        return records
