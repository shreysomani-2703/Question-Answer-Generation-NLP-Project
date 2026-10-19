"""
Dataset splitting with optional stratification by question type.

Produces reproducible train / val / test partitions from a flat list
of records.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED


class DatasetSplitter:
    """Split a list of records into train / val / test subsets.

    Tries to stratify by question type (wh‑word); falls back to a
    plain random split if stratification fails (e.g. too few samples
    per class).
    """

    _WH_PATTERN = re.compile(
        r"^(who|what|when|where|why|which|how|whom|whose)\b",
        re.IGNORECASE,
    )

    def split(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Partition *records* into train / val / test.

        Parameters
        ----------
        records : list of dict
            Flat list produced by earlier pipeline stages.

        Returns
        -------
        dict
            ``{"train": [...], "val": [...], "test": [...]}``.
        """
        labels = self._extract_question_types(records)
        indices = np.arange(len(records))

        # val + test fraction
        val_test_frac = VAL_SPLIT + TEST_SPLIT
        # relative fraction of test within (val + test)
        test_relative = TEST_SPLIT / val_test_frac if val_test_frac > 0 else 0.5

        try:
            train_idx, valtest_idx = train_test_split(
                indices,
                test_size=val_test_frac,
                random_state=RANDOM_SEED,
                stratify=labels,
            )
            valtest_labels = [labels[i] for i in valtest_idx]
            val_idx, test_idx = train_test_split(
                valtest_idx,
                test_size=test_relative,
                random_state=RANDOM_SEED,
                stratify=valtest_labels,
            )
            print("[DatasetSplitter] Stratified split succeeded.")
        except ValueError:
            # Fallback: non‑stratified
            print("[DatasetSplitter] Stratification failed — using random split.")
            train_idx, valtest_idx = train_test_split(
                indices,
                test_size=val_test_frac,
                random_state=RANDOM_SEED,
            )
            val_idx, test_idx = train_test_split(
                valtest_idx,
                test_size=test_relative,
                random_state=RANDOM_SEED,
            )

        result = {
            "train": [records[i] for i in train_idx],
            "val": [records[i] for i in val_idx],
            "test": [records[i] for i in test_idx],
        }

        for name, recs in result.items():
            print(f"  {name}: {len(recs):,} records")

        return result

    # ── helpers ───────────────────────────────────────────────────────
    def _extract_question_types(
        self, records: List[Dict[str, Any]]
    ) -> List[str]:
        """Return a wh‑word label for each record's question."""
        types: List[str] = []
        for rec in records:
            q = rec.get("question", "")
            m = self._WH_PATTERN.match(q.strip())
            types.append(m.group(1).lower() if m else "other")
        return types
