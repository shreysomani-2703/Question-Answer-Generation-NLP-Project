"""
Phase 1 orchestration pipeline.

Ties together loading, preprocessing, BIO tagging, splitting, and
output serialisation.  Exposes ``load_lstm_data`` and ``load_t5_data``
as the **stable integration points** for Phase 2A / 2B.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    LSTM_OUTPUT_DIR,
    MAX_VOCAB_SIZE,
    MIN_WORD_FREQ,
    GLOVE_DIM,
    T5_OUTPUT_DIR,
    T5_PROMPT_TEMPLATE,
)
from phase1_pipeline.loader import SquadLoader
from phase1_pipeline.preprocessor import TextPreprocessor
from phase1_pipeline.answer_extractor import AnswerExtractor
from phase1_pipeline.bio_tagger import BIOTagger
from phase1_pipeline.splitter import DatasetSplitter


class Phase1Pipeline:
    """End‑to‑end Phase 1 data pipeline.

    Usage
    -----
    >>> p = Phase1Pipeline()
    >>> data = p.run()            # returns {"lstm_data": …, "t5_data": …}
    >>> p.save_outputs(data["lstm_data"], data["t5_data"])

    After saving, future phases load data via:
    >>> train = p.load_lstm_data("train")   # Phase 2A
    >>> train = p.load_t5_data("train")     # Phase 2B
    """

    def __init__(self) -> None:
        self.loader = SquadLoader()
        self.preprocessor = TextPreprocessor()
        self.answer_extractor = AnswerExtractor()
        self.bio_tagger = BIOTagger()
        self.splitter = DatasetSplitter()

    # ── main entry point ─────────────────────────────────────────────
    def run(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Execute the full pipeline and return structured data.

        Returns
        -------
        dict
            ``{"lstm_data": {"train": [...], "val": [...], "test": [...]},
              "t5_data":  {"train": [...], "val": [...], "test": [...]}}``.
        """
        # 1 ── Load
        print("=" * 60)
        print("STEP 1: Loading SQuAD dataset")
        print("=" * 60)
        raw = self.loader.load()
        # Combine train + validation for a unified processing pass,
        # then re‑split with our own ratios.
        all_records: List[Dict[str, Any]] = raw["train"] + raw["validation"]
        print(f"Total records: {len(all_records):,}")

        # 2 ── Preprocess
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing text")
        print("=" * 60)
        all_records = self.preprocessor.process_batch(all_records)

        # 3 ── Align SQuAD answers
        print("\n" + "=" * 60)
        print("STEP 3: Aligning answer spans")
        print("=" * 60)
        for rec in tqdm(all_records, desc="Answer alignment", unit="rec"):
            rec["answer_span"] = self.answer_extractor.align_squad_answer(
                context=rec["context"],
                answer_text=rec["answer_text"],
                answer_start=rec["answer_start"],
            )

        # 4 ── BIO tagging
        print("\n" + "=" * 60)
        print("STEP 4: BIO tagging")
        print("=" * 60)
        all_records = self.bio_tagger.tag_batch(all_records)

        # 5 ── Split
        print("\n" + "=" * 60)
        print("STEP 5: Splitting dataset")
        print("=" * 60)
        splits = self.splitter.split(all_records)

        # 6 ── Format for both down‑stream consumers
        print("\n" + "=" * 60)
        print("STEP 6: Formatting outputs")
        print("=" * 60)
        lstm_data: Dict[str, List[Dict[str, Any]]] = {}
        t5_data: Dict[str, List[Dict[str, Any]]] = {}

        for split_name, recs in splits.items():
            lstm_data[split_name] = self._format_lstm(recs)
            t5_data[split_name] = self._format_t5(recs)
            print(
                f"  {split_name} → LSTM: {len(lstm_data[split_name]):,}  "
                f"T5: {len(t5_data[split_name]):,}"
            )

        return {"lstm_data": lstm_data, "t5_data": t5_data}

    # ── serialisation ────────────────────────────────────────────────
    def save_outputs(
        self,
        lstm_data: Dict[str, List[Dict[str, Any]]],
        t5_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Persist LSTM and T5 data to disk.

        Parameters
        ----------
        lstm_data : dict
            ``{"train": [...], "val": [...], "test": [...]}``.
        t5_data : dict
            ``{"train": [...], "val": [...], "test": [...]}``.
        """
        # ── LSTM outputs ─────────────────────────────────────────────
        os.makedirs(LSTM_OUTPUT_DIR, exist_ok=True)
        for split_name, records in lstm_data.items():
            path = os.path.join(LSTM_OUTPUT_DIR, f"{split_name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(records, f)
            print(f"[save] {path}  ({len(records):,} records)")

        # Build and save vocabulary
        vocab = self._build_vocab(lstm_data)
        vocab_path = os.path.join(LSTM_OUTPUT_DIR, "vocab.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"[save] {vocab_path}  (vocab_size={len(vocab['word2idx']):,})")

        # ── T5 outputs ───────────────────────────────────────────────
        os.makedirs(T5_OUTPUT_DIR, exist_ok=True)
        for split_name, records in t5_data.items():
            path = os.path.join(T5_OUTPUT_DIR, f"{split_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"[save] {path}  ({len(records):,} records)")

    # ── integration points (DO NOT CHANGE SIGNATURES) ────────────────
    @staticmethod
    def load_lstm_data(split: str) -> List[Dict[str, Any]]:
        """Load LSTM‑ready data for a given split.

        Parameters
        ----------
        split : str
            One of ``"train"``, ``"val"``, ``"test"``.

        Returns
        -------
        list of dict
            Each dict has keys: *tokens*, *bio_tags*, *answer_text*,
            *question*, *id*.
        """
        path = os.path.join(LSTM_OUTPUT_DIR, f"{split}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_t5_data(split: str) -> List[Dict[str, Any]]:
        """Load T5‑ready data for a given split.

        Parameters
        ----------
        split : str
            One of ``"train"``, ``"val"``, ``"test"``.

        Returns
        -------
        list of dict
            Each dict has keys: *input_text*, *target_text*, *id*.
        """
        path = os.path.join(T5_OUTPUT_DIR, f"{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── private formatters ───────────────────────────────────────────
    @staticmethod
    def _format_lstm(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert processed records into the LSTM consumer schema."""
        out: List[Dict[str, Any]] = []
        for rec in records:
            out.append(
                {
                    "tokens": rec["processed_context"]["tokens"],
                    "bio_tags": rec["bio_tags"],
                    "answer_text": rec["answer_text"],
                    "question": rec["question"],
                    "id": rec["id"],
                }
            )
        return out

    @staticmethod
    def _format_t5(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert processed records into the T5 consumer schema."""
        out: List[Dict[str, Any]] = []
        for rec in records:
            input_text = T5_PROMPT_TEMPLATE.format(
                context=rec["context"],
                answer=rec["answer_text"],
            )
            out.append(
                {
                    "input_text": input_text,
                    "target_text": rec["question"],
                    "id": rec["id"],
                }
            )
        return out

    # ── vocabulary builder ───────────────────────────────────────────
    @staticmethod
    def _build_vocab(
        lstm_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Build word→idx mapping and a placeholder embedding matrix.

        Only words appearing ≥ ``MIN_WORD_FREQ`` times (across train
        split) are included, up to ``MAX_VOCAB_SIZE`` entries.

        Returns
        -------
        dict
            Keys: *word2idx*, *idx2word*, *embedding_matrix*.
        """
        counter: Counter = Counter()
        for rec in lstm_data.get("train", []):
            for tok in rec["tokens"]:
                counter[tok.lower()] += 1

        # Special tokens
        word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        idx = len(word2idx)

        for word, freq in counter.most_common(MAX_VOCAB_SIZE):
            if freq < MIN_WORD_FREQ:
                break
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1

        idx2word: Dict[int, str] = {v: k for k, v in word2idx.items()}

        # Placeholder embedding matrix (filled in Phase 2A with GloVe)
        vocab_size = len(word2idx)
        embedding_matrix = np.zeros((vocab_size, GLOVE_DIM), dtype=np.float32)

        print(f"[vocab] {vocab_size:,} words (min_freq={MIN_WORD_FREQ})")

        return {
            "word2idx": word2idx,
            "idx2word": idx2word,
            "embedding_matrix": embedding_matrix,
        }
