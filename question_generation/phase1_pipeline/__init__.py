"""
Phase 1 — Shared Data Pipeline
===============================

This package loads SQuAD 1.1, preprocesses texts, performs BIO tagging,
splits the data, and serialises outputs in two formats:

* **LSTM‑ready** (pickle) — used by Phase 2A
* **T5‑ready** (JSON) — used by Phase 2B

Quick start
-----------
>>> from phase1_pipeline import run_pipeline
>>> run_pipeline()
"""

from phase1_pipeline.loader import SquadLoader
from phase1_pipeline.preprocessor import TextPreprocessor
from phase1_pipeline.answer_extractor import AnswerExtractor
from phase1_pipeline.bio_tagger import BIOTagger
from phase1_pipeline.splitter import DatasetSplitter
from phase1_pipeline.pipeline import Phase1Pipeline


__all__ = [
    "Phase1Pipeline",
    "SquadLoader",
    "TextPreprocessor",
    "AnswerExtractor",
    "BIOTagger",
    "DatasetSplitter",
    "run_pipeline",
]


def run_pipeline() -> None:
    """Convenience function: run the full pipeline and save outputs.

    Prints a completion summary with record counts and output paths.
    """
    import os
    from config import LSTM_OUTPUT_DIR, T5_OUTPUT_DIR

    pipeline = Phase1Pipeline()
    data = pipeline.run()
    pipeline.save_outputs(data["lstm_data"], data["t5_data"])

    print("\n" + "=" * 60)
    print("✓ PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  LSTM outputs → {os.path.abspath(LSTM_OUTPUT_DIR)}")
    print(f"  T5 outputs   → {os.path.abspath(T5_OUTPUT_DIR)}")

    for fmt, d in [("LSTM", data["lstm_data"]), ("T5", data["t5_data"])]:
        for split_name, recs in d.items():
            print(f"    {fmt}/{split_name}: {len(recs):,} records")

    # Print first 3 records of each format
    print("\n" + "-" * 60)
    print("Sample LSTM records (first 3):")
    print("-" * 60)
    for rec in data["lstm_data"]["train"][:3]:
        print({
            "id": rec["id"],
            "tokens_preview": rec["tokens"][:10],
            "bio_tags_preview": rec["bio_tags"][:10],
            "answer_text": rec["answer_text"],
            "question": rec["question"][:80],
        })

    print("\n" + "-" * 60)
    print("Sample T5 records (first 3):")
    print("-" * 60)
    for rec in data["t5_data"]["train"][:3]:
        print({
            "id": rec["id"],
            "input_text": rec["input_text"][:120] + "...",
            "target_text": rec["target_text"],
        })
