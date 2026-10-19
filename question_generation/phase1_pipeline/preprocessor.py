"""
Text preprocessing with spaCy.

Provides tokenisation, lemmatisation, sentence segmentation,
and a cleaned string form for every piece of text.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List

import spacy
from tqdm import tqdm


class TextPreprocessor:
    """Preprocess raw text using the spaCy ``en_core_web_sm`` model.

    The processor is intentionally lightweight — it does not modify the
    original character offsets so that downstream BIO tagging can still
    align tokens to answer spans.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Load the spaCy language model.

        Parameters
        ----------
        spacy_model : str
            Name of the spaCy model to load.
        """
        self.nlp = spacy.load(spacy_model, disable=["ner"])
        # raise pipe‐length limit so long contexts don't error out
        self.nlp.max_length = 2_000_000

    # ── single‐text processing ───────────────────────────────────────
    def process(self, text: str) -> Dict[str, Any]:
        """Process a single text string.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        dict
            Keys: *tokens*, *lemmas*, *sentences*, *cleaned*.
        """
        doc = self.nlp(text)

        tokens: List[str] = [tok.text for tok in doc]
        lemmas: List[str] = [tok.lemma_ for tok in doc]
        sentences: List[str] = [sent.text.strip() for sent in doc.sents]
        cleaned: str = self._clean(text)

        return {
            "tokens": tokens,
            "lemmas": lemmas,
            "sentences": sentences,
            "cleaned": cleaned,
        }

    # ── batch processing ─────────────────────────────────────────────
    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add ``processed_context`` and ``processed_question`` to every record.

        Parameters
        ----------
        records : list of dict
            Each dict must contain *context* and *question* keys.

        Returns
        -------
        list of dict
            The same records, augmented with processed fields.
        """
        processed: List[Dict[str, Any]] = []
        for rec in tqdm(records, desc="Preprocessing", unit="rec"):
            rec = deepcopy(rec)
            rec["processed_context"] = self.process(rec["context"])
            rec["processed_question"] = self.process(rec["question"])
            processed.append(rec)
        return processed

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _clean(text: str) -> str:
        """Lowercase, collapse whitespace, keep only alphanumerics + punctuation."""
        text = text.lower()
        # keep letters, digits, standard punctuation, and whitespace
        text = re.sub(r"[^a-z0-9\s.,;:!?'\"\-()/]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
