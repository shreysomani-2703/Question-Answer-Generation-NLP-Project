"""
Answer candidate extraction via NER and noun‑chunk key‑phrases.

Also provides a helper to align existing SQuAD answers to the
same span‑dict format so every downstream consumer sees a
uniform schema.
"""

from __future__ import annotations

from typing import Any, Dict, List

import spacy


class AnswerExtractor:
    """Extract potential answer spans from a context string.

    Uses spaCy named‑entity recognition and noun chunks to build
    a list of candidate spans.
    """

    # Entity types we care about
    _TARGET_ENTS = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT"}

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Load the spaCy model *with* NER enabled.

        Parameters
        ----------
        spacy_model : str
            Name of the spaCy model to load.
        """
        self.nlp = spacy.load(spacy_model)
        self.nlp.max_length = 2_000_000

    # ── public API ───────────────────────────────────────────────────
    def extract_candidates(self, context: str) -> List[Dict[str, Any]]:
        """Return de‑duplicated candidate answer spans.

        Parameters
        ----------
        context : str
            The passage to extract candidates from.

        Returns
        -------
        list of dict
            Each dict has keys *text*, *start_char*, *end_char*, *type*.
        """
        doc = self.nlp(context)
        spans: List[Dict[str, Any]] = []

        # ── Named entities ───────────────────────────────────────────
        for ent in doc.ents:
            if ent.label_ in self._TARGET_ENTS:
                spans.append(
                    {
                        "text": ent.text,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "type": ent.label_,
                    }
                )

        # ── Noun‑chunk key‑phrases (> 1 token) ──────────────────────
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:
                spans.append(
                    {
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "type": "KEYPHRASE",
                    }
                )

        return self._deduplicate(spans)

    def align_squad_answer(
        self,
        context: str,
        answer_text: str,
        answer_start: int,
    ) -> Dict[str, Any]:
        """Wrap a known SQuAD answer into the candidate‑span schema.

        Parameters
        ----------
        context : str
            The full context passage.
        answer_text : str
            The answer string.
        answer_start : int
            Character offset of the answer inside *context*.

        Returns
        -------
        dict
            Keys: *text*, *start_char*, *end_char*, *type*.
        """
        end_char = answer_start + len(answer_text)
        # Sanity check: make sure the slice matches
        extracted = context[answer_start:end_char]
        if extracted != answer_text:
            # Try to find the answer in context as fallback
            idx = context.find(answer_text)
            if idx != -1:
                answer_start = idx
                end_char = answer_start + len(answer_text)

        return {
            "text": answer_text,
            "start_char": answer_start,
            "end_char": end_char,
            "type": "SQUAD_ANSWER",
        }

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _deduplicate(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping spans, keeping the longest one."""
        if not spans:
            return spans

        # Sort by start_char, then by descending length
        spans.sort(key=lambda s: (s["start_char"], -(s["end_char"] - s["start_char"])))
        kept: List[Dict[str, Any]] = [spans[0]]

        for span in spans[1:]:
            prev = kept[-1]
            # If current span overlaps with the last kept span, skip it
            # (the longer one was kept because of the sort order)
            if span["start_char"] < prev["end_char"]:
                # Keep the longer span
                if (span["end_char"] - span["start_char"]) > (
                    prev["end_char"] - prev["start_char"]
                ):
                    kept[-1] = span
            else:
                kept.append(span)

        return kept
