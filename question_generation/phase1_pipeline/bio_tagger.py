"""
BIO sequence tagger for answer spans.

Converts character‑level answer offsets into token‑level BIO tags
(B‑ANS / I‑ANS / O) that an LSTM‑based model can use as copy or
pointer signals.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from tqdm import tqdm


class BIOTagger:
    """Produce BIO tags for tokenised context given answer character offsets."""

    # ── public API ───────────────────────────────────────────────────
    @staticmethod
    def tag(
        tokens: List[str],
        answer_start_char: int,
        answer_end_char: int,
        original_text: str,
    ) -> List[str]:
        """Assign a BIO tag to every token.

        The method reconstructs character positions by walking through
        *original_text* and aligning each token, so it is robust to
        whitespace variations produced by the tokeniser.

        Parameters
        ----------
        tokens : list of str
            Token strings (e.g. from spaCy).
        answer_start_char : int
            Start character offset of the answer in *original_text*.
        answer_end_char : int
            End character offset (exclusive) of the answer.
        original_text : str
            The raw context string that was tokenised.

        Returns
        -------
        list of str
            One tag per token: ``"B-ANS"``, ``"I-ANS"``, or ``"O"``.
        """
        tags: List[str] = []
        cursor = 0  # current character position in original_text
        answer_started = False

        for token in tokens:
            # Find where this token actually starts in the original text
            tok_start = original_text.find(token, cursor)
            if tok_start == -1:
                # Fallback: try case‑insensitive search
                tok_start = original_text.lower().find(token.lower(), cursor)
            if tok_start == -1:
                # Token not found — tag as O and move on
                tags.append("O")
                continue

            tok_end = tok_start + len(token)

            # Check overlap between [tok_start, tok_end) and [answer_start, answer_end)
            if tok_end > answer_start_char and tok_start < answer_end_char:
                if not answer_started:
                    tags.append("B-ANS")
                    answer_started = True
                else:
                    tags.append("I-ANS")
            else:
                if answer_started and tok_start >= answer_end_char:
                    answer_started = False  # we've passed the answer
                tags.append("O")

            cursor = tok_end

        return tags

    # ── batch helpers ────────────────────────────────────────────────
    def tag_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add ``bio_tags`` to each record.

        Each record must already contain:
        - ``processed_context.tokens`` (list of str)
        - ``answer_start`` (int, char offset)
        - ``answer_text`` (str)
        - ``context`` (str, original text)

        Parameters
        ----------
        records : list of dict

        Returns
        -------
        list of dict
            Same records with an added *bio_tags* key.
        """
        tagged: List[Dict[str, Any]] = []
        for rec in tqdm(records, desc="BIO tagging", unit="rec"):
            rec = deepcopy(rec)
            tokens = rec["processed_context"]["tokens"]
            answer_start = rec["answer_start"]
            answer_end = answer_start + len(rec["answer_text"])

            rec["bio_tags"] = self.tag(
                tokens=tokens,
                answer_start_char=answer_start,
                answer_end_char=answer_end,
                original_text=rec["context"],
            )
            tagged.append(rec)
        return tagged
