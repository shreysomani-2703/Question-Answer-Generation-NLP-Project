"""
Inference engine for the LSTM Seq2Seq question generation model.

Supports greedy decoding and beam search with length penalty.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class InferenceEngine:
    """Generate questions from a trained :class:`Seq2SeqQG` model.

    Parameters
    ----------
    model : nn.Module
        Trained Seq2SeqQG model (in eval mode).
    word2idx : dict
    idx2word : dict
    device : torch.device
    beam_width : int
    max_len : int
        Maximum number of decoding steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        device: torch.device,
        beam_width: int = 4,
        max_len: int = 64,
    ) -> None:
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.beam_width = beam_width
        self.max_len = max_len

        self.sos_idx = word2idx.get("<SOS>", 2)
        self.eos_idx = word2idx.get("<EOS>", 3)
        self.pad_idx = word2idx.get("<PAD>", 0)
        self.unk_idx = word2idx.get("<UNK>", 1)

    # ── helpers ──────────────────────────────────────────────────────
    def _ids_to_text(self, ids: List[int]) -> str:
        """Convert a list of token ids to a string, skipping special tokens."""
        skip = {self.pad_idx, self.sos_idx, self.eos_idx}
        words = [
            self.idx2word.get(i, "<UNK>") for i in ids if i not in skip
        ]
        return " ".join(words)

    # ── greedy decode ────────────────────────────────────────────────
    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        bio_ids: torch.Tensor,
        src_len: torch.Tensor,
    ) -> str:
        """Greedily decode a single example.

        Parameters
        ----------
        src_ids : LongTensor ``(L,)``  (1‑D, no batch dim)
        bio_ids : LongTensor ``(L,)``
        src_len : LongTensor scalar or ``(1,)``

        Returns
        -------
        str
            Generated question string.
        """
        self.model.eval()

        # Add batch dimension
        src_ids = src_ids.unsqueeze(0).to(self.device)
        bio_ids = bio_ids.unsqueeze(0).to(self.device)
        if src_len.dim() == 0:
            src_len = src_len.unsqueeze(0)

        encoder_outputs, hidden, cell = self.model.encoder(
            src_ids, bio_ids, src_len
        )

        input_token = torch.LongTensor([self.sos_idx]).to(self.device)
        generated_ids: List[int] = []

        for _ in range(self.max_len):
            logits, hidden, cell, _ = self.model.decoder(
                input_token, encoder_outputs, hidden, cell
            )
            next_id = logits.argmax(dim=-1).item()

            if next_id == self.eos_idx:
                break

            generated_ids.append(next_id)
            input_token = torch.LongTensor([next_id]).to(self.device)

        return self._ids_to_text(generated_ids)

    # ── beam decode ──────────────────────────────────────────────────
    @torch.no_grad()
    def beam_decode(
        self,
        src_ids: torch.Tensor,
        bio_ids: torch.Tensor,
        src_len: torch.Tensor,
    ) -> str:
        """Beam search decoding for a single example.

        Parameters
        ----------
        src_ids : LongTensor ``(L,)``
        bio_ids : LongTensor ``(L,)``
        src_len : LongTensor scalar or ``(1,)``

        Returns
        -------
        str
            Best question hypothesis found by beam search.
        """
        self.model.eval()

        # Add batch dimension
        src_ids = src_ids.unsqueeze(0).to(self.device)
        bio_ids = bio_ids.unsqueeze(0).to(self.device)
        if src_len.dim() == 0:
            src_len = src_len.unsqueeze(0)

        encoder_outputs, hidden, cell = self.model.encoder(
            src_ids, bio_ids, src_len
        )

        # Each beam: (log_prob, token_ids, hidden, cell)
        beams: List[Tuple[float, List[int], torch.Tensor, torch.Tensor]] = [
            (0.0, [self.sos_idx], hidden, cell)
        ]
        completed: List[Tuple[float, List[int]]] = []

        for step in range(self.max_len):
            candidates: List[Tuple[float, List[int], torch.Tensor, torch.Tensor]] = []

            for log_prob, token_ids, h, c in beams:
                last_token = torch.LongTensor([token_ids[-1]]).to(self.device)
                logits, new_h, new_c, _ = self.model.decoder(
                    last_token, encoder_outputs, h, c
                )
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                # Top-k next tokens
                topk_log_probs, topk_ids = log_probs.topk(self.beam_width)

                for i in range(self.beam_width):
                    next_id = topk_ids[i].item()
                    new_log_prob = log_prob + topk_log_probs[i].item()
                    new_token_ids = token_ids + [next_id]

                    if next_id == self.eos_idx:
                        # Length penalty: ((5 + len) / 6) ^ 0.7
                        length = len(new_token_ids) - 1  # exclude SOS
                        penalty = ((5.0 + length) / 6.0) ** 0.7
                        completed.append((new_log_prob / penalty, new_token_ids))
                    else:
                        candidates.append(
                            (new_log_prob, new_token_ids, new_h, new_c)
                        )

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[: self.beam_width]

            # Early exit if all beams completed
            if len(completed) >= self.beam_width:
                break

        # If no beam completed, take the best active beam
        if not completed:
            completed = [
                (b[0] / max(len(b[1]) - 1, 1), b[1]) for b in beams
            ]

        # Best hypothesis
        completed.sort(key=lambda x: -x[0])
        best_ids = completed[0][1]

        return self._ids_to_text(best_ids)

    # ── batch generation ─────────────────────────────────────────────
    def batch_generate(
        self,
        test_data: List[Dict[str, Any]],
        use_beam: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate questions for all records in ``test_data``.

        Parameters
        ----------
        test_data : list of dict
            Phase 1 records with keys *tokens*, *bio_tags*, *answer_text*,
            *question*, *id*.
        use_beam : bool
            If True use beam search, otherwise greedy.

        Returns
        -------
        list of dict
            Each dict: ``{id, generated_question, reference_question,
            context, answer}``.
        """
        self.model.eval()
        predictions: List[Dict[str, Any]] = []

        for rec in test_data:
            tokens = rec["tokens"]
            bio_tags = rec["bio_tags"]

            # Build tensors
            src_ids = torch.LongTensor(
                [self.word2idx.get(tok.lower(), self.unk_idx) for tok in tokens]
            )
            bio2idx = {"O": 0, "B-ANS": 1, "I-ANS": 2}
            bio_ids = torch.LongTensor(
                [bio2idx.get(tag, 0) for tag in bio_tags]
            )
            src_len = torch.LongTensor([len(tokens)])

            if use_beam:
                generated = self.beam_decode(src_ids, bio_ids, src_len)
            else:
                generated = self.greedy_decode(src_ids, bio_ids, src_len)

            predictions.append(
                {
                    "id": rec["id"],
                    "generated_question": generated,
                    "reference_question": rec["question"],
                    "context": " ".join(tokens),
                    "answer": rec["answer_text"],
                }
            )

        return predictions

    # ── save predictions ─────────────────────────────────────────────
    @staticmethod
    def save_predictions(predictions: List[Dict[str, Any]], path: str) -> None:
        """Write predictions to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {path}")
