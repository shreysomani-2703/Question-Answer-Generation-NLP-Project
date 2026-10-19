"""
T5InferenceEngine — load a fine-tuned checkpoint and generate questions.

Supports both single-example and batched generation. Prediction dicts
conform to the Phase 3/4/5 integration contract::

    {id, generated_question, reference_question, input_text}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5InferenceEngine:
    """Load a saved T5 checkpoint and generate questions.

    Parameters
    ----------
    model_dir : str
        Root model directory (contains ``checkpoint/``).
    device : torch.device
        Target device.
    """

    def __init__(self, model_dir: str, device: torch.device) -> None:
        checkpoint_path = os.path.join(model_dir, "checkpoint")
        self.tokenizer  = T5Tokenizer.from_pretrained(checkpoint_path)
        self.model      = T5ForConditionalGeneration.from_pretrained(
            checkpoint_path,
        ).to(device)
        self.model.eval()
        self.device = device
        print(f"T5 inference engine loaded from {checkpoint_path}")

    # ── single-example generation ────────────────────────────────────
    def generate_question(
        self, context: str, answer: str, num_beams: int = 4,
    ) -> str:
        """Generate a question for a (context, answer) pair."""
        input_text = f"generate question: {context} answer: {answer}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=64,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # ── batched generation ───────────────────────────────────────────
    def batch_generate(
        self,
        test_data: List[Dict[str, Any]],
        batch_size: int = 8,
        num_beams: int = 4,
    ) -> List[Dict[str, Any]]:
        """Generate predictions for an entire test set.

        Parameters
        ----------
        test_data : list of dict
            Phase 1 T5-ready dicts with ``input_text``, ``target_text``, ``id``.
        batch_size : int
            Number of examples per forward pass.
        num_beams : int
            Beam width for beam search decoding.

        Returns
        -------
        list of dict
            Each dict: ``{id, generated_question, reference_question, input_text}``.
        """
        results: List[Dict[str, Any]] = []

        for i in range(0, len(test_data), batch_size):
            chunk = test_data[i : i + batch_size]
            input_texts = [d["input_text"] for d in chunk]

            enc = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **enc,
                    num_beams=num_beams,
                    max_new_tokens=64,
                    early_stopping=True,
                )

            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for item, q in zip(chunk, decoded):
                results.append({
                    "id":                 item["id"],
                    "generated_question": q,
                    "reference_question": item["target_text"],
                    "input_text":         item["input_text"],
                })

            batch_num = i // batch_size + 1
            if batch_num % 10 == 0:
                print(
                    f"  Generated {min(i + batch_size, len(test_data))}"
                    f"/{len(test_data)} predictions..."
                )

        return results

    # ── persistence ──────────────────────────────────────────────────
    def save_predictions(self, predictions: List[Dict[str, Any]], path: str) -> None:
        """Write predictions to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {path}")
