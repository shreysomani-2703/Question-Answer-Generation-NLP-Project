"""
T5Trainer — fine-tune T5-small for question generation.

Handles the full training loop with:
  • Linear warmup scheduler
  • Gradient clipping
  • BLEU-based early stopping
  • Checkpoint saving (best BLEU)
  • Training log serialisation
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class T5Trainer:
    """Train a T5 model for conditional question generation.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"t5-small"``).
    output_dir : str
        Root directory for checkpoints and logs.
    device : torch.device
        Target device (``cuda`` or ``cpu``).
    """

    def __init__(self, model_name: str, output_dir: str, device: torch.device) -> None:
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._model     = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device     = device
        self.output_dir = output_dir
        self.training_log: List[Dict[str, Any]] = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoint"), exist_ok=True)

        param_count = sum(p.numel() for p in self._model.parameters())
        print(f"Loaded {model_name} — {param_count:,} parameters")
        print(f"Model will be saved to: {output_dir}/checkpoint")

    # ── public properties ────────────────────────────────────────────
    @property
    def tokenizer(self) -> T5Tokenizer:
        return self._tokenizer

    @property
    def model(self) -> T5ForConditionalGeneration:
        return self._model

    # ── validation metrics ───────────────────────────────────────────
    def _compute_val_metrics(self, val_loader: DataLoader) -> tuple:
        """Return ``(avg_val_loss, corpus_bleu)`` over *val_loader*."""
        self._model.eval()
        total_loss = 0.0
        n_batches  = 0
        references: List[List[List[str]]] = []
        hypotheses: List[List[str]]       = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                loss = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                total_loss += loss.item()
                n_batches  += 1

                gen_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    num_beams=1,
                    max_new_tokens=64,
                    early_stopping=True,
                )
                decoded = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for hyp, ref in zip(decoded, batch["target_text"]):
                    hypotheses.append(hyp.lower().split())
                    references.append([ref.lower().split()])

        sf   = SmoothingFunction().method1
        bleu = corpus_bleu(references, hypotheses, smoothing_function=sf)
        return total_loss / max(n_batches, 1), bleu

    # ── training loop ────────────────────────────────────────────────
    def train(
        self,
        train_dataset,
        val_dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        early_stop_patience: int,
    ) -> float:
        """Run the full training loop. Returns best validation BLEU."""

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        optimizer   = AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps,
        )

        best_val_bleu  = 0.0
        best_val_loss  = float("inf")
        patience_count = 0

        # ── Startup banner ───────────────────────────────────────────
        print("=" * 55)
        print("T5 TRAINING CONFIGURATION")
        print(f"  Model:           t5-small")
        print(f"  Epochs:          {epochs}")
        print(f"  Batch size:      {batch_size}")
        print(f"  Learning rate:   {lr}")
        print(f"  Warmup steps:    {warmup_steps}")
        print(f"  Total steps:     {total_steps}")
        print(f"  Early stop pat:  {early_stop_patience}")
        print(f"  Train batches:   {len(train_loader)}")
        print(f"  Val batches:     {len(val_loader)}")
        print(f"  Device:          {self.device}")
        print("=" * 55)

        # ── Epoch loop ───────────────────────────────────────────────
        for epoch in range(epochs):
            self._model.train()
            total_train_loss = 0.0

            for step, batch in enumerate(train_loader):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                optimizer.zero_grad()
                loss = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

                if (step + 1) % 100 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  Step {step+1}/{len(train_loader)} | "
                        f"Loss: {total_train_loss/(step+1):.4f} | "
                        f"LR: {current_lr:.2e}"
                    )

            train_loss = total_train_loss / len(train_loader)
            val_loss, val_bleu = self._compute_val_metrics(val_loader)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val BLEU: {val_bleu:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # ── Checkpoint logic ─────────────────────────────────────
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                best_val_loss = val_loss
                self._model.save_pretrained(
                    os.path.join(self.output_dir, "checkpoint"),
                )
                self._tokenizer.save_pretrained(
                    os.path.join(self.output_dir, "checkpoint"),
                )
                print(f"  → New best BLEU {val_bleu:.4f} — checkpoint saved")
                patience_count = 0
            else:
                patience_count += 1
                print(
                    f"  → No BLEU improvement "
                    f"({patience_count}/{early_stop_patience})"
                )
                if patience_count >= early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch+1}. "
                        f"Best Val BLEU: {best_val_bleu:.4f}"
                    )
                    break

            self.training_log.append({
                "epoch":      epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "val_bleu":   round(val_bleu,   6),
                "lr":         current_lr,
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Persist training log ─────────────────────────────────────
        log_path = os.path.join(self.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Training log saved to {log_path}")

        return best_val_bleu
