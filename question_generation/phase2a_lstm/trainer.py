"""
Training loop for the LSTM Seq2Seq question generation model.

Handles:
- Epoch training with teacher forcing
- Validation with BLEU scoring via NLTK
- Early stopping and best‑model checkpointing
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EARLY_STOPPING_PATIENCE, MAX_DECODE_LEN, MAX_GRAD_NORM


class Trainer:
    """Training orchestrator for :class:`Seq2SeqQG`.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Cross‑entropy loss with ``ignore_index=0`` (PAD).
    device : torch.device
    idx2word : dict
        Index → word mapping for decoding during evaluation.
    word2idx : dict
        Word → index mapping (used to look up special tokens).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        idx2word: Dict[int, str],
        word2idx: Dict[str, int],
        scheduler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.scheduler = scheduler
        self.eos_idx = word2idx.get("<EOS>", 3)
        self.pad_idx = word2idx.get("<PAD>", 0)
        self.sos_idx = word2idx.get("<SOS>", 2)

    # ── Training epoch ───────────────────────────────────────────────
    def train_epoch(
        self, loader: torch.utils.data.DataLoader, tf_ratio: float
    ) -> float:
        """Run one training epoch.

        Returns
        -------
        float
            Mean training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="  Train", leave=False):
            src = batch["src_ids"].to(self.device)
            bio = batch["bio_ids"].to(self.device)
            tgt = batch["tgt_ids"].to(self.device)
            src_lens = batch["src_lens"]

            self.optimizer.zero_grad()

            # Forward: output shape (B, tgt_len-1, vocab)
            output = self.model(src, bio, src_lens, tgt, tf_ratio)

            # Target: skip SOS column → (B, tgt_len-1)
            target = tgt[:, 1:]

            # Reshape for cross‑entropy
            output_flat = output.reshape(-1, output.size(-1))
            target_flat = target.reshape(-1)

            loss = self.criterion(output_flat, target_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), MAX_GRAD_NORM
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── Evaluation ───────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(
        self, loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Evaluate on a validation/test set.

        Returns
        -------
        (mean_loss, corpus_bleu_score)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        hypotheses: List[List[str]] = []
        references: List[List[List[str]]] = []

        for batch in tqdm(loader, desc="  Eval ", leave=False):
            src = batch["src_ids"].to(self.device)
            bio = batch["bio_ids"].to(self.device)
            tgt = batch["tgt_ids"].to(self.device)
            src_lens = batch["src_lens"]

            # Teacher forcing = 0 for evaluation
            output = self.model(src, bio, src_lens, tgt, teacher_forcing_ratio=0.0)
            target = tgt[:, 1:]

            output_flat = output.reshape(-1, output.size(-1))
            target_flat = target.reshape(-1)

            loss = self.criterion(output_flat, target_flat)
            total_loss += loss.item()
            n_batches += 1

            # Greedy decode for BLEU (from model output logits)
            preds = output.argmax(dim=-1)  # (B, tgt_len-1)

            for i in range(preds.size(0)):
                # Hypothesis
                hyp_tokens: List[str] = []
                for t in range(preds.size(1)):
                    idx = preds[i, t].item()
                    if idx == self.eos_idx:
                        break
                    word = self.idx2word.get(idx, "<UNK>")
                    if word not in ("<PAD>", "<SOS>", "<EOS>"):
                        hyp_tokens.append(word)
                hypotheses.append(hyp_tokens)

                # Reference (from target ids, skip PAD/SOS/EOS)
                ref_tokens: List[str] = []
                for t in range(target.size(1)):
                    idx = target[i, t].item()
                    if idx == self.eos_idx:
                        break
                    word = self.idx2word.get(idx, "<UNK>")
                    if word not in ("<PAD>", "<SOS>", "<EOS>"):
                        ref_tokens.append(word)
                references.append([ref_tokens])

        mean_loss = total_loss / max(n_batches, 1)

        smoothie = SmoothingFunction().method1
        bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)

        return mean_loss, bleu

    # ── Full training loop ───────────────────────────────────────────
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        tf_start: float,
        tf_decay: float,
        save_dir: str,
    ) -> float:
        """Run the full training loop with early stopping.

        Returns
        -------
        float
            Best validation BLEU achieved.
        """
        os.makedirs(save_dir, exist_ok=True)

        tf_ratio = tf_start
        best_val_bleu = -1.0
        patience_counter = 0
        log: List[Dict[str, Any]] = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, tf_ratio)
            val_loss, val_bleu = self.evaluate(val_loader)

            # Step LR scheduler on val_loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            tf_ratio = max(0.0, tf_ratio - tf_decay)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val BLEU: {val_bleu:.4f} | "
                f"TF: {tf_ratio:.2f} | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping monitors val_bleu (higher is better)
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, "best_model.pt"),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\u2192 Early stopping triggered at epoch {epoch + 1} (best Val BLEU: {best_val_bleu:.4f})")
                    break

            log.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu": val_bleu,
                    "lr": current_lr,
                }
            )

        # Persist training log
        log_path = os.path.join(save_dir, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved to {log_path}")

        return best_val_bleu
