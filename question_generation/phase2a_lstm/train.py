"""
Phase 2A — Train the LSTM Seq2Seq question generation model.

Main entry point.  Run from the ``question_generation/`` directory::

    python -m phase2a_lstm.train
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys

# ── Path setup ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Reproducibility ──────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────
from config import (
    BATCH_SIZE,
    BEAM_WIDTH,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    EPOCHS,
    GLOVE_PATH,
    HIDDEN_SIZE,
    LEARNING_RATE,
    LSTM_MODEL_DIR,
    LSTM_OUTPUT_DIR,
    MAX_DECODE_LEN,
    MAX_GRAD_NORM,
    NUM_LAYERS,
    TEACHER_FORCING_DECAY,
    TEACHER_FORCING_RATIO_START,
)


def main() -> None:
    # ── 1. Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 2. Load Phase 1 data ─────────────────────────────────────────
    from phase1_pipeline.pipeline import Phase1Pipeline

    pipeline = Phase1Pipeline()
    train_data = pipeline.load_lstm_data("train")
    val_data = pipeline.load_lstm_data("val")
    test_data = pipeline.load_lstm_data("test")
    print(
        f"Loaded {len(train_data)} train / "
        f"{len(val_data)} val / "
        f"{len(test_data)} test"
    )

    # ── 3. Load vocabulary ───────────────────────────────────────────
    vocab_path = os.path.join(LSTM_OUTPUT_DIR, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    embedding_matrix = vocab.get("embedding_matrix", None)

    print(f"Vocabulary size: {len(word2idx):,}")

    # ── 4. Load / build embeddings ───────────────────────────────────
    # Phase 1 stores a zeros placeholder; replace with GloVe if available
    if embedding_matrix is None or embedding_matrix.sum() == 0.0:
        from phase2a_lstm.embeddings import EmbeddingLoader

        embedding_matrix = EmbeddingLoader.load_glove(
            GLOVE_PATH, word2idx, EMBEDDING_DIM
        )

    # ── 5. Build datasets and loaders ────────────────────────────────
    from phase2a_lstm.dataset import LSTMDataset, collate_fn

    train_ds = LSTMDataset(train_data, word2idx)
    val_ds = LSTMDataset(val_data, word2idx)
    test_ds = LSTMDataset(test_data, word2idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── 6. Smoke test — batch shapes ────────────────────────────────
    batch = next(iter(train_loader))
    print("src_ids shape:", batch["src_ids"].shape)
    print("bio_ids shape:", batch["bio_ids"].shape)
    print("tgt_ids shape:", batch["tgt_ids"].shape)

    # ── 7. Build model ───────────────────────────────────────────────
    from phase2a_lstm.model import Decoder, Encoder, Seq2SeqQG

    vocab_size = len(word2idx)
    encoder = Encoder(
        vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
        embedding_matrix,
    )
    decoder = Decoder(
        vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    )
    model = Seq2SeqQG(encoder, decoder, device).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── 8. Smoke test — forward pass ─────────────────────────────────
    src = batch["src_ids"].to(device)
    bio = batch["bio_ids"].to(device)
    tgt = batch["tgt_ids"].to(device)
    lens = batch["src_lens"]

    out = model(src, bio, lens, tgt, teacher_forcing_ratio=0.5)
    print("Forward pass output shape:", out.shape)

    # ── 9. Optimizer & criterion ─────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ── 10. Training configuration verification ─────────────────────
    print("=" * 50)
    print("TRAINING CONFIGURATION VERIFICATION")
    print(f"  Epochs:                  {EPOCHS}")
    print(f"  Early stop patience:     {EARLY_STOPPING_PATIENCE}")
    print(f"  Teacher forcing start:   {TEACHER_FORCING_RATIO_START}")
    print(f"  Teacher forcing decay:   {TEACHER_FORCING_DECAY}")
    print(f"  Teacher forcing at ep20: {max(0.0, TEACHER_FORCING_RATIO_START - TEACHER_FORCING_DECAY * 20):.2f}")
    print(f"  Learning rate:           {LEARNING_RATE}")
    print(f"  Batch size:              {BATCH_SIZE}")
    print(f"  Hidden size:             {HIDDEN_SIZE}")
    print(f"  Device:                  {device}")
    print(f"  Train examples:          {len(train_data)}")
    print(f"  Val examples:            {len(val_data)}")
    print("=" * 50)

    # ── 11. Train ────────────────────────────────────────────────────
    from phase2a_lstm.trainer import Trainer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    trainer = Trainer(model, optimizer, criterion, device, idx2word, word2idx, scheduler=scheduler)
    trainer.train(
        train_loader,
        val_loader,
        EPOCHS,
        TEACHER_FORCING_RATIO_START,
        TEACHER_FORCING_DECAY,
        LSTM_MODEL_DIR,
    )

    # ── 11. Load best model & generate test predictions ──────────────
    best_path = os.path.join(LSTM_MODEL_DIR, "best_model.pt")
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    model.eval()

    from phase2a_lstm.inference import InferenceEngine

    engine = InferenceEngine(
        model, word2idx, idx2word, device, BEAM_WIDTH, MAX_DECODE_LEN
    )
    predictions = engine.batch_generate(test_data, use_beam=True)
    pred_path = os.path.join(LSTM_MODEL_DIR, "predictions.json")
    engine.save_predictions(predictions, pred_path)

    # ── 12. Sample output ────────────────────────────────────────────
    print()
    print("Sample generated question:", predictions[0]["generated_question"])
    print("Reference question:       ", predictions[0]["reference_question"])

    # ── 13. Summary ──────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("=== Phase 2A Complete ===")
    print(f"Predictions saved to: {pred_path}")
    print(f"Model saved to: {best_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
