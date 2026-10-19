"""
Smoke test for Phase 2B — runs steps 1-6 of train.py only.
Exits before full training begins.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# ── 1. Config & device ──────────────────────────────────────────────
from config import (
    T5_MODEL_NAME,
    T5_MAX_INPUT_LENGTH,
    T5_MAX_TARGET_LENGTH,
    T5_BATCH_SIZE,
    T5_LEARNING_RATE,
    T5_WEIGHT_DECAY,
    T5_EPOCHS,
    T5_WARMUP_RATIO,
    T5_NUM_BEAMS,
    T5_EARLY_STOP_PAT,
    T5_MODEL_DIR,
    T5_OUTPUT_DIR,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 2. Load Phase 1 data ────────────────────────────────────────────
try:
    from phase1_pipeline.pipeline import Phase1Pipeline
    pipeline   = Phase1Pipeline()
    train_data = pipeline.load_t5_data("train")
    val_data   = pipeline.load_t5_data("val")
    test_data  = pipeline.load_t5_data("test")
    print("Loaded data via Phase1Pipeline")
except Exception as e:
    print(f"Phase1Pipeline failed ({e}), loading JSON directly...")
    train_data = json.load(open(os.path.join(T5_OUTPUT_DIR, "train.json")))
    val_data   = json.load(open(os.path.join(T5_OUTPUT_DIR, "val.json")))
    test_data  = json.load(open(os.path.join(T5_OUTPUT_DIR, "test.json")))

print(
    f"Loaded {len(train_data)} train / {len(val_data)} val / "
    f"{len(test_data)} test examples"
)

# ── 3. Verify sample format ─────────────────────────────────────────
print(f"Sample input  : {train_data[0]['input_text'][:100]}...")
print(f"Sample target : {train_data[0]['target_text']}")

# ── 4. Build trainer ────────────────────────────────────────────────
from phase2b_t5.trainer import T5Trainer

trainer = T5Trainer(T5_MODEL_NAME, T5_MODEL_DIR, device)

# ── 5. Build datasets (using only first 100 examples for speed) ─────
from phase2b_t5.dataset import T5QGDataset

print("Tokenizing datasets (small subset for smoke test)...")
train_ds = T5QGDataset(
    train_data[:100], trainer.tokenizer, T5_MAX_INPUT_LENGTH, T5_MAX_TARGET_LENGTH,
)
val_ds = T5QGDataset(
    val_data[:20], trainer.tokenizer, T5_MAX_INPUT_LENGTH, T5_MAX_TARGET_LENGTH,
)
print(f"Datasets ready — train: {len(train_ds)}, val: {len(val_ds)}")

# ── 6. Smoke test — one batch forward pass ──────────────────────────
from torch.utils.data import DataLoader

tmp = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
b   = next(iter(tmp))
del tmp

print(
    f"Batch shapes — input_ids: {b['input_ids'].shape}, "
    f"labels: {b['labels'].shape}"
)

with torch.no_grad():
    out = trainer.model(
        input_ids=b["input_ids"].to(device),
        attention_mask=b["attention_mask"].to(device),
        labels=b["labels"].to(device),
    )

assert not torch.isnan(out.loss), "NaN loss on first batch — check data"
print(f"Smoke test passed — initial loss: {out.loss.item():.4f}")

# ── Config verification ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("T5 TRAINING CONFIGURATION")
print(f"  Model:           {T5_MODEL_NAME}")
print(f"  Epochs:          {T5_EPOCHS}")
print(f"  Batch size:      {T5_BATCH_SIZE}")
print(f"  Learning rate:   {T5_LEARNING_RATE}")
print(f"  Weight decay:    {T5_WEIGHT_DECAY}")
print(f"  Warmup ratio:    {T5_WARMUP_RATIO}")
print(f"  Num beams:       {T5_NUM_BEAMS}")
print(f"  Early stop pat:  {T5_EARLY_STOP_PAT}")
print(f"  Max input len:   {T5_MAX_INPUT_LENGTH}")
print(f"  Max target len:  {T5_MAX_TARGET_LENGTH}")
print(f"  Model dir:       {T5_MODEL_DIR}")
print(f"  Device:          {device}")
print("=" * 55)

# ── Quick generation test ────────────────────────────────────────────
print("\nRunning a quick generation test (untrained model — expect gibberish)...")
gen_ids = trainer.model.generate(
    b["input_ids"][:1].to(device),
    attention_mask=b["attention_mask"][:1].to(device),
    num_beams=4,
    max_new_tokens=64,
    early_stopping=True,
)
sample_q = trainer.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print(f"  Generated (untrained): {sample_q}")
print(f"  Reference:             {train_data[0]['target_text']}")

print("\n✓ ALL SMOKE TESTS PASSED — ready for full training.")
print("  Run:  python -m phase2b_t5.train")
