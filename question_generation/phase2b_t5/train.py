"""
Phase 2B — T5 fine-tuning entry point.

Run from the ``question_generation/`` directory::

    python -m phase2b_t5.train

Steps:
  1. Load Phase 1 data (pipeline → JSON fallback)
  2. Build T5Trainer + pre-tokenised datasets
  3. Smoke-test one forward pass
  4. Full training with early stopping
  5. Run inference on test set
  6. Save predictions
"""

import os
import sys
import json

# ── Ensure project root is on sys.path ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# ── 5. Build datasets ───────────────────────────────────────────────
from phase2b_t5.dataset import T5QGDataset

print("Tokenizing datasets (this takes ~1-2 minutes)...")
train_ds = T5QGDataset(
    train_data, trainer.tokenizer, T5_MAX_INPUT_LENGTH, T5_MAX_TARGET_LENGTH,
)
val_ds = T5QGDataset(
    val_data, trainer.tokenizer, T5_MAX_INPUT_LENGTH, T5_MAX_TARGET_LENGTH,
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

# ── 7. Train ────────────────────────────────────────────────────────
best_bleu = trainer.train(
    train_ds,
    val_ds,
    epochs=T5_EPOCHS,
    batch_size=T5_BATCH_SIZE,
    lr=T5_LEARNING_RATE,
    weight_decay=T5_WEIGHT_DECAY,
    warmup_ratio=T5_WARMUP_RATIO,
    early_stop_patience=T5_EARLY_STOP_PAT,
)

# ── 8. Inference on test set ────────────────────────────────────────
from phase2b_t5.inference import T5InferenceEngine

engine = T5InferenceEngine(T5_MODEL_DIR, device)
print("Generating test predictions...")
predictions = engine.batch_generate(
    test_data,
    batch_size=T5_BATCH_SIZE,
    num_beams=T5_NUM_BEAMS,
)
pred_path = os.path.join(T5_MODEL_DIR, "predictions.json")
engine.save_predictions(predictions, pred_path)

# ── 9. Sample predictions ───────────────────────────────────────────
for i in range(min(3, len(predictions))):
    print(f"\nSample {i+1}:")
    print(f"  Generated : {predictions[i]['generated_question']}")
    print(f"  Reference : {predictions[i]['reference_question']}")

# ── 10. Final summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 2B COMPLETE")
print(f"  Best Val BLEU : {best_bleu:.4f}")
print(f"  Predictions   : {pred_path}")
print(f"  Model         : {T5_MODEL_DIR}checkpoint/")
print(f"  Training log  : {T5_MODEL_DIR}training_log.json")
print("=" * 55)
