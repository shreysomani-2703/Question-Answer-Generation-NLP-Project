"""
Dry-run verification script for Phase 2A training fixes.
Checks all 4 verification items without running full training.
"""
import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DRY-RUN VERIFICATION OF PHASE 2A FIXES")
print("=" * 60)

# ── CHECK 1: Config values ──────────────────────────────────────
print("\n[CHECK 1] Config verification block:")
from config import (
    EPOCHS, EARLY_STOPPING_PATIENCE, TEACHER_FORCING_RATIO_START,
    TEACHER_FORCING_DECAY, LEARNING_RATE, BATCH_SIZE, HIDDEN_SIZE,
)

print(f"  Epochs:                  {EPOCHS}")
print(f"  Early stop patience:     {EARLY_STOPPING_PATIENCE}")
print(f"  Teacher forcing start:   {TEACHER_FORCING_RATIO_START}")
print(f"  Teacher forcing decay:   {TEACHER_FORCING_DECAY}")
print(f"  Teacher forcing at ep20: {max(0.0, TEACHER_FORCING_RATIO_START - TEACHER_FORCING_DECAY * 20):.2f}")
print(f"  Learning rate:           {LEARNING_RATE}")
print(f"  Batch size:              {BATCH_SIZE}")
print(f"  Hidden size:             {HIDDEN_SIZE}")

# Validate expected values
errors = []
if EPOCHS != 25:
    errors.append(f"EPOCHS should be 25, got {EPOCHS}")
if EARLY_STOPPING_PATIENCE != 5:
    errors.append(f"EARLY_STOPPING_PATIENCE should be 5, got {EARLY_STOPPING_PATIENCE}")
if TEACHER_FORCING_DECAY != 0.03:
    errors.append(f"TEACHER_FORCING_DECAY should be 0.03, got {TEACHER_FORCING_DECAY}")
if TEACHER_FORCING_RATIO_START != 0.5:
    errors.append(f"TEACHER_FORCING_RATIO_START should be 0.5, got {TEACHER_FORCING_RATIO_START}")

if errors:
    for e in errors:
        print(f"  [FAIL] {e}")
else:
    print("  [OK] All config values correct")

# ── CHECK 2: Trainer print format includes LR column ────────────
print("\n[CHECK 2] Trainer epoch print format:")
from phase2a_lstm.trainer import Trainer

src = inspect.getsource(Trainer.train)
if "LR: {current_lr" in src:
    print("  [OK] Print format includes LR column")
else:
    print("  [FAIL] LR column NOT found in print format")

# ── CHECK 3: Early stopping monitors val_bleu, not val_loss ─────
print("\n[CHECK 3] Early stopping metric:")
if "best_val_bleu" in src and "val_bleu > best_val_bleu" in src:
    print("  [OK] Early stopping monitors val_bleu (higher is better)")
else:
    print("  [FAIL] Early stopping NOT monitoring val_bleu correctly")

if "best_val_loss" in src:
    print("  [FAIL] Still references best_val_loss — needs cleanup")
else:
    print("  [OK] No residual best_val_loss references")

# ── CHECK 4: GloVe coverage diagnostics ─────────────────────────
print("\n[CHECK 4] GloVe coverage verification:")
from phase2a_lstm.embeddings import EmbeddingLoader

emb_src = inspect.getsource(EmbeddingLoader.load_glove)
if "GloVe coverage:" in emb_src and "coverage < 50.0" in emb_src:
    print("  [OK] GloVe coverage percentage is printed with warnings")
else:
    print("  [FAIL] GloVe coverage diagnostics missing or incomplete")

if "GloVe coverage is good" in emb_src:
    print("  [OK] Good coverage confirmation message present")
else:
    print("  [FAIL] Good coverage message missing")

# ── CHECK 5: Scheduler support in Trainer ───────────────────────
print("\n[CHECK 5] LR scheduler integration:")
init_src = inspect.getsource(Trainer.__init__)
if "scheduler" in init_src and "self.scheduler" in init_src:
    print("  [OK] Trainer.__init__ accepts scheduler parameter")
else:
    print("  [FAIL] Trainer.__init__ does NOT accept scheduler")

if "self.scheduler.step" in src:
    print("  [OK] scheduler.step() called in training loop")
else:
    print("  [FAIL] scheduler.step() NOT called in training loop")

# ── CHECK 6: train.py has config verification block ──────────────
print("\n[CHECK 6] train.py startup verification:")
train_path = os.path.join(os.path.dirname(__file__), "phase2a_lstm", "train.py")
with open(train_path, "r") as f:
    train_src = f.read()

if "TRAINING CONFIGURATION VERIFICATION" in train_src:
    print("  [OK] Config verification block present in train.py")
else:
    print("  [FAIL] Config verification block NOT found in train.py")

if "ReduceLROnPlateau" in train_src:
    print("  [OK] ReduceLROnPlateau scheduler created in train.py")
else:
    print("  [FAIL] ReduceLROnPlateau NOT found in train.py")

if "scheduler=scheduler" in train_src:
    print("  [OK] Scheduler passed to Trainer constructor")
else:
    print("  [FAIL] Scheduler NOT passed to Trainer")

print("\n" + "=" * 60)
print("DRY-RUN VERIFICATION COMPLETE")
print("=" * 60)
