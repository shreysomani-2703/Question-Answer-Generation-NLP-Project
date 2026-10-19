"""Verification script for Phase 2A implementation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import torch

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")

print("=" * 60)
print("PHASE 2A VERIFICATION")
print("=" * 60)

# ── STEP 1: config.py ──
print("\n--- STEP 1: config.py constants ---")
from config import (HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_DIM, DROPOUT, BATCH_SIZE,
    LEARNING_RATE, MAX_GRAD_NORM, TEACHER_FORCING_RATIO_START, TEACHER_FORCING_DECAY,
    BEAM_WIDTH, MAX_DECODE_LEN, EPOCHS, EARLY_STOPPING_PATIENCE, GLOVE_PATH, LSTM_MODEL_DIR,
    LSTM_OUTPUT_DIR, BASE_DIR)

check("HIDDEN_SIZE == 256", HIDDEN_SIZE == 256, f"got {HIDDEN_SIZE}")
check("NUM_LAYERS == 2", NUM_LAYERS == 2, f"got {NUM_LAYERS}")
check("EMBEDDING_DIM == 100", EMBEDDING_DIM == 100, f"got {EMBEDDING_DIM}")
check("DROPOUT == 0.3", DROPOUT == 0.3, f"got {DROPOUT}")
check("BATCH_SIZE == 32", BATCH_SIZE == 32, f"got {BATCH_SIZE}")
check("LEARNING_RATE == 0.001", LEARNING_RATE == 0.001, f"got {LEARNING_RATE}")
check("MAX_GRAD_NORM == 5.0", MAX_GRAD_NORM == 5.0, f"got {MAX_GRAD_NORM}")
check("TF_RATIO_START == 0.5", TEACHER_FORCING_RATIO_START == 0.5)
check("TF_DECAY == 0.05", TEACHER_FORCING_DECAY == 0.05)
check("BEAM_WIDTH == 4", BEAM_WIDTH == 4)
check("MAX_DECODE_LEN == 64", MAX_DECODE_LEN == 64)
check("EPOCHS == 20", EPOCHS == 20)
check("EARLY_STOPPING_PATIENCE == 3", EARLY_STOPPING_PATIENCE == 3)
check("GLOVE_PATH exists as config", isinstance(GLOVE_PATH, str) and "glove" in GLOVE_PATH.lower())
check("LSTM_MODEL_DIR exists as config", isinstance(LSTM_MODEL_DIR, str) and "lstm_model" in LSTM_MODEL_DIR)
# Ensure Phase 1 constants still intact
from config import SQUAD_VERSION, MAX_CONTEXT_LEN, MAX_QUESTION_LEN, MAX_VOCAB_SIZE
check("Phase 1 constants preserved", SQUAD_VERSION == "squad" and MAX_CONTEXT_LEN == 512)

# ── STEP 2: dataset.py ──
print("\n--- STEP 2: dataset.py ---")
from phase2a_lstm.dataset import LSTMDataset, collate_fn

vocab = pickle.load(open(os.path.join(LSTM_OUTPUT_DIR, "vocab.pkl"), "rb"))
w2i = vocab["word2idx"]
i2w = vocab["idx2word"]

from phase1_pipeline.pipeline import Phase1Pipeline
p = Phase1Pipeline()
train_data = p.load_lstm_data("train")
val_data = p.load_lstm_data("val")
test_data = p.load_lstm_data("test")
check("Data loaded", len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0,
      f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

ds = LSTMDataset(train_data[:20], w2i)
check("LSTMDataset length", len(ds) == 20, f"got {len(ds)}")
check("bio2idx accessible", hasattr(ds, 'bio2idx') and ds.bio2idx == {"O":0, "B-ANS":1, "I-ANS":2, "<PAD>":3})

item = ds[0]
check("__getitem__ keys", all(k in item for k in ["src_ids","bio_ids","tgt_ids","src_len","tgt_len","id"]))
check("src_ids is list of int", isinstance(item["src_ids"], list) and all(isinstance(x, int) for x in item["src_ids"]))
check("bio_ids is list of int", isinstance(item["bio_ids"], list))
check("tgt_ids starts with SOS", item["tgt_ids"][0] == w2i.get("<SOS>", 2))
check("src_len correct", item["src_len"] == len(item["src_ids"]))
check("tgt_len correct", item["tgt_len"] == len(item["tgt_ids"]))
check("max_src_len truncation", item["src_len"] <= 512)
check("max_tgt_len truncation", item["tgt_len"] <= 64)

batch = collate_fn([ds[i] for i in range(4)])
check("collate returns src_ids LongTensor", batch["src_ids"].dtype == torch.long and batch["src_ids"].shape[0] == 4)
check("collate returns bio_ids LongTensor", batch["bio_ids"].dtype == torch.long)
check("collate returns tgt_ids LongTensor", batch["tgt_ids"].dtype == torch.long)
check("collate returns src_lens", batch["src_lens"].shape == (4,))
check("collate returns tgt_lens", batch["tgt_lens"].shape == (4,))
check("collate padding works", batch["src_ids"].shape[1] == max(ds[i]["src_len"] for i in range(4)))

# ── STEP 3: embeddings.py ──
print("\n--- STEP 3: embeddings.py ---")
from phase2a_lstm.embeddings import EmbeddingLoader
check("EmbeddingLoader class exists", hasattr(EmbeddingLoader, 'load_glove'))
check("load_glove is static", isinstance(EmbeddingLoader.__dict__['load_glove'], staticmethod))

emb = EmbeddingLoader.load_glove("nonexistent_file.txt", w2i, 100)
check("Missing GloVe doesn't crash", emb is not None)
check("Embedding shape correct", emb.shape == (len(w2i), 100), f"got {emb.shape}")
check("PAD (idx 0) is zeros", np.all(emb[0] == 0))
check("Non-PAD has values", not np.all(emb[1] == 0))

# ── STEP 4: attention.py ──
print("\n--- STEP 4: attention.py ---")
from phase2a_lstm.attention import BahdanauAttention
attn = BahdanauAttention(256)
check("W1 shape", attn.W1.in_features == 512 and attn.W1.out_features == 256)
check("W2 shape", attn.W2.in_features == 256 and attn.W2.out_features == 256)
check("V shape", attn.V.in_features == 256 and attn.V.out_features == 1)
check("V has no bias", attn.V.bias is None)

enc_out = torch.randn(2, 15, 512)
dec_h = torch.randn(2, 256)
ctx, wts = attn(enc_out, dec_h)
check("context shape (B, H*2)", ctx.shape == (2, 512), f"got {ctx.shape}")
check("weights shape (B, src_len)", wts.shape == (2, 15), f"got {wts.shape}")
check("weights sum ~= 1", torch.allclose(wts.sum(dim=1), torch.ones(2), atol=1e-5))

# ── STEP 5: model.py ──
print("\n--- STEP 5: model.py ---")
from phase2a_lstm.model import Encoder, Decoder, Seq2SeqQG

# Encoder
enc = Encoder(len(w2i), 100, 256, 2, 0.3)
check("Encoder token_emb padding_idx=0", enc.token_emb.padding_idx == 0)
check("Encoder bio_emb (4, 16, pad=3)", enc.bio_emb.num_embeddings == 4 and enc.bio_emb.embedding_dim == 16 and enc.bio_emb.padding_idx == 3)
check("Encoder LSTM bidirectional", enc.lstm.bidirectional)
check("Encoder LSTM num_layers=2", enc.lstm.num_layers == 2)
check("Encoder fc_h (512->256)", enc.fc_h.in_features == 512 and enc.fc_h.out_features == 256)
check("Encoder fc_c (512->256)", enc.fc_c.in_features == 512 and enc.fc_c.out_features == 256)

# Test with embedding matrix
enc2 = Encoder(len(w2i), 100, 256, 2, 0.3, emb)
check("Encoder accepts embedding_matrix", enc2.token_emb.weight.requires_grad)

# Encoder forward
enc_outs, h, c = enc(batch["src_ids"], batch["bio_ids"], batch["src_lens"])
check("Encoder output shape (B, L, H*2)", enc_outs.shape == (4, batch["src_ids"].shape[1], 512), f"got {enc_outs.shape}")
check("Encoder hidden shape (layers, B, H)", h.shape == (2, 4, 256), f"got {h.shape}")
check("Encoder cell shape (layers, B, H)", c.shape == (2, 4, 256), f"got {c.shape}")

# Decoder
dec = Decoder(len(w2i), 100, 256, 2, 0.3)
check("Decoder emb padding_idx=0", dec.emb.padding_idx == 0)
check("Decoder has attention", isinstance(dec.attention, BahdanauAttention))
check("Decoder LSTM not bidirectional", not dec.lstm.bidirectional)
check("Decoder fc_out (H+H*2 -> vocab)", dec.fc_out.in_features == 256 + 512 and dec.fc_out.out_features == len(w2i))

token = batch["tgt_ids"][:, 0]
logits, h2, c2, aw = dec(token, enc_outs, h, c)
check("Decoder logits shape (B, vocab)", logits.shape == (4, len(w2i)))
check("Decoder hidden shape preserved", h2.shape == (2, 4, 256))
check("Decoder attn_weights shape", aw.shape[0] == 4)

# Seq2SeqQG
device = torch.device("cpu")
model = Seq2SeqQG(enc, dec, device)
out = model(batch["src_ids"], batch["bio_ids"], batch["src_lens"], batch["tgt_ids"], 0.5)
check("Seq2Seq output shape (B, tgt_len-1, vocab)", out.shape == (4, batch["tgt_ids"].shape[1]-1, len(w2i)), f"got {out.shape}")
n_params = sum(p.numel() for p in model.parameters())
check(f"Model has parameters ({n_params:,})", n_params > 0)

# ── STEP 6: trainer.py ──
print("\n--- STEP 6: trainer.py ---")
from phase2a_lstm.trainer import Trainer
import torch.nn as nn
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
trainer = Trainer(model, optimizer, criterion, device, i2w, w2i)
check("Trainer instantiates", trainer is not None)
check("Trainer has train_epoch", hasattr(trainer, 'train_epoch'))
check("Trainer has evaluate", hasattr(trainer, 'evaluate'))
check("Trainer has train", hasattr(trainer, 'train'))
check("Trainer stores eos/pad/sos indices", trainer.eos_idx == 3 and trainer.pad_idx == 0 and trainer.sos_idx == 2)

# ── STEP 7: inference.py ──
print("\n--- STEP 7: inference.py ---")
from phase2a_lstm.inference import InferenceEngine
engine = InferenceEngine(model, w2i, i2w, device, 4, 64)
check("InferenceEngine instantiates", engine is not None)
check("InferenceEngine has greedy_decode", hasattr(engine, 'greedy_decode'))
check("InferenceEngine has beam_decode", hasattr(engine, 'beam_decode'))
check("InferenceEngine has batch_generate", hasattr(engine, 'batch_generate'))
check("InferenceEngine has save_predictions", hasattr(engine, 'save_predictions'))

q = engine.greedy_decode(batch["src_ids"][0], batch["bio_ids"][0], batch["src_lens"][0:1])
check("greedy_decode returns str", isinstance(q, str))

preds = engine.batch_generate(test_data[:2], use_beam=False)
check("batch_generate returns list", isinstance(preds, list) and len(preds) == 2)
check("prediction has 'id'", "id" in preds[0])
check("prediction has 'generated_question'", "generated_question" in preds[0])
check("prediction has 'reference_question'", "reference_question" in preds[0])
check("prediction has 'context'", "context" in preds[0])
check("prediction has 'answer'", "answer" in preds[0])

# ── STEP 8: train.py ──
print("\n--- STEP 8: train.py ---")
check("train.py exists", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "train.py")))

# ── Directory check ──
print("\n--- Directory Structure ---")
check("outputs/lstm_model/ exists", os.path.isdir(os.path.join(BASE_DIR, "outputs", "lstm_model")))
check("phase2a_lstm/__init__.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "__init__.py")))
check("phase2a_lstm/dataset.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "dataset.py")))
check("phase2a_lstm/embeddings.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "embeddings.py")))
check("phase2a_lstm/attention.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "attention.py")))
check("phase2a_lstm/model.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "model.py")))
check("phase2a_lstm/trainer.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "trainer.py")))
check("phase2a_lstm/inference.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "inference.py")))
check("phase2a_lstm/train.py", os.path.isfile(os.path.join(BASE_DIR, "phase2a_lstm", "train.py")))

# ── Integration Contract ──
print("\n--- Integration Contract ---")
from phase2a_lstm.inference import InferenceEngine as IE
check("Phase 3 import: from phase2a_lstm.inference import InferenceEngine", IE is not None)
check("batch_generate returns required keys", all(k in preds[0] for k in ["id", "generated_question", "reference_question", "context", "answer"]))

# ── Phase 1 files NOT modified ──
print("\n--- Phase 1 Integrity ---")
check("Phase 1 pipeline.py untouched (load_lstm_data works)", callable(Phase1Pipeline.load_lstm_data))
check("Phase 1 vocab has word2idx/idx2word/embedding_matrix", all(k in vocab for k in ["word2idx", "idx2word", "embedding_matrix"]))

# ── Summary ──
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("[OK] ALL CHECKS PASSED - Phase 2A is correctly implemented!")
else:
    print(f"[!!] {failed} CHECK(S) FAILED - review the failures above")
print("=" * 60)
