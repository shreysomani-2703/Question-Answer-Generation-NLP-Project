"""
Central configuration for the Question Generation pipeline.

All hyperparameters, paths, and constants are defined here so that
every module in the project imports from a single source of truth.
"""

import os

# ── Dataset ──────────────────────────────────────────────────────────
SQUAD_VERSION: str = "squad"  # HuggingFace dataset name for SQuAD 1.1

# ── Token / sequence limits ─────────────────────────────────────────
MAX_CONTEXT_LEN: int = 512   # max tokens for T5 input
MAX_QUESTION_LEN: int = 64   # max tokens for T5 target

# ── LSTM vocabulary ─────────────────────────────────────────────────
MAX_VOCAB_SIZE: int = 50_000  # upper bound on vocabulary size
MIN_WORD_FREQ: int = 2        # minimum frequency to include a word
GLOVE_DIM: int = 100          # GloVe embedding dimension

# ── Data splits ─────────────────────────────────────────────────────
TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1
RANDOM_SEED: int = 42

# ── Paths (relative to the question_generation/ root) ───────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
LSTM_OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs", "lstm_ready")
T5_OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs", "t5_ready")

# ── T5 prompt format ────────────────────────────────────────────────
T5_PROMPT_TEMPLATE: str = "generate question: {context} answer: {answer}"

# ── Phase 2A: LSTM Seq2Seq hyperparameters ──────────────────────────
HIDDEN_SIZE: int = 256
NUM_LAYERS: int = 2
EMBEDDING_DIM: int = 100
DROPOUT: float = 0.3
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
MAX_GRAD_NORM: float = 5.0
TEACHER_FORCING_RATIO_START: float = 0.5
TEACHER_FORCING_DECAY: float = 0.03
BEAM_WIDTH: int = 4
MAX_DECODE_LEN: int = 64
EPOCHS: int = 25
EARLY_STOPPING_PATIENCE: int = 5

# ── Phase 2A: Paths ────────────────────────────────────────────────
GLOVE_PATH: str = os.path.join(BASE_DIR, "data", "glove.6B.100d.txt")
LSTM_MODEL_DIR: str = os.path.join(BASE_DIR, "outputs", "lstm_model")

# ── Phase 2B: T5 Fine-tuning hyperparameters ───────────────────────
T5_MODEL_NAME: str        = "t5-small"
T5_MAX_INPUT_LENGTH: int  = 512
T5_MAX_TARGET_LENGTH: int = 64
T5_BATCH_SIZE: int        = 8
T5_LEARNING_RATE: float   = 3e-4
T5_WEIGHT_DECAY: float    = 0.01
T5_EPOCHS: int            = 5
T5_WARMUP_RATIO: float    = 0.1
T5_NUM_BEAMS: int         = 4
T5_EARLY_STOP_PAT: int    = 2

# ── Phase 2B: Paths ────────────────────────────────────────────────
T5_MODEL_DIR: str = os.path.join(BASE_DIR, "outputs", "t5_model") + os.sep
