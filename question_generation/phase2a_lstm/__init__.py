"""
Phase 2A — LSTM Seq2Seq with Bahdanau Attention for Question Generation.

Public API
----------
>>> from phase2a_lstm import Seq2SeqQG, Encoder, Decoder
>>> from phase2a_lstm import BahdanauAttention
>>> from phase2a_lstm import Trainer
>>> from phase2a_lstm import InferenceEngine
>>> from phase2a_lstm import LSTMDataset, collate_fn
"""

from phase2a_lstm.model import Seq2SeqQG, Encoder, Decoder
from phase2a_lstm.attention import BahdanauAttention
from phase2a_lstm.trainer import Trainer
from phase2a_lstm.inference import InferenceEngine
from phase2a_lstm.dataset import LSTMDataset, collate_fn
