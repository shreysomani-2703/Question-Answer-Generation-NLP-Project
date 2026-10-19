"""
Encoder–Decoder Seq2Seq model with Bahdanau attention for question generation.

Architecture
------------
- **Encoder**: Bidirectional LSTM that encodes source tokens + BIO tags.
- **Decoder**: Unidirectional LSTM with additive attention.
- **Seq2SeqQG**: Wraps encoder and decoder, handles teacher forcing.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from phase2a_lstm.attention import BahdanauAttention


# =====================================================================
# Encoder
# =====================================================================
class Encoder(nn.Module):
    """Bidirectional LSTM encoder with token + BIO tag embeddings.

    Parameters
    ----------
    vocab_size : int
    embedding_dim : int
    hidden_size : int
    num_layers : int
    dropout : float
    embedding_matrix : np.ndarray or None
        Pre‑trained embedding weights (e.g. GloVe).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        embedding_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── Token embedding ──
        self.token_emb = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        if embedding_matrix is not None:
            self.token_emb.weight.data.copy_(
                torch.from_numpy(embedding_matrix).float()
            )
            self.token_emb.weight.requires_grad = True

        # ── BIO tag embedding (4 tags, dim=16, pad_idx=3) ──
        self.bio_emb = nn.Embedding(4, 16, padding_idx=3)

        lstm_input_dim = embedding_dim + 16

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Projection from bidirectional hidden → decoder‑sized hidden
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        bio_ids: torch.Tensor,
        src_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode source sequences.

        Parameters
        ----------
        src_ids : LongTensor  ``(B, L)``
        bio_ids : LongTensor  ``(B, L)``
        src_lens : LongTensor ``(B,)``

        Returns
        -------
        encoder_outputs : Tensor ``(B, L, hidden_size * 2)``
        hidden : Tensor ``(num_layers, B, hidden_size)``
        cell : Tensor   ``(num_layers, B, hidden_size)``
        """
        # Embedding concat: (B, L, embedding_dim + 16)
        emb = self.dropout(
            torch.cat(
                [self.token_emb(src_ids), self.bio_emb(bio_ids)], dim=-1
            )
        )

        # Pack → LSTM → unpack
        packed = pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        encoder_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(
            encoder_outputs, batch_first=True
        )

        # hidden shape: (num_layers * 2, B, hidden_size)
        # Merge forward/backward for each layer
        batch_size = src_ids.size(0)
        hidden_proj_list = []
        cell_proj_list = []
        for layer in range(self.num_layers):
            fwd = hidden[2 * layer]       # (B, hidden_size)
            bwd = hidden[2 * layer + 1]   # (B, hidden_size)
            hidden_proj_list.append(
                torch.tanh(self.fc_h(torch.cat([fwd, bwd], dim=-1)))
            )
            fwd_c = cell[2 * layer]
            bwd_c = cell[2 * layer + 1]
            cell_proj_list.append(
                torch.tanh(self.fc_c(torch.cat([fwd_c, bwd_c], dim=-1)))
            )

        hidden_proj = torch.stack(hidden_proj_list, dim=0)  # (num_layers, B, H)
        cell_proj = torch.stack(cell_proj_list, dim=0)      # (num_layers, B, H)

        return encoder_outputs, hidden_proj, cell_proj


# =====================================================================
# Decoder
# =====================================================================
class Decoder(nn.Module):
    """Unidirectional LSTM decoder with Bahdanau attention.

    Parameters
    ----------
    vocab_size, embedding_dim, hidden_size, num_layers, dropout
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)

        lstm_input = embedding_dim + hidden_size * 2
        self.lstm = nn.LSTM(
            lstm_input,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_size + hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_id: torch.Tensor,
        encoder_outputs: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decoder single‑step forward.

        Parameters
        ----------
        token_id : LongTensor ``(B,)``
        encoder_outputs : Tensor ``(B, src_len, hidden_size * 2)``
        hidden : Tensor ``(num_layers, B, hidden_size)``
        cell : Tensor   ``(num_layers, B, hidden_size)``

        Returns
        -------
        logits : Tensor       ``(B, vocab_size)``
        hidden : Tensor       ``(num_layers, B, hidden_size)``
        cell : Tensor         ``(num_layers, B, hidden_size)``
        attn_weights : Tensor ``(B, src_len)``
        """
        # (B,) → (B, 1, embedding_dim)
        emb = self.dropout(self.emb(token_id)).unsqueeze(1)

        # Attention uses the top‑layer decoder hidden
        decoder_hidden_top = hidden[-1]  # (B, hidden_size)
        context, attn_weights = self.attention(
            encoder_outputs, decoder_hidden_top
        )

        # LSTM input: concat embedding + context
        lstm_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)  # (B, hidden_size)

        # Output projection: concat LSTM output + context → vocab logits
        logits = self.fc_out(torch.cat([output, context], dim=-1))

        return logits, hidden, cell, attn_weights


# =====================================================================
# Seq2Seq wrapper
# =====================================================================
class Seq2SeqQG(nn.Module):
    """Full encoder–decoder model with teacher forcing.

    Parameters
    ----------
    encoder : Encoder
    decoder : Decoder
    device : torch.device
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src_ids: torch.Tensor,
        bio_ids: torch.Tensor,
        src_lens: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Parameters
        ----------
        src_ids : LongTensor ``(B, src_len)``
        bio_ids : LongTensor ``(B, src_len)``
        src_lens : LongTensor ``(B,)``
        tgt_ids : LongTensor ``(B, tgt_len)``
        teacher_forcing_ratio : float

        Returns
        -------
        Tensor ``(B, tgt_len - 1, vocab_size)``
        """
        encoder_outputs, hidden, cell = self.encoder(
            src_ids, bio_ids, src_lens
        )

        tgt_len = tgt_ids.shape[1]
        all_logits = []

        input_token = tgt_ids[:, 0]  # SOS token

        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(
                input_token, encoder_outputs, hidden, cell
            )
            all_logits.append(logits.unsqueeze(1))

            if random.random() < teacher_forcing_ratio:
                input_token = tgt_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        return torch.cat(all_logits, dim=1)  # (B, tgt_len-1, vocab_size)
