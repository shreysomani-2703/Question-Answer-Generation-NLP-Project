"""
Bahdanau (additive) attention mechanism for the Seq2Seq decoder.

Computes alignment scores between encoder outputs and the current
decoder hidden state, producing a context vector used by the decoder
at every generation step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention (Bahdanau et al., 2015).

    Parameters
    ----------
    hidden_size : int
        Decoder hidden size.  Encoder outputs are expected to have
        dimension ``hidden_size * 2`` (bidirectional).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(hidden_size * 2, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context and weights.

        Parameters
        ----------
        encoder_outputs : Tensor
            ``(batch, src_len, hidden_size * 2)``
        decoder_hidden : Tensor
            ``(batch, hidden_size)`` — last‑layer decoder hidden state.

        Returns
        -------
        context : Tensor
            ``(batch, hidden_size * 2)``
        weights : Tensor
            ``(batch, src_len)`` — normalised attention weights.
        """
        # score: (batch, src_len, 1)
        score = self.V(
            torch.tanh(
                self.W1(encoder_outputs)
                + self.W2(decoder_hidden).unsqueeze(1)
            )
        )

        # weights: (batch, src_len, 1)
        weights = F.softmax(score, dim=1)

        # context: (batch, hidden_size * 2)
        context = (weights * encoder_outputs).sum(dim=1)

        return context, weights.squeeze(-1)
