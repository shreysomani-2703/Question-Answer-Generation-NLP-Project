"""
GloVe embedding loader for the LSTM encoder.

Reads pre‑trained GloVe vectors from a text file and builds an embedding
matrix aligned with the project vocabulary.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np


class EmbeddingLoader:
    """Static utility for loading pre‑trained GloVe embeddings."""

    @staticmethod
    def load_glove(
        glove_path: str,
        word2idx: Dict[str, int],
        embedding_dim: int = 100,
    ) -> np.ndarray:
        """Build an embedding matrix from a GloVe text file.

        Parameters
        ----------
        glove_path : str
            Path to the GloVe vectors file (e.g. ``glove.6B.100d.txt``).
        word2idx : dict
            Word → index mapping produced by Phase 1.
        embedding_dim : int
            Dimensionality of the GloVe vectors (default 100).

        Returns
        -------
        np.ndarray
            Shape ``(len(word2idx), embedding_dim)``.  The PAD token
            (index 0) is kept as all zeros.  Words not found in GloVe
            receive a small random initialisation.
        """
        vocab_size = len(word2idx)

        # Initialise with small random values; PAD (0) stays zero.
        embedding_matrix = np.random.uniform(
            -0.1, 0.1, (vocab_size, embedding_dim)
        ).astype(np.float32)
        embedding_matrix[0] = np.zeros(embedding_dim, dtype=np.float32)  # PAD

        if not os.path.isfile(glove_path):
            print(
                f"[WARNING] GloVe file not found at '{glove_path}'. "
                f"Using random initialisation for all embeddings."
            )
            return embedding_matrix

        # Read GloVe vectors
        found = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in word2idx:
                    idx = word2idx[word]
                    vec = np.array(parts[1:], dtype=np.float32)
                    if vec.shape[0] == embedding_dim:
                        embedding_matrix[idx] = vec
                        found += 1

        coverage = (found / len(word2idx)) * 100
        print(f"GloVe coverage: {found} / {len(word2idx)} vocab words ({coverage:.1f}%)")
        if coverage < 50.0:
            print("WARNING: GloVe coverage below 50%. Check that GLOVE_PATH points to the correct file.")
            print(f"Expected file at: {glove_path}")
            print("Download with: wget http://nlp.stanford.edu/data/glove.6B.zip")
        elif coverage < 70.0:
            print("WARNING: GloVe coverage is low. Performance will be degraded.")
        else:
            print("GloVe coverage is good.")
        return embedding_matrix
