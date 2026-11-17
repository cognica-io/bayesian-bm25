#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Block-level BM25 bounds."""

from __future__ import annotations

import numpy as np

from bayesian_bm25.probability import BayesianProbabilityTransform


class BlockMaxIndex:
    """Block-max index for BMW-style upper bounds (Paper 1, Section 6.2).

    Partitions documents into fixed-size blocks and stores per-block
    maximum BM25 contribution for each term.  Block-level upper bounds
    are tighter than global WAND upper bounds because they exploit
    locality in the posting list.

    Parameters
    ----------
    block_size : int
        Number of documents per block (default 128).
    """

    def __init__(self, block_size: int = 128) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self._block_size = block_size
        self._block_maxes: np.ndarray | None = None  # (n_terms, n_blocks)
        self._n_docs: int = 0
        self._n_terms: int = 0

    def build(self, score_matrix: np.ndarray) -> None:
        """Build block-max index from a per-term score matrix.

        Parameters
        ----------
        score_matrix : array of shape (n_terms, n_docs)
            Per-term BM25 contributions for each document.
        """
        score_matrix = np.asarray(score_matrix, dtype=np.float64)
        if score_matrix.ndim != 2:
            raise ValueError(
                f"score_matrix must be 2D (n_terms, n_docs), got {score_matrix.ndim}D"
            )
        n_terms, n_docs = score_matrix.shape
        self._n_terms = n_terms
        self._n_docs = n_docs

        bs = self._block_size
        n_blocks = (n_docs + bs - 1) // bs

        block_maxes = np.empty((n_terms, n_blocks), dtype=np.float64)
        for b in range(n_blocks):
            start = b * bs
            end = min(start + bs, n_docs)
            block_maxes[:, b] = np.max(score_matrix[:, start:end], axis=1)

        self._block_maxes = block_maxes

    def block_upper_bound(self, term_idx: int, block_id: int) -> float:
        """Return the per-term BM25 upper bound for a specific block.

        Parameters
        ----------
        term_idx : int
            Term index in the score matrix.
        block_id : int
            Block index.

        Returns
        -------
        Maximum BM25 contribution of ``term_idx`` in ``block_id``.
        """
        if self._block_maxes is None:
            raise RuntimeError("Call build() before block_upper_bound().")
        return float(self._block_maxes[term_idx, block_id])

    def bayesian_block_upper_bound(
        self,
        term_idx: int,
        block_id: int,
        transform: BayesianProbabilityTransform,
        p_max: float = 0.9,
    ) -> float:
        """Bayesian probability upper bound for a block (Corollary 7.4.2).

        Delegates to ``transform.wand_upper_bound()`` using the
        block-level BM25 maximum, producing a tighter bound than the
        global WAND upper bound.

        Parameters
        ----------
        term_idx : int
            Term index.
        block_id : int
            Block index.
        transform : BayesianProbabilityTransform
            Configured transform for score-to-probability conversion.
        p_max : float
            Maximum prior for the WAND upper bound formula.

        Returns
        -------
        Bayesian probability upper bound for the block.
        """
        block_max = self.block_upper_bound(term_idx, block_id)
        return float(transform.wand_upper_bound(block_max, p_max))

    @property
    def block_size(self) -> int:
        """Number of documents per block."""
        return self._block_size

    @property
    def n_blocks(self) -> int:
        """Total number of blocks."""
        if self._block_maxes is None:
            raise RuntimeError("Call build() before accessing n_blocks.")
        return self._block_maxes.shape[1]
