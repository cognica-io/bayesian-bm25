#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""BM25 scorer with Bayesian probability transforms.

Integrates bm25s as the search backend with the Bayesian probability
framework to return calibrated relevance probabilities instead of raw
BM25 scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import bm25s
except ImportError as exc:
    raise ImportError(
        "bm25s is required for BayesianBM25Scorer. "
        "Install it with: pip install bayesian-bm25[scorer]"
    ) from exc

from bayesian_bm25.probability import BayesianProbabilityTransform


class BayesianBM25Scorer:
    """BM25 scorer that returns Bayesian-calibrated probabilities.

    Parameters
    ----------
    k1 : float
        BM25 k1 parameter (term frequency saturation).
    b : float
        BM25 b parameter (document length normalisation).
    method : str
        BM25 variant: "robertson", "lucene", or "atire".
    alpha : float or None
        Sigmoid steepness.  If None, auto-estimated from corpus statistics
        during indexing.
    beta : float or None
        Sigmoid midpoint.  If None, auto-estimated from corpus statistics
        during indexing.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "robertson",
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        self._bm25 = bm25s.BM25(k1=k1, b=b, method=method)
        self._user_alpha = alpha
        self._user_beta = beta
        self._transform: BayesianProbabilityTransform | None = None
        self._doc_lengths: np.ndarray | None = None
        self._avgdl: float | None = None
        self._corpus_tokens: list[list[str]] | None = None

    @property
    def num_docs(self) -> int:
        """Number of indexed documents."""
        return int(self._bm25.scores["num_docs"])

    @property
    def doc_lengths(self) -> np.ndarray:
        """Document lengths (token counts)."""
        if self._doc_lengths is None:
            raise RuntimeError("Call index() before accessing doc_lengths.")
        return self._doc_lengths

    @property
    def avgdl(self) -> float:
        """Average document length."""
        if self._avgdl is None:
            raise RuntimeError("Call index() before accessing avgdl.")
        return self._avgdl

    def index(self, corpus_tokens: list[list[str]], show_progress: bool = True) -> None:
        """Build the BM25 index and compute document statistics.

        Parameters
        ----------
        corpus_tokens : list of list of str
            Each inner list is a document's tokens.
        show_progress : bool
            Whether to show progress bars during indexing.
        """
        self._corpus_tokens = corpus_tokens
        self._bm25.index(corpus_tokens, show_progress=show_progress)

        self._doc_lengths = np.array(
            [len(doc) for doc in corpus_tokens], dtype=np.float64
        )
        self._avgdl = float(np.mean(self._doc_lengths))

        alpha, beta = self._estimate_parameters(corpus_tokens)
        self._transform = BayesianProbabilityTransform(alpha=alpha, beta=beta)

    def _estimate_parameters(
        self, corpus_tokens: list[list[str]]
    ) -> tuple[float, float]:
        """Auto-estimate alpha and beta from the corpus score distribution.

        Samples documents as pseudo-queries to build a score distribution,
        then sets:
          - beta = median(scores)  -- sigmoid midpoint at typical score
          - alpha = 1 / std(scores) -- normalise steepness to score spread

        User-supplied values override the estimates.
        """
        if self._user_alpha is not None and self._user_beta is not None:
            return self._user_alpha, self._user_beta

        n = len(corpus_tokens)
        sample_size = min(n, 50)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n, size=sample_size, replace=False)

        all_scores: list[float] = []
        for idx in sample_indices:
            query_tokens = corpus_tokens[idx]
            if not query_tokens:
                continue
            query = query_tokens[:5]
            scores = self._bm25.get_scores(query)
            nonzero = scores[scores > 0]
            if len(nonzero) > 0:
                all_scores.extend(nonzero.tolist())

        if not all_scores:
            return (self._user_alpha or 1.0, self._user_beta or 0.0)

        score_array = np.array(all_scores)
        estimated_beta = float(np.median(score_array))
        score_std = float(np.std(score_array))
        estimated_alpha = 1.0 / score_std if score_std > 0 else 1.0

        alpha = self._user_alpha if self._user_alpha is not None else estimated_alpha
        beta = self._user_beta if self._user_beta is not None else estimated_beta
        return alpha, beta

    def retrieve(
        self,
        query_tokens: list[list[str]],
        k: int = 10,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k documents with Bayesian probabilities.

        Parameters
        ----------
        query_tokens : list of list of str
            Tokenised queries (one inner list per query).
        k : int
            Number of documents to retrieve per query.
        show_progress : bool
            Whether to show progress bars.

        Returns
        -------
        doc_ids : ndarray of shape (num_queries, k)
            Document indices.
        probabilities : ndarray of shape (num_queries, k)
            Calibrated relevance probabilities.
        """
        if self._transform is None:
            raise RuntimeError("Call index() before retrieve().")

        results = self._bm25.retrieve(
            query_tokens, k=k, sorted=True, show_progress=show_progress
        )
        doc_ids = results.documents
        bm25_scores = results.scores

        probabilities = self._scores_to_probabilities(
            doc_ids, bm25_scores, query_tokens
        )
        return doc_ids, probabilities

    def get_probabilities(
        self,
        query_tokens: list[str],
    ) -> np.ndarray:
        """Get probabilities for ALL documents (dense array).

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.

        Returns
        -------
        probabilities : ndarray of shape (num_docs,)
            Calibrated relevance probability for every document.
        """
        if self._transform is None:
            raise RuntimeError("Call index() before get_probabilities().")

        bm25_scores = self._bm25.get_scores(query_tokens)
        doc_ids = np.arange(len(bm25_scores))
        probabilities = self._scores_to_probabilities(
            doc_ids.reshape(1, -1),
            bm25_scores.astype(np.float64).reshape(1, -1),
            [query_tokens],
        )
        return probabilities.squeeze(0)

    def _compute_tf(self, doc_tokens: list[str], query_tokens: list[str]) -> float:
        """Compute total term frequency of query tokens in a document."""
        query_set = set(query_tokens)
        return float(sum(1 for t in doc_tokens if t in query_set))

    def _scores_to_probabilities(
        self,
        doc_ids: np.ndarray,
        bm25_scores: np.ndarray,
        query_tokens_batch: list[list[str]],
    ) -> np.ndarray:
        """Convert raw BM25 scores to probabilities for given doc IDs."""
        assert self._transform is not None
        assert self._doc_lengths is not None
        assert self._corpus_tokens is not None

        probabilities = np.zeros_like(bm25_scores, dtype=np.float64)

        for q_idx in range(doc_ids.shape[0]):
            query = query_tokens_batch[q_idx]
            for d_idx in range(doc_ids.shape[1]):
                did = int(doc_ids[q_idx, d_idx])
                score = float(bm25_scores[q_idx, d_idx])

                if score <= 0:
                    probabilities[q_idx, d_idx] = 0.0
                    continue

                doc_len = self._doc_lengths[did]
                doc_len_ratio = doc_len / self._avgdl
                tf = self._compute_tf(self._corpus_tokens[did], query)

                probabilities[q_idx, d_idx] = self._transform.score_to_probability(
                    score, tf, doc_len_ratio
                )

        return probabilities
