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
    base_rate : float, str, or None
        Corpus-level base rate of relevance.

        - None (default): no base rate correction.
        - "auto": auto-estimate from corpus score distribution during
          indexing by counting documents above the 95th percentile.
        - float in (0, 1): explicit base rate.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "robertson",
        alpha: float | None = None,
        beta: float | None = None,
        base_rate: float | str | None = None,
    ) -> None:
        self._bm25 = bm25s.BM25(k1=k1, b=b, method=method)
        self._user_alpha = alpha
        self._user_beta = beta
        self._user_base_rate = base_rate
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

    @property
    def base_rate(self) -> float | None:
        """Corpus-level base rate of relevance, or None if not set."""
        if self._transform is None:
            return None
        return self._transform.base_rate

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

        # Sample pseudo-query scores once, reuse for both estimation steps
        per_query_scores = self._sample_pseudo_query_scores(corpus_tokens)

        alpha, beta = self._estimate_parameters(per_query_scores)

        # Resolve base_rate
        base_rate: float | None = None
        if self._user_base_rate == "auto":
            base_rate = self._estimate_base_rate(
                per_query_scores, len(corpus_tokens)
            )
        elif isinstance(self._user_base_rate, (int, float)):
            base_rate = float(self._user_base_rate)

        self._transform = BayesianProbabilityTransform(
            alpha=alpha, beta=beta, base_rate=base_rate
        )

    def _sample_pseudo_query_scores(
        self, corpus_tokens: list[list[str]]
    ) -> list[np.ndarray]:
        """Sample documents as pseudo-queries and return per-query nonzero scores.

        Returns a list of 1-D arrays, one per pseudo-query, containing only
        the nonzero BM25 scores for that query.
        """
        n = len(corpus_tokens)
        sample_size = min(n, 50)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n, size=sample_size, replace=False)

        per_query_scores: list[np.ndarray] = []
        for idx in sample_indices:
            query_tokens = corpus_tokens[idx]
            if not query_tokens:
                continue
            query = query_tokens[:5]
            scores = self._bm25.get_scores(query)
            nonzero = scores[scores > 0]
            if len(nonzero) > 0:
                per_query_scores.append(nonzero)

        return per_query_scores

    def _estimate_parameters(
        self, per_query_scores: list[np.ndarray]
    ) -> tuple[float, float]:
        """Auto-estimate alpha and beta from the corpus score distribution.

        Uses pre-computed pseudo-query scores to set:
          - beta = median(scores)  -- sigmoid midpoint at typical score
          - alpha = 1 / std(scores) -- normalise steepness to score spread

        User-supplied values override the estimates.
        """
        if self._user_alpha is not None and self._user_beta is not None:
            return self._user_alpha, self._user_beta

        if not per_query_scores:
            return (self._user_alpha or 1.0, self._user_beta or 0.0)

        all_scores = np.concatenate(per_query_scores)
        estimated_beta = float(np.median(all_scores))
        score_std = float(np.std(all_scores))
        estimated_alpha = 1.0 / score_std if score_std > 0 else 1.0

        alpha = self._user_alpha if self._user_alpha is not None else estimated_alpha
        beta = self._user_beta if self._user_beta is not None else estimated_beta
        return alpha, beta

    def _estimate_base_rate(
        self, per_query_scores: list[np.ndarray], n_docs: int
    ) -> float:
        """Auto-estimate the corpus-level base rate of relevance.

        For each pseudo-query, counts documents scoring above the 95th
        percentile of non-zero scores.  The average count divided by
        corpus size gives the base rate estimate, clamped to [1e-6, 0.5].
        """
        if not per_query_scores:
            return 1e-6

        high_count_ratios: list[float] = []
        for scores in per_query_scores:
            threshold = float(np.percentile(scores, 95))
            n_above = int(np.sum(scores >= threshold))
            high_count_ratios.append(n_above / n_docs)

        base_rate = float(np.mean(high_count_ratios))
        return float(np.clip(base_rate, 1e-6, 0.5))

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

    def _compute_tf_batch(
        self, doc_ids: np.ndarray, query_tokens: list[str]
    ) -> np.ndarray:
        """Compute total term frequencies for multiple documents against a query."""
        query_set = set(query_tokens)
        corpus_tokens = self._corpus_tokens
        return np.array(
            [sum(1 for t in corpus_tokens[did] if t in query_set) for did in doc_ids],
            dtype=np.float64,
        )

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
            scores = bm25_scores[q_idx]

            active_mask = scores > 0
            if not np.any(active_mask):
                continue

            active_ids = doc_ids[q_idx][active_mask].astype(int)
            active_scores = scores[active_mask]

            doc_len_ratios = self._doc_lengths[active_ids] / self._avgdl
            tfs = self._compute_tf_batch(active_ids, query)

            probabilities[q_idx, active_mask] = (
                self._transform.score_to_probability(
                    active_scores, tfs, doc_len_ratios
                )
            )

        return probabilities
