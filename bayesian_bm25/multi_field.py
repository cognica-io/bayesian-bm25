#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Multi-field BM25 scorer with Bayesian probability fusion.

Manages per-field ``BayesianBM25Scorer`` instances and fuses field-level
probabilities via ``log_odds_conjunction``.  This enables first-class
multi-field search (e.g., title + body) with calibrated output.
"""

from __future__ import annotations

import numpy as np

from bayesian_bm25.fusion import _resolve_alpha, log_odds_conjunction
from bayesian_bm25.scorer import BayesianBM25Scorer

_VALID_ALPHA_VALUES = (float, int, str, type(None))


class MultiFieldScorer:
    """Multi-field BM25 scorer that fuses per-field Bayesian probabilities.

    Parameters
    ----------
    fields : list of str
        Ordered list of field names (e.g., ``["title", "body"]``).
    field_weights : dict mapping field name to float, or None
        Per-field weights for log-odds conjunction.  Must be non-negative
        and sum to 1.  If None, equal weights are used.
    alpha : float, str, or None
        Confidence scaling exponent for log-odds conjunction.
        ``"auto"`` resolves to 0.5 (sqrt(n) scaling).
    base_rate : float, str, or None
        Passed through to each per-field ``BayesianBM25Scorer``.
    k1 : float
        BM25 k1 parameter.
    b : float
        BM25 b parameter.
    method : str
        BM25 variant: ``"robertson"``, ``"lucene"``, or ``"atire"``.
    """

    def __init__(
        self,
        fields: list[str],
        field_weights: dict[str, float] | None = None,
        alpha: float | str | None = "auto",
        base_rate: float | str | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "robertson",
    ) -> None:
        if not fields:
            raise ValueError("fields must be a non-empty list")
        if len(fields) != len(set(fields)):
            raise ValueError("fields must not contain duplicates")

        self._fields = list(fields)
        self._alpha = alpha
        self._base_rate = base_rate
        self._k1 = k1
        self._b = b
        self._method = method

        # Resolve field weights
        if field_weights is None:
            n = len(fields)
            self._field_weights = {f: 1.0 / n for f in fields}
        else:
            for f in fields:
                if f not in field_weights:
                    raise ValueError(
                        f"field_weights missing key {f!r}"
                    )
            weight_sum = sum(field_weights[f] for f in fields)
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"field_weights must sum to 1, got {weight_sum}"
                )
            self._field_weights = {f: field_weights[f] for f in fields}

        # Per-field scorers (populated by index())
        self._scorers: dict[str, BayesianBM25Scorer] = {}
        self._num_docs: int = 0

    @property
    def num_docs(self) -> int:
        """Number of indexed documents."""
        return self._num_docs

    @property
    def fields(self) -> list[str]:
        """Ordered list of field names."""
        return list(self._fields)

    @property
    def field_weights(self) -> dict[str, float]:
        """Per-field weights for log-odds conjunction."""
        return dict(self._field_weights)

    def index(
        self,
        documents: list[dict[str, list[str]]],
        show_progress: bool = True,
    ) -> None:
        """Build per-field BM25 indexes.

        Parameters
        ----------
        documents : list of dict
            Each document is a dict mapping field name to a list of tokens.
            Every document must contain all fields.
        show_progress : bool
            Whether to show progress bars during indexing.
        """
        for i, doc in enumerate(documents):
            for field in self._fields:
                if field not in doc:
                    raise ValueError(
                        f"Document {i} missing field {field!r}"
                    )

        self._scorers = {}
        for field in self._fields:
            scorer = BayesianBM25Scorer(
                k1=self._k1,
                b=self._b,
                method=self._method,
                base_rate=self._base_rate,
            )
            field_corpus = [doc[field] for doc in documents]
            scorer.index(field_corpus, show_progress=show_progress)
            self._scorers[field] = scorer

        self._num_docs = len(documents)

    def get_probabilities(self, query_tokens: list[str]) -> np.ndarray:
        """Get fused probabilities for all documents (dense array).

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.

        Returns
        -------
        probabilities : ndarray of shape (num_docs,)
            Fused relevance probability for every document.
        """
        if not self._scorers:
            raise RuntimeError("Call index() before get_probabilities().")

        # Collect per-field probabilities: shape (num_docs, num_fields)
        field_probs = np.column_stack([
            self._scorers[field].get_probabilities(query_tokens)
            for field in self._fields
        ])

        # Build weights array in field order
        weights = np.array(
            [self._field_weights[f] for f in self._fields],
            dtype=np.float64,
        )

        # Resolve alpha for the conjunction
        effective_alpha = _resolve_alpha(self._alpha, default=0.5)

        return log_odds_conjunction(
            field_probs, alpha=effective_alpha, weights=weights
        )

    def retrieve(
        self,
        query_tokens: list[str],
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k documents by fused probability.

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.
        k : int
            Number of documents to retrieve.

        Returns
        -------
        doc_ids : ndarray of shape (k,)
            Document indices sorted by descending probability.
        probabilities : ndarray of shape (k,)
            Fused relevance probabilities for the top-k documents.
        """
        probs = self.get_probabilities(query_tokens)
        k = min(k, len(probs))
        top_k_ids = np.argsort(probs)[::-1][:k]
        return top_k_ids, probs[top_k_ids]

    def add_documents(
        self,
        new_documents: list[dict[str, list[str]]],
        show_progress: bool = True,
    ) -> None:
        """Add documents to the multi-field index.

        Since the underlying BM25 engine requires IDF recomputation,
        this method appends the new documents and rebuilds per-field
        indexes.

        Parameters
        ----------
        new_documents : list of dict
            Documents to add, same format as ``index()``.
        show_progress : bool
            Whether to show progress bars during reindexing.
        """
        if not self._scorers:
            raise RuntimeError("Call index() before add_documents().")

        for i, doc in enumerate(new_documents):
            for field in self._fields:
                if field not in doc:
                    raise ValueError(
                        f"New document {i} missing field {field!r}"
                    )

        for field in self._fields:
            new_field_corpus = [doc[field] for doc in new_documents]
            self._scorers[field].add_documents(
                new_field_corpus, show_progress=show_progress
            )

        self._num_docs += len(new_documents)
