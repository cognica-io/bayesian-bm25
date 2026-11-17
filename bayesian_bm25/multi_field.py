#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Multi-field BM25 scorer with Bayesian probability fusion.

Manages per-field ``BayesianBM25Scorer`` instances. Query mode pools
prior-free field evidence and solves one final corpus count and ceiling.
Corpus mode retains the earlier posterior opinion pool for comparison.
"""

from __future__ import annotations

import numpy as np

from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.count_calibration import CountCalibration, calibrate_count
from bayesian_bm25.fusion import _resolve_alpha, log_odds_fusion
from bayesian_bm25.probability import _EPSILON, logit, sigmoid
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
        calibration: str = "corpus",
        expected_relevant: float | None = None,
        kappa: float | None = None,
        ceiling: float | None = None,
        calibration_context: CalibrationContext | None = None,
    ) -> None:
        if not fields:
            raise ValueError("fields must be a non-empty list")
        if len(fields) != len(set(fields)):
            raise ValueError("fields must not contain duplicates")

        self._calibration_context = (
            CalibrationContext(
                reference_population="corpus",
                count_source="convention" if expected_relevant is None else "supplied",
                slope_source="convention" if kappa is None else "supplied",
                ceiling_source="convention" if ceiling is None else "supplied",
            ) if calibration_context is None else calibration_context
        )
        if not isinstance(self._calibration_context, CalibrationContext):
            raise ValueError("calibration_context must be a CalibrationContext")
        if self._calibration_context.reference_population != "corpus":
            raise ValueError("the scorer calibrates the corpus reference population")
        expected_relevant = 1.0 if expected_relevant is None else float(expected_relevant)
        kappa = 1.0 if kappa is None else float(kappa)
        ceiling = 1.0 if ceiling is None else float(ceiling)
        if not np.isfinite(expected_relevant) or expected_relevant <= 0.0:
            raise ValueError("expected_relevant must be positive and finite")
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be positive and finite")
        if not np.isfinite(ceiling) or not 0.0 < ceiling <= 1.0:
            raise ValueError("ceiling must be finite and in (0, 1]")
        self._fields = list(fields)
        self._alpha = alpha
        self._base_rate = base_rate
        self._k1 = k1
        self._b = b
        self._method = method
        if calibration not in ("corpus", "query"):
            raise ValueError("calibration must be 'corpus' or 'query'")
        if calibration == "corpus" and ceiling != 1.0:
            raise ValueError("ceiling requires calibration='query'")
        self._calibration = calibration
        self._expected_relevant = expected_relevant
        self._kappa = kappa
        self._ceiling = ceiling

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
            if any(not np.isfinite(w) or w < 0.0 for w in self._field_weights.values()):
                raise ValueError("field_weights must be finite and non-negative")

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
        if (
            self._calibration == "query"
            and self._expected_relevant >= self._ceiling * len(documents)
        ):
            raise ValueError("expected_relevant must be less than ceiling * population size")
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
                calibration=self._calibration,
                expected_relevant=self._expected_relevant,
                kappa=self._kappa,
                calibration_context=self._calibration_context,
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

        if self._calibration == "query":
            evidence, model = self._query_count_model(query_tokens)
            return np.asarray(model.probability(evidence))
        return np.asarray(sigmoid(self.get_log_odds(query_tokens)))

    def _query_count_model(self, query_tokens: list[str]) -> tuple[np.ndarray, CountCalibration]:
        """Pool each field's evidence and constrain the fused corpus count."""
        if not self._scorers:
            raise RuntimeError("Call index() before scoring.")
        columns = []
        for field in self._fields:
            scorer = self._scorers[field]
            scores = scorer._raw_scores(query_tokens)
            calibration = scorer._calibration_from_scores(scores)
            columns.append(calibration.evidence(scores))
        weights = np.array([self._field_weights[field] for field in self._fields])
        pooled = np.asarray(log_odds_fusion(
            np.column_stack(columns), alpha=_resolve_alpha(self._alpha, default=0.5),
            weights=weights,
        ))
        model = calibrate_count(
            pooled, self._expected_relevant, ceiling=self._ceiling,
            context=self._calibration_context,
        )
        return pooled, model

    def get_ranking_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Keep the latent fused ordering when a posterior ceiling saturates."""
        if self._calibration == "query":
            evidence, model = self._query_count_model(query_tokens)
            return np.asarray(model.ranking_score(evidence))
        return self.get_log_odds(query_tokens)

    def get_log_odds(self, query_tokens: list[str]) -> np.ndarray:
        """Get fused log-odds for all documents (dense array).

        The per-field posteriors are pooled in log-odds space without
        passing through a probability, so documents with strong evidence
        keep their order where the fused probability would round to 1.
        A field in which a document matches no query term has
        probability 0; its log-odds are bounded below at the logit of
        the probability clamp, which is the value
        :meth:`get_probabilities` uses for it.

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.

        Returns
        -------
        log_odds : ndarray of shape (num_docs,)
            Fused log-odds of relevance for every document.
        """
        if not self._scorers:
            raise RuntimeError("Call index() before get_log_odds().")
        if self._calibration == "query":
            evidence, model = self._query_count_model(query_tokens)
            return np.asarray(model.log_odds(evidence))

        field_log_odds = np.column_stack([
            self._scorers[field].get_log_odds(query_tokens)
            for field in self._fields
        ])
        field_log_odds = np.maximum(field_log_odds, float(logit(_EPSILON)))

        weights = np.array(
            [self._field_weights[f] for f in self._fields],
            dtype=np.float64,
        )
        effective_alpha = _resolve_alpha(self._alpha, default=0.5)

        return np.asarray(
            log_odds_fusion(field_log_odds, alpha=effective_alpha, weights=weights),
            dtype=np.float64,
        )

    def retrieve(
        self,
        query_tokens: list[str],
        k: int = 10,
        return_log_odds: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k documents by fused relevance.

        Documents are ranked on the fused log-odds, which do not
        saturate; the returned values are probabilities unless
        ``return_log_odds`` is set.

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.
        k : int
            Number of documents to retrieve.
        return_log_odds : bool
            Return fused log-odds instead of probabilities.

        Returns
        -------
        doc_ids : ndarray of shape (k,)
            Document indices sorted by descending relevance.
        probabilities : ndarray of shape (k,)
            Fused relevance probabilities (or log-odds) for the top-k
            documents.
        """
        if self._calibration == "query":
            evidence, model = self._query_count_model(query_tokens)
            ranking = np.asarray(model.ranking_score(evidence))
            log_odds = np.asarray(model.log_odds(evidence))
        else:
            log_odds = self.get_log_odds(query_tokens)
            ranking = log_odds
        k = min(k, len(log_odds))
        top_k_ids = np.argsort(-ranking, kind="stable")[:k]
        if return_log_odds:
            return top_k_ids, log_odds[top_k_ids]
        return top_k_ids, np.asarray(sigmoid(log_odds[top_k_ids]), dtype=np.float64)

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
