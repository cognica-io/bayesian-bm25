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

import math
import operator
from dataclasses import replace

import numpy as np

try:
    import bm25s
except ImportError as exc:
    raise ImportError(
        "bm25s is required for BayesianBM25Scorer. "
        "Install it with: pip install bayesian-bm25[scorer]"
    ) from exc

from bayesian_bm25.block_max_index import BlockMaxIndex as BlockMaxIndex
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.probability import BayesianProbabilityTransform, sigmoid
from bayesian_bm25.query_calibration import (
    QueryCalibration,
    calibrate_query,
    score_dispersion,
)
from bayesian_bm25.retrieval_result import RetrievalResult

_VALID_BASE_RATE_METHODS = ("percentile", "mixture", "elbow")
_VALID_CALIBRATIONS = ("corpus", "query")


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
          indexing.
        - float in (0, 1): explicit base rate.
    base_rate_method : str
        Estimation method when ``base_rate="auto"``.

        - ``"percentile"`` (default): 95th-percentile heuristic.
        - ``"mixture"``: 2-component Gaussian EM (numpy only, 20 iterations).
        - ``"elbow"``: knee point in sorted score curve.
    calibration : str
        How scores become probabilities.

        - ``"corpus"`` (default): one slope and one midpoint for the
          whole corpus, estimated from pseudo-queries at indexing time,
          with the composite document prior and the optional base rate.
          A corpus-level slope is the right order of magnitude only for
          queries that resemble the pseudo-queries: applied to a query
          twenty times as long it multiplies the evidence by roughly
          twenty.
        - ``"query"``: per-query calibration.  The slope is
          ``kappa / sigma_q`` with ``sigma_q`` the dispersion of the
          matching scores of each query, so it follows the scale of the
          query's scores; the offset is fixed by the count constraint
          ``sum_d P(R=1 | s_d) = expected_relevant`` over the whole
          corpus.  The posterior is a monotone transform of the BM25
          score for each query, so it preserves the BM25 ranking, and
          at most ``expected_relevant / p`` documents can receive
          probability ``p`` or more.  ``alpha``, ``beta`` and
          ``base_rate`` do not enter this mode.
    kappa : float
        Dimensionless slope for ``calibration="query"``: the evidence
        gained per standard deviation of a query's matching scores.
    expected_relevant : float
        Expected number of relevant documents per query for
        ``calibration="query"``.  Must be smaller than the corpus size.
        Neither ``kappa`` nor ``expected_relevant`` can be identified
        from scores alone; they are conventions unless fitted to
        relevance labels.
    ceiling : float
        Posterior ceiling in (0, 1] for query calibration. The expected
        count must be less than ``ceiling * num_docs``. The ceiling is
        supplied or learned from judgments; scores do not identify it.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "robertson",
        alpha: float | None = None,
        beta: float | None = None,
        base_rate: float | str | None = None,
        base_rate_method: str = "percentile",
        calibration: str = "corpus",
        kappa: float | None = None,
        expected_relevant: float | None = None,
        ceiling: float | None = None,
        calibration_context: CalibrationContext | None = None,
    ) -> None:
        supplied_count = expected_relevant is not None
        supplied_slope = kappa is not None
        supplied_ceiling = ceiling is not None
        kappa = 1.0 if kappa is None else float(kappa)
        expected_relevant = 1.0 if expected_relevant is None else float(expected_relevant)
        ceiling = 1.0 if ceiling is None else float(ceiling)
        if base_rate_method not in _VALID_BASE_RATE_METHODS:
            raise ValueError(
                f"base_rate_method must be one of {_VALID_BASE_RATE_METHODS}, "
                f"got {base_rate_method!r}"
            )
        if calibration not in _VALID_CALIBRATIONS:
            raise ValueError(
                f"calibration must be one of {_VALID_CALIBRATIONS}, "
                f"got {calibration!r}"
            )
        if not math.isfinite(kappa) or not (kappa > 0.0):
            raise ValueError(f"kappa must be positive and finite, got {kappa}")
        if not math.isfinite(expected_relevant) or not (expected_relevant > 0.0):
            raise ValueError(
                f"expected_relevant must be positive and finite, got {expected_relevant}"
            )
        if not math.isfinite(ceiling) or not (0.0 < ceiling <= 1.0):
            raise ValueError("ceiling must be finite and in (0, 1]")
        if calibration != "query" and ceiling != 1.0:
            raise ValueError("ceiling requires calibration='query'")
        self._bm25 = bm25s.BM25(k1=k1, b=b, method=method)
        self._user_alpha = alpha
        self._user_beta = beta
        self._user_base_rate = base_rate
        self._base_rate_method = base_rate_method
        self._calibration = calibration
        self._kappa = float(kappa)
        self._expected_relevant = float(expected_relevant)
        self._ceiling = float(ceiling)
        self._calibration_context = (
            CalibrationContext(
                reference_population="corpus",
                count_source="supplied" if supplied_count else "convention",
                slope_source="supplied" if supplied_slope else "convention",
                ceiling_source="supplied" if supplied_ceiling else "convention",
            ) if calibration_context is None else calibration_context
        )
        if not isinstance(self._calibration_context, CalibrationContext):
            raise ValueError("calibration_context must be a CalibrationContext")
        if self._calibration_context.reference_population != "corpus":
            raise ValueError("the scorer calibrates the corpus reference population")
        self._transform: BayesianProbabilityTransform | None = None
        self._doc_lengths: np.ndarray | None = None
        self._avgdl: float | None = None
        self._corpus_tokens: list[list[str]] | None = None
        self._doc_token_sets: list[set[str]] | None = None

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

    @property
    def calibration(self) -> str:
        """Calibration mode: ``"corpus"`` or ``"query"``."""
        return self._calibration

    @property
    def kappa(self) -> float:
        """Dimensionless slope used by ``calibration="query"``."""
        return self._kappa

    @property
    def expected_relevant(self) -> float:
        """Expected relevant documents per query for ``calibration="query"``."""
        return self._expected_relevant

    @property
    def ceiling(self) -> float:
        """Default posterior ceiling for per-query calibration."""
        return self._ceiling

    def index(self, corpus_tokens: list[list[str]], show_progress: bool = True) -> None:
        """Build the BM25 index and compute document statistics.

        Parameters
        ----------
        corpus_tokens : list of list of str
            Each inner list is a document's tokens.
        show_progress : bool
            Whether to show progress bars during indexing.
        """
        if (
            self._calibration == "query"
            and self._expected_relevant >= self._ceiling * len(corpus_tokens)
        ):
            raise ValueError(
                f"expected_relevant ({self._expected_relevant}) must be smaller "
                f"than the corpus size times ceiling ({self._ceiling * len(corpus_tokens):g})"
            )
        self._corpus_tokens = corpus_tokens
        self._doc_token_sets = [set(tokens) for tokens in corpus_tokens]
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
        # Scores that are equal up to rounding have no scale to estimate.
        score_std = score_dispersion(all_scores)
        estimated_alpha = 1.0 / score_std if score_std > 0 else 1.0

        alpha = self._user_alpha if self._user_alpha is not None else estimated_alpha
        beta = self._user_beta if self._user_beta is not None else estimated_beta
        return alpha, beta

    def _estimate_base_rate(
        self, per_query_scores: list[np.ndarray], n_docs: int
    ) -> float:
        """Auto-estimate the corpus-level base rate of relevance.

        Dispatches to the method selected by ``base_rate_method``:

        - ``"percentile"``: 95th-percentile heuristic (original default).
        - ``"mixture"``: 2-component Gaussian EM.
        - ``"elbow"``: knee point in sorted score curve.

        All methods clamp the result to [1e-6, 0.5].
        """
        if not per_query_scores:
            return 1e-6

        method = self._base_rate_method
        if method == "percentile":
            return self._base_rate_percentile(per_query_scores, n_docs)
        if method == "mixture":
            return self._base_rate_mixture(per_query_scores)
        if method == "elbow":
            return self._base_rate_elbow(per_query_scores)
        raise ValueError(
            f"Unknown base_rate_method: {method!r}"
        )

    @staticmethod
    def _base_rate_percentile(
        per_query_scores: list[np.ndarray], n_docs: int
    ) -> float:
        """95th-percentile heuristic: fraction of docs above the 95th pct."""
        high_count_ratios: list[float] = []
        for scores in per_query_scores:
            threshold = float(np.percentile(scores, 95))
            n_above = int(np.sum(scores >= threshold))
            high_count_ratios.append(n_above / n_docs)

        base_rate = float(np.mean(high_count_ratios))
        return float(np.clip(base_rate, 1e-6, 0.5))

    @staticmethod
    def _base_rate_mixture(per_query_scores: list[np.ndarray]) -> float:
        """2-component Gaussian EM to separate relevant/non-relevant scores.

        Fits a mixture of two Gaussians to the pooled score distribution.
        The component with the higher mean is assumed to be the "relevant"
        population; its mixing weight is the base rate estimate.
        """
        all_scores = np.concatenate(per_query_scores)
        if len(all_scores) < 2:
            return 1e-6

        # Initialize: split at median
        median = float(np.median(all_scores))
        lo_mask = all_scores <= median
        hi_mask = ~lo_mask

        mu0 = float(np.mean(all_scores[lo_mask])) if np.any(lo_mask) else median - 1.0
        mu1 = float(np.mean(all_scores[hi_mask])) if np.any(hi_mask) else median + 1.0
        var0 = float(np.var(all_scores[lo_mask])) if np.any(lo_mask) else 1.0
        var1 = float(np.var(all_scores[hi_mask])) if np.any(hi_mask) else 1.0
        var0 = max(var0, 1e-8)
        var1 = max(var1, 1e-8)
        pi1 = 0.5  # mixing weight of the "relevant" component

        for _ in range(20):
            # E-step: responsibilities
            std0 = np.sqrt(var0)
            std1 = np.sqrt(var1)
            log_p0 = -0.5 * ((all_scores - mu0) / std0) ** 2 - np.log(std0)
            log_p1 = -0.5 * ((all_scores - mu1) / std1) ** 2 - np.log(std1)

            # Log-sum-exp for numerical stability
            log_w0 = np.log(max(1.0 - pi1, 1e-10)) + log_p0
            log_w1 = np.log(max(pi1, 1e-10)) + log_p1
            log_total = np.logaddexp(log_w0, log_w1)
            gamma = np.exp(log_w1 - log_total)  # responsibility of component 1

            # M-step
            n_eff1 = float(np.sum(gamma))
            n_eff0 = float(np.sum(1.0 - gamma))

            if n_eff0 < 1e-8 or n_eff1 < 1e-8:
                break

            mu0 = float(np.sum((1.0 - gamma) * all_scores) / n_eff0)
            mu1 = float(np.sum(gamma * all_scores) / n_eff1)
            var0 = max(float(np.sum((1.0 - gamma) * (all_scores - mu0) ** 2) / n_eff0), 1e-8)
            var1 = max(float(np.sum(gamma * (all_scores - mu1) ** 2) / n_eff1), 1e-8)
            pi1 = n_eff1 / len(all_scores)

        # The higher-mean component is the "relevant" population
        base_rate = pi1 if mu1 >= mu0 else 1.0 - pi1
        return float(np.clip(base_rate, 1e-6, 0.5))

    @staticmethod
    def _base_rate_elbow(per_query_scores: list[np.ndarray]) -> float:
        """Knee point in sorted score curve.

        Sorts pooled scores descending, draws a line from the first to
        the last point, and finds the index with maximum perpendicular
        distance.  The fraction of scores above the knee is the base
        rate estimate.
        """
        all_scores = np.sort(np.concatenate(per_query_scores))[::-1]
        n = len(all_scores)
        if n < 3:
            return 1e-6

        # Line from (0, scores[0]) to (n-1, scores[-1])
        x = np.arange(n, dtype=np.float64)
        y = all_scores

        # Direction vector of the line
        dx = float(n - 1)
        dy = float(y[-1] - y[0])
        line_len = np.sqrt(dx * dx + dy * dy)

        if line_len < 1e-12:
            return 1e-6

        # Perpendicular distance from each point to the line
        distances = np.abs(dy * x - dx * (y - y[0])) / line_len
        knee_idx = int(np.argmax(distances))

        # Fraction of scores at or above the knee
        base_rate = max(1, knee_idx) / n
        return float(np.clip(base_rate, 1e-6, 0.5))

    def add_documents(
        self,
        new_corpus_tokens: list[list[str]],
        show_progress: bool = True,
    ) -> None:
        """Add documents to the index.

        Since the underlying BM25 engine requires IDF recomputation when the
        corpus changes, this method appends the new documents to the existing
        corpus and rebuilds the full index.  All previously computed BM25
        statistics (alpha, beta, base_rate) are re-estimated from the updated
        corpus.

        Parameters
        ----------
        new_corpus_tokens : list of list of str
            Tokenised documents to add.
        show_progress : bool
            Whether to show progress bars during reindexing.
        """
        if self._corpus_tokens is None:
            raise RuntimeError("Call index() before add_documents().")
        combined = self._corpus_tokens + new_corpus_tokens
        self.index(combined, show_progress=show_progress)

    def retrieve(
        self,
        query_tokens: list[list[str]],
        k: int = 10,
        show_progress: bool = False,
        explain: bool = False,
        return_log_odds: bool = False,
        *,
        expected_relevant: np.ndarray | float | None = None,
        kappa: np.ndarray | float | None = None,
        ceiling: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | RetrievalResult:
        """Retrieve top-k documents with Bayesian probabilities.

        Parameters
        ----------
        query_tokens : list of list of str
            Tokenised queries (one inner list per query).
        k : int
            Number of documents to retrieve per query.
        show_progress : bool
            Whether to show progress bars.
        explain : bool
            When True, returns a ``RetrievalResult`` with per-document
            ``BM25SignalTrace`` explanations.  When False (default),
            returns a backward-compatible ``(doc_ids, probabilities)``
            tuple.
        return_log_odds : bool
            Return posterior log-odds in place of probabilities.  A
            probability rounds to 1 for log-odds above about 36.7 and is
            clamped well before that, so documents with strong evidence
            tie; log-odds keep their order.  Under
            ``calibration="corpus"`` a retrieved document with a zero
            score has probability 0 and log-odds ``-inf``.
        expected_relevant, kappa, ceiling : float, array, or None
            Query-calibration overrides, either one scalar or one value
            per query. Counts refer to the whole indexed corpus. A result
            limit is not a relevance count. None uses the scorer default.
            These overrides require ``calibration="query"``.

        Returns
        -------
        tuple of (doc_ids, probabilities) when ``explain=False``.
        RetrievalResult when ``explain=True``.
        """
        if self._transform is None:
            raise RuntimeError("Call index() before retrieve().")

        try:
            depth = operator.index(k)
        except TypeError as exc:
            raise ValueError("k must be a positive integer") from exc
        if isinstance(k, bool) or depth < 1 or depth > self.num_docs:
            raise ValueError("k must be a positive integer no larger than the corpus size")
        if self._calibration != "query" and any(
            value is not None for value in (expected_relevant, kappa, ceiling)
        ):
            raise ValueError("query parameter overrides require calibration='query'")

        n_queries = len(query_tokens)
        counts = self._query_parameter(expected_relevant, self._expected_relevant, n_queries)
        slopes = self._query_parameter(kappa, self._kappa, n_queries)
        ceilings = self._query_parameter(ceiling, self._ceiling, n_queries)
        doc_ids = np.empty((n_queries, depth), dtype=np.int64)
        bm25_scores = np.empty((n_queries, depth), dtype=np.float64)
        values = np.empty((n_queries, depth), dtype=np.float64)
        ranking_scores = np.empty_like(values)
        posterior_log_odds = np.empty_like(values)
        calibrations: list[QueryCalibration] | None = [] if self._calibration == "query" else None
        population_ids = np.arange(self.num_docs, dtype=np.int64)

        for q_idx, query in bm25s.tqdm(
            enumerate(query_tokens), total=n_queries, disable=not show_progress,
            desc="Score and calibrate", leave=False,
        ):
            scores = self._raw_scores(query)
            if calibrations is not None:
                model = self._calibration_from_scores(
                    scores,
                    expected_relevant=None if expected_relevant is None else float(counts[q_idx]),
                    kappa=None if kappa is None else float(slopes[q_idx]),
                    ceiling=None if ceiling is None else float(ceilings[q_idx]),
                )
                calibrations.append(model)
                ranking = np.asarray(model.ranking_score(scores))
                log_odds = np.asarray(model.log_odds(scores))
            else:
                # A document prior can move a document outside raw BM25's
                # top-k into the posterior top-k. Score the population
                # before selecting rather than re-sorting a truncated list.
                log_odds = self._scores_to_probabilities(
                    population_ids[None, :], scores[None, :], [query], return_log_odds=True,
                )[0]
                ranking = log_odds

            eligible = population_ids
            if depth < self.num_docs:
                threshold = np.partition(ranking, self.num_docs - depth)[self.num_docs - depth]
                eligible = np.flatnonzero(ranking >= threshold)
            selected = eligible[np.argsort(-ranking[eligible], kind="stable")[:depth]]
            doc_ids[q_idx] = selected
            bm25_scores[q_idx] = scores[selected]
            ranking_scores[q_idx] = ranking[selected]
            posterior_log_odds[q_idx] = log_odds[selected]
            if return_log_odds:
                values[q_idx] = log_odds[selected]
            elif calibrations is not None:
                values[q_idx] = model.probability(scores[selected])
            else:
                values[q_idx] = self._scores_to_probabilities(
                    selected[None, :], scores[selected][None, :], [query],
                )[0]

        if not explain:
            return doc_ids, values

        from bayesian_bm25.debug import FusionDebugger

        explanations: list[list] = []

        for q_idx in range(doc_ids.shape[0]):
            query = query_tokens[q_idx]
            debugger = FusionDebugger(self._transform)
            query_explanations = []
            for rank in range(doc_ids.shape[1]):
                did = int(doc_ids[q_idx, rank])
                score = float(bm25_scores[q_idx, rank])
                if (score > 0 or calibrations is not None) and self._doc_lengths is not None:
                    dl_ratio = float(self._doc_lengths[did] / self._avgdl)
                    tf = float(len(set(query) & self._doc_token_sets[did]))
                    if calibrations is None:
                        trace = debugger.trace_bm25(score, tf, dl_ratio)
                    else:
                        trace = debugger.trace_query_calibration(
                            calibrations[q_idx], score, tf, dl_ratio,
                        )
                else:
                    trace = None
                query_explanations.append(trace)
            explanations.append(query_explanations)

        return RetrievalResult(
            doc_ids=doc_ids,
            probabilities=values,
            explanations=explanations,
            calibrations=calibrations,
            ranking_scores=ranking_scores,
            posterior_log_odds=posterior_log_odds,
        )

    @staticmethod
    def _query_parameter(
        value: np.ndarray | float | None, default: float, n_queries: int,
    ) -> np.ndarray:
        """Resolve scalar or per-query parameter overrides."""
        array = np.asarray(default if value is None else value, dtype=np.float64)
        if array.ndim == 0:
            return np.full(n_queries, float(array))
        if array.shape != (n_queries,):
            raise ValueError("query parameter arrays must have one value per query")
        return array

    def _raw_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Score one query, including the empty query's known zero vector."""
        if not isinstance(query_tokens, list):
            raise ValueError("query_tokens must be a list of tokens")
        if not query_tokens:
            return np.zeros(self.num_docs, dtype=np.float64)
        return np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float64)

    def calibrate_query(
        self,
        query_tokens: list[str],
        *,
        expected_relevant: float | None = None,
        kappa: float | None = None,
        ceiling: float | None = None,
    ) -> QueryCalibration:
        """Per-query calibration parameters for one query.

        Scores every document, sets the slope to ``kappa / sigma_q``
        from the dispersion of the matching scores, and solves the count
        constraint over the whole corpus for the offset.  Documents that
        match no query term enter the constraint in closed form at score
        zero.

        The result does not depend on the ``calibration`` mode of the
        scorer, so the per-query parameters can be inspected even when
        probabilities are produced with ``calibration="corpus"``.

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.

        Returns
        -------
        QueryCalibration
            Slope, offset and prevalence for this query.
        """
        if self._transform is None:
            raise RuntimeError("Call index() before calibrate_query().")
        scores = self._raw_scores(query_tokens)
        return self._calibration_from_scores(
            scores, expected_relevant=expected_relevant, kappa=kappa, ceiling=ceiling,
        )

    def _calibration_from_scores(
        self,
        scores: np.ndarray,
        *,
        expected_relevant: float | None = None,
        kappa: float | None = None,
        ceiling: float | None = None,
    ) -> QueryCalibration:
        """Calibrate one query from its full score vector."""
        if self._transform is None:
            raise RuntimeError("index() must be called before scoring")

        n_docs = len(scores)
        count = self._expected_relevant if expected_relevant is None else expected_relevant
        slope = self._kappa if kappa is None else kappa
        cap = self._ceiling if ceiling is None else ceiling
        context = replace(
            self._calibration_context,
            count_source=(
                self._calibration_context.count_source if expected_relevant is None else "supplied"
            ),
            slope_source=self._calibration_context.slope_source if kappa is None else "supplied",
            ceiling_source=(
                self._calibration_context.ceiling_source if ceiling is None else "supplied"
            ),
        )
        if count >= cap * n_docs:
            raise ValueError(
                f"expected_relevant ({count}) must be smaller "
                f"than the corpus size times ceiling ({cap * n_docs:g})"
            )
        # The corpus-level scale stands in when a query's matching scores
        # have no dispersion (fewer than two matches, or all equal).
        corpus_scale = 1.0 / self._transform.alpha
        matched = scores[scores > 0]
        if matched.size == 0:
            # No document matches: every document carries score zero, so
            # every posterior equals the prevalence.
            return calibrate_query(
                np.zeros(1, dtype=np.float64),
                kappa=slope,
                expected_relevant=count,
                ceiling=cap,
                population_size=n_docs,
                scale=corpus_scale,
                context=context,
            )
        return calibrate_query(
            matched,
            kappa=slope,
            expected_relevant=count,
            ceiling=cap,
            population_size=n_docs,
            fallback_scale=corpus_scale,
            context=context,
        )

    def get_log_odds(
        self,
        query_tokens: list[str],
        *,
        expected_relevant: float | None = None,
        kappa: float | None = None,
        ceiling: float | None = None,
    ) -> np.ndarray:
        """Get posterior log-odds for ALL documents (dense array).

        This is the quantity to rank, prune and fuse on.  Under
        ``calibration="corpus"`` a document that matches no query term
        has probability 0, hence log-odds ``-inf``; under
        ``calibration="query"`` it carries the finite log-odds of a
        score of zero.

        Parameters
        ----------
        query_tokens : list of str
            Tokens for a single query.

        Returns
        -------
        log_odds : ndarray of shape (num_docs,)
            Posterior log-odds of relevance for every document.
        """
        if self._transform is None:
            raise RuntimeError("Call index() before get_log_odds().")

        bm25_scores = self._raw_scores(query_tokens)
        if self._calibration == "query":
            return np.asarray(
                self._calibration_from_scores(
                    bm25_scores, expected_relevant=expected_relevant, kappa=kappa, ceiling=ceiling,
                ).log_odds(bm25_scores),
                dtype=np.float64,
            )
        if any(value is not None for value in (expected_relevant, kappa, ceiling)):
            raise ValueError("query parameter overrides require calibration='query'")

        doc_ids = np.arange(len(bm25_scores))
        log_odds = self._scores_to_probabilities(
            doc_ids.reshape(1, -1),
            bm25_scores.reshape(1, -1),
            [query_tokens],
            return_log_odds=True,
        )
        return log_odds.squeeze(0)

    def get_ranking_scores(
        self,
        query_tokens: list[str],
        *,
        expected_relevant: float | None = None,
        kappa: float | None = None,
        ceiling: float | None = None,
    ) -> np.ndarray:
        """Scores for exact ordering before a capped posterior can saturate.

        Query calibration returns latent log-odds. Corpus calibration
        returns the posterior log-odds, including its document prior.
        """
        if self._transform is None:
            raise RuntimeError("Call index() before get_ranking_scores().")
        if self._calibration != "query":
            return self.get_log_odds(
                query_tokens, expected_relevant=expected_relevant, kappa=kappa, ceiling=ceiling,
            )
        scores = self._raw_scores(query_tokens)
        model = self._calibration_from_scores(
            scores, expected_relevant=expected_relevant, kappa=kappa, ceiling=ceiling,
        )
        return np.asarray(model.ranking_score(scores), dtype=np.float64)

    def get_probabilities(
        self,
        query_tokens: list[str],
        *,
        expected_relevant: float | None = None,
        kappa: float | None = None,
        ceiling: float | None = None,
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

        if self._calibration == "query":
            return np.asarray(
                sigmoid(self.get_log_odds(
                    query_tokens, expected_relevant=expected_relevant, kappa=kappa, ceiling=ceiling,
                )), dtype=np.float64,
            )
        if any(value is not None for value in (expected_relevant, kappa, ceiling)):
            raise ValueError("query parameter overrides require calibration='query'")

        bm25_scores = self._raw_scores(query_tokens)
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
        doc_token_sets = self._doc_token_sets
        return np.array(
            [len(query_set & doc_token_sets[did]) for did in doc_ids],
            dtype=np.float64,
        )

    def _scores_to_probabilities(
        self,
        doc_ids: np.ndarray,
        bm25_scores: np.ndarray,
        query_tokens_batch: list[list[str]],
        return_log_odds: bool = False,
    ) -> np.ndarray:
        """Convert raw BM25 scores to probabilities for given doc IDs.

        Corpus-level calibration.  A document with a zero score keeps
        probability 0, which is log-odds ``-inf`` when
        ``return_log_odds`` is True.
        """
        if (
            self._transform is None
            or self._doc_lengths is None
            or self._corpus_tokens is None
            or self._doc_token_sets is None
        ):
            raise RuntimeError("index() must be called before scoring")

        if return_log_odds:
            probabilities = np.full(bm25_scores.shape, -np.inf, dtype=np.float64)
        else:
            probabilities = np.zeros_like(bm25_scores, dtype=np.float64)
        to_value = (
            self._transform.score_to_log_odds
            if return_log_odds
            else self._transform.score_to_probability
        )

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

            probabilities[q_idx, active_mask] = to_value(
                active_scores, tfs, doc_len_ratios
            )

        return probabilities
