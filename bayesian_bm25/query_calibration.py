#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Per-query calibration of a scalar retrieval score.

Implements Section 3 of "From Scores to Evidence":

* the standardized slope ``alpha_q = kappa / sigma_q`` (Definition 3.12),
  which makes the evidence equivariant to the scale of each query's
  scores (Proposition 3.11);
* the consistent intercept (Theorem 3.7), which fixes the offset of the
  posterior log-odds by the count constraint
  ``sum_d P(R=1 | s_d) = R_q``;
* Procedure 3.14, which combines the two.

The latent ranking score is ``alpha_q * s - c_q``. The posterior is
``ceiling * sigmoid(alpha_q * s - c_q)`` (Definition 3.23). At ceiling
one the ranking score is also the posterior log-odds; otherwise the
posterior log-odds and evidence are bounded, nonlinear functions of it.
The latent score preserves ranking without finite-precision saturation
and maps a score upper bound to a safe pruning bound.

Nothing here is specific to BM25.  Applied to a cosine similarity the
same procedure gives the monotone vector calibration of Section 4.8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.count_calibration import CountCalibration, calibrate_count, ceiling_log_odds
from bayesian_bm25.probability import sigmoid

_VALID_DISPERSIONS = ("std", "mad", "iqr")
_VALID_INTERCEPTS = ("exact", "dilute")

# Consistency factors that make the robust dispersions estimate the
# standard deviation of a Gaussian sample.
_MAD_TO_STD = 1.482602218505602
_IQR_TO_STD = 1.3489795003921634

def score_dispersion(scores: np.ndarray, method: str = "std") -> float:
    """Dispersion ``sigma_q`` of a query's scores (Definition 3.12).

    Parameters
    ----------
    scores : array
        Scores of the documents that match the query.  Documents that
        match no query term must be left out: a distribution with a large
        atom at zero has a standard deviation that reflects how many
        documents match, not how scores vary among them.
    method : str
        ``"std"`` (standard deviation), ``"mad"`` (median absolute
        deviation, scaled to the standard deviation of a Gaussian), or
        ``"iqr"`` (interquartile range, scaled likewise).

    Returns
    -------
    float
        The dispersion, or 0.0 when fewer than two scores are given or
        all of them are equal.  Scores that differ only by rounding
        error count as equal.
    """
    if method not in _VALID_DISPERSIONS:
        raise ValueError(
            f"method must be one of {_VALID_DISPERSIONS}, got {method!r}"
        )
    s = np.asarray(scores, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    # Test literal variation before taking a mean: equal float64 values
    # can have a nonzero np.std from mean-rounding alone. A tolerance
    # relative to the raw score location would break shift equivariance.
    if s.size < 2 or float(np.ptp(s)) == 0.0:
        return 0.0
    if method == "std":
        dispersion = float(np.std(s))
    elif method == "mad":
        median = float(np.median(s))
        dispersion = _MAD_TO_STD * float(np.median(np.abs(s - median)))
    else:
        q75, q25 = np.percentile(s, [75.0, 25.0])
        dispersion = float(q75 - q25) / _IQR_TO_STD
    return dispersion


@dataclass(frozen=True)
class QueryCalibration:
    """Calibration of one query's scores (Procedure 3.14).

    The latent ranking score is ``slope * s - offset`` and the posterior
    is ``ceiling * sigmoid(ranking_score)``. Only at ceiling one are the
    posterior log-odds affine, with
    ``offset = slope * midpoint - logit(prevalence)``.

    Attributes
    ----------
    slope : float
        Slope of evidence in the score, ``alpha_q = kappa / scale``.
    offset : float
        Offset ``c_q`` of the posterior log-odds.
    prevalence : float
        Prior probability of relevance ``pi_q = R_q / N_q``.
    expected_relevant : float
        Expected number of relevant documents ``R_q``.
    population_size : float
        Size ``N_q`` of the reference population.
    scale : float
        Dispersion ``sigma_q`` used to standardize the score.
    kappa : float
        Dimensionless slope ``kappa`` (the discriminability index under
        the equal-variance Gaussian reading of Section 3.1).
    """

    slope: float
    offset: float
    prevalence: float
    expected_relevant: float
    population_size: float
    scale: float
    kappa: float
    ceiling: float = 1.0
    context: CalibrationContext = field(default_factory=CalibrationContext)

    @property
    def count_calibration(self) -> CountCalibration:
        """The shared count model applied after the score's linear term."""
        return CountCalibration(
            self.offset, self.expected_relevant, self.population_size, self.ceiling, self.context,
        )

    @property
    def prior_log_odds(self) -> float:
        """Log-odds of the prevalence, ``logit(pi_q)``."""
        return self.count_calibration.prior_log_odds

    @property
    def midpoint(self) -> float:
        """Score ``beta_q`` at which the evidence changes sign."""
        latent_prior = math.log(self.expected_relevant) - math.log(
            self.ceiling * self.population_size - self.expected_relevant
        )
        return (self.offset + latent_prior) / self.slope

    @property
    def even_odds_score(self) -> float:
        """Score ``theta_q`` at which a document is as likely relevant as not.

        No finite score reaches one half when ``ceiling <= 0.5``;
        that case raises ``ValueError``.
        """
        if self.ceiling <= 0.5:
            raise ValueError("no finite even-odds score exists when ceiling <= 0.5")
        return (self.offset - math.log(2.0 * self.ceiling - 1.0)) / self.slope

    def ranking_score(
        self,
        scores: np.ndarray | float,
        document_prior_log_odds: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        """Latent ranking score ``slope * s - offset``.

        Parameters
        ----------
        scores : float or array
            Scores of the documents to calibrate.
        document_prior_log_odds : float, array, or None
            Optional document-quality prior ``logit P_0(d)`` appended as
            one further additive term (Definition 3.15).  Pass the same
            values that were given to :func:`calibrate_query`, so that
            the offset matches.

        Returns
        -------
        Latent score for ranking and pruning. The document prior may
        re-rank relative to BM25. Use :meth:`evidence` for fusion of
        modeled likelihood ratios when a ceiling is present.
        """
        s = np.asarray(scores, dtype=np.float64)
        result = self.slope * s - self.offset
        if document_prior_log_odds is not None:
            result = result + np.asarray(document_prior_log_odds, dtype=np.float64)
        return float(result) if np.ndim(result) == 0 else result

    def log_odds(
        self,
        scores: np.ndarray | float,
        document_prior_log_odds: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        """Actual posterior log-odds, including the relevance ceiling.

        Retain :meth:`ranking_score` for ordering: capped log-odds can
        reach their floating-point upper limit even in log space.
        """
        return ceiling_log_odds(
            self.ranking_score(scores, document_prior_log_odds), self.ceiling,
        )

    def evidence(
        self,
        scores: np.ndarray | float,
        document_prior_log_odds: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        """Modeled likelihood-ratio evidence, with the prior removed.

        The evidence is the posterior log-odds with the prior removed.
        It is the quantity to add across conditionally independent
        signals, with the prior added once (Theorem 5.13).
        """
        result = np.asarray(self.log_odds(scores, document_prior_log_odds)) - self.prior_log_odds
        return float(result) if np.ndim(result) == 0 else result

    def probability(
        self,
        scores: np.ndarray | float,
        document_prior_log_odds: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        """Posterior ``ceiling * sigmoid(ranking_score)``.

        For reporting and thresholding only.  At most
        ``expected_relevant / p`` documents of the reference population
        can exceed probability ``p`` (Theorem 3.7).
        """
        result = self.ceiling * np.asarray(
            sigmoid(self.ranking_score(scores, document_prior_log_odds))
        )
        return float(result) if result.ndim == 0 else result

    def upper_bound_ranking_score(
        self,
        score_upper_bound: np.ndarray | float,
        document_prior_upper_bound: float | None = None,
    ) -> np.ndarray | float:
        """Exact bound on latent ranking scores (Theorem 3.20).

        A BM25 score upper bound ``U`` maps to the log-odds upper bound
        ``slope * U - offset``, valid for every document with score at
        most ``U``.  With a document-quality prior, pass its maximum
        ``max_d P_0(d)`` to keep the bound exact.

        Parameters
        ----------
        score_upper_bound : float or array
            Upper bound(s) on the score, for example a WAND term bound
            or a Block-Max WAND block maximum.
        document_prior_upper_bound : float or None
            Maximum of the document-quality prior, in (0, 1).

        Returns
        -------
        Upper bound(s) on the posterior log-odds.
        """
        u = np.asarray(score_upper_bound, dtype=np.float64)
        result = self.slope * u - self.offset
        if document_prior_upper_bound is not None:
            p_max = float(document_prior_upper_bound)
            if not (0.0 < p_max < 1.0):
                raise ValueError(
                    "document_prior_upper_bound must be in (0, 1), "
                    f"got {document_prior_upper_bound}"
                )
            result = result + (math.log(p_max) - math.log1p(-p_max))
        return float(result) if np.ndim(result) == 0 else result

    def upper_bound_log_odds(
        self,
        score_upper_bound: np.ndarray | float,
        document_prior_upper_bound: float | None = None,
    ) -> np.ndarray | float:
        """Safe bound on actual posterior log-odds after the ceiling."""
        return ceiling_log_odds(
            self.upper_bound_ranking_score(score_upper_bound, document_prior_upper_bound),
            self.ceiling,
        )


def calibrate_query(
    scores: np.ndarray,
    *,
    kappa: float | None = None,
    expected_relevant: float | None = None,
    ceiling: float | None = None,
    population_size: float | None = None,
    unscored_score: float = 0.0,
    dispersion: str = "std",
    scale: float | None = None,
    fallback_scale: float | None = None,
    match_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    document_prior_log_odds: np.ndarray | None = None,
    unscored_prior_log_odds: float = 0.0,
    intercept: str = "exact",
    context: CalibrationContext | None = None,
) -> QueryCalibration:
    """Calibrate the scores of one query (Procedure 3.14).

    1. ``sigma_q`` is the dispersion of the matching scores and
       ``alpha_q = kappa / sigma_q`` (Definition 3.12).
    2. The offset ``c_q`` is the root of the count constraint
       ``sum_d sigmoid(alpha_q * s_d - c_q) = R_q / ceiling`` over the
       reference population (Corollary 3.24).

    Parameters
    ----------
    scores : array
        Scores of the scored documents of the reference population.  For
        BM25 these are the documents that match at least one query term;
        for a similarity they are all documents.
    kappa : float
        Dimensionless slope shared across queries; must be positive.
        The slope, count, and ceiling cannot be identified from scores
        alone (Propositions 3.13 and 3.26).
    expected_relevant : float
        Expected number of relevant documents ``R_q`` in the reference
        population.  In the dilute regime every posterior is
        proportional to it (Eq. 3.4).
    ceiling : float
        Posterior upper asymptote in (0, 1]. The count must be less than
        ``ceiling * population_size``. A lower ceiling redistributes
        mass from the head while preserving the supplied total.
    population_size : float or None
        Size ``N_q`` of the reference population.  When it exceeds the
        number of scored documents, the remainder are documents whose
        score is ``unscored_score``; they enter the count constraint in
        closed form.  None means the population is exactly the scored
        documents.
    unscored_score : float
        Known common score of documents that were not scored (0 for a
        lexical non-match, Remark 3.10). This is not an estimate of
        unknown dense or pruned scores; use a weighted tail sample for
        those populations.
    dispersion : str
        Dispersion statistic: ``"std"``, ``"mad"`` or ``"iqr"``.
    scale : float or None
        Explicit ``sigma_q``.  When given, ``dispersion`` is not used.
        This is how index-derived statistics enter (Section 3.4).
    fallback_scale : float or None
        ``sigma_q`` to use when the matching scores have no dispersion
        (fewer than two matching documents, or all scores equal), for
        example a corpus-level scale.  Without it such a query raises
        ``ValueError``.
    match_mask : boolean array or None
        Which entries of ``scores`` count as matching documents for the
        dispersion.  None means all of them.  Every entry of ``scores``
        enters the count constraint regardless of the mask.
    weights : array or None
        Multiplicity of each entry of ``scores`` in the count
        constraint.  Use it when the population is represented by the
        retrieved list plus a weighted sample of the tail.
    document_prior_log_odds : array or None
        Document-quality prior ``logit P_0(d)`` of the scored documents
        (Definition 3.15).  The offset is then solved with the prior
        term included.
    unscored_prior_log_odds : float
        ``logit P_0`` shared by the documents that were not scored.
    intercept : str
        ``"exact"`` solves the count constraint; ``"dilute"`` returns
        the closed form of Corollary 3.8, which overestimates the offset
        whenever some posterior is not small.

    Returns
    -------
    QueryCalibration
        The slope, the offset and the quantities they were derived from.
    """
    if context is None:
        context = CalibrationContext(
            count_source="convention" if expected_relevant is None else "supplied",
            slope_source="convention" if kappa is None else "supplied",
            ceiling_source="convention" if ceiling is None else "supplied",
        )
    kappa = 1.0 if kappa is None else float(kappa)
    expected_relevant = 1.0 if expected_relevant is None else float(expected_relevant)
    ceiling = 1.0 if ceiling is None else float(ceiling)
    if intercept not in _VALID_INTERCEPTS:
        raise ValueError(
            f"intercept must be one of {_VALID_INTERCEPTS}, got {intercept!r}"
        )
    if not (kappa > 0.0) or not math.isfinite(kappa):
        raise ValueError(f"kappa must be positive and finite, got {kappa}")
    if not math.isfinite(ceiling) or not (0.0 < ceiling <= 1.0):
        raise ValueError("ceiling must be finite and in (0, 1]")

    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("scores must not be empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")

    if weights is None:
        w = None
        n_scored = float(s.size)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != s.shape:
            raise ValueError(
                f"weights shape {w.shape} does not match scores shape {s.shape}"
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0.0) or not np.any(w > 0.0):
            raise ValueError("weights must be finite and non-negative with a positive sum")
        n_scored = float(np.sum(w))
        if not math.isfinite(n_scored):
            raise ValueError("weights must have a finite sum")

    if population_size is None:
        n_population = n_scored
    else:
        n_population = float(population_size)
        if not math.isfinite(n_population):
            raise ValueError("population_size must be finite")
        if n_population < n_scored:
            raise ValueError(
                f"population_size {population_size} is smaller than the "
                f"number of scored documents {n_scored:g}"
            )
    n_unscored = n_population - n_scored

    r = float(expected_relevant)
    if not math.isfinite(r) or not (0.0 < r < ceiling * n_population):
        raise ValueError(
            "expected_relevant must lie strictly between 0 and the population "
            f"size times ceiling ({ceiling * n_population:g}), got {expected_relevant}"
        )

    # Step 1: standardized slope.
    if scale is None:
        matching = s if match_mask is None else s[np.asarray(match_mask, dtype=bool)]
        sigma_q = score_dispersion(matching, dispersion)
        if not (sigma_q > 0.0):
            if fallback_scale is None:
                raise ValueError(
                    "the matching scores have no dispersion, so the standardized "
                    "slope is undefined; pass scale or fallback_scale"
                )
            sigma_q = float(fallback_scale)
    else:
        sigma_q = float(scale)
    if not (sigma_q > 0.0) or not math.isfinite(sigma_q):
        raise ValueError(f"the score scale must be positive and finite, got {sigma_q}")
    alpha_q = kappa / sigma_q

    # Step 2: offset from the count constraint over the whole population.
    g = alpha_q * s
    if document_prior_log_odds is not None:
        prior = np.asarray(document_prior_log_odds, dtype=np.float64).ravel()
        if prior.shape != s.shape:
            raise ValueError(
                f"document_prior_log_odds shape {prior.shape} does not match "
                f"scores shape {s.shape}"
            )
        g = g + prior

    if n_unscored > 0.0:
        g = np.append(g, alpha_q * float(unscored_score) + float(unscored_prior_log_odds))
        multiplicity = np.ones(s.size, dtype=np.float64) if w is None else w
        g_weights: np.ndarray | None = np.append(multiplicity, n_unscored)
    else:
        g_weights = w

    count_calibration = calibrate_count(
        g, r, ceiling=ceiling, weights=g_weights, intercept=intercept, context=context,
    )

    return QueryCalibration(
        slope=alpha_q,
        offset=count_calibration.offset,
        prevalence=r / n_population,
        expected_relevant=r,
        population_size=n_population,
        scale=sigma_q,
        kappa=float(kappa),
        ceiling=float(ceiling),
        context=count_calibration.context,
    )
