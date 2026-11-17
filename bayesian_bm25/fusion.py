#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Probabilistic score combination functions.

Implements AND, OR, NOT, and log-odds conjunction for combining multiple
probability estimates.  The log-odds conjunction (from "From Bayesian
Inference to Neural Computation") resolves the shrinkage problem of
naive probabilistic AND by using geometric-mean log-odds with an
agreement bonus.

All functions operate on numpy arrays and are batch-friendly.
"""

from __future__ import annotations

import numpy as np

from bayesian_bm25.evidence_fusion import posterior_to_evidence
from bayesian_bm25.probability import _clamp_probability, logit, sigmoid
from bayesian_bm25.query_calibration import score_dispersion


def cosine_to_probability(
    score: np.ndarray | float,
) -> np.ndarray | float:
    """Convert cosine similarity to probability (Definition 7.1.2).

    Maps cosine similarity in [-1, 1] to probability in (0, 1) via
    P_vector = (1 + score) / 2, with epsilon clamping for numerical
    stability.

    Parameters
    ----------
    score : float or array
        Cosine similarity score(s) in [-1, 1].

    Returns
    -------
    Probability value(s) in (0, 1).
    """
    score = np.asarray(score, dtype=np.float64)
    result = _clamp_probability((1.0 + score) / 2.0)
    return float(result) if result.ndim == 0 else result


def prob_not(prob: np.ndarray | float) -> np.ndarray | float:
    """Probabilistic NOT via complement rule (Eq. 35).

    Computes P(NOT R) = 1 - P(R).  In log-odds space this corresponds
    to negation: logit(1 - p) = -logit(p), so NOT simply flips the
    sign of evidence.

    Parameters
    ----------
    prob : float or array
        Probability value(s) to negate.

    Returns
    -------
    Complement probability: 1 - p, clamped to (epsilon, 1 - epsilon).
    """
    prob = _clamp_probability(np.asarray(prob, dtype=np.float64))
    result = _clamp_probability(1.0 - prob)
    return float(result) if np.ndim(result) == 0 else result


def prob_and(probs: np.ndarray) -> np.ndarray | float:
    """Probabilistic AND via product rule in log-space (Eq. 33-34).

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.

    Returns
    -------
    Combined probability: exp(sum(ln(p_i))) along the last axis.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    result = np.exp(np.sum(np.log(probs), axis=-1))
    return float(result) if np.ndim(result) == 0 else result


def prob_or(probs: np.ndarray) -> np.ndarray | float:
    """Probabilistic OR via complement rule in log-space (Eq. 36-37).

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.

    Returns
    -------
    Combined probability: 1 - exp(sum(ln(1 - p_i))) along the last axis.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    result = 1.0 - np.exp(np.sum(np.log(1.0 - probs), axis=-1))
    return float(result) if np.ndim(result) == 0 else result


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(sigmoid(x)) = -log(1 + exp(-x)), stable for large |x|."""
    return -np.logaddexp(0.0, -x)


def log_odds_not(logits: np.ndarray | float) -> np.ndarray | float:
    """Probabilistic NOT in log-odds space: logit(1 - p) = -logit(p).

    Exact for every finite input; no clamping is involved.
    """
    result = -np.asarray(logits, dtype=np.float64)
    return float(result) if result.ndim == 0 else result


def log_odds_and(logits: np.ndarray) -> np.ndarray | float:
    """Probabilistic AND (product rule) evaluated in log-odds space.

    Returns ``logit(prod_i p_i)`` for ``p_i = sigmoid(logit_i)`` along
    the last axis.  Working with ``log p_i = log_sigmoid(logit_i)``
    keeps the result strictly increasing in every input for arbitrarily
    large magnitudes, where the probability-valued :func:`prob_and`
    rounds to 0 or 1 and documents tie.

    Parameters
    ----------
    logits : array of shape (..., n)
        Log-odds of the events to combine.  The last axis is reduced.

    Returns
    -------
    Log-odds of the conjunction.
    """
    x = np.asarray(logits, dtype=np.float64)
    log_p = np.sum(_log_sigmoid(x), axis=-1)
    # logit(P) = log P - log(1 - P) with P = exp(log_p) <= 1.
    result = log_p - np.log(-np.expm1(np.minimum(log_p, -np.finfo(np.float64).tiny)))
    return float(result) if np.ndim(result) == 0 else result


def log_odds_or(logits: np.ndarray) -> np.ndarray | float:
    """Probabilistic OR (complement rule) evaluated in log-odds space.

    Returns ``logit(1 - prod_i (1 - p_i))`` for ``p_i = sigmoid(logit_i)``
    along the last axis.  With ``log(1 - p_i) = log_sigmoid(-logit_i)``
    the result stays strictly increasing in every input: the
    probability-valued :func:`prob_or` rounds to exactly 1 once any
    input passes about 36.7, after which the ranking is decided by
    tie-breaking.

    Parameters
    ----------
    logits : array of shape (..., n)
        Log-odds of the events to combine.  The last axis is reduced.

    Returns
    -------
    Log-odds of the disjunction.
    """
    x = np.asarray(logits, dtype=np.float64)
    log_q = np.sum(_log_sigmoid(-x), axis=-1)
    # logit(P) = log(1 - Q) - log Q with Q = exp(log_q) <= 1.
    result = np.log(-np.expm1(np.minimum(log_q, -np.finfo(np.float64).tiny))) - log_q
    return float(result) if np.ndim(result) == 0 else result


def effective_signal_count(
    n_signals: int,
    correlation: float | np.ndarray,
) -> float:
    """Effective number of signals under correlated evidence (Theorem 5.8).

    For ``n`` equally strong signals whose evidence logits are jointly
    Gaussian given relevance, exact fusion multiplies the mean evidence
    by

        n_eff = n / (1 + (n - 1) * r)

    when every pair has correlation ``r``.  Independent signals
    (``r = 0``) give ``n_eff = n``; fully redundant signals (``r = 1``)
    give ``n_eff = 1``.  For a general correlation matrix ``Omega`` the
    multiplier is ``1' Omega^-1 1``, which reduces to the formula above
    under equicorrelation.

    Parameters
    ----------
    n_signals : int
        Number of signals ``n`` (>= 1).
    correlation : float or array of shape (n, n)
        Pairwise correlation ``r`` of the evidence among non-relevant
        documents, or the full correlation matrix.  Estimate it on a
        uniform sample of the corpus, not on the candidate pool: a pool
        formed as the union of each signal's top results induces a
        spurious negative correlation.

    Returns
    -------
    float
        The effective number of signals.
    """
    n = int(n_signals)
    if n < 1:
        raise ValueError(f"n_signals must be >= 1, got {n_signals}")

    if np.ndim(correlation) == 2:
        omega = _validated_correlation_matrix(correlation, n)
        ones = np.ones(n, dtype=np.float64)
        return float(ones @ np.linalg.solve(omega, ones))

    if n == 1:
        return 1.0
    r = float(correlation)
    if not (-1.0 / (n - 1) < r <= 1.0):
        raise ValueError(
            f"correlation must be in (-1/(n-1), 1] = ({-1.0 / (n - 1):g}, 1], got {r}"
        )
    return n / (1.0 + (n - 1) * r)


def confidence_exponent(
    n_signals: int,
    correlation: float | np.ndarray,
) -> float:
    """Confidence exponent ``rho`` implied by a correlation (Theorem 5.8).

    The log-odds conjunction scales the mean evidence by ``n ** rho``.
    That is exact when ``n ** rho`` equals the effective number of
    signals, i.e.

        rho = 1 - log(1 + (n - 1) * r) / log(n).

    ``r = 0`` gives ``rho = 1`` (evidence sums), ``r = 1`` gives
    ``rho = 0`` (the log-odds mean), and the square-root law
    ``rho = 1/2`` corresponds to ``r = 1 / (1 + sqrt(n))``.

    Parameters
    ----------
    n_signals : int
        Number of signals ``n`` (>= 2; the exponent is undefined for a
        single signal, which passes through unchanged for every ``rho``).
    correlation : float or array of shape (n, n)
        Pairwise correlation of the evidence, or the correlation matrix.

    Returns
    -------
    float
        The exponent ``rho``.  It exceeds 1 for negatively correlated
        signals.
    """
    n = int(n_signals)
    if n < 2:
        raise ValueError(
            f"confidence_exponent needs at least two signals, got {n_signals}"
        )
    return float(np.log(effective_signal_count(n, correlation)) / np.log(n))


def _validated_correlation_matrix(correlation: np.ndarray, n: int) -> np.ndarray:
    """Check that ``correlation`` is an n x n positive definite correlation matrix."""
    omega = np.asarray(correlation, dtype=np.float64)
    if omega.shape != (n, n):
        raise ValueError(
            f"correlation matrix must have shape ({n}, {n}), got {omega.shape}"
        )
    if not np.allclose(omega, omega.T, atol=1e-10):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(omega), 1.0, atol=1e-10):
        raise ValueError("correlation matrix must have a unit diagonal")
    try:
        np.linalg.cholesky(omega)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation matrix must be positive definite") from exc
    return omega


def gaussian_evidence_weights(
    strengths: np.ndarray,
    correlation: float | np.ndarray,
) -> np.ndarray:
    """Exact fusion weights for correlated Gaussian evidence (Theorem 5.8).

    If the evidence logits ``e_i`` are each a valid log-likelihood ratio
    and are jointly Gaussian given relevance with standard deviations
    ``tau_i`` and correlation matrix ``Omega``, the joint log-likelihood
    ratio is ``sum_i v_i * e_i`` with

        v = T^-1 Omega^-1 tau,    T = diag(tau).

    The weights multiply the evidences directly; they are not
    normalized to sum to one.  They need not be positive: for two
    signals ``v_2 < 0`` whenever ``r > tau_2 / tau_1``, because a weak
    signal that is highly correlated with a strong one is best used to
    cancel the strong signal's noise.

    Parameters
    ----------
    strengths : array of shape (n,)
        Standard deviations ``tau_i`` of the evidence logits given
        relevance.  For a log-linear calibration with dimensionless
        slope ``kappa_i`` (Definition 3.12), ``tau_i = kappa_i``.
    correlation : float or array of shape (n, n)
        Pairwise correlation shared by all pairs, or the full
        correlation matrix ``Omega``.

    Returns
    -------
    array of shape (n,)
        The weights ``v``.
    """
    tau = np.asarray(strengths, dtype=np.float64).ravel()
    n = tau.size
    if n < 1:
        raise ValueError("strengths must not be empty")
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("strengths must be positive and finite")

    if np.ndim(correlation) == 0:
        r = float(correlation)
        omega = np.full((n, n), r, dtype=np.float64)
        np.fill_diagonal(omega, 1.0)
    else:
        omega = np.asarray(correlation, dtype=np.float64)
    omega = _validated_correlation_matrix(omega, n)

    return np.linalg.solve(omega, tau) / tau


def evidence_correlation(evidence: np.ndarray) -> np.ndarray:
    """Correlation matrix of the columns of an evidence matrix.

    Parameters
    ----------
    evidence : array of shape (m, n)
        Evidence logits of ``m`` documents under ``n`` signals.  The
        correlation that enters Theorem 5.8 is the one among
        non-relevant documents, so a uniform sample of the corpus is the
        right input; no relevance labels are needed.

    Returns
    -------
    array of shape (n, n)
        Pearson correlation matrix.
    """
    x = np.asarray(evidence, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(
            "evidence must be a 2D array with at least two rows, "
            f"got shape {x.shape}"
        )
    # A column that is constant up to rounding has no correlation with
    # anything; np.corrcoef would return noise for it.
    if any(score_dispersion(column) <= 0.0 for column in x.T):
        raise ValueError("every evidence column must have positive variance")
    return np.atleast_2d(np.corrcoef(x, rowvar=False))


_SQRT_N_ALPHA = 0.5  # alpha=0.5 implements the sqrt(n) scaling law (Theorem 4.2.1)


def _resolve_alpha(alpha: float | str | None, default: float) -> float:
    """Resolve alpha parameter: "auto" -> sqrt(n) scaling, None -> default."""
    if alpha is None:
        return default
    if isinstance(alpha, str):
        if alpha != "auto":
            raise ValueError(
                f"alpha must be a float, None, or 'auto', got {alpha!r}"
            )
        return _SQRT_N_ALPHA
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite")
    return value


def _apply_gating(
    logits: np.ndarray, gating: str, beta: float = 1.0,
) -> np.ndarray:
    """Apply sparse-signal gating to logit values before aggregation.

    Parameters
    ----------
    logits : array
        Log-odds values to gate.
    gating : str
        Gating function: "none", "relu", "swish", "gelu", or "softplus".

        - "relu": MAP estimate under sparse prior (Theorem 6.5.3).
          Zeroes out weak/negative evidence: max(0, logit).
        - "swish": Bayes estimate under sparse prior (Theorem 6.7.4).
          Soft gating: logit * sigmoid(beta * logit).  When beta=1.0
          this is the standard swish (Theorem 6.7.6).
        - "gelu": Bayesian expected signal under Gaussian noise model
          (Theorem 6.8.1, Proposition 6.8.2).  Approximated as
          logit * sigmoid(1.702 * logit), which matches Swish_1.702.
          The ``beta`` parameter is ignored for gelu.
        - "softplus": Smooth ReLU that preserves all evidence
          (Remark 6.5.4).  Computes log(1 + exp(beta * logit)) / beta.
          Unlike ReLU, never zeroes out evidence entirely, making it
          suitable for small datasets where discarding any signal is
          costly.  beta=1.0 is the standard softplus; beta -> inf
          approaches ReLU.  Because softplus(x) > x for all finite x,
          it inflates all logits (both positive and negative), producing
          higher fused probabilities than other gating modes.  Consider
          using a lower ``alpha`` to compensate for the increased
          confidence when needed.
    beta : float
        Sharpness parameter for swish and softplus gating.  Controls
        the transition sharpness: beta=1.0 gives the standard form,
        beta -> inf approaches ReLU.  For swish, beta -> 0 approaches
        x/2 (Theorem 6.7.6).  Ignored when gating is "gelu".
    """
    if gating == "none":
        return logits
    if gating == "relu":
        return np.maximum(0.0, logits)
    if gating == "swish":
        return logits * sigmoid(beta * logits)
    if gating == "gelu":
        return logits * sigmoid(1.702 * logits)
    if gating == "softplus":
        return np.logaddexp(0.0, beta * logits) / beta
    raise ValueError(
        f"gating must be 'none', 'relu', 'swish', 'gelu', or 'softplus', "
        f"got {gating!r}"
    )


def log_odds_fusion(
    logits: np.ndarray,
    alpha: float | str | None = None,
    weights: np.ndarray | None = None,
    gating: str = "none",
    gating_beta: float = 1.0,
    max_logit: float | None = None,
    prior_logit: np.ndarray | float | None = None,
    input_prior_log_odds: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Pool evidence logits and return the fused log-odds.

    This is the log-odds conjunction (Definition 5.7) evaluated without
    ever leaving log-odds space:

        fused = prior_logit + n^alpha * sum_i w_i * gate(logit_i),

    with ``w_i = 1/n`` when no weights are given.  ``alpha = 1`` with
    uniform weights is the plain sum of evidence, the Bayes rule for
    conditionally independent signals (Theorem 5.13); ``alpha = 0`` is
    the log-odds mean, a normalized Logarithmic Opinion Pool
    (Theorem 5.6).  :func:`confidence_exponent` gives the exponent
    implied by a measured correlation between signals (Theorem 5.8).

    Unlike :func:`log_odds_conjunction`, neither the inputs nor the
    output pass through a probability, so nothing is clamped or rounded
    and the result stays strictly increasing in every input.  Rank and
    prune on the returned value; apply ``sigmoid`` only to report or
    threshold a probability.

    Parameters
    ----------
    logits : array of shape (..., n)
        Evidence logits to combine.  The last axis is reduced.  Pass
        prior-free evidence (a posterior logit minus its own prior
        logit) together with ``prior_logit`` so that the prior is
        counted once.
    alpha : float, str, or None
        Confidence scaling exponent.  ``"auto"`` resolves to 0.5.  When
        None, uses 0.5 in unweighted mode and 0.0 in weighted mode, as
        :func:`log_odds_conjunction` does.
    weights : array of shape (n,) or None
        Per-signal reliability weights; non-negative, summing to 1.
    gating : str
        Gating applied to each logit before pooling: ``"none"``,
        ``"relu"``, ``"swish"``, ``"gelu"`` or ``"softplus"``.
    gating_beta : float
        Sharpness parameter for swish and softplus gating.
    max_logit : float or None
        Clip each gated logit to ``[-max_logit, +max_logit]`` before
        pooling.  None (default) disables clipping.
    prior_logit : float, array, or None
        Prior log-odds added once after pooling.
    input_prior_log_odds : float, array, or None
        When the inputs are posterior logits, remove each source prior
        before gating and pooling. This requires an explicit target
        ``prior_logit``. None pools the supplied logits as given.

    Returns
    -------
    Fused log-odds.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("logits must have a non-empty signal axis")
    if np.any(np.isnan(x)):
        raise ValueError("logits must not contain NaN")
    if input_prior_log_odds is not None:
        if prior_logit is None:
            raise ValueError("removing source priors requires an explicit target prior_logit")
        x = np.asarray(posterior_to_evidence(x, input_prior_log_odds))
    n = x.shape[-1]
    gated = _apply_gating(x, gating, beta=gating_beta)

    if max_logit is not None:
        gated = np.clip(gated, -max_logit, max_logit)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if weights.shape != (n,) or not np.all(np.isfinite(weights)):
            raise ValueError(
                "weights must be a finite non-negative vector with one value per signal"
            )
        if abs(float(np.sum(weights)) - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1, got {float(np.sum(weights))}"
            )

        effective_alpha = _resolve_alpha(alpha, default=0.0)

        # Log-OP with confidence scaling:
        # n^alpha * sum(w_i * logit_i)  (Theorem 8.3 + Section 4.2)
        contributions = np.zeros_like(gated)
        np.multiply(gated, weights, out=contributions, where=weights != 0.0)
        fused = (n ** effective_alpha) * np.sum(contributions, axis=-1)
    else:
        effective_alpha = _resolve_alpha(alpha, default=0.5)

        # Mean log-odds (Eq. 20) with multiplicative confidence
        # scaling (Eq. 23).
        fused = np.mean(gated, axis=-1) * (n ** effective_alpha)

    if prior_logit is not None:
        fused = fused + np.asarray(prior_logit, dtype=np.float64)

    return float(fused) if np.ndim(fused) == 0 else fused


def log_odds_conjunction(
    probs: np.ndarray,
    alpha: float | str | None = None,
    weights: np.ndarray | None = None,
    gating: str = "none",
    gating_beta: float = 1.0,
    max_logit: float | None = None,
    prior_logit: np.ndarray | float | None = None,
    input_prior_log_odds: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Log-odds conjunction with multiplicative confidence scaling (Paper 2, Section 4).

    Resolves the shrinkage problem of naive probabilistic AND by:
      1. Computing the mean log-odds (Eq. 20)
      2. Multiplicative confidence scaling by n^alpha (Eq. 23)
      3. Converting back to probability via sigmoid (Eq. 26)

    When ``weights`` are provided, uses the Log-OP (Log-linear Opinion
    Pool) formulation from Paper 2, Theorem 8.3 / Remark 8.4:
    sigma(n^alpha * sum(w_i * logit(P_i))) where sum(w_i) = 1 and
    w_i >= 0.  Per-signal weights (Theorem 8.3) and confidence scaling
    by signal count (Section 4.2) are orthogonal and compose
    multiplicatively.

    The multiplicative formulation (rather than additive) preserves the
    sign of evidence (Theorem 4.2.2), preventing accidental inversion
    of irrelevance signals (Remark 4.2.4).  Working directly in log-odds
    space avoids the nonlinear residual introduced by geometric-mean
    aggregation in probability space (Remark 4.1.3).

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.
    alpha : float, str, or None
        Confidence scaling exponent.  Higher values amplify the effect
        of multiple agreeing signals.  ``"auto"`` resolves to 0.5
        (sqrt(n) scaling law, Theorem 4.2.1).  When None (default),
        uses 0.5 in unweighted mode and 0.0 in weighted mode,
        preserving backward compatibility for both calling conventions.
    weights : array of shape (n,) or None
        Per-signal reliability weights for the Log-OP formulation.
        Must be non-negative and sum to 1.  When None (default),
        uses the unweighted mean log-odds with n^alpha scaling.
    gating : str
        Sparse-signal gating applied to logit values before aggregation.

        - ``"none"`` (default): no gating.
        - ``"relu"``: MAP under sparse prior (Theorem 6.5.3): max(0, logit).
        - ``"swish"``: Bayes under sparse prior (Theorem 6.7.4):
          logit * sigmoid(beta * logit).
        - ``"gelu"``: Bayesian expected signal under Gaussian noise model
          (Theorem 6.8.1): logit * sigmoid(1.702 * logit).
        - ``"softplus"``: smooth ReLU preserving all evidence
          (Remark 6.5.4): log(1 + exp(beta * logit)) / beta.
          Inflates all logits (softplus(x) > x), so consider a lower
          ``alpha`` to temper the increased confidence.
    gating_beta : float
        Sharpness parameter for swish and softplus gating.  Only used
        when ``gating="swish"`` or ``gating="softplus"``.  Default 1.0
        preserves existing behavior.
    max_logit : float or None
        Maximum absolute logit value.  Logits are clipped to
        [-max_logit, +max_logit] after gating and before scaling.
        Prevents sigmoid saturation that destroys ranking
        discrimination when input probabilities are extreme
        (e.g., density-ratio calibrated vector scores near 0 or 1).
        None (default) disables clipping for backward compatibility.

    Returns
    -------
    Combined probability after log-odds conjunction.

    Notes
    -----
    The inputs are clamped to [epsilon, 1 - epsilon] and the output is a
    probability, so candidates whose evidence exceeds the clamp tie.
    Use :func:`log_odds_fusion` to rank, prune or fuse on log-odds
    directly.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))

    # Steps 1-2: pooled log-odds with confidence scaling (Eq. 20, 23).
    fused = log_odds_fusion(
        logit(probs),
        alpha=alpha,
        weights=weights,
        gating=gating,
        gating_beta=gating_beta,
        max_logit=max_logit,
        prior_logit=prior_logit,
        input_prior_log_odds=input_prior_log_odds,
    )

    # Step 3: back to probability (Eq. 26)
    result = sigmoid(fused)
    return float(result) if np.ndim(result) == 0 else result


def balanced_log_odds_fusion(
    sparse_probs: np.ndarray,
    dense_similarities: np.ndarray,
    weight: float = 0.5,
    *,
    from_log_odds: bool = False,
) -> np.ndarray | float:
    """Balanced log-odds fusion for hybrid sparse-dense retrieval.

    Combines Bayesian BM25 probabilities with dense cosine similarities
    by normalizing both signals in logit space.  Min-max normalization
    ensures each signal contributes equally, preventing the heavy-tailed
    sparse logits (from sigmoid unwrapping) from drowning the dense signal.

    Pipeline:
      1. sparse_probs -> logit(p_sparse)
      2. dense_similarities -> cosine_to_probability -> logit(p_dense)
      3. Min-max normalize each logit array to [0, 1]
      4. Return weight * logit_dense_norm + (1 - weight) * logit_sparse_norm

    Parameters
    ----------
    sparse_probs : array
        Bayesian BM25 probabilities in (0, 1), e.g. from
        ``BayesianBM25Scorer.get_probabilities()``.
    dense_similarities : array
        Cosine similarities in [-1, 1] from a dense encoder.
    weight : float
        Weight for the dense signal (default 0.5 for equal weighting).
        The sparse weight is ``1 - weight``.
    from_log_odds : bool
        When True, ``sparse_probs`` holds posterior log-odds instead of
        probabilities (for example from
        ``BayesianProbabilityTransform.score_to_log_odds()``), and no
        clamping is applied to it.  A probability input saturates at
        the clamp, about 23.0 in log-odds; candidates beyond it tie and
        the lexical signal stops discriminating among them.

    Returns
    -------
    Fusion scores (not probabilities).  Higher is more relevant.

    Notes
    -----
    When all values in a signal are identical (zero variance), the
    ``_min_max_normalize`` step maps that signal to all zeros.  This
    means a zero-variance signal contributes nothing to the fusion
    result -- only the other signal determines the final ranking.
    """
    sparse_probs = np.asarray(sparse_probs, dtype=np.float64)
    dense_similarities = np.asarray(dense_similarities, dtype=np.float64)

    logit_sparse = _to_log_odds(sparse_probs, from_log_odds)
    logit_dense = logit(cosine_to_probability(dense_similarities))

    logit_sparse_norm = _min_max_normalize(logit_sparse)
    logit_dense_norm = _min_max_normalize(logit_dense)

    result = weight * logit_dense_norm + (1.0 - weight) * logit_sparse_norm
    return float(result) if np.ndim(result) == 0 else result


def _to_log_odds(
    values: np.ndarray | float,
    from_log_odds: bool,
    input_prior_log_odds: np.ndarray | float | None = None,
) -> np.ndarray:
    """Return the input as log-odds.

    When ``from_log_odds`` is True the values are already log-odds and
    are passed through untouched.  Otherwise they are probabilities,
    which are clamped to [epsilon, 1 - epsilon] before the logit.
    """
    values = np.asarray(values, dtype=np.float64)
    result = values if from_log_odds else np.asarray(logit(_clamp_probability(values)))
    if input_prior_log_odds is not None:
        result = np.asarray(posterior_to_evidence(result, input_prior_log_odds))
    return result


def _validated_input_priors(
    input_prior_log_odds: np.ndarray | float | None,
    n_signals: int,
    base_rate: float | None,
) -> np.ndarray | None:
    """Validate source priors and require the target prior for evidence fusion."""
    if input_prior_log_odds is None:
        return None
    if base_rate is None:
        raise ValueError("input_prior_log_odds requires an explicit target base_rate")
    priors = np.asarray(input_prior_log_odds, dtype=np.float64)
    if priors.shape not in ((), (n_signals,)) or not np.all(np.isfinite(priors)):
        raise ValueError("input_prior_log_odds must be finite with one value per signal")
    return priors.copy()


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].  Returns zeros if range is negligible."""
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def __getattr__(name: str):
    """Preserve class imports while keeping one main class per module."""
    if name == "LearnableLogOddsWeights":
        from bayesian_bm25.learnable_log_odds_weights import LearnableLogOddsWeights
        return LearnableLogOddsWeights
    if name == "AttentionLogOddsWeights":
        from bayesian_bm25.attention_log_odds_weights import AttentionLogOddsWeights
        return AttentionLogOddsWeights
    if name == "MultiHeadAttentionLogOddsWeights":
        from bayesian_bm25.multi_head_attention_log_odds_weights import (
            MultiHeadAttentionLogOddsWeights,
        )
        return MultiHeadAttentionLogOddsWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
