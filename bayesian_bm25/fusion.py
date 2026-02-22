#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Probabilistic score combination functions.

Implements AND, OR, and log-odds conjunction for combining multiple
probability estimates.  The log-odds conjunction (from "From Bayesian
Inference to Neural Computation") resolves the shrinkage problem of
naive probabilistic AND by using geometric-mean log-odds with an
agreement bonus.

All functions operate on numpy arrays and are batch-friendly.
"""

from __future__ import annotations

import numpy as np

from bayesian_bm25.probability import _EPSILON, _clamp_probability, logit, sigmoid


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


def log_odds_conjunction(
    probs: np.ndarray,
    alpha: float | None = None,
    weights: np.ndarray | None = None,
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
    alpha : float or None
        Confidence scaling exponent.  Higher values amplify the effect
        of multiple agreeing signals.  When None (default), uses 0.5
        in unweighted mode and 0.0 in weighted mode, preserving
        backward compatibility for both calling conventions.
    weights : array of shape (n,) or None
        Per-signal reliability weights for the Log-OP formulation.
        Must be non-negative and sum to 1.  When None (default),
        uses the unweighted mean log-odds with n^alpha scaling.

    Returns
    -------
    Combined probability after log-odds conjunction.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    n = probs.shape[-1]

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if abs(float(np.sum(weights)) - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1, got {float(np.sum(weights))}"
            )

        effective_alpha = 0.0 if alpha is None else alpha

        # Log-OP with confidence scaling:
        # sigma(n^alpha * sum(w_i * logit(P_i)))  (Theorem 8.3 + Section 4.2)
        l_weighted = (n ** effective_alpha) * np.sum(
            weights * logit(probs), axis=-1
        )
        result = sigmoid(l_weighted)
        return float(result) if np.ndim(result) == 0 else result

    effective_alpha = 0.5 if alpha is None else alpha

    # Step 1: mean log-odds (Eq. 20)
    l_bar = np.mean(logit(probs), axis=-1)

    # Step 2: multiplicative confidence scaling (Eq. 23)
    l_adjusted = l_bar * (n ** effective_alpha)

    # Step 3: back to probability (Eq. 26)
    result = sigmoid(l_adjusted)
    return float(result) if np.ndim(result) == 0 else result
