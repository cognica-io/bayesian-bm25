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
    alpha: float = 0.5,
) -> np.ndarray | float:
    """Log-odds conjunction with agreement bonus (Paper 2, Section 4).

    Resolves the shrinkage problem of naive probabilistic AND by:
      1. Computing the geometric mean in probability space (Eq. 20)
      2. Converting to log-odds and adding an agreement bonus (Eq. 21)
      3. Converting back to probability (Eq. 24)

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.
    alpha : float
        Agreement bonus scaling factor.  Higher values give more weight
        to the number of agreeing signals.

    Returns
    -------
    Combined probability after log-odds conjunction.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    n = probs.shape[-1]

    # Step 1: geometric mean -- exp(mean(log(p_i)))
    geometric_mean = np.exp(np.mean(np.log(probs), axis=-1))

    # Step 2: log-odds with agreement bonus
    log_odds = logit(geometric_mean) + alpha * np.log(n)

    # Step 3: back to probability
    result = sigmoid(log_odds)
    return float(result) if np.ndim(result) == 0 else result
