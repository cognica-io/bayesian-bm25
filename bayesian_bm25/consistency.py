#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Marginal consistency of a calibrated posterior (the count constraint).

A posterior over a reference population must average to the prior
(Lemma 2.8 of "From Scores to Evidence").  For a
log-odds model ``logit P_d = g_d - c`` this pins the offset ``c``: it is
the unique root of

    sum_d w_d * sigmoid(g_d - c) = R,

where ``R`` is the expected number of relevant documents in the
population (Theorem 3.7).  The left side is strictly decreasing in
``c``, so the root exists and is unique whenever ``0 < R < sum_d w_d``.

The same one-dimensional problem restores calibration in the large after
any pooling of evidence (post-fusion consistency, Section 5.3).

All functions are numpy-only and work on log-odds, so that no
probability is ever rounded to 0 or 1 before the root is found
(Remark 3.19).
"""

from __future__ import annotations

import math

import numpy as np

from bayesian_bm25.probability import sigmoid


def _as_population(
    log_odds: np.ndarray | float,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Validate a weighted population of log-odds values.

    Returns the flattened log-odds, the flattened weights (or None for
    unit weights), and the total weight (the population size).
    """
    g = np.asarray(log_odds, dtype=np.float64).ravel()
    if g.size == 0:
        raise ValueError("log_odds must not be empty")
    if not np.all(np.isfinite(g)):
        raise ValueError("log_odds must be finite")

    if weights is None:
        return g, None, float(g.size)

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != g.shape:
        raise ValueError(
            f"weights shape {w.shape} does not match log_odds shape {g.shape}"
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(w))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a positive finite sum")
    # Zero-mass entries are outside the population. In particular, their
    # logits must not set the log-sum-exp shift or the root bracket.
    active = w > 0.0
    if not np.all(active):
        g, w = g[active], w[active]
    return g, w, total


def _log_sum_exp(g: np.ndarray, w: np.ndarray | None) -> float:
    """log sum_d w_d * exp(g_d), computed without overflow."""
    g_max = float(np.max(g))
    shifted = np.exp(g - g_max)
    total = float(np.sum(shifted)) if w is None else float(np.sum(w * shifted))
    return g_max + math.log(total)


def expected_count(
    log_odds: np.ndarray | float,
    offset: float = 0.0,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Expected number of relevant documents: sum_d w_d * sigmoid(g_d - c).

    Parameters
    ----------
    log_odds : array
        Log-odds ``g_d`` of the documents of the reference population,
        before the offset is subtracted.
    offset : float
        Offset ``c`` subtracted from every log-odds value.
    weights : array or None
        Multiplicity of each entry.  Use it to represent many documents
        that share one value, such as the documents that match no query
        term, without materialising them.  None means unit weights.

    Returns
    -------
    float
        The sum of the posterior probabilities over the population.
    """
    g, w, _ = _as_population(log_odds, weights)
    p = np.asarray(sigmoid(g - float(offset)), dtype=np.float64)
    return float(np.sum(p)) if w is None else float(np.sum(w * p))


def dilute_offset(
    log_odds: np.ndarray | float,
    expected_relevant: float,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Offset in the dilute limit (Corollary 3.8).

    When no posterior is large, ``sigmoid(x)`` is close to ``exp(x)`` and
    the count constraint reduces to

        c = log sum_d w_d * exp(g_d) - log R,

    so that calibration is ``R`` times a softmax over the population.
    Because ``sigmoid(x) <= exp(x)``, this value is never below the exact
    root of :func:`solve_offset`; it is exact only in the limit.

    Parameters
    ----------
    log_odds : array
        Log-odds ``g_d`` before the offset is subtracted.
    expected_relevant : float
        Expected number of relevant documents ``R``; must be positive.
    weights : array or None
        Multiplicity of each entry (None means unit weights).

    Returns
    -------
    float
        The dilute-limit offset.
    """
    g, w, _ = _as_population(log_odds, weights)
    r = float(expected_relevant)
    if not (r > 0.0) or not math.isfinite(r):
        raise ValueError(
            f"expected_relevant must be positive and finite, got {expected_relevant}"
        )
    return _log_sum_exp(g, w) - math.log(r)


def solve_offset(
    log_odds: np.ndarray | float,
    expected_relevant: float,
    *,
    weights: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Solve the count constraint for the offset (Theorem 3.7).

    Finds the unique ``c`` with

        sum_d w_d * sigmoid(g_d - c) = R.

    The iteration is a Newton step on ``log F(c) - log R``, where
    ``F`` is the left side: in the dilute regime ``log F`` is nearly
    linear in ``c``, so the step is nearly exact.  Every step is
    safeguarded by a bracket that is known to contain the root,

        g_min - logit(R / W)  <=  c  <=  min(g_max - logit(R / W), c_dilute),

    with ``W`` the population size and ``c_dilute`` the value of
    :func:`dilute_offset`. Steps outside the bracket and steps that fail
    to halve the residual are replaced by bisection. This prevents cycles
    across plateaus between well-separated groups of scores.

    Parameters
    ----------
    log_odds : array
        Log-odds ``g_d`` of the documents of the reference population,
        before the offset is subtracted (for example ``alpha * s_d``).
    expected_relevant : float
        Expected number of relevant documents ``R``.  Must satisfy
        ``0 < R < W`` where ``W`` is the population size.
    weights : array or None
        Multiplicity of each entry (None means unit weights).
    tolerance : float
        Convergence threshold on ``|log F(c) - log R|``, the relative
        error of the count.
    max_iterations : int
        Maximum number of safeguarded Newton iterations.

    Returns
    -------
    float
        The offset ``c``.  The calibrated posterior log-odds are
        ``g_d - c``.
    """
    g, w, total = _as_population(log_odds, weights)
    r = float(expected_relevant)
    if not math.isfinite(r) or not (0.0 < r < total):
        raise ValueError(
            "expected_relevant must lie strictly between 0 and the population "
            f"size {total:g}, got {expected_relevant}"
        )
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    log_r = math.log(r)
    prior_logit = log_r - math.log(total - r)
    lo = float(np.min(g)) - prior_logit
    hi = float(np.max(g)) - prior_logit
    hi = min(hi, _log_sum_exp(g, w) - log_r)
    if hi <= lo:
        # Every document carries the same log-odds: the bracket is a point.
        return lo

    c = hi
    previous_residual = math.inf
    for _ in range(max_iterations):
        p = np.asarray(sigmoid(g - c), dtype=np.float64)
        if w is None:
            count = float(np.sum(p))
            slope = float(np.sum(p * (1.0 - p)))
        else:
            count = float(np.sum(w * p))
            slope = float(np.sum(w * p * (1.0 - p)))

        if count <= 0.0:
            # Every posterior underflowed: the offset is far too large.
            hi = c
            c = 0.5 * (lo + hi)
            previous_residual = math.inf
            continue

        residual = math.log(count) - log_r
        if abs(residual) <= tolerance:
            return c

        if residual > 0.0:
            lo = c
        else:
            hi = c

        # d/dc log F(c) = -slope / count, so the Newton step is
        # c + residual * count / slope.
        c_next = c + residual * count / slope if slope > 0.0 else math.nan
        if not (lo < c_next < hi) or abs(residual) > 0.5 * previous_residual:
            c_next = 0.5 * (lo + hi)
        previous_residual = abs(residual)

        if hi - lo <= 4.0 * np.finfo(np.float64).eps * max(1.0, abs(c_next)):
            # The bracket has collapsed to floating-point resolution.
            return c_next
        c = c_next

    raise RuntimeError(
        f"solve_offset did not converge in {max_iterations} iterations "
        f"(bracket [{lo}, {hi}])"
    )
