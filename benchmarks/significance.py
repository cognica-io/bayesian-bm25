#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Uncertainty estimates for per-query retrieval metrics.

Queries are the sampling unit: a dataset's score is a mean over
queries, so confidence intervals come from resampling queries and
comparisons between two systems are paired on the query.

* :func:`bootstrap_ci` -- percentile bootstrap interval for a mean.
* :func:`paired_bootstrap_ci` -- interval for a difference of means.
* :func:`paired_randomization_test` -- Fisher randomization (sign-flip)
  test of the null hypothesis that two systems are exchangeable.
* :func:`holm_correction` -- Holm step-down adjustment for a family of
  comparisons.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array of shape (n_queries,)
        Per-query metric values.
    n_resamples : int
        Number of bootstrap resamples of the queries.
    confidence : float
        Coverage of the interval.
    seed : int
        Seed of the resampling.

    Returns
    -------
    (mean, lower, upper)
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    means = _resampled_means(x, n_resamples, rng)
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(np.mean(x)), float(lower), float(upper)


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for mean(a) - mean(b), paired by query.

    Returns
    -------
    (mean_difference, lower, upper)
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"paired samples must have the same shape, got {a.shape} and {b.shape}"
        )
    return bootstrap_ci(
        a - b, n_resamples=n_resamples, confidence=confidence, seed=seed,
    )


def paired_randomization_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_permutations: int = 100000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization test for a difference of means.

    Under the null hypothesis the two systems are exchangeable within
    each query, so the sign of every per-query difference is equally
    likely to be flipped.  The p-value is the fraction of sign
    assignments whose absolute mean difference is at least the observed
    one, with the observed assignment counted once (so the p-value is
    never zero).

    Returns
    -------
    float
        Two-sided p-value.
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"paired samples must have the same shape, got {a.shape} and {b.shape}"
        )
    diff = a - b
    observed = abs(float(np.mean(diff)))
    if not np.any(diff != 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    n = diff.size
    exceed = 0
    done = 0
    # Chunked to bound memory: each chunk draws (chunk, n) signs.
    chunk = max(1, min(n_permutations, 4_000_000 // max(n, 1)))
    while done < n_permutations:
        size = min(chunk, n_permutations - done)
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
        stats = np.abs(signs @ diff) / n
        # Tolerance guards against rounding in the permuted sums.
        exceed += int(np.sum(stats >= observed - 1e-15))
        done += size
    return (exceed + 1) / (n_permutations + 1)


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values for a family of comparisons.

    Controls the family-wise error rate without assuming independence.

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values.

    Returns
    -------
    list of float
        Adjusted p-values, in the input order.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def _resampled_means(
    x: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Means of ``n_resamples`` bootstrap resamples of ``x``."""
    n = x.size
    means = np.empty(n_resamples, dtype=np.float64)
    chunk = max(1, min(n_resamples, 4_000_000 // max(n, 1)))
    done = 0
    while done < n_resamples:
        size = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        means[done:done + size] = np.mean(x[idx], axis=1)
        done += size
    return means
