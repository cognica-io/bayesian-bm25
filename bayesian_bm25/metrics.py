#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Calibration metrics for evaluating probability quality.

Provides Expected Calibration Error (ECE), Brier score, and reliability
diagram data for assessing how well predicted probabilities match actual
relevance rates.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual relevance
    rates.  Lower is better.  Perfect calibration = 0.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probabilities)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities > lo) & (probabilities <= hi)
        if lo == 0:
            mask = (probabilities >= lo) & (probabilities <= hi)
        count = np.sum(mask)
        if count == 0:
            continue
        avg_prob = np.mean(probabilities[mask])
        avg_label = np.mean(labels[mask])
        ece += (count / total) * abs(avg_prob - avg_label)

    return float(ece)


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Brier score: mean squared error between probabilities and labels.

    Decomposes into calibration + discrimination.  Lower is better.
    A constant prediction of base rate achieves the reference score.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))


def reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Compute reliability diagram data: (avg_predicted, avg_actual, count) per bin.

    Perfect calibration means avg_predicted == avg_actual for every bin.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities > lo) & (probabilities <= hi)
        if lo == 0:
            mask = (probabilities >= lo) & (probabilities <= hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_pred = float(np.mean(probabilities[mask]))
        avg_actual = float(np.mean(labels[mask]))
        bins.append((avg_pred, avg_actual, count))
    return bins
