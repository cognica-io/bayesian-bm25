#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""IR evaluation metrics for benchmarking."""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    relevance = np.asarray(relevance, dtype=np.float64)[:k]
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Normalised DCG at rank k."""
    actual = dcg_at_k(relevance, k)
    ideal = dcg_at_k(np.sort(relevance)[::-1], k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def precision_at_k(relevance: np.ndarray, k: int) -> float:
    """Precision at rank k (binary relevance)."""
    relevance = np.asarray(relevance, dtype=np.float64)[:k]
    if len(relevance) == 0:
        return 0.0
    return float(np.sum(relevance > 0) / k)


def average_precision(relevance: np.ndarray) -> float:
    """Average Precision for a single query (binary relevance)."""
    relevance = np.asarray(relevance, dtype=np.float64)
    if np.sum(relevance) == 0:
        return 0.0
    precisions = []
    relevant_count = 0
    for i, rel in enumerate(relevance):
        if rel > 0:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return float(np.mean(precisions))


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
