#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""IR evaluation metrics for benchmarking.

Calibration metrics (ECE, Brier score, reliability diagram) are re-exported
from the main ``bayesian_bm25.metrics`` module.  IR ranking metrics (DCG,
NDCG, precision, AP) remain here as benchmark-only utilities.
"""

from __future__ import annotations

import numpy as np

# Re-export calibration metrics from the main package
from bayesian_bm25.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)

__all__ = [
    "average_precision",
    "best_f1_threshold",
    "brier_score",
    "dcg_at_k",
    "expected_calibration_error",
    "ndcg_at_k",
    "precision_at_k",
    "reliability_diagram",
    "threshold_f1",
]


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


def threshold_f1(
    values: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the decision ``value >= threshold``."""
    predicted = values >= threshold
    tp = np.sum(predicted & (labels == 1))
    fp = np.sum(predicted & (labels == 0))
    fn = np.sum(~predicted & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def best_f1_threshold(
    values: np.ndarray,
    labels: np.ndarray,
    n_missed: int = 0,
) -> tuple[float, float]:
    """Threshold that maximizes pooled F1, by exact search.

    Every distinct value is a candidate threshold, so nothing is lost to
    a grid.  A grid that is uniform in the transformed variable is
    coarse wherever the transform is flat, which makes two monotone
    transforms of the same score appear to differ when they cannot.

    Parameters
    ----------
    values : array
        Scores of the candidate documents.
    labels : array
        Binary relevance labels of the same documents.
    n_missed : int
        Number of relevant documents that are absent from ``values`` and
        can never be selected.

    Returns
    -------
    (threshold, f1)
        Documents with ``value >= threshold`` are selected.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    hits = np.cumsum(labels[order])
    selected = np.arange(1, len(values) + 1, dtype=np.float64)
    total_relevant = float(hits[-1]) + n_missed
    if total_relevant == 0.0:
        return float(sorted_values[0]), 0.0

    f1 = 2.0 * hits / (selected + total_relevant)
    # A threshold is only realizable at the last element of a run of ties.
    boundary = np.ones(len(values), dtype=bool)
    boundary[:-1] = sorted_values[:-1] > sorted_values[1:]
    f1 = np.where(boundary, f1, -1.0)
    best = int(np.argmax(f1))
    return float(sorted_values[best]), float(f1[best])
