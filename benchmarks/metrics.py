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
    "brier_score",
    "dcg_at_k",
    "expected_calibration_error",
    "ndcg_at_k",
    "precision_at_k",
    "reliability_diagram",
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
