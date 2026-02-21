#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Bayesian BM25 -- probabilistic transforms for BM25 retrieval scores."""

from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.fusion import (
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_or,
)
from bayesian_bm25.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)

__all__ = [
    "BayesianProbabilityTransform",
    "brier_score",
    "cosine_to_probability",
    "expected_calibration_error",
    "log_odds_conjunction",
    "prob_and",
    "prob_or",
    "reliability_diagram",
]

# Lazy import for scorer (requires optional bm25s dependency)
def __getattr__(name: str):
    if name == "BayesianBM25Scorer":
        from bayesian_bm25.scorer import BayesianBM25Scorer
        return BayesianBM25Scorer
    raise AttributeError(f"module 'bayesian_bm25' has no attribute {name!r}")
