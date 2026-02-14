#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Bayesian BM25 -- probabilistic transforms for BM25 retrieval scores."""

from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.fusion import log_odds_conjunction, prob_and, prob_or

__all__ = [
    "BayesianProbabilityTransform",
    "log_odds_conjunction",
    "prob_and",
    "prob_or",
]

# Lazy import for scorer (requires optional bm25s dependency)
def __getattr__(name: str):
    if name == "BayesianBM25Scorer":
        from bayesian_bm25.scorer import BayesianBM25Scorer
        return BayesianBM25Scorer
    raise AttributeError(f"module 'bayesian_bm25' has no attribute {name!r}")
