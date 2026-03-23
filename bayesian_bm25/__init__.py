#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Bayesian BM25 -- probabilistic transforms for BM25 retrieval scores."""

from importlib.metadata import version as _metadata_version

from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    LearnableLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    balanced_log_odds_fusion,
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_not,
    prob_or,
)
from bayesian_bm25.metrics import (
    CalibrationReport,
    brier_score,
    calibration_report,
    expected_calibration_error,
    log_loss,
    reliability_diagram,
)
from bayesian_bm25.probability import BayesianProbabilityTransform

__version__ = _metadata_version("bayesian-bm25")

__all__ = [
    "__version__",
    "AttentionLogOddsWeights",
    "BayesianProbabilityTransform",
    "CalibrationReport",
    "LearnableLogOddsWeights",
    "MultiHeadAttentionLogOddsWeights",
    "VectorProbabilityTransform",
    "balanced_log_odds_fusion",
    "brier_score",
    "calibration_report",
    "cosine_to_probability",
    "expected_calibration_error",
    "log_loss",
    "ivf_density_prior",
    "knn_density_prior",
    "log_odds_conjunction",
    "prob_and",
    "prob_not",
    "prob_or",
    "reliability_diagram",
]

# Lazy imports for optional modules
def __getattr__(name: str):
    if name == "BayesianBM25Scorer":
        from bayesian_bm25.scorer import BayesianBM25Scorer
        return BayesianBM25Scorer
    if name == "RetrievalResult":
        from bayesian_bm25.scorer import RetrievalResult
        return RetrievalResult
    if name == "BlockMaxIndex":
        from bayesian_bm25.scorer import BlockMaxIndex
        return BlockMaxIndex
    if name == "MultiFieldScorer":
        from bayesian_bm25.multi_field import MultiFieldScorer
        return MultiFieldScorer
    if name == "FusionDebugger":
        from bayesian_bm25.debug import FusionDebugger
        return FusionDebugger
    if name == "PlattCalibrator":
        from bayesian_bm25.calibration import PlattCalibrator
        return PlattCalibrator
    if name == "IsotonicCalibrator":
        from bayesian_bm25.calibration import IsotonicCalibrator
        return IsotonicCalibrator
    if name == "TemporalBayesianTransform":
        from bayesian_bm25.probability import TemporalBayesianTransform
        return TemporalBayesianTransform
    if name == "VectorProbabilityTransform":
        from bayesian_bm25.vector_probability import VectorProbabilityTransform
        return VectorProbabilityTransform
    if name == "ivf_density_prior":
        from bayesian_bm25.vector_probability import ivf_density_prior
        return ivf_density_prior
    if name == "knn_density_prior":
        from bayesian_bm25.vector_probability import knn_density_prior
        return knn_density_prior
    raise AttributeError(f"module 'bayesian_bm25' has no attribute {name!r}")
