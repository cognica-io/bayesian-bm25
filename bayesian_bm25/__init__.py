#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Bayesian BM25 -- probabilistic transforms for BM25 retrieval scores."""

from importlib.metadata import version as _metadata_version

from bayesian_bm25.calibration_accumulator import CalibrationAccumulator
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.consistency import dilute_offset, expected_count, solve_offset
from bayesian_bm25.count_calibration import CountCalibration, calibrate_count
from bayesian_bm25.evidence_fusion import evidence_upper_bound, fuse_evidence, posterior_to_evidence
from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    LearnableLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    balanced_log_odds_fusion,
    confidence_exponent,
    cosine_to_probability,
    effective_signal_count,
    evidence_correlation,
    gaussian_evidence_weights,
    log_odds_and,
    log_odds_conjunction,
    log_odds_fusion,
    log_odds_not,
    log_odds_or,
    prob_and,
    prob_not,
    prob_or,
)
from bayesian_bm25.metrics import (
    CalibrationReport,
    brier_score,
    calibration_report,
    calibration_report_from_log_odds,
    expected_calibration_error,
    log_loss,
    reliability_diagram,
)
from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.query_calibration import (
    QueryCalibration,
    calibrate_query,
    score_dispersion,
)
from bayesian_bm25.tail_scale import tail_scale

__version__ = _metadata_version("bayesian-bm25")

__all__ = [
    "__version__",
    "AttentionLogOddsWeights",
    "BayesianProbabilityTransform",
    "CalibrationReport",
    "CalibrationContext",
    "CalibrationAccumulator",
    "CountCalibration",
    "LearnableLogOddsWeights",
    "MultiHeadAttentionLogOddsWeights",
    "QueryCalibration",
    "VectorProbabilityTransform",
    "balanced_log_odds_fusion",
    "brier_score",
    "calibrate_query",
    "calibrate_count",
    "calibration_report",
    "calibration_report_from_log_odds",
    "confidence_exponent",
    "cosine_to_probability",
    "dilute_offset",
    "effective_signal_count",
    "evidence_correlation",
    "evidence_upper_bound",
    "fuse_evidence",
    "expected_calibration_error",
    "expected_count",
    "gaussian_evidence_weights",
    "log_loss",
    "ivf_density_prior",
    "knn_density_prior",
    "log_odds_and",
    "log_odds_conjunction",
    "log_odds_fusion",
    "log_odds_not",
    "log_odds_or",
    "prob_and",
    "prob_not",
    "prob_or",
    "posterior_to_evidence",
    "reliability_diagram",
    "score_dispersion",
    "solve_offset",
    "tail_scale",
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
