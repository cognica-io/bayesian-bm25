#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tables of Section 6 of "From Scores to Evidence".

Reads the result files written by the benchmark scripts and writes every
table of the paper's evaluation to its own Markdown file, formatted as
it appears in the paper (``table01.md`` ... ``table23.md``), together
with the measurements the text quotes without a table
(``supplement_*.md``).

The inputs, all in one directory, are

* ``reevaluation.json`` and ``reevaluation_mpnet.json`` from
  ``benchmarks/reevaluation.py`` (the second with
  ``--model all-mpnet-base-v2``);
* ``hybrid_beir.json``, ``hybrid_beir_ivf.json`` (query background), and
  ``hybrid_beir_ivf_residuals.json`` (the named residual ablation);
* ``protocol_diagnosis.json`` from ``benchmarks/protocol_diagnosis.py``;
* ``ceiling_slope.json`` and ``ceiling_slope_mpnet.json``.

``--legacy`` explicitly selects the older 19-table, id-identity archive.

Usage:
    python -m benchmarks.paper_tables --results benchmarks/results --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

import numpy as np

from benchmarks.reevaluation_report import (
    DATASET_LABELS,
    DATASET_ORDER,
    PerQuery,
    comparison_table,
    load_hybrid,
    load_hybrid_diagnostics,
    load_reevaluation,
    ranking_table,
)

# Adjusted p-value below which a paired difference is set in bold.
SIGNIFICANCE_LEVEL = 0.05

# A cell format: a format string, or a function of the value.
Formatter = str | Callable[[float], str]

# ---------------------------------------------------------------------------
# Row definitions: (key, cells ...) in the order of the paper
# ---------------------------------------------------------------------------

# Table 1: the configurations of the first version's Table 1.
MAIN_ROWS = [
    ("BM25", "BM25", "`BM25`", "none"),
    ("Dense", "Dense", "`Dense`", "none"),
    ("RRF", "RRF ($k = 60$)", "`RRF`", "none"),
    ("Convex", "Convex (min-max, $w = 0.5$)", "`Convex`", "none"),
    (
        "Bayesian-LogOdds-Local",
        "Log-odds, per-query standardized",
        "`Bayesian-LogOdds-Local`",
        "none",
    ),
    ("Bayesian-Balanced", "Log-odds, per-query range-normalized", "`Bayesian-Balanced`", "none"),
    ("Bayesian-LogOdds", "Log-odds, corpus-level parameters", "`Bayesian-LogOdds`", "none"),
    (
        "Bayesian-LogOdds-BR",
        "Log-odds, corpus-level parameters + base rate",
        "`Bayesian-LogOdds-BR`",
        "none",
    ),
    (
        "Bayesian-Vector-Balanced",
        "Density-ratio vector evidence",
        "`Bayesian-Vector-Balanced`",
        "none",
    ),
    (
        "Bayesian-Vector-Softplus",
        "Density-ratio vector evidence, softplus-gated",
        "`Bayesian-Vector-Softplus`",
        "none",
    ),
    ("Bayesian-Gated-ReLU", "Gated (ReLU)", "`Bayesian-Gated-ReLU`", "none"),
    ("Bayesian-Gated-Swish", "Gated (Swish)", "`Bayesian-Gated-Swish`", "none"),
    ("Bayesian-OR", "Probabilistic OR", "`Bayesian-OR`", "none"),
    ("Bayesian-Attn-Norm", "Query-adaptive weights", "`Bayesian-Attn-Norm`", "5-fold CV"),
    ("Bayesian-MultiHead-Norm", "Multi-head weights", "`Bayesian-MultiHead-Norm`", "5-fold CV"),
    (
        "Bayesian-Vector-Attn",
        "Density-ratio evidence + weights",
        "`Bayesian-Vector-Attn`",
        "5-fold CV",
    ),
]

COMPARISONS = [
    ("Bayesian-LogOdds-Local", "RRF", "Per-query standardized vs RRF"),
    ("Bayesian-LogOdds-Local", "Convex", "Per-query standardized vs Convex"),
    ("Bayesian-Balanced", "RRF", "Per-query range-normalized vs RRF"),
    ("Bayesian-Balanced", "Convex", "Per-query range-normalized vs Convex"),
    ("Bayesian-Balanced", "Bayesian-LogOdds-Local", "Per-query range-normalized vs standardized"),
    ("Bayesian-LogOdds", "RRF", "Corpus-level parameters vs RRF"),
    (
        "Bayesian-LogOdds",
        "Bayesian-LogOdds-Local",
        "Corpus-level parameters vs per-query standardized",
    ),
    ("Bayesian-Attn-Norm", "Bayesian-Balanced", "Query-adaptive weights (CV) vs uniform weights"),
    ("Bayesian-Vector-Balanced", "Dense", "Density-ratio vector evidence vs Dense"),
    (
        "Bayesian-Vector-Balanced",
        "Bayesian-Balanced",
        "Density-ratio vector evidence vs range-normalized",
    ),
    ("Convex", "RRF", "Convex vs RRF"),
]

SLOPE_ROWS = [
    ("Additive-Corpus", "Corpus-level", "Corpus-level"),
    ("Slope-LexQuery-DenseCorpus", "Per-query", "Corpus-level"),
    ("Slope-LexCorpus-DenseQuery", "Corpus-level", "Per-query"),
    ("Additive-Query", "Per-query", "Per-query"),
]

SLOPE_COMPARISONS = [
    ("Slope-LexQuery-DenseCorpus", "Additive-Corpus", "Lexical slope: per-query vs corpus-level"),
    ("Slope-LexCorpus-DenseQuery", "Additive-Corpus", "Dense slope: per-query vs corpus-level"),
    ("Additive-Query", "Additive-Corpus", "Both slopes: per-query vs corpus-level"),
]

ABLATION_ROWS = [
    ("Balanced", "Per-query range-normalized (`Bayesian-Balanced`)"),
    ("Balanced-NoPrior", "$-$ document-quality prior"),
    (
        "Balanced-NoPrior-Consistent",
        "$-$ floor for unmatched candidates (they carry the evidence of $s = 0$)",
    ),
    (
        "Balanced-NoPrior-Consistent-Cos",
        "$-$ logit transform of the similarity ($=$ convex combination)",
    ),
    ("Balanced-NoPrior-Consistent-Cos-Std", "$-$ range (dispersion over the pool instead)"),
    (
        "Additive-Query",
        "$-$ pool statistics (dispersion over the matching documents and the corpus)",
    ),
    ("Local", "Per-query standardized (`Bayesian-LogOdds-Local`)"),
    ("Local+Consistent", "$+$ unmatched candidates carry the evidence of $s = 0$"),
    ("Additive-Query-MAD", "Standardized over the population, robust dispersion (MAD)"),
    ("Additive-Query+Prior", "Standardized over the population, with the document-quality prior"),
]

PROTOCOL_CELLS = [
    "kept/probability",
    "kept/log-odds",
    "excluded/probability",
    "excluded/log-odds",
]

PROTOCOL_ROWS = [
    ("BM25", "BM25"),
    ("Dense", "Dense"),
    ("RRF", "RRF"),
    ("Convex", "Convex"),
    ("Standardized (Local)", "Log-odds, per-query standardized"),
    ("Range-normalized (Balanced)", "Log-odds, per-query range-normalized"),
    ("Corpus-level conjunction", "Log-odds, corpus-level parameters"),
    ("Probabilistic OR", "Probabilistic OR"),
]

# Rows of PROTOCOL_ROWS that form a probability and can therefore saturate.
PROTOCOL_PROBABILISTIC = PROTOCOL_ROWS[4:]

SATURATION_ROWS = [
    ("query_tokens_mean", "Query length in tokens (mean)", "{:.1f}"),
    ("sigma_scale_ratio", "$\\sigma_q$ over the corpus-level scale (mean)", "{:.1f}"),
    ("max_evidence_corpus_mean", "Largest corpus-level evidence (mean)", "{:.1f}"),
    ("max_evidence_corpus_p95", "Largest corpus-level evidence (95th percentile)", "{:.1f}"),
    ("queries_with_likelihood_one", "Queries with a likelihood that rounds to 1", "{:.1%}"),
    ("tied_top100_mean", "Tied among the 100 highest probabilities (mean)", "{:.1f}"),
    ("tied_top100_log_odds_mean", "Tied among the 100 highest log-odds (mean)", "{:.1f}"),
    (
        "queries_with_rounding_ties_top10",
        "Queries with a tie created by rounding in the top 10",
        "{:.1%}",
    ),
]

# Rows of the first version's lexical calibration table, by benchmark name.
BASE_RATE_ROWS = [
    ("Bayesian (no base rate)", "No base rate (prior $\\tfrac12$)", "none"),
    ("Bayesian (base_rate=auto)", "Base rate from percentile rule", "none"),
    ("Bayesian (base_rate=0.001)", "Base rate fixed at $0.001$", "none"),
    ("Batch fit (no base rate)", "Supervised fit (C1), no base rate", "train half"),
    ("Batch fit (base_rate=auto)", "Supervised fit (C1) + percentile base rate", "train half"),
    ("Batch fit (prior_aware)", "Supervised fit, prior-aware (C2)", "train half"),
    ("Batch fit (prior_free)", "Supervised fit, prior-free (C3)", "train half"),
    ("Min-max normalization", "Global min-max rescaling", "train half (range)"),
    ("Platt scaling", "Platt scaling", "train half"),
]

LEXICAL_ROWS = [
    ("Lex-Corpus-NoBaseRate", "Corpus-level, no base rate", "none"),
    ("Lex-Corpus-Percentile", "Corpus-level, percentile base rate", "none"),
    ("Lex-Corpus-0.001", "Corpus-level, base rate $0.001$", "none"),
    ("Lex-Query-R0.5", "Per-query, $R_q = 0.5$", "none"),
    ("Lex-Query-R1", "Per-query, $R_q = 1$", "none"),
    ("Lex-Query-R2", "Per-query, $R_q = 2$", "none"),
    ("Lex-Query-R5", "Per-query, $R_q = 5$", "none"),
    ("Lex-Query-R10", "Per-query, $R_q = 10$", "none"),
    ("Lex-Query-R20", "Per-query, $R_q = 20$", "none"),
    ("Lex-Query-R50", "Per-query, $R_q = 50$", "none"),
    ("Lex-Query-Rmean", "Per-query, $R_q =$ mean count of the training queries", "count (CV)"),
    ("Lex-Query-Roracle", "Per-query, $R_q =$ true count of the query", "oracle count"),
    ("Lex-Query-Fit", "Per-query, $\\kappa$ fitted, $R_q =$ training mean", "5-fold CV"),
    ("Lex-Platt-Raw", "Platt scaling of the raw score", "5-fold CV"),
    ("Lex-Platt-Standardized", "Platt scaling after per-query standardization", "5-fold CV"),
    ("Lex-MinMax", "Global min-max rescaling", "range (CV)"),
]

# The fused output of Theorem 5.13 on the candidate pool: the sequence of
# Section 6.8, then the corrections of Section 5.3 applied to its second
# step, with the background of the density ratio as a Gaussian and as a
# kernel estimate of the marginal.
UNIFIED_ROWS = [
    ("Unified-Corpus", "(5.2), corpus-level parameters", "none"),
    ("Unified-Query", "(5.2), per-query parameters ($\\kappa = 1$, $R_q = 1$)", "none"),
    ("Unified-Fit", "(5.2), per-query parameters fitted to labels", "5-fold CV"),
    ("Unified-Query-Consistent", "Per-query, $+$ post-fusion consistency", "none"),
    ("Unified-Query-Pooled", "Per-query, pooled with $\\rho = \\tfrac12$", "none"),
    ("Unified-Query-Pooled-Consistent", "Per-query, pooled, $+$ post-fusion consistency", "none"),
    ("Unified-Query-Marginal", "Per-query, background estimated as the marginal", "none"),
    (
        "Unified-Query-Marginal-Consistent",
        "Per-query, marginal, $+$ post-fusion consistency",
        "none",
    ),
    (
        "Unified-Query-Marginal-Pooled-Consistent",
        "Per-query, marginal, pooled, $+$ post-fusion consistency",
        "none",
    ),
    ("Unified-Fit-Pooled-Consistent", "Fitted, pooled, $+$ post-fusion consistency", "5-fold CV"),
]

# The exact form of Remark 4.5 inside (5.2), with the same corrections.
UNIFIED_EXACT_ROWS = [
    ("Unified-Query-Exact", "Per-query, exact form of Remark 4.5", "none"),
    ("Unified-Query-Exact-Consistent", "$+$ post-fusion consistency", "none"),
    ("Unified-Query-Exact-Pooled", "Pooled with $\\rho = \\tfrac12$", "none"),
    ("Unified-Query-Exact-Pooled-Consistent", "Pooled, $+$ post-fusion consistency", "none"),
    ("Unified-Query-Exact-Gaussian", "Exact form, Gaussian background", "none"),
    ("Unified-Fit-Exact", "Fitted, exact form", "5-fold CV"),
    ("Unified-Fit-Exact-Consistent", "$+$ post-fusion consistency", "5-fold CV"),
    ("Unified-Fit-Exact-Pooled-Consistent", "Pooled, $+$ post-fusion consistency", "5-fold CV"),
]

# The forms of the density-ratio vector evidence inside (5.2), per-query
# lexical parameters.
UNIFIED_VARIANT_ROWS = [
    ("Unified-Query", "Density ratio (4.1), Gaussian background", "none"),
    ("Unified-Query-Monotone", "$+$ monotone projection", "none"),
    ("Unified-Query-Marginal", "Density ratio (4.1), background estimated as the marginal", "none"),
    (
        "Unified-Query-Exact",
        "Exact form of Remark 4.5, background estimated as the marginal",
        "none",
    ),
    ("Unified-Query-Exact-Monotone", "$+$ monotone projection", "none"),
    ("Unified-Query-Exact-Gaussian", "Exact form, Gaussian background", "none"),
    ("Unified-Query-Uniform", "Density ratio (4.1), unweighted estimate", "none"),
    ("Unified-Query-Bandwidth2", "Density ratio (4.1), bandwidth doubled", "none"),
    ("Unified-Query-Sharpened", "Density ratio (4.1), sharpened weights", "none"),
]

# The pooled fusion of Section 5.3 with the log-linear vector evidence of
# Section 4.8.
POOLED_ROWS = [
    ("Fused-Corpus", "Corpus-level parameters, evidence summed", "none"),
    ("Fused-Query-Additive-rho1", "Per-query, evidence summed ($\\rho = 1$)", "none"),
    ("Fused-Query-Additive-rho0.5", "Per-query, pooled with $\\rho = \\tfrac12$", "none"),
    (
        "Fused-Query-Additive-rhoMeasured",
        "Per-query, pooled with $\\rho$ from the measured correlation",
        "none",
    ),
    ("Fused-Query-Consistent-rho0.5", "$\\rho = \\tfrac12$ + post-fusion consistency", "none"),
    ("Fused-Query-Consistent-rhoMeasured", "Measured $\\rho$ + post-fusion consistency", "none"),
    ("Fused-Fit-Rmean", "Per-query, slopes and count fitted to labels", "5-fold CV"),
    ("Fused-Query-Consistent-Rmean", "Pooled + consistency, training-mean count", "count (CV)"),
    ("Fused-Tail-Consistent-R1", "Dense tail units, pooled + consistency", "none"),
    ("Fused-Tail-Consistent-Rmean", "The same, training-mean count", "count (CV)"),
]

CORRELATION_ROWS = [
    ("corr_all", "Correlation of the two signals over the corpus", "{:.2f}"),
    ("corr_pool", "Correlation over the candidate pool", "{:.2f}"),
    ("rho_from_corr_all", "Confidence exponent $\\rho$ implied (Theorem 5.8)", "{:.2f}"),
    (
        "additive_count_ratio_median",
        "Sum of the posteriors when the evidence is summed, over $R_q$ (median)",
        "{:.1f}",
    ),
]

WEIGHT_ROWS = [
    ("VPT-DensityPrior", "No second modality (Section 4.5)"),
    ("VPT-BM25Weights", "Cross-modal BM25 weights"),
]

BANDWIDTH_ROWS = [
    ("VPT-BW-0.2", "$c = 0.2$"),
    ("VPT-BW-0.5", "$c = 0.5$"),
    ("VPT-BW-1.0", "$c = 1.0$"),
    ("VPT-BW-2.0", "$c = 2.0$"),
]

VECTOR_ROWS = [
    ("Vec-GlobalSigmoid", "Global sigmoid on distance", "none"),
    ("Vec-Arctan", "Arctangent normalization", "none"),
    ("Vec-Platt-Raw", "Platt scaling", "5-fold CV"),
    ("Vec-DensityRatio", "Density ratio (4.1), Gaussian background", "none"),
    (
        "Vec-DensityRatio-Marginal",
        "Density ratio (4.1), background estimated as the marginal",
        "none",
    ),
    ("Vec-DensityRatio-Exact-Gaussian", "Exact form of Remark 4.5, Gaussian background", "none"),
    ("Vec-DensityRatio-Exact", "Exact form, background estimated as the marginal", "none"),
    ("Vec-Query-R1", "Log-linear model of Section 4.8, $R_q = 1$", "none"),
    ("Vec-Tail-R1", "Log-linear model, tail units, $R_q = 1$", "none"),
]

VECTOR_RANK_ROWS = [
    ("Dense", "Dense alone"),
    ("Additive-Query-DensityRatio", "Density ratio (4.1), Gaussian background"),
    ("Additive-Query-DensityRatio-Monotone", "$+$ monotone projection"),
    (
        "Additive-Query-DensityRatio-Marginal",
        "Density ratio (4.1), background estimated as the marginal",
    ),
    (
        "Additive-Query-DensityRatio-Exact",
        "Exact form of Remark 4.5, background estimated as the marginal",
    ),
    ("Additive-Query-DensityRatio-Exact-Monotone", "$+$ monotone projection"),
    ("Additive-Query-DensityRatio-Exact-Gaussian", "Exact form, Gaussian background"),
    ("Additive-Query-DensityRatio-Uniform", "Density ratio (4.1), unweighted estimate"),
    ("Additive-Query-DensityRatio-Bandwidth2", "Density ratio (4.1), bandwidth doubled"),
    ("Additive-Query-DensityRatio-Sharpened", "Density ratio (4.1), sharpened weights"),
    ("Additive-Query", "Log-linear model of Section 4.8"),
    ("Additive-Query-Dense2", "Log-linear model, slope $2/\\sigma_q$"),
    ("Additive-Query-Dense4", "Log-linear model, slope $4/\\sigma_q$"),
    ("Additive-Query-Tail", "Log-linear model, slope $1/m_q$ (tail units)"),
    ("Additive-Query-GaussianQuadratic", "$\\tfrac12 z^2$ alone"),
]

VECTOR_FORM_COMPARISONS = [
    ("Additive-Query-DensityRatio", "Dense", "(5.2) with the density ratio vs Dense"),
    ("Additive-Query-DensityRatio", "Additive-Query", "Density ratio (4.1) vs log-linear model"),
    (
        "Additive-Query-DensityRatio-Monotone",
        "Additive-Query-DensityRatio",
        "Monotone projection vs none",
    ),
    (
        "Additive-Query-DensityRatio-Marginal",
        "Additive-Query-DensityRatio",
        "(4.1): marginal vs Gaussian background",
    ),
    (
        "Additive-Query-DensityRatio-Exact-Gaussian",
        "Additive-Query-DensityRatio",
        "Gaussian background: exact form vs (4.1)",
    ),
    (
        "Additive-Query-DensityRatio-Exact",
        "Additive-Query-DensityRatio-Marginal",
        "Marginal background: exact form vs (4.1)",
    ),
    (
        "Additive-Query-DensityRatio-Exact",
        "Additive-Query-DensityRatio",
        "Exact form with the marginal vs (4.1) with the Gaussian",
    ),
    (
        "Additive-Query-DensityRatio-Uniform",
        "Additive-Query-DensityRatio",
        "Unweighted estimate vs cross-modal weights",
    ),
    (
        "Additive-Query-DensityRatio-Bandwidth2",
        "Additive-Query-DensityRatio",
        "Bandwidth doubled vs Definition 4.8",
    ),
    (
        "Additive-Query-DensityRatio-Sharpened",
        "Additive-Query-DensityRatio",
        "Sharpened vs unsharpened weights",
    ),
]

# Comparisons among the vector-calibration ablations of the hybrid benchmark.
WEIGHT_COMPARISONS = [
    ("VPT-BM25Weights", "VPT-DensityPrior", "Cross-modal vs weights without a second modality"),
    ("VPT-BM25Weights", "Dense", "Cross-modal weights vs Dense"),
    ("VPT-BW-2.0", "VPT-BW-0.2", "Bandwidth factor 2.0 vs 0.2"),
]

THRESHOLD_ROWS = [
    ("Raw BM25", "Raw BM25 score"),
    ("Corpus-level log-odds", "Corpus-level log-odds"),
    ("Corpus-level log-odds + prior", "Corpus-level log-odds + document-quality prior"),
    ("Per-query z-score", "Per-query standardized score"),
    ("Per-query min-max", "Score over the best score of the query"),
    ("Per-query log-odds (R=1)", "Per-query posterior, $R_q = 1$"),
    ("Per-query log-odds (R oracle)", "Per-query posterior, true count"),
]

# The experiment of Section 6.8: the fusion rule fixed at (5.2).
SEQUENCE_ROWS = [
    ("RRF", "RRF ($k = 60$)", "none"),
    ("Convex", "Convex (min-max, $w = 0.5$)", "none"),
    ("Additive-Corpus-DensityRatio", "(5.2), corpus-level parameters", "none"),
    (
        "Additive-Query-DensityRatio",
        "(5.2), per-query parameters ($\\kappa = 1$, $R_q = 1$)",
        "none",
    ),
    ("Additive-QueryFit-DensityRatio", "(5.2), per-query parameters fitted to labels", "5-fold CV"),
    (
        "Additive-Query-DensityRatio-Marginal",
        "(5.2), per-query parameters, background estimated as the marginal",
        "none",
    ),
    ("Additive-Query", "(5.2), per-query parameters, log-linear vector evidence", "none"),
    (
        "Additive-Query-Tail",
        "(5.2), per-query parameters, log-linear vector evidence in tail units",
        "none",
    ),
    (
        "Additive-Query-Fit",
        "(5.2), log-linear vector evidence, both slopes fitted to labels",
        "5-fold CV",
    ),
]

SEQUENCE_COMPARISONS = [
    (
        "Additive-Query-DensityRatio",
        "Additive-Corpus-DensityRatio",
        "(5.2): per-query vs corpus-level parameters",
    ),
    (
        "Additive-QueryFit-DensityRatio",
        "Additive-Query-DensityRatio",
        "(5.2): fitted vs conventions",
    ),
    ("Additive-Query-DensityRatio", "RRF", "(5.2), per-query vs RRF"),
    ("Additive-Query-DensityRatio", "Convex", "(5.2), per-query vs Convex"),
    ("Additive-Query-DensityRatio", "Bayesian-Balanced", "(5.2), per-query vs range-normalized"),
    (
        "Additive-Query-DensityRatio",
        "Additive-Query",
        "(5.2): density ratio vs log-linear vector evidence",
    ),
    (
        "Additive-Query-DensityRatio-Marginal",
        "Additive-Query-DensityRatio",
        "(5.2): marginal vs Gaussian background",
    ),
    (
        "Additive-Query-DensityRatio-Marginal",
        "Additive-Query",
        "(5.2): density ratio with the marginal vs log-linear",
    ),
    ("Additive-Query-DensityRatio-Marginal", "RRF", "(5.2), marginal background vs RRF"),
    ("Additive-Query-DensityRatio-Marginal", "Convex", "(5.2), marginal background vs Convex"),
    ("Additive-Query", "RRF", "(5.2), log-linear vs RRF"),
    ("Additive-Query", "Convex", "(5.2), log-linear vs Convex"),
    ("Additive-Query-Fit", "Additive-Query", "(5.2), log-linear: fitted slopes vs conventions"),
    (
        "Additive-Query-Fit",
        "Additive-Query-DensityRatio",
        "(5.2): log-linear with fitted slopes vs density ratio without labels",
    ),
]

SECOND_ENCODER_ROWS = [
    ("BM25", "BM25", "none"),
    ("Dense", "Dense", "none"),
    ("RRF", "RRF ($k = 60$)", "none"),
    ("Convex", "Convex (min-max, $w = 0.5$)", "none"),
    ("Local", "Log-odds, per-query standardized", "none"),
    ("Balanced", "Log-odds, per-query range-normalized", "none"),
    ("Additive-Corpus-DensityRatio", "(5.2), corpus-level parameters", "none"),
    (
        "Additive-Query-DensityRatio",
        "(5.2), per-query parameters ($\\kappa = 1$, $R_q = 1$)",
        "none",
    ),
    ("Additive-QueryFit-DensityRatio", "(5.2), per-query parameters fitted to labels", "5-fold CV"),
    (
        "Additive-Query-DensityRatio-Marginal",
        "(5.2), per-query parameters, background estimated as the marginal",
        "none",
    ),
    ("Additive-Query", "(5.2), per-query parameters, log-linear vector evidence", "none"),
    (
        "Additive-Query-Tail",
        "(5.2), per-query parameters, log-linear vector evidence in tail units",
        "none",
    ),
    (
        "Additive-Query-Fit",
        "(5.2), log-linear vector evidence, both slopes fitted to labels",
        "5-fold CV",
    ),
]

SECOND_ENCODER_COMPARISONS = [
    (
        "Additive-Query-DensityRatio",
        "Additive-Corpus-DensityRatio",
        "(5.2): per-query vs corpus-level parameters",
    ),
    ("Additive-Query-DensityRatio", "RRF", "(5.2), per-query vs RRF"),
    ("Additive-Query-DensityRatio", "Convex", "(5.2), per-query vs Convex"),
    ("Additive-Query-DensityRatio", "Balanced", "(5.2), per-query vs range-normalized"),
    (
        "Additive-Query-DensityRatio-Marginal",
        "Additive-Query-DensityRatio",
        "(5.2): marginal vs Gaussian background",
    ),
    (
        "Additive-Query-DensityRatio",
        "Additive-Query",
        "(5.2): density ratio vs log-linear vector evidence",
    ),
    ("Additive-Query-Fit", "Additive-Query", "(5.2), log-linear: fitted slopes vs conventions"),
    ("Balanced", "Convex", "Per-query range-normalized vs Convex"),
    ("Local", "Convex", "Per-query standardized vs Convex"),
]

# Background statistics of the IVF index, cosine distance.
IVF_BACKGROUND_ROWS = [
    ("residual_mean", "Intra-cell residuals: mean", "{:.3f}"),
    ("residual_std", "Intra-cell residuals: standard deviation", "{:.3f}"),
    ("query_distance_mean", "Query to corpus, from the centroids: mean", "{:.3f}"),
    ("query_distance_std", "Query to corpus, from the centroids: standard deviation", "{:.3f}"),
]

INTERCEPT_ROWS = [
    (
        "offset_dilute_error_mean",
        "Dilute form (Corollary 3.8): offset minus exact, mean",
        "{:+.2f}",
    ),
    ("offset_dilute_abs_error_p95", "Dilute form: 95th percentile of the error", "{:.2f}"),
    ("offset_topk_abs_error_mean", "Retrieved list only (top 1000): mean absolute error", "{:.3f}"),
    (
        "offset_topk_bound_abs_error_mean",
        "Retrieved list + tail bounded by the $k$-th score: mean absolute error",
        "{:.3f}",
    ),
    (
        "offset_sampled_abs_error_mean",
        "Retrieved list + 1000 sampled documents: mean absolute error",
        "{:.4f}",
    ),
    (
        "offset_factorized_error_mean",
        "Per-term statistics (independence): offset minus exact, mean",
        "{:+.2f}",
    ),
    (
        "psi_gap_negative_fraction",
        "Queries for which the factorized $\\psi_q$ is a lower bound",
        "{:.1%}",
    ),
    ("sigma_factorized_ratio_mean", "Factorized $\\sigma_q$ over the exact one (mean)", "{:.2f}"),
]


def format_count_ratio(value: float) -> str:
    """Three significant digits, without an exponent for large ratios."""
    if value >= 999.5:
        return f"{value:.0f}"
    return f"{value:.3g}"


# Metric tuples printed in one cell, separated by " / ".
SKILL: list[tuple[str, Formatter]] = [("log_loss_skill", "{:+.2f}")]
RELIABILITY: list[tuple[str, Formatter]] = [
    ("ece", "{:.4f}"),
    ("expected_over_observed", format_count_ratio),
    ("calibration_slope", "{:.2f}"),
]
SKILL_COUNT_SLOPE: list[tuple[str, Formatter]] = [
    ("log_loss_skill", "{:+.2f}"),
    ("expected_over_observed", format_count_ratio),
    ("calibration_slope", "{:.2f}"),
]
ECE_SKILL_COUNT_SLOPE: list[tuple[str, Formatter]] = [
    ("ece", "{:.3f}"),
    ("log_loss_skill", "{:+.2f}"),
    ("expected_over_observed", format_count_ratio),
    ("calibration_slope", "{:.2f}"),
]


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _datasets(results: dict) -> list[str]:
    return [d for d in DATASET_ORDER if d in results]


def _dataset_header(first: list[str], datasets: list[str]) -> list[str]:
    header = first + [DATASET_LABELS[d] for d in datasets]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]


def _format_value(value: float | None, fmt: Formatter) -> str:
    if value is None:
        return "--"
    if value == float("-inf"):
        return "$-\\infty$"
    if value == float("inf"):
        return "$\\infty$"
    if value != value:
        return "n/a"
    return fmt(value) if callable(fmt) else fmt.format(value)


def with_derived_diagnostics(results: dict) -> dict:
    """Add the diagnostics that are ratios of stored ones.

    ``sigma_scale_ratio`` is the mean dispersion of a query's matching
    scores over the corpus-level scale ``1 / alpha``.
    """
    for result in results.values():
        diagnostics = result["diagnostics"]
        diagnostics["sigma_scale_ratio"] = (
            diagnostics["sigma_lex_mean"] / diagnostics["corpus_scale"]
        )
    return results


def diagnostics_table(results: dict, rows: list[tuple[str, str, str]]) -> str:
    """One row per diagnostic quantity, one column per dataset."""
    datasets = _datasets(results)
    lines = _dataset_header(["Quantity"], datasets)
    for key, label, fmt in rows:
        cells = [_format_value(results[d]["diagnostics"].get(key), fmt) for d in datasets]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def ivf_background_table(ivf_diagnostics: dict[str, dict]) -> str:
    """The two candidate backgrounds of an IVF index, one column per dataset.

    ``ivf_diagnostics`` is the ``diagnostics`` entry of a hybrid benchmark
    run with ``--dense-backend ivf``.
    """
    datasets = [d for d in DATASET_ORDER if d in ivf_diagnostics]
    lines = _dataset_header(["Quantity"], datasets)
    for key, label, fmt in IVF_BACKGROUND_ROWS:
        cells = [_format_value(ivf_diagnostics[d]["ivf_background"][key], fmt) for d in datasets]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def calibration_table(
    results: dict,
    group: str,
    rows: list[tuple[str, str, str]],
    pool: str,
    metrics: list[tuple[str, Formatter]],
) -> str:
    """Calibration metrics of ``rows`` over ``pool``, one cell per dataset.

    ``rows`` lists ``(key, label, labels used)``; the metrics of a cell
    are joined by `` / `` in the order of ``metrics``.
    """
    datasets = _datasets(results)
    lines = _dataset_header(["Method", "Labels"], datasets)
    for key, label, labels in rows:
        cells = []
        for dataset in datasets:
            summary = results[dataset]["calibration"][group].get(key, {}).get(pool)
            if summary is None:
                cells.append("--")
            else:
                cells.append(" / ".join(_format_value(summary.get(m), fmt) for m, fmt in metrics))
        lines.append("| " + " | ".join([label, labels] + cells) + " |")
    return "\n".join(lines)


def protocol_table(protocol: dict, dataset: str = "arguana") -> str:
    """NDCG@10 under the four combinations of the two first-version defects."""
    cells_of = protocol[dataset]
    if "text/log-odds" in cells_of:
        lines = [
            "| Configuration | Id identity, probability | Id identity, log-odds "
            "| Text identity, probability | Text identity, log-odds "
            "| Tied in top 100 (text, probability) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        cells_to_read = ["id/probability", "id/log-odds", "text/probability", "text/log-odds"]
        tied_cell = "text/probability"
    else:
        return legacy_protocol_table(protocol, dataset)
    for key, label in PROTOCOL_ROWS:
        cells = [f"{100 * cells_of[cell]['ndcg'][key]:.2f}" for cell in cells_to_read]
        tied = cells_of[tied_cell]["tied_top100"].get(key)
        cells.append("--" if tied is None else f"{tied:.1f}")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def legacy_protocol_table(protocol: dict, dataset: str = "arguana") -> str:
    """The kept/excluded comparison used by the older manuscript."""
    lines = [
        "| Configuration | Ids kept, probability (first version) | Ids kept, log-odds "
        "| Ids excluded, probability | Ids excluded, log-odds (corrected) "
        "| Tied in top 100 (probability) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    cells_of = protocol[dataset]
    for key, label in PROTOCOL_ROWS:
        cells = [f"{100 * cells_of[cell]['ndcg'][key]:.2f}" for cell in PROTOCOL_CELLS]
        tied = cells_of["excluded/probability"]["tied_top100"].get(key)
        cells.append("--" if tied is None else f"{tied:.1f}")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def probability_cost_table(protocol: dict) -> str:
    """Change in NDCG@10 from ranking probabilities instead of log-odds."""
    datasets = _datasets(protocol)
    lines = _dataset_header(["Configuration"], datasets)
    for key, label in PROTOCOL_PROBABILISTIC:
        cells = []
        for dataset in datasets:
            prefix = "text" if "text/log-odds" in protocol[dataset] else "excluded"
            log_odds = protocol[dataset][f"{prefix}/log-odds"]["ndcg"][key]
            probability = protocol[dataset][f"{prefix}/probability"]["ndcg"][key]
            cells.append(f"{100 * (probability - log_odds):+.2f}")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def base_rate_table(base_rate: dict) -> str:
    """ECE and Brier score of the corpus-level lexical posterior (two datasets)."""
    results = base_rate["results"]
    lines = [
        "| Method | Labels | NFCorpus ECE | NFCorpus Brier | SciFact ECE | SciFact Brier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, label, labels in BASE_RATE_ROWS:
        cells = []
        for dataset in ("nfcorpus", "scifact"):
            entry = results[dataset][key]
            cells += [f"{entry['ECE']:.4f}", f"{entry['Brier']:.4f}"]
        lines.append("| " + " | ".join([label, labels] + cells) + " |")
    return "\n".join(lines)


def backend_table(
    rows: list[tuple[str, str]],
    backends: list[tuple[str, PerQuery]],
    first_column: str,
) -> str:
    """Rows of the hybrid benchmark under several dense backends."""
    datasets = [d for d in DATASET_ORDER if d in backends[0][1]]
    header = [first_column, "Search"] + [DATASET_LABELS[d] for d in datasets] + ["Average"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key, label in rows:
        for backend, per_query in backends:
            table = ranking_table(
                per_query,
                [(key, label, backend)],
                "NDCG@10",
                bold_among=[],
                leading=(first_column, "Search"),
            )
            lines.append(table.splitlines()[-1])
    return "\n".join(lines)


def threshold_table(results: dict) -> str:
    """Held-out F1 of one threshold: micro-average / macro-average."""
    datasets = _datasets(results)
    lines = _dataset_header(["Thresholded quantity"], datasets)
    for key, label in THRESHOLD_ROWS:
        cells = []
        for dataset in datasets:
            entry = results[dataset]["threshold"][key]
            cells.append(f"{entry['test_micro_f1']:.3f} / {entry['test_macro_f1']:.3f}")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def admitted_table(results: dict) -> str:
    """Documents a held-out query is given, and queries that are given none."""
    datasets = _datasets(results)
    lines = _dataset_header(["Thresholded quantity"], datasets)
    for key, label in THRESHOLD_ROWS:
        cells = []
        for dataset in datasets:
            entry = results[dataset]["threshold"][key]
            cells.append(f"{entry['admitted_mean']:.1f} ({entry['admitted_zero_fraction']:.0%})")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def fits_table(results: dict) -> str:
    """Slopes fitted to labels, averaged over the folds, against Theorem 5.8."""
    datasets = _datasets(results)
    lines = _dataset_header(["Quantity"], datasets)
    rows: list[tuple[str, Callable[[dict], float]]] = [
        ("Lexical slope $\\kappa_{\\text{lex}}$, fitted alone", lambda f: f["kappa_lex"]),
        ("Dense slope $\\kappa_{\\text{vec}}$, fitted alone", lambda f: f["kappa_vec"]),
        ("Lexical slope, fitted jointly", lambda f: f["kappas_fused"][0]),
        ("Dense slope, fitted jointly", lambda f: f["kappas_fused"][1]),
        ("Lexical slope, predicted by Theorem 5.8 (ii)", lambda f: f["kappas_fused_predicted"][0]),
        ("Dense slope, predicted by Theorem 5.8 (ii)", lambda f: f["kappas_fused_predicted"][1]),
        ("Mean number of relevant documents per query", lambda f: f["relevant_mean"]),
    ]
    for label, getter in rows:
        cells = [
            f"{float(np.mean([getter(fit) for fit in results[d]['fits']])):.2f}" for d in datasets
        ]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def tail_fits_table(results: dict) -> str:
    """The paper's solo slope fits in bulk and tail units."""
    datasets = _datasets(results)
    lines = _dataset_header(["Quantity"], datasets)
    lines = [line + (" Average |" if i == 0 else " --- |") for i, line in enumerate(lines)]
    for label, key in (
        ("Lexical slope in bulk units", "kappa_lex"),
        ("Lexical slope in tail units", "theta_lex"),
        ("Dense slope in bulk units", "kappa_vec"),
        ("Dense slope in tail units", "theta_vec"),
    ):
        values = [float(np.mean([fit[key] for fit in results[d]["fits"]])) for d in datasets]
        cells = [f"{value:.2f}" for value in values] + [f"{np.mean(values):.2f}"]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_tables(
    results: dict,
    per_query: PerQuery,
    hybrid: PerQuery,
    ivf: PerQuery,
    ivf_query: PerQuery,
    ivf_diagnostics: dict[str, dict],
    protocol: dict,
    base_rate: dict,
    second_results: dict,
    second_per_query: PerQuery,
    *,
    n_resamples: int = 10000,
    n_permutations: int = 100000,
) -> dict[str, str]:
    """Every table of the paper, keyed by file name without extension."""
    results = with_derived_diagnostics(results)
    datasets = _datasets(results)
    merged: PerQuery = {d: {**hybrid[d], **per_query[d]} for d in datasets}
    label_free = [row[0] for row in MAIN_ROWS if row[3] == "none"]
    resampling = dict(n_resamples=n_resamples, n_permutations=n_permutations)

    tables: dict[str, str] = {}

    # Section 6.2: ranking quality.
    tables["table01"] = ranking_table(
        merged,
        MAIN_ROWS,
        "NDCG@10",
        bold_among=label_free,
        leading=("Configuration", "Benchmark id", "Labels"),
    )
    tables["table02"] = comparison_table(
        merged,
        COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["table03"] = ranking_table(
        merged,
        SLOPE_ROWS,
        "NDCG@10",
        leading=("Lexical slope", "Dense slope"),
    )
    tables["table04"] = ranking_table(merged, ABLATION_ROWS, "NDCG@10", bold_among=[])

    # Section 6.3: the first version's ArguAna results.
    tables["table05"] = protocol_table(protocol)
    tables["table06"] = diagnostics_table(results, SATURATION_ROWS)
    tables["table07"] = probability_cost_table(protocol)

    # Section 6.4: probability calibration.
    if base_rate is not None:
        tables["table08"] = base_rate_table(base_rate)
    tables["table09"] = calibration_table(results, "lexical", LEXICAL_ROWS, "all", SKILL)
    tables["table10"] = calibration_table(
        results,
        "lexical",
        LEXICAL_ROWS,
        "all",
        RELIABILITY,
    )
    tables["table11"] = calibration_table(
        results,
        "lexical",
        LEXICAL_ROWS,
        "top10",
        SKILL_COUNT_SLOPE,
    )
    tables["table12"] = calibration_table(
        results,
        "fused",
        UNIFIED_ROWS,
        "pool",
        SKILL_COUNT_SLOPE,
    )
    tables["table13"] = calibration_table(
        results,
        "fused",
        POOLED_ROWS,
        "pool",
        SKILL_COUNT_SLOPE,
    )

    # Section 6.5: vector calibration ablations.
    backends = [("exact", hybrid), ("IVF", ivf)]
    tables["table14"] = backend_table(WEIGHT_ROWS, backends, "Weight source")
    tables["table15"] = calibration_table(
        results,
        "vector",
        VECTOR_ROWS,
        "retrieved",
        ECE_SKILL_COUNT_SLOPE,
    )
    tables["table16"] = ranking_table(
        merged,
        VECTOR_RANK_ROWS,
        "NDCG@10",
        leading=("Vector evidence in (5.2)",),
    )

    # Sections 6.6 and 6.8.
    tables["table17"] = threshold_table(results)
    sequence_label_free = [row[0] for row in SEQUENCE_ROWS if row[2] == "none"]
    tables["table18"] = ranking_table(
        merged,
        SEQUENCE_ROWS,
        "NDCG@10",
        bold_among=sequence_label_free,
        leading=("Configuration", "Labels"),
    )
    second_label_free = [row[0] for row in SECOND_ENCODER_ROWS if row[2] == "none"]
    tables["table19"] = ranking_table(
        second_per_query,
        SECOND_ENCODER_ROWS,
        "NDCG@10",
        bold_among=second_label_free,
        leading=("Configuration", "Labels"),
    )

    # Measurements quoted in the text without a table of their own.
    tables["supplement_map"] = ranking_table(
        merged,
        MAIN_ROWS,
        "MAP@10",
        bold_among=label_free,
        leading=("Configuration", "Benchmark id", "Labels"),
    )
    tables["supplement_recall"] = ranking_table(
        merged,
        MAIN_ROWS,
        "Recall@10",
        bold_among=label_free,
        leading=("Configuration", "Benchmark id", "Labels"),
    )
    tables["supplement_balanced_variants"] = ranking_table(
        merged,
        [
            ("Bayesian-Balanced", "`Bayesian-Balanced`"),
            ("Bayesian-Balanced-Mix", "`Bayesian-Balanced-Mix`"),
            ("Bayesian-Balanced-Elbow", "`Bayesian-Balanced-Elbow`"),
        ],
        "NDCG@10",
        bold_among=[],
    )
    tables["supplement_slope_comparisons"] = comparison_table(
        merged,
        SLOPE_COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["supplement_vector_form_comparisons"] = comparison_table(
        merged,
        VECTOR_FORM_COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["supplement_weight_comparisons_exact"] = comparison_table(
        hybrid,
        WEIGHT_COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["supplement_weight_comparisons_ivf"] = comparison_table(
        ivf,
        WEIGHT_COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["supplement_sequence_comparisons"] = comparison_table(
        merged,
        SEQUENCE_COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["supplement_unified_top10"] = calibration_table(
        results,
        "fused",
        UNIFIED_ROWS,
        "top10",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_unified_exact_pool"] = calibration_table(
        results,
        "fused",
        UNIFIED_EXACT_ROWS,
        "pool",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_unified_dense_list"] = calibration_table(
        results,
        "fused",
        UNIFIED_ROWS,
        "dense_list",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_unified_variants_pool"] = calibration_table(
        results,
        "fused",
        UNIFIED_VARIANT_ROWS,
        "pool",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_unified_variants_dense_list"] = calibration_table(
        results,
        "fused",
        UNIFIED_VARIANT_ROWS,
        "dense_list",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_pooled_corpus"] = calibration_table(
        results,
        "fused",
        POOLED_ROWS,
        "all",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_pooled_top10"] = calibration_table(
        results,
        "fused",
        POOLED_ROWS,
        "top10",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_lexical_matched"] = calibration_table(
        results,
        "lexical",
        LEXICAL_ROWS,
        "matched",
        SKILL_COUNT_SLOPE,
    )
    tables["supplement_correlation"] = diagnostics_table(results, CORRELATION_ROWS)
    tables["supplement_fits"] = fits_table(results)
    tables["supplement_intercept"] = diagnostics_table(results, INTERCEPT_ROWS)
    tables["supplement_bandwidth"] = backend_table(BANDWIDTH_ROWS, backends, "Bandwidth factor")
    tables["supplement_ivf_backgrounds"] = backend_table(
        WEIGHT_ROWS
        + [("Bayesian-Vector-Balanced", "`Bayesian-Vector-Balanced`"), ("Dense", "Dense")],
        [
            ("exact", hybrid),
            ("IVF, residual background", ivf),
            ("IVF, per-query background", ivf_query),
        ],
        "Configuration",
    )
    tables["supplement_ivf_background_statistics"] = ivf_background_table(ivf_diagnostics)
    tables["supplement_vector_top10"] = calibration_table(
        results,
        "vector",
        VECTOR_ROWS,
        "top10",
        ECE_SKILL_COUNT_SLOPE,
    )
    tables["supplement_admitted"] = admitted_table(results)
    tables["supplement_second_encoder_comparisons"] = comparison_table(
        second_per_query,
        SECOND_ENCODER_COMPARISONS,
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        **resampling,
    )
    tables["supplement_second_encoder_correlation"] = diagnostics_table(
        second_results,
        CORRELATION_ROWS,
    )
    tables["supplement_second_encoder_fits"] = fits_table(second_results)
    return tables


CEILING_ROWS = [
    ("conventional", "Theorem 3.7 (ceiling = 1, slope = 1)", "count (CV)"),
    ("fit_ceiling", "Ceiling fitted", "5-fold CV"),
    ("fit_slope", "Slope fitted", "5-fold CV"),
    ("fit_both", "Ceiling and slope fitted", "5-fold CV"),
    ("sure30", "Ceiling from 30 judged sure documents", "count (CV) + 30"),
    ("stratified90", "Ceiling and slope from 90 judged documents", "count (CV) + 90"),
    ("transfer_ceiling", "Ceiling of the other collections", "count (CV); source labels"),
    ("transfer_both", "Ceiling and slope of the other collections", "count (CV); source labels"),
]


def _ceiling_methods(entry: dict) -> dict:
    methods = dict(entry["calibration"])
    methods.update(entry.get("transfer", {}).get("calibration", {}))
    return methods


def ceiling_table(
    results: dict,
    pool: str,
    *,
    channel: str = "fused",
    rows: list | None = None,
) -> str:
    """A population-specific ceiling table using the public metric names."""
    proxy = {
        name: {"calibration": {channel: _ceiling_methods(result["channels"][channel])}}
        for name, result in results.items()
    }
    return calibration_table(
        proxy,
        channel,
        CEILING_ROWS if rows is None else rows,
        pool,
        SKILL_COUNT_SLOPE,
    )


def ceiling_constants_table(results: dict) -> str:
    datasets = _datasets(results)
    lines = _dataset_header(["Quantity"], datasets)

    def append(label, getter):
        lines.append("| " + " | ".join([label] + [getter(results[d]) for d in datasets]) + " |")

    def pair(parameters):
        return f"{parameters['ceiling']:g} / {parameters['slope']:.2f}"

    for channel in ("lexical", "dense", "fused"):
        append(
            f"{channel.title()}: ceiling / slope, all queries",
            lambda r, channel=channel: pair(r["channels"][channel]["global_fits"]["both"]),
        )
    append(
        "Fused: ceiling alone, all queries",
        lambda r: f"{r['channels']['fused']['global_fits']['ceiling']['ceiling']:g}",
    )
    append(
        "Fused: slope alone, all queries",
        lambda r: f"{r['channels']['fused']['global_fits']['slope']['slope']:.2f}",
    )
    append(
        "Fused: documents with conventional posterior >= 0.9",
        lambda r: str(r["channels"]["fused"]["sure_documents"]["count"]),
    )
    append(
        "Fused: judged relevant among them",
        lambda r: _format_value(
            r["channels"]["fused"]["sure_documents"]["relevant_fraction"], "{:.2f}"
        ),
    )

    def sure_estimates(result):
        draws = result["channels"]["fused"]["parameters"]["sure30"]
        values = [
            p["raw_ceiling_estimate"]
            for draw in draws
            for p in draw
            if p["raw_ceiling_estimate"] is not None
        ]
        return f"{np.mean(values):.2f} $\\pm$ {np.std(values):.2f}" if values else "--"

    append("Fused: ceiling from sure documents, mean and sd over CV draws", sure_estimates)

    def medians(result):
        draws = result["channels"]["fused"]["parameters"]["stratified90"]
        values = np.array([[p["ceiling"], p["slope"]] for draw in draws for p in draw])
        return f"{np.median(values[:, 0]):g} / {np.median(values[:, 1]):.2f}"

    append("Fused: ceiling / slope from stratified judgments, medians", medians)
    append(
        "Fused: ceiling / slope of the other collections",
        lambda r: (
            pair(r["channels"]["fused"]["transfer"]["parameters"]["transfer_both"][0][0])
            if "transfer" in r["channels"]["fused"]
            else "--"
        ),
    )

    def relevant_mean(result):
        summary = result["channels"]["fused"]["calibration"]["conventional"]["all"]
        return f"{summary['observed_relevant'] / result['n_queries']:.2f}"

    append("Mean relevant count per query", relevant_mean)
    return "\n".join(lines)


def quarter_table(results: dict) -> str:
    datasets = _datasets(results)
    lines = _dataset_header(["Posterior"], datasets)
    rows = [
        ("conventional", "Theorem 3.7"),
        ("full_ceiling", "Ceiling of the full corpus"),
        ("full_both", "Ceiling and slope of the full corpus"),
        ("quarter_both", "Ceiling and slope chosen on the part"),
    ]
    for key, label in rows:
        cells = []
        for dataset in datasets:
            entry = results[dataset]["channels"]["fused"]["quarter"]["calibration"][key]
            cells.append(
                " / ".join(
                    _format_value(entry[pool]["log_loss_skill"], "{:+.2f}")
                    for pool in ("top10", "pool")
                )
            )
        lines.append("| " + " | ".join([label] + cells) + " |")
    for key, label, fmt in (
        ("n_docs", "Documents kept", "{:,.0f}"),
        ("mean_relevant", "Mean relevant count per query", "{:.2f}"),
        ("zero_relevant_query_fraction", "Queries with no relevant document left", "{:.0%}"),
    ):
        cells = [_format_value(results[d]["quarter"][key], fmt) for d in datasets]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def count_sensitivity_table(results: dict) -> str:
    datasets = _datasets(results)
    lines = _dataset_header(["Count supplied", "Posterior"], datasets)
    for count in ("R1", "mean_x0.25", "mean_x0.5", "mean_x1", "mean_x2", "mean_x4"):
        for key, label in (
            ("conventional", "Theorem 3.7"),
            ("sure30", "Ceiling from sure documents"),
        ):
            cells = []
            for dataset in datasets:
                entry = results[dataset]["count_sensitivity"][count]["calibration"].get(key)
                if entry is None:
                    cells.append("--")
                    continue
                cells.append(
                    " / ".join(
                        _format_value(entry[pool]["log_loss_skill"], "{:+.2f}")
                        for pool in ("top10", "pool")
                    )
                )
            lines.append("| " + " | ".join([count, label] + cells) + " |")
    return "\n".join(lines)


def ceiling_uncertainty_table(results: dict) -> str:
    datasets = _datasets(results)
    lines = _dataset_header(["Posterior", "Pool"], datasets)
    for key, label, _ in CEILING_ROWS[:6]:
        for pool in ("top10", "pool"):
            cells = []
            for dataset in datasets:
                entry = results[dataset]["channels"]["fused"]["calibration"].get(key)
                if entry is None:
                    cells.append("--")
                    continue
                summary = entry[pool]
                lo, hi = summary["uncertainty"]["log_loss_skill_95ci"]
                cells.append(
                    f"{summary['log_loss_skill']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                    if lo is not None
                    else "--"
                )
            lines.append("| " + " | ".join([label, pool] + cells) + " |")
    return "\n".join(lines)


def build_scores_to_evidence_tables(
    results: dict,
    per_query: PerQuery,
    hybrid: PerQuery,
    ivf: PerQuery,
    ivf_residuals: PerQuery,
    ivf_diagnostics: dict,
    protocol: dict,
    second_results: dict,
    second_per_query: PerQuery,
    ceiling: dict,
    second_ceiling: dict,
    *,
    hybrid_diagnostics: dict,
    n_resamples: int = 10000,
    n_permutations: int = 100000,
) -> dict[str, str]:
    """The 23-table paper protocol; the older base-rate table is not inserted."""
    for source in (results, second_results, ceiling, second_ceiling):
        if any(entry.get("identity") != "text" for entry in source.values()):
            raise ValueError("From Scores to Evidence tables require recorded text identity")
    for measured, constants in ((results, ceiling), (second_results, second_ceiling)):
        for name in _datasets(measured):
            if measured[name]["dataset_fingerprint"] != constants[name]["dataset_fingerprint"]:
                raise ValueError(f"{name}: ceiling and reevaluation populations differ")
    old = build_tables(
        results,
        per_query,
        hybrid,
        ivf_residuals,
        ivf,
        ivf_diagnostics,
        protocol,
        None,
        second_results,
        second_per_query,
        n_resamples=n_resamples,
        n_permutations=n_permutations,
    )
    tables = {name: table for name, table in old.items() if not name.startswith("table")}
    for number in range(1, 8):
        tables[f"table{number:02d}"] = old[f"table{number:02d}"]
    for number in range(8, 19):
        tables[f"table{number:02d}"] = old[f"table{number + 1:02d}"]
    tables["table13"] = backend_table(
        WEIGHT_ROWS,
        [("exact", hybrid), ("IVF", ivf), ("IVF, residual background", ivf_residuals)],
        "Weight source",
    )
    tables["supplement_bandwidth"] = backend_table(
        BANDWIDTH_ROWS,
        [("exact", hybrid), ("IVF", ivf), ("IVF, residual background", ivf_residuals)],
        "Bandwidth factor",
    )
    tables["table19"] = ceiling_table(ceiling, "top10")
    tables["table20"] = ceiling_table(ceiling, "pool")
    tables["table21"] = ceiling_constants_table(ceiling)
    tables["table22"] = quarter_table(ceiling)
    tables["table23"] = count_sensitivity_table(ceiling)
    tables["supplement_ceiling_uncertainty"] = ceiling_uncertainty_table(ceiling)
    tables["supplement_tail_fits"] = tail_fits_table(results)
    tables["supplement_second_encoder_tail_fits"] = tail_fits_table(second_results)
    for channel in ("lexical", "dense"):
        tables[f"supplement_ceiling_{channel}_head"] = ceiling_table(
            ceiling, "top10", channel=channel
        )
        tables[f"supplement_ceiling_{channel}_list"] = ceiling_table(
            ceiling, "pool", channel=channel
        )
    for number, builder in (
        (19, lambda c: ceiling_table(c, "top10")),
        (20, lambda c: ceiling_table(c, "pool")),
        (21, ceiling_constants_table),
        (22, quarter_table),
        (23, count_sensitivity_table),
    ):
        tables[f"supplement_second_encoder_table{number}"] = builder(second_ceiling)
    tables["supplement_tail_ranking"] = ranking_table(
        per_query,
        [
            ("Additive-Query", "Bulk units"),
            ("Additive-Query-Tail", "Dense tail units"),
            ("Additive-Query-BothTail", "Both tail units"),
            ("Rank-Product", "Product of ranks"),
        ]
        + [
            (f"Additive-Query-Tail{depth}", f"Dense tail at depth {depth}")
            for depth in (10, 30, 100, 300, 1000)
        ],
        "NDCG@10",
    )
    tables["supplement_tail_comparisons"] = comparison_table(
        per_query,
        [
            ("Additive-Query-Tail", "Additive-Query", "Tail vs bulk units"),
            ("Additive-Query-Tail", "RRF", "Tail units vs RRF"),
            ("Additive-Query-Tail", "Convex", "Tail units vs convex"),
            ("Additive-Query-Tail", "Additive-Query-DensityRatio", "Tail vs Gaussian ratio"),
            ("Rank-Product", "RRF", "Product of ranks vs RRF"),
        ],
        "NDCG@10",
        significance_level=SIGNIFICANCE_LEVEL,
        n_resamples=n_resamples,
        n_permutations=n_permutations,
    )
    return tables


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tables of Section 6 of From Scores to Evidence",
    )
    parser.add_argument(
        "--results",
        default=os.path.join("benchmarks", "results"),
        help="Directory holding the result files (default: benchmarks/results)",
    )
    parser.add_argument("--out", required=True, help="Directory to write the tables to")
    parser.add_argument("--resamples", type=int, default=10000, help="Bootstrap resamples")
    parser.add_argument(
        "--legacy", action="store_true", help="Render the older id-identity archive"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100000,
        help="Sign-flip samples of the randomization test",
    )
    args = parser.parse_args()

    def path(name: str) -> str:
        return os.path.join(args.results, name)

    results, per_query = load_reevaluation(path("reevaluation.json"))
    second_results, second_per_query = load_reevaluation(path("reevaluation_mpnet.json"))
    if args.legacy:
        tables = build_tables(
            results,
            per_query,
            load_hybrid(path("hybrid_beir.json")),
            load_hybrid(path("hybrid_beir_ivf.json")),
            load_hybrid(path("hybrid_beir_ivf_query.json")),
            load_hybrid_diagnostics(path("hybrid_beir_ivf.json")),
            _load_json(path("protocol_diagnosis.json")),
            _load_json(path("base_rate.json")),
            second_results,
            second_per_query,
            n_resamples=args.resamples,
            n_permutations=args.permutations,
        )
    else:
        ceilings = []
        for filename in ("ceiling_slope.json", "ceiling_slope_mpnet.json"):
            payload = _load_json(path(filename))
            if (
                payload.get("sure_ceiling_estimator") != "observed_relevant_fraction"
                or payload.get("tail_depth") != 100
            ):
                raise ValueError(
                    f"{filename}: paper fraction estimator and tail depth 100 required"
                )
            ceilings.append(payload["results"])
        for filename in (
            "hybrid_beir.json",
            "hybrid_beir_ivf.json",
            "hybrid_beir_ivf_residuals.json",
        ):
            payload = _load_json(path(filename))
            if payload.get("protocol", {}).get("identity") != "text":
                raise ValueError(
                    f"{filename}: text identity metadata is required; --legacy is explicit"
                )
            for name, diagnostics in payload["diagnostics"].items():
                if diagnostics["dataset_fingerprint"] != results[name]["dataset_fingerprint"]:
                    raise ValueError(f"{filename}, {name}: reference populations differ")
        tables = build_scores_to_evidence_tables(
            results,
            per_query,
            load_hybrid(path("hybrid_beir.json")),
            load_hybrid(path("hybrid_beir_ivf.json")),
            load_hybrid(path("hybrid_beir_ivf_residuals.json")),
            load_hybrid_diagnostics(path("hybrid_beir_ivf.json")),
            _load_json(path("protocol_diagnosis.json")),
            second_results,
            second_per_query,
            ceilings[0],
            ceilings[1],
            hybrid_diagnostics=load_hybrid_diagnostics(path("hybrid_beir.json")),
            n_resamples=args.resamples,
            n_permutations=args.permutations,
        )

    os.makedirs(args.out, exist_ok=True)
    for name, text in tables.items():
        with open(os.path.join(args.out, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
    print(f"{len(tables)} tables written to {args.out}")


if __name__ == "__main__":
    main()
