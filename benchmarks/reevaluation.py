#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Ranking and calibration measurements of "From Scores to Evidence".

Carries out the measurements of Sections 6.2 to 6.7 of "From Scores to
Evidence: Bayesian Calibration and Identifiability in Hybrid Retrieval"
(Tables 3, 4, 6, 8 to 12 and 14 to 18), the sequence of its Section 6.7
included, under the protocol of its Section 6.1:

* a document is identified by its exact (title, body) text and a query
  by its exact text (``--identity text``, the default; ``--identity id``
  is the rule of the standard evaluators);
* for each query the documents that carry its id and, under text
  identity, the documents whose body is its text are excluded before
  calibration and before scoring;
* every method is ranked on log-odds (or raw fusion scores) in double
  precision, never on a rounded probability;
* anything fitted to relevance labels is fitted on queries disjoint
  from those it is evaluated on (5-fold cross-validation over queries);
* per-query metric values are kept, so that confidence intervals and
  paired tests can be computed afterwards (see
  ``benchmarks/reevaluation_report.py``).

Experiments, per dataset:

  ranking      Baselines, the unified posterior (5.2) with calibration
               improved in steps (corpus-level, per-query with
               conventions, per-query fitted), a 2x2 isolation of the
               lexical and dense slopes, the log-linear vector evidence
               in bulk and in tail units (Section 4.8), and
               one-factor-at-a-time ablations between the
               range-normalized and the standardized per-query fusion.
  lexical      Calibration of the lexical posterior as a function of
               how the expected number of relevant documents is set.
  fused        Calibration of the fused posterior: the unified posterior
               (5.2) with density-ratio vector evidence on the candidate
               pool, and the pooled fusion with log-linear vector
               evidence over the corpus and on the pool; summed, pooled
               with a confidence exponent, and with post-fusion
               consistency.
  vector       Calibration of the dense signal and ranking of the
               density-ratio evidence: the form (4.1) and the exact
               posterior, a Gaussian background and a kernel estimate
               of the marginal, the monotone projection, and the
               weights and bandwidth of the relevant-distance estimate.
  threshold    Transfer of a single relevance threshold across queries.
  diagnostics  Saturation of the corpus-level calibration, correlation
               between the two signals, and the cost levels of the
               intercept.

Dependencies:
    pip install bayesian-bm25[scorer] sentence-transformers pytrec-eval-0.5
    pip install PyStemmer scipy

Usage:
    python -m benchmarks.reevaluation -d <beir-dir> -o results.json
    python -m benchmarks.reevaluation -d <beir-dir> --datasets scifact
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from bayesian_bm25.calibration import PlattCalibrator
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.consistency import dilute_offset, solve_offset
from bayesian_bm25.fusion import gaussian_evidence_weights
from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    _clamp_probability,
    logit,
    sigmoid,
)
from bayesian_bm25.query_calibration import score_dispersion
from bayesian_bm25.tail_scale import tail_scale
from bayesian_bm25.vector_probability import VectorProbabilityTransform
from benchmarks.calibration_accumulator import CalibrationAccumulator
from benchmarks.evidence_protocol import TAIL_DEPTH
from benchmarks.hybrid_beir import load_beir_dataset
from benchmarks.metrics import best_f1_threshold
from benchmarks.prepared_dataset import PreparedDataset, prepare_dataset
from benchmarks.query_signals import QuerySignals, query_signals
from benchmarks.ranking_eval import evaluate_run, mean_metrics, ranked_run

# Conventions for the expected number of relevant documents per query.
RELEVANT_COUNTS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DEFAULT_RELEVANT = 1.0

# Log-odds assigned to a lexical non-match by the first version's
# range-normalized fusion: the logit of the clamped probability 1e-10.
FLOOR_LOG_ODDS = float(logit(1e-10))

# Scales of the pooled evidence that correspond to confidence exponents
# rho for two signals: scale = 2 ** (rho - 1).
CONFIDENCE_EXPONENTS = (0.0, 0.5, 1.0)

N_TAIL_SAMPLE = 2000

# Variants of the density-ratio vector evidence: name suffix, options of
# the calibration, importance weights, bandwidth factor.  The form (4.1)
# is paired with the Gaussian background and the exact form of Remark 4.5
# with the kernel estimate of the marginal; "-Marginal" and
# "-Exact-Gaussian" are the two other combinations, which separate the
# effect of the background from that of the posterior form.
DENSITY_RATIO_VARIANTS = (
    ("", dict(posterior="approximate", background="gaussian"), "lexical", 1.0),
    (
        "-Monotone",
        dict(posterior="approximate", background="gaussian", monotone=True),
        "lexical",
        1.0,
    ),
    ("-Marginal", dict(posterior="approximate", background="marginal"), "lexical", 1.0),
    ("-Exact", dict(posterior="exact", background="marginal"), "lexical", 1.0),
    (
        "-Exact-Monotone",
        dict(posterior="exact", background="marginal", monotone=True),
        "lexical",
        1.0,
    ),
    ("-Exact-Gaussian", dict(posterior="exact", background="gaussian"), "lexical", 1.0),
    # Sensitivity of the approximate form to its ingredients: the kernel
    # bandwidth, the importance weights, and the sharpening of the weights
    # that the reference implementation applies in its automatic mode.
    ("-Bandwidth2", dict(posterior="approximate", background="gaussian"), "lexical", 2.0),
    ("-Uniform", dict(posterior="approximate", background="gaussian"), "uniform", 1.0),
    ("-Sharpened", dict(posterior="approximate", background="gaussian"), "sharpened", 1.0),
)

# Variants of the density ratio whose fused output is also measured after
# the two corrections of Section 5.3: the pooled rule (5.1) with
# rho = 1/2 in place of the sum, and post-fusion consistency.
CORRECTED_VARIANTS = ("", "-Marginal", "-Exact")

# Scale of the summed evidence of two signals under the pooled rule (5.1)
# with rho = 1/2: n ** (rho - 1).
POOLED_SCALE = 2.0**-0.5

# Calibration of the unified posterior (5.2) along the sequence of
# Section 6.8: corpus-level parameters, per-query parameters under the
# conventions, per-query parameters fitted to labels.  "-Consistent" adds
# post-fusion consistency: one intercept re-solved over the pool.
# "-Pooled" replaces the sum of the two evidence terms by the pooled rule.
UNIFIED_NAMES = (
    ["Unified-Corpus"]
    + [f"Unified-Query{suffix}" for suffix, _, _, _ in DENSITY_RATIO_VARIANTS]
    + [
        f"Unified-Query{suffix}{correction}"
        for suffix in CORRECTED_VARIANTS
        for correction in ("-Consistent", "-Pooled", "-Pooled-Consistent")
    ]
    + [
        f"Unified-Fit{suffix}{correction}"
        for suffix in ("", "-Exact")
        for correction in ("", "-Consistent", "-Pooled", "-Pooled-Consistent")
    ]
)

# A variance computed as a difference of second moments is zero up to
# this fraction of the second moment when all the scores are equal.
_VARIANCE_RELATIVE_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _min_max(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; zeros when the range is negligible."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _positive_median_and_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and 1/std of the positive entries (first-version pool statistics)."""
    positive = values[values > 0.0]
    if len(positive) == 0:
        return 0.0, 1.0
    std = float(np.std(positive))
    return float(np.median(positive)), (1.0 / std if std > 0.0 else 1.0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN when either input is constant."""
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k largest values (order unspecified)."""
    if len(values) <= k:
        return np.arange(len(values))
    return np.argpartition(values, len(values) - k)[len(values) - k :]


# ---------------------------------------------------------------------------
# Per-query statistics
# ---------------------------------------------------------------------------


def lexical_scale(sig: QuerySignals, corpus_scale: float, method: str = "std") -> float:
    """Dispersion of the matching scores (Definition 3.12).

    Falls back to the corpus-level scale when the query has fewer than
    two matching documents or its matching scores are all equal.
    """
    sigma = score_dispersion(sig.bm25[sig.matched], method)
    return sigma if sigma > 0.0 else corpus_scale


def document_prior_features(
    prepared: PreparedDataset,
    sig: QuerySignals,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Term-frequency and length features of the composite document prior.

    The term-frequency feature is the number of distinct query terms a
    document contains, as in the reference scorer; it is 0 for a
    document that matches no query term.
    """
    scorer = prepared.scorer
    corpus_idx = sig.doc_index[positions]
    tfs = np.zeros(len(positions), dtype=np.float64)
    active = sig.bm25[positions] > 0.0
    if np.any(active):
        tfs[active] = scorer._compute_tf_batch(
            corpus_idx[active],
            prepared.query_tokens[sig.index],
        )
    doc_len_ratios = scorer._doc_lengths[corpus_idx] / scorer._avgdl
    return tfs, doc_len_ratios


def corpus_lexical_log_odds(
    prepared: PreparedDataset,
    scores: np.ndarray,
    *,
    prior_log_odds: np.ndarray | None = None,
    base_rate: float | None = None,
) -> np.ndarray:
    """Corpus-level lexical posterior log-odds for the given scores."""
    log_odds = prepared.corpus_alpha * (scores - prepared.corpus_beta)
    if prior_log_odds is not None:
        log_odds = log_odds + prior_log_odds
    if base_rate is not None:
        log_odds = log_odds + float(logit(base_rate))
    return log_odds


def factorized_statistics(
    prepared: PreparedDataset,
    q_idx: int,
    population: int,
    alpha: float | None = None,
) -> tuple[float, float]:
    """Intercept and dispersion of a query from per-term index statistics.

    Treats the term contributions as independent across the population
    (Section 3.3, level 3).  For each query term ``t`` with document
    frequency ``p_t`` and contributions ``w_t(d)``, the cumulant
    generating function of the score factorizes,

        psi_q(alpha) = sum_t log(1 - p_t + p_t * E[exp(alpha * w_t) | t in d]),

    and the mean and variance of the score are sums of per-term
    moments.  Every quantity here could be stored at index time; they
    are computed from the posting lists to measure how much the
    independence assumption costs.

    Parameters
    ----------
    prepared : PreparedDataset
        The dataset (its BM25 index supplies the posting lists).
    q_idx : int
        Position of the query.
    population : int
        Size of the reference population.
    alpha : float or None
        Slope at which to evaluate ``psi_q``.  None uses the slope
        implied by the factorized dispersion, ``1 / sigma``.

    Returns
    -------
    (psi, sigma)
        The factorized cumulant generating function at ``alpha`` and the
        factorized dispersion of the matching scores.
    """
    bm25 = prepared.scorer._bm25
    data = bm25.scores["data"]
    indices = bm25.scores["indices"]
    indptr = bm25.scores["indptr"]
    excluded = prepared.excluded_for_query(q_idx)

    token_ids, counts = np.unique(
        np.asarray(bm25.get_tokens_ids(prepared.query_tokens[q_idx]), dtype=np.int64),
        return_counts=True,
    )
    contributions: list[np.ndarray] = []
    frequencies: list[float] = []
    for token_id, count in zip(token_ids, counts, strict=True):
        start, end = int(indptr[token_id]), int(indptr[token_id + 1])
        weights = data[start:end].astype(np.float64)
        if excluded.size > 0:
            weights = weights[~np.isin(indices[start:end], excluded)]
        if len(weights) == 0:
            continue
        # A repeated query token contributes once per occurrence.
        contributions.append(count * weights)
        frequencies.append(len(weights) / population)

    if not contributions:
        return 0.0, 0.0

    # Moments of the score over the whole population, then conditioned
    # on at least one term matching.
    mean = 0.0
    variance = 0.0
    log_no_match = 0.0
    for weights, p in zip(contributions, frequencies, strict=True):
        first = float(np.mean(weights))
        second = float(np.mean(weights * weights))
        mean += p * first
        variance += p * second - (p * first) ** 2
        log_no_match += math.log1p(-min(p, 1.0 - 1e-12))
    matched = -math.expm1(log_no_match)
    mean_matched = mean / matched
    second_matched = (variance + mean * mean) / matched
    variance_matched = second_matched - mean_matched**2
    # The variance is a difference of two second moments.  When every
    # matching score is the same (one term, equal contributions) it is
    # zero up to cancellation error, which must not be read as a scale.
    if variance_matched <= _VARIANCE_RELATIVE_TOLERANCE * second_matched:
        sigma = 0.0
    else:
        sigma = math.sqrt(variance_matched)

    slope = alpha if alpha is not None else (1.0 / sigma if sigma > 0.0 else 0.0)
    psi = 0.0
    for weights, p in zip(contributions, frequencies, strict=True):
        psi += math.log1p(p * float(np.mean(np.expm1(slope * weights))))
    return psi, sigma


# ---------------------------------------------------------------------------
# Fusion on the candidate pool
# ---------------------------------------------------------------------------


def fuse_candidates(
    prepared: PreparedDataset,
    sig: QuerySignals,
    *,
    lex_model: str,
    lex_norm: str,
    unmatched: str,
    dense_model: str,
    dense_norm: str,
    lex_prior: bool = False,
    lex_dispersion: str = "std",
    weight: float = 0.5,
    prior_features: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Fused score of every candidate under one configuration.

    The configuration is a point in the space spanned by the choices
    that separate the fusion variants of the first version from each
    other and from the additive posterior:

    lex_model
        ``"raw"`` uses the BM25 score itself as the lexical term;
        ``"corpus"`` uses the corpus-level posterior log-odds
        ``alpha * (s - beta) + logit(base_rate)``.
    lex_prior
        Add the composite document-quality prior to the lexical term.
    lex_norm
        ``"none"``; ``"std_match"`` divides by the dispersion of the
        matching scores over the whole population (Definition 3.12);
        ``"std_pool"`` centres at the median and divides by the standard
        deviation of the matched candidates; ``"range_pool"`` min-max
        normalizes over the candidates.
    unmatched
        How a candidate with no lexical match is scored:
        ``"consistent"`` gives it the lexical term of a score of zero;
        ``"floor"`` gives it the log-odds of the clamped probability
        1e-10; ``"dense_only"`` scores it by the dense term alone at
        half weight while matched candidates get the square-root-of-two
        confidence scaling (the first version's log-odds conjunction).
    dense_model
        ``"cos"`` uses the cosine similarity; ``"atanh"`` uses
        ``logit((1 + cos) / 2)``.
    dense_norm
        ``"corpus"`` multiplies by the corpus-level scale;
        ``"std_all"`` divides by the standard deviation over the whole
        population; ``"std_pool"`` centres at the median and divides by
        the standard deviation of the candidates with positive cosine;
        ``"range_pool"`` min-max normalizes over the candidates.
    weight
        Weight of the dense term; the lexical term gets ``1 - weight``.

    Returns
    -------
    array of shape (n_candidates,)
        Fused scores; higher is more relevant.
    """
    cand = sig.candidates
    s = sig.bm25[cand]
    c = sig.dense[cand]
    matched = s > 0.0

    # ----- lexical term ---------------------------------------------------
    if lex_model == "raw":
        lexical = s.copy()
    elif lex_model == "corpus":
        lexical = corpus_lexical_log_odds(
            prepared,
            s,
            base_rate=prepared.corpus_base_rate,
        )
    else:
        raise ValueError(f"unknown lex_model {lex_model!r}")

    if lex_prior:
        if prior_features is None:
            prior_features = document_prior_features(prepared, sig, cand)
        tfs, doc_len_ratios = prior_features
        lexical = lexical + np.asarray(
            logit(BayesianProbabilityTransform.composite_prior(tfs, doc_len_ratios)),
            dtype=np.float64,
        )

    if unmatched == "floor":
        lexical = np.where(matched, lexical, FLOOR_LOG_ODDS)
    elif unmatched not in ("consistent", "dense_only"):
        raise ValueError(f"unknown unmatched {unmatched!r}")

    if lex_norm == "none":
        pass
    elif lex_norm == "std_match":
        if lex_model != "raw":
            raise ValueError("std_match is defined for lex_model='raw'")
        corpus_scale = 1.0 / prepared.corpus_alpha
        lexical = lexical / lexical_scale(sig, corpus_scale, lex_dispersion)
    elif lex_norm == "std_pool":
        pool = lexical[matched]
        if len(pool) > 0:
            std = float(np.std(pool))
            lexical = (lexical - float(np.median(pool))) * (1.0 / std if std > 0.0 else 1.0)
    elif lex_norm == "range_pool":
        lexical = _min_max(lexical)
    else:
        raise ValueError(f"unknown lex_norm {lex_norm!r}")

    # ----- dense term -----------------------------------------------------
    if dense_model == "cos":
        dense = c.copy()
        dense_population = sig.dense
    elif dense_model == "atanh":
        dense = np.asarray(logit(_clamp_probability((1.0 + c) / 2.0)), dtype=np.float64)
        dense_population = np.asarray(
            logit(_clamp_probability((1.0 + sig.dense) / 2.0)),
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown dense_model {dense_model!r}")

    if dense_norm == "corpus":
        dense = prepared.dense_kappa * dense
    elif dense_norm == "std_all":
        dense = dense / float(np.std(dense_population))
    elif dense_norm == "std_pool":
        median, scale = _positive_median_and_scale(dense)
        dense = scale * (dense - median)
    elif dense_norm == "range_pool":
        dense = _min_max(dense)
    else:
        raise ValueError(f"unknown dense_norm {dense_norm!r}")

    # ----- pooling --------------------------------------------------------
    if unmatched == "dense_only":
        both = (weight * dense + (1.0 - weight) * lexical) * math.sqrt(2.0)
        return np.where(matched, both, weight * dense)
    return weight * dense + (1.0 - weight) * lexical


# Named ranking configurations built from fuse_candidates.  The comments
# give the role of each one in the paper.
FUSION_CONFIGS: dict[str, dict] = {
    # The additive posterior (5.2) with calibration improved in steps.
    "Additive-Corpus": dict(
        lex_model="corpus",
        lex_norm="none",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="corpus",
    ),
    "Additive-Query": dict(
        lex_model="raw",
        lex_norm="std_match",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="std_all",
    ),
    # 2x2 isolation of the slopes: which signal's scale is per query.
    "Slope-LexQuery-DenseCorpus": dict(
        lex_model="raw",
        lex_norm="std_match",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="corpus",
    ),
    "Slope-LexCorpus-DenseQuery": dict(
        lex_model="corpus",
        lex_norm="none",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="std_all",
    ),
    # The per-query standardized variant of the first version and the
    # steps from it to the additive posterior.
    "Local": dict(
        lex_model="raw",
        lex_norm="std_pool",
        unmatched="dense_only",
        dense_model="cos",
        dense_norm="std_pool",
    ),
    "Local+Consistent": dict(
        lex_model="raw",
        lex_norm="std_pool",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="std_pool",
    ),
    # The range-normalized variant of the first version and the steps
    # from it to the additive posterior.
    "Balanced": dict(
        lex_model="corpus",
        lex_prior=True,
        lex_norm="range_pool",
        unmatched="floor",
        dense_model="atanh",
        dense_norm="range_pool",
    ),
    "Balanced-NoPrior": dict(
        lex_model="corpus",
        lex_norm="range_pool",
        unmatched="floor",
        dense_model="atanh",
        dense_norm="range_pool",
    ),
    "Balanced-NoPrior-Consistent": dict(
        lex_model="corpus",
        lex_norm="range_pool",
        unmatched="consistent",
        dense_model="atanh",
        dense_norm="range_pool",
    ),
    "Balanced-NoPrior-Consistent-Cos": dict(
        lex_model="corpus",
        lex_norm="range_pool",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="range_pool",
    ),
    "Balanced-NoPrior-Consistent-Cos-Std": dict(
        lex_model="corpus",
        lex_norm="std_pool",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="std_pool",
    ),
    # Single-factor variations of the additive posterior.
    "Additive-Query-MAD": dict(
        lex_model="raw",
        lex_norm="std_match",
        lex_dispersion="mad",
        unmatched="consistent",
        dense_model="cos",
        dense_norm="std_all",
    ),
    "Additive-Query-Atanh": dict(
        lex_model="raw",
        lex_norm="std_match",
        unmatched="consistent",
        dense_model="atanh",
        dense_norm="std_all",
    ),
    "Additive-Query-Floor": dict(
        lex_model="raw",
        lex_norm="std_match",
        unmatched="floor",
        dense_model="cos",
        dense_norm="std_all",
    ),
    "Additive-Query-DenseOnly": dict(
        lex_model="raw",
        lex_norm="std_match",
        unmatched="dense_only",
        dense_model="cos",
        dense_norm="std_all",
    ),
}

# Dense weights for the sensitivity sweep of the additive posterior.
WEIGHT_SWEEP = (0.3, 0.4, 0.6, 0.7)


def score_rrf(sig: QuerySignals, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion over the two result lists."""
    scores = np.zeros(len(sig.candidates), dtype=np.float64)
    in_bm25 = sig.bm25_rank > 0
    in_dense = sig.dense_rank > 0
    scores[in_bm25] += 1.0 / (k + sig.bm25_rank[in_bm25])
    scores[in_dense] += 1.0 / (k + sig.dense_rank[in_dense])
    return scores


def score_convex(sig: QuerySignals, weight: float = 0.5) -> np.ndarray:
    """Convex combination of min-max normalized raw scores."""
    cand = sig.candidates
    return weight * _min_max(sig.dense[cand]) + (1.0 - weight) * _min_max(sig.bm25[cand])


def tail_units(
    sig: QuerySignals,
    depth: int = TAIL_DEPTH,
    *,
    fallback_lexical_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed population scores in lexical and dense mean-excess units.

    Only the retrieved head sets each unit. No candidate count is used
    as a corpus relevance count. A flat head has no tail unit; this
    experiment explicitly falls back to the original bulk unit. An
    entirely constant signal carries no discriminating score difference.
    """
    lex = sig.bm25[sig.matched]
    lex_scale = tail_scale(lex, k=min(depth, len(lex))) if len(lex) >= 2 else 0.0
    vec_scale = tail_scale(sig.dense, k=min(depth, len(sig.dense)))
    if not math.isfinite(fallback_lexical_scale) or fallback_lexical_scale <= 0.0:
        raise ValueError("fallback_lexical_scale must be finite and positive")
    if lex_scale <= 0.0:
        lex_scale = fallback_lexical_scale
    if vec_scale <= 0.0:
        vec_scale = float(np.std(sig.dense))
        if vec_scale <= 0.0:
            vec_scale = 1.0
    return sig.bm25 / lex_scale, sig.dense / vec_scale


def rank_product(sig: QuerySignals, *, tie_method: str = "ordinal") -> np.ndarray:
    """Sum log(N / rank); non-matches carry the observed score zero.

    Population-position ties reproduce the supplied experiment. Average
    ties are a separately named sensitivity comparison.
    """
    from scipy.stats import rankdata

    n = len(sig.bm25)
    lex_rank = rankdata(-sig.bm25, method=tie_method)
    vec_rank = rankdata(-sig.dense, method=tie_method)
    lexical = math.log(n) - np.log(lex_rank)
    lexical[~sig.matched] = 0.0
    dense = math.log(n) - np.log(vec_rank)
    return lexical[sig.candidates] + dense[sig.candidates]


# ---------------------------------------------------------------------------
# Posteriors over the whole population
# ---------------------------------------------------------------------------


def per_query_lexical_log_odds(
    z_lex: np.ndarray,
    expected_relevant: float,
    kappa: float = 1.0,
) -> np.ndarray:
    """Per-query lexical posterior log-odds (Procedure 3.14)."""
    g = kappa * z_lex
    return g - solve_offset(g, expected_relevant)


def fused_log_odds(
    z_lex: np.ndarray,
    z_vec: np.ndarray,
    expected_relevant: float,
    *,
    kappa_lex: float = 1.0,
    kappa_vec: float = 1.0,
    scale: float = 1.0,
    consistent: bool = True,
) -> np.ndarray:
    """Fused posterior log-odds over the population.

    With ``consistent=False`` this is the additive posterior (5.2) with
    the pooled evidence scaled by ``scale = n ** (rho - 1)``: each
    signal's evidence is a log-likelihood ratio whose intercept is
    solved on its own, and the prior is added once.  With
    ``consistent=True`` a single intercept is re-solved after pooling
    (post-fusion consistency), which restores calibration in the large
    whatever the dependence between the signals.
    """
    g_lex = kappa_lex * z_lex
    g_vec = kappa_vec * z_vec
    if consistent:
        g = scale * (g_lex + g_vec)
        return g - solve_offset(g, expected_relevant)

    prior = math.log(expected_relevant) - math.log(len(z_lex) - expected_relevant)
    evidence_lex = g_lex - solve_offset(g_lex, expected_relevant) - prior
    evidence_vec = g_vec - solve_offset(g_vec, expected_relevant) - prior
    return prior + scale * (evidence_lex + evidence_vec)


def unified_posterior(
    sig: QuerySignals,
    lexical_log_odds: np.ndarray,
    prior_logit: float,
    *,
    posterior: str = "approximate",
    background: str = "gaussian",
    monotone: bool = False,
    weighting: str = "lexical",
    bandwidth_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The unified posterior (5.2) of Theorem 5.13 on a query's candidates.

    ``logit P = logit(pi) + lexical evidence + log f_R(delta) / f_G(delta)``.

    The relevant-distance density ``f_R`` is the importance-weighted
    kernel estimate of Definition 4.8 over the query's retrieved
    distances, with the Silverman bandwidth of that definition.  The
    weights are the relevance probabilities the lexical signal assigns
    to the retrieved documents (Section 4.3).  The background ``f_G``
    is a Gaussian with the mean and variance of the query's distances
    over the corpus, or a kernel estimate of their marginal density.
    With ``posterior="exact"`` the posterior is ``pi f_R / f_G``, which
    is Bayes' rule when ``f_G`` is the marginal (Remark 4.5).

    Parameters
    ----------
    sig : QuerySignals
        Signals of the query.
    lexical_log_odds : array of shape (P,)
        Posterior log-odds of the lexical signal alone over the
        population: the prior plus the lexical evidence.
    prior_logit : float
        Log-odds of the prevalence that ``lexical_log_odds`` contains.
    posterior, monotone, bandwidth_factor
        Passed to ``VectorProbabilityTransform.calibrate_with_sample``.
    background : str
        ``"gaussian"`` (mean and variance of the corpus distances) or
        ``"marginal"`` (kernel estimate of their density, with the
        bandwidth of the relevant-distance estimate).
    weighting : str
        Importance weights: ``"lexical"`` (the lexical relevance
        probabilities), ``"sharpened"`` (the same after the softmax
        sharpening of the reference implementation's automatic mode), or
        ``"uniform"`` (the unweighted estimate).

    Returns
    -------
    (fused, vector_log_odds)
        Posterior log-odds of (5.2) for every candidate, and the
        posterior log-odds of the vector signal alone (the prior plus
        the vector evidence).
    """
    cand = sig.candidates
    distances = 1.0 - sig.dense
    in_dense = sig.dense_rank > 0
    order = np.argsort(sig.dense_rank[in_dense], kind="stable")
    sample_positions = cand[in_dense][order]
    sample_distances = distances[sample_positions]
    if weighting in ("lexical", "sharpened"):
        weights = np.asarray(sigmoid(lexical_log_odds[sample_positions]), dtype=np.float64)
        if weighting == "sharpened":
            weights = VectorProbabilityTransform._sharpen_weights(weights)
    elif weighting == "uniform":
        weights = np.ones(len(sample_distances), dtype=np.float64)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    if background not in ("gaussian", "marginal"):
        raise ValueError(f"unknown background {background!r}")

    vpt = VectorProbabilityTransform.fit_background(
        distances,
        base_rate=float(sigmoid(prior_logit)),
        background_model="gaussian",
    )
    vector_log_odds = np.asarray(
        vpt.calibrate_with_sample(
            distances[cand],
            sample_distances,
            weights=weights,
            method="kde",
            bandwidth_factor=bandwidth_factor,
            background_distances=distances if background == "marginal" else None,
            posterior=posterior,
            monotone=monotone,
            return_log_odds=True,
        ),
        dtype=np.float64,
    )
    fused = lexical_log_odds[cand] + (vector_log_odds - prior_logit)
    return fused, vector_log_odds


# ---------------------------------------------------------------------------
# Supervised fits on training queries
# ---------------------------------------------------------------------------


def training_sample(
    sig: QuerySignals,
    rng: np.random.Generator,
    n_tail: int = N_TAIL_SAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sample of a query's population for fitting.

    Every candidate and every relevant document is kept with weight 1;
    the remaining documents are represented by a uniform sample whose
    weight makes the sample total equal the population size.
    """
    population = len(sig.bm25)
    core = np.union1d(sig.candidates, np.flatnonzero(sig.labels > 0.0))
    in_core = np.zeros(population, dtype=bool)
    in_core[core] = True
    rest = np.flatnonzero(~in_core)
    if len(rest) > n_tail:
        tail = rng.choice(rest, size=n_tail, replace=False)
        tail_weight = len(rest) / n_tail
    else:
        tail = rest
        tail_weight = 1.0
    positions = np.concatenate([core, tail])
    weights = np.concatenate([np.ones(len(core)), np.full(len(tail), tail_weight)])
    return positions, weights


def fit_kappas(
    features: list[np.ndarray],
    labels: list[np.ndarray],
    weights: list[np.ndarray],
) -> np.ndarray:
    """Fit dimensionless slopes with one intercept per query.

    Maximizes the likelihood of ``logit P = sum_j kappa_j * z_j - c_q``
    over the slopes.  Each query's intercept is profiled out by the
    count constraint with the observed number of relevant documents,
    which is what maximum likelihood imposes on a per-query intercept
    (Proposition 3.22).

    Parameters
    ----------
    features : list of arrays of shape (m_q, n_signals)
        Standardized scores of each training query's sample.
    labels : list of arrays of shape (m_q,)
        Binary relevance labels.
    weights : list of arrays of shape (m_q,)
        Sample weights.

    Returns
    -------
    array of shape (n_signals,)
        The fitted slopes.
    """
    usable = [
        i
        for i in range(len(features))
        if 0.0 < float(np.sum(weights[i] * labels[i])) < float(np.sum(weights[i]))
    ]
    n_signals = features[0].shape[1]
    # Normalizing by the number of relevant documents keeps the objective
    # and its gradient of order one; normalizing by the population size
    # would shrink them below the optimizer's tolerances.
    total_weight = float(sum(np.sum(weights[i] * labels[i]) for i in usable))

    def objective(log_kappa: np.ndarray) -> tuple[float, np.ndarray]:
        kappa = np.exp(log_kappa)
        loss = 0.0
        grad = np.zeros(n_signals, dtype=np.float64)
        for i in usable:
            g = features[i] @ kappa
            observed = float(np.sum(weights[i] * labels[i]))
            log_odds = g - solve_offset(g, observed, weights=weights[i])
            loss += float(np.sum(weights[i] * (np.logaddexp(0.0, log_odds) - labels[i] * log_odds)))
            residual = weights[i] * (np.asarray(sigmoid(log_odds)) - labels[i])
            grad += residual @ features[i]
        # The intercepts are at their optimum, so the gradient with
        # respect to the slopes is the partial gradient.
        return loss / total_weight, grad * kappa / total_weight

    result = minimize(
        objective,
        np.zeros(n_signals, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(math.log(1e-3), math.log(1e3))] * n_signals,
    )
    return np.exp(result.x)


def fit_platt(
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    weights: list[np.ndarray],
) -> PlattCalibrator:
    """Platt scaling on the pooled training samples (convergent solver)."""
    calibrator = PlattCalibrator()
    calibrator.fit(
        np.concatenate(scores),
        np.concatenate(labels),
        sample_weight=np.concatenate(weights),
        max_iterations=200,
        tolerance=1e-9,
    )
    return calibrator


# ---------------------------------------------------------------------------
# Threshold transfer
# ---------------------------------------------------------------------------


def threshold_transfer(
    values: list[np.ndarray],
    labels: list[np.ndarray],
    missed: list[int],
    folds: np.ndarray,
) -> dict[str, float]:
    """Cross-validated transfer of one threshold across queries.

    For each fold the threshold that maximizes pooled F1 on the training
    queries is applied to the held-out queries.  Reports the pooled
    (micro) F1 over all held-out decisions, the mean per-query (macro)
    F1, the mean training F1, and the dispersion of the number of
    documents the threshold admits per query.
    """
    n_folds = int(np.max(folds)) + 1
    tp = fp = fn = 0.0
    per_query_f1: list[float] = []
    admitted: list[float] = []
    train_f1: list[float] = []

    for fold in range(n_folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        threshold, f1 = best_f1_threshold(
            np.concatenate([values[i] for i in train]),
            np.concatenate([labels[i] for i in train]),
            int(sum(missed[i] for i in train)),
        )
        train_f1.append(f1)
        for i in test:
            selected = values[i] >= threshold
            q_tp = float(np.sum(labels[i][selected]))
            q_fp = float(np.sum(selected)) - q_tp
            q_fn = float(np.sum(labels[i])) - q_tp + missed[i]
            tp += q_tp
            fp += q_fp
            fn += q_fn
            denom = 2.0 * q_tp + q_fp + q_fn
            per_query_f1.append(2.0 * q_tp / denom if denom > 0.0 else 0.0)
            admitted.append(float(np.sum(selected)))

    denom = 2.0 * tp + fp + fn
    admitted_arr = np.asarray(admitted)
    return {
        "train_micro_f1": float(np.mean(train_f1)),
        "test_micro_f1": 2.0 * tp / denom if denom > 0.0 else 0.0,
        "test_macro_f1": float(np.mean(per_query_f1)),
        "admitted_mean": float(np.mean(admitted_arr)),
        "admitted_median": float(np.median(admitted_arr)),
        "admitted_zero_fraction": float(np.mean(admitted_arr == 0)),
    }


# ---------------------------------------------------------------------------
# Main per-dataset pipeline
# ---------------------------------------------------------------------------


def _accumulators(
    names: list[str],
    pools: list[str],
) -> dict[str, dict[str, CalibrationAccumulator]]:
    return {
        name: {
            pool: CalibrationAccumulator(
                context=CalibrationContext(
                    reference_population="corpus" if pool == "all" else pool,
                    count_source="judgments",
                    slope_source="diagnostic",
                    ceiling_source="not_applicable",
                )
            )
            for pool in pools
        }
        for name in names
    }


def _update_pools(
    accumulators: dict[str, CalibrationAccumulator],
    log_odds: np.ndarray,
    labels: np.ndarray,
    pools: dict[str, np.ndarray | None],
) -> None:
    """Feed one query's posterior log-odds to the accumulator of each pool.

    ``pools`` maps a pool name to the population positions it covers,
    or to None for the whole population.
    """
    for pool, positions in pools.items():
        if positions is None:
            accumulators[pool].update(log_odds, labels)
        else:
            accumulators[pool].update(log_odds[positions], labels[positions])


def _update_unified(
    accumulators: dict[str, CalibrationAccumulator],
    posterior: np.ndarray,
    labels: np.ndarray,
    in_dense: np.ndarray,
    k: int,
) -> None:
    """Feed a posterior over the candidate pool to its accumulators.

    The pools are the whole candidate pool, the candidates of the dense
    result list (the distances from which the relevant-distance density
    was estimated), and the ``k`` best candidates.
    """
    accumulators["pool"].update(posterior, labels)
    accumulators["dense_list"].update(posterior[in_dense], labels[in_dense])
    top = _top_k(posterior, k)
    accumulators["top10"].update(posterior[top], labels[top])


def _update_corrected(
    fused_acc: dict[str, dict[str, CalibrationAccumulator]],
    name: str,
    fused: np.ndarray,
    prior_logit: float,
    expected_relevant: float,
    labels: np.ndarray,
    in_dense: np.ndarray,
    k: int,
) -> None:
    """Feed a unified posterior after the corrections of Section 5.3.

    ``fused`` is the posterior log-odds of (5.2) over the candidate
    pool: the prior plus the sum of the two evidence terms.

    * ``<name>-Consistent``: post-fusion consistency, one intercept so
      that the posteriors of the pool sum to ``expected_relevant``.
    * ``<name>-Pooled``: the pooled rule (5.1) with ``rho = 1/2``, the
      summed evidence scaled by ``2 ** -0.5`` and the prior added once.
      The ranking is that of the sum.
    * ``<name>-Pooled-Consistent``: both.
    """
    pooled = prior_logit + POOLED_SCALE * (fused - prior_logit)
    for suffix, posterior in (
        ("-Consistent", fused - solve_offset(fused, expected_relevant)),
        ("-Pooled", pooled),
        ("-Pooled-Consistent", pooled - solve_offset(pooled, expected_relevant)),
    ):
        _update_unified(fused_acc[f"{name}{suffix}"], posterior, labels, in_dense, k)


def _spread(values: np.ndarray, positions: np.ndarray, size: int) -> np.ndarray:
    """Place values given at some population positions into a full-length array."""
    full = np.full(size, np.nan, dtype=np.float64)
    full[positions] = values
    return full


def _summaries(
    accumulators: dict[str, dict[str, CalibrationAccumulator]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        name: {pool: acc.summary() for pool, acc in pools.items() if acc.count > 0}
        for name, pools in accumulators.items()
    }


def run_dataset(
    prepared: PreparedDataset,
    *,
    k: int = 10,
    retrieve_k: int = 1000,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Run every experiment on one dataset."""
    t_start = time.time()
    n_queries = prepared.n_queries
    corpus_scale = 1.0 / prepared.corpus_alpha
    corpus_ids = np.asarray(prepared.corpus_ids)
    rng = np.random.default_rng(seed)

    folds = np.empty(n_queries, dtype=np.int64)
    folds[rng.permutation(n_queries)] = np.arange(n_queries) % n_folds

    runs: dict[str, dict[str, dict[str, float]]] = {}

    def add_run(name: str, sig: QuerySignals, positions: np.ndarray, scores: np.ndarray) -> None:
        runs.setdefault(name, {})[sig.qid] = ranked_run(
            corpus_ids[sig.doc_index[positions]],
            scores,
        )

    # ----- calibration accumulators ----------------------------------------
    lexical_names = (
        ["Lex-Corpus-NoBaseRate", "Lex-Corpus-Percentile", "Lex-Corpus-0.001"]
        + [f"Lex-Query-R{r:g}" for r in RELEVANT_COUNTS]
        + ["Lex-Query-Roracle"]
    )
    lexical_cv_names = [
        "Lex-Query-Rmean",
        "Lex-Query-Fit",
        "Lex-Query-Fit-Roracle",
        "Lex-Platt-Raw",
        "Lex-Platt-Standardized",
        "Lex-MinMax",
    ]
    lexical_acc = _accumulators(
        lexical_names + lexical_cv_names,
        ["all", "matched", "top10"],
    )

    fused_names = ["Fused-Corpus"]
    for rho in CONFIDENCE_EXPONENTS:
        fused_names.append(f"Fused-Query-Additive-rho{rho:g}")
        fused_names.append(f"Fused-Query-Consistent-rho{rho:g}")
    fused_names.append("Fused-Query-Consistent-Roracle")
    fused_cv_names = [
        "Fused-Query-Consistent-rhoMeasured",
        "Fused-Query-Additive-rhoMeasured",
        "Fused-Fit-Rmean",
        "Fused-Fit-Roracle",
        "Fused-Query-Consistent-Rmean",
        "Fused-Tail-Consistent-R1",
        "Fused-Tail-Consistent-Rmean",
    ]
    # The pooled fusion is defined over the whole population; it is also
    # checked on the candidate pool, where the unified posterior lives.
    fused_acc = _accumulators(fused_names + fused_cv_names, ["all", "pool", "top10"])
    # The unified posterior (5.2) with density-ratio vector evidence is
    # the output of the retrieve-then-evaluate pipeline, so it is checked
    # on the candidate pool and on its head.
    fused_acc.update(_accumulators(UNIFIED_NAMES, ["pool", "dense_list", "top10"]))

    vector_names = [
        "Vec-GlobalSigmoid",
        "Vec-Arctan",
        "Vec-Query-R1",
        "Vec-Query-Roracle",
        "Vec-Tail-R1",
    ] + [f"Vec-DensityRatio{suffix}" for suffix, _, _, _ in DENSITY_RATIO_VARIANTS]
    vector_cv_names = ["Vec-Platt-Raw", "Vec-Platt-Standardized"]
    vector_acc = _accumulators(vector_names + vector_cv_names, ["retrieved", "top10"])

    # ----- state collected in the first pass ---------------------------------
    stats: dict[str, list[float]] = {
        key: []
        for key in (
            "n_match",
            "n_relevant",
            "query_tokens",
            "sigma_lex",
            "sigma_vec",
            "corr_all",
            "corr_matched",
            "corr_pool",
            "max_evidence_corpus",
            "tied_top100",
            "tied_top10",
            "tied_top100_log_odds",
            "tied_top10_log_odds",
            "likelihood_rounds_to_one",
            "offset_exact",
            "offset_dilute",
            "offset_topk",
            "offset_topk_bound",
            "offset_sampled",
            "offset_factorized",
            "psi_gap",
            "sigma_factorized_ratio",
            "top_error_factorized_full",
            "additive_count_ratio",
            "density_ratio_clipped",
            "pool_dense_weight_log_ratio",
            "floor_lexical_weight_ratio",
            "sigma_over_tail_lex",
            "sigma_over_tail_vec",
            "gaussian_tail_log_density_error",
            "dense_relevant_recall",
            "lexical_tail_fallback",
            "dense_tail_fallback",
        )
    }
    train_positions: list[np.ndarray] = []
    train_weights: list[np.ndarray] = []
    train_in_pool: list[np.ndarray] = []
    train_z: list[np.ndarray] = []
    train_tail: list[np.ndarray] = []
    train_raw: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    relevant_counts = np.zeros(n_queries, dtype=np.float64)
    matched_minmax: list[tuple[float, float]] = []

    threshold_names = [
        "Raw BM25",
        "Corpus-level log-odds",
        "Corpus-level log-odds + prior",
        "Per-query z-score",
        "Per-query log-odds (R=1)",
        "Per-query log-odds (R oracle)",
        "Per-query min-max",
    ]
    threshold_values: dict[str, list[np.ndarray]] = {n: [] for n in threshold_names}
    threshold_labels: list[np.ndarray] = []
    threshold_missed: list[int] = []

    # =========================================================================
    # Pass 1: everything that needs no relevance labels for fitting
    # =========================================================================
    for q_idx in range(n_queries):
        sig = query_signals(prepared, q_idx, retrieve_k)
        cand = sig.candidates
        labels = sig.labels
        matched = sig.matched
        n_match = int(np.sum(matched))
        population = len(sig.bm25)
        n_relevant = sig.n_relevant
        relevant_counts[q_idx] = n_relevant

        sigma_lex = lexical_scale(sig, corpus_scale)
        sigma_vec = score_dispersion(sig.dense)
        if not sigma_vec > 0.0:
            raise ValueError(
                f"query {sig.qid!r}: the similarities of the population have no "
                "dispersion, so the dense evidence is undefined"
            )
        z_lex = sig.bm25 / sigma_lex
        z_vec = sig.dense / sigma_vec
        tail_lex, tail_vec = tail_units(sig, TAIL_DEPTH, fallback_lexical_scale=sigma_lex)
        lexical_head = sig.bm25[sig.matched]
        lexical_tail = (
            tail_scale(lexical_head, k=min(TAIL_DEPTH, len(lexical_head)))
            if len(lexical_head) >= 2
            else 0.0
        )
        dense_tail = tail_scale(sig.dense, k=min(TAIL_DEPTH, len(sig.dense)))
        stats["lexical_tail_fallback"].append(float(lexical_tail <= 0.0))
        stats["dense_tail_fallback"].append(float(dense_tail <= 0.0))
        if lexical_tail > 0.0:
            stats["sigma_over_tail_lex"].append(sigma_lex / lexical_tail)
        stats["sigma_over_tail_vec"].append(
            sigma_vec * float(np.ptp(tail_vec)) / float(np.ptp(sig.dense))
        )

        stats["n_match"].append(n_match)
        stats["n_relevant"].append(n_relevant)
        stats["query_tokens"].append(len(prepared.query_tokens[q_idx]))
        stats["sigma_lex"].append(sigma_lex)
        stats["sigma_vec"].append(sigma_vec)
        stats["corr_all"].append(_safe_corr(sig.bm25, sig.dense))
        stats["corr_matched"].append(_safe_corr(sig.bm25[matched], sig.dense[matched]))
        stats["corr_pool"].append(_safe_corr(sig.bm25[cand], sig.dense[cand]))

        # ----- what pool statistics and the floor do to the relative weights ----
        # Standardizing over the candidate pool instead of the population
        # multiplies the weight of the dense term relative to the lexical
        # one by (sigma_lex_pool / sigma_vec_pool) / (sigma_lex / sigma_vec).
        cand_bm25 = sig.bm25[cand]
        cand_matched = cand_bm25 > 0.0
        sigma_lex_pool = score_dispersion(cand_bm25[cand_matched])
        sigma_vec_pool = score_dispersion(sig.dense[cand][sig.dense[cand] > 0.0])
        if sigma_lex_pool > 0.0 and sigma_vec_pool > 0.0:
            stats["pool_dense_weight_log_ratio"].append(
                math.log((sigma_lex_pool / sigma_vec_pool) / (sigma_lex / sigma_vec))
            )
        # Under range normalization the floor for non-matches stretches the
        # range of the lexical term, which divides the lexical differences
        # among matched candidates by this factor.
        if np.any(cand_matched) and not np.all(cand_matched):
            lexical_cand = corpus_lexical_log_odds(
                prepared,
                cand_bm25,
                base_rate=prepared.corpus_base_rate,
            )
            top = float(np.max(lexical_cand))
            consistent_range = top - float(np.min(lexical_cand))
            floor_range = top - min(FLOOR_LOG_ODDS, float(np.min(lexical_cand)))
            if floor_range > 0.0:
                stats["floor_lexical_weight_ratio"].append(consistent_range / floor_range)

        # ----- ranking: baselines ---------------------------------------------
        in_bm25 = sig.bm25_rank > 0
        in_dense = sig.dense_rank > 0
        add_run("BM25", sig, cand[in_bm25], sig.bm25[cand[in_bm25]])
        add_run("Dense", sig, cand[in_dense], sig.dense[cand[in_dense]])
        add_run("RRF", sig, cand, score_rrf(sig))
        add_run("Convex", sig, cand, score_convex(sig))
        add_run("Rank-Product", sig, cand, rank_product(sig))
        add_run("Additive-Query-Tail", sig, cand, z_lex[cand] + tail_vec[cand])
        add_run("Additive-Query-BothTail", sig, cand, tail_lex[cand] + tail_vec[cand])
        for multiple in (2, 4):
            add_run(
                f"Additive-Query-Dense{multiple}",
                sig,
                cand,
                z_lex[cand] + multiple * z_vec[cand],
            )
        # On the upper similarity tail, -log Gaussian density contributes
        # z^2/2. The candidate pool also contains lexical-only documents;
        # record the literal quadratic ablation there, including its shape.
        centered_vec = (sig.dense - float(np.mean(sig.dense))) / sigma_vec
        add_run(
            "Additive-Query-GaussianQuadratic",
            sig,
            cand,
            z_lex[cand] + 0.5 * centered_vec[cand] ** 2,
        )
        for depth in (10, 30, 100, 300, 1000):
            _, vec_at_depth = tail_units(sig, depth, fallback_lexical_scale=sigma_lex)
            add_run(
                f"Additive-Query-Tail{depth}",
                sig,
                cand,
                z_lex[cand] + vec_at_depth[cand],
            )

        # ----- ranking: fusion configurations ------------------------------------
        prior_features = document_prior_features(prepared, sig, cand)
        for name, config in FUSION_CONFIGS.items():
            add_run(
                name,
                sig,
                cand,
                fuse_candidates(prepared, sig, prior_features=prior_features, **config),
            )
        add_run(
            "Additive-Query+Prior",
            sig,
            cand,
            fuse_candidates(
                prepared,
                sig,
                prior_features=prior_features,
                **FUSION_CONFIGS["Additive-Query"],
            )
            + 0.5
            * np.asarray(
                logit(BayesianProbabilityTransform.composite_prior(*prior_features)),
                dtype=np.float64,
            ),
        )
        for weight in WEIGHT_SWEEP:
            add_run(
                f"Additive-Query-w{weight:g}",
                sig,
                cand,
                fuse_candidates(
                    prepared,
                    sig,
                    weight=weight,
                    **FUSION_CONFIGS["Additive-Query"],
                ),
            )

        # ----- lexical posteriors over the population ----------------------------
        matched_pos = np.flatnonzero(matched)
        top10_lex = matched_pos[_top_k(sig.bm25[matched_pos], k)] if n_match else matched_pos
        lexical_pools = {"all": None, "matched": matched_pos, "top10": top10_lex}

        tfs, doc_len_ratios = document_prior_features(prepared, sig, matched_pos)
        prior_log_odds = np.asarray(
            logit(BayesianProbabilityTransform.composite_prior(tfs, doc_len_ratios)),
            dtype=np.float64,
        )
        for name, base_rate in (
            ("Lex-Corpus-NoBaseRate", None),
            ("Lex-Corpus-Percentile", prepared.corpus_base_rate),
            ("Lex-Corpus-0.001", 0.001),
        ):
            # The first-version estimator: a document that matches no
            # query term has probability 0.
            log_odds = np.full(population, -np.inf, dtype=np.float64)
            log_odds[matched_pos] = corpus_lexical_log_odds(
                prepared,
                sig.bm25[matched_pos],
                prior_log_odds=prior_log_odds,
                base_rate=base_rate,
            )
            top = matched_pos[_top_k(log_odds[matched_pos], k)] if n_match else matched_pos
            _update_pools(
                lexical_acc[name],
                log_odds,
                labels,
                {"all": None, "matched": matched_pos, "top10": top},
            )

        lexical_r1: np.ndarray | None = None
        for r in RELEVANT_COUNTS:
            log_odds = per_query_lexical_log_odds(z_lex, r)
            if r == DEFAULT_RELEVANT:
                lexical_r1 = log_odds
            _update_pools(lexical_acc[f"Lex-Query-R{r:g}"], log_odds, labels, lexical_pools)
        assert lexical_r1 is not None

        lexical_oracle: np.ndarray | None = None
        if 0 < n_relevant < population:
            lexical_oracle = per_query_lexical_log_odds(z_lex, float(n_relevant))
            _update_pools(
                lexical_acc["Lex-Query-Roracle"],
                lexical_oracle,
                labels,
                lexical_pools,
            )

        # ----- saturation of the corpus-level calibration --------------------------
        if n_match:
            transform = prepared.scorer._transform
            probabilities = np.asarray(
                transform.score_to_probability(sig.bm25[matched_pos], tfs, doc_len_ratios),
                dtype=np.float64,
            )
            head = np.sort(probabilities)[::-1]
            head100 = head[:100]
            head10 = head[:10]
            stats["tied_top100"].append(len(head100) - len(np.unique(head100)))
            stats["tied_top10"].append(len(head10) - len(np.unique(head10)))
            # The same posterior in log-odds: ties that remain here are
            # genuine (documents with equal scores and priors); the
            # excess above is what rounding and clamping create.
            exact = np.sort(
                np.asarray(
                    transform.score_to_log_odds(sig.bm25[matched_pos], tfs, doc_len_ratios),
                    dtype=np.float64,
                )
            )[::-1]
            stats["tied_top100_log_odds"].append(
                100 - len(np.unique(exact[:100]))
                if len(exact) >= 100
                else len(exact) - len(np.unique(exact))
            )
            stats["tied_top10_log_odds"].append(len(exact[:10]) - len(np.unique(exact[:10])))
            stats["max_evidence_corpus"].append(
                prepared.corpus_alpha * (float(np.max(sig.bm25)) - prepared.corpus_beta)
            )
            stats["likelihood_rounds_to_one"].append(
                float(np.sum(np.asarray(transform.likelihood(sig.bm25[matched_pos])) == 1.0))
            )

        # ----- cost levels of the intercept (lexical, R = 1) -----------------------
        offset_exact = float(z_lex[0] - lexical_r1[0])
        stats["offset_exact"].append(offset_exact)
        stats["offset_dilute"].append(dilute_offset(z_lex, DEFAULT_RELEVANT))
        if n_match:
            top_lex = matched_pos[_top_k(sig.bm25[matched_pos], retrieve_k)]
            n_top = len(top_lex)
            n_rest = population - n_top
            if n_rest > 0:
                # Retrieved list only: every unscored document is treated
                # as a non-match.
                stats["offset_topk"].append(
                    solve_offset(
                        np.append(z_lex[top_lex], 0.0),
                        DEFAULT_RELEVANT,
                        weights=np.append(np.ones(n_top), n_rest),
                    )
                )
                # Unscored matches bounded above by the k-th score.
                n_unscored_matches = n_match - n_top
                kth = float(np.min(z_lex[top_lex]))
                stats["offset_topk_bound"].append(
                    solve_offset(
                        np.concatenate([z_lex[top_lex], [kth, 0.0]]),
                        DEFAULT_RELEVANT,
                        weights=np.concatenate(
                            [
                                np.ones(n_top),
                                [max(n_unscored_matches, 1e-12), population - n_match + 1e-12],
                            ]
                        ),
                    )
                )
                # Retrieved list plus a uniform sample of the rest.
                in_top = np.zeros(population, dtype=bool)
                in_top[top_lex] = True
                rest = np.flatnonzero(~in_top)
                sample = rng.choice(rest, size=min(1000, len(rest)), replace=False)
                stats["offset_sampled"].append(
                    solve_offset(
                        np.concatenate([z_lex[top_lex], z_lex[sample]]),
                        DEFAULT_RELEVANT,
                        weights=np.concatenate(
                            [np.ones(n_top), np.full(len(sample), len(rest) / len(sample))]
                        ),
                    )
                )
            else:
                for key in ("offset_topk", "offset_topk_bound", "offset_sampled"):
                    stats[key].append(offset_exact)
            # Per-term index statistics (independence across terms), in
            # the dilute form: c = log N + psi_q(alpha) - log R.
            psi_fact, sigma_fact = factorized_statistics(
                prepared,
                q_idx,
                population,
                alpha=1.0 / sigma_lex,
            )
            dilute_from_psi = math.log(population) - math.log(DEFAULT_RELEVANT)
            stats["offset_factorized"].append(dilute_from_psi + psi_fact)
            # The gap between the factorized and the exact cumulant
            # generating function is the effect of association.
            stats["psi_gap"].append(dilute_from_psi + psi_fact - stats["offset_dilute"][-1])
            # Slope and intercept both from index statistics.  The slope
            # differs from the exact one, so the two calibrations are
            # compared on the posterior log-odds of the best document.
            # A query whose matching scores have no dispersion falls back
            # to the corpus-level scale, as the exact estimator does.
            top_score = float(np.max(sig.bm25))
            exact_top = top_score / sigma_lex - offset_exact
            scale_fact = sigma_fact if sigma_fact > 0.0 else corpus_scale
            psi_full, _ = factorized_statistics(
                prepared,
                q_idx,
                population,
                alpha=1.0 / scale_fact,
            )
            stats["top_error_factorized_full"].append(
                top_score / scale_fact - (dilute_from_psi + psi_full) - exact_top
            )
            stats["sigma_factorized_ratio"].append(scale_fact / sigma_lex)
        else:
            # No document matches: every level sees the same population.
            for key in (
                "offset_topk",
                "offset_topk_bound",
                "offset_sampled",
                "offset_factorized",
            ):
                stats[key].append(offset_exact)
            stats["psi_gap"].append(0.0)
            stats["top_error_factorized_full"].append(0.0)
            stats["sigma_factorized_ratio"].append(1.0)

        # ----- fused posteriors over the population ----------------------------------
        fused_corpus = corpus_lexical_log_odds(
            prepared,
            sig.bm25,
            base_rate=prepared.corpus_base_rate,
        ) + prepared.dense_kappa * (prepared.dense_beta - (1.0 - sig.dense))
        _update_pools(
            fused_acc["Fused-Corpus"],
            fused_corpus,
            labels,
            {"all": None, "pool": cand, "top10": _top_k(fused_corpus, k)},
        )

        fused_pools = {"all": None, "pool": cand, "top10": _top_k(z_lex + z_vec, k)}
        for rho in CONFIDENCE_EXPONENTS:
            scale = 2.0 ** (rho - 1.0)
            for consistent in (False, True):
                log_odds = fused_log_odds(
                    z_lex,
                    z_vec,
                    DEFAULT_RELEVANT,
                    scale=scale,
                    consistent=consistent,
                )
                kind = "Consistent" if consistent else "Additive"
                _update_pools(
                    fused_acc[f"Fused-Query-{kind}-rho{rho:g}"],
                    log_odds,
                    labels,
                    fused_pools,
                )
                if not consistent and rho == 1.0:
                    stats["additive_count_ratio"].append(
                        float(np.sum(sigmoid(log_odds))) / DEFAULT_RELEVANT
                    )
        if 0 < n_relevant < population:
            log_odds = fused_log_odds(z_lex, z_vec, float(n_relevant))
            _update_pools(
                fused_acc["Fused-Query-Consistent-Roracle"],
                log_odds,
                labels,
                fused_pools,
            )

        # ----- dense signal: calibration on the retrieved list -------------------------
        retrieved = cand[in_dense]
        distances = 1.0 - sig.dense
        vector_pools = {
            "retrieved": retrieved,
            "top10": retrieved[_top_k(sig.dense[retrieved], k)],
        }

        _update_pools(
            vector_acc["Vec-GlobalSigmoid"],
            prepared.dense_kappa * (prepared.dense_beta - distances),
            labels,
            vector_pools,
        )
        _update_pools(
            vector_acc["Vec-Arctan"],
            np.asarray(
                logit(_clamp_probability((2.0 / np.pi) * np.arctan(5.0 * sig.dense))),
                dtype=np.float64,
            ),
            labels,
            vector_pools,
        )
        _update_pools(
            vector_acc["Vec-Query-R1"],
            z_vec - solve_offset(z_vec, DEFAULT_RELEVANT),
            labels,
            vector_pools,
        )
        _update_pools(
            vector_acc["Vec-Tail-R1"],
            tail_vec - solve_offset(tail_vec, DEFAULT_RELEVANT),
            labels,
            vector_pools,
        )
        if n_relevant:
            stats["dense_relevant_recall"].append(float(np.sum(labels[retrieved])) / n_relevant)
        if 0 < n_relevant < population:
            _update_pools(
                vector_acc["Vec-Query-Roracle"],
                z_vec - solve_offset(z_vec, float(n_relevant)),
                labels,
                vector_pools,
            )

        # ----- the unified posterior (5.2), density-ratio vector evidence -------------------
        # Step 1 of Section 6.8: corpus-level parameters.  The lexical
        # posterior has the corpus-level slope, midpoint and base rate; a
        # document that matches no query term carries the term at s = 0.
        lexical_corpus = corpus_lexical_log_odds(
            prepared,
            sig.bm25,
            base_rate=prepared.corpus_base_rate,
        )
        fused, _ = unified_posterior(
            sig,
            lexical_corpus,
            float(logit(prepared.corpus_base_rate)),
        )
        add_run("Additive-Corpus-DensityRatio", sig, cand, fused)
        _update_unified(fused_acc["Unified-Corpus"], fused, labels[cand], in_dense, k)

        # Step 2: per-query parameters under the conventions kappa = 1
        # and R_q = 1, with the variants of the vector evidence.
        prevalence = DEFAULT_RELEVANT / population
        prior_logit = math.log(prevalence) - math.log1p(-prevalence)
        gaussian_vector = None
        for suffix, options, weighting, bandwidth_factor in DENSITY_RATIO_VARIANTS:
            fused, vector_log_odds = unified_posterior(
                sig,
                lexical_r1,
                prior_logit,
                weighting=weighting,
                bandwidth_factor=bandwidth_factor,
                **options,
            )
            if suffix == "":
                gaussian_vector = vector_log_odds.copy()
            elif suffix == "-Marginal":
                stats["gaussian_tail_log_density_error"].append(
                    float(np.mean((gaussian_vector - vector_log_odds)[in_dense]))
                )
            if suffix == "-Exact":
                stats["density_ratio_clipped"].append(
                    float(np.mean(vector_log_odds >= float(logit(1.0 - 1e-10)) - 1e-6))
                )
            add_run(f"Additive-Query-DensityRatio{suffix}", sig, cand, fused)
            _update_pools(
                vector_acc[f"Vec-DensityRatio{suffix}"],
                _spread(vector_log_odds, cand, population),
                labels,
                vector_pools,
            )
            _update_unified(
                fused_acc[f"Unified-Query{suffix}"],
                fused,
                labels[cand],
                in_dense,
                k,
            )
            if suffix in CORRECTED_VARIANTS:
                _update_corrected(
                    fused_acc,
                    f"Unified-Query{suffix}",
                    fused,
                    prior_logit,
                    DEFAULT_RELEVANT,
                    labels[cand],
                    in_dense,
                    k,
                )

        # ----- threshold transfer inputs ------------------------------------------------------
        threshold_labels.append(labels[matched_pos].astype(np.float32))
        threshold_missed.append(int(np.sum(labels[~matched])))
        s_matched = sig.bm25[matched_pos]
        corpus_log_odds = corpus_lexical_log_odds(prepared, s_matched)
        threshold_values["Raw BM25"].append(s_matched)
        threshold_values["Corpus-level log-odds"].append(corpus_log_odds)
        threshold_values["Corpus-level log-odds + prior"].append(corpus_log_odds + prior_log_odds)
        mean_matched = float(np.mean(s_matched)) if n_match else 0.0
        threshold_values["Per-query z-score"].append((s_matched - mean_matched) / sigma_lex)
        threshold_values["Per-query log-odds (R=1)"].append(lexical_r1[matched_pos])
        threshold_values["Per-query log-odds (R oracle)"].append(
            lexical_oracle[matched_pos] if lexical_oracle is not None else lexical_r1[matched_pos]
        )
        max_matched = float(np.max(s_matched)) if n_match else 1.0
        threshold_values["Per-query min-max"].append(s_matched / max_matched)

        # ----- training sample for the supervised fits --------------------------------------
        positions, weights = training_sample(sig, rng)
        train_positions.append(positions)
        train_weights.append(weights)
        train_in_pool.append(np.isin(positions, cand))
        train_z.append(np.column_stack([z_lex[positions], z_vec[positions]]))
        train_tail.append(np.column_stack([tail_lex[positions], tail_vec[positions]]))
        train_raw.append(np.column_stack([sig.bm25[positions], sig.dense[positions]]))
        train_labels.append(labels[positions])
        matched_minmax.append(
            (float(np.min(s_matched)), float(np.max(s_matched))) if n_match else (np.inf, -np.inf)
        )

        if (q_idx + 1) % 200 == 0:
            print(
                f"    pass 1: {q_idx + 1}/{n_queries} queries ({time.time() - t_start:.0f}s)",
                flush=True,
            )

    # =========================================================================
    # Supervised fits, one per fold
    # =========================================================================
    pass1_seconds = time.time() - t_start
    fit_start = time.time()
    mean_corr = float(np.nanmean(stats["corr_all"]))
    measured_scale = 1.0 / (1.0 + mean_corr)
    fits: list[dict] = []
    for fold in range(n_folds):
        train = np.flatnonzero(folds != fold)
        z = [train_z[i] for i in train]
        tail = [train_tail[i] for i in train]
        raw = [train_raw[i] for i in train]
        y = [train_labels[i] for i in train]
        w = [train_weights[i] for i in train]

        platt_lex_raw = fit_platt([r[:, 0] for r in raw], y, w)
        platt_lex_std = fit_platt([f[:, 0] for f in z], y, w)
        platt_vec_raw = fit_platt([r[:, 1] for r in raw], y, w)
        platt_vec_std = fit_platt([f[:, 1] for f in z], y, w)
        lows = [matched_minmax[i][0] for i in train]
        highs = [matched_minmax[i][1] for i in train]
        # Slopes fitted on the candidate pool alone: the population is
        # then the documents on which the ranking is actually decided.
        pool = [train_in_pool[i] for i in train]
        kappas_pool = fit_kappas(
            [f[m] for f, m in zip(z, pool, strict=True)],
            [labels_q[m] for labels_q, m in zip(y, pool, strict=True)],
            [np.ones(int(np.sum(m))) for m in pool],
        )
        kappa_lex_solo = float(fit_kappas([f[:, :1] for f in z], y, w)[0])
        kappa_vec_solo = float(fit_kappas([f[:, 1:] for f in z], y, w)[0])
        # Theorem 5.8 (ii): with solo strengths tau and correlation r the
        # joint log-likelihood ratio has coefficients
        # (tau_i - r * tau_j) / (1 - r^2) on the standardized scores.
        predicted = gaussian_evidence_weights(
            [kappa_lex_solo, kappa_vec_solo], mean_corr
        ) * np.array([kappa_lex_solo, kappa_vec_solo])
        fits.append(
            {
                "kappa_lex": kappa_lex_solo,
                "kappa_vec": kappa_vec_solo,
                "theta_lex": float(fit_kappas([f[:, :1] for f in tail], y, w)[0]),
                "theta_vec": float(fit_kappas([f[:, 1:] for f in tail], y, w)[0]),
                "kappas_fused_predicted": [float(v) for v in predicted],
                "kappas_fused": [float(v) for v in fit_kappas(z, y, w)],
                "kappas_fused_pool": [float(v) for v in kappas_pool],
                "relevant_mean": float(np.mean(relevant_counts[train])),
                "platt_lex_raw": [platt_lex_raw.a, platt_lex_raw.b],
                "platt_lex_std": [platt_lex_std.a, platt_lex_std.b],
                "platt_vec_raw": [platt_vec_raw.a, platt_vec_raw.b],
                "platt_vec_std": [platt_vec_std.a, platt_vec_std.b],
                "minmax": [float(np.min(lows)), float(np.max(highs))],
            }
        )
        print(
            f"    fold {fold}: solo kappa lex={kappa_lex_solo:.3f} vec={kappa_vec_solo:.3f} | "
            f"fused fitted=({fits[-1]['kappas_fused'][0]:.3f}, "
            f"{fits[-1]['kappas_fused'][1]:.3f}) "
            f"predicted=({predicted[0]:.3f}, {predicted[1]:.3f}) "
            f"R_mean={fits[-1]['relevant_mean']:.2f}",
            flush=True,
        )

    # =========================================================================
    # Pass 2: configurations that use fitted parameters (held-out queries)
    # =========================================================================
    fit_seconds = time.time() - fit_start
    pass2_start = time.time()
    for q_idx in range(n_queries):
        sig = query_signals(prepared, q_idx, retrieve_k)
        fit = fits[int(folds[q_idx])]
        cand = sig.candidates
        labels = sig.labels
        matched_pos = np.flatnonzero(sig.matched)
        n_match = len(matched_pos)
        population = len(sig.bm25)
        n_relevant = sig.n_relevant
        z_lex = sig.bm25 / lexical_scale(sig, corpus_scale)
        z_vec = sig.dense / float(np.std(sig.dense))
        _, tail_vec = tail_units(
            sig,
            TAIL_DEPTH,
            fallback_lexical_scale=lexical_scale(sig, corpus_scale),
        )
        top10_lex = matched_pos[_top_k(sig.bm25[matched_pos], k)] if n_match else matched_pos
        lexical_pools = {"all": None, "matched": matched_pos, "top10": top10_lex}
        r_mean = min(fit["relevant_mean"], population - 1.0)

        _update_pools(
            lexical_acc["Lex-Query-Rmean"],
            per_query_lexical_log_odds(z_lex, r_mean),
            labels,
            lexical_pools,
        )
        _update_pools(
            lexical_acc["Lex-Query-Fit"],
            per_query_lexical_log_odds(z_lex, r_mean, fit["kappa_lex"]),
            labels,
            lexical_pools,
        )
        if 0 < n_relevant < population:
            _update_pools(
                lexical_acc["Lex-Query-Fit-Roracle"],
                per_query_lexical_log_odds(z_lex, float(n_relevant), fit["kappa_lex"]),
                labels,
                lexical_pools,
            )
        a, b = fit["platt_lex_raw"]
        _update_pools(lexical_acc["Lex-Platt-Raw"], a * sig.bm25 + b, labels, lexical_pools)
        a, b = fit["platt_lex_std"]
        _update_pools(
            lexical_acc["Lex-Platt-Standardized"],
            a * z_lex + b,
            labels,
            lexical_pools,
        )
        lo, hi = fit["minmax"]
        _update_pools(
            lexical_acc["Lex-MinMax"],
            np.asarray(
                logit(_clamp_probability(np.clip((sig.bm25 - lo) / (hi - lo), 0.0, 1.0))),
                dtype=np.float64,
            ),
            labels,
            lexical_pools,
        )

        # ----- fused: fitted slopes ---------------------------------------------------
        kappa_lex, kappa_vec = fit["kappas_fused"]
        fitted_scores = kappa_lex * z_lex + kappa_vec * z_vec
        add_run("Additive-Query-Fit", sig, cand, fitted_scores[cand])
        # Step 3 of Section 6.8 for the unified posterior (5.2): the
        # lexical slope and the expected count fitted to labels.
        lexical_fit = per_query_lexical_log_odds(z_lex, r_mean, fit["kappa_lex"])
        prior_fit = math.log(r_mean) - math.log(population - r_mean)
        for name, posterior, background in (
            ("Unified-Fit", "approximate", "gaussian"),
            ("Unified-Fit-Exact", "exact", "marginal"),
        ):
            fused, _ = unified_posterior(
                sig,
                lexical_fit,
                prior_fit,
                posterior=posterior,
                background=background,
            )
            if posterior == "approximate":
                add_run("Additive-QueryFit-DensityRatio", sig, cand, fused)
            in_dense = sig.dense_rank > 0
            _update_unified(fused_acc[name], fused, labels[cand], in_dense, k)
            _update_corrected(
                fused_acc,
                name,
                fused,
                prior_fit,
                r_mean,
                labels[cand],
                in_dense,
                k,
            )
        pool_lex, pool_vec = fit["kappas_fused_pool"]
        add_run(
            "Additive-Query-FitPool",
            sig,
            cand,
            pool_lex * z_lex[cand] + pool_vec * z_vec[cand],
        )
        fit_pools = {"all": None, "pool": cand, "top10": _top_k(fitted_scores, k)}
        _update_pools(
            fused_acc["Fused-Fit-Rmean"],
            fused_log_odds(z_lex, z_vec, r_mean, kappa_lex=kappa_lex, kappa_vec=kappa_vec),
            labels,
            fit_pools,
        )
        if 0 < n_relevant < population:
            _update_pools(
                fused_acc["Fused-Fit-Roracle"],
                fused_log_odds(
                    z_lex,
                    z_vec,
                    float(n_relevant),
                    kappa_lex=kappa_lex,
                    kappa_vec=kappa_vec,
                ),
                labels,
                fit_pools,
            )

        for name, vector, count in (
            ("Fused-Query-Consistent-Rmean", z_vec, r_mean),
            ("Fused-Tail-Consistent-R1", tail_vec, 1.0),
            ("Fused-Tail-Consistent-Rmean", tail_vec, r_mean),
        ):
            _update_pools(
                fused_acc[name],
                fused_log_odds(z_lex, vector, count, scale=POOLED_SCALE),
                labels,
                {"all": None, "pool": cand, "top10": _top_k(z_lex + vector, k)},
            )

        # ----- fused: exponent from the measured correlation (label-free) ----------------
        fused_pools = {"all": None, "pool": cand, "top10": _top_k(z_lex + z_vec, k)}
        for consistent in (False, True):
            kind = "Consistent" if consistent else "Additive"
            _update_pools(
                fused_acc[f"Fused-Query-{kind}-rhoMeasured"],
                fused_log_odds(
                    z_lex,
                    z_vec,
                    DEFAULT_RELEVANT,
                    scale=measured_scale,
                    consistent=consistent,
                ),
                labels,
                fused_pools,
            )

        # ----- dense signal: Platt scaling ---------------------------------------------------
        retrieved = cand[sig.dense_rank > 0]
        vector_pools = {
            "retrieved": retrieved,
            "top10": retrieved[_top_k(sig.dense[retrieved], k)],
        }
        for name, key, values in (
            ("Vec-Platt-Raw", "platt_vec_raw", sig.dense),
            ("Vec-Platt-Standardized", "platt_vec_std", z_vec),
        ):
            a, b = fit[key]
            _update_pools(vector_acc[name], a * values + b, labels, vector_pools)

    # =========================================================================
    # Evaluation
    # =========================================================================
    ranking: dict[str, dict] = {}
    for name, run in runs.items():
        per_query = evaluate_run(prepared.qrels, run, k=k)
        ranking[name] = {
            "mean": mean_metrics(per_query),
            "per_query": {
                metric: [values[qid] for qid in prepared.query_ids]
                for metric, values in per_query.items()
            },
        }

    pass2_seconds = time.time() - pass2_start
    thresholds = {
        name: threshold_transfer(
            threshold_values[name],
            threshold_labels,
            threshold_missed,
            folds,
        )
        for name in threshold_names
    }

    def mean_of(key: str) -> float:
        values = np.asarray(stats[key], dtype=np.float64)
        return float(np.nanmean(values)) if len(values) else float("nan")

    offsets = np.asarray(stats["offset_exact"])
    diagnostics = {
        "query_tokens_mean": mean_of("query_tokens"),
        "query_tokens_median": float(np.median(stats["query_tokens"])),
        "matched_fraction_mean": mean_of("n_match") / prepared.n_docs,
        "relevant_mean": mean_of("n_relevant"),
        "sigma_lex_cv": float(np.std(stats["sigma_lex"]) / np.mean(stats["sigma_lex"])),
        "sigma_lex_ratio_p95_p5": float(
            np.percentile(stats["sigma_lex"], 95) / np.percentile(stats["sigma_lex"], 5)
        ),
        "sigma_lex_mean": mean_of("sigma_lex"),
        "sigma_vec_mean": mean_of("sigma_vec"),
        "pool_dense_weight_ratio_gmean": math.exp(mean_of("pool_dense_weight_log_ratio")),
        "floor_lexical_weight_ratio_mean": mean_of("floor_lexical_weight_ratio"),
        "relevant_unmatched_fraction": float(
            np.sum(threshold_missed) / max(float(np.sum(relevant_counts)), 1.0)
        ),
        "queries_with_unmatched_relevant": float(np.mean(np.asarray(threshold_missed) > 0)),
        "corpus_scale": corpus_scale,
        "sigma_vec_cv": float(np.std(stats["sigma_vec"]) / np.mean(stats["sigma_vec"])),
        "corr_all": mean_of("corr_all"),
        "corr_matched": mean_of("corr_matched"),
        "corr_pool": mean_of("corr_pool"),
        "rho_from_corr_all": float(1.0 - math.log(1.0 + mean_corr) / math.log(2.0)),
        "max_evidence_corpus_mean": mean_of("max_evidence_corpus"),
        "max_evidence_corpus_p95": float(np.percentile(stats["max_evidence_corpus"], 95)),
        "tied_top100_mean": mean_of("tied_top100"),
        "tied_top10_mean": mean_of("tied_top10"),
        "tied_top100_log_odds_mean": mean_of("tied_top100_log_odds"),
        "tied_top10_log_odds_mean": mean_of("tied_top10_log_odds"),
        "queries_with_tied_top10": float(np.mean(np.asarray(stats["tied_top10"]) > 0)),
        "queries_with_tied_top100": float(np.mean(np.asarray(stats["tied_top100"]) > 0)),
        "queries_with_rounding_ties_top10": float(
            np.mean(np.asarray(stats["tied_top10"]) > np.asarray(stats["tied_top10_log_odds"]))
        ),
        "queries_with_rounding_ties_top100": float(
            np.mean(np.asarray(stats["tied_top100"]) > np.asarray(stats["tied_top100_log_odds"]))
        ),
        "likelihood_rounds_to_one_mean": mean_of("likelihood_rounds_to_one"),
        "queries_with_likelihood_one": float(
            np.mean(np.asarray(stats["likelihood_rounds_to_one"]) > 0)
        ),
        "additive_count_ratio_mean": mean_of("additive_count_ratio"),
        "additive_count_ratio_median": float(np.median(stats["additive_count_ratio"])),
        "density_ratio_clipped_mean": mean_of("density_ratio_clipped"),
        "sigma_over_tail_lex_median": float(np.median(stats["sigma_over_tail_lex"])),
        "sigma_over_tail_vec_median": float(np.median(stats["sigma_over_tail_vec"])),
        "gaussian_tail_log_density_error_mean": mean_of("gaussian_tail_log_density_error"),
        "dense_relevant_recall_mean": mean_of("dense_relevant_recall"),
        "lexical_tail_fallback_fraction": mean_of("lexical_tail_fallback"),
        "dense_tail_fallback_fraction": mean_of("dense_tail_fallback"),
    }
    diagnostics["psi_gap_mean"] = mean_of("psi_gap")
    diagnostics["psi_gap_negative_fraction"] = float(np.mean(np.asarray(stats["psi_gap"]) <= 1e-9))
    diagnostics["sigma_factorized_ratio_mean"] = mean_of("sigma_factorized_ratio")
    diagnostics["sigma_factorized_below_fraction"] = float(
        np.mean(np.asarray(stats["sigma_factorized_ratio"]) <= 1.0 + 1e-9)
    )
    top_error = np.asarray(stats["top_error_factorized_full"])
    diagnostics["top_error_factorized_full_mean"] = float(np.mean(top_error))
    diagnostics["top_error_factorized_full_abs_mean"] = float(np.mean(np.abs(top_error)))
    for key in (
        "offset_dilute",
        "offset_topk",
        "offset_topk_bound",
        "offset_sampled",
        "offset_factorized",
    ):
        values = np.asarray(stats[key])
        if len(values) == len(offsets):
            delta = values - offsets
            diagnostics[f"{key}_error_mean"] = float(np.mean(delta))
            diagnostics[f"{key}_abs_error_mean"] = float(np.mean(np.abs(delta)))
            diagnostics[f"{key}_abs_error_p95"] = float(np.percentile(np.abs(delta), 95))

    print(f"  {prepared.name}: done in {time.time() - t_start:.0f}s", flush=True)

    return {
        "dataset": prepared.name,
        "n_docs": prepared.n_docs,
        "n_queries": n_queries,
        "n_identical_ids_excluded": int(np.sum(prepared.excluded >= 0)),
        "identity": prepared.identity,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "reference_population": "query_excluded_corpus",
        "relevance_target": "judged_relevance",
        "degenerate_tail_policy": "use_recorded_bulk_unit",
        "tail_depth": TAIL_DEPTH,
        "preparation_seconds": prepared.preparation_seconds,
        "evaluation_seconds": time.time() - t_start,
        "timing": {
            "preparation_seconds": prepared.preparation_seconds,
            "label_free_experiments_seconds": pass1_seconds,
            "supervised_fit_seconds": fit_seconds,
            "held_out_experiments_seconds": pass2_seconds,
            "total_seconds": prepared.preparation_seconds + time.time() - t_start,
        },
        "corpus": {
            "alpha": prepared.corpus_alpha,
            "beta": prepared.corpus_beta,
            "base_rate": prepared.corpus_base_rate,
            "dense_kappa": prepared.dense_kappa,
            "dense_beta": prepared.dense_beta,
        },
        "fits": fits,
        "ranking": ranking,
        "calibration": {
            "lexical": _summaries(lexical_acc),
            "fused": _summaries(fused_acc),
            "vector": _summaries(vector_acc),
        },
        "threshold": thresholds,
        "diagnostics": diagnostics,
    }


def _run_worker(options: dict, name: str) -> tuple[str, dict]:
    dataset_dir = os.path.join(options["beir_dir"], name)
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"dataset directory not found: {dataset_dir}")
    print(f"\n=== {name} ===", flush=True)
    prepared = prepare_dataset(
        dataset_dir,
        name,
        options["model"],
        options["cache_dir"],
        ignore_identical_ids=not options["keep_identical_ids"],
        identity=options["identity"],
    )
    if options.get("refresh_tail"):
        from benchmarks.tail_experiments import refresh_tail_experiments

        previous = json.loads(Path(options["output"]).read_text(encoding="ascii"))["results"][name]
        return name, refresh_tail_experiments(
            prepared,
            previous,
            k=options["top_k"],
            retrieve_k=options["retrieve_k"],
        )
    return name, run_dataset(
        prepared,
        k=options["top_k"],
        retrieve_k=options["retrieve_k"],
        n_folds=options["folds"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluation of the unified Bayesian hybrid search framework",
    )
    parser.add_argument("--identity", choices=["id", "text"], default="text")
    parser.add_argument("-d", "--beir-dir", required=True, help="Path to BEIR data directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"],
        help="BEIR datasets to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Evaluation depth")
    parser.add_argument(
        "-R",
        "--retrieve-k",
        type=int,
        default=1000,
        help="Retrieval depth per signal (default: 1000)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--keep-identical-ids",
        action="store_true",
        help="Do not exclude the document whose id equals the query's",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Embedding cache directory (default: {beir-dir}/.cache/embeddings)",
    )
    parser.add_argument("-o", "--output", default=None, help="Path to save JSON results")
    parser.add_argument("--workers", type=int, default=1, help="Independent dataset processes")
    parser.add_argument("--resume", action="store_true", help="Reuse validated completed datasets")
    parser.add_argument(
        "--refresh-tail",
        action="store_true",
        help="Refresh tail rows on identical saved populations without repeating density fits",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.beir_dir, ".cache", "embeddings")
    all_results: dict[str, dict] = {}
    metadata = {
        "model": args.model,
        "k": args.top_k,
        "retrieve_k": args.retrieve_k,
        "identical_ids_excluded": not args.keep_identical_ids,
        "identity": args.identity,
        "identity_version": 1,
    }
    if args.resume or args.refresh_tail:
        if not args.output:
            raise SystemExit("resume requires an output file")
        if Path(args.output).exists():
            previous = json.loads(Path(args.output).read_text(encoding="ascii"))
            if any(previous.get(key) != value for key, value in metadata.items()):
                raise SystemExit("resume protocol metadata differs")
            all_results = previous["results"]
            for name, result in all_results.items():
                data = load_beir_dataset(os.path.join(args.beir_dir, name), identity=args.identity)
                if (
                    len(result["fits"]) != args.folds
                    or result["dataset_fingerprint"] != data["dataset_fingerprint"]
                    or "Additive-Query-Tail" not in result["ranking"]
                ):
                    raise SystemExit(f"{name}: resumed data, folds, or score protocol differs")

    def record(name, result):
        all_results[name] = result
        ranking = result["ranking"]
        for method in ("BM25", "Dense", "RRF", "Convex", "Additive-Query", "Balanced"):
            print(
                f"    {method:<16} NDCG@{args.top_k} = "
                f"{100 * ranking[method]['mean'][f'NDCG@{args.top_k}']:.2f}"
            )

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(args.output + ".tmp")
            temporary.write_text(
                json.dumps(dict(metadata, n_folds=args.folds, seed=42, results=all_results)),
                encoding="ascii",
            )
            temporary.replace(args.output)
            print(f"  results saved to {args.output}", flush=True)

    if args.refresh_tail and any(name not in all_results for name in args.datasets):
        raise SystemExit("tail refresh requires completed results for each requested dataset")
    pending = (
        list(args.datasets)
        if args.refresh_tail
        else [name for name in args.datasets if name not in all_results]
    )
    options = dict(vars(args), cache_dir=cache_dir)
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    if args.workers == 1:
        for name in pending:
            record(*_run_worker(options, name))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(_run_worker, options, name) for name in pending]
            for job in as_completed(jobs):
                record(*job.result())


if __name__ == "__main__":
    main()
