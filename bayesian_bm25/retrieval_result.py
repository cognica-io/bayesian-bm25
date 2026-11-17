#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Structured retrieval output and per-query calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalResult:
    """Result object returned by ``retrieve(explain=True)``.

    Attributes
    ----------
    doc_ids : ndarray of shape (num_queries, k)
        Document indices for each query.
    probabilities : ndarray of shape (num_queries, k)
        Modeled relevance probabilities.
    explanations : list of list of BM25SignalTrace, or None
        Per-query, per-document trace through the probability pipeline.
        ``explanations[q][r]`` is the trace for query *q*, rank *r*.
        None when ``explain=False``.
    calibrations : list of QueryCalibration, or None
        Per-query calibration parameters (slope, offset, prevalence),
        one per query.  None unless the scorer was built with
        ``calibration="query"``.
    ranking_scores : ndarray or None
        Latent ranking values selected before probability conversion.
        These remain ordered when a ceiling saturates posterior log-odds.
    posterior_log_odds : ndarray or None
        Actual posterior log-odds of the selected documents, regardless
        of whether the requested output values are probabilities.
    """

    doc_ids: np.ndarray
    probabilities: np.ndarray
    explanations: list | None
    calibrations: list | None = None
    ranking_scores: np.ndarray | None = None
    posterior_log_odds: np.ndarray | None = None
