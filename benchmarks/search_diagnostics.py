#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark-local search diagnostics for query-adaptive dense gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.simple_ivf import IVFSearchResult, SimpleIVF

_EPSILON = 1e-12


@dataclass
class SearchDiagnostics:
    """Query-local retrieval diagnostics for backend-agnostic gating.

    ``accepted_distances`` are the distances for the kept neighborhood
    around the query. ``contrast_distances`` are distances from a nearby
    background shell or trace. ``purity`` and ``coverage`` are optional
    reliability terms for backend-specific routing quality.
    """

    accepted_distances: np.ndarray
    contrast_distances: np.ndarray
    purity: float = 1.0
    coverage: float = 1.0

    def __post_init__(self) -> None:
        self.accepted_distances = np.asarray(
            self.accepted_distances, dtype=np.float64,
        )
        self.contrast_distances = np.asarray(
            self.contrast_distances, dtype=np.float64,
        )
        self.purity = float(np.clip(self.purity, 0.0, 1.0))
        self.coverage = float(np.clip(self.coverage, 0.0, 1.0))

    @property
    def cohesion(self) -> float:
        if len(self.accepted_distances) == 0:
            return 1.0
        return float(np.mean(self.accepted_distances))

    @property
    def separation(self) -> float:
        if len(self.contrast_distances) == 0:
            return self.cohesion
        return float(np.mean(self.contrast_distances))

    @property
    def reliability(self) -> float:
        return float(np.clip(self.purity * self.coverage, 0.0, 1.0))


def _scores_to_distances(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 - scores


def build_exact_search_diagnostics(
    dense_top_scores: np.ndarray,
    *,
    local_k: int = 10,
    shell_k: int = 10,
) -> SearchDiagnostics:
    """Build diagnostics from exact or exact-like top-rank shells."""
    dense_top_scores = np.asarray(dense_top_scores, dtype=np.float64)
    if len(dense_top_scores) == 0:
        return SearchDiagnostics([], [], purity=0.0, coverage=0.0)

    local_k = max(1, min(local_k, len(dense_top_scores)))
    accepted = _scores_to_distances(dense_top_scores[:local_k])

    shell_start = local_k
    shell_end = min(shell_start + shell_k, len(dense_top_scores))
    if shell_end <= shell_start:
        contrast = np.empty(0, dtype=np.float64)
    else:
        contrast = _scores_to_distances(dense_top_scores[shell_start:shell_end])

    return SearchDiagnostics(
        accepted,
        contrast,
        purity=1.0,
        coverage=1.0,
    )


def build_ivf_search_diagnostics(
    dense_top_scores: np.ndarray,
    top_cell_ids: np.ndarray,
    search_result: IVFSearchResult,
    dense_index: SimpleIVF,
    *,
    local_k: int = 10,
    shell_k: int = 10,
) -> SearchDiagnostics:
    """Build diagnostics from IVF candidate shells and routing purity."""
    dense_top_scores = np.asarray(dense_top_scores, dtype=np.float64)
    top_cell_ids = np.asarray(top_cell_ids, dtype=np.int32)
    if len(dense_top_scores) == 0 or len(top_cell_ids) == 0:
        return SearchDiagnostics([], [], purity=0.0, coverage=0.0)

    local_k = max(1, min(local_k, len(dense_top_scores), len(top_cell_ids)))
    local_scores = dense_top_scores[:local_k]
    local_cells = top_cell_ids[:local_k]

    unique_cells, counts = np.unique(local_cells, return_counts=True)
    primary_cell = int(unique_cells[np.argmax(counts)])
    primary_mask = local_cells == primary_cell
    purity = float(np.mean(primary_mask))
    accepted_scores = local_scores[primary_mask]
    if len(accepted_scores) == 0:
        accepted_scores = local_scores
        purity = 1.0 / float(local_k)
    accepted = _scores_to_distances(accepted_scores)

    candidate_scores = np.asarray(search_result.candidate_scores, dtype=np.float64)
    candidate_cell_ids = np.asarray(search_result.candidate_cell_ids, dtype=np.int32)
    shell_mask = candidate_cell_ids != primary_cell
    shell_scores = candidate_scores[shell_mask]
    if len(shell_scores) > 0:
        shell_k = max(1, min(shell_k, len(shell_scores)))
        top_shell = np.argpartition(-shell_scores, shell_k - 1)[:shell_k]
        contrast = _scores_to_distances(shell_scores[top_shell])
    else:
        centroid_scores = np.asarray(search_result.centroid_scores, dtype=np.float64)
        other_mask = np.ones(len(centroid_scores), dtype=bool)
        if 0 <= primary_cell < len(other_mask):
            other_mask[primary_cell] = False
        if not np.any(other_mask):
            contrast = np.empty(0, dtype=np.float64)
        else:
            other_ids = np.nonzero(other_mask)[0]
            best_other_local = int(np.argmax(centroid_scores[other_mask]))
            other_cell = int(other_ids[best_other_local])
            centroid_distance = 1.0 - float(centroid_scores[other_cell])
            residual_distance = float(
                0.5 * (
                    dense_index.cell_residual_means[other_cell]
                    + dense_index.cell_residual_q90[other_cell]
                )
            )
            contrast = np.asarray(
                [min(2.0, centroid_distance + residual_distance)],
                dtype=np.float64,
            )

    return SearchDiagnostics(
        accepted,
        contrast,
        purity=purity,
        coverage=1.0,
    )


def separability_gate(
    diagnostics: SearchDiagnostics,
    *,
    min_gate: float = 0.02,
    max_gate: float = 0.98,
) -> float:
    """Map diagnostics to a silhouette-like gate in ``[min_gate, max_gate]``."""
    if len(diagnostics.accepted_distances) == 0:
        return min_gate

    a = max(float(diagnostics.cohesion), 0.0)
    b = max(float(diagnostics.separation), 0.0)
    denom = max(a, b, _EPSILON)
    score = max(0.0, (b - a) / denom)
    score *= diagnostics.reliability
    return float(np.clip(score, min_gate, max_gate))
