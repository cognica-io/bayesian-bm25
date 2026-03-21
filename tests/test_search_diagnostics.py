#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for benchmark-local search diagnostics."""

import numpy as np

from benchmarks.search_diagnostics import (
    SearchDiagnostics,
    build_exact_search_diagnostics,
    build_ivf_search_diagnostics,
    separability_gate,
)
from benchmarks.simple_ivf import SimpleIVF


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def test_exact_search_diagnostics_builds_shell_statistics():
    scores = np.array([0.95, 0.90, 0.50, 0.45], dtype=np.float64)
    diagnostics = build_exact_search_diagnostics(scores, local_k=2, shell_k=2)

    np.testing.assert_allclose(diagnostics.accepted_distances, [0.05, 0.10])
    np.testing.assert_allclose(diagnostics.contrast_distances, [0.50, 0.55])
    assert diagnostics.purity == 1.0
    assert diagnostics.coverage == 1.0
    assert separability_gate(diagnostics) > 0.80


def test_separability_gate_penalizes_unreliable_diagnostics():
    strong = SearchDiagnostics([0.10, 0.12], [0.50, 0.52], purity=1.0, coverage=1.0)
    weak = SearchDiagnostics([0.10, 0.12], [0.50, 0.52], purity=0.25, coverage=0.50)

    assert separability_gate(weak) < separability_gate(strong)


def test_ivf_search_diagnostics_use_primary_cell_and_contrast_shell():
    emb = _normalize_rows(
        np.array(
            [
                [1.0, 0.0],
                [0.98, 0.10],
                [0.96, -0.12],
                [0.0, 1.0],
                [0.08, 0.99],
                [-0.10, 0.97],
            ],
            dtype=np.float32,
        )
    )
    index = SimpleIVF.build(emb, n_cells=2, max_iterations=8, seed=42)
    query = _normalize_rows(np.array([[1.0, 0.10]], dtype=np.float32))[0]
    result = index.search(query, k=4, nprobe=2)

    diagnostics = build_ivf_search_diagnostics(
        result.scores,
        result.cell_ids,
        result,
        index,
        local_k=3,
        shell_k=2,
    )

    assert len(diagnostics.accepted_distances) >= 1
    assert len(diagnostics.contrast_distances) >= 1
    assert 0.0 < diagnostics.purity <= 1.0
    assert diagnostics.separation > diagnostics.cohesion
    assert separability_gate(diagnostics) > 0.02
