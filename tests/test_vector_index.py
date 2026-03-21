#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for benchmarks.simple_ivf."""

import numpy as np

from benchmarks.simple_ivf import SimpleIVF


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class TestSimpleIVF:
    def test_build_exposes_background_stats(self):
        emb = _normalize_rows(
            np.array(
                [
                    [1.0, 0.0],
                    [0.98, 0.15],
                    [0.95, -0.10],
                    [0.0, 1.0],
                    [0.10, 0.97],
                    [-0.08, 0.99],
                ],
                dtype=np.float32,
            )
        )
        index = SimpleIVF.build(emb, n_cells=2, max_iterations=8, seed=42)
        assert index.n_docs == 6
        assert index.n_cells == 2
        assert len(index.background_distances) == 6
        assert np.all(index.background_distances >= 0.0)
        assert index.avg_population > 0.0
        assert len(index.cell_residual_means) == 2
        assert len(index.cell_residual_q90) == 2

    def test_search_returns_nearest_cluster(self):
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
        query = np.array([1.0, 0.02], dtype=np.float32)
        result = index.search(query, k=3, nprobe=1)

        assert len(result.indices) == 3
        assert len(result.candidate_indices) >= 3
        assert set(result.indices.tolist()).issubset({0, 1, 2})
        assert np.all(np.diff(result.scores) <= 1e-12)
        assert len(result.centroid_scores) == index.n_cells

    def test_score_documents_matches_exact_dot_product(self):
        emb = _normalize_rows(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        index = SimpleIVF.build(emb, n_cells=2, max_iterations=4, seed=7)
        query = _normalize_rows(np.array([[1.0, 0.5]], dtype=np.float32))[0]
        doc_indices = np.array([0, 2], dtype=np.int32)

        expected = np.asarray(emb[doc_indices] @ query, dtype=np.float64)
        actual = index.score_documents(query, doc_indices)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
