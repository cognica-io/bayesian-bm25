#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for benchmarks.simple_ivf."""

import numpy as np
import pytest

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


class TestQueryBackground:
    """Index-derived moments of a query's distance over the corpus."""

    @staticmethod
    def _corpus(seed: int = 0, n_docs: int = 4000, dim: int = 48, n_topics: int = 30):
        rng = np.random.default_rng(seed)
        topics = _normalize_rows(rng.normal(size=(n_topics, dim)))
        assignment = rng.integers(0, n_topics, size=n_docs)
        corpus = _normalize_rows(
            topics[assignment] + 0.6 * rng.normal(size=(n_docs, dim))
        )
        queries = _normalize_rows(
            topics[rng.integers(0, n_topics, size=20)] + 0.6 * rng.normal(size=(20, dim))
        )
        return corpus, queries

    def test_recovers_the_moments_of_the_query_distances(self):
        corpus, queries = self._corpus()
        index = SimpleIVF.build(corpus, n_cells=40, max_iterations=25, seed=1)
        for query in queries:
            result = index.search(query, k=200, nprobe=6)
            mean, std = index.query_background(result)
            distances = 1.0 - corpus @ query
            assert abs(mean - float(np.mean(distances))) < 0.01
            assert std == pytest.approx(float(np.std(distances)), rel=0.25)

    def test_build_time_residuals_are_a_different_variable(self):
        # Distances between a document and its own centroid are much
        # smaller than distances between a query and a document.
        corpus, queries = self._corpus(seed=2)
        index = SimpleIVF.build(corpus, n_cells=40, max_iterations=25, seed=1)
        residual_mean = float(np.mean(index.background_distances))
        query_mean = float(np.mean(1.0 - corpus @ queries[0]))
        assert residual_mean < 0.7 * query_mean

    def test_probing_every_cell_is_exact(self):
        corpus, queries = self._corpus(seed=3, n_docs=500)
        index = SimpleIVF.build(corpus, n_cells=10, max_iterations=10, seed=1)
        result = index.search(queries[0], k=50, nprobe=index.n_cells)
        mean, std = index.query_background(result)
        distances = 1.0 - corpus @ queries[0]
        assert mean == pytest.approx(float(np.mean(distances)), abs=1e-6)
        assert std == pytest.approx(float(np.std(distances)), abs=1e-6)

    def test_cell_sums_are_stored(self):
        corpus, _ = self._corpus(seed=4, n_docs=300)
        index = SimpleIVF.build(corpus, n_cells=8, max_iterations=10, seed=1)
        assert index.cell_cosine_sums.shape == (index.n_cells,)
        assert index.cell_cosine_square_sums.shape == (index.n_cells,)
        assert np.all(index.cell_cosine_square_sums <= index.cell_cosine_sums + 1e-9)

    def test_all_probed_query_copies_are_absent_from_both_moments(self):
        corpus, queries = self._corpus(seed=5, n_docs=300)
        index = SimpleIVF.build(corpus, n_cells=8, max_iterations=10, seed=1)
        result = index.search(queries[0], k=50, nprobe=index.n_cells)
        excluded = np.array([0, 8, 19, 22])
        mean, std = index.query_background(result, excluded_indices=excluded)
        distances = np.delete(1.0 - corpus @ queries[0], excluded)
        assert mean == pytest.approx(float(np.mean(distances)), abs=1e-6)
        assert std == pytest.approx(float(np.std(distances)), abs=1e-6)

    def test_unprobed_exclusions_require_the_query_vector(self):
        corpus, queries = self._corpus(seed=6, n_docs=300)
        index = SimpleIVF.build(corpus, n_cells=8, max_iterations=10, seed=1)
        result = index.search(queries[0], k=20, nprobe=1)
        unprobed = np.setdiff1d(np.arange(300), result.candidate_indices)[:4]
        with pytest.raises(ValueError, match="query_vector"):
            index.query_background(result, excluded_indices=unprobed)
        actual, _ = index.query_background(
            result, excluded_indices=unprobed, query_vector=queries[0],
        )
        exact = float(np.mean(np.delete(1.0 - corpus @ queries[0], unprobed)))
        assert actual == pytest.approx(exact, abs=0.01)
