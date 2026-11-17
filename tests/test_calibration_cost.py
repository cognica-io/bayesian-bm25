#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""The measured pipeline uses the same count population as the scorer."""

import numpy as np

from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.additional.calibration_cost import measure_scores


def test_cost_pipeline_retains_exact_count_and_separates_density_cost():
    rng = np.random.default_rng(42)
    corpus = [["a", "b"] if i % 3 else ["c", "b", "b"] for i in range(200)]
    scorer = BayesianBM25Scorer(method="lucene")
    scorer.index(corpus, show_progress=False)
    embeddings = rng.normal(size=(200, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = rng.normal(size=16).astype(np.float32)
    query /= np.linalg.norm(query)
    times, accuracy = measure_scores(
        scorer, ["a"], embeddings, query, np.array([0, 2]), retrieve_k=30, rng=rng,
    )
    assert accuracy["relative_count_error"] < 1e-9
    assert 0 < accuracy["head_count"] <= 10
    assert times["scoring_to_capped_output_ms"] > times["lexical_scale_and_count_ms"]
    assert times["scoring_to_density_ratio_output_ms"] > times["gaussian_ratio_kde_and_fusion_ms"]
    assert all(value >= 0.0 for value in times.values())
