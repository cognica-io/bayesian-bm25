#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tail units, equivariance, and separation of scale and population."""

import numpy as np
import pytest

from bayesian_bm25 import calibrate_query, tail_scale


class TestTailScale:
    def test_mean_excess_uses_the_retrieval_threshold(self):
        # The best two exceed the third score by six and two units.
        assert tail_scale(np.array([1.0, 7.0, -100.0, 3.0]), k=3) == 4.0

    def test_order_and_unretrieved_scores_do_not_change_the_scale(self):
        scores = np.array([10.0, 5.0, 3.0, -1.0, -100.0])
        assert tail_scale(scores, k=3) == tail_scale(scores[::-1], k=3)
        assert tail_scale(scores, k=3) == tail_scale(scores[:3])

    def test_affine_equivariance(self):
        scores = np.array([-0.2, 0.0, 0.3, 0.7, 0.8])
        assert tail_scale(20.0 * scores + 3.0, k=4) == pytest.approx(
            20.0 * tail_scale(scores, k=4)
        )

    def test_large_depth_uses_the_available_list(self):
        scores = np.array([1.0, 2.0, 4.0])
        assert tail_scale(scores, k=1000) == tail_scale(scores)

    def test_score_location_does_not_invent_a_degenerate_tail(self):
        scores = np.array([1.0, 2.0, 3.0])
        assert tail_scale(scores + 1e12) == tail_scale(scores)

    def test_ties_have_no_inverse_tail_scale(self):
        for scores in (np.array([]), np.array([1.0]), np.ones(10)):
            assert tail_scale(scores) == 0.0
        assert tail_scale(np.array([5.0, 5.0, 4.0]), k=2) == 0.0

    def test_tail_scale_does_not_replace_the_calibration_population(self):
        population = np.concatenate([np.linspace(0.0, 0.5, 1000), [0.8, 0.9, 1.0]])
        scale = tail_scale(population, k=3)
        model = calibrate_query(population, scale=scale, expected_relevant=2.0, ceiling=0.6)
        assert np.sum(model.probability(population)) == pytest.approx(2.0)
        assert model.population_size == len(population)
        assert model.slope == pytest.approx(1.0 / scale)

    @pytest.mark.parametrize("depth", [0, 1, -1, 2.5, True])
    def test_rejects_invalid_depths(self, depth):
        with pytest.raises(ValueError, match="k must be an integer"):
            tail_scale(np.arange(4.0), k=depth)

    @pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_scores(self, score):
        with pytest.raises(ValueError, match="scores must be finite"):
            tail_scale(np.array([0.5, score]))
