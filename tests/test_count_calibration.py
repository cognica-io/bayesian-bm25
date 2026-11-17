#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Count conservation and numerical invariants of a ceiling posterior."""

import math

import numpy as np
import pytest

from bayesian_bm25 import calibrate_count
from bayesian_bm25.consistency import solve_offset
from bayesian_bm25.count_calibration import ceiling_log_odds
from bayesian_bm25.probability import sigmoid


class TestCountCalibration:
    @pytest.mark.parametrize("ceiling", [0.1, 0.4, 0.9, 1.0])
    def test_preserves_the_weighted_relevance_count(self, ceiling):
        logits = np.array([-20.0, -2.0, 1.0, 3.0, 15.0])
        weights = np.array([1000.0, 40.0, 10.0, 1.0, 1.0])
        model = calibrate_count(logits, 2.5, ceiling=ceiling, weights=weights)
        probabilities = model.probability(logits)
        assert np.dot(weights, probabilities) == pytest.approx(2.5, rel=1e-9)
        assert np.all(probabilities <= ceiling)
        assert model.population_size == np.sum(weights)

    def test_lowering_the_ceiling_moves_mass_from_head_to_tail(self):
        logits = np.array([12.0, 10.0, 5.0, 1.0, 0.0, -2.0])
        base = calibrate_count(logits, 1.2)
        capped = calibrate_count(logits, 1.2, ceiling=0.4)
        before, after = base.probability(logits), capped.probability(logits)
        assert after[0] < before[0]
        assert after[-1] > before[-1]
        assert after.sum() == pytest.approx(before.sum())
        assert capped.offset < base.offset

    def test_ceiling_one_preserves_the_original_model(self):
        logits = np.linspace(-100.0, 100.0, 301)
        model = calibrate_count(logits, 2.0)
        assert model.offset == solve_offset(logits, 2.0)
        np.testing.assert_array_equal(model.log_odds(logits), model.ranking_score(logits))

    def test_ranking_scores_remain_ordered_at_the_ceiling(self):
        model = calibrate_count(np.arange(20.0), 1.2, ceiling=0.4)
        strong = np.array([100.0, 200.0, 1000.0])
        assert np.all(np.diff(model.ranking_score(strong)) > 0.0)
        assert np.unique(model.log_odds(strong)).size == 1
        assert np.all(model.probability(strong) == 0.4)

    def test_log_odds_match_the_ceiling_model_including_extreme_tails(self):
        x = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        for ceiling in (0.01, 0.4, 0.99, 1.0):
            result = ceiling_log_odds(x, ceiling)
            assert np.all(np.isfinite(result))
            np.testing.assert_allclose(sigmoid(result), ceiling * sigmoid(x), atol=1e-15)
        assert ceiling_log_odds(-1000.0, 0.4) == pytest.approx(-1000.0 + math.log(0.4))

    def test_prior_free_evidence_matches_the_actual_posterior(self):
        logits = np.arange(20.0)
        model = calibrate_count(logits, 2.0, ceiling=0.4)
        np.testing.assert_allclose(
            model.evidence(logits) + model.prior_log_odds, model.log_odds(logits),
        )
        assert model.prevalence == pytest.approx(0.1)

    def test_bounds_dominate_latent_and_posterior_scores(self):
        logits = np.linspace(-10.0, 20.0, 100)
        model = calibrate_count(logits, 3.0, ceiling=0.6)
        assert np.all(model.ranking_score(logits) <= model.upper_bound_ranking_score(20.0))
        assert np.all(model.log_odds(logits) <= model.upper_bound_log_odds(20.0))

    def test_dilute_limit_does_not_depend_on_the_ceiling(self):
        logits = np.linspace(-1.0, 1.0, 10000)
        base = calibrate_count(logits, 0.01)
        capped = calibrate_count(logits, 0.01, ceiling=0.2)
        np.testing.assert_allclose(base.probability(logits), capped.probability(logits), rtol=1e-5)

    def test_zero_weight_outlier_does_not_shift_or_underflow_the_solution(self):
        logits = np.array([1e6, -2.0, 0.0, 2.0])
        weights = np.array([0.0, 1.0, 2.0, 1.0])
        model = calibrate_count(logits, 1.0, ceiling=0.4, weights=weights)
        assert np.dot(weights, model.probability(logits)) == pytest.approx(1.0)

    def test_scalar_predictions_return_floats(self):
        model = calibrate_count(np.arange(5.0), 1.0, ceiling=0.4)
        for method in (model.ranking_score, model.log_odds, model.probability, model.evidence):
            assert isinstance(method(2.0), float)

    @pytest.mark.parametrize("ceiling", [0.0, -0.1, 1.1, np.nan, np.inf])
    def test_rejects_invalid_ceilings(self, ceiling):
        with pytest.raises(ValueError, match="ceiling"):
            calibrate_count(np.arange(5.0), 1.0, ceiling=ceiling)

    @pytest.mark.parametrize("count", [0.0, -1.0, 2.0, 3.0, np.nan, np.inf])
    def test_rejects_infeasible_counts(self, count):
        with pytest.raises(ValueError, match="expected_relevant"):
            calibrate_count(np.arange(5.0), count, ceiling=0.4)
