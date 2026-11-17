#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.consistency (the count constraint)."""

import numpy as np
import pytest

from bayesian_bm25.consistency import dilute_offset, expected_count, solve_offset
from bayesian_bm25.probability import sigmoid


class TestExpectedCount:
    def test_sum_of_posteriors(self):
        g = np.array([-2.0, 0.0, 3.0])
        expected = float(np.sum(sigmoid(g - 0.5)))
        assert expected_count(g, 0.5) == pytest.approx(expected)

    def test_weights_are_multiplicities(self):
        g = np.array([1.0, -1.0])
        weighted = expected_count(g, 0.0, weights=np.array([3.0, 2.0]))
        expanded = expected_count(np.array([1.0, 1.0, 1.0, -1.0, -1.0]), 0.0)
        assert weighted == pytest.approx(expanded)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            expected_count(np.array([]))

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            expected_count(np.array([0.0, np.inf]))


class TestSolveOffset:
    def test_separated_score_groups_do_not_cycle(self):
        scores = np.array([
            36.9267, 25.2375, 24.7536, 23.9235, 21.9126, 20.7009, 19.0397,
            19.0344, 17.4130, 16.6625, 16.4189, 16.2124, 15.9371, 14.5375,
            14.1433, 14.1107, 14.0226, 13.9091, 13.8640, 13.8102, 8.3381,
        ])
        weights = np.append(np.ones(20), 2136.0)
        offset = solve_offset(scores, 1.6635, weights=weights)
        assert expected_count(scores, offset, weights=weights) == pytest.approx(1.6635, rel=1e-9)

    @pytest.mark.parametrize("target", [0.01, 0.5, 1.0, 7.0, 250.0, 4000.0])
    def test_count_constraint_is_met(self, target):
        rng = np.random.default_rng(0)
        g = rng.normal(0.0, 3.0, size=5000)
        c = solve_offset(g, target)
        assert expected_count(g, c) == pytest.approx(target, rel=1e-8)

    def test_root_is_unique_and_count_is_decreasing(self):
        rng = np.random.default_rng(1)
        g = rng.normal(0.0, 2.0, size=500)
        c = solve_offset(g, 3.0)
        assert expected_count(g, c - 1e-3) > 3.0 > expected_count(g, c + 1e-3)

    def test_weighted_population_matches_expanded_population(self):
        g = np.array([4.0, 2.5, 0.0])
        weights = np.array([1.0, 3.0, 1000.0])
        expanded = np.concatenate([[4.0], np.full(3, 2.5), np.zeros(1000)])
        assert solve_offset(g, 2.0, weights=weights) == pytest.approx(
            solve_offset(expanded, 2.0), abs=1e-9
        )

    def test_larger_target_gives_smaller_offset(self):
        rng = np.random.default_rng(2)
        g = rng.normal(size=300)
        offsets = [solve_offset(g, r) for r in (0.5, 1.0, 5.0, 50.0)]
        assert all(a > b for a, b in zip(offsets, offsets[1:], strict=False))

    def test_identical_log_odds_have_closed_form(self):
        g = np.full(40, 1.7)
        c = solve_offset(g, 8.0)
        # 40 * sigmoid(1.7 - c) = 8  =>  c = 1.7 - logit(0.2)
        assert c == pytest.approx(1.7 - np.log(0.2 / 0.8))

    def test_shift_invariance(self):
        rng = np.random.default_rng(3)
        g = rng.normal(size=200)
        assert solve_offset(g + 10.0, 2.0) == pytest.approx(
            solve_offset(g, 2.0) + 10.0, abs=1e-9
        )

    def test_extreme_magnitudes_do_not_overflow(self):
        g = np.array([800.0, 500.0, -300.0, -900.0])
        c = solve_offset(g, 1.5)
        assert expected_count(g, c) == pytest.approx(1.5, rel=1e-8)

    def test_target_must_be_inside_population(self):
        g = np.zeros(10)
        with pytest.raises(ValueError, match="strictly between"):
            solve_offset(g, 10.0)
        with pytest.raises(ValueError, match="strictly between"):
            solve_offset(g, 0.0)

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            solve_offset(np.zeros(3), 1.0, weights=np.array([1.0, -1.0, 1.0]))

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError, match="does not match"):
            solve_offset(np.zeros(3), 1.0, weights=np.ones(2))


class TestDiluteOffset:
    def test_closed_form(self):
        g = np.array([0.3, -1.2, 2.0])
        expected = np.log(np.sum(np.exp(g))) - np.log(0.7)
        assert dilute_offset(g, 0.7) == pytest.approx(expected)

    def test_never_below_the_exact_root(self):
        # sigmoid(x) <= exp(x), so the dilute value is an upper bound.
        rng = np.random.default_rng(4)
        for _ in range(20):
            g = rng.normal(0.0, rng.uniform(0.5, 5.0), size=400)
            target = float(rng.uniform(0.1, 100.0))
            assert dilute_offset(g, target) >= solve_offset(g, target) - 1e-9

    def test_exact_in_the_dilute_regime(self):
        rng = np.random.default_rng(5)
        g = rng.normal(0.0, 1.0, size=100000)
        # One expected relevant document among 1e5: every posterior is tiny.
        assert dilute_offset(g, 1.0) == pytest.approx(solve_offset(g, 1.0), abs=1e-3)

    def test_overestimates_when_a_posterior_is_large(self):
        g = np.concatenate([[30.0], np.zeros(1000)])
        # The top document's posterior is close to one, so the dilute form
        # fails exactly where it matters.
        assert dilute_offset(g, 1.0) - solve_offset(g, 1.0) > 5.0

    def test_stable_for_large_log_odds(self):
        g = np.array([1000.0, 999.0])
        assert np.isfinite(dilute_offset(g, 1.0))

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="positive"):
            dilute_offset(np.zeros(3), 0.0)


class TestMarginalConsistency:
    """Lemma 2.8: the posteriors average to the prior iff the densities exist."""

    def test_densities_normalize_exactly_at_the_consistent_offset(self):
        rng = np.random.default_rng(6)
        scores = rng.gamma(2.0, 2.0, size=2000)
        alpha, prevalence = 0.8, 0.01
        n = len(scores)

        c = solve_offset(alpha * scores, prevalence * n)
        posterior = np.asarray(sigmoid(alpha * scores - c))

        # f0 = F (1 - P) / (1 - pi), f1 = F P / pi with F uniform on the sample.
        f0_mass = np.sum((1.0 - posterior) / (1.0 - prevalence)) / n
        f1_mass = np.sum(posterior / prevalence) / n
        assert f0_mass == pytest.approx(1.0, abs=1e-9)
        assert f1_mass == pytest.approx(1.0, abs=1e-9)

    def test_any_other_offset_breaks_normalization(self):
        rng = np.random.default_rng(7)
        scores = rng.gamma(2.0, 2.0, size=2000)
        alpha, prevalence = 0.8, 0.01
        n = len(scores)
        c = solve_offset(alpha * scores, prevalence * n) + 0.5
        posterior = np.asarray(sigmoid(alpha * scores - c))
        assert abs(np.sum(posterior / prevalence) / n - 1.0) > 0.1
