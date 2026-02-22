#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.fusion module."""

import numpy as np
import pytest

from bayesian_bm25.fusion import (
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_or,
)


class TestCosineToProbability:
    def test_max_similarity(self):
        """score=1.0 -> ~1.0 (clamped to 1-epsilon)."""
        result = cosine_to_probability(1.0)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_min_similarity(self):
        """score=-1.0 -> ~0.0 (clamped to epsilon)."""
        result = cosine_to_probability(-1.0)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_zero_similarity(self):
        """score=0.0 -> 0.5."""
        result = cosine_to_probability(0.0)
        assert result == pytest.approx(0.5)

    def test_bounds(self):
        """Output is always in (0, 1) with clamping."""
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = cosine_to_probability(score)
            assert 0 < result < 1, f"Out of bounds for score={score}: {result}"

    def test_array_input(self):
        scores = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = cosine_to_probability(scores)
        assert result.shape == (5,)
        np.testing.assert_allclose(
            result, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-5
        )

    def test_monotonicity(self):
        scores = np.linspace(-1.0, 1.0, 20)
        result = cosine_to_probability(scores)
        assert np.all(np.diff(result) > 0)


class TestProbAnd:
    def test_two_values(self):
        result = prob_and(np.array([0.8, 0.9]))
        assert result == pytest.approx(0.72)

    def test_all_ones(self):
        result = prob_and(np.array([1.0, 1.0, 1.0]))
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_contains_zero(self):
        # With clamping, zero becomes epsilon
        result = prob_and(np.array([0.5, 0.0]))
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_single_value(self):
        result = prob_and(np.array([0.7]))
        assert result == pytest.approx(0.7)

    def test_shrinkage(self):
        """Product rule causes shrinkage: AND of 0.9 and 0.9 is only 0.81."""
        result = prob_and(np.array([0.9, 0.9]))
        assert result == pytest.approx(0.81)
        assert result < 0.9  # Shrinkage

    def test_batched(self):
        probs = np.array([[0.8, 0.9], [0.5, 0.5]])
        result = prob_and(probs)
        np.testing.assert_allclose(result, [0.72, 0.25])


class TestProbOr:
    def test_two_values(self):
        result = prob_or(np.array([0.8, 0.9]))
        assert result == pytest.approx(0.98)

    def test_all_zeros(self):
        result = prob_or(np.array([0.0, 0.0]))
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_contains_one(self):
        result = prob_or(np.array([0.5, 1.0]))
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_single_value(self):
        result = prob_or(np.array([0.7]))
        assert result == pytest.approx(0.7)

    def test_complement(self):
        """P(A or B) = 1 - (1-A)(1-B)."""
        a, b = 0.6, 0.7
        result = prob_or(np.array([a, b]))
        expected = 1.0 - (1.0 - a) * (1.0 - b)
        assert result == pytest.approx(expected)

    def test_batched(self):
        probs = np.array([[0.8, 0.9], [0.5, 0.5]])
        result = prob_or(probs)
        np.testing.assert_allclose(result, [0.98, 0.75])


class TestLogOddsConjunction:
    """Verify properties from Paper 2, Section 4.5."""

    def test_agreement_amplification(self):
        """Two agreeing high probabilities should amplify above the input.

        With alpha=0.5: l_bar=logit(0.9)=2.197,
        l_adjusted=2.197*sqrt(2)=3.107, sigmoid(3.107)=0.957.
        """
        result = log_odds_conjunction(np.array([0.9, 0.9]))
        assert result == pytest.approx(0.957, abs=0.01)
        assert result > 0.9  # Amplification, not shrinkage

    def test_moderate_agreement(self):
        """(0.7, 0.7) -> ~0.77.

        l_bar=logit(0.7)=0.847, l_adjusted=0.847*sqrt(2)=1.198,
        sigmoid(1.198)=0.768.
        """
        result = log_odds_conjunction(np.array([0.7, 0.7]))
        assert result == pytest.approx(0.77, abs=0.02)
        assert result > 0.7  # Still amplified

    def test_disagreement_moderation(self):
        """(0.7, 0.3) -> exactly 0.5 (symmetric logits cancel).

        logit(0.7)=0.847, logit(0.3)=-0.847, l_bar=0, sigmoid(0)=0.5.
        """
        result = log_odds_conjunction(np.array([0.7, 0.3]))
        assert result == pytest.approx(0.5, abs=0.01)
        # Symmetric logits cancel to exact uncertainty
        assert 0.49 < result < 0.51

    def test_agreement_low(self):
        """(0.3, 0.3) -> ~0.23, moderated rather than shrunk to 0.09.

        l_bar=logit(0.3)=-0.847, l_adjusted=-0.847*sqrt(2)=-1.198,
        sigmoid(-1.198)=0.232.
        """
        result = log_odds_conjunction(np.array([0.3, 0.3]))
        assert result == pytest.approx(0.23, abs=0.02)
        # Should be higher than naive AND (0.09)
        assert result > prob_and(np.array([0.3, 0.3]))

    def test_irrelevance_preservation(self):
        """(0.5, 0.5) should stay exactly 0.5.

        logit(0.5)=0, l_bar=0, l_adjusted=0, sigmoid(0)=0.5.
        """
        result = log_odds_conjunction(np.array([0.5, 0.5]))
        assert result == pytest.approx(0.5, abs=0.01)

    def test_single_signal_identity(self):
        """With n=1 and alpha=0, a single signal should pass through."""
        result = log_odds_conjunction(np.array([0.8]), alpha=0.0)
        assert result == pytest.approx(0.8, abs=0.01)

    def test_bounds(self):
        """Output should always be in (0, 1)."""
        test_cases = [
            np.array([0.01, 0.01]),
            np.array([0.99, 0.99]),
            np.array([0.01, 0.99]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ]
        for probs in test_cases:
            result = log_odds_conjunction(probs)
            assert 0 < result < 1, f"Out of bounds for {probs}: {result}"

    def test_more_signals_amplify(self):
        """Adding more agreeing signals should increase confidence."""
        two = log_odds_conjunction(np.array([0.8, 0.8]))
        three = log_odds_conjunction(np.array([0.8, 0.8, 0.8]))
        assert three > two

    def test_alpha_effect(self):
        """Higher alpha gives stronger agreement bonus."""
        probs = np.array([0.8, 0.8])
        low_alpha = log_odds_conjunction(probs, alpha=0.1)
        high_alpha = log_odds_conjunction(probs, alpha=1.0)
        assert high_alpha > low_alpha

    def test_batched(self):
        probs = np.array([[0.9, 0.9], [0.3, 0.3]])
        result = log_odds_conjunction(probs)
        assert result.shape == (2,)
        assert result[0] > 0.9  # High agreement amplified
        assert result[1] < 0.5  # Low agreement stays low


class TestWeightedLogOddsConjunction:
    """Verify weighted Log-OP from Paper 2, Theorem 8.3 / Remark 8.4."""

    def test_uniform_weights_match_unweighted_alpha_zero(self):
        """Uniform weights should produce same result as unweighted alpha=0.

        With alpha=0 unweighted: l_adjusted = l_bar * 1 = mean(logit(P_i)).
        With uniform weights w_i=1/n: sum(w_i * logit(P_i)) = mean(logit(P_i)).
        """
        probs = np.array([0.7, 0.8])
        uniform_w = np.array([0.5, 0.5])
        weighted = log_odds_conjunction(probs, weights=uniform_w)
        unweighted_alpha0 = log_odds_conjunction(probs, alpha=0.0)
        assert weighted == pytest.approx(unweighted_alpha0, abs=1e-10)

    def test_higher_weight_on_high_probability(self):
        """Weighting the high-probability signal more should increase result."""
        probs = np.array([0.9, 0.3])
        w_high_first = np.array([0.8, 0.2])
        w_high_second = np.array([0.2, 0.8])
        result_high_first = log_odds_conjunction(probs, weights=w_high_first)
        result_high_second = log_odds_conjunction(probs, weights=w_high_second)
        assert result_high_first > result_high_second

    def test_higher_weight_on_low_probability(self):
        """Weighting the low-probability signal more should decrease result."""
        probs = np.array([0.9, 0.2])
        w_equal = np.array([0.5, 0.5])
        w_low_heavy = np.array([0.2, 0.8])
        result_equal = log_odds_conjunction(probs, weights=w_equal)
        result_low = log_odds_conjunction(probs, weights=w_low_heavy)
        assert result_low < result_equal

    def test_weights_must_sum_to_one(self):
        probs = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="weights must sum to 1"):
            log_odds_conjunction(probs, weights=np.array([0.3, 0.3]))

    def test_weights_must_be_nonnegative(self):
        probs = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="weights must be non-negative"):
            log_odds_conjunction(probs, weights=np.array([-0.5, 1.5]))

    def test_batched_with_weights(self):
        probs = np.array([[0.9, 0.1], [0.8, 0.8]])
        weights = np.array([0.7, 0.3])
        result = log_odds_conjunction(probs, weights=weights)
        assert result.shape == (2,)
        # First batch: high-prob signal weighted heavily -> > 0.5
        assert result[0] > 0.5
        # Second batch: both signals agree at 0.8 -> > 0.5
        assert result[1] > 0.5

    def test_single_signal_full_weight(self):
        """A single signal with weight=1 should pass through."""
        result = log_odds_conjunction(np.array([0.8]), weights=np.array([1.0]))
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_three_signals(self):
        """Three signals with non-uniform weights."""
        probs = np.array([0.9, 0.9, 0.1])
        w = np.array([0.4, 0.4, 0.2])
        result = log_odds_conjunction(probs, weights=w)
        # Two high signals dominate -> result > 0.5
        assert result > 0.5

    def test_weighted_with_explicit_alpha(self):
        """When alpha is explicitly set, n^alpha scaling is applied in weighted mode."""
        probs = np.array([0.8, 0.8])
        w = np.array([0.5, 0.5])
        # alpha=0.0: no scaling (n^0 = 1)
        result_alpha0 = log_odds_conjunction(probs, alpha=0.0, weights=w)
        # alpha=0.5: n^0.5 = sqrt(2) scaling
        result_alpha05 = log_odds_conjunction(probs, alpha=0.5, weights=w)
        # Higher alpha -> stronger amplification for agreeing signals
        assert result_alpha05 > result_alpha0

    def test_weighted_alpha_none_backward_compatible(self):
        """alpha=None in weighted mode behaves like old code (alpha ignored)."""
        probs = np.array([0.7, 0.8, 0.6])
        w = np.array([0.4, 0.4, 0.2])
        # Old behavior: sigma(sum(w_i * logit(P_i))) with no n^alpha
        from bayesian_bm25.probability import logit as _logit, sigmoid as _sigmoid
        expected = _sigmoid(np.sum(w * _logit(probs)))
        result = log_odds_conjunction(probs, weights=w)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_weighted_alpha_zero_matches_current(self):
        """Explicit alpha=0.0 in weighted mode matches no-scaling behavior."""
        probs = np.array([0.7, 0.8, 0.6])
        w = np.array([0.4, 0.4, 0.2])
        result_none = log_odds_conjunction(probs, weights=w)
        result_zero = log_odds_conjunction(probs, alpha=0.0, weights=w)
        assert result_zero == pytest.approx(result_none, abs=1e-10)
