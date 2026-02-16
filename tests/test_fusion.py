#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.fusion module."""

import numpy as np
import pytest

from bayesian_bm25.fusion import log_odds_conjunction, prob_and, prob_or


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
