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

        With alpha=0.5: geometric_mean=0.9, logit(0.9)=2.197,
        bonus=0.5*ln(2)=0.347, sigmoid(2.544)=0.927.
        """
        result = log_odds_conjunction(np.array([0.9, 0.9]))
        assert result == pytest.approx(0.927, abs=0.01)
        assert result > 0.9  # Amplification, not shrinkage

    def test_moderate_agreement(self):
        """(0.7, 0.7) -> ~0.77."""
        result = log_odds_conjunction(np.array([0.7, 0.7]))
        assert result == pytest.approx(0.77, abs=0.03)
        assert result > 0.7  # Still amplified

    def test_disagreement_moderation(self):
        """(0.7, 0.3) -> ~0.54, near 0.5 (uncertain)."""
        result = log_odds_conjunction(np.array([0.7, 0.3]))
        assert result == pytest.approx(0.54, abs=0.05)
        # Should be close to 0.5 (maximum uncertainty)
        assert 0.45 < result < 0.65

    def test_agreement_low(self):
        """(0.3, 0.3) -> ~0.38, moderated rather than shrunk to 0.09."""
        result = log_odds_conjunction(np.array([0.3, 0.3]))
        assert result == pytest.approx(0.38, abs=0.05)
        # Should be higher than naive AND (0.09)
        assert result > prob_and(np.array([0.3, 0.3]))

    def test_irrelevance_preservation(self):
        """(0.5, 0.5) should stay near 0.5."""
        result = log_odds_conjunction(np.array([0.5, 0.5]))
        assert result == pytest.approx(0.5, abs=0.1)

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
