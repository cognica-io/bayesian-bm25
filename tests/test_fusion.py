#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.fusion module."""

import pickle

import numpy as np
import pytest

from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    LearnableLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    balanced_log_odds_fusion,
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_not,
    prob_or,
)
from bayesian_bm25.probability import logit, sigmoid


class TestVersion:
    def test_version_accessible(self):
        """__version__ is accessible and matches pyproject.toml."""
        import bayesian_bm25

        assert hasattr(bayesian_bm25, "__version__")
        assert isinstance(bayesian_bm25.__version__, str)
        # Minimal semver sanity: at least "X.Y.Z"
        parts = bayesian_bm25.__version__.split(".")
        assert len(parts) >= 2, f"Unexpected version format: {bayesian_bm25.__version__}"

    def test_version_in_all(self):
        """__version__ is exported in __all__."""
        import bayesian_bm25

        assert "__version__" in bayesian_bm25.__all__


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

    def test_boundary_negative_one(self):
        """cos=-1 maps to approximately 0 (epsilon)."""
        result = cosine_to_probability(-1.0)
        assert result > 0
        assert result < 0.01

    def test_boundary_zero(self):
        """cos=0 maps to exactly 0.5."""
        result = cosine_to_probability(0.0)
        assert result == pytest.approx(0.5)

    def test_boundary_positive_one(self):
        """cos=1 maps to approximately 1 (1-epsilon)."""
        result = cosine_to_probability(1.0)
        assert result > 0.99
        assert result < 1.0

    def test_strict_monotonicity(self):
        """Strictly increasing over [-1, 1] with fine granularity."""
        scores = np.linspace(-1.0, 1.0, 1000)
        result = cosine_to_probability(scores)
        diffs = np.diff(result)
        assert np.all(diffs > 0), "cosine_to_probability must be strictly increasing"


class TestProbNot:
    def test_complement(self):
        """NOT 0.8 = 0.2."""
        result = prob_not(0.8)
        assert result == pytest.approx(0.2)

    def test_half(self):
        """NOT 0.5 = 0.5 (uncertainty is self-complementary)."""
        result = prob_not(0.5)
        assert result == pytest.approx(0.5)

    def test_near_zero(self):
        """NOT of near-zero -> near-one."""
        result = prob_not(0.01)
        assert result == pytest.approx(0.99)

    def test_near_one(self):
        """NOT of near-one -> near-zero."""
        result = prob_not(0.99)
        assert result == pytest.approx(0.01)

    def test_involution(self):
        """NOT(NOT(p)) = p (double negation)."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert prob_not(prob_not(p)) == pytest.approx(p, abs=1e-9)

    def test_bounds(self):
        """Output is always in (0, 1) with clamping."""
        for p in [0.0, 0.5, 1.0]:
            result = prob_not(p)
            assert 0 < result < 1, f"Out of bounds for p={p}: {result}"

    def test_array_input(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = prob_not(probs)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, [0.9, 0.7, 0.5, 0.3, 0.1])

    def test_log_odds_negation(self):
        """logit(NOT p) = -logit(p)."""
        from bayesian_bm25.probability import logit

        for p in [0.2, 0.4, 0.6, 0.8]:
            assert logit(prob_not(p)) == pytest.approx(-logit(p), abs=1e-9)

    def test_de_morgan_and(self):
        """De Morgan: NOT(A AND B) = OR(NOT A, NOT B)."""
        a, b = 0.7, 0.8
        lhs = prob_not(prob_and(np.array([a, b])))
        rhs = prob_or(np.array([prob_not(a), prob_not(b)]))
        assert lhs == pytest.approx(rhs, abs=1e-9)

    def test_de_morgan_or(self):
        """De Morgan: NOT(A OR B) = AND(NOT A, NOT B)."""
        a, b = 0.7, 0.8
        lhs = prob_not(prob_or(np.array([a, b])))
        rhs = prob_and(np.array([prob_not(a), prob_not(b)]))
        assert lhs == pytest.approx(rhs, abs=1e-9)


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
        from bayesian_bm25.probability import logit as _logit
        from bayesian_bm25.probability import sigmoid as _sigmoid
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


class TestLearnableLogOddsWeights:
    """Tests for the LearnableLogOddsWeights class (Remark 5.3.2)."""

    def test_initial_uniform_weights(self):
        """Initial weights should be uniform 1/n (Naive Bayes init)."""
        for n in [1, 2, 3, 5, 10]:
            learner = LearnableLogOddsWeights(n_signals=n)
            expected = np.full(n, 1.0 / n)
            np.testing.assert_allclose(learner.weights, expected, atol=1e-15)

    def test_n_signals_less_than_one_raises(self):
        """n_signals < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_signals must be >= 1"):
            LearnableLogOddsWeights(n_signals=0)
        with pytest.raises(ValueError, match="n_signals must be >= 1"):
            LearnableLogOddsWeights(n_signals=-1)

    def test_call_matches_log_odds_conjunction_at_init(self):
        """At uniform init, __call__ matches log_odds_conjunction with uniform weights."""
        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
        probs = np.array([0.7, 0.8, 0.6])
        result = learner(probs)
        expected = log_odds_conjunction(
            probs, alpha=0.0, weights=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        )
        assert result == pytest.approx(expected, abs=1e-10)

    def test_call_batched(self):
        """__call__ handles batched inputs."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        probs = np.array([[0.8, 0.9], [0.3, 0.7]])
        result = learner(probs)
        assert result.shape == (2,)
        # Each row should match independent call
        for i in range(2):
            expected = learner(probs[i])
            assert result[i] == pytest.approx(expected, abs=1e-10)

    def test_call_use_averaged(self):
        """use_averaged=True uses Polyak-averaged weights."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        probs = np.array([0.7, 0.8])
        # At init, averaged weights = raw weights = uniform
        result_raw = learner(probs, use_averaged=False)
        result_avg = learner(probs, use_averaged=True)
        assert result_raw == pytest.approx(result_avg, abs=1e-10)

    def test_weights_simplex(self):
        """Weights always sum to 1 and are non-negative."""
        learner = LearnableLogOddsWeights(n_signals=4, alpha=0.0)
        # Perturb logits manually
        learner._logits = np.array([1.0, -2.0, 0.5, 3.0])
        w = learner.weights
        assert np.all(w >= 0)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-10)

    def test_fit_learns_reliable_signal(self):
        """fit() should learn to weight a reliable signal higher than a noisy one."""
        rng = np.random.RandomState(42)
        m = 500
        labels = rng.randint(0, 2, size=m).astype(np.float64)

        # Signal 0: reliable (high prob for label=1, low for label=0)
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        # Signal 1: noisy (random around 0.5)
        signal_1 = rng.uniform(0.3, 0.7, size=m)

        probs = np.column_stack([signal_0, signal_1])

        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        learner.fit(probs, labels, learning_rate=0.1, max_iterations=2000)

        # Reliable signal should get higher weight
        assert learner.weights[0] > learner.weights[1]
        assert learner.weights[0] > 0.6

    def test_fit_dimension_mismatch_raises(self):
        """fit() raises ValueError on dimension mismatch."""
        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
        probs = np.array([[0.5, 0.5]])  # 2 signals, not 3
        labels = np.array([1.0])
        with pytest.raises(ValueError, match="n_signals"):
            learner.fit(probs, labels)

    def test_fit_resets_online_state(self):
        """fit() resets online state (n_updates, EMA, averaged weights)."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        # Simulate some online updates first
        learner._n_updates = 10
        learner._grad_logits_ema = np.array([0.5, -0.3])

        probs = np.array([[0.8, 0.2], [0.7, 0.3]])
        labels = np.array([1.0, 0.0])
        learner.fit(probs, labels)

        assert learner._n_updates == 0
        np.testing.assert_allclose(learner._grad_logits_ema, [0.0, 0.0])
        np.testing.assert_allclose(learner.averaged_weights, learner.weights, atol=1e-10)

    def test_update_moves_toward_informative_signal(self):
        """update() should move weights from uniform toward the informative signal."""
        rng = np.random.RandomState(123)
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)

        for _ in range(200):
            label = float(rng.randint(0, 2))
            # Signal 0: informative
            p0 = 0.9 if label == 1 else 0.1
            # Signal 1: noise
            p1 = rng.uniform(0.3, 0.7)
            learner.update(np.array([p0, p1]), label, learning_rate=0.05)

        # Signal 0 should have higher weight
        assert learner.weights[0] > learner.weights[1]

    def test_update_dimension_mismatch_raises(self):
        """update() raises ValueError on dimension mismatch."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        with pytest.raises(ValueError, match="n_signals"):
            learner.update(np.array([0.5, 0.5, 0.5]), 1.0)

    def test_update_accepts_mini_batches(self):
        """update() accepts mini-batch inputs."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1.0, 0.0])
        # Should not raise
        learner.update(probs, labels)
        assert learner._n_updates == 1

    def test_averaged_weights_smoother_than_raw(self):
        """Averaged weights should be smoother than raw weights over noisy updates."""
        rng = np.random.RandomState(99)
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)

        raw_history = []
        avg_history = []

        for _ in range(100):
            label = float(rng.randint(0, 2))
            p0 = 0.8 if label == 1 else 0.2
            p1 = rng.uniform(0.2, 0.8)
            learner.update(np.array([p0, p1]), label, learning_rate=0.1)
            raw_history.append(learner.weights[0])
            avg_history.append(learner.averaged_weights[0])

        # Variance of averaged weights should be lower
        raw_var = np.var(raw_history[-50:])
        avg_var = np.var(avg_history[-50:])
        assert avg_var < raw_var

    def test_properties(self):
        """Properties return correct values."""
        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.5)
        assert learner.n_signals == 3
        assert learner.alpha == 0.5
        assert learner.weights.shape == (3,)
        assert learner.averaged_weights.shape == (3,)

    def test_averaged_weights_returns_copy(self):
        """averaged_weights should return a copy, not a reference."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        w1 = learner.averaged_weights
        w2 = learner.averaged_weights
        assert w1 is not w2
        # Mutating the returned array should not affect internal state
        w1[0] = 999.0
        assert learner.averaged_weights[0] != 999.0

    def test_softmax_numerical_stability(self):
        """Softmax should handle large logit differences without overflow."""
        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
        learner._logits = np.array([1000.0, 0.0, -1000.0])
        w = learner.weights
        assert np.all(np.isfinite(w))
        assert np.sum(w) == pytest.approx(1.0, abs=1e-10)
        assert w[0] == pytest.approx(1.0, abs=1e-10)  # Dominant logit
        assert w[2] == pytest.approx(0.0, abs=1e-10)  # Negligible logit

    def test_numerical_gradient(self):
        """Analytical gradient matches finite-difference approximation."""
        from bayesian_bm25.probability import _clamp_probability
        from bayesian_bm25.probability import logit as _logit
        from bayesian_bm25.probability import sigmoid as _sigmoid

        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
        learner._logits = np.array([0.5, -0.3, 0.1])

        probs = np.array([[0.8, 0.3, 0.6], [0.4, 0.9, 0.5]])
        labels = np.array([1.0, 0.0])

        n = learner.n_signals
        scale = n ** learner.alpha
        x = _logit(_clamp_probability(probs))

        # Analytical gradient
        w = learner._softmax(learner._logits)
        x_bar_w = np.sum(w * x, axis=-1)
        p = np.atleast_1d(np.asarray(_sigmoid(scale * x_bar_w), dtype=np.float64))
        error = p - labels
        analytical_grad = np.mean(
            scale * error[:, np.newaxis] * w[np.newaxis, :] * (x - x_bar_w[:, np.newaxis]),
            axis=0,
        )

        # Finite-difference gradient
        eps = 1e-5
        fd_grad = np.zeros(n)
        for j in range(n):
            logits_plus = learner._logits.copy()
            logits_plus[j] += eps
            w_plus = learner._softmax(logits_plus)
            x_bar_plus = np.sum(w_plus * x, axis=-1)
            p_plus = np.atleast_1d(np.asarray(_sigmoid(scale * x_bar_plus), dtype=np.float64))
            loss_plus = -np.mean(labels * np.log(np.clip(p_plus, 1e-10, None))
                                 + (1 - labels) * np.log(np.clip(1 - p_plus, 1e-10, None)))

            logits_minus = learner._logits.copy()
            logits_minus[j] -= eps
            w_minus = learner._softmax(logits_minus)
            x_bar_minus = np.sum(w_minus * x, axis=-1)
            p_minus = np.atleast_1d(np.asarray(_sigmoid(scale * x_bar_minus), dtype=np.float64))
            loss_minus = -np.mean(labels * np.log(np.clip(p_minus, 1e-10, None))
                                  + (1 - labels) * np.log(np.clip(1 - p_minus, 1e-10, None)))

            fd_grad[j] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(analytical_grad, fd_grad, atol=1e-4)

    def test_pickle_roundtrip(self):
        """Pickle round-trip preserves logits, weights, and online state."""
        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.5)
        learner._logits = np.array([1.0, -0.5, 0.3])
        learner._n_updates = 5
        learner._grad_logits_ema = np.array([0.1, -0.2, 0.05])
        learner._weights_avg = learner._softmax(learner._logits).copy()

        restored = pickle.loads(pickle.dumps(learner))

        assert restored.n_signals == learner.n_signals
        assert restored.alpha == learner.alpha
        np.testing.assert_allclose(restored._logits, learner._logits)
        np.testing.assert_allclose(restored.weights, learner.weights)
        np.testing.assert_allclose(restored.averaged_weights, learner.averaged_weights)
        assert restored._n_updates == learner._n_updates
        np.testing.assert_allclose(restored._grad_logits_ema, learner._grad_logits_ema)


class TestBalancedLogOddsFusion:
    """Tests for balanced_log_odds_fusion (hybrid sparse-dense retrieval)."""

    def test_equal_weight(self):
        """weight=0.5 gives equal contribution from sparse and dense."""
        sparse = np.array([0.8, 0.6, 0.3])
        dense = np.array([0.5, 0.7, 0.9])
        result = balanced_log_odds_fusion(sparse, dense, weight=0.5)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_sparse_only_weight(self):
        """weight=0.0 means only sparse contributes."""
        sparse = np.array([0.8, 0.6, 0.3])
        dense = np.array([0.5, 0.7, 0.9])
        result = balanced_log_odds_fusion(sparse, dense, weight=0.0)
        # Result should depend only on sparse signal
        result_diff_dense = balanced_log_odds_fusion(
            sparse, np.array([0.1, 0.2, 0.3]), weight=0.0
        )
        np.testing.assert_allclose(result, result_diff_dense)

    def test_dense_only_weight(self):
        """weight=1.0 means only dense contributes."""
        sparse = np.array([0.8, 0.6, 0.3])
        dense = np.array([0.5, 0.7, 0.9])
        result = balanced_log_odds_fusion(sparse, dense, weight=1.0)
        # Result should depend only on dense signal
        result_diff_sparse = balanced_log_odds_fusion(
            np.array([0.1, 0.2, 0.3]), dense, weight=1.0
        )
        np.testing.assert_allclose(result, result_diff_sparse)

    def test_monotonicity_sparse(self):
        """Within a single call, higher sparse probs produce higher fusion scores.

        Since _min_max_normalize is range-relative (removing absolute scale),
        monotonicity is tested within a single call: a document with higher
        sparse probability should receive a higher fusion score.
        """
        sparse = np.array([0.3, 0.6, 0.9])
        dense = np.array([0.5, 0.5, 0.5])  # uniform -> contributes zeros
        result = balanced_log_odds_fusion(sparse, dense, weight=0.3)
        # Sparse drives ranking because dense is uniform (normalized to zeros)
        assert np.all(np.diff(result) >= 0)

    def test_monotonicity_dense(self):
        """Within a single call, higher dense sims produce higher fusion scores.

        Since _min_max_normalize is range-relative (removing absolute scale),
        monotonicity is tested within a single call: a document with higher
        dense similarity should receive a higher fusion score.
        """
        sparse = np.array([0.5, 0.5, 0.5])  # uniform -> contributes zeros
        dense = np.array([-0.5, 0.0, 0.8])
        result = balanced_log_odds_fusion(sparse, dense, weight=0.7)
        # Dense drives ranking because sparse is uniform (normalized to zeros)
        assert np.all(np.diff(result) >= 0)

    def test_identical_sparse_scores(self):
        """All-same sparse scores produce zeros from _min_max_normalize."""
        sparse = np.array([0.5, 0.5, 0.5])
        dense = np.array([0.3, 0.6, 0.9])
        result = balanced_log_odds_fusion(sparse, dense, weight=0.5)
        # Sparse contributes zero; only dense drives the result
        assert result.shape == (3,)
        # Ranking should follow dense signal
        assert result[2] > result[1] > result[0]

    def test_scalar_input(self):
        """Single float inputs produce a finite float result."""
        result = balanced_log_odds_fusion(
            np.float64(0.7), np.float64(0.5), weight=0.5
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_output_is_numeric(self):
        """Result is always finite."""
        sparse = np.array([0.01, 0.5, 0.99])
        dense = np.array([-0.99, 0.0, 0.99])
        result = balanced_log_odds_fusion(sparse, dense)
        assert np.all(np.isfinite(result))


class TestAlphaAuto:
    """Tests for alpha='auto' (sqrt(n) scaling, Theorem 4.2.1)."""

    def test_auto_matches_half_unweighted(self):
        """alpha='auto' produces same result as alpha=0.5 in unweighted mode."""
        probs = np.array([0.8, 0.9])
        auto = log_odds_conjunction(probs, alpha="auto")
        explicit = log_odds_conjunction(probs, alpha=0.5)
        assert auto == pytest.approx(explicit, abs=1e-12)

    def test_auto_matches_half_weighted(self):
        """alpha='auto' produces same result as alpha=0.5 in weighted mode."""
        probs = np.array([0.8, 0.9])
        w = np.array([0.6, 0.4])
        auto = log_odds_conjunction(probs, alpha="auto", weights=w)
        explicit = log_odds_conjunction(probs, alpha=0.5, weights=w)
        assert auto == pytest.approx(explicit, abs=1e-12)

    def test_auto_amplifies_agreement(self):
        """alpha='auto' amplifies agreement like alpha=0.5."""
        probs = np.array([0.9, 0.9])
        result = log_odds_conjunction(probs, alpha="auto")
        assert result > 0.9  # Agreement amplification

    def test_auto_batched(self):
        """alpha='auto' works with batched inputs."""
        probs = np.array([[0.9, 0.9], [0.3, 0.3]])
        result = log_odds_conjunction(probs, alpha="auto")
        assert result.shape == (2,)
        assert result[0] > 0.9
        assert result[1] < 0.5

    def test_invalid_alpha_string_raises(self):
        """Non-'auto' string raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be"):
            log_odds_conjunction(np.array([0.5, 0.5]), alpha="invalid")

    def test_learnable_alpha_auto(self):
        """LearnableLogOddsWeights accepts alpha='auto'."""
        learner = LearnableLogOddsWeights(n_signals=3, alpha="auto")
        assert learner.alpha == 0.5

    def test_learnable_alpha_auto_call(self):
        """LearnableLogOddsWeights with alpha='auto' produces valid output."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha="auto")
        probs = np.array([0.8, 0.7])
        result = learner(probs)
        assert 0 < result < 1

    def test_none_defaults_preserved(self):
        """alpha=None still uses 0.5 unweighted, 0.0 weighted (backward compat)."""
        probs = np.array([0.8, 0.9])
        none_unweighted = log_odds_conjunction(probs, alpha=None)
        half_unweighted = log_odds_conjunction(probs, alpha=0.5)
        assert none_unweighted == pytest.approx(half_unweighted, abs=1e-12)

        w = np.array([0.6, 0.4])
        none_weighted = log_odds_conjunction(probs, alpha=None, weights=w)
        zero_weighted = log_odds_conjunction(probs, alpha=0.0, weights=w)
        assert none_weighted == pytest.approx(zero_weighted, abs=1e-12)


class TestGating:
    """Tests for sparse signal gating (ReLU/Swish, Theorems 6.5.3, 6.7.4)."""

    def test_none_gating_identity(self):
        """gating='none' gives same result as no gating."""
        probs = np.array([0.8, 0.9])
        result_none = log_odds_conjunction(probs, gating="none")
        result_default = log_odds_conjunction(probs)
        assert result_none == pytest.approx(result_default, abs=1e-12)

    def test_relu_zeros_weak_evidence(self):
        """ReLU gating ignores signals below 0.5 (negative logits)."""
        # 0.3 has logit < 0, so ReLU zeroes it out
        probs = np.array([0.9, 0.3])
        result_relu = log_odds_conjunction(probs, gating="relu")
        result_none = log_odds_conjunction(probs, gating="none")
        # With ReLU, the 0.3 signal is zeroed: effectively only 0.9 contributes
        # so the result should be higher than without gating
        assert result_relu > result_none

    def test_relu_all_above_half(self):
        """ReLU gating on all-above-0.5 signals is close to no gating."""
        probs = np.array([0.8, 0.9, 0.7])
        result_relu = log_odds_conjunction(probs, gating="relu")
        result_none = log_odds_conjunction(probs, gating="none")
        # All logits are positive, so ReLU does not change them
        assert result_relu == pytest.approx(result_none, abs=1e-12)

    def test_swish_soft_gate(self):
        """Swish gating is between 'none' and 'relu' for mixed signals."""
        probs = np.array([0.9, 0.3])
        result_none = log_odds_conjunction(probs, gating="none")
        result_swish = log_odds_conjunction(probs, gating="swish")
        result_relu = log_odds_conjunction(probs, gating="relu")
        # Swish is a soft version of ReLU: result should be between none and relu
        assert result_none < result_swish < result_relu

    def test_swish_all_above_half(self):
        """Swish with all signals > 0.5 is close to (but below) no gating."""
        probs = np.array([0.8, 0.9])
        result_swish = log_odds_conjunction(probs, gating="swish")
        result_none = log_odds_conjunction(probs, gating="none")
        # Swish(x) = x*sigmoid(x) < x for all finite x > 0, so swish
        # gating always attenuates positive logits slightly
        assert result_swish < result_none
        # But it should be close (within 6%) for moderate positive logits
        assert abs(result_swish - result_none) < 0.06

    def test_gating_with_weights(self):
        """Gating works with weighted mode."""
        probs = np.array([0.9, 0.3])
        w = np.array([0.5, 0.5])
        result_none = log_odds_conjunction(probs, weights=w, gating="none")
        result_relu = log_odds_conjunction(probs, weights=w, gating="relu")
        # ReLU zeroes the 0.3 signal (negative logit), boosting the result
        assert result_relu > result_none

    def test_gating_with_alpha_auto(self):
        """Gating works together with alpha='auto'."""
        probs = np.array([0.9, 0.3, 0.8])
        result = log_odds_conjunction(probs, alpha="auto", gating="relu")
        assert 0 < result < 1

    def test_invalid_gating_raises(self):
        """Invalid gating value raises ValueError."""
        with pytest.raises(ValueError, match="gating must be"):
            log_odds_conjunction(np.array([0.5, 0.5]), gating="invalid")

    def test_relu_batched(self):
        """ReLU gating works with batched inputs."""
        probs = np.array([[0.9, 0.3], [0.3, 0.9]])
        result = log_odds_conjunction(probs, gating="relu")
        assert result.shape == (2,)
        # Both batches have one signal zeroed out, result should be similar
        assert result[0] == pytest.approx(result[1], abs=1e-12)

    def test_swish_batched(self):
        """Swish gating works with batched inputs."""
        probs = np.array([[0.9, 0.3], [0.8, 0.8]])
        result = log_odds_conjunction(probs, gating="swish")
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
        assert np.all(result < 1)


class TestAttentionLogOddsWeights:
    """Tests for AttentionLogOddsWeights (Paper 2, Section 8)."""

    def test_init_shapes(self):
        """Weight matrix and bias have correct shapes."""
        attn = AttentionLogOddsWeights(n_signals=3, n_query_features=5)
        assert attn.n_signals == 3
        assert attn.n_query_features == 5
        assert attn.weights_matrix.shape == (3, 5)
        assert attn.alpha == 0.5

    def test_init_alpha_auto(self):
        """alpha='auto' resolves to 0.5."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3, alpha="auto")
        assert attn.alpha == 0.5

    def test_init_invalid_n_signals(self):
        with pytest.raises(ValueError, match="n_signals"):
            AttentionLogOddsWeights(n_signals=0, n_query_features=3)

    def test_init_invalid_n_query_features(self):
        with pytest.raises(ValueError, match="n_query_features"):
            AttentionLogOddsWeights(n_signals=2, n_query_features=0)

    def test_call_single_sample(self):
        """Single-sample call returns a float in (0, 1)."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        result = attn(probs, qf)
        assert isinstance(result, float)
        assert 0 < result < 1

    def test_call_batched(self):
        """Batched call returns array of correct shape."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        probs = np.array([[0.8, 0.7], [0.3, 0.9]])
        qf = np.array([[1.0, 0.5, -0.3], [0.2, -0.1, 0.8]])
        result = attn(probs, qf)
        assert result.shape == (2,)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_different_queries_produce_different_weights(self):
        """Different query features lead to different fusion results."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        probs = np.array([0.9, 0.3])
        qf1 = np.array([1.0, 0.0, 0.0])
        qf2 = np.array([0.0, 0.0, 1.0])
        r1 = attn(probs, qf1)
        r2 = attn(probs, qf2)
        # With random init, different features produce different weights
        assert r1 != pytest.approx(r2, abs=1e-6)

    def test_fit_learns_informative_features(self):
        """fit() should learn to use informative query features."""
        rng = np.random.RandomState(42)
        m = 300
        n_signals = 2
        n_qf = 3

        labels = rng.randint(0, 2, size=m).astype(np.float64)

        # Signal 0: reliable, Signal 1: noisy
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        signal_1 = rng.uniform(0.3, 0.7, size=m)
        probs = np.column_stack([signal_0, signal_1])

        # Query features: feature 0 indicates which signal is reliable
        query_features = rng.randn(m, n_qf)
        query_features[:, 0] = 1.0  # constant feature favoring signal 0

        attn = AttentionLogOddsWeights(n_signals=n_signals, n_query_features=n_qf, alpha=0.0)
        attn.fit(
            probs, labels, query_features,
            learning_rate=0.1, max_iterations=2000,
        )

        # After training, fusion should produce high probs for relevant docs
        test_qf = np.array([1.0, 0.0, 0.0])
        result_high = attn(np.array([0.9, 0.5]), test_qf)
        result_low = attn(np.array([0.1, 0.5]), test_qf)
        assert result_high > result_low

    def test_update_moves_parameters(self):
        """update() changes internal parameters from initial state."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        W_before = attn.weights_matrix.copy()

        for _ in range(50):
            probs = np.array([0.9, 0.1])
            qf = np.array([1.0, 0.0])
            attn.update(probs, 1.0, qf, learning_rate=0.05)

        W_after = attn.weights_matrix
        assert not np.allclose(W_before, W_after)

    def test_use_averaged(self):
        """use_averaged=True uses Polyak-averaged parameters."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2, alpha=0.0)
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5])

        # At init, averaged = raw
        r_raw = attn(probs, qf, use_averaged=False)
        r_avg = attn(probs, qf, use_averaged=True)
        assert r_raw == pytest.approx(r_avg, abs=1e-10)

    def test_weights_matrix_returns_copy(self):
        """weights_matrix returns a copy, not a reference."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        w1 = attn.weights_matrix
        w2 = attn.weights_matrix
        assert w1 is not w2
        w1[0, 0] = 999.0
        assert attn.weights_matrix[0, 0] != 999.0

    def test_fit_resets_online_state(self):
        """fit() resets online state."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        # Simulate some online updates
        attn.update(np.array([0.8, 0.2]), 1.0, np.array([1.0, 0.0]))
        assert attn._n_updates == 1

        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1.0, 0.0])
        qf = np.array([[1.0, 0.0], [0.0, 1.0]])
        attn.fit(probs, labels, qf)

        assert attn._n_updates == 0
        np.testing.assert_allclose(attn._grad_W_ema, 0.0)
        np.testing.assert_allclose(attn._grad_b_ema, 0.0)

    def test_softmax_numerical_stability(self):
        """Softmax handles large values without overflow."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        z = np.array([[1000.0, -1000.0], [0.0, 0.0]])
        result = attn._softmax(z)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-10)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_normalize_default_false(self):
        """Default normalize is False."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        assert attn.normalize is False

    def test_normalize_property(self):
        """normalize property reflects constructor argument."""
        attn = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, normalize=True
        )
        assert attn.normalize is True

    def test_call_normalize_rescales(self):
        """Normalize=True produces different results from normalize=False.

        Two signals with very different logit scales: without normalization
        the high-logit signal dominates; with normalization both contribute
        equally after per-column min-max rescaling.
        """
        # Signal 0: high dynamic range (logits span ~8 units)
        # Signal 1: low dynamic range (logits span ~0.4 units)
        probs = np.array([
            [0.99, 0.55],
            [0.50, 0.50],
            [0.01, 0.45],
        ])
        # Per-row query features (batched call requires matching rows)
        qf = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        attn_plain = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, normalize=False
        )
        attn_norm = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, normalize=True
        )
        # Copy weights so both use the same parameters
        attn_norm._W = attn_plain._W.copy()
        attn_norm._b = attn_plain._b.copy()
        attn_norm._W_avg = attn_plain._W_avg.copy()
        attn_norm._b_avg = attn_plain._b_avg.copy()

        r_plain = attn_plain(probs, qf)
        r_norm = attn_norm(probs, qf)
        # Results should differ due to normalization
        assert not np.allclose(r_plain, r_norm)

    def test_call_normalize_single_sample_fallthrough(self):
        """1D probs with normalize=True falls through to non-normalized path."""
        attn_norm = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, normalize=True
        )
        attn_plain = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, normalize=False
        )
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        r_norm = attn_norm(probs, qf)
        r_plain = attn_plain(probs, qf)
        # 1D input cannot be normalized (no candidates to normalize across),
        # so both paths should produce identical results.
        assert np.isclose(r_norm, r_plain)
        assert 0 < r_norm < 1

    def test_fit_normalize_with_query_ids(self):
        """fit with query_ids groups normalization correctly."""
        rng = np.random.RandomState(42)
        m = 200
        n_signals = 2
        n_qf = 2

        labels = rng.randint(0, 2, size=m).astype(np.float64)
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        signal_1 = rng.uniform(0.3, 0.7, size=m)
        probs = np.column_stack([signal_0, signal_1])
        query_features = np.ones((m, n_qf), dtype=np.float64)
        # 10 queries, each with 20 candidates
        query_ids = np.repeat(np.arange(10), 20)

        attn = AttentionLogOddsWeights(
            n_signals=n_signals, n_query_features=n_qf,
            alpha=0.0, normalize=True,
        )
        # Should not raise
        attn.fit(
            probs, labels, query_features,
            query_ids=query_ids,
            learning_rate=0.1, max_iterations=500,
        )
        # After training, fusion should produce reasonable results
        test_probs = np.array([[0.9, 0.5], [0.1, 0.5]])
        test_qf = np.array([1.0, 0.0])
        results = attn(test_probs, test_qf)
        assert results.shape == (2,)

    def test_fit_normalize_without_query_ids(self):
        """fit without query_ids normalizes whole batch."""
        rng = np.random.RandomState(42)
        m = 100
        labels = rng.randint(0, 2, size=m).astype(np.float64)
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        signal_1 = rng.uniform(0.3, 0.7, size=m)
        probs = np.column_stack([signal_0, signal_1])
        query_features = np.ones((m, 2), dtype=np.float64)

        attn = AttentionLogOddsWeights(
            n_signals=2, n_query_features=2,
            alpha=0.0, normalize=True,
        )
        # Should not raise -- normalizes the whole batch as one group
        attn.fit(
            probs, labels, query_features,
            learning_rate=0.1, max_iterations=500,
        )
        assert attn._n_updates == 0  # fit resets online state

    def test_fit_normalize_query_ids_vs_global(self):
        """Per-query normalization (query_ids) produces different weights
        than global batch normalization when signal scales differ per query."""
        rng = np.random.RandomState(123)
        n_per_q = 50
        n_signals = 2
        n_qf = 2

        # Query A: signal 0 has logits in [-10, 10], signal 1 in [-1, 1]
        # Query B: signal 0 has logits in [-1, 1], signal 1 in [-10, 10]
        # Global normalization mixes these scales; per-query fixes them.
        labels_a = rng.randint(0, 2, size=n_per_q).astype(np.float64)
        labels_b = rng.randint(0, 2, size=n_per_q).astype(np.float64)

        from bayesian_bm25.fusion import sigmoid
        probs_a = np.column_stack([
            sigmoid(np.where(labels_a == 1, rng.uniform(3, 10, n_per_q),
                             rng.uniform(-10, -3, n_per_q))),
            sigmoid(rng.uniform(-1, 1, n_per_q)),
        ])
        probs_b = np.column_stack([
            sigmoid(rng.uniform(-1, 1, n_per_q)),
            sigmoid(np.where(labels_b == 1, rng.uniform(3, 10, n_per_q),
                             rng.uniform(-10, -3, n_per_q))),
        ])

        probs = np.vstack([probs_a, probs_b])
        labels = np.concatenate([labels_a, labels_b])
        qf = np.ones((2 * n_per_q, n_qf), dtype=np.float64)
        query_ids = np.array([0] * n_per_q + [1] * n_per_q)

        attn_global = AttentionLogOddsWeights(
            n_signals=n_signals, n_query_features=n_qf,
            alpha=0.0, normalize=True,
        )
        attn_global.fit(probs, labels, qf,
                        learning_rate=0.1, max_iterations=300)

        attn_perq = AttentionLogOddsWeights(
            n_signals=n_signals, n_query_features=n_qf,
            alpha=0.0, normalize=True,
        )
        attn_perq.fit(probs, labels, qf, query_ids=query_ids,
                      learning_rate=0.1, max_iterations=300)

        # The learned weight matrices should differ
        assert not np.allclose(attn_global._W, attn_perq._W, atol=1e-3)

    def test_normalize_uniform_signal_zeros_out(self):
        """Signal with all-same values normalizes to zeros.

        _min_max_normalize maps constant arrays to all-zeros, so a
        uniform signal contributes nothing after normalization.
        """
        # Signal 0: varied, Signal 1: constant 0.5
        probs = np.array([
            [0.9, 0.5],
            [0.5, 0.5],
            [0.1, 0.5],
        ])
        qf = np.array([1.0, 0.0])

        attn = AttentionLogOddsWeights(
            n_signals=2, n_query_features=2, normalize=True
        )
        result = attn(probs, qf)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Feature 2: Generalized Swish / Swish_beta (Theorem 6.7.6)
# ---------------------------------------------------------------------------

class TestGatingBeta:
    """Tests for gating_beta parameter in log_odds_conjunction."""

    def test_beta_one_matches_existing_swish(self):
        """gating_beta=1.0 matches existing swish behavior."""
        probs = np.array([0.9, 0.3, 0.7])
        result_default = log_odds_conjunction(probs, gating="swish")
        result_beta1 = log_odds_conjunction(probs, gating="swish", gating_beta=1.0)
        assert result_beta1 == pytest.approx(result_default, abs=1e-12)

    def test_beta_zero_approaches_half(self):
        """gating_beta -> 0 makes swish approach x/2 (Theorem 6.7.6 limit)."""
        probs = np.array([0.9, 0.7])
        # With very small beta, swish(x) = x * sigmoid(beta*x) -> x * 0.5 = x/2
        result = log_odds_conjunction(probs, gating="swish", gating_beta=0.001)
        # Compute expected: mean of x/2 values, scaled by n^alpha
        from bayesian_bm25.probability import _clamp_probability
        x = logit(_clamp_probability(np.asarray(probs)))
        mean_half = float(np.mean(x / 2.0))
        expected = float(sigmoid(mean_half * (2 ** 0.5)))
        assert result == pytest.approx(expected, abs=0.01)

    def test_beta_large_approaches_relu(self):
        """gating_beta -> large makes swish approach ReLU (Theorem 6.7.6 limit)."""
        probs = np.array([0.9, 0.3])
        result_relu = log_odds_conjunction(probs, gating="relu")
        result_large_beta = log_odds_conjunction(probs, gating="swish", gating_beta=100.0)
        assert result_large_beta == pytest.approx(result_relu, abs=0.01)

    def test_swish_beta_1702_matches_gelu(self):
        """gating="swish", gating_beta=1.702 matches gating="gelu"."""
        probs = np.array([0.9, 0.3, 0.7])
        result_swish = log_odds_conjunction(probs, gating="swish", gating_beta=1.702)
        result_gelu = log_odds_conjunction(probs, gating="gelu")
        assert result_swish == pytest.approx(result_gelu, abs=1e-10)


# ---------------------------------------------------------------------------
# Feature 1: GELU Gating (Theorem 6.8.1, Proposition 6.8.2)
# ---------------------------------------------------------------------------

class TestGELUGating:
    """Tests for GELU gating in log_odds_conjunction."""

    def test_gelu_between_swish_and_relu_positive(self):
        """GELU result between swish and relu for positive logits."""
        probs = np.array([0.8, 0.9, 0.7])  # all > 0.5, positive logits
        result_none = log_odds_conjunction(probs, gating="none")
        result_swish = log_odds_conjunction(probs, gating="swish")
        result_gelu = log_odds_conjunction(probs, gating="gelu")
        result_relu = log_odds_conjunction(probs, gating="relu")
        # For positive logits, gating attenuates: none > relu > gelu > swish
        # But all are close since all logits positive
        # GELU is between swish(beta=1) and swish(beta=inf)=relu
        assert result_swish < result_gelu < result_relu

    def test_gelu_matches_swish_1702(self):
        """GELU approximates Swish_1.702 (Proposition 6.8.2)."""
        probs = np.array([0.9, 0.3, 0.7, 0.2])
        result_gelu = log_odds_conjunction(probs, gating="gelu")
        result_swish = log_odds_conjunction(probs, gating="swish", gating_beta=1.702)
        assert result_gelu == pytest.approx(result_swish, abs=1e-10)

    def test_gelu_batched(self):
        """GELU gating works with batched inputs."""
        probs = np.array([[0.9, 0.3], [0.8, 0.8]])
        result = log_odds_conjunction(probs, gating="gelu")
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_gelu_with_weights(self):
        """GELU gating works with weighted mode."""
        probs = np.array([0.9, 0.3])
        w = np.array([0.5, 0.5])
        result = log_odds_conjunction(probs, weights=w, gating="gelu")
        assert 0 < result < 1

    def test_gelu_ignores_gating_beta(self):
        """gating='gelu' ignores the gating_beta parameter."""
        probs = np.array([0.9, 0.3, 0.7])
        result_default = log_odds_conjunction(probs, gating="gelu")
        result_custom_beta = log_odds_conjunction(probs, gating="gelu", gating_beta=5.0)
        assert result_default == pytest.approx(result_custom_beta, abs=1e-12)


# ---------------------------------------------------------------------------
# Feature 9: Vectorize Attention Forward Pass
# ---------------------------------------------------------------------------

class TestAttentionVectorized:
    """Tests for vectorized AttentionLogOddsWeights forward pass."""

    def test_vectorized_matches_loop(self):
        """Vectorized path matches per-row loop for random samples."""
        rng = np.random.default_rng(42)
        m = 100
        n_signals = 3
        n_qf = 4
        attn = AttentionLogOddsWeights(n_signals=n_signals, n_query_features=n_qf)

        probs = rng.uniform(0.1, 0.9, size=(m, n_signals))
        qf = rng.standard_normal(size=(m, n_qf))

        # Vectorized result
        result_vectorized = attn(probs, qf)

        # Per-row loop result
        result_loop = np.empty(m, dtype=np.float64)
        for i in range(m):
            result_loop[i] = attn(probs[i], qf[i])

        np.testing.assert_allclose(result_vectorized, result_loop, atol=1e-10)

    def test_single_query_broadcast(self):
        """Single query features broadcast across all candidates."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        probs = np.array([[0.8, 0.7], [0.3, 0.9], [0.6, 0.6]])
        qf = np.array([1.0, 0.5, -0.3])  # single query
        result = attn(probs, qf)
        assert result.shape == (3,)
        assert np.all(result > 0)
        assert np.all(result < 1)


# ---------------------------------------------------------------------------
# Feature 10: base_rate in Fusion Classes
# ---------------------------------------------------------------------------

class TestLearnableBaseRate:
    """Tests for base_rate in LearnableLogOddsWeights."""

    def test_none_preserves_existing(self):
        """base_rate=None preserves existing behavior."""
        learner_none = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        learner_explicit = LearnableLogOddsWeights(n_signals=2, alpha=0.0, base_rate=None)
        probs = np.array([0.7, 0.8])
        assert learner_none(probs) == pytest.approx(learner_explicit(probs), abs=1e-12)

    def test_base_rate_half_neutral(self):
        """base_rate=0.5 is neutral (logit(0.5)=0)."""
        learner_none = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        learner_half = LearnableLogOddsWeights(n_signals=2, alpha=0.0, base_rate=0.5)
        probs = np.array([0.7, 0.8])
        assert learner_none(probs) == pytest.approx(learner_half(probs), abs=1e-8)

    def test_low_base_rate_shifts_down(self):
        """Low base_rate shifts output downward."""
        learner_none = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        learner_low = LearnableLogOddsWeights(n_signals=2, alpha=0.0, base_rate=0.01)
        probs = np.array([0.7, 0.8])
        assert learner_low(probs) < learner_none(probs)

    def test_high_base_rate_shifts_up(self):
        """High base_rate shifts output upward."""
        learner_none = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        learner_high = LearnableLogOddsWeights(n_signals=2, alpha=0.0, base_rate=0.99)
        probs = np.array([0.7, 0.8])
        assert learner_high(probs) > learner_none(probs)

    def test_invalid_base_rate_raises(self):
        """Invalid base_rate raises ValueError."""
        with pytest.raises(ValueError, match="base_rate must be in"):
            LearnableLogOddsWeights(n_signals=2, base_rate=0.0)
        with pytest.raises(ValueError, match="base_rate must be in"):
            LearnableLogOddsWeights(n_signals=2, base_rate=1.0)
        with pytest.raises(ValueError, match="base_rate must be in"):
            LearnableLogOddsWeights(n_signals=2, base_rate=-0.1)

    def test_base_rate_property(self):
        """base_rate property returns correct value."""
        learner = LearnableLogOddsWeights(n_signals=2, base_rate=0.1)
        assert learner.base_rate == 0.1

    def test_fit_with_base_rate(self):
        """fit() works with base_rate set."""
        rng = np.random.RandomState(42)
        m = 200
        labels = rng.randint(0, 2, size=m).astype(np.float64)
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        signal_1 = rng.uniform(0.3, 0.7, size=m)
        probs = np.column_stack([signal_0, signal_1])

        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0, base_rate=0.3)
        learner.fit(probs, labels, learning_rate=0.1, max_iterations=1000)
        assert learner.weights[0] > learner.weights[1]

    def test_update_with_base_rate(self):
        """update() works with base_rate set."""
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0, base_rate=0.1)
        for _ in range(20):
            learner.update(np.array([0.9, 0.5]), 1.0, learning_rate=0.05)
            learner.update(np.array([0.1, 0.5]), 0.0, learning_rate=0.05)
        assert learner._n_updates == 40


class TestAttentionBaseRate:
    """Tests for base_rate in AttentionLogOddsWeights."""

    def test_none_preserves_existing(self):
        """base_rate=None preserves existing behavior."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        r1 = attn(probs, qf)
        # Re-create with explicit None
        attn2 = AttentionLogOddsWeights(n_signals=2, n_query_features=3, base_rate=None)
        r2 = attn2(probs, qf)
        # Both use seed=0 default, should match
        assert r1 == pytest.approx(r2, abs=1e-12)

    def test_base_rate_half_neutral(self):
        """base_rate=0.5 is neutral (logit(0.5)=0)."""
        attn_none = AttentionLogOddsWeights(n_signals=2, n_query_features=3, seed=42)
        attn_half = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, seed=42, base_rate=0.5
        )
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        assert attn_none(probs, qf) == pytest.approx(attn_half(probs, qf), abs=1e-8)

    def test_low_base_rate_shifts_down(self):
        """Low base_rate shifts output downward."""
        attn_none = AttentionLogOddsWeights(n_signals=2, n_query_features=3, seed=42)
        attn_low = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, seed=42, base_rate=0.01
        )
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        assert attn_low(probs, qf) < attn_none(probs, qf)

    def test_high_base_rate_shifts_up(self):
        """High base_rate shifts output upward."""
        attn_none = AttentionLogOddsWeights(n_signals=2, n_query_features=3, seed=42)
        attn_high = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, seed=42, base_rate=0.99
        )
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        assert attn_high(probs, qf) > attn_none(probs, qf)

    def test_invalid_base_rate_raises(self):
        """Invalid base_rate raises ValueError."""
        with pytest.raises(ValueError, match="base_rate must be in"):
            AttentionLogOddsWeights(n_signals=2, n_query_features=3, base_rate=0.0)

    def test_base_rate_property(self):
        """base_rate property returns correct value."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=3, base_rate=0.1)
        assert attn.base_rate == 0.1

    def test_fit_with_base_rate(self):
        """fit() works with base_rate set."""
        rng = np.random.RandomState(42)
        m = 200
        labels = rng.randint(0, 2, size=m).astype(np.float64)
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        signal_1 = rng.uniform(0.3, 0.7, size=m)
        probs = np.column_stack([signal_0, signal_1])
        qf = np.ones((m, 2), dtype=np.float64)

        attn = AttentionLogOddsWeights(
            n_signals=2, n_query_features=2, alpha=0.0, base_rate=0.3
        )
        attn.fit(probs, labels, qf, learning_rate=0.1, max_iterations=500)
        result = attn(np.array([0.9, 0.5]), np.array([1.0, 0.0]))
        assert 0 < result < 1

    def test_update_with_base_rate(self):
        """update() works with base_rate set."""
        attn = AttentionLogOddsWeights(
            n_signals=2, n_query_features=2, base_rate=0.1
        )
        for _ in range(10):
            attn.update(np.array([0.9, 0.5]), 1.0, np.array([1.0, 0.0]))
        assert attn._n_updates == 10


# ---------------------------------------------------------------------------
# Feature 5: Multi-Head Attention Fusion (Remark 8.6, Corollary 8.7.2)
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionLogOddsWeights:
    """Tests for MultiHeadAttentionLogOddsWeights."""

    def test_single_head_matches_attention(self):
        """Single head matches AttentionLogOddsWeights with seed=0."""
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=1, n_signals=2, n_query_features=3
        )
        single = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, seed=0
        )
        probs = np.array([0.8, 0.7])
        qf = np.array([1.0, 0.5, -0.3])
        r_mh = mh(probs, qf)
        r_single = single(probs, qf)
        assert r_mh == pytest.approx(r_single, abs=1e-10)

    def test_output_in_0_1(self):
        """Output always in (0, 1)."""
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=4, n_signals=3, n_query_features=2
        )
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.1, 0.9, size=(20, 3))
        qf = rng.standard_normal(size=(20, 2))
        result = mh(probs, qf)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_fit_reduces_bce(self):
        """fit() reduces BCE loss."""
        rng = np.random.RandomState(42)
        m = 200
        labels = rng.randint(0, 2, size=m).astype(np.float64)
        signal_0 = np.where(labels == 1, 0.85, 0.15)
        signal_1 = rng.uniform(0.3, 0.7, size=m)
        probs = np.column_stack([signal_0, signal_1])
        qf = np.ones((m, 2), dtype=np.float64)

        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=3, n_signals=2, n_query_features=2, alpha=0.0
        )
        # Measure BCE before training
        pred_before = np.array([mh(probs[i], qf[i]) for i in range(m)])
        pred_before = np.clip(pred_before, 1e-10, 1.0 - 1e-10)
        bce_before = -np.mean(
            labels * np.log(pred_before) + (1 - labels) * np.log(1 - pred_before)
        )

        mh.fit(probs, labels, qf, learning_rate=0.1, max_iterations=500)

        pred_after = np.array([mh(probs[i], qf[i]) for i in range(m)])
        pred_after = np.clip(pred_after, 1e-10, 1.0 - 1e-10)
        bce_after = -np.mean(
            labels * np.log(pred_after) + (1 - labels) * np.log(1 - pred_after)
        )
        assert bce_after < bce_before

    def test_different_heads_different_weights(self):
        """Different heads produce different weights after training (diversity)."""
        rng = np.random.RandomState(42)
        m = 200
        labels = rng.randint(0, 2, size=m).astype(np.float64)
        probs = np.column_stack([
            np.where(labels == 1, 0.85, 0.15),
            rng.uniform(0.3, 0.7, size=m),
        ])
        qf = np.ones((m, 2), dtype=np.float64)

        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=3, n_signals=2, n_query_features=2, alpha=0.0
        )
        mh.fit(probs, labels, qf, learning_rate=0.1, max_iterations=300)

        # Check that heads have different weight matrices
        w0 = mh.heads[0].weights_matrix
        w1 = mh.heads[1].weights_matrix
        assert not np.allclose(w0, w1, atol=1e-3)

    def test_batched_inputs(self):
        """Batched inputs work."""
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=2, n_signals=2, n_query_features=3
        )
        probs = np.array([[0.8, 0.7], [0.3, 0.9]])
        qf = np.array([[1.0, 0.5, -0.3], [0.2, -0.1, 0.8]])
        result = mh(probs, qf)
        assert result.shape == (2,)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_n_heads_property(self):
        """n_heads property returns correct value."""
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=4, n_signals=2, n_query_features=3
        )
        assert mh.n_heads == 4
        assert len(mh.heads) == 4

    def test_invalid_n_heads(self):
        """n_heads < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_heads"):
            MultiHeadAttentionLogOddsWeights(
                n_heads=0, n_signals=2, n_query_features=3
            )

    def test_update(self):
        """update() changes parameters for all heads."""
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=2, n_signals=2, n_query_features=2
        )
        W_before = [h.weights_matrix.copy() for h in mh.heads]

        for _ in range(20):
            mh.update(np.array([0.9, 0.1]), 1.0, np.array([1.0, 0.0]))

        for i, head in enumerate(mh.heads):
            assert not np.allclose(W_before[i], head.weights_matrix)


# ---------------------------------------------------------------------------
# Feature 6: Exact Attention Pruning (Theorem 8.7.1, Corollary 8.7.2)
# ---------------------------------------------------------------------------

class TestAttentionPruning:
    """Tests for compute_upper_bounds and prune on AttentionLogOddsWeights."""

    def test_upper_bound_ge_actual(self):
        """Upper bound >= actual fused probability for all candidates."""
        rng = np.random.default_rng(42)
        attn = AttentionLogOddsWeights(n_signals=3, n_query_features=2)
        m = 50
        probs = rng.uniform(0.1, 0.9, size=(m, 3))
        qf = rng.standard_normal(size=(m, 2))

        # Use probs as its own upper bound (conservative but valid)
        upper_bounds = attn.compute_upper_bounds(probs, qf)
        actual = attn(probs, qf)

        # Upper bound should be >= actual for every candidate
        for i in range(m):
            assert upper_bounds[i] >= actual[i] - 1e-10, (
                f"idx={i}: ub={upper_bounds[i]}, actual={actual[i]}"
            )

    def test_pruned_below_threshold(self):
        """Pruned candidates all have actual probability below threshold."""
        rng = np.random.default_rng(42)
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        m = 50
        probs = rng.uniform(0.1, 0.9, size=(m, 2))
        qf = rng.standard_normal(size=(1, 2))

        actual = attn(probs, qf)
        threshold = float(np.median(actual))

        surviving_idx, fused = attn.prune(probs, qf, threshold)

        # All surviving candidates should have actual >= threshold (approximately)
        pruned_idx = np.setdiff1d(np.arange(m), surviving_idx)
        if len(pruned_idx) > 0:
            # Using probs as upper bound (same as actual), so prune is exact
            pruned_actual = actual[pruned_idx]
            assert np.all(pruned_actual < threshold + 1e-10)

    def test_surviving_includes_above_threshold(self):
        """Surviving candidates include all actually above-threshold documents."""
        rng = np.random.default_rng(42)
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        m = 50
        probs = rng.uniform(0.1, 0.9, size=(m, 2))
        # Use higher upper bounds to make pruning conservative
        upper_probs = np.clip(probs + 0.1, 0.0, 1.0)
        qf = rng.standard_normal(size=(1, 2))

        actual = attn(probs, qf)
        threshold = float(np.median(actual))

        surviving_idx, _ = attn.prune(probs, qf, threshold, upper_bound_probs=upper_probs)

        # Every doc with actual >= threshold must be in surviving
        truly_above = np.where(actual >= threshold)[0]
        for idx in truly_above:
            assert idx in surviving_idx, (
                f"Doc {idx} with actual={actual[idx]} >= {threshold} was pruned"
            )

    def test_empty_when_all_below(self):
        """Empty result when all below threshold."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        probs = np.array([[0.1, 0.1], [0.2, 0.2]])
        qf = np.array([1.0, 0.0])
        surviving_idx, fused = attn.prune(probs, qf, threshold=0.99)
        assert len(surviving_idx) == 0
        assert len(fused) == 0

    def test_no_pruning_when_all_above(self):
        """No pruning when all above threshold."""
        attn = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        probs = np.array([[0.9, 0.9], [0.8, 0.8], [0.85, 0.85]])
        qf = np.array([1.0, 0.0])
        surviving_idx, fused = attn.prune(probs, qf, threshold=0.01)
        assert len(surviving_idx) == 3
        assert len(fused) == 3


class TestMultiHeadPruning:
    """Tests for pruning on MultiHeadAttentionLogOddsWeights."""

    def test_upper_bound_ge_actual(self):
        """Multi-head upper bound >= actual for all candidates."""
        rng = np.random.default_rng(42)
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=3, n_signals=2, n_query_features=2
        )
        m = 30
        probs = rng.uniform(0.1, 0.9, size=(m, 2))
        qf = rng.standard_normal(size=(m, 2))

        upper_bounds = mh.compute_upper_bounds(probs, qf)
        actual = mh(probs, qf)

        for i in range(m):
            assert upper_bounds[i] >= actual[i] - 1e-10

    def test_prune_returns_correct_shapes(self):
        """prune() returns correct shapes."""
        mh = MultiHeadAttentionLogOddsWeights(
            n_heads=2, n_signals=2, n_query_features=2
        )
        probs = np.array([[0.8, 0.7], [0.3, 0.9], [0.5, 0.5]])
        qf = np.array([1.0, 0.0])
        surviving_idx, fused = mh.prune(probs, qf, threshold=0.3)
        assert surviving_idx.ndim == 1
        assert fused.ndim == 1
        assert len(surviving_idx) == len(fused)
