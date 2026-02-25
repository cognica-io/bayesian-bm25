#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.fusion module."""

import numpy as np
import pytest

from bayesian_bm25.fusion import (
    LearnableLogOddsWeights,
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_not,
    prob_or,
)


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
        from bayesian_bm25.probability import logit as _logit, sigmoid as _sigmoid, _clamp_probability

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
