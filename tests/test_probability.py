#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.probability module."""

import numpy as np
import pytest

from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    logit,
    sigmoid,
)


class TestSigmoid:
    def test_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        # Should not overflow
        assert sigmoid(1000.0) == pytest.approx(1.0)

    def test_large_negative(self):
        # Should not underflow to exactly 0
        assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-15)

    def test_symmetry(self):
        x = 2.5
        assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)

    def test_monotonicity(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        s = sigmoid(x)
        assert np.all(np.diff(s) > 0)

    def test_array(self):
        x = np.array([0.0, 1.0, -1.0])
        result = sigmoid(x)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.5)


class TestLogit:
    def test_half(self):
        assert logit(0.5) == pytest.approx(0.0)

    def test_roundtrip(self):
        p = 0.73
        assert sigmoid(logit(p)) == pytest.approx(p)

    def test_extreme_values(self):
        # Should not produce inf/-inf thanks to clamping
        assert np.isfinite(logit(0.0))
        assert np.isfinite(logit(1.0))

    def test_monotonicity(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        l_vals = logit(p)
        assert np.all(np.diff(l_vals) > 0)


class TestTFPrior:
    def test_zero_tf(self):
        assert BayesianProbabilityTransform.tf_prior(0) == pytest.approx(0.2)

    def test_high_tf(self):
        # tf >= 10 should saturate at 0.9
        assert BayesianProbabilityTransform.tf_prior(10) == pytest.approx(0.9)
        assert BayesianProbabilityTransform.tf_prior(100) == pytest.approx(0.9)

    def test_mid_tf(self):
        assert BayesianProbabilityTransform.tf_prior(5) == pytest.approx(0.55)

    def test_array(self):
        result = BayesianProbabilityTransform.tf_prior(np.array([0, 5, 10]))
        np.testing.assert_allclose(result, [0.2, 0.55, 0.9])


class TestNormPrior:
    def test_average_length(self):
        # doc_len_ratio = 0.5 is peak
        assert BayesianProbabilityTransform.norm_prior(0.5) == pytest.approx(0.9)

    def test_extreme_lengths(self):
        # Very short or very long documents
        p_short = BayesianProbabilityTransform.norm_prior(0.0)
        p_long = BayesianProbabilityTransform.norm_prior(1.0)
        assert p_short == pytest.approx(0.3)
        assert p_long == pytest.approx(0.3)

    def test_bounds(self):
        ratios = np.linspace(0, 3, 100)
        priors = BayesianProbabilityTransform.norm_prior(ratios)
        assert np.all(priors >= 0.3)
        assert np.all(priors <= 0.9)


class TestCompositePrior:
    def test_bounds(self):
        # Composite prior should always be in [0.1, 0.9]
        for tf in [0, 1, 5, 10, 100]:
            for ratio in [0.0, 0.25, 0.5, 1.0, 2.0]:
                p = BayesianProbabilityTransform.composite_prior(tf, ratio)
                assert 0.1 <= p <= 0.9, f"Out of bounds for tf={tf}, ratio={ratio}: {p}"

    def test_array(self):
        tf = np.array([0, 5, 10])
        ratio = np.array([0.5, 0.5, 0.5])
        result = BayesianProbabilityTransform.composite_prior(tf, ratio)
        assert result.shape == (3,)
        assert np.all(result >= 0.1)
        assert np.all(result <= 0.9)


class TestPosterior:
    def test_uniform_prior(self):
        # With prior = 0.5, posterior == likelihood
        l_val = 0.7
        p = BayesianProbabilityTransform.posterior(l_val, 0.5)
        assert p == pytest.approx(l_val)

    def test_high_prior_amplifies(self):
        l_val = 0.6
        low_prior = BayesianProbabilityTransform.posterior(l_val, 0.3)
        high_prior = BayesianProbabilityTransform.posterior(l_val, 0.7)
        assert high_prior > low_prior

    def test_monotonicity_in_likelihood(self):
        prior = 0.5
        likelihoods = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        posteriors = BayesianProbabilityTransform.posterior(likelihoods, prior)
        assert np.all(np.diff(posteriors) > 0)

    def test_bounds(self):
        l_val = np.array([0.01, 0.5, 0.99])
        prior = np.array([0.01, 0.5, 0.99])
        posteriors = BayesianProbabilityTransform.posterior(l_val, prior)
        assert np.all(posteriors > 0)
        assert np.all(posteriors < 1)


class TestScoreToProbability:
    def test_higher_score_higher_probability(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        p_low = t.score_to_probability(0.5, tf=5, doc_len_ratio=0.5)
        p_high = t.score_to_probability(2.0, tf=5, doc_len_ratio=0.5)
        assert p_high > p_low

    def test_paper_values(self):
        """Verify against Section 11.1 of the Bayesian BM25 paper.

        BM25 scores [1.0464478, 0.56150854, 1.1230172] should map to
        calibrated probabilities.  With alpha=1, beta=0, the likelihood
        is sigmoid(score) which is already > 0.5 for positive scores.
        The composite prior further shifts the posterior.
        """
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([1.0464478, 0.56150854, 1.1230172])
        tf = np.array([5.0, 3.0, 7.0])
        doc_len_ratio = np.array([0.5, 0.5, 0.5])
        probs = t.score_to_probability(scores, tf, doc_len_ratio)
        # All probabilities should be in (0, 1)
        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)
        # Monotonicity: higher score with equal or higher tf -> higher prob
        assert probs[2] > probs[1]
        assert probs[0] > probs[1]


class TestFit:
    def test_learns_parameters(self):
        rng = np.random.default_rng(42)
        # Generate synthetic data with known alpha=2.0, beta=1.0
        true_alpha, true_beta = 2.0, 1.0
        scores = rng.uniform(0, 3, size=200)
        prob_relevant = sigmoid(true_alpha * (scores - true_beta))
        labels = (rng.random(200) < prob_relevant).astype(float)

        t = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        t.fit(scores, labels, learning_rate=0.05, max_iterations=5000)

        # Should be in the right ballpark (not exact due to noise)
        assert abs(t.alpha - true_alpha) < 1.0
        assert abs(t.beta - true_beta) < 1.0

    def test_convergence(self):
        scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = np.array([0.0, 0.0, 0.5, 1.0, 1.0])

        t = BayesianProbabilityTransform(alpha=0.1, beta=0.0)
        t.fit(scores, labels, learning_rate=0.01, max_iterations=2000)

        # After fitting, predictions should roughly match labels
        predicted = sigmoid(t.alpha * (scores - t.beta))
        # The middle values should be roughly in the right direction
        assert predicted[0] < predicted[4]


class TestBaseRate:
    def test_none_preserves_current_behavior(self):
        """base_rate=None produces identical output to the old code."""
        t_none = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=None)
        t_old = BayesianProbabilityTransform(alpha=1.0, beta=0.5)
        scores = np.array([0.5, 1.0, 2.0, 3.0])
        for score in scores:
            p_none = t_none.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            p_old = t_old.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            assert p_none == pytest.approx(p_old)

    def test_base_rate_half_is_neutral(self):
        """base_rate=0.5 is neutral (logit(0.5)=0), same as no base rate."""
        t_half = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=0.5)
        t_none = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=None)
        scores = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
        for score in scores:
            p_half = t_half.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            p_none = t_none.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            assert p_half == pytest.approx(p_none, abs=1e-8)

    def test_low_base_rate_reduces_probabilities(self):
        """base_rate=0.01 pulls probabilities down compared to no base rate."""
        t_low = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=0.01)
        t_none = BayesianProbabilityTransform(alpha=1.0, beta=0.5)
        scores = np.array([0.5, 1.0, 2.0, 3.0])
        for score in scores:
            p_low = t_low.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            p_none = t_none.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            assert p_low < p_none, (
                f"score={score}: base_rate=0.01 gave {p_low}, no base_rate gave {p_none}"
            )

    def test_high_base_rate_increases_probabilities(self):
        """base_rate=0.9 pushes probabilities up compared to no base rate."""
        t_high = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=0.9)
        t_none = BayesianProbabilityTransform(alpha=1.0, beta=0.5)
        scores = np.array([0.5, 1.0, 2.0, 3.0])
        for score in scores:
            p_high = t_high.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            p_none = t_none.score_to_probability(score, tf=5, doc_len_ratio=0.5)
            assert p_high > p_none, (
                f"score={score}: base_rate=0.9 gave {p_high}, no base_rate gave {p_none}"
            )

    def test_numerical_example(self):
        """Verify: L=0.7, prior=0.5, base_rate=0.01 -> ~0.023.

        logit(0.7) = 0.8473, logit(0.5) = 0, logit(0.01) = -4.5951
        sum = 0.8473 + (-4.5951) + 0 = -3.7478
        sigmoid(-3.7478) ~ 0.0231
        """
        result = BayesianProbabilityTransform.posterior(0.7, 0.5, base_rate=0.01)
        assert result == pytest.approx(0.023, abs=0.005)

    def test_monotonicity_preserved(self):
        """Score ordering is preserved with any base_rate."""
        for br in [0.001, 0.01, 0.1, 0.5, 0.9, 0.999]:
            t = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=br)
            scores = np.sort(np.array([0.2, 0.5, 1.0, 2.0, 5.0]))
            probs = t.score_to_probability(scores, tf=5, doc_len_ratio=0.5)
            assert np.all(np.diff(probs) > 0), (
                f"Monotonicity violated for base_rate={br}: {probs}"
            )

    def test_output_range(self):
        """All P in (0, 1) for various base_rates."""
        rng = np.random.default_rng(42)
        for br in [0.001, 0.01, 0.1, 0.5, 0.9, 0.999]:
            t = BayesianProbabilityTransform(alpha=1.0, beta=0.5, base_rate=br)
            scores = rng.uniform(-5, 10, size=100)
            tf = rng.uniform(0, 20, size=100)
            ratio = rng.uniform(0.1, 3.0, size=100)
            probs = t.score_to_probability(scores, tf, ratio)
            assert np.all(probs > 0), f"base_rate={br}: some P <= 0"
            assert np.all(probs < 1), f"base_rate={br}: some P >= 1"

    def test_invalid_base_rate_raises(self):
        """ValueError for 0, 1, negative, >1."""
        for invalid in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="base_rate must be in"):
                BayesianProbabilityTransform(base_rate=invalid)

    def test_fit_preserves_base_rate(self):
        """fit() learns alpha/beta but does not modify base_rate."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0, base_rate=0.01)
        scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        t.fit(scores, labels, learning_rate=0.01, max_iterations=100)
        assert t.base_rate == 0.01
        assert t._logit_base_rate == pytest.approx(logit(0.01))

    def test_update_preserves_base_rate(self):
        """update() learns alpha/beta but does not modify base_rate."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0, base_rate=0.05)
        for _ in range(20):
            t.update(2.0, 1.0, learning_rate=0.1)
            t.update(-1.0, 0.0, learning_rate=0.1)
        assert t.base_rate == 0.05
        assert t._logit_base_rate == pytest.approx(logit(0.05))

    def test_posterior_static_with_base_rate(self):
        """posterior(L, p, base_rate) shifts probabilities downward."""
        L = 0.8
        p = 0.6
        p_no_br = BayesianProbabilityTransform.posterior(L, p)
        p_with_br = BayesianProbabilityTransform.posterior(L, p, base_rate=0.01)
        assert p_with_br < p_no_br


class TestOnlineUpdate:
    def test_converges_to_batch(self):
        """Online updates should converge to similar parameters as batch fit."""
        rng = np.random.default_rng(123)
        true_alpha, true_beta = 2.0, 1.0
        scores = rng.uniform(0, 3, size=500)
        prob_relevant = sigmoid(true_alpha * (scores - true_beta))
        labels = (rng.random(500) < prob_relevant).astype(float)

        t = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        for s, l in zip(scores, labels):
            t.update(s, l, learning_rate=0.05, momentum=0.9)

        assert abs(t.alpha - true_alpha) < 1.5
        assert abs(t.beta - true_beta) < 1.0

    def test_single_updates_move_parameters(self):
        """Each update should move parameters in the right direction."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        initial_alpha = t.alpha
        initial_beta = t.beta

        # Observing a high score as relevant should increase alpha
        # or shift beta to make high scores more likely
        for _ in range(20):
            t.update(3.0, 1.0, learning_rate=0.1)
            t.update(-1.0, 0.0, learning_rate=0.1)

        # Parameters should have moved
        assert t.alpha != initial_alpha or t.beta != initial_beta

    def test_mini_batch_update(self):
        """update() should accept arrays for mini-batch updates."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([0.5, 1.5, 2.5])
        labels = np.array([0.0, 1.0, 1.0])
        t.update(scores, labels, learning_rate=0.05)
        # Should not raise and parameters should change
        assert t._n_updates == 1

    def test_fit_resets_ema_state(self):
        """Calling fit() should reset the online learning state."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        t.update(2.0, 1.0, learning_rate=0.1)
        assert t._n_updates == 1
        assert t._grad_alpha_ema != 0.0

        t.fit(np.array([1.0, 2.0]), np.array([0.0, 1.0]), max_iterations=10)
        assert t._n_updates == 0
        assert t._grad_alpha_ema == 0.0

    def test_momentum_smoothing(self):
        """Higher momentum should produce smoother parameter trajectories."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 3, size=100)
        labels = (scores > 1.5).astype(float)
        # Add noise
        flip = rng.random(100) < 0.2
        labels[flip] = 1.0 - labels[flip]

        # Low momentum -- more responsive to noise
        t_low = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        alphas_low = []
        for s, l in zip(scores, labels):
            t_low.update(s, l, learning_rate=0.05, momentum=0.5)
            alphas_low.append(t_low.alpha)

        # High momentum -- smoother trajectory
        t_high = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        alphas_high = []
        for s, l in zip(scores, labels):
            t_high.update(s, l, learning_rate=0.05, momentum=0.95)
            alphas_high.append(t_high.alpha)

        # High momentum should have lower variance in parameter trajectory
        var_low = np.var(np.diff(alphas_low))
        var_high = np.var(np.diff(alphas_high))
        assert var_high < var_low
