#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for the log-odds fusion functions and the Theorem 5.8 helpers."""

import numpy as np
import pytest

from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    LearnableLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    balanced_log_odds_fusion,
    confidence_exponent,
    effective_signal_count,
    evidence_correlation,
    gaussian_evidence_weights,
    log_odds_and,
    log_odds_conjunction,
    log_odds_fusion,
    log_odds_not,
    log_odds_or,
    prob_and,
    prob_not,
    prob_or,
)
from bayesian_bm25.probability import logit, sigmoid


class TestLogOddsFusion:
    def test_matches_conjunction_for_moderate_evidence(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0.05, 0.95, size=(50, 3))
        np.testing.assert_allclose(
            sigmoid(log_odds_fusion(logit(probs))), log_odds_conjunction(probs),
        )

    def test_matches_weighted_conjunction(self):
        rng = np.random.default_rng(1)
        probs = rng.uniform(0.05, 0.95, size=(50, 3))
        weights = np.array([0.5, 0.3, 0.2])
        np.testing.assert_allclose(
            sigmoid(log_odds_fusion(logit(probs), alpha=0.5, weights=weights)),
            log_odds_conjunction(probs, alpha=0.5, weights=weights),
        )

    @pytest.mark.parametrize("gating", ["relu", "swish", "gelu", "softplus"])
    def test_matches_gated_conjunction(self, gating):
        rng = np.random.default_rng(2)
        probs = rng.uniform(0.05, 0.95, size=(20, 2))
        np.testing.assert_allclose(
            sigmoid(log_odds_fusion(logit(probs), gating=gating)),
            log_odds_conjunction(probs, gating=gating),
        )

    def test_alpha_one_is_the_sum_of_evidence(self):
        logits = np.array([[1.5, -0.5, 2.0], [0.2, 0.3, -4.0]])
        np.testing.assert_allclose(
            log_odds_fusion(logits, alpha=1.0), np.sum(logits, axis=-1),
        )

    def test_alpha_zero_is_the_mean(self):
        logits = np.array([[1.5, -0.5, 2.0], [0.2, 0.3, -4.0]])
        np.testing.assert_allclose(
            log_odds_fusion(logits, alpha=0.0), np.mean(logits, axis=-1),
        )

    def test_prior_is_added_once(self):
        logits = np.array([1.0, 2.0])
        base = log_odds_fusion(logits, alpha=1.0)
        assert log_odds_fusion(logits, alpha=1.0, prior_logit=-7.0) == pytest.approx(
            base - 7.0
        )

    def test_does_not_saturate(self):
        # Evidence far beyond what a probability can resolve: the
        # probability-valued conjunction ties, the log-odds do not.
        logits = np.array([[60.0, 55.0], [61.0, 55.0], [900.0, 55.0]])
        fused = log_odds_fusion(logits)
        assert fused[0] < fused[1] < fused[2]
        probabilities = log_odds_conjunction(sigmoid(logits))
        assert probabilities[0] == probabilities[1] == probabilities[2]

    def test_max_logit_clips_inputs(self):
        fused = log_odds_fusion(np.array([100.0, -100.0]), alpha=1.0, max_logit=10.0)
        assert fused == pytest.approx(0.0)

    def test_scalar_output_for_a_single_document(self):
        assert isinstance(log_odds_fusion(np.array([0.5, 1.5])), float)

    def test_single_signal_passes_through(self):
        assert log_odds_fusion(np.array([2.5])) == pytest.approx(2.5)

    def test_rejects_weights_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            log_odds_fusion(np.array([1.0, 2.0]), weights=np.array([0.5, 0.2]))


class TestLogOddsBooleans:
    def test_not_is_negation(self):
        logits = np.array([-3.0, 0.0, 2.0])
        np.testing.assert_allclose(log_odds_not(logits), -logits)
        np.testing.assert_allclose(sigmoid(log_odds_not(logits)), prob_not(sigmoid(logits)))

    def test_and_matches_probability_rule(self):
        rng = np.random.default_rng(3)
        probs = rng.uniform(0.05, 0.95, size=(40, 3))
        np.testing.assert_allclose(
            sigmoid(log_odds_and(logit(probs))), prob_and(probs), rtol=1e-9,
        )

    def test_or_matches_probability_rule(self):
        rng = np.random.default_rng(4)
        probs = rng.uniform(0.05, 0.95, size=(40, 3))
        np.testing.assert_allclose(
            sigmoid(log_odds_or(logit(probs))), prob_or(probs), rtol=1e-9,
        )

    def test_or_stays_strictly_increasing_where_the_probability_rounds_to_one(self):
        logits = np.array([[40.0, -3.0], [41.0, -3.0], [400.0, -3.0]])
        fused = log_odds_or(logits)
        assert fused[0] < fused[1] < fused[2]
        assert np.all(prob_or(sigmoid(logits)) == prob_or(sigmoid(logits))[0])

    def test_and_stays_strictly_increasing_for_strong_evidence(self):
        logits = np.array([[40.0, 45.0], [41.0, 45.0], [300.0, 45.0]])
        fused = log_odds_and(logits)
        assert fused[0] < fused[1] < fused[2]

    def test_or_with_a_zero_probability_input(self):
        # A signal with probability 0 (log-odds -inf) leaves the other one.
        fused = log_odds_or(np.array([-np.inf, 1.25]))
        assert fused == pytest.approx(1.25)

    def test_or_of_tiny_probabilities_is_their_sum(self):
        fused = log_odds_or(np.array([-200.0, -200.0]))
        assert fused == pytest.approx(-200.0 + np.log(2.0), abs=1e-9)

    def test_de_morgan(self):
        logits = np.array([0.7, -1.3, 2.2])
        assert log_odds_not(log_odds_and(logits)) == pytest.approx(
            log_odds_or(log_odds_not(logits))
        )


class TestEffectiveSignalCount:
    def test_independent_signals_count_fully(self):
        assert effective_signal_count(5, 0.0) == pytest.approx(5.0)

    def test_redundant_signals_count_once(self):
        assert effective_signal_count(5, 1.0) == pytest.approx(1.0)

    def test_formula(self):
        assert effective_signal_count(4, 0.25) == pytest.approx(4.0 / 1.75)

    def test_matrix_input_agrees_with_equicorrelation(self):
        n, r = 4, 0.3
        omega = np.full((n, n), r)
        np.fill_diagonal(omega, 1.0)
        assert effective_signal_count(n, omega) == pytest.approx(
            effective_signal_count(n, r)
        )

    def test_single_signal(self):
        assert effective_signal_count(1, 0.9) == 1.0

    def test_rejects_invalid_correlation(self):
        with pytest.raises(ValueError, match="correlation must be in"):
            effective_signal_count(3, -0.5)
        with pytest.raises(ValueError, match="n_signals must be"):
            effective_signal_count(0, 0.1)


class TestConfidenceExponent:
    def test_endpoints(self):
        assert confidence_exponent(3, 0.0) == pytest.approx(1.0)
        assert confidence_exponent(3, 1.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [2, 3, 10, 50])
    def test_square_root_law(self, n):
        # rho = 1/2 is exact when r = 1 / (1 + sqrt(n)).
        r = 1.0 / (1.0 + np.sqrt(n))
        assert confidence_exponent(n, r) == pytest.approx(0.5)

    def test_square_root_law_correlations_quoted_in_the_paper(self):
        assert 1.0 / (1.0 + np.sqrt(2)) == pytest.approx(0.41, abs=0.005)
        assert 1.0 / (1.0 + np.sqrt(3)) == pytest.approx(0.37, abs=0.005)
        assert 1.0 / (1.0 + np.sqrt(10)) == pytest.approx(0.24, abs=0.005)

    def test_exponent_reproduces_effective_count(self):
        n, r = 6, 0.35
        assert n ** confidence_exponent(n, r) == pytest.approx(
            effective_signal_count(n, r)
        )

    def test_negative_correlation_exceeds_one(self):
        assert confidence_exponent(2, -0.2) > 1.0

    def test_undefined_for_a_single_signal(self):
        with pytest.raises(ValueError, match="at least two signals"):
            confidence_exponent(1, 0.0)


class TestGaussianEvidenceWeights:
    def test_independent_signals_have_unit_weights(self):
        np.testing.assert_allclose(
            gaussian_evidence_weights([1.0, 2.0, 0.5], 0.0), np.ones(3),
        )

    def test_equal_equicorrelated_signals(self):
        n, r = 4, 0.3
        np.testing.assert_allclose(
            gaussian_evidence_weights(np.full(n, 1.5), r),
            np.full(n, 1.0 / (1.0 + (n - 1) * r)),
        )

    def test_two_signal_closed_form(self):
        tau1, tau2, r = 2.0, 1.0, 0.3
        v = gaussian_evidence_weights([tau1, tau2], r)
        assert v[0] == pytest.approx((tau1 - r * tau2) / (tau1 * (1 - r * r)))
        assert v[1] == pytest.approx((tau2 - r * tau1) / (tau2 * (1 - r * r)))

    def test_weak_correlated_signal_gets_a_negative_weight(self):
        # v_2 < 0 exactly when r > tau_2 / tau_1.
        assert gaussian_evidence_weights([2.0, 1.0], 0.6)[1] < 0.0
        assert gaussian_evidence_weights([2.0, 1.0], 0.4)[1] > 0.0

    def test_recovers_the_joint_log_likelihood_ratio(self):
        # Theorem 5.8 (i)-(ii): evidence logits with class means
        # +/- tau^2 / 2 are valid log-likelihood ratios, and their joint
        # log-likelihood ratio is the weighted sum.
        rng = np.random.default_rng(5)
        tau = np.array([1.5, 0.8, 1.1])
        omega = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
        sigma = np.outer(tau, tau) * omega
        mean = 0.5 * tau**2
        e = rng.multivariate_normal(mean, sigma, size=200)

        inverse = np.linalg.inv(sigma)
        log_ratio = (
            -0.5 * np.einsum("ij,jk,ik->i", e - mean, inverse, e - mean)
            + 0.5 * np.einsum("ij,jk,ik->i", e + mean, inverse, e + mean)
        )
        np.testing.assert_allclose(
            e @ gaussian_evidence_weights(tau, omega), log_ratio, rtol=1e-9, atol=1e-9,
        )

    def test_each_evidence_is_its_own_log_likelihood_ratio(self):
        # With class means +/- tau^2 / 2 and variance tau^2 the
        # log-likelihood ratio of e is e itself.
        tau = 1.3
        e = np.linspace(-4.0, 4.0, 9)
        log_ratio = (
            -0.5 * ((e - 0.5 * tau**2) / tau) ** 2
            + 0.5 * ((e + 0.5 * tau**2) / tau) ** 2
        )
        np.testing.assert_allclose(log_ratio, e, atol=1e-12)

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="positive"):
            gaussian_evidence_weights([1.0, 0.0], 0.1)
        with pytest.raises(ValueError, match="positive definite"):
            gaussian_evidence_weights([1.0, 1.0], np.array([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="shape"):
            gaussian_evidence_weights([1.0, 1.0], np.eye(3))


class TestEvidenceCorrelation:
    def test_recovers_a_known_correlation(self):
        rng = np.random.default_rng(6)
        cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        x = rng.multivariate_normal([0.0, 0.0], cov, size=200000)
        corr = evidence_correlation(x)
        assert corr.shape == (2, 2)
        assert corr[0, 1] == pytest.approx(0.6, abs=0.01)

    def test_candidate_pool_induces_a_spurious_negative_correlation(self):
        # Two independent signals; a pool formed as the union of each
        # signal's top results selects documents on which at least one is
        # high, which correlates them negatively.
        rng = np.random.default_rng(7)
        x = rng.normal(size=(100000, 2))
        assert abs(evidence_correlation(x)[0, 1]) < 0.02
        top = np.union1d(np.argsort(-x[:, 0])[:1000], np.argsort(-x[:, 1])[:1000])
        assert evidence_correlation(x[top])[0, 1] < -0.5

    def test_rejects_constant_columns(self):
        with pytest.raises(ValueError, match="positive variance"):
            evidence_correlation(np.column_stack([np.ones(5), np.arange(5.0)]))

    def test_rejects_columns_that_are_constant_up_to_rounding(self):
        constant = np.full(3, 0.1)
        assert np.std(constant) > 0.0
        with pytest.raises(ValueError, match="positive variance"):
            evidence_correlation(np.column_stack([constant, np.arange(3.0)]))


class TestLogOddsInputsAndOutputs:
    def test_balanced_fusion_from_log_odds(self):
        rng = np.random.default_rng(8)
        sparse = rng.uniform(0.05, 0.95, size=30)
        dense = rng.uniform(-0.2, 0.9, size=30)
        np.testing.assert_allclose(
            balanced_log_odds_fusion(logit(sparse), dense, from_log_odds=True),
            balanced_log_odds_fusion(sparse, dense),
        )

    def test_balanced_fusion_keeps_strong_lexical_evidence_apart(self):
        # Beyond the probability clamp the lexical signal stops
        # discriminating; given as log-odds it still does.
        log_odds = np.array([30.0, 60.0, 90.0, -5.0])
        dense = np.array([0.5, 0.5, 0.5, 0.5])
        from_probabilities = balanced_log_odds_fusion(sigmoid(log_odds), dense)
        assert from_probabilities[0] == from_probabilities[1] == from_probabilities[2]
        direct = balanced_log_odds_fusion(log_odds, dense, from_log_odds=True)
        assert direct[0] < direct[1] < direct[2]

    def test_learnable_weights(self):
        rng = np.random.default_rng(9)
        probs = rng.uniform(0.05, 0.95, size=(25, 3))
        model = LearnableLogOddsWeights(n_signals=3, alpha=0.5)
        expected = model(probs)
        np.testing.assert_allclose(model(logit(probs), from_log_odds=True), expected)
        np.testing.assert_allclose(
            sigmoid(model(probs, return_log_odds=True)), expected,
        )

    def test_learnable_weights_fit_from_log_odds(self):
        rng = np.random.default_rng(10)
        probs = rng.uniform(0.05, 0.95, size=(200, 2))
        labels = (probs[:, 0] > 0.5).astype(float)
        a = LearnableLogOddsWeights(n_signals=2)
        a.fit(probs, labels, max_iterations=50)
        b = LearnableLogOddsWeights(n_signals=2)
        b.fit(logit(probs), labels, max_iterations=50, from_log_odds=True)
        np.testing.assert_allclose(a.weights, b.weights)

    def test_attention_weights(self):
        rng = np.random.default_rng(11)
        probs = rng.uniform(0.05, 0.95, size=(25, 2))
        features = rng.normal(size=3)
        model = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        expected = model(probs, features)
        np.testing.assert_allclose(
            model(logit(probs), features, from_log_odds=True), expected,
        )
        np.testing.assert_allclose(
            sigmoid(model(probs, features, return_log_odds=True)), expected,
        )

    def test_attention_fit_from_log_odds(self):
        rng = np.random.default_rng(12)
        probs = rng.uniform(0.05, 0.95, size=(120, 2))
        labels = (probs[:, 1] > 0.5).astype(float)
        features = rng.normal(size=(120, 3))
        a = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        a.fit(probs, labels, features, max_iterations=30)
        b = AttentionLogOddsWeights(n_signals=2, n_query_features=3)
        b.fit(logit(probs), labels, features, max_iterations=30, from_log_odds=True)
        np.testing.assert_allclose(a.weights_matrix, b.weights_matrix)

    def test_attention_ranks_strong_evidence_in_log_odds(self):
        model = AttentionLogOddsWeights(n_signals=2, n_query_features=1)
        log_odds = np.array([[50.0, 40.0], [70.0, 40.0], [90.0, 40.0]])
        features = np.array([1.0])
        fused = model(log_odds, features, from_log_odds=True, return_log_odds=True)
        assert fused[0] < fused[1] < fused[2]
        saturated = model(sigmoid(log_odds), features)
        assert saturated[0] == saturated[1] == saturated[2]

    def test_attention_pruning_in_log_odds(self):
        rng = np.random.default_rng(13)
        log_odds = rng.normal(0.0, 3.0, size=(40, 2))
        features = np.array([0.5, -0.5])
        model = AttentionLogOddsWeights(n_signals=2, n_query_features=2)
        survivors, fused = model.prune(
            log_odds, features, threshold=0.0,
            from_log_odds=True, return_log_odds=True,
        )
        full = model(log_odds, features, from_log_odds=True, return_log_odds=True)
        np.testing.assert_array_equal(survivors, np.flatnonzero(full >= 0.0))
        np.testing.assert_allclose(fused, full[survivors])

    def test_multi_head_attention(self):
        rng = np.random.default_rng(14)
        probs = rng.uniform(0.05, 0.95, size=(25, 2))
        features = rng.normal(size=3)
        model = MultiHeadAttentionLogOddsWeights(
            n_heads=3, n_signals=2, n_query_features=3,
        )
        expected = model(probs, features)
        np.testing.assert_allclose(
            model(logit(probs), features, from_log_odds=True), expected,
        )
        np.testing.assert_allclose(
            sigmoid(model(probs, features, return_log_odds=True)), expected,
        )
        bounds = model.compute_upper_bounds(probs, features)
        np.testing.assert_allclose(
            sigmoid(model.compute_upper_bounds(probs, features, return_log_odds=True)),
            bounds,
        )
