#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Prior-once fusion, signed coefficients, and end-to-end model semantics."""

import math

import numpy as np
import pytest

from bayesian_bm25 import (
    AttentionLogOddsWeights,
    BayesianProbabilityTransform,
    LearnableLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    evidence_upper_bound,
    fuse_evidence,
    gaussian_evidence_weights,
    log_odds_fusion,
    posterior_to_evidence,
)
from bayesian_bm25.debug import FusionDebugger
from bayesian_bm25.multi_field import MultiFieldScorer
from bayesian_bm25.probability import sigmoid


class TestEvidenceContract:
    def test_inactive_opinion_does_not_turn_infinite_input_into_nan(self):
        values = log_odds_fusion(np.array([np.inf, 2.0]), weights=np.array([0.0, 1.0]))
        assert values == 2.0
        with pytest.raises(ValueError, match="finite non-negative"):
            log_odds_fusion(np.array([1.0, 2.0]), weights=np.array([np.nan, 1.0]))

    def test_source_priors_are_removed_and_target_prior_is_added_once(self):
        prior = math.log(0.001 / 0.999)
        logits = np.full(2, math.log(0.2 / 0.8))
        evidence = posterior_to_evidence(logits, prior)
        exact = fuse_evidence(evidence, prior_logit=prior)
        pooled = log_odds_fusion(
            logits, alpha=0.5, input_prior_log_odds=prior, prior_logit=prior,
        )
        assert sigmoid(exact) == pytest.approx(0.9842364532)
        assert sigmoid(pooled) == pytest.approx(0.7110224938)
        assert sigmoid(log_odds_fusion(logits, alpha=0.5)) < 0.2

    def test_source_priors_require_an_explicit_target(self):
        with pytest.raises(ValueError, match="explicit target"):
            log_odds_fusion(np.ones(2), input_prior_log_odds=-5.0)
        with pytest.raises(ValueError, match="explicit target"):
            LearnableLogOddsWeights(2, input_prior_log_odds=-5.0)

    def test_gaussian_coefficients_are_used_without_normalization(self):
        coefficients = gaussian_evidence_weights(np.array([2.0, 0.5]), 0.8)
        assert coefficients[1] < 0.0
        evidence = np.array([[1.0, 2.0], [3.0, -1.0]])
        np.testing.assert_allclose(
            fuse_evidence(evidence, coefficients=coefficients, prior_logit=-7.0),
            evidence @ coefficients - 7.0,
        )

    def test_signed_pruning_uses_lower_bounds_for_negative_coefficients(self):
        rng = np.random.default_rng(42)
        lower = np.array([-2.0, -4.0, -1.0])
        upper = np.array([5.0, 2.0, 3.0])
        coefficients = np.array([2.0, -3.0, 0.0])
        bound = evidence_upper_bound(lower, upper, coefficients=coefficients, prior_logit=-6.0)
        samples = rng.uniform(lower, upper, size=(10000, 3))
        actual = fuse_evidence(samples, coefficients=coefficients, prior_logit=-6.0)
        assert np.all(actual <= bound)
        assert bound == pytest.approx(16.0)

    def test_zero_coefficient_ignores_an_infinite_input(self):
        assert fuse_evidence(
            np.array([2.0, -np.inf]), coefficients=np.array([1.0, 0.0]), prior_logit=-4.0,
        ) == -2.0

    @pytest.mark.parametrize("kind", ["learnable", "attention", "multi_head"])
    def test_learned_models_share_the_evidence_contract(self, kind):
        prior = math.log(0.001 / 0.999)
        options = dict(alpha=0.5, base_rate=0.001, input_prior_log_odds=prior)
        logits = np.full(2, math.log(0.2 / 0.8))
        if kind == "learnable":
            model = LearnableLogOddsWeights(2, **options)
            actual = model(logits, from_log_odds=True, return_log_odds=True)
        elif kind == "attention":
            model = AttentionLogOddsWeights(2, 1, **options)
            actual = model(logits, np.zeros(1), from_log_odds=True, return_log_odds=True)
        else:
            model = MultiHeadAttentionLogOddsWeights(3, 2, 1, **options)
            actual = model(logits, np.zeros(1), from_log_odds=True, return_log_odds=True)
        assert sigmoid(actual) == pytest.approx(0.7110224938)

    def test_fit_and_online_update_use_the_same_source_priors(self):
        rng = np.random.default_rng(0)
        prior = np.array([-6.0, -5.0])
        logits = rng.normal(size=(30, 2))
        labels = rng.integers(0, 2, size=30)
        posterior_model = LearnableLogOddsWeights(2, base_rate=0.01, input_prior_log_odds=prior)
        evidence_model = LearnableLogOddsWeights(2, base_rate=0.01)
        posterior_model.fit(logits, labels, from_log_odds=True, max_iterations=20)
        evidence_model.fit(logits - prior, labels, from_log_odds=True, max_iterations=20)
        posterior_model.update(logits[:2], labels[:2], from_log_odds=True)
        evidence_model.update(logits[:2] - prior, labels[:2], from_log_odds=True)
        np.testing.assert_allclose(posterior_model.weights, evidence_model.weights)

    def test_debugger_records_evidence_and_target_prior(self):
        prior = math.log(0.001 / 0.999)
        debugger = FusionDebugger(BayesianProbabilityTransform())
        trace = debugger.trace_fusion(
            np.full(2, math.log(0.2 / 0.8)), from_log_odds=True,
            input_prior_log_odds=prior, prior_logit=prior, alpha=0.5,
        )
        assert trace.fused_probability == pytest.approx(0.7110224938)
        assert trace.prior_logit == prior
        assert trace.input_prior_log_odds == [prior, prior]
        assert sigmoid(trace.posterior_log_odds) == pytest.approx(trace.fused_probability)

    def test_query_multi_field_constrains_the_fused_population_count(self):
        scorer = MultiFieldScorer(
            ["title", "body"], method="lucene", calibration="query",
            expected_relevant=1.2, ceiling=0.4,
        )
        scorer.index([
            {"title": ["foo"], "body": ["foo", "foo"]},
            {"title": ["bar"], "body": ["foo", "bar"]},
            {"title": ["foo", "bar"], "body": ["baz"]},
            {"title": ["baz"], "body": ["bar", "bar"]},
        ], show_progress=False)
        probabilities = scorer.get_probabilities(["foo"])
        assert np.sum(probabilities) == pytest.approx(1.2)
        assert np.all(probabilities <= 0.4)
        np.testing.assert_allclose(sigmoid(scorer.get_log_odds(["foo"])), probabilities)

    def test_normalized_attention_bounds_and_pruning_preserve_scores(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(100, 2))
        upper = logits + rng.exponential(size=(100, 2))
        features = rng.normal(size=(100, 3))
        model = AttentionLogOddsWeights(2, 3, normalize=True)
        actual = model(logits, features, from_log_odds=True, return_log_odds=True)
        bounds = model.compute_upper_bounds(
            upper, features, from_log_odds=True, return_log_odds=True,
        )
        assert np.all(actual <= bounds + 1e-12)
        indices, survived = model.prune(
            logits, features, 0.5, upper, from_log_odds=True, return_log_odds=True,
        )
        assert set(np.flatnonzero(actual >= 0.5)).issubset(indices)
        np.testing.assert_allclose(survived, actual[indices])
