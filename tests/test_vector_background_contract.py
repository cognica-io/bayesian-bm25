#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Density roles, retrieved-range recall, and actual posterior evidence."""

import math

import numpy as np
import pytest

from bayesian_bm25 import VectorProbabilityTransform
from bayesian_bm25.probability import sigmoid


class TestVectorBackgroundContract:
    def test_a_fitted_population_sample_is_a_marginal_kde(self):
        distances = np.linspace(0.1, 1.0, 100)
        model = VectorProbabilityTransform.fit_background(distances, base_rate=0.1)
        assert model.background_kind == "marginal"
        assert model.background_model == "kde"
        logits = model.calibrate(
            distances, weights=np.ones(100), method="kde", return_log_odds=True,
        )
        # Equal relevant and marginal shapes contain no information.
        np.testing.assert_allclose(sigmoid(logits), 0.1, atol=1e-12)
        np.testing.assert_allclose(
            model.evidence(distances, weights=np.ones(100), method="kde"), 0.0, atol=1e-12,
        )

    def test_exact_marginal_posterior_includes_retrieved_relevance_recall(self):
        distances = np.linspace(0.1, 1.0, 100)
        model = VectorProbabilityTransform.fit_background(distances, base_rate=0.1)
        full = model.calibrate(distances, weights=np.ones(100), method="kde", return_log_odds=True)
        partial = model.calibrate(
            distances, weights=np.ones(100), method="kde", relevance_recall=0.25,
            return_log_odds=True,
        )
        np.testing.assert_allclose(sigmoid(partial), 0.25 * sigmoid(full), atol=1e-12)

    def test_nonrelevant_background_adds_the_recall_log_factor(self):
        distances = np.linspace(0.2, 0.9, 40)
        weights = np.linspace(0.9, 0.1, 40)
        model = VectorProbabilityTransform(0.8, 0.2, base_rate=0.01, background_kind="nonrelevant")
        full = model.evidence(distances, weights=weights, method="kde")
        partial = model.evidence(distances, weights=weights, method="kde", relevance_recall=0.2)
        np.testing.assert_allclose(partial - full, math.log(0.2))

    def test_exact_formula_requires_a_marginal_background(self):
        model = VectorProbabilityTransform(0.8, 0.2, base_rate=0.01)
        with pytest.raises(ValueError, match="requires a marginal background"):
            model.calibrate(np.array([0.2, 0.3]), posterior="exact")

    def test_separate_sample_evidence_uses_the_actual_posterior(self):
        sample = np.linspace(0.2, 0.8, 40)
        model = VectorProbabilityTransform.fit_background(
            np.linspace(0.0, 1.0, 100), base_rate=0.01,
        )
        points = np.array([0.25, 0.4, 0.7])
        options = dict(weights=np.linspace(0.9, 0.1, 40), method="kde", relevance_recall=0.8)
        evidence = model.evidence_with_sample(points, sample, **options)
        posterior = model.calibrate_with_sample(points, sample, return_log_odds=True, **options)
        np.testing.assert_allclose(evidence + math.log(0.01 / 0.99), posterior)

    def test_rare_priors_are_not_epsilon_clamped(self):
        prior = 1e-14
        model = VectorProbabilityTransform(0.8, 0.2, base_rate=prior)
        assert model._logit_base_rate == pytest.approx(math.log(prior) - math.log1p(-prior))

    def test_scalar_marginal_density_ratio_is_supported(self):
        model = VectorProbabilityTransform.fit_background(np.linspace(0.2, 0.9, 40))
        assert isinstance(model.log_density_ratio(0.5, 1.0), float)

    @pytest.mark.parametrize("recall", [0.0, -0.1, 1.1, np.nan, np.inf])
    def test_invalid_recall_is_rejected(self, recall):
        model = VectorProbabilityTransform(0.8, 0.2)
        with pytest.raises(ValueError, match="relevance_recall"):
            model.calibrate(np.array([0.2, 0.3]), relevance_recall=recall)
