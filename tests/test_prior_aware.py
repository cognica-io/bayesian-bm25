#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for prior-aware training modes (C1/C2/C3 conditions, Algorithm 8.3.1)."""

import numpy as np
import pytest

from bayesian_bm25.probability import BayesianProbabilityTransform, sigmoid


class TestBalancedMode:
    """Balanced mode (C1) should produce identical results to the original fit()."""

    def test_same_as_default(self):
        """Explicit mode='balanced' gives the same result as default."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 3, size=200)
        true_alpha, true_beta = 2.0, 1.0
        prob_relevant = sigmoid(true_alpha * (scores - true_beta))
        labels = (rng.random(200) < prob_relevant).astype(float)

        t_default = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        t_default.fit(scores, labels, learning_rate=0.05, max_iterations=3000)

        t_balanced = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        t_balanced.fit(
            scores, labels, learning_rate=0.05, max_iterations=3000,
            mode="balanced",
        )

        assert t_balanced.alpha == pytest.approx(t_default.alpha)
        assert t_balanced.beta == pytest.approx(t_default.beta)

    def test_training_mode_stored(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([0.0, 1.0, 2.0])
        labels = np.array([0.0, 0.5, 1.0])
        t.fit(scores, labels, max_iterations=10, mode="balanced")
        assert t._training_mode == "balanced"


class TestPriorAwareMode:
    """Prior-aware mode (C2): trains on the full Bayesian posterior."""

    def test_learns_with_prior(self):
        """prior_aware mode should learn alpha/beta that produce good posteriors."""
        rng = np.random.default_rng(42)
        n = 500
        scores = rng.uniform(0, 5, size=n)
        tfs = rng.uniform(0, 15, size=n)
        doc_len_ratios = rng.uniform(0.2, 2.0, size=n)

        # Generate labels from a known model
        true_alpha, true_beta = 1.5, 2.0
        likelihoods = sigmoid(true_alpha * (scores - true_beta))
        priors = BayesianProbabilityTransform.composite_prior(tfs, doc_len_ratios)
        posteriors = likelihoods * priors / (
            likelihoods * priors + (1.0 - likelihoods) * (1.0 - priors)
        )
        labels = (rng.random(n) < posteriors).astype(float)

        t = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        t.fit(
            scores, labels,
            learning_rate=0.01, max_iterations=5000,
            mode="prior_aware",
            tfs=tfs, doc_len_ratios=doc_len_ratios,
        )

        # Should learn parameters in the right ballpark
        assert abs(t.alpha - true_alpha) < 2.0
        assert abs(t.beta - true_beta) < 2.0
        assert t._training_mode == "prior_aware"

    def test_requires_tfs_and_doc_len_ratios(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([1.0, 2.0])
        labels = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="tfs and doc_len_ratios are required"):
            t.fit(scores, labels, mode="prior_aware")

        with pytest.raises(ValueError, match="tfs and doc_len_ratios are required"):
            t.fit(scores, labels, mode="prior_aware", tfs=np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="tfs and doc_len_ratios are required"):
            t.fit(
                scores, labels, mode="prior_aware",
                doc_len_ratios=np.array([0.5, 0.5]),
            )

    def test_posterior_used_as_prediction(self):
        """In prior_aware mode, the prediction includes the composite prior."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=1.0)

        # High TF and average doc length -> high prior -> boosted posterior
        scores = np.array([2.0, 2.0])
        labels = np.array([1.0, 1.0])
        tfs = np.array([10.0, 10.0])
        doc_len_ratios = np.array([0.5, 0.5])

        initial_alpha = t.alpha
        t.fit(
            scores, labels,
            learning_rate=0.01, max_iterations=100,
            mode="prior_aware",
            tfs=tfs, doc_len_ratios=doc_len_ratios,
        )
        # Parameters should have changed
        assert t.alpha != initial_alpha or t.beta != 1.0


class TestPriorFreeMode:
    """Prior-free mode (C3): trains like balanced, but inference uses prior=0.5."""

    def test_same_training_as_balanced(self):
        """prior_free trains identically to balanced (only inference differs)."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 3, size=200)
        true_alpha, true_beta = 2.0, 1.0
        prob_relevant = sigmoid(true_alpha * (scores - true_beta))
        labels = (rng.random(200) < prob_relevant).astype(float)

        t_balanced = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        t_balanced.fit(
            scores, labels, learning_rate=0.05, max_iterations=3000,
            mode="balanced",
        )

        t_free = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
        t_free.fit(
            scores, labels, learning_rate=0.05, max_iterations=3000,
            mode="prior_free",
        )

        # Same alpha/beta because training is identical
        assert t_free.alpha == pytest.approx(t_balanced.alpha)
        assert t_free.beta == pytest.approx(t_balanced.beta)

    def test_inference_uses_uniform_prior(self):
        """In prior_free mode, score_to_probability uses prior=0.5."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=1.0)
        t.fit(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            max_iterations=100,
            mode="prior_free",
        )

        # With prior=0.5, posterior = likelihood
        score = 2.5
        prob = t.score_to_probability(score, tf=10, doc_len_ratio=0.5)
        likelihood = t.likelihood(score)
        assert prob == pytest.approx(float(likelihood), abs=1e-8)

    def test_prior_free_vs_balanced_inference_differs(self):
        """prior_free and balanced produce different inference results."""
        scores = np.array([0.0, 1.0, 2.0, 3.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])

        t_balanced = BayesianProbabilityTransform(alpha=1.0, beta=1.0)
        t_balanced.fit(scores, labels, max_iterations=100, mode="balanced")

        t_free = BayesianProbabilityTransform(alpha=1.0, beta=1.0)
        t_free.fit(scores, labels, max_iterations=100, mode="prior_free")

        # Same parameters, different inference
        test_score = 2.5
        p_balanced = t_balanced.score_to_probability(test_score, tf=10, doc_len_ratio=0.5)
        p_free = t_free.score_to_probability(test_score, tf=10, doc_len_ratio=0.5)

        # With tf=10 and ratio=0.5, composite_prior > 0.5, so balanced > free
        assert p_balanced != pytest.approx(p_free, abs=1e-4)


class TestModeValidation:
    def test_invalid_mode_raises(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([1.0, 2.0])
        labels = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="mode must be one of"):
            t.fit(scores, labels, mode="invalid")

    def test_invalid_update_mode_raises(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        with pytest.raises(ValueError, match="mode must be one of"):
            t.update(1.0, 1.0, mode="invalid")


class TestOnlineUpdateModes:
    def test_update_prior_aware_requires_tf(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        with pytest.raises(ValueError, match="tf and doc_len_ratio are required"):
            t.update(1.0, 1.0, mode="prior_aware")

    def test_update_prior_aware(self):
        """Online prior_aware updates should move parameters."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        initial_alpha = t.alpha
        for _ in range(50):
            t.update(
                3.0, 1.0,
                learning_rate=0.1,
                mode="prior_aware",
                tf=5.0, doc_len_ratio=0.5,
            )
            t.update(
                -1.0, 0.0,
                learning_rate=0.1,
                mode="prior_aware",
                tf=1.0, doc_len_ratio=2.0,
            )
        assert t.alpha != initial_alpha
        assert t._training_mode == "prior_aware"

    def test_update_inherits_fit_mode(self):
        """update() without mode= uses the mode from the last fit()."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        t.fit(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.5, 1.0]),
            max_iterations=10,
            mode="prior_free",
        )
        assert t._training_mode == "prior_free"

        # update() without mode= should use prior_free (balanced training)
        t.update(2.0, 1.0, learning_rate=0.01)
        assert t._training_mode == "prior_free"

    def test_update_mode_override(self):
        """Explicit mode= in update() overrides the stored mode."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        t._training_mode = "balanced"
        t.update(
            2.0, 1.0,
            learning_rate=0.01,
            mode="prior_aware",
            tf=5.0, doc_len_ratio=0.5,
        )
        assert t._training_mode == "prior_aware"


class TestConvergenceComparison:
    """Compare convergence behaviour across training modes."""

    def test_all_modes_converge(self):
        """All three modes should produce reasonable predictions after training."""
        rng = np.random.default_rng(42)
        n = 300
        scores = rng.uniform(0, 4, size=n)
        tfs = rng.uniform(0, 15, size=n)
        doc_len_ratios = rng.uniform(0.2, 2.0, size=n)

        true_alpha, true_beta = 1.5, 2.0
        likelihoods = sigmoid(true_alpha * (scores - true_beta))
        labels = (rng.random(n) < likelihoods).astype(float)

        for mode_name in ["balanced", "prior_aware", "prior_free"]:
            t = BayesianProbabilityTransform(alpha=0.5, beta=0.0)
            kwargs = dict(
                learning_rate=0.01,
                max_iterations=5000,
                mode=mode_name,
            )
            if mode_name == "prior_aware":
                kwargs["tfs"] = tfs
                kwargs["doc_len_ratios"] = doc_len_ratios

            t.fit(scores, labels, **kwargs)

            # Alpha should be positive and parameters should have moved
            assert t.alpha > 0, f"mode={mode_name}: alpha={t.alpha}"
            assert t.alpha != 0.5 or t.beta != 0.0, (
                f"mode={mode_name}: parameters did not move"
            )
