#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for the log-odds path of VectorProbabilityTransform.

Covers the log-domain density ratio, the exact posterior and the
monotone projection of Remark 4.5, and the exclusion of duplicates of
the query (Remark 4.12).
"""

import math

import numpy as np
import pytest

from bayesian_bm25.probability import logit, sigmoid
from bayesian_bm25.vector_probability import (
    VectorProbabilityTransform,
    _gaussian_pdf,
    _kernel_density,
    _log_binned_kernel_density,
    _log_gaussian_pdf,
    _log_kernel_density,
)


def _candidates(seed: int = 0):
    """Three relevant documents near the query among 997 background ones.

    The relevant documents sit below the bulk, separated from it by a
    gap that is smaller than their own distance from the query.
    """
    rng = np.random.default_rng(seed)
    relevant = np.array([0.30, 0.33, 0.40])
    background = np.clip(rng.normal(0.80, 0.07, size=997), 0.58, 1.2)
    return np.concatenate([relevant, background])


class TestLogDensities:
    def test_log_gaussian_matches_density(self):
        x = np.linspace(-2.0, 3.0, 11)
        np.testing.assert_allclose(
            np.exp(_log_gaussian_pdf(x, 0.5, 0.7)), _gaussian_pdf(x, 0.5, 0.7),
        )

    def test_log_gaussian_is_exact_where_the_density_underflows(self):
        # 60 standard deviations out the density is 0 in double precision.
        assert _gaussian_pdf(60.0, 0.0, 1.0) == 0.0
        assert _log_gaussian_pdf(60.0, 0.0, 1.0) == pytest.approx(
            -0.5 * 3600.0 - 0.5 * math.log(2.0 * math.pi)
        )

    def test_log_kernel_density_matches_density(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(0.4, 0.1, size=200)
        weights = rng.uniform(0.1, 1.0, size=200)
        grid = np.linspace(0.0, 0.8, 50)
        np.testing.assert_allclose(
            np.exp(_log_kernel_density(grid, sample, weights, 0.05)),
            _kernel_density(grid, sample, weights, 0.05),
            rtol=1e-9,
        )

    def test_log_kernel_density_keeps_decreasing_far_from_the_sample(self):
        sample = np.array([0.2, 0.25, 0.3])
        weights = np.ones(3)
        far = _log_kernel_density(np.array([5.0, 6.0, 7.0]), sample, weights, 0.02)
        assert np.all(np.isfinite(far))
        assert far[0] > far[1] > far[2]

    def test_zero_weights_are_ignored(self):
        sample = np.array([0.2, 0.9])
        grid = np.array([0.2, 0.5])
        np.testing.assert_allclose(
            _log_kernel_density(grid, sample, np.array([1.0, 0.0]), 0.1),
            _log_kernel_density(grid, sample[:1], np.array([1.0]), 0.1),
        )

    def test_binned_density_approximates_the_exact_one(self):
        rng = np.random.default_rng(2)
        sample = rng.normal(0.8, 0.1, size=50000)
        grid = np.linspace(0.5, 1.1, 40)
        exact = _log_kernel_density(grid, sample, np.ones_like(sample), 0.02)
        binned = _log_binned_kernel_density(grid, sample, 0.02)
        np.testing.assert_allclose(binned, exact, atol=1e-3)


class TestLogOddsOutput:
    def test_log_odds_agree_with_probabilities(self):
        distances = _candidates()
        vct = VectorProbabilityTransform.fit_background(distances, base_rate=0.003)
        weights = np.where(distances < 0.35, 1.0, 0.01)
        probabilities = vct.calibrate(distances, weights=weights, method="kde")
        log_odds = vct.calibrate(
            distances, weights=weights, method="kde", return_log_odds=True,
        )
        unsaturated = np.abs(log_odds) < 20.0
        np.testing.assert_allclose(
            sigmoid(log_odds[unsaturated]), probabilities[unsaturated], rtol=1e-9,
        )

    def test_log_odds_do_not_saturate_near_the_query(self):
        # Close candidates lie far in the tail of the background, where a
        # probability is clamped; their log-odds keep their order.
        vct = VectorProbabilityTransform(mu_G=0.9, sigma_G=0.05, base_rate=0.01)
        sample = np.array([0.10, 0.12, 0.14, 0.16])
        eval_points = np.array([0.10, 0.20, 0.30])
        weights = np.ones(4)
        log_odds = vct.calibrate_with_sample(
            eval_points, sample, weights=weights, method="kde",
            return_log_odds=True,
        )
        probabilities = vct.calibrate_with_sample(
            eval_points, sample, weights=weights, method="kde",
        )
        assert len(np.unique(log_odds)) == 3
        assert probabilities[0] == probabilities[1] == probabilities[2]

    def test_scalar_input(self):
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1, base_rate=0.01)
        sample = np.array([0.2, 0.25, 0.3, 0.7, 0.8, 0.9])
        value = vct.calibrate_with_sample(0.25, sample, return_log_odds=True)
        assert isinstance(value, float)

    def test_log_density_ratio_is_exact_in_the_background_tail(self):
        vct = VectorProbabilityTransform(mu_G=1.0, sigma_G=0.05)
        # 15 sigma from the background mean: f_G underflows the old floor.
        near = vct.log_density_ratio(np.array([0.25, 0.20]), np.array([1.0, 1.0]))
        assert near[1] > near[0]
        assert near[0] == pytest.approx(
            -float(_log_gaussian_pdf(0.25, 1.0, 0.05)), rel=1e-12,
        )


class TestExactPosterior:
    def test_formula(self):
        # P = pi * f_R / f_G  =>  logit P = e + log(pi) - log(1 - pi * exp(e)).
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1, base_rate=0.02)
        evidence = np.array([-2.0, 0.0, 1.5, 3.0])
        exact = vct._posterior_log_odds(evidence, "exact")
        expected = evidence + math.log(0.02) - np.log1p(-0.02 * np.exp(evidence))
        np.testing.assert_allclose(exact, expected, rtol=1e-12)

    def test_correction_to_the_approximate_form(self):
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1, base_rate=0.02)
        evidence = np.array([-3.0, 0.0, 2.0, 3.5])
        exact = vct._posterior_log_odds(evidence, "exact")
        approximate = vct._posterior_log_odds(evidence, "approximate")
        correction = np.log((1.0 - 0.02) / (1.0 - 0.02 * np.exp(evidence)))
        np.testing.assert_allclose(exact - approximate, correction, rtol=1e-10)

    def test_approximate_form_is_capped_when_the_background_is_the_marginal(self):
        # f_R / f_G <= 1 / pi, so sigmoid(e + logit(pi)) <= 1 / (2 - pi).
        prevalence = 0.01
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1, base_rate=prevalence)
        largest = math.log(1.0 / prevalence)
        cap = float(sigmoid(vct._posterior_log_odds(np.array([largest]), "approximate")[0]))
        assert cap == pytest.approx(1.0 / (2.0 - prevalence))

    def test_exact_posterior_averages_to_the_prevalence(self):
        # With f_G the marginal of the scored population, the exact
        # posterior satisfies marginal consistency by construction; the
        # approximate one averages to less.
        rng = np.random.default_rng(3)
        prevalence = 0.05
        n = 200000
        relevant = rng.random(n) < prevalence
        distances = np.where(
            relevant, rng.normal(0.3, 0.05, size=n), rng.normal(0.8, 0.1, size=n),
        )

        def density(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

        f_r = density(distances, 0.3, 0.05)
        f_g = prevalence * f_r + (1.0 - prevalence) * density(distances, 0.8, 0.1)
        evidence = np.log(f_r / f_g)

        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1, base_rate=prevalence)
        exact = sigmoid(vct._posterior_log_odds(evidence, "exact"))
        approximate = sigmoid(vct._posterior_log_odds(evidence, "approximate"))
        assert float(np.mean(exact)) == pytest.approx(prevalence, rel=0.02)
        assert float(np.mean(approximate)) < 0.8 * prevalence
        assert float(np.max(approximate)) <= 1.0 / (2.0 - prevalence) + 1e-12

    def test_ratio_above_the_bound_is_clipped_below_one(self):
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1, base_rate=0.02)
        clipped = vct._posterior_log_odds(np.array([50.0, 80.0]), "exact")
        assert np.all(np.isfinite(clipped))
        assert clipped[0] == clipped[1] == pytest.approx(float(logit(1.0 - 1e-10)), rel=1e-6)

    def test_empirical_background_bounds_the_posterior(self):
        # With the background estimated from the scored population by the
        # same kernel, the exact posterior is a probability without
        # clipping for nearly every candidate.
        distances = _candidates()
        vct = VectorProbabilityTransform.fit_background(distances, base_rate=3.0 / 1000.0)
        weights = np.where(distances < 0.45, 1.0, 0.0)
        log_odds = vct.calibrate(
            distances, weights=weights, method="kde", bandwidth_factor=1.0,
            posterior="exact", background_distances=distances, return_log_odds=True,
        )
        probabilities = np.asarray(sigmoid(log_odds))
        assert np.all(probabilities[:3] > 0.5)
        assert np.all(probabilities[3:] < 0.01)

    def test_invalid_posterior(self):
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.1)
        with pytest.raises(ValueError, match="posterior must be one of"):
            vct.calibrate(np.array([0.2, 0.5, 0.9]), posterior="bayes")


class TestMonotoneProjection:
    def test_evidence_is_non_increasing_in_distance(self):
        rng = np.random.default_rng(4)
        distances = np.sort(rng.uniform(0.1, 1.0, size=300))
        # Weights concentrated mid-range make the raw ratio non-monotone.
        weights = np.exp(-0.5 * ((distances - 0.45) / 0.03) ** 2)
        vct = VectorProbabilityTransform(mu_G=0.9, sigma_G=0.2, base_rate=0.01)

        raw = vct.calibrate(
            distances, weights=weights, method="kde", bandwidth_factor=1.0,
            return_log_odds=True,
        )
        assert np.any(np.diff(raw) > 1e-9)

        projected = vct.calibrate(
            distances, weights=weights, method="kde", bandwidth_factor=1.0,
            monotone=True, return_log_odds=True,
        )
        assert np.all(np.diff(projected) <= 1e-12)

    def test_projection_preserves_the_distance_ranking(self):
        rng = np.random.default_rng(5)
        distances = rng.uniform(0.1, 1.0, size=200)
        weights = np.exp(-0.5 * ((distances - 0.5) / 0.05) ** 2)
        vct = VectorProbabilityTransform(mu_G=0.9, sigma_G=0.2, base_rate=0.01)
        projected = vct.calibrate(
            distances, weights=weights, method="kde", monotone=True,
            return_log_odds=True,
        )
        order = np.argsort(distances)
        assert np.all(np.diff(projected[order]) <= 1e-12)

    def test_projection_leaves_a_monotone_ratio_unchanged(self):
        distances = np.linspace(0.1, 0.6, 40)
        weights = np.exp(-distances / 0.05)
        vct = VectorProbabilityTransform(mu_G=0.9, sigma_G=0.3, base_rate=0.01)
        raw = vct.calibrate(
            distances, weights=weights, method="gmm", return_log_odds=True,
        )
        if np.all(np.diff(raw) <= 0.0):
            projected = vct.calibrate(
                distances, weights=weights, method="gmm", monotone=True,
                return_log_odds=True,
            )
            np.testing.assert_allclose(projected, raw, atol=1e-9)


class TestDuplicateExclusion:
    """Remark 4.12: a copy of the query captures gap detection."""

    def test_query_copy_collapses_gap_detection(self):
        distances = _candidates()
        vct = VectorProbabilityTransform.fit_background(distances, base_rate=0.003)
        clean = vct.calibrate(distances, method="auto")
        assert np.all(clean[:3] > 0.5)

        # A copy of the query sits at distance zero.  The largest gap is
        # now the one between the copy and every genuine candidate, so
        # gap detection gives all weight to the copy and the relevant
        # density collapses to a single kernel at zero.
        with_copy = np.concatenate([[0.0], distances])
        vct_copy = VectorProbabilityTransform.fit_background(with_copy, base_rate=0.003)
        collapsed = vct_copy.calibrate(with_copy, method="auto")
        assert np.all(collapsed[1:4] < 1e-7)
        # Every genuine candidate is tied at the numerical floor.
        floor = float(np.min(collapsed))
        assert int(np.sum(collapsed[1:] == floor)) == len(distances)

    def test_external_weights_degrade_far_less(self):
        distances = _candidates()
        weights = np.where(distances < 0.45, 0.9, 0.02)
        vct = VectorProbabilityTransform.fit_background(distances, base_rate=0.003)
        clean = vct.calibrate(distances, weights=weights, method="kde")

        with_copy = np.concatenate([[0.0], distances])
        weights_copy = np.concatenate([[1.0], weights])
        vct_copy = VectorProbabilityTransform.fit_background(with_copy, base_rate=0.003)
        degraded = vct_copy.calibrate(with_copy, weights=weights_copy, method="kde")

        # Imperfect weights retain background mass (Proposition 4.9).
        # In the separated relevant region the marginal-density posterior
        # approaches that mixture's purity, not an invented high certainty.
        purity = float(np.sum(weights[:3]) / np.sum(weights))
        np.testing.assert_allclose(clean[:3], purity, rtol=1e-3)
        # The copy adds one kernel instead of replacing the estimate.
        assert np.all(degraded[1:4] > 0.1)

    def test_duplicate_threshold_restores_the_estimate(self):
        distances = _candidates()
        vct = VectorProbabilityTransform.fit_background(distances, base_rate=0.003)
        clean = vct.calibrate(distances, method="auto", return_log_odds=True)

        with_copy = np.concatenate([[0.0], distances])
        restored = vct.calibrate_with_sample(
            distances, with_copy, method="auto",
            duplicate_threshold=1e-6, return_log_odds=True,
        )
        np.testing.assert_allclose(restored, clean, rtol=1e-9)

    def test_threshold_filters_weights_and_priors_consistently(self):
        sample = np.array([0.0, 0.2, 0.25, 0.3, 0.8, 0.85, 0.9])
        weights = np.array([1.0, 0.9, 0.8, 0.7, 0.05, 0.05, 0.05])
        vct = VectorProbabilityTransform(mu_G=0.85, sigma_G=0.1, base_rate=0.01)
        filtered = vct.calibrate_with_sample(
            sample[1:], sample, weights=weights, method="kde",
            duplicate_threshold=1e-6, return_log_odds=True,
        )
        reference = vct.calibrate_with_sample(
            sample[1:], sample[1:], weights=weights[1:], method="kde",
            return_log_odds=True,
        )
        np.testing.assert_allclose(filtered, reference)
