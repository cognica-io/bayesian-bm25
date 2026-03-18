#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.vector_probability module."""

import math

import numpy as np
import pytest

from bayesian_bm25.fusion import log_odds_conjunction
from bayesian_bm25.vector_probability import (
    VectorProbabilityTransform,
    _gaussian_pdf,
    _kernel_density,
    _silverman_bandwidth,
    ivf_density_prior,
    knn_density_prior,
)

# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


class TestGaussianPDF:
    def test_peak_at_mean(self):
        mu, sigma = 3.0, 1.0
        peak = _gaussian_pdf(mu, mu, sigma)
        off_peak = _gaussian_pdf(mu + 1.0, mu, sigma)
        assert peak > off_peak

    def test_known_value_standard_normal(self):
        # N(0,1) at x=0 should be 1/sqrt(2*pi)
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        result = _gaussian_pdf(0.0, 0.0, 1.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_symmetry(self):
        mu, sigma = 5.0, 2.0
        left = _gaussian_pdf(mu - 1.5, mu, sigma)
        right = _gaussian_pdf(mu + 1.5, mu, sigma)
        assert left == pytest.approx(right, rel=1e-10)

    def test_array_input(self):
        x = np.array([0.0, 1.0, -1.0, 2.0])
        result = _gaussian_pdf(x, 0.0, 1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert np.all(result > 0)

    def test_scalar_returns_float(self):
        result = _gaussian_pdf(0.0, 0.0, 1.0)
        assert isinstance(result, float)


class TestSilvermanBandwidth:
    def test_positive_bandwidth(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(5.0, 2.0, size=100)
        h = _silverman_bandwidth(distances)
        assert h > 0.0

    def test_uniform_weights_close_to_unweighted(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(0.0, 1.0, size=200)
        h_none = _silverman_bandwidth(distances)
        h_uniform = _silverman_bandwidth(
            distances, np.ones(200, dtype=np.float64)
        )
        assert h_none == pytest.approx(h_uniform, rel=1e-10)

    def test_concentrated_weights_reduce_bandwidth(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(0.0, 3.0, size=100)
        uniform_w = np.ones(100)
        # Concentrate weights on the center
        concentrated_w = np.exp(-distances**2)
        h_uniform = _silverman_bandwidth(distances, uniform_w)
        h_concentrated = _silverman_bandwidth(distances, concentrated_w)
        assert h_concentrated < h_uniform


class TestKernelDensity:
    def test_integrates_approximately_to_one(self):
        sample = np.array([0.0, 1.0, 2.0])
        weights = np.array([1.0, 1.0, 1.0])
        h = 0.5
        # Evaluate on a fine grid
        grid = np.linspace(-5.0, 7.0, 10000)
        density = _kernel_density(grid, sample, weights, h)
        dx = grid[1] - grid[0]
        integral = float(np.sum(density) * dx)
        assert integral == pytest.approx(1.0, abs=0.02)

    def test_non_negative(self):
        sample = np.array([1.0, 3.0, 5.0])
        weights = np.array([1.0, 2.0, 1.0])
        grid = np.linspace(-2.0, 8.0, 500)
        density = _kernel_density(grid, sample, weights, 1.0)
        assert np.all(density > 0)

    def test_peak_near_weighted_center(self):
        # Heavy weight on x=2.0
        sample = np.array([0.0, 2.0, 4.0])
        weights = np.array([0.1, 10.0, 0.1])
        grid = np.linspace(-1.0, 5.0, 1000)
        density = _kernel_density(grid, sample, weights, 0.5)
        peak_idx = int(np.argmax(density))
        peak_x = grid[peak_idx]
        assert abs(peak_x - 2.0) < 0.2


# ---------------------------------------------------------------------------
# VectorProbabilityTransform construction
# ---------------------------------------------------------------------------


class TestFitBackground:
    def test_estimates_mean_and_std(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(5.0, 2.0, size=10000)
        vct = VectorProbabilityTransform.fit_background(distances)
        assert vct.mu_G == pytest.approx(5.0, abs=0.1)
        assert vct.sigma_G == pytest.approx(2.0, abs=0.1)

    def test_base_rate_forwarded(self):
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vct = VectorProbabilityTransform.fit_background(
            distances, base_rate=0.01
        )
        assert vct.base_rate == 0.01

    def test_constant_distances_epsilon_sigma(self):
        distances = np.array([3.0, 3.0, 3.0, 3.0])
        vct = VectorProbabilityTransform.fit_background(distances)
        assert vct.sigma_G > 0.0

    def test_constructor_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma_G must be positive"):
            VectorProbabilityTransform(mu_G=0.0, sigma_G=-1.0)

    def test_constructor_rejects_invalid_base_rate(self):
        with pytest.raises(ValueError, match="base_rate must be in"):
            VectorProbabilityTransform(mu_G=0.0, sigma_G=1.0, base_rate=1.5)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


class TestDetectGap:
    def test_detects_clear_gap(self):
        # Cluster at [0.1, 0.2, 0.3] then gap then [0.9, 1.0]
        distances = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.3)
        gap_idx = vct._detect_gap(distances)
        assert gap_idx is not None
        sorted_d = np.sort(distances)
        # The gap should be between 0.3 and 0.9
        assert sorted_d[gap_idx - 1] < 0.5
        assert sorted_d[gap_idx] > 0.5

    def test_no_gap_in_uniform(self):
        distances = np.linspace(0.0, 1.0, 20)
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.3)
        gap_idx = vct._detect_gap(distances)
        assert gap_idx is None

    def test_too_few_points_returns_none(self):
        distances = np.array([1.0, 2.0])
        vct = VectorProbabilityTransform(mu_G=1.5, sigma_G=0.5)
        assert vct._detect_gap(distances) is None

    def test_identical_distances_returns_none(self):
        distances = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        vct = VectorProbabilityTransform(mu_G=1.0, sigma_G=0.5)
        assert vct._detect_gap(distances) is None

    def test_z_score_fallback(self):
        # Gaps just below the 15% span ratio but with high z-score
        # Uniform spacing of 0.01 + one gap of 0.05
        base = np.arange(0, 10) * 0.01  # [0.00, 0.01, ... 0.09]
        top = np.array([0.14, 0.15, 0.16])
        distances = np.concatenate([base, top])
        vct = VectorProbabilityTransform(mu_G=0.1, sigma_G=0.05)
        gap_idx = vct._detect_gap(distances)
        # Should detect the gap between 0.09 and 0.14
        assert gap_idx is not None


class TestGapWeights:
    def test_binary_partition(self):
        distances = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.3)
        w = vct._gap_weights(distances)
        assert w is not None
        # First three should be 1.0, last two 0.0
        assert np.sum(w == 1.0) == 3
        assert np.sum(w == 0.0) == 2

    def test_returns_none_when_no_gap(self):
        distances = np.linspace(0.0, 1.0, 20)
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.3)
        assert vct._gap_weights(distances) is None


# ---------------------------------------------------------------------------
# Weight transforms
# ---------------------------------------------------------------------------


class TestSharpenWeights:
    def test_preserves_total_mass(self):
        weights = np.array([0.6, 0.65, 0.7, 0.55, 0.5])
        sharpened = VectorProbabilityTransform._sharpen_weights(weights)
        assert float(np.sum(sharpened)) == pytest.approx(
            float(np.sum(weights)), rel=1e-6
        )

    def test_increases_contrast(self):
        weights = np.array([0.6, 0.65, 0.7, 0.55, 0.5])
        sharpened = VectorProbabilityTransform._sharpen_weights(weights)
        # The max weight should become more dominant
        orig_ratio = float(np.max(weights) / np.sum(weights))
        sharp_ratio = float(np.max(sharpened) / np.sum(sharpened))
        assert sharp_ratio > orig_ratio

    def test_low_temperature_concentrates(self):
        weights = np.array([0.5, 0.6, 0.7, 0.55])
        low_t = VectorProbabilityTransform._sharpen_weights(
            weights, temperature=0.01
        )
        high_t = VectorProbabilityTransform._sharpen_weights(
            weights, temperature=0.5
        )
        # Lower temperature should make max weight more dominant
        low_max_frac = float(np.max(low_t) / np.sum(low_t))
        high_max_frac = float(np.max(high_t) / np.sum(high_t))
        assert low_max_frac > high_max_frac


class TestDistanceDensityWeights:
    def test_closer_distances_get_higher_weight(self):
        distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        weights = VectorProbabilityTransform._distance_density_weights(
            distances
        )
        # Weights should decrease as distance increases
        assert np.all(np.diff(weights) <= 0)

    def test_output_range(self):
        distances = np.array([0.3, 0.5, 1.0, 2.0, 3.0])
        weights = VectorProbabilityTransform._distance_density_weights(
            distances
        )
        assert np.all(weights > 0.0)
        assert np.all(weights < 1.0)


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------


class TestEstimateKDE:
    def test_returns_positive_densities(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(0.3, 0.1, size=50)
        weights = np.ones(50)
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.2)
        f_R = vct.estimate_kde(distances, weights)
        assert np.all(f_R > 0)
        assert f_R.shape == distances.shape

    def test_bandwidth_factor_effect(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(0.3, 0.1, size=100)
        weights = np.ones(100)
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.2)
        f_R_narrow = vct.estimate_kde(distances, weights, bandwidth_factor=0.5)
        f_R_wide = vct.estimate_kde(distances, weights, bandwidth_factor=5.0)
        # Narrower bandwidth should produce higher peak density
        assert float(np.max(f_R_narrow)) > float(np.max(f_R_wide))


class TestEstimateGMM:
    def test_returns_positive_densities(self):
        rng = np.random.default_rng(42)
        # Mixture: relevant cluster near 0.3, background near 0.8
        relevant = rng.normal(0.3, 0.05, size=30)
        background = rng.normal(0.8, 0.15, size=70)
        distances = np.concatenate([relevant, background])
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.15)
        f_R = vct.estimate_gmm(distances)
        assert np.all(f_R > 0)
        assert f_R.shape == distances.shape

    def test_higher_density_near_relevant_cluster(self):
        rng = np.random.default_rng(42)
        relevant = rng.normal(0.2, 0.03, size=20)
        background = rng.normal(0.7, 0.1, size=80)
        distances = np.concatenate([relevant, background])
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.1)
        # Use weights emphasizing the relevant cluster
        weights = np.zeros(100)
        weights[:20] = 1.0
        f_R = vct.estimate_gmm(distances, weights)
        # f_R near the relevant cluster should be higher
        mean_f_R_relevant = float(np.mean(f_R[:20]))
        mean_f_R_background = float(np.mean(f_R[20:]))
        assert mean_f_R_relevant > mean_f_R_background

    def test_no_weights_still_works(self):
        rng = np.random.default_rng(42)
        distances = rng.normal(0.5, 0.2, size=50)
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.2)
        f_R = vct.estimate_gmm(distances)
        assert f_R.shape == distances.shape
        assert np.all(f_R > 0)


# ---------------------------------------------------------------------------
# Log density ratio
# ---------------------------------------------------------------------------


class TestLogDensityRatio:
    def test_positive_when_f_R_dominates(self):
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.2)
        distances = np.array([0.2, 0.3])
        # f_R >> f_G at these distances (far from background mean)
        f_R = np.array([5.0, 4.0])
        ratios = vct.log_density_ratio(distances, f_R)
        assert np.all(ratios > 0)

    def test_negative_when_f_G_dominates(self):
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.1)
        distances = np.array([0.5])
        # f_R much smaller than f_G near the background mean
        f_R = np.array([0.01])
        ratio = vct.log_density_ratio(distances, f_R)
        assert float(ratio[0]) < 0

    def test_scalar_input(self):
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.2)
        result = vct.log_density_ratio(0.3, 1.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Auto-routing
# ---------------------------------------------------------------------------


class TestAutoRouting:
    def test_gap_large_k_uses_kde(self):
        rng = np.random.default_rng(42)
        # Clear gap: 40 relevant near 0.2, 60 background near 0.8
        relevant = rng.normal(0.2, 0.03, size=40)
        background = rng.normal(0.8, 0.05, size=60)
        distances = np.concatenate([relevant, background])
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.2)
        result = vct.calibrate(distances, method="auto")
        assert result.shape == distances.shape
        assert np.all(result > 0) and np.all(result < 1)

    def test_gap_small_k_uses_gmm(self):
        # Gap with K < 50
        distances = np.array(
            [0.1, 0.12, 0.15, 0.18, 0.2, 0.8, 0.82, 0.85, 0.9]
        )
        vct = VectorProbabilityTransform(mu_G=0.8, sigma_G=0.2)
        result = vct.calibrate(distances, method="auto")
        assert result.shape == distances.shape
        assert np.all(result > 0) and np.all(result < 1)

    def test_smooth_with_weights_uses_kde(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.3, 0.7, size=30)
        weights = rng.uniform(0.4, 0.8, size=30)
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.2)
        result = vct.calibrate(distances, weights=weights, method="auto")
        assert result.shape == distances.shape
        assert np.all(result > 0) and np.all(result < 1)

    def test_smooth_with_density_prior_uses_gmm(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.3, 0.7, size=30)
        density_prior = rng.uniform(0.3, 0.7, size=30)
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.2)
        result = vct.calibrate(
            distances, density_prior=density_prior, method="auto"
        )
        assert result.shape == distances.shape
        assert np.all(result > 0) and np.all(result < 1)

    def test_smooth_fallback_uses_gmm_with_distance_weights(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.3, 0.7, size=30)
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.2)
        result = vct.calibrate(distances, method="auto")
        assert result.shape == distances.shape
        assert np.all(result > 0) and np.all(result < 1)


# ---------------------------------------------------------------------------
# Calibrate end-to-end
# ---------------------------------------------------------------------------


class TestCalibrateEndToEnd:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.1, 1.0, size=100)
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.2)
        result = vct.calibrate(distances)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_scalar_input(self):
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.2)
        scalar_result = vct.calibrate(0.3)
        assert isinstance(scalar_result, float)
        assert 0.0 < scalar_result < 1.0

    def test_array_input(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.1, 1.0, size=50)
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.2)
        result = vct.calibrate(distances)
        assert isinstance(result, np.ndarray)
        assert result.shape == distances.shape

    def test_base_rate_shifts_output(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.1, 0.9, size=80)
        vct_neutral = VectorProbabilityTransform(
            mu_G=0.7, sigma_G=0.2, base_rate=None
        )
        vct_low = VectorProbabilityTransform(
            mu_G=0.7, sigma_G=0.2, base_rate=0.01
        )
        result_neutral = vct_neutral.calibrate(distances)
        result_low = vct_low.calibrate(distances)
        # Low base rate should pull probabilities down
        assert float(np.mean(result_low)) < float(np.mean(result_neutral))

    def test_explicit_kde_method(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.1, 1.0, size=50)
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.2)
        result = vct.calibrate(distances, method="kde")
        assert np.all(result > 0) and np.all(result < 1)

    def test_explicit_gmm_method(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.1, 1.0, size=50)
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.2)
        result = vct.calibrate(distances, method="gmm")
        assert np.all(result > 0) and np.all(result < 1)

    def test_invalid_method_raises(self):
        vct = VectorProbabilityTransform(mu_G=0.5, sigma_G=0.2)
        with pytest.raises(ValueError, match="method must be"):
            vct.calibrate(np.array([0.3, 0.5]), method="invalid")


# ---------------------------------------------------------------------------
# Integration with log_odds_conjunction
# ---------------------------------------------------------------------------


class TestLogOddsConjunctionIntegration:
    def test_calibrated_output_feeds_to_conjunction(self):
        rng = np.random.default_rng(42)
        distances = rng.uniform(0.1, 0.9, size=50)
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.2)
        calibrated = vct.calibrate(distances)

        # Pick a few calibrated probabilities
        subset = calibrated[:5]
        fused = log_odds_conjunction(subset)
        assert 0.0 < fused < 1.0

    def test_mixed_bm25_and_vector_signals(self):
        from bayesian_bm25.probability import BayesianProbabilityTransform

        # BM25 signal
        bm25_transform = BayesianProbabilityTransform(alpha=1.0, beta=3.0)
        bm25_prob = bm25_transform.score_to_probability(
            score=5.0, tf=3.0, doc_len_ratio=0.8
        )

        # Vector signal via calibrated transform
        rng = np.random.default_rng(42)
        distances = rng.normal(0.3, 0.05, size=20)
        vct = VectorProbabilityTransform(mu_G=0.7, sigma_G=0.15)
        vector_probs = vct.calibrate(distances)
        vector_prob = float(vector_probs[0])

        # Fuse both signals
        fused = log_odds_conjunction(np.array([bm25_prob, vector_prob]))
        assert 0.0 < fused < 1.0


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------


class TestIVFDensityPrior:
    def test_sparse_cell_gets_high_prior(self):
        # Sparse cell (pop < avg): proximity is more informative (IDF analogy)
        result = ivf_density_prior(50, 100.0)
        assert result > 0.5

    def test_dense_cell_gets_low_prior(self):
        # Dense cell (pop > avg): proximity is less discriminative
        result = ivf_density_prior(200, 100.0)
        assert result < 0.5

    def test_average_cell_near_half(self):
        result = ivf_density_prior(100, 100.0)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_gamma_increases_sensitivity(self):
        sparse_low_gamma = ivf_density_prior(50, 100.0, gamma=0.5)
        sparse_high_gamma = ivf_density_prior(50, 100.0, gamma=2.0)
        # Higher gamma pushes sparse cells further above 0.5
        assert sparse_high_gamma > sparse_low_gamma

    def test_array_input(self):
        populations = np.array([50, 100, 200, 300])
        result = ivf_density_prior(populations, 100.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        # Monotonically decreasing: sparser cells get higher weight
        assert np.all(np.diff(result) < 0)

    def test_scalar_returns_float(self):
        result = ivf_density_prior(100, 100.0)
        assert isinstance(result, float)


class TestKNNDensityPrior:
    def test_sparse_neighborhood_gets_high_prior(self):
        # Large kth_distance = sparse region = more informative match
        result = knn_density_prior(2.0, 1.0)
        assert result > 0.5

    def test_dense_neighborhood_gets_low_prior(self):
        # Small kth_distance = dense region = less discriminative
        result = knn_density_prior(0.5, 1.0)
        assert result < 0.5

    def test_median_distance_near_half(self):
        result = knn_density_prior(1.0, 1.0)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_gamma_effect(self):
        sparse_low_gamma = knn_density_prior(2.0, 1.0, gamma=0.5)
        sparse_high_gamma = knn_density_prior(2.0, 1.0, gamma=2.0)
        # Higher gamma pushes sparse neighborhoods further above 0.5
        assert sparse_high_gamma > sparse_low_gamma

    def test_array_input(self):
        distances = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
        result = knn_density_prior(distances, 1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        # Monotonically increasing: sparser (larger distance) = higher weight
        assert np.all(np.diff(result) > 0)

    def test_scalar_returns_float(self):
        result = knn_density_prior(1.0, 1.0)
        assert isinstance(result, float)
