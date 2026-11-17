#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for the convergent Platt solver and the isotonic regression helper."""

import numpy as np
import pytest

from bayesian_bm25.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    isotonic_regression,
)
from bayesian_bm25.probability import sigmoid


def _imbalanced_scores(seed: int = 0, n: int = 100000):
    """Unstandardized scores with a relevance rate of about 0.2 percent."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.002).astype(np.float64)
    scores = np.where(
        labels == 1.0, rng.normal(14.0, 4.0, size=n), rng.normal(6.0, 3.0, size=n),
    ).clip(0.01)
    return scores, labels


def _cross_entropy(calibrator: PlattCalibrator, scores, labels) -> float:
    z = calibrator.a * scores + calibrator.b
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


class TestPlattNewton:
    def test_recovers_the_generating_parameters(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(0.0, 2.0, size=200000)
        labels = (rng.random(200000) < sigmoid(1.3 * scores - 0.7)).astype(np.float64)
        calibrator = PlattCalibrator()
        calibrator.fit(scores, labels)
        assert calibrator.a == pytest.approx(1.3, abs=0.03)
        assert calibrator.b == pytest.approx(-0.7, abs=0.03)

    def test_reaches_the_stationary_point(self):
        scores, labels = _imbalanced_scores()
        calibrator = PlattCalibrator()
        calibrator.fit(scores, labels)
        predicted = np.asarray(calibrator.calibrate(scores))
        # Both gradient components vanish at the maximum-likelihood fit.
        assert abs(float(np.mean(predicted - labels))) < 1e-9
        assert abs(float(np.mean((predicted - labels) * scores))) < 1e-7

    def test_predictions_sum_to_the_number_of_positives(self):
        # Proposition 3.22: at a stationary point of the intercept the
        # predicted probabilities sum to the observed count.
        scores, labels = _imbalanced_scores(seed=2)
        calibrator = PlattCalibrator()
        calibrator.fit(scores, labels)
        assert float(np.sum(calibrator.calibrate(scores))) == pytest.approx(
            float(np.sum(labels)), rel=1e-6,
        )

    def test_fixed_step_gradient_descent_fails_where_newton_converges(self):
        # Unstandardized, heavily imbalanced scores: fixed-step gradient
        # descent from zero stops with a slope of the wrong sign, which
        # inverts the ranking.  Newton's method reaches the optimum.
        scores, labels = _imbalanced_scores(seed=3)
        gradient = PlattCalibrator(a=0.0, b=0.0)
        gradient.fit(
            scores, labels, method="gradient",
            learning_rate=0.01, max_iterations=5000, tolerance=1e-7,
        )
        newton = PlattCalibrator()
        newton.fit(scores, labels)

        assert gradient.a < 0.0
        assert newton.a > 0.0
        assert _cross_entropy(newton, scores, labels) < 0.5 * _cross_entropy(
            gradient, scores, labels,
        )

    def test_result_does_not_depend_on_the_initial_parameters(self):
        scores, labels = _imbalanced_scores(seed=4, n=20000)
        first = PlattCalibrator(a=1.0, b=0.0)
        first.fit(scores, labels)
        second = PlattCalibrator(a=-50.0, b=300.0)
        second.fit(scores, labels)
        assert first.a == pytest.approx(second.a)
        assert first.b == pytest.approx(second.b)

    def test_invariant_to_the_scale_of_the_scores(self):
        scores, labels = _imbalanced_scores(seed=5, n=20000)
        base = PlattCalibrator()
        base.fit(scores, labels)
        scaled = PlattCalibrator()
        scaled.fit(1000.0 * scores + 7.0, labels)
        np.testing.assert_allclose(
            scaled.calibrate(1000.0 * scores + 7.0), base.calibrate(scores), rtol=1e-6,
        )

    def test_sample_weight_equals_replication(self):
        rng = np.random.default_rng(6)
        scores = rng.normal(size=300)
        labels = (rng.random(300) < sigmoid(scores)).astype(np.float64)
        counts = rng.integers(1, 5, size=300)
        weighted = PlattCalibrator()
        weighted.fit(scores, labels, sample_weight=counts)
        replicated = PlattCalibrator()
        replicated.fit(np.repeat(scores, counts), np.repeat(labels, counts))
        assert weighted.a == pytest.approx(replicated.a, rel=1e-6)
        assert weighted.b == pytest.approx(replicated.b, rel=1e-6)

    def test_fractional_labels_fit_grouped_data(self):
        # A bin with relevant fraction y and count n contributes like n
        # samples of which a fraction y is positive.
        scores = np.array([-2.0, 0.0, 2.0])
        fractions = np.array([0.1, 0.5, 0.9])
        counts = np.array([1000.0, 1000.0, 1000.0])
        grouped = PlattCalibrator()
        grouped.fit(scores, fractions, sample_weight=counts)
        np.testing.assert_allclose(grouped.calibrate(scores), fractions, atol=0.02)

    def test_constant_scores_give_the_base_rate(self):
        calibrator = PlattCalibrator()
        calibrator.fit(np.full(100, 3.0), np.array([1.0] * 20 + [0.0] * 80))
        assert calibrator.a == 0.0
        assert calibrator.calibrate(3.0) == pytest.approx(0.2)

    def test_separable_data_terminates(self):
        scores = np.concatenate([np.linspace(-5.0, -1.0, 50), np.linspace(1.0, 5.0, 50)])
        labels = (scores > 0.0).astype(np.float64)
        calibrator = PlattCalibrator()
        calibrator.fit(scores, labels)
        assert np.isfinite(calibrator.a) and calibrator.a > 0.0
        assert np.all((np.asarray(calibrator.calibrate(scores)) > 0.5) == (labels == 1.0))

    def test_log_odds(self):
        calibrator = PlattCalibrator(a=2.0, b=-1.0)
        scores = np.array([-1.0, 0.0, 3.0])
        np.testing.assert_allclose(calibrator.log_odds(scores), 2.0 * scores - 1.0)
        np.testing.assert_allclose(
            sigmoid(calibrator.log_odds(scores)), calibrator.calibrate(scores),
        )
        assert isinstance(calibrator.log_odds(1.5), float)

    def test_gradient_method_matches_the_original_update(self):
        rng = np.random.default_rng(7)
        scores = rng.normal(size=200)
        labels = (rng.random(200) < sigmoid(scores)).astype(np.float64)
        calibrator = PlattCalibrator(a=0.5, b=0.0)
        calibrator.fit(
            scores, labels, method="gradient", learning_rate=0.1, max_iterations=1,
        )
        predicted = np.clip(np.asarray(sigmoid(0.5 * scores)), 1e-10, 1 - 1e-10)
        error = predicted - labels
        assert calibrator.a == pytest.approx(0.5 - 0.1 * float(np.mean(error * scores)))
        assert calibrator.b == pytest.approx(-0.1 * float(np.mean(error)))

    def test_invalid_arguments(self):
        calibrator = PlattCalibrator()
        with pytest.raises(ValueError, match="method must be one of"):
            calibrator.fit(np.zeros(3), np.zeros(3), method="lbfgs")
        with pytest.raises(ValueError, match="does not match"):
            calibrator.fit(np.zeros(3), np.zeros(2))
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.fit(np.array([]), np.array([]))
        with pytest.raises(ValueError, match="non-negative"):
            calibrator.fit(np.zeros(3), np.zeros(3), sample_weight=np.array([1.0, -1.0, 1.0]))


class TestIsotonicRegression:
    def test_monotone_input_is_unchanged(self):
        values = np.array([0.1, 0.4, 0.4, 2.0])
        np.testing.assert_allclose(isotonic_regression(values), values)

    def test_violators_are_pooled(self):
        fitted = isotonic_regression(np.array([1.0, 3.0, 2.0, 2.5, 0.5, 4.0]))
        np.testing.assert_allclose(fitted, [1.0, 2.0, 2.0, 2.0, 2.0, 4.0])

    def test_non_increasing_fit(self):
        fitted = isotonic_regression(
            np.array([5.0, 3.0, 4.0, 1.0, 2.0]), increasing=False,
        )
        np.testing.assert_allclose(fitted, [5.0, 3.5, 3.5, 1.5, 1.5])
        assert np.all(np.diff(fitted) <= 0.0)

    def test_weights_shift_the_pooled_value(self):
        fitted = isotonic_regression(
            np.array([3.0, 1.0]), weights=np.array([1.0, 3.0]),
        )
        np.testing.assert_allclose(fitted, [1.5, 1.5])

    def test_fit_is_the_least_squares_monotone_sequence(self):
        rng = np.random.default_rng(8)
        values = rng.normal(size=60).cumsum() * 0.1 + rng.normal(size=60)
        fitted = isotonic_regression(values)
        assert np.all(np.diff(fitted) >= -1e-12)
        # No monotone perturbation improves the squared error.
        base = float(np.sum((fitted - values) ** 2))
        for _ in range(200):
            candidate = np.sort(fitted + rng.normal(scale=0.05, size=60))
            assert float(np.sum((candidate - values) ** 2)) >= base - 1e-9

    def test_preserves_the_weighted_mean(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=100)
        weights = rng.uniform(0.5, 2.0, size=100)
        fitted = isotonic_regression(values, weights)
        assert float(np.sum(weights * fitted)) == pytest.approx(
            float(np.sum(weights * values))
        )

    def test_rejects_non_positive_weights(self):
        with pytest.raises(ValueError, match="must be positive"):
            isotonic_regression(np.array([1.0, 2.0]), weights=np.array([1.0, 0.0]))

    def test_calibrator_uses_the_same_blocks(self):
        rng = np.random.default_rng(10)
        scores = rng.uniform(0.0, 1.0, size=500)
        labels = (rng.random(500) < scores).astype(np.float64)
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores, labels)
        fitted = isotonic_regression(labels[np.argsort(scores)])
        # The calibrator's breakpoints are the distinct block values.
        assert np.all(np.diff(calibrator._y) >= 0.0)
        np.testing.assert_allclose(
            np.unique(np.round(calibrator._y, 12)), np.unique(np.round(fitted, 12)),
        )
