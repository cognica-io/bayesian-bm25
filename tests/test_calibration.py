#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.calibration module."""

import numpy as np
import pytest

from bayesian_bm25.calibration import IsotonicCalibrator, PlattCalibrator
from bayesian_bm25.fusion import log_odds_conjunction
from bayesian_bm25.probability import sigmoid


class TestPlattCalibratorDefaultParameters:
    def test_default_parameters(self):
        cal = PlattCalibrator()
        assert cal.a == pytest.approx(1.0)
        assert cal.b == pytest.approx(0.0)


class TestPlattCalibratorOutputRange:
    def test_calibrate_output_range(self):
        cal = PlattCalibrator(a=2.0, b=-1.0)
        rng = np.random.default_rng(42)
        scores = rng.uniform(-10, 10, size=200)
        result = cal.calibrate(scores)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)


class TestPlattCalibratorMonotonic:
    def test_calibrate_monotonic(self):
        cal = PlattCalibrator(a=1.5, b=-0.5)
        scores = np.linspace(-5.0, 5.0, 100)
        result = cal.calibrate(scores)
        assert np.all(np.diff(result) > 0)


class TestPlattCalibratorScalar:
    def test_calibrate_scalar(self):
        cal = PlattCalibrator(a=1.0, b=0.0)
        result = cal.calibrate(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(sigmoid(0.5))


class TestPlattCalibratorArray:
    def test_calibrate_array(self):
        cal = PlattCalibrator(a=1.0, b=0.0)
        scores = np.array([0.0, 1.0, -1.0, 2.5])
        result = cal.calibrate(scores)
        assert isinstance(result, np.ndarray)
        assert result.shape == scores.shape


class TestPlattCalibratorFitRecoversParameters:
    def test_fit_recovers_parameters(self):
        rng = np.random.default_rng(42)
        true_a = 2.0
        true_b = -1.0
        scores = rng.uniform(-3, 3, size=2000)
        probs = sigmoid(true_a * scores + true_b)
        labels = (rng.random(2000) < probs).astype(float)

        cal = PlattCalibrator(a=0.5, b=0.0)
        cal.fit(
            scores,
            labels,
            learning_rate=0.01,
            max_iterations=5000,
            tolerance=1e-8,
        )

        assert abs(cal.a - true_a) < 0.5
        assert abs(cal.b - true_b) < 0.5


class TestPlattCalibratorCallable:
    def test_callable(self):
        cal = PlattCalibrator(a=1.5, b=-0.3)
        scores = np.array([-1.0, 0.0, 1.0, 2.0])
        result_calibrate = cal.calibrate(scores)
        result_call = cal(scores)
        np.testing.assert_allclose(result_call, result_calibrate)

        scalar_calibrate = cal.calibrate(0.7)
        scalar_call = cal(0.7)
        assert scalar_call == pytest.approx(scalar_calibrate)


class TestPlattCalibratorLogOddsConjunction:
    def test_output_feeds_to_log_odds_conjunction(self):
        cal = PlattCalibrator(a=1.0, b=0.0)
        scores = np.array([0.5, 1.0, 1.5])
        calibrated = cal.calibrate(scores)

        # Calibrated outputs must be valid probabilities for log_odds_conjunction
        result = log_odds_conjunction(calibrated)
        assert 0.0 < result < 1.0


class TestIsotonicCalibratorMonotone:
    def test_fit_produces_monotone(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 5, size=200)
        probs = sigmoid(2.0 * scores - 3.0)
        labels = (rng.random(200) < probs).astype(float)

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        sorted_scores = np.sort(scores)
        calibrated = cal.calibrate(sorted_scores)
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-12), (
            f"Non-monotone values found: min diff = {diffs.min()}"
        )


class TestIsotonicCalibratorInterpolation:
    def test_interpolation_works(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0.0, 0.0, 0.5, 1.0, 1.0])

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        # Score between known breakpoints should be interpolated
        mid = cal.calibrate(2.5)
        low = cal.calibrate(2.0)
        high = cal.calibrate(3.0)
        assert low <= mid <= high


class TestIsotonicCalibratorHandlesTies:
    def test_handles_ties(self):
        scores = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        labels = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        # Should not raise and should produce valid output
        result = cal.calibrate(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)


class TestIsotonicCalibratorBeforeFitRaises:
    def test_calibrate_before_fit_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="Call fit"):
            cal.calibrate(1.0)


class TestIsotonicCalibratorScalar:
    def test_calibrate_scalar(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([0.0, 0.5, 1.0])

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        result = cal.calibrate(2.0)
        assert isinstance(result, float)


class TestIsotonicCalibratorArray:
    def test_calibrate_array(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([0.0, 0.25, 0.75, 1.0])

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        test_scores = np.array([1.5, 2.5, 3.5])
        result = cal.calibrate(test_scores)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_scores.shape


class TestIsotonicCalibratorExtremeScores:
    def test_extreme_scores(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        # Score below training range gets the leftmost boundary value
        below = cal.calibrate(-100.0)
        at_min = cal.calibrate(1.0)
        assert below == pytest.approx(at_min, abs=1e-6)

        # Score above training range gets the rightmost boundary value
        above = cal.calibrate(100.0)
        at_max = cal.calibrate(5.0)
        assert above == pytest.approx(at_max, abs=1e-6)


class TestIsotonicCalibratorCallable:
    def test_callable(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([0.0, 0.3, 0.7, 1.0])

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        test_scores = np.array([1.5, 2.5, 3.5])
        result_calibrate = cal.calibrate(test_scores)
        result_call = cal(test_scores)
        np.testing.assert_allclose(result_call, result_calibrate)

        scalar_calibrate = cal.calibrate(2.0)
        scalar_call = cal(2.0)
        assert scalar_call == pytest.approx(scalar_calibrate)


class TestIsotonicCalibratorLogOddsConjunction:
    def test_output_feeds_to_log_odds_conjunction(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 5, size=100)
        probs = sigmoid(1.5 * scores - 2.0)
        labels = (rng.random(100) < probs).astype(float)

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)

        test_scores = np.array([1.0, 2.5, 4.0])
        calibrated = cal.calibrate(test_scores)

        # Calibrated outputs must be valid probabilities for log_odds_conjunction
        result = log_odds_conjunction(calibrated)
        assert 0.0 < result < 1.0
