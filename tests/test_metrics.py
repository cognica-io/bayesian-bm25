#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.metrics module."""

import numpy as np
import pytest

from bayesian_bm25.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """ECE = 0 when predicted probabilities match labels exactly."""
        probs = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0)

    def test_worst_calibration(self):
        """ECE > 0 when predictions are inverted."""
        probs = np.array([0.9, 0.9, 0.1, 0.1])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(probs, labels)
        assert ece > 0.5

    def test_constant_prediction(self):
        """ECE for constant prediction equals |p - base_rate|."""
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        probs = np.full(5, 0.5)
        ece = expected_calibration_error(probs, labels, n_bins=1)
        # With 1 bin, ECE = |0.5 - 0.4| = 0.1
        assert ece == pytest.approx(0.1)

    def test_bounds(self):
        """ECE is in [0, 1]."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, size=1000)
        labels = (rng.random(1000) < 0.3).astype(float)
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_n_bins_parameter(self):
        """Different n_bins produce valid results."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, size=100)
        labels = (rng.random(100) < probs).astype(float)
        for n_bins in [2, 5, 10, 20, 50]:
            ece = expected_calibration_error(probs, labels, n_bins=n_bins)
            assert 0.0 <= ece <= 1.0


class TestBrierScore:
    def test_perfect_prediction(self):
        """Brier score = 0 for perfect predictions."""
        probs = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(probs, labels) == pytest.approx(0.0)

    def test_worst_prediction(self):
        """Brier score = 1 for completely wrong predictions."""
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(probs, labels) == pytest.approx(1.0)

    def test_constant_half(self):
        """Brier score for constant 0.5 prediction = 0.25."""
        probs = np.full(100, 0.5)
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        assert brier_score(probs, labels) == pytest.approx(0.25)

    def test_bounds(self):
        """Brier score is in [0, 1] for valid probabilities."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, size=1000)
        labels = (rng.random(1000) < 0.3).astype(float)
        bs = brier_score(probs, labels)
        assert 0.0 <= bs <= 1.0

    def test_better_calibration_lower_score(self):
        """Better-calibrated predictions have lower Brier score."""
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        good_probs = np.array([0.1, 0.2, 0.1, 0.8, 0.9, 0.8])
        bad_probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert brier_score(good_probs, labels) < brier_score(bad_probs, labels)


class TestReliabilityDiagram:
    def test_returns_tuples(self):
        """Each entry is (avg_pred, avg_actual, count)."""
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        bins = reliability_diagram(probs, labels)
        assert len(bins) > 0
        for avg_pred, avg_actual, count in bins:
            assert 0.0 <= avg_pred <= 1.0
            assert 0.0 <= avg_actual <= 1.0
            assert count > 0

    def test_total_count_equals_n(self):
        """Sum of counts across all bins equals total number of samples."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, size=200)
        labels = (rng.random(200) < 0.5).astype(float)
        bins = reliability_diagram(probs, labels)
        total = sum(count for _, _, count in bins)
        assert total == 200

    def test_perfect_calibration_diagonal(self):
        """For perfectly calibrated data, avg_pred ~ avg_actual."""
        rng = np.random.default_rng(42)
        n = 10000
        probs = rng.uniform(0, 1, size=n)
        labels = (rng.random(n) < probs).astype(float)
        bins = reliability_diagram(probs, labels, n_bins=5)
        for avg_pred, avg_actual, count in bins:
            if count >= 100:
                assert abs(avg_pred - avg_actual) < 0.1

    def test_empty_bins_excluded(self):
        """Bins with no samples are not included."""
        probs = np.array([0.05, 0.05, 0.95, 0.95])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        bins = reliability_diagram(probs, labels, n_bins=10)
        # Only the extreme bins should have data
        assert len(bins) <= 3

    def test_n_bins_parameter(self):
        """Different n_bins produce valid results."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, size=100)
        labels = (rng.random(100) < 0.5).astype(float)
        for n_bins in [2, 5, 10, 20]:
            bins = reliability_diagram(probs, labels, n_bins=n_bins)
            assert len(bins) > 0
            assert len(bins) <= n_bins


class TestMainPackageExport:
    def test_import_from_main_package(self):
        """Metrics are importable from bayesian_bm25 directly."""
        import bayesian_bm25

        assert hasattr(bayesian_bm25, "expected_calibration_error")
        assert hasattr(bayesian_bm25, "brier_score")
        assert hasattr(bayesian_bm25, "reliability_diagram")

    def test_import_from_benchmarks(self):
        """Backward compatibility: metrics still importable from benchmarks."""
        from benchmarks.metrics import (
            brier_score as bs,
            expected_calibration_error as ece,
            reliability_diagram as rd,
        )

        probs = np.array([0.5, 0.5])
        labels = np.array([0.0, 1.0])
        assert ece(probs, labels) >= 0
        assert bs(probs, labels) >= 0
        assert len(rd(probs, labels)) > 0
