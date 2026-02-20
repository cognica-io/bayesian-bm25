#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for WAND upper bound (Theorem 6.1.2)."""

import numpy as np
import pytest

from bayesian_bm25.probability import BayesianProbabilityTransform


class TestWANDUpperBound:
    def test_upper_bound_exceeds_actual(self):
        """Upper bound >= any actual probability for the same BM25 score.

        The upper bound uses p_max=0.9, so for any tf/doc_len_ratio combo
        (which produces composite_prior <= 0.9), the upper bound dominates.
        """
        t = BayesianProbabilityTransform(alpha=1.5, beta=2.0)
        bm25_score = 3.0
        upper = t.wand_upper_bound(bm25_score)

        # Try many tf / doc_len_ratio combos
        for tf in [0, 1, 5, 10, 50]:
            for ratio in [0.1, 0.5, 1.0, 2.0, 5.0]:
                actual = t.score_to_probability(bm25_score, tf, ratio)
                assert upper >= actual, (
                    f"Upper bound {upper} < actual {actual} "
                    f"for tf={tf}, ratio={ratio}"
                )

    def test_monotonicity(self):
        """Higher BM25 upper bound -> higher Bayesian upper bound."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=1.0)
        bounds = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        upper_bounds = t.wand_upper_bound(bounds)
        assert np.all(np.diff(upper_bounds) > 0)

    def test_output_range(self):
        """Output is always in (0, 1)."""
        t = BayesianProbabilityTransform(alpha=2.0, beta=0.5)
        bounds = np.array([0.01, 0.5, 1.0, 5.0, 100.0])
        upper_bounds = t.wand_upper_bound(bounds)
        assert np.all(upper_bounds > 0)
        assert np.all(upper_bounds < 1)

    def test_without_base_rate(self):
        """Upper bound works correctly without base_rate."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=1.0, base_rate=None)
        result = t.wand_upper_bound(5.0)
        assert 0 < result < 1
        # With high BM25 score and p_max=0.9, upper bound should be high
        assert result > 0.5

    def test_with_base_rate(self):
        """Upper bound incorporates base_rate when set."""
        t_none = BayesianProbabilityTransform(alpha=1.0, beta=1.0, base_rate=None)
        t_low = BayesianProbabilityTransform(alpha=1.0, beta=1.0, base_rate=0.01)
        bound = 5.0
        upper_none = t_none.wand_upper_bound(bound)
        upper_low = t_low.wand_upper_bound(bound)
        # Low base rate should give lower upper bound
        assert upper_low < upper_none

    def test_uses_p_max_0_9(self):
        """Default p_max=0.9 matches the composite_prior maximum."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        default_bound = t.wand_upper_bound(3.0)
        explicit_bound = t.wand_upper_bound(3.0, p_max=0.9)
        assert default_bound == pytest.approx(explicit_bound)

    def test_pruning_safety_random(self):
        """For random documents, probability never exceeds upper bound."""
        rng = np.random.default_rng(42)
        t = BayesianProbabilityTransform(alpha=1.5, beta=1.0)

        bm25_upper_bound = 8.0
        upper = t.wand_upper_bound(bm25_upper_bound)

        scores = rng.uniform(0, bm25_upper_bound, size=500)
        tfs = rng.uniform(0, 20, size=500)
        ratios = rng.uniform(0.1, 3.0, size=500)

        for i in range(500):
            actual = t.score_to_probability(scores[i], tfs[i], ratios[i])
            assert upper >= actual - 1e-10, (
                f"Pruning violation: upper={upper}, actual={actual}, "
                f"score={scores[i]}, tf={tfs[i]}, ratio={ratios[i]}"
            )

    def test_pruning_safety_with_base_rate(self):
        """Pruning safety holds with base_rate too."""
        rng = np.random.default_rng(123)
        t = BayesianProbabilityTransform(alpha=2.0, beta=0.5, base_rate=0.05)

        bm25_upper_bound = 5.0
        upper = t.wand_upper_bound(bm25_upper_bound)

        scores = rng.uniform(0, bm25_upper_bound, size=500)
        tfs = rng.uniform(0, 20, size=500)
        ratios = rng.uniform(0.1, 3.0, size=500)

        for i in range(500):
            actual = t.score_to_probability(scores[i], tfs[i], ratios[i])
            assert upper >= actual - 1e-10

    def test_scalar_output(self):
        """Scalar input produces scalar output."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        result = t.wand_upper_bound(3.0)
        assert isinstance(result, float)

    def test_array_output(self):
        """Array input produces array output."""
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        result = t.wand_upper_bound(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
