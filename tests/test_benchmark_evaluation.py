#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for the evaluation utilities of the benchmarks.

Covers double-precision ranking (benchmarks.ranking_eval), bootstrap
intervals and paired tests (benchmarks.significance), and the streaming
calibration statistics (benchmarks.calibration_accumulator).
"""

import numpy as np
import pytest

from bayesian_bm25.metrics import brier_score, expected_calibration_error, log_loss
from bayesian_bm25.probability import logit, sigmoid
from benchmarks.calibration_accumulator import CalibrationAccumulator
from benchmarks.ranking_eval import evaluate_run, ranked_run, raw_run
from benchmarks.significance import (
    bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    paired_randomization_test,
)

pytrec_eval = pytest.importorskip("pytrec_eval")


class TestRankedRun:
    def test_scores_encode_the_rank(self):
        run = ranked_run(["a", "b", "c"], np.array([0.2, 0.9, 0.5]))
        assert run == {"b": 3.0, "c": 2.0, "a": 1.0}

    def test_depth_truncates(self):
        scores = np.arange(50, dtype=np.float64)
        run = ranked_run([f"d{i}" for i in range(50)], scores, depth=10)
        assert set(run) == {f"d{i}" for i in range(40, 50)}
        assert run["d49"] == 10.0

    def test_double_precision_differences_decide_the_order(self):
        # The two scores are equal in single precision.
        scores = np.array([30.0, 30.0 + 1e-9])
        assert np.float32(scores[0]) == np.float32(scores[1])
        run = ranked_run(["z", "a"], scores)
        assert run["a"] > run["z"]

    def test_exact_ties_break_by_descending_document_id(self):
        run = ranked_run(["a", "ab", "b"], np.array([1.0, 1.0, 1.0]))
        assert run["b"] > run["ab"] > run["a"]

    def test_ties_at_the_cutoff_are_resolved_by_the_full_rule(self):
        run = ranked_run(["a", "b", "c", "d"], np.array([5.0, 1.0, 1.0, 1.0]), depth=2)
        assert set(run) == {"a", "d"}

    def test_empty_input(self):
        assert ranked_run([], np.array([])) == {}

    def test_handles_infinite_scores(self):
        run = ranked_run(["a", "b", "c"], np.array([-np.inf, 2.0, 1.0]))
        assert run["b"] > run["c"] > run["a"]


class TestEvaluateRun:
    def test_evaluator_compares_raw_scores_in_single_precision(self):
        # The relevant document has the strictly higher double-precision
        # score but the lower document id.
        qrels = {"q": {"a": 1}}
        scores = np.array([30.0 + 1e-9, 30.0])
        raw = evaluate_run(qrels, {"q": raw_run(["a", "z"], scores)})
        ranked = evaluate_run(qrels, {"q": ranked_run(["a", "z"], scores)})
        assert raw["NDCG@10"]["q"] < 1.0
        assert ranked["NDCG@10"]["q"] == pytest.approx(1.0)

    def test_queries_missing_from_the_run_score_zero(self):
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        result = evaluate_run(qrels, {"q1": {"a": 1.0}})
        assert result["NDCG@10"] == {"q1": pytest.approx(1.0), "q2": 0.0}
        assert result["Recall@10"]["q2"] == 0.0

    def test_metric_names(self):
        result = evaluate_run({"q": {"a": 1}}, {"q": {"a": 1.0}}, k=5)
        assert set(result) == {"NDCG@5", "MAP@5", "Recall@5"}


class TestBootstrap:
    def test_interval_contains_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.4, 0.2, size=500)
        mean, lower, upper = bootstrap_ci(values, n_resamples=2000)
        assert mean == pytest.approx(float(np.mean(values)))
        assert lower < mean < upper

    def test_interval_width_matches_the_standard_error(self):
        rng = np.random.default_rng(1)
        values = rng.normal(0.0, 1.0, size=1000)
        _, lower, upper = bootstrap_ci(values, n_resamples=4000)
        expected = 2 * 1.96 * float(np.std(values)) / np.sqrt(1000)
        assert upper - lower == pytest.approx(expected, rel=0.1)

    def test_constant_values_have_a_degenerate_interval(self):
        mean, lower, upper = bootstrap_ci(np.full(20, 0.3), n_resamples=100)
        assert mean == pytest.approx(0.3)
        assert lower == pytest.approx(0.3)
        assert upper == pytest.approx(0.3)

    def test_paired_interval_is_tighter_than_unpaired_for_correlated_systems(self):
        rng = np.random.default_rng(2)
        difficulty = rng.normal(size=400)
        a = difficulty + rng.normal(scale=0.1, size=400) + 0.05
        b = difficulty + rng.normal(scale=0.1, size=400)
        diff, lower, upper = paired_bootstrap_ci(a, b, n_resamples=2000)
        assert diff == pytest.approx(float(np.mean(a - b)))
        assert lower > 0.0
        _, lower_a, upper_a = bootstrap_ci(a, n_resamples=2000)
        assert upper - lower < 0.5 * (upper_a - lower_a)

    def test_rejects_mismatched_pairs(self):
        with pytest.raises(ValueError, match="same shape"):
            paired_bootstrap_ci(np.zeros(3), np.zeros(4))


class TestRandomizationTest:
    def test_identical_systems(self):
        values = np.linspace(0.0, 1.0, 50)
        assert paired_randomization_test(values, values) == 1.0

    def test_clear_difference_is_significant(self):
        rng = np.random.default_rng(3)
        b = rng.normal(size=300)
        a = b + 0.3 + rng.normal(scale=0.1, size=300)
        assert paired_randomization_test(a, b, n_permutations=5000) < 0.001

    def test_noise_is_not_significant(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=300)
        b = a + rng.normal(scale=0.5, size=300)
        assert paired_randomization_test(a, b, n_permutations=5000) > 0.05

    def test_p_value_is_never_zero(self):
        a = np.arange(1.0, 21.0)
        b = np.zeros(20)
        p = paired_randomization_test(a, b, n_permutations=1000)
        assert 0.0 < p <= 2.0 / 1001 + 1e-12

    def test_agrees_with_the_exact_sign_flip_distribution(self):
        # Six pairs: all 64 sign assignments can be enumerated.
        diff = np.array([0.5, -0.2, 0.9, 0.4, -0.1, 0.7])
        observed = abs(float(np.mean(diff)))
        signs = np.array(np.meshgrid(*[[-1, 1]] * 6)).reshape(6, -1).T
        exact = float(np.mean(np.abs(signs @ diff) / 6 >= observed - 1e-12))
        estimate = paired_randomization_test(diff, np.zeros(6), n_permutations=200000)
        assert estimate == pytest.approx(exact, abs=0.01)


class TestHolm:
    def test_adjustment(self):
        adjusted = holm_correction([0.01, 0.04, 0.03])
        assert adjusted == pytest.approx([0.03, 0.06, 0.06])

    def test_capped_at_one_and_monotone(self):
        adjusted = holm_correction([0.5, 0.9, 0.2])
        assert max(adjusted) <= 1.0
        order = np.argsort([0.5, 0.9, 0.2])
        assert np.all(np.diff(np.asarray(adjusted)[order]) >= 0.0)

    def test_single_comparison_is_unchanged(self):
        assert holm_correction([0.02]) == [0.02]


class TestCalibrationAccumulator:
    def _sample(self, seed: int = 0, n: int = 50000):
        rng = np.random.default_rng(seed)
        log_odds = rng.normal(-3.0, 2.0, size=n)
        labels = (rng.random(n) < sigmoid(log_odds)).astype(np.float64)
        return log_odds, labels

    def test_matches_the_library_metrics(self):
        log_odds, labels = self._sample()
        accumulator = CalibrationAccumulator()
        for chunk in np.array_split(np.arange(len(labels)), 7):
            accumulator.update(log_odds[chunk], labels[chunk])
        summary = accumulator.summary()
        probabilities = np.asarray(sigmoid(log_odds))
        assert summary["n"] == len(labels)
        assert summary["brier"] == pytest.approx(brier_score(probabilities, labels))
        assert summary["log_loss"] == pytest.approx(log_loss(probabilities, labels))
        assert summary["ece"] == pytest.approx(
            expected_calibration_error(probabilities, labels), abs=1e-6,
        )

    def test_calibrated_predictions(self):
        log_odds, labels = self._sample(seed=1, n=400000)
        accumulator = CalibrationAccumulator()
        accumulator.update(log_odds, labels)
        summary = accumulator.summary()
        assert summary["expected_over_observed"] == pytest.approx(1.0, abs=0.02)
        assert summary["calibration_slope"] == pytest.approx(1.0, abs=0.03)
        assert summary["calibration_intercept"] == pytest.approx(0.0, abs=0.05)
        assert summary["log_loss_skill"] > 0.0
        assert summary["brier_skill"] > 0.0

    def test_overconfident_predictions_have_a_slope_below_one(self):
        log_odds, labels = self._sample(seed=2, n=400000)
        accumulator = CalibrationAccumulator()
        accumulator.update(2.0 * log_odds, labels)
        assert accumulator.summary()["calibration_slope"] == pytest.approx(0.5, abs=0.03)

    def test_constant_prevalence_predictor_has_zero_skill(self):
        _, labels = self._sample(seed=3)
        prevalence = float(np.mean(labels))
        accumulator = CalibrationAccumulator()
        accumulator.update(np.full(len(labels), float(logit(prevalence))), labels)
        summary = accumulator.summary()
        assert summary["log_loss_skill"] == pytest.approx(0.0, abs=1e-9)
        assert summary["brier_skill"] == pytest.approx(0.0, abs=1e-9)

    def test_ece_cannot_see_an_inverted_ranking_at_low_prevalence(self):
        # At a prevalence near 0.001 a predictor with no resolution, or
        # with an inverted ranking, scores well on ECE; the skill scores
        # expose it.
        rng = np.random.default_rng(4)
        n = 500000
        labels = (rng.random(n) < 0.001).astype(np.float64)
        scores = np.where(labels == 1.0, rng.normal(3.0, 1.0, n), rng.normal(0.0, 1.0, n))
        inverted = CalibrationAccumulator()
        inverted.update(-scores - 7.0, labels)
        summary = inverted.summary()
        assert summary["ece"] < 0.01
        assert summary["log_loss_skill"] < 0.0
        assert summary["calibration_slope"] < 0.0

    def test_probability_zero_for_a_relevant_document_is_an_infinite_loss(self):
        accumulator = CalibrationAccumulator()
        accumulator.update(np.array([-np.inf, 0.0]), np.array([1.0, 0.0]))
        summary = accumulator.summary()
        assert summary["log_loss"] == float("inf")
        assert summary["log_loss_skill"] == float("-inf")

    def test_probability_zero_for_a_non_relevant_document_costs_nothing(self):
        accumulator = CalibrationAccumulator()
        accumulator.update(np.array([-np.inf, 0.0]), np.array([0.0, 1.0]))
        assert accumulator.summary()["log_loss"] == pytest.approx(0.5 * np.log(2.0))

    def test_reliability_rows(self):
        accumulator = CalibrationAccumulator(n_bins=2)
        accumulator.update(
            np.asarray(logit(np.array([0.1, 0.2, 0.8]))), np.array([0.0, 1.0, 1.0]),
        )
        rows = accumulator.reliability()
        assert rows[0] == (pytest.approx(0.15), pytest.approx(0.5), 2)
        assert rows[1] == (pytest.approx(0.8), pytest.approx(1.0), 1)

    def test_empty_accumulator(self):
        with pytest.raises(ValueError, match="no data"):
            CalibrationAccumulator().summary()

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="does not match"):
            CalibrationAccumulator().update(np.zeros(3), np.zeros(2))
