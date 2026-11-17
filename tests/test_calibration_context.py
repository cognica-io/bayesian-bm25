#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Calibration provenance, evaluation populations, and boundary behavior."""

import math

import numpy as np
import pytest

from bayesian_bm25 import (
    BayesianBM25Scorer,
    CalibrationAccumulator,
    CalibrationContext,
    calibrate_count,
    calibrate_query,
    calibration_report_from_log_odds,
)


class TestCalibrationContext:
    def test_count_factory_distinguishes_its_default_ceiling_from_a_supplied_one(self):
        default = calibrate_count(np.arange(5.0), 1.0)
        supplied = calibrate_count(np.arange(5.0), 1.0, ceiling=1.0)
        assert default.context.ceiling_source == "convention"
        assert supplied.context.ceiling_source == "supplied"

    def test_query_defaults_are_recorded_as_conventions(self):
        model = calibrate_query(np.arange(20.0))
        assert model.context.count_source == "convention"
        assert model.context.slope_source == "convention"
        supplied = calibrate_query(np.arange(20.0), expected_relevant=2.0, ceiling=0.4)
        assert supplied.context.count_source == "supplied"
        assert supplied.context.ceiling_source == "supplied"

    def test_explicit_population_and_sources_are_preserved(self):
        context = CalibrationContext(
            reference_population="candidates", count_source="training_mean",
            slope_source="judgments", ceiling_source="judgments",
            relevance_target="reader_usefulness",
        )
        model = calibrate_count(np.arange(20.0), 2.0, ceiling=0.4, context=context)
        assert model.context == context
        assert model.context.as_dict()["reference_population"] == "candidates"

    def test_scorer_overrides_record_their_sources(self):
        context = CalibrationContext(
            reference_population="corpus", count_source="training_mean",
            slope_source="judgments", ceiling_source="judgments",
        )
        scorer = BayesianBM25Scorer(
            calibration="query", method="lucene", calibration_context=context,
        )
        scorer.index([["foo"], ["foo", "foo"], ["bar"]], show_progress=False)
        inherited = scorer.retrieve([["foo"]], k=2, explain=True).calibrations[0]
        assert inherited.context == context
        overridden = scorer.retrieve(
            [["foo"]], k=2, expected_relevant=0.5, explain=True,
        ).calibrations[0]
        assert overridden.context.count_source == "supplied"
        assert overridden.context.slope_source == "judgments"

    def test_zero_count_is_rejected_as_a_degenerate_likelihood_ratio_model(self):
        for factory in (
            lambda: calibrate_count(np.arange(5.0), 0.0),
            lambda: calibrate_query(np.arange(5.0), expected_relevant=0.0),
        ):
            with pytest.raises(ValueError, match="strictly between"):
                factory()

    def test_public_report_names_the_evaluated_population(self):
        labels = np.array([0.0, 0.0, 0.0, 1.0])
        logits = np.full(4, math.log(1.0 / 3.0))
        context = CalibrationContext(reference_population="top10", count_source="judgments")
        report = calibration_report_from_log_odds(logits, labels, context=context)
        assert report.context.reference_population == "top10"
        assert report.count_ratio == pytest.approx(1.0)
        assert report.log_loss_skill == pytest.approx(0.0, abs=1e-12)

    def test_weighted_evaluation_matches_an_explicit_population(self):
        logits = np.array([-3.0, -1.0, 1.0, 3.0])
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        weights = np.array([10, 2, 3, 1])
        weighted = CalibrationAccumulator()
        weighted.update(logits, labels, sample_weight=weights)
        expanded = CalibrationAccumulator()
        expanded.update(np.repeat(logits, weights), np.repeat(labels, weights))
        for key in ("n", "ece", "brier", "log_loss", "log_loss_skill", "count_ratio"):
            assert weighted.summary()[key] == pytest.approx(expanded.summary()[key])

    def test_both_infinite_probability_limits_have_the_correct_loss(self):
        correct = CalibrationAccumulator()
        correct.update(np.array([-np.inf, np.inf]), np.array([0.0, 1.0]))
        assert correct.summary()["log_loss"] == 0.0
        incorrect = CalibrationAccumulator()
        incorrect.update(np.array([-np.inf, np.inf]), np.array([1.0, 0.0]))
        assert incorrect.summary()["log_loss"] == np.inf

    def test_unjudged_positive_count_is_not_invented(self):
        accumulator = CalibrationAccumulator()
        accumulator.update(np.array([-2.0, -1.0]), np.zeros(2))
        summary = accumulator.summary()
        assert math.isnan(summary["count_ratio"])
        assert summary["observed_relevant"] == 0.0
        assert summary["expected_relevant"] > 0.0

    @pytest.mark.parametrize("name", ["reference_population", "count_source", "relevance_target"])
    def test_context_fields_cannot_be_empty(self, name):
        with pytest.raises(ValueError, match=name):
            CalibrationContext(**{name: ""})
