#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Population, held-out judgments, and size-transfer contracts."""

import json

import numpy as np
import pytest

from benchmarks import ceiling_slope
from benchmarks.additional.judgment_baselines import fit_convex, fit_logistic, predict_logistic
from benchmarks.ceiling_query import make_ceiling_query
from benchmarks.query_signals import query_signals
from tests.test_reevaluation_helpers import _prepared


def test_paper_grid_is_complete():
    assert len(ceiling_slope.CEILINGS) == 12
    np.testing.assert_allclose(ceiling_slope.SLOPES, 2.0 ** (np.arange(-6, 7) / 4.0))
    assert ceiling_slope.GRID.shape == (156, 2)


def test_judgment_samples_never_include_held_out_queries_or_repeat_a_pair():
    folds = np.array([0, 1, 0, 1])
    catalog = {"query": np.repeat(np.arange(4), 90), "labels": np.tile([0, 1], 180)}
    logits = np.tile(np.tile([3.0, 0.0, -2.0], 120), (2, 1))
    plans = ceiling_slope._sampling_plans(logits, catalog, folds, n_draws=10, seed=42)
    for fold, plan in enumerate(plans):
        for observations in plan["sure"] + plan["stratified"]:
            assert len(np.unique(observations)) == len(observations)
            assert np.all(folds[catalog["query"][observations]] != fold)
        assert all(len(x) == 30 for x in plan["sure"])
        assert all(len(x) == 90 for x in plan["stratified"])


def test_insufficient_strata_are_reported_without_replacement():
    catalog = {"query": np.array([0, 0, 1, 1]), "labels": np.array([1, 0, 0, 1])}
    plans = ceiling_slope._sampling_plans(
        np.full((2, 4), 3.0),
        catalog,
        np.array([0, 1]),
        n_draws=2,
        seed=42,
    )
    assert plans[0]["available_by_stratum"] == [2, 0, 0]
    assert len(plans[0]["sure"][0]) == 2
    parameters = ceiling_slope._sure_ceiling(catalog["labels"], plans[0]["sure"][0])
    assert parameters["n_judgments"] == 2
    assert parameters["status"] == "insufficient_sure_documents"
    assert parameters["ceiling"] == 0.5
    assert parameters["ceiling"] == parameters["raw_ceiling_estimate"]
    assert not parameters["boundary_projection"]
    unavailable = ceiling_slope._sure_ceiling(catalog["labels"], np.array([], dtype=int))
    assert unavailable["ceiling"] is None
    assert unavailable["source"] == "unavailable_no_sure_documents"


def test_quarters_preserve_document_evidence_and_units():
    prepared = _prepared()
    full = make_ceiling_query(query_signals(prepared, 0), 1.0 / prepared.corpus_alpha)
    kept = np.arange(0, prepared.n_docs, 4)
    part = full.restrict(kept, retrieve_k=30)
    selected = np.isin(full.signals.doc_index, kept)
    for channel in ceiling_slope.CHANNELS:
        np.testing.assert_array_equal(part.evidence(channel), full.evidence(channel)[selected])


def test_same_budget_logistic_and_convex_estimation_use_only_supplied_observations():
    rng = np.random.default_rng(42)
    x = rng.uniform(size=(90, 2))
    y = (2.0 * x[:, 1] - x[:, 0] > 0.5).astype(float)
    logistic = fit_logistic(x, y)
    convex = fit_convex(x, y)
    assert logistic["n_judgments"] == convex["n_judgments"] == 90
    assert logistic["coefficients"][0] < 0 < logistic["coefficients"][1]
    assert np.corrcoef(predict_logistic(logistic, x), 2 * x[:, 1] - x[:, 0])[0, 1] > 0.98
    assert convex["dense_weight"] > 0.5


def test_complete_experiment_uses_disjoint_folds_and_literal_count_solves(monkeypatch):
    monkeypatch.setattr(
        ceiling_slope,
        "GRID",
        np.array(
            [
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
                [1.0, 1.0],
            ]
        ),
    )
    prepared = _prepared(n_queries=8, n_docs=200, seed=15)
    result = ceiling_slope.run_dataset(
        prepared,
        n_folds=2,
        n_draws=2,
        retrieve_k=30,
    )
    assert result["quarter"]["scores_and_units"] == "fixed_from_full_corpus"
    json.dumps(result)
    for channel in ceiling_slope.CHANNELS:
        entry = result["channels"][channel]
        assert entry["max_relative_count_error"] < 1e-9
        assert entry["quarter"]["max_relative_count_error"] < 1e-9
        for plan in entry["judgments"]:
            for budget in ("sure", "stratified"):
                for observations in plan[budget]:
                    for qid, _, _ in observations:
                        qi = prepared.query_ids.index(qid)
                        assert result["folds"][qi] != plan["fold"]
        conventional = entry["calibration"]["conventional"]["all"]
        assert conventional["expected_relevant"] == pytest.approx(
            sum(result["training_counts"][f] for f in result["folds"]),
            rel=1e-9,
        )
        assert {"fit_ceiling", "fit_slope", "fit_both", "sure30", "stratified90"} <= set(
            entry["calibration"].keys() | entry["unavailable"].keys(),
        )
