#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""An independent tail refresh must retain other paper experiments."""

import copy

import pytest

from benchmarks.merge_results import merge_results
from benchmarks.reevaluation import run_dataset
from benchmarks.tail_experiments import refresh_tail_experiments
from tests.test_reevaluation_helpers import _prepared


def test_tail_refresh_agrees_with_a_complete_run_and_preserves_density_results():
    prepared = _prepared(n_docs=200, n_queries=8, seed=15)
    original = run_dataset(prepared, retrieve_k=30, n_folds=2)
    refreshed = refresh_tail_experiments(prepared, copy.deepcopy(original), retrieve_k=30)
    for name in ("Additive-Query-Tail", "Additive-Query-BothTail", "Rank-Product"):
        assert original["ranking"][name] == refreshed["ranking"][name]
    assert (
        original["ranking"]["Additive-Query-DensityRatio"]
        == refreshed["ranking"]["Additive-Query-DensityRatio"]
    )
    for name in ("Fused-Query-Consistent-Rmean", "Fused-Tail-Consistent-Rmean"):
        for pool in ("all", "pool", "top10"):
            expected = original["calibration"]["fused"][name][pool]
            actual = refreshed["calibration"]["fused"][name][pool]
            for metric in ("log_loss", "count_ratio", "calibration_slope"):
                assert actual[metric] == pytest.approx(expected[metric], nan_ok=True)
    assert original["fits"] == refreshed["fits"]
    assert "Additive-Query-FitTail" not in original["ranking"]


def test_merge_requires_disjoint_datasets_and_matching_metadata():
    first = {"model": "encoder", "identity": "text", "results": {"one": {"n": 1}}}
    second = {"model": "encoder", "identity": "text", "results": {"two": {"n": 2}}}
    assert set(merge_results([first, second])["results"]) == {"one", "two"}
    with pytest.raises(ValueError, match="overlap"):
        merge_results([first, first])
    with pytest.raises(ValueError, match="metadata"):
        merge_results([first, dict(second, identity="id")])
