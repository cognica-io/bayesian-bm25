#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Protect the original two-signal, five-collection benchmark contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_bm25.fusion import AttentionLogOddsWeights
from benchmarks import base_rate, hybrid_beir
from benchmarks.hybrid_beir import (
    DEFAULT_DATASETS,
    REQUIRED_DATASET_FILES,
    fusion_rrf,
    validate_dataset_inputs,
)
from benchmarks.ranking_eval import evaluate_run, ranked_run


def _dataset_inputs(root, datasets=DEFAULT_DATASETS):
    for name in datasets:
        for relative in REQUIRED_DATASET_FILES:
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="ascii")


def test_default_protocol_has_original_five_collections_and_one_encoder():
    args = hybrid_beir.build_argument_parser().parse_args([])
    assert args.datasets == ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"]
    assert args.model == "all-MiniLM-L6-v2"
    assert not hasattr(args, "model2")
    assert args.retrieve_k == 1000
    assert args.top_k == 10
    assert args.identity == "text"


def test_only_an_explicit_user_choice_changes_the_collection_set():
    args = hybrid_beir.build_argument_parser().parse_args(["--datasets", "arguana", "scidocs"])
    assert args.datasets == ["arguana", "scidocs"]


@pytest.mark.parametrize("arguments", [
    ["--model2", "all-mpnet-base-v2"],
    ["--model", "all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    ["--dense-calibration", "platt"],
])
def test_three_signal_experiment_options_cannot_change_this_protocol(arguments):
    with pytest.raises(SystemExit) as error:
        hybrid_beir.build_argument_parser().parse_args(arguments)
    assert error.value.code == 2


def test_all_requested_collections_are_validated(tmp_path):
    _dataset_inputs(tmp_path)
    validate_dataset_inputs(tmp_path, list(DEFAULT_DATASETS))


@pytest.mark.parametrize("missing", ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"])
def test_missing_collection_aborts_before_any_benchmark(tmp_path, monkeypatch, missing):
    _dataset_inputs(tmp_path, [name for name in DEFAULT_DATASETS if name != missing])
    calls = []
    monkeypatch.setattr(hybrid_beir, "run_dataset", lambda *a, **k: calls.append(a))
    with pytest.raises(FileNotFoundError, match=missing):
        hybrid_beir.main(["--beir-dir", str(tmp_path)])
    assert not calls


def test_missing_qrels_cannot_silently_remove_a_dataset(tmp_path):
    _dataset_inputs(tmp_path)
    (tmp_path / "scidocs" / "qrels" / "test.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="scidocs/qrels/test.tsv"):
        validate_dataset_inputs(tmp_path, list(DEFAULT_DATASETS))


@pytest.mark.parametrize("datasets", [[], ["fiqa", "fiqa"]])
def test_empty_or_duplicate_collection_sets_are_rejected(tmp_path, datasets):
    with pytest.raises(ValueError):
        validate_dataset_inputs(tmp_path, datasets)


@pytest.mark.parametrize("option", ["--top-k", "--retrieve-k"])
def test_invalid_depth_aborts_before_loading_data(tmp_path, option):
    with pytest.raises(ValueError, match="depths must be positive"):
        hybrid_beir.main(["--beir-dir", str(tmp_path), option, "0"])


def test_actual_entrypoint_runs_all_five_with_the_same_single_encoder(tmp_path, monkeypatch):
    _dataset_inputs(tmp_path)
    output = tmp_path / "results.json"
    calls = []
    checkpoints = []

    def run_dataset(directory, name, model, **options):
        if output.exists():
            checkpoints.append(json.loads(output.read_text(encoding="ascii"))["protocol"])
        calls.append((Path(directory).name, name, model, options))
        return {"BM25": {"NDCG@10": 0.3}}, {}, {}, {}

    monkeypatch.setattr(hybrid_beir, "run_dataset", run_dataset)
    monkeypatch.setattr(hybrid_beir, "print_results_table", lambda *a, **k: None)
    hybrid_beir.main(["--beir-dir", str(tmp_path), "--output", str(output)])
    assert [name for _, name, _, _ in calls] == list(DEFAULT_DATASETS)
    assert all(model == "all-MiniLM-L6-v2" for _, _, model, _ in calls)
    assert all(options["identity"] == "text" for _, _, _, options in calls)
    assert all(checkpoint["status"] == "running" for checkpoint in checkpoints)
    result = json.loads(output.read_text(encoding="ascii"))
    assert result["protocol"]["signal_count"] == 2
    assert result["protocol"]["signals"] == ["BM25", "all-MiniLM-L6-v2"]
    assert result["protocol"]["datasets_requested"] == list(DEFAULT_DATASETS)
    assert result["protocol"]["datasets_completed"] == list(DEFAULT_DATASETS)
    assert result["protocol"]["status"] == "complete"
    assert list(result["results"]) == list(DEFAULT_DATASETS)


def test_held_out_queries_are_scored_even_without_judgments_in_the_candidate_pool():
    corpus_ids = [f"d{i}" for i in range(20)]
    cache = {}
    for query_index in range(12):
        labels = np.zeros(20)
        judged = np.zeros(20, dtype=bool)
        if query_index != 0:
            labels[0] = 1.0
            judged[0] = True
        cache[str(query_index)] = {
            "union_idx": np.arange(20),
            "cand_log_odds_br": np.linspace(-2.0, 2.0, 20),
            "cand_dense_log_odds": np.linspace(1.0, -1.0, 20),
            "cand_labels": labels,
            "cand_judged": judged,
            "features": np.array([1.0, 0.2, 0.5]),
        }
    run = hybrid_beir.cross_validated_attention_run(
        "test", cache, corpus_ids, "features",
        lambda: AttentionLogOddsWeights(n_signals=2, n_query_features=3),
        normalize=False, n_folds=2, max_iterations=1,
    )
    assert run is not None
    assert set(run) == set(cache)
    assert len(run["0"]) == len(corpus_ids)


def test_rrf_missing_documents_contribute_exactly_zero():
    actual = fusion_rrf(np.array([1, 0, 2, 0]), np.array([0, 1, 2, 0]))
    np.testing.assert_allclose(actual, [1 / 61, 1 / 61, 2 / 62, 0.0])


def test_adding_unretrieved_candidates_cannot_change_rrf_scores():
    lexical = np.array([1, 0, 2])
    dense = np.array([0, 1, 2])
    before = fusion_rrf(lexical, dense)
    after = fusion_rrf(np.append(lexical, 0), np.append(dense, 0))
    np.testing.assert_array_equal(after[:3], before)
    assert after[3] == 0.0


def test_rrf_agrees_with_independent_ranked_list_reference():
    rng = np.random.default_rng(42)
    expected = np.zeros(100)
    arrays = []
    for length in [30, 75]:
        positions = rng.choice(100, size=length, replace=False)
        ranks = np.zeros(100, dtype=np.int64)
        ranks[positions] = np.arange(1, length + 1)
        arrays.append(ranks)
        for rank, position in enumerate(positions, 1):
            expected[position] += 1.0 / (60 + rank)
    np.testing.assert_allclose(fusion_rrf(*arrays), expected)


def test_empty_retrieval_runs_have_no_rrf_contribution():
    np.testing.assert_array_equal(fusion_rrf(np.zeros(3), np.zeros(3)), np.zeros(3))
    assert fusion_rrf(np.array([]), np.array([])).size == 0


def test_map_counts_unretrieved_relevant_documents():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    qrels = {"q": {"retrieved": 1, "missing": 1}}
    result = pytrec_eval.RelevanceEvaluator(qrels, {"map"}).evaluate({"q": {"retrieved": 1.0}})
    assert result["q"]["map"] == 0.5


def test_partial_tuning_results_do_not_become_a_full_collection_average(capsys):
    metrics = {"NDCG@10": 0.5, "MAP@10": 0.4, "Recall@10": 0.6}
    hybrid_beir.print_results_table({
        "arguana": {"BM25": metrics},
        "fiqa": {"BM25": metrics, "Bayesian-Balanced-Tuned": metrics},
    })
    output = capsys.readouterr().out
    averages = [line for line in output.splitlines() if line.startswith("Average")]
    assert len(averages) == 3
    assert all("n/a (1/2)" in line for line in averages)
    assert "n/a (1/2 datasets)" in output


def test_ranking_keeps_distinctions_destroyed_by_probability_rounding():
    from bayesian_bm25.probability import sigmoid

    log_odds = np.array([40.0, 41.0])
    assert sigmoid(log_odds)[0] == sigmoid(log_odds)[1] == 1.0
    run = ranked_run(["z", "a"], log_odds)
    result = evaluate_run({"q": {"a": 1}}, {"q": run}, k=1)
    assert result["NDCG@1"]["q"] == 1.0


def test_sparse_ranking_uses_all_qrels_for_ndcg_and_map():
    dataset = SimpleNamespace(doc_ids=["a"], qrels={"q": {"a": 1, "missed": 1}})
    ndcg, ap, precision = base_rate._ranking_metrics(
        np.array([1.0]), dataset, "q", np.array([True]), 10,
    )
    assert ndcg == pytest.approx(1.0 / (1.0 + 1.0 / np.log2(3.0)))
    assert ap == 0.5
    assert precision == 0.1


def test_sparse_calibration_keeps_matching_documents_with_underflowed_probability():
    dataset = SimpleNamespace(
        doc_ids=["a", "b"], queries=[("q", ["word"])], qrels={"q": {"a": 1, "b": 1}},
    )
    scorer = SimpleNamespace(
        _bm25=SimpleNamespace(get_scores=lambda tokens: np.ones(2)),
        get_probabilities=lambda tokens: np.array([0.0, 0.5]),
        get_log_odds=lambda tokens: np.array([-1000.0, 0.0]),
    )
    result = base_rate.evaluate_bayesian(scorer, dataset)
    assert result["Brier"] == pytest.approx(0.625)
    assert result["ECE"] == pytest.approx(0.75)
