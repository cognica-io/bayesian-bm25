#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Additional equal-budget baselines; not part of the paper model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from bayesian_bm25.calibration_accumulator import CalibrationAccumulator
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.count_calibration import calibrate_count
from benchmarks.additional.judgment_baselines import (
    fit_convex,
    fit_logistic,
    predict_convex,
    predict_logistic,
)
from benchmarks.ceiling_query import CeilingQuery, make_ceiling_query
from benchmarks.ceiling_slope import (
    CHANNELS,
    _base_logits,
    _catalog,
    _judgment_record,
    _loss,
    _sampling_plans,
    _summarize_evaluations,
)
from benchmarks.prepared_dataset import prepare_dataset
from benchmarks.query_signals import query_signals
from benchmarks.ranking_eval import evaluate_run, mean_metrics, ranked_run


def _convex_features(query: CeilingQuery, positions: np.ndarray) -> np.ndarray:
    candidates = query.signals.candidates
    columns = []
    for scores in (query.signals.bm25, query.signals.dense):
        lo, hi = float(np.min(scores[candidates])), float(np.max(scores[candidates]))
        columns.append(
            np.clip((scores[positions] - lo) / (hi - lo), 0.0, 1.0)
            if hi > lo
            else np.zeros(len(positions))
        )
    return np.column_stack(columns)


def _logistic_features(query: CeilingQuery, count: float, positions: np.ndarray) -> np.ndarray:
    columns = []
    for channel in ("lexical", "dense"):
        evidence = query.evidence(channel)
        model = calibrate_count(evidence, count)
        columns.append(model.log_odds(evidence[positions]))
    return np.column_stack(columns)


def _feature_columns(features: np.ndarray, channel: str) -> np.ndarray:
    if channel == "lexical":
        return features[:, :1]
    if channel == "dense":
        return features[:, 1:]
    return features


def run_equal_budget(
    queries: list[CeilingQuery],
    channel: str,
    catalog: dict,
    plans: list[dict],
    folds: np.ndarray,
    counts: np.ndarray,
    *,
    k: int,
    corpus_ids: np.ndarray,
    qrels: dict,
) -> dict:
    """Fit and evaluate on precisely the observations used by each budget row."""
    n_draws = len(plans[0]["sure"])
    parameters = {}
    for budget in ("sure", "stratified"):
        parameters[f"logistic_{budget}"] = [[None for _ in plans] for _ in range(n_draws)]
        if channel == "fused":
            parameters[f"convex_{budget}"] = [[None for _ in plans] for _ in range(n_draws)]
    for fold, plan in enumerate(plans):
        selected = np.unique(np.concatenate(plan["sure"] + plan["stratified"]))
        features = np.empty((len(selected), 2))
        convex = np.empty_like(features)
        for qi, query in enumerate(queries):
            start, end = catalog["starts"][qi : qi + 2]
            local = np.flatnonzero((selected >= start) & (selected < end))
            if not len(local):
                continue
            positions = catalog["positions"][qi][selected[local] - start]
            features[local] = _logistic_features(query, float(counts[fold]), positions)
            convex[local] = _convex_features(query, positions)
        for budget in ("sure", "stratified"):
            for draw, observations in enumerate(plan[budget]):
                if not len(observations):
                    continue
                selected_positions = np.searchsorted(selected, observations)
                labels = catalog["labels"][observations]
                parameters[f"logistic_{budget}"][draw][fold] = fit_logistic(
                    _feature_columns(features[selected_positions], channel),
                    labels,
                )
                if channel == "fused":
                    parameters[f"convex_{budget}"][draw][fold] = fit_convex(
                        convex[selected_positions],
                        labels,
                    )

    unavailable = {
        name: {
            "status": "insufficient_judgments",
            "missing_fold_draw_pairs": [
                [fold, draw]
                for draw, fits in enumerate(draws)
                for fold, fit in enumerate(fits)
                if fit is None
            ],
        }
        for name, draws in parameters.items()
        if any(fit is None for fits in draws for fit in fits)
    }
    available = {name: draws for name, draws in parameters.items() if name not in unavailable}
    context = CalibrationContext(
        reference_population="query_excluded_corpus",
        count_source="training_mean_in_features",
        slope_source="same_document_judgments",
        ceiling_source="not_applicable",
        relevance_target="judged_relevance",
    )
    accumulators = {
        name: [
            {pool: CalibrationAccumulator(context=context) for pool in ("all", "pool", "top10")}
            for _ in draws
        ]
        for name, draws in available.items()
    }
    statistics = {
        name: [{pool: [] for pool in ("all", "pool", "top10")} for _ in draws]
        for name, draws in available.items()
    }
    runs = {name: [{} for _ in draws] for name, draws in available.items()}
    for qi, query in enumerate(queries):
        fold = int(folds[qi])
        all_positions = np.arange(len(query.signals.labels))
        features = _feature_columns(
            _logistic_features(query, float(counts[fold]), all_positions),
            channel,
        )
        convex = _convex_features(query, all_positions)
        candidates = query.positions(channel)
        for name, draws in available.items():
            for draw, fits in enumerate(draws):
                if name.startswith("convex"):
                    scores, logits = predict_convex(fits[fold], convex)
                else:
                    logits = predict_logistic(fits[fold], features)
                    scores = logits
                head = candidates[np.argsort(-scores[candidates], kind="stable")[:k]]
                for pool, positions in {
                    "all": all_positions,
                    "pool": candidates,
                    "top10": head,
                }.items():
                    x, y = logits[positions], query.signals.labels[positions]
                    accumulators[name][draw][pool].update(x, y)
                    statistics[name][draw][pool].append(
                        [
                            len(y),
                            float(np.sum(y)),
                            float(np.sum(expit(x))),
                            float(np.sum(_loss(x, y))),
                        ]
                    )
                runs[name][draw][query.signals.qid] = ranked_run(
                    corpus_ids[query.signals.doc_index[candidates]],
                    scores[candidates],
                    depth=k,
                )
    ranking = {}
    for name, draws in runs.items():
        values = [evaluate_run(qrels, run, k=k) for run in draws]
        means = [mean_metrics(v) for v in values]
        ranking[name] = {
            "mean": {metric: float(np.mean([m[metric] for m in means])) for metric in means[0]},
            "per_query": {
                metric: [
                    float(np.mean([v[metric][q.signals.qid] for v in values])) for q in queries
                ]
                for metric in means[0]
            },
            "n_draws": n_draws,
        }
    return {
        "calibration": _summarize_evaluations(accumulators, statistics),
        "parameters": parameters,
        "ranking": ranking,
        "unavailable": unavailable,
        "features": "count_constrained_single_signal_log_odds",
        "convex_objective": "same_sample_log_loss_with_fitted_monotone_logistic_map",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--beir-dir", required=True)
    parser.add_argument("--paper-results", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.paper_results).resolve():
        raise ValueError("additional results require a separate output file")
    paper = json.loads(Path(args.paper_results).read_text(encoding="ascii"))
    payload = {
        "experiment": "additional_equal_budget_baselines",
        "model": paper["model"],
        "identity": paper["identity"],
        "tail_depth": paper["tail_depth"],
        "paper_results": args.paper_results,
        "results": {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, result in paper["results"].items():
        prepared = prepare_dataset(
            str(Path(args.beir_dir) / name),
            name,
            paper["model"],
            str(Path(args.beir_dir) / ".cache" / "embeddings"),
            identity=paper["identity"],
        )
        if prepared.dataset_fingerprint != result["dataset_fingerprint"]:
            raise ValueError("additional comparisons require the same population and labels")
        queries = [
            make_ceiling_query(
                query_signals(prepared, qi, paper["retrieve_k"]), 1.0 / prepared.corpus_alpha
            )
            for qi in range(prepared.n_queries)
        ]
        folds, counts = np.asarray(result["folds"]), np.asarray(result["training_counts"])
        payload["results"][name] = {}
        for ci, channel in enumerate(CHANNELS):
            catalog = _catalog(queries, channel)
            plans = _sampling_plans(
                _base_logits(queries, channel, catalog, counts),
                catalog,
                folds,
                n_draws=paper["n_draws"],
                seed=paper["seed"] + ci * 1000,
            )
            if (
                _judgment_record(queries, catalog, plans)
                != result["channels"][channel]["judgments"]
            ):
                raise ValueError(
                    "additional comparisons must use exactly the saved paper judgments"
                )
            payload["results"][name][channel] = run_equal_budget(
                queries,
                channel,
                catalog,
                plans,
                folds,
                counts,
                k=paper["k"],
                corpus_ids=np.asarray(prepared.corpus_ids),
                qrels=prepared.qrels,
            )
        temporary = Path(str(output) + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")
        temporary.replace(output)


if __name__ == "__main__":
    main()
