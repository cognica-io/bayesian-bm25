#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Reproduce Section 6.8 of From Scores to Evidence.

All intercepts use the query-excluded corpus, never the retrieved count.
Judgment budgets are additional to the training-mean relevant count.
Unjudged documents have label zero. Canonical queries define CV folds.
Random quarters retain the original scores and the original score units.

Usage:
    python -m benchmarks.ceiling_slope -d benchmarks/.cache/beir -o results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import expit

from bayesian_bm25.calibration_accumulator import CalibrationAccumulator
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.count_calibration import calibrate_count, ceiling_log_odds
from benchmarks.ceiling_query import CeilingQuery, make_ceiling_query
from benchmarks.evidence_protocol import QUARTER_SEED, TAIL_DEPTH
from benchmarks.logistic_population import LogisticPopulation
from benchmarks.prepared_dataset import PreparedDataset, prepare_dataset
from benchmarks.query_signals import query_signals

CEILINGS = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
SLOPES = 2.0 ** (np.arange(-6, 7) / 4.0)
GRID = np.array([(w, k) for k in SLOPES for w in CEILINGS])
CHANNELS = ("lexical", "dense", "fused")
DATASETS = ("arguana", "fiqa", "nfcorpus", "scidocs", "scifact")
PROTOCOL_VERSION = 3
GRID_VERSION = 2


def _loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 1.0, np.logaddexp(0.0, -logits), np.logaddexp(0.0, logits))


def _folds(n_queries: int, n_folds: int, seed: int) -> np.ndarray:
    if not 2 <= n_folds <= n_queries:
        raise ValueError("fold count must lie between 2 and the canonical query count")
    folds = np.empty(n_queries, dtype=np.int64)
    folds[np.random.default_rng(seed).permutation(n_queries)] = np.arange(n_queries) % n_folds
    return folds


def _training_counts(queries: list[CeilingQuery], folds: np.ndarray) -> np.ndarray:
    counts = np.array([q.signals.n_relevant for q in queries], dtype=np.float64)
    result = np.array([np.mean(counts[folds != f]) for f in range(int(np.max(folds)) + 1)])
    if np.any(result <= 0.0):
        raise ValueError(
            "a training fold has zero relevance count; the nondegenerate model is undefined"
        )
    return result


def _catalog(queries: list[CeilingQuery], channel: str) -> dict:
    positions = [q.positions(channel) for q in queries]
    starts = np.cumsum([0] + [len(p) for p in positions])
    return {
        "positions": positions,
        "starts": starts,
        "query": np.repeat(np.arange(len(queries)), np.diff(starts)),
        "labels": np.concatenate(
            [q.signals.labels[p] for q, p in zip(queries, positions, strict=True)]
        ),
    }


def _base_logits(
    queries: list[CeilingQuery],
    channel: str,
    catalog: dict,
    counts: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(counts), len(catalog["labels"])))
    for qi, query in enumerate(queries):
        evidence = query.evidence(channel)
        positions = catalog["positions"][qi]
        start, end = catalog["starts"][qi : qi + 2]
        for fold, count in enumerate(counts):
            model = calibrate_count(evidence, float(count))
            result[fold, start:end] = model.log_odds(evidence[positions])
    return result


def _sample(
    available: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # No replacement: using a judgment twice cannot enlarge a label budget.
    return np.sort(rng.choice(available, size=min(budget, len(available)), replace=False))


def _sampling_plans(
    logits: np.ndarray,
    catalog: dict,
    folds: np.ndarray,
    *,
    n_draws: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    plans = []
    for fold, values in enumerate(logits):
        training = folds[catalog["query"]] != fold
        p = expit(values)
        strata = [
            np.flatnonzero(training & (p >= 0.9)),
            np.flatnonzero(training & (p >= 0.3) & (p < 0.9)),
            np.flatnonzero(training & (p >= 0.03) & (p < 0.3)),
        ]
        plans.append(
            {
                "sure": [_sample(strata[0], 30, rng) for _ in range(n_draws)],
                "stratified": [
                    np.sort(np.concatenate([_sample(s, 30, rng) for s in strata]))
                    for _ in range(n_draws)
                ],
                "available_by_stratum": [len(s) for s in strata],
                "sure_relevant_fraction": (
                    float(np.mean(catalog["labels"][strata[0]])) if len(strata[0]) else None
                ),
            }
        )
    return plans


def _grid_logits(
    population: LogisticPopulation,
    evidence: np.ndarray,
    positions: np.ndarray,
    count: float,
) -> np.ndarray:
    predictions = np.empty((len(GRID), len(positions)))
    for gi, (ceiling, slope) in enumerate(GRID):
        offset = population.solve(float(slope), float(count / ceiling))
        predictions[gi] = ceiling_log_odds(slope * evidence[positions] - offset, float(ceiling))
    return predictions


def _fit_grids(
    queries: list[CeilingQuery],
    quarters: list[CeilingQuery],
    channel: str,
    catalog: dict,
    plans: list[dict],
    folds: np.ndarray,
    counts: np.ndarray,
    quarter_counts: np.ndarray,
    *,
    n_draws: int,
) -> dict:
    """Training NLL for every fixed grid pair and held-out query fold."""
    n_folds, n_queries = len(counts), len(queries)
    losses = np.empty((n_folds, n_queries, len(GRID)))
    quarter_losses = np.empty_like(losses)
    sample_losses = np.zeros((n_folds, n_draws, len(GRID)))
    global_losses = np.zeros(len(GRID))
    global_count = float(np.mean([q.signals.n_relevant for q in queries]))
    max_bound, fallbacks, node_total, population_total = 0.0, 0, 0, 0
    started = time.perf_counter()
    for qi, (query, quarter) in enumerate(zip(queries, quarters, strict=True)):
        evidence = query.evidence(channel)
        positions = catalog["positions"][qi]
        labels = query.signals.labels[positions]
        population = LogisticPopulation(evidence)
        part_evidence = quarter.evidence(channel)
        part_positions = quarter.positions(channel)
        part_labels = quarter.signals.labels[part_positions]
        part = LogisticPopulation(part_evidence)
        start, end = catalog["starts"][qi : qi + 2]
        for fold in range(n_folds):
            point_losses = _loss(
                _grid_logits(population, evidence, positions, float(counts[fold])),
                labels,
            )
            losses[fold, qi] = np.sum(point_losses, axis=1)
            for draw, observations in enumerate(plans[fold]["stratified"]):
                local = observations[(observations >= start) & (observations < end)] - start
                if len(local):
                    sample_losses[fold, draw] += np.sum(point_losses[:, local], axis=1)
            quarter_losses[fold, qi] = np.sum(
                _loss(
                    _grid_logits(part, part_evidence, part_positions, float(quarter_counts[fold])),
                    part_labels,
                ),
                axis=1,
            )
        global_losses += np.sum(
            _loss(_grid_logits(population, evidence, positions, global_count), labels),
            axis=1,
        )
        for approximation in (population, part):
            max_bound = max(max_bound, approximation.max_relative_count_bound)
            fallbacks += approximation.fallback_count
            node_total += len(approximation.nodes)
            population_total += approximation.population
        if (qi + 1) % 100 == 0:
            print(
                f"    {channel} grid {qi + 1}/{n_queries} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    return {
        "losses": losses,
        "quarter_losses": quarter_losses,
        "sample_losses": sample_losses,
        "global_losses": global_losses,
        "max_relative_count_bound": np.array(max_bound),
        "fallback_count": np.array(fallbacks),
        "node_fraction": np.array(node_total / population_total),
    }


def _best(losses: np.ndarray, *, mode: str = "both") -> dict:
    allowed = np.ones(len(GRID), dtype=bool)
    if mode == "ceiling":
        allowed = GRID[:, 1] == 1.0
    elif mode == "slope":
        allowed = GRID[:, 0] == 1.0
    elif mode != "both":
        raise ValueError("unknown grid fit mode")
    indices = np.flatnonzero(allowed)
    position = indices[int(np.argmin(losses[indices]))]
    return {"ceiling": float(GRID[position, 0]), "slope": float(GRID[position, 1])}


def _sure_ceiling(labels: np.ndarray, observations: np.ndarray) -> dict:
    raw = float(np.mean(labels[observations])) if len(observations) else None
    return {
        "ceiling": raw,
        "slope": 1.0,
        "raw_ceiling_estimate": raw,
        "n_judgments": len(observations),
        "requested_judgments": 30,
        "status": "complete_budget" if len(observations) == 30 else "insufficient_sure_documents",
        "estimator": "observed_relevant_fraction",
        "source": "judgments" if raw is not None else "unavailable_no_sure_documents",
        "boundary_projection": False,
    }


def _parameter_rows(
    fitted: dict,
    plans: list[dict],
    catalog: dict,
    folds: np.ndarray,
    counts: np.ndarray,
    *,
    n_draws: int,
) -> tuple[dict, dict]:
    rows = {"conventional": [[{"ceiling": 1.0, "slope": 1.0} for _ in counts]]}
    quarter_rows = {"conventional": rows["conventional"]}
    for mode in ("ceiling", "slope", "both"):
        parameters = [
            _best(np.sum(fitted["losses"][f, folds != f], axis=0), mode=mode)
            for f in range(len(counts))
        ]
        rows[f"fit_{mode}"] = [parameters]
        if mode in ("ceiling", "both"):
            quarter_rows[f"full_{mode}"] = [parameters]
    quarter_rows["quarter_both"] = [
        [_best(np.sum(fitted["quarter_losses"][f, folds != f], axis=0)) for f in range(len(counts))]
    ]
    rows["sure30"] = [
        [_sure_ceiling(catalog["labels"], p["sure"][draw]) for p in plans]
        for draw in range(n_draws)
    ]
    rows["stratified90"] = [
        [
            dict(
                _best(fitted["sample_losses"][f, draw]),
                n_judgments=len(plans[f]["stratified"][draw]),
                requested_judgments=90,
            )
            for f in range(len(counts))
        ]
        for draw in range(n_draws)
    ]
    return rows, quarter_rows


def _bootstrap(statistics: np.ndarray, *, seed: int = 42, n_draws: int = 1000) -> dict:
    """Query bootstrap of sums: n, observed, expected and negative log likelihood."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(statistics), size=(n_draws, len(statistics)))
    summed = np.sum(statistics[draws], axis=1)
    n, observed, expected, loss = summed.T
    prevalence = observed / n
    entropy = -(
        prevalence * np.log(np.maximum(prevalence, 1e-300))
        + (1.0 - prevalence) * np.log(np.maximum(1.0 - prevalence, 1e-300))
    )
    skill = 1.0 - np.divide(loss / n, entropy, out=np.full(n_draws, np.nan), where=entropy > 0)
    ratio = np.divide(expected, observed, out=np.full(n_draws, np.nan), where=observed > 0)

    def interval(values):
        finite = values[np.isfinite(values)]
        return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]

    return {
        "unit": "canonical_query",
        "n_resamples": n_draws,
        "log_loss_skill_95ci": interval(skill),
        "count_ratio_95ci": interval(ratio),
    }


def _evaluate_rows(
    queries: list[CeilingQuery],
    channel: str,
    folds: np.ndarray,
    counts: np.ndarray,
    rows: dict,
    *,
    k: int = 10,
) -> dict:
    """Evaluate chosen constants using literal full-population count solves."""
    requested_rows = rows
    unavailable = {}
    for name, draws in rows.items():
        invalid = []
        for draw, parameters in enumerate(draws):
            for fold, chosen in enumerate(parameters):
                cap = chosen["ceiling"]
                minimum_population = min(
                    len(q.signals.labels) for qi, q in enumerate(queries) if folds[qi] == fold
                )
                if cap is None or not 0.0 < cap <= 1.0 or counts[fold] >= cap * minimum_population:
                    invalid.append([draw, fold])
        if invalid:
            unavailable[name] = {
                "status": "empirical_ceiling_infeasible_for_supplied_count",
                "draw_fold_pairs": invalid,
            }
    rows = {name: draws for name, draws in rows.items() if name not in unavailable}
    context = CalibrationContext(
        reference_population="query_excluded_corpus",
        count_source="training_mean",
        slope_source="recorded_by_method",
        ceiling_source="recorded_by_method",
        relevance_target="judged_relevance",
    )
    accumulators = {
        name: [
            {pool: CalibrationAccumulator(context=context) for pool in ("all", "pool", "top10")}
            for _ in draws
        ]
        for name, draws in rows.items()
    }
    per_query = {
        name: [{pool: [] for pool in ("all", "pool", "top10")} for _ in draws]
        for name, draws in rows.items()
    }
    count_errors = []
    for qi, query in enumerate(queries):
        fold = int(folds[qi])
        evidence = query.evidence(channel)
        labels = query.signals.labels
        positions = {
            "all": np.arange(len(labels)),
            "pool": query.positions(channel),
            "top10": query.head(channel, k),
        }
        cache = {}
        for name, draws in rows.items():
            for draw, parameters in enumerate(draws):
                chosen = parameters[fold]
                ceiling, slope = chosen["ceiling"], chosen["slope"]
                count = float(counts[fold])
                key = (ceiling, slope, count)
                if key not in cache:
                    score = slope * evidence
                    model = calibrate_count(score, count, ceiling=ceiling, context=context)
                    cache[key] = model.log_odds(score)
                    count_errors.append(abs(float(np.sum(expit(cache[key]))) / count - 1.0))
                logits = cache[key]
                for pool, indices in positions.items():
                    x, y = logits[indices], labels[indices]
                    accumulators[name][draw][pool].update(x, y)
                    per_query[name][draw][pool].append(
                        [
                            len(y),
                            float(np.sum(y)),
                            float(np.sum(expit(x))),
                            float(np.sum(_loss(x, y))),
                        ]
                    )
    return {
        "calibration": _summarize_evaluations(accumulators, per_query),
        "parameters": requested_rows,
        "unavailable": unavailable,
        "max_relative_count_error": max(count_errors, default=0.0),
    }


def _summarize_evaluations(accumulators: dict, per_query: dict) -> dict:
    result = {}
    for name, draws in accumulators.items():
        pools = {}
        for pool in ("all", "pool", "top10"):
            summaries = [d[pool].summary() for d in draws]
            summary = dict(summaries[0])
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    summary[key] = float(np.mean([s[key] for s in summaries]))
            statistics = np.asarray([d[pool] for d in per_query[name]])
            summary["uncertainty"] = _bootstrap(np.mean(statistics, axis=0))
            summary["draw_sd_log_loss_skill"] = float(
                np.std([s["log_loss_skill"] for s in summaries])
            )
            summary["n_draws"] = len(draws)
            summary["per_query_sums"] = np.mean(statistics, axis=0).tolist()
            summary["evaluation_population"] = {
                "all": "query_excluded_corpus",
                "pool": "channel_result_list",
                "top10": "top10_of_channel_result_list",
            }[pool]
            pools[pool] = summary
        result[name] = pools
    return result


def _judgment_record(queries: list[CeilingQuery], catalog: dict, plans: list[dict]) -> list[dict]:
    records = []
    for fold, plan in enumerate(plans):
        entry = {
            "fold": fold,
            "available_by_stratum": plan["available_by_stratum"],
            "sure_relevant_fraction": plan["sure_relevant_fraction"],
        }
        for budget in ("sure", "stratified"):
            entry[budget] = []
            for observations in plan[budget]:
                pairs = []
                for position in observations:
                    qi = int(catalog["query"][position])
                    query = queries[qi]
                    local = position - catalog["starts"][qi]
                    document = catalog["positions"][qi][local]
                    pairs.append(
                        [
                            query.signals.qid,
                            int(query.signals.doc_index[document]),
                            int(catalog["labels"][position]),
                        ]
                    )
                entry[budget].append(pairs)
        records.append(entry)
    return records


def _count_sensitivity(
    queries: list[CeilingQuery],
    folds: np.ndarray,
    counts: np.ndarray,
    *,
    k: int,
    seed: int,
) -> dict:
    catalog = _catalog(queries, "fused")
    result = {}
    conventions = [("R1", np.ones_like(counts))] + [
        (f"mean_x{multiple:g}", multiple * counts) for multiple in (0.25, 0.5, 1.0, 2.0, 4.0)
    ]
    for index, (name, supplied) in enumerate(conventions):
        base = _base_logits(queries, "fused", catalog, supplied)
        plans = _sampling_plans(base, catalog, folds, n_draws=1, seed=seed + index)
        rows = {
            "conventional": [[{"ceiling": 1.0, "slope": 1.0} for _ in counts]],
            "sure30": [[_sure_ceiling(catalog["labels"], p["sure"][0]) for p in plans]],
        }
        result[name] = _evaluate_rows(queries, "fused", folds, supplied, rows, k=k)
        result[name]["supplied_counts"] = supplied.tolist()
        result[name]["judgments"] = _judgment_record(queries, catalog, plans)
    return result


def run_dataset(
    prepared: PreparedDataset,
    *,
    k: int = 10,
    retrieve_k: int = 1000,
    n_folds: int = 5,
    n_draws: int = 10,
    seed: int = 42,
    grid_cache_dir: str | None = None,
) -> dict:
    """Fit constants and evaluate held-out canonical queries on full and quarter corpora."""
    started = time.perf_counter()
    folds = _folds(prepared.n_queries, n_folds, seed)
    queries = [
        make_ceiling_query(query_signals(prepared, i, retrieve_k), 1.0 / prepared.corpus_alpha)
        for i in range(prepared.n_queries)
    ]
    kept = np.sort(
        np.random.default_rng(QUARTER_SEED).choice(
            prepared.n_docs,
            size=prepared.n_docs // 4,
            replace=False,
        )
    )
    quarters = [q.restrict(kept, retrieve_k) for q in queries]
    counts = _training_counts(queries, folds)
    quarter_counts = _training_counts(quarters, folds)
    result = {
        "dataset": prepared.name,
        "identity": prepared.identity,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "n_docs": prepared.n_docs,
        "n_queries": prepared.n_queries,
        "query_ids": prepared.query_ids,
        "corpus_ids": prepared.corpus_ids,
        "folds": folds.tolist(),
        "training_counts": counts.tolist(),
        "quarter_training_counts": quarter_counts.tolist(),
        "channels": {},
        "reference_population": "query_excluded_corpus",
        "relevance_target": "judged_relevance",
        "quarter": {
            "kept_document_indices": kept.tolist(),
            "n_docs": len(kept),
            "mean_relevant": float(np.mean([q.signals.n_relevant for q in quarters])),
            "zero_relevant_query_fraction": float(
                np.mean([q.signals.n_relevant == 0 for q in quarters])
            ),
            "scores_and_units": "fixed_from_full_corpus",
        },
    }
    for ci, channel in enumerate(CHANNELS):
        stage = time.perf_counter()
        catalog = _catalog(queries, channel)
        base = _base_logits(queries, channel, catalog, counts)
        plans = _sampling_plans(base, catalog, folds, n_draws=n_draws, seed=seed + ci * 1000)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "protocol_version": GRID_VERSION,
                    "dataset": prepared.dataset_fingerprint,
                    "model": prepared.model_name,
                    "channel": channel,
                    "folds": n_folds,
                    "draws": n_draws,
                    "seed": seed,
                    "retrieve_k": retrieve_k,
                    "tail_depth": TAIL_DEPTH,
                    "quarter_seed": QUARTER_SEED,
                    "grid": GRID.tolist(),
                    "identity": prepared.identity,
                },
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        cache_path = None
        if grid_cache_dir is not None:
            Path(grid_cache_dir).mkdir(parents=True, exist_ok=True)
            cache_path = Path(grid_cache_dir) / f"{prepared.name}-{channel}-{fingerprint}.npz"
        cache_hit = cache_path is not None and cache_path.exists()
        if cache_hit:
            with np.load(cache_path) as cached:
                fitted = dict(cached)
            print(f"    {channel}: grid cache hit", flush=True)
        else:
            fitted = _fit_grids(
                queries,
                quarters,
                channel,
                catalog,
                plans,
                folds,
                counts,
                quarter_counts,
                n_draws=n_draws,
            )
            if cache_path is not None:
                np.savez_compressed(cache_path, **fitted)
        fit_seconds = time.perf_counter() - stage
        evaluation_start = time.perf_counter()
        rows, quarter_rows = _parameter_rows(
            fitted,
            plans,
            catalog,
            folds,
            counts,
            n_draws=n_draws,
        )
        channel_result = _evaluate_rows(queries, channel, folds, counts, rows, k=k)
        channel_result["quarter"] = _evaluate_rows(
            quarters,
            channel,
            folds,
            quarter_counts,
            quarter_rows,
            k=k,
        )
        channel_result["global_fits"] = {
            mode: _best(fitted["global_losses"], mode=mode) for mode in ("ceiling", "slope", "both")
        }
        channel_result["global_grid_loss"] = fitted["global_losses"].tolist()
        channel_result["n_list_observations"] = len(catalog["labels"])
        channel_result["judgments"] = _judgment_record(queries, catalog, plans)
        channel_result["grid_solver"] = {
            key: float(fitted[key])
            for key in ("max_relative_count_bound", "fallback_count", "node_fraction")
        }
        all_logits = base[folds[catalog["query"]], np.arange(len(catalog["labels"]))]
        sure = all_logits >= math.log(9.0)
        channel_result["sure_documents"] = {
            "count": int(np.sum(sure)),
            "relevant_fraction": float(np.mean(catalog["labels"][sure])) if np.any(sure) else None,
        }
        channel_result["timing"] = {
            "grid_and_sampling_seconds": fit_seconds,
            "evaluation_seconds": time.perf_counter() - evaluation_start,
            "grid_cache_hit": cache_hit,
        }
        result["channels"][channel] = channel_result
        print(
            f"  {prepared.name} {channel}: finished ({time.perf_counter() - stage:.1f}s)",
            flush=True,
        )
    result["count_sensitivity"] = _count_sensitivity(queries, folds, counts, k=k, seed=seed + 10000)
    result["timing"] = {
        "preparation_seconds": prepared.preparation_seconds,
        "experiments_seconds": time.perf_counter() - started,
        "total_seconds": prepared.preparation_seconds + time.perf_counter() - started,
    }
    return result


def add_collection_transfer(
    prepared: PreparedDataset,
    results: dict,
    *,
    k: int,
    retrieve_k: int,
) -> None:
    """Leave-one-collection-out constants, selected without target labels."""
    name = prepared.name
    others = [r for dataset, r in results.items() if dataset != name]
    if not others:
        results[name]["transfer_status"] = "requires_another_collection"
        return
    folds = np.asarray(results[name]["folds"])
    counts = np.asarray(results[name]["training_counts"])
    queries = [
        make_ceiling_query(query_signals(prepared, i, retrieve_k), 1.0 / prepared.corpus_alpha)
        for i in range(prepared.n_queries)
    ]
    for channel in CHANNELS:
        source_losses = [np.asarray(r["channels"][channel]["global_grid_loss"]) for r in others]
        losses = np.mean([loss / np.min(loss) for loss in source_losses], axis=0)
        rows = {
            f"transfer_{mode}": [[_best(losses, mode=mode) for _ in counts]]
            for mode in ("ceiling", "both")
        }
        results[name]["channels"][channel]["transfer"] = _evaluate_rows(
            queries,
            channel,
            folds,
            counts,
            rows,
            k=k,
        )
    results[name]["transfer_status"] = "complete"
    results[name]["transfer_source_collections"] = [r["dataset"] for r in others]
    results[name]["transfer_objective"] = "mean_source_relative_log_loss_regret"


def _checkpoint_protocol(options: dict) -> dict:
    protocol = {
        key: options[key]
        for key in (
            "model",
            "identity",
            "folds",
            "draws",
            "retrieve_k",
            "top_k",
        )
    }
    return dict(
        protocol,
        protocol_version=PROTOCOL_VERSION,
        tail_depth=TAIL_DEPTH,
        quarter_seed=QUARTER_SEED,
    )


def _run_worker(options: dict, name: str) -> tuple[str, dict]:
    print(f"\n=== {name}: ceiling and slope ===", flush=True)
    prepared = prepare_dataset(
        os.path.join(options["beir_dir"], name),
        name,
        options["model"],
        options["cache_dir"],
        identity=options["identity"],
    )
    try:
        result = run_dataset(
            prepared,
            k=options["top_k"],
            retrieve_k=options["retrieve_k"],
            n_folds=options["folds"],
            n_draws=options["draws"],
            grid_cache_dir=options["grid_cache_dir"],
        )
        checkpoint = Path(options["output"] + ".datasets")
        checkpoint.mkdir(parents=True, exist_ok=True)
        protocol = _checkpoint_protocol(options)
        temporary = checkpoint / f"{name}.json.tmp"
        temporary.write_text(json.dumps({"protocol": protocol, "result": result}), encoding="ascii")
        temporary.replace(checkpoint / f"{name}.json")
        return name, result
    except Exception:
        traceback.print_exc()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--beir-dir", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--identity", choices=["id", "text"], default="text")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--grid-cache-dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("-R", "--retrieve-k", type=int, default=1000)
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Independent dataset processes")
    args = parser.parse_args()
    cache_dir = args.cache_dir or os.path.join(args.beir_dir, ".cache", "embeddings")
    grid_cache = args.grid_cache_dir or os.path.join(args.beir_dir, ".cache", "ceiling_slope")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model": args.model,
        "identity": args.identity,
        "identity_version": 1,
        "k": args.top_k,
        "retrieve_k": args.retrieve_k,
        "n_folds": args.folds,
        "n_draws": args.draws,
        "seed": 42,
        "tail_depth": TAIL_DEPTH,
        "quarter_seed": QUARTER_SEED,
        "sure_ceiling_estimator": "observed_relevant_fraction",
        "fit_count_protocol": "training_fold_mean_for_all_training_and_test_predictions",
        "ceiling_grid": CEILINGS.tolist(),
        "slope_grid": SLOPES.tolist(),
        "unjudged_label": 0,
        "count_source": "training_mean",
        "relevance_target": "judged_relevance",
        "small_strata": "all_available_without_replacement_and_actual_budget_recorded",
    }
    results = {}
    if args.resume and Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in metadata.items()):
            raise SystemExit("resume metadata differs from this protocol")
        results = previous["results"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = Path(args.output + ".tmp")
        temporary.write_text(
            json.dumps(dict(metadata, results=results), ensure_ascii=True), encoding="ascii"
        )
        temporary.replace(args.output)

    options = dict(vars(args), cache_dir=cache_dir, grid_cache_dir=grid_cache)
    if args.resume:
        for name in args.datasets:
            checkpoint = Path(args.output + ".datasets") / f"{name}.json"
            if name in results or not checkpoint.exists():
                continue
            restored = json.loads(checkpoint.read_text(encoding="ascii"))
            if restored["protocol"] == _checkpoint_protocol(options):
                results[name] = restored["result"]
        save()
    pending = [name for name in args.datasets if name not in results]
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    if args.workers == 1:
        for name in pending:
            _, results[name] = _run_worker(options, name)
            save()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(_run_worker, options, name) for name in pending]
            for job in as_completed(jobs):
                name, results[name] = job.result()
                save()
    for name in args.datasets:
        if results[name].get("transfer_status") == "complete" and set(
            results[name].get("transfer_source_collections", [])
        ) == set(results) - {name}:
            continue
        prepared = prepare_dataset(
            os.path.join(args.beir_dir, name),
            name,
            args.model,
            cache_dir,
            identity=args.identity,
        )
        add_collection_transfer(prepared, results, k=args.top_k, retrieve_k=args.retrieve_k)
        del prepared
        save()


if __name__ == "__main__":
    main()
