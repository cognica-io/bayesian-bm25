#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Refresh tail-unit measurements without repeating independent density fits.

Density fits and bulk-unit slopes are independent of these measurements.
Solo slopes in tail units replay the original CV training samples.
"""

from __future__ import annotations

import time

import numpy as np

from bayesian_bm25.consistency import solve_offset
from bayesian_bm25.query_calibration import score_dispersion
from bayesian_bm25.tail_scale import tail_scale
from benchmarks.evidence_protocol import TAIL_DEPTH
from benchmarks.prepared_dataset import PreparedDataset
from benchmarks.query_signals import query_signals
from benchmarks.ranking_eval import evaluate_run, mean_metrics, ranked_run
from benchmarks.reevaluation import (
    POOLED_SCALE,
    _accumulators,
    _summaries,
    _top_k,
    _update_pools,
    fit_kappas,
    fused_log_odds,
    lexical_scale,
    rank_product,
    tail_units,
    training_sample,
)


def refresh_tail_experiments(
    prepared: PreparedDataset,
    result: dict,
    *,
    k: int = 10,
    retrieve_k: int = 1000,
    seed: int = 42,
) -> dict:
    started = time.perf_counter()
    if prepared.dataset_fingerprint != result["dataset_fingerprint"]:
        raise ValueError("tail refresh requires the identical population and judgments")
    rng = np.random.default_rng(seed)
    n_folds = len(result["fits"])
    folds = np.empty(prepared.n_queries, dtype=np.int64)
    folds[rng.permutation(prepared.n_queries)] = np.arange(prepared.n_queries) % n_folds
    corpus_ids = np.asarray(prepared.corpus_ids)
    runs: dict[str, dict] = {}
    vector_acc = _accumulators(["Vec-Tail-R1"], ["retrieved", "top10"])
    names = [
        "Fused-Query-Consistent-Rmean",
        "Fused-Tail-Consistent-R1",
        "Fused-Tail-Consistent-Rmean",
    ]
    fused_acc = _accumulators(names, ["all", "pool", "top10"])
    train_features, train_labels, train_weights = [], [], []
    ratios_lex, ratios_dense, fallback_lex, fallback_dense = [], [], [], []

    def add(name, sig, scores):
        runs.setdefault(name, {})[sig.qid] = ranked_run(
            corpus_ids[sig.doc_index[sig.candidates]],
            scores,
        )

    for qi in range(prepared.n_queries):
        sig = query_signals(prepared, qi, retrieve_k)
        sigma_lex = lexical_scale(sig, 1.0 / prepared.corpus_alpha)
        sigma_vec = score_dispersion(sig.dense)
        tail_lex, tail_vec = tail_units(sig, TAIL_DEPTH, fallback_lexical_scale=sigma_lex)
        z_lex, z_vec = sig.bm25 / sigma_lex, sig.dense / sigma_vec
        cand = sig.candidates
        add("Additive-Query-Tail", sig, z_lex[cand] + tail_vec[cand])
        add("Additive-Query-BothTail", sig, tail_lex[cand] + tail_vec[cand])
        add("Rank-Product", sig, rank_product(sig))
        for depth in (10, 30, 100, 300, 1000):
            _, dense = tail_units(sig, depth, fallback_lexical_scale=sigma_lex)
            add(f"Additive-Query-Tail{depth}", sig, z_lex[cand] + dense[cand])
        retrieved = cand[sig.dense_rank > 0]
        head = retrieved[_top_k(sig.dense[retrieved], k)]
        _update_pools(
            vector_acc["Vec-Tail-R1"],
            tail_vec - solve_offset(tail_vec, 1.0),
            sig.labels,
            {"retrieved": retrieved, "top10": head},
        )
        r_mean = result["fits"][int(folds[qi])]["relevant_mean"]
        for name, vector, count in (
            ("Fused-Query-Consistent-Rmean", z_vec, r_mean),
            ("Fused-Tail-Consistent-R1", tail_vec, 1.0),
            ("Fused-Tail-Consistent-Rmean", tail_vec, r_mean),
        ):
            values = fused_log_odds(z_lex, vector, count, scale=POOLED_SCALE)
            _update_pools(
                fused_acc[name],
                values,
                sig.labels,
                {"all": None, "pool": cand, "top10": _top_k(z_lex + vector, k)},
            )
        lexical = sig.bm25[sig.matched]
        unit_lex = tail_scale(lexical, k=TAIL_DEPTH)
        unit_vec = tail_scale(sig.dense, k=TAIL_DEPTH)
        fallback_lex.append(unit_lex <= 0.0)
        fallback_dense.append(unit_vec <= 0.0)
        if unit_lex > 0.0:
            ratios_lex.append(sigma_lex / unit_lex)
        if unit_vec > 0.0:
            ratios_dense.append(sigma_vec / unit_vec)

        # Replay the preceding sampled-offset diagnostic's RNG operations.
        matched = np.flatnonzero(sig.matched)
        if len(matched):
            top = matched[_top_k(sig.bm25[matched], retrieve_k)]
            if len(top) < len(sig.bm25):
                rest = np.flatnonzero(~np.isin(np.arange(len(sig.bm25)), top))
                rng.choice(rest, size=min(1000, len(rest)), replace=False)
        positions, weights = training_sample(sig, rng)
        train_features.append(np.column_stack([tail_lex[positions], tail_vec[positions]]))
        train_labels.append(sig.labels[positions])
        train_weights.append(weights)

    for fold in range(n_folds):
        train = np.flatnonzero(folds != fold)
        for column, key in ((slice(0, 1), "theta_lex"), (slice(1, 2), "theta_vec")):
            result["fits"][fold][key] = float(
                fit_kappas(
                    [train_features[i][:, column] for i in train],
                    [train_labels[i] for i in train],
                    [train_weights[i] for i in train],
                )[0]
            )

    for name, run in runs.items():
        metrics = evaluate_run(prepared.qrels, run, k=k)
        result["ranking"][name] = {
            "mean": mean_metrics(metrics),
            "per_query": {
                metric: [values[qid] for qid in prepared.query_ids]
                for metric, values in metrics.items()
            },
        }
    result["calibration"]["vector"].update(_summaries(vector_acc))
    result["calibration"]["fused"].update(_summaries(fused_acc))
    result["diagnostics"].update(
        {
            "sigma_over_tail_lex_median": float(np.median(ratios_lex)),
            "sigma_over_tail_vec_median": float(np.median(ratios_dense)),
            "lexical_tail_fallback_fraction": float(np.mean(fallback_lex)),
            "dense_tail_fallback_fraction": float(np.mean(fallback_dense)),
        }
    )
    result["tail_depth"] = TAIL_DEPTH
    result["rank_product_tie_policy"] = "population_position_with_unmatched_score_zero"
    result.setdefault("timing", {})["tail_refresh_seconds"] = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    print(f"  {prepared.name}: tail depth {TAIL_DEPTH} refreshed ({elapsed:.1f}s)", flush=True)
    return result
