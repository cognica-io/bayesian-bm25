#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Base rate prior comparison.

Compares probability calibration and ranking quality across different
base rate configurations:

  1. Raw BM25 (baseline)
  2. Bayesian BM25 -- no base rate
  3. Bayesian BM25 -- base_rate="auto"
  4. Bayesian BM25 -- explicit base_rate values (0.001, 0.01, 0.05, 0.1)
  5. Bayesian BM25 (batch fit) -- no base rate
  6. Bayesian BM25 (batch fit) -- base_rate="auto"

Datasets: BEIR/NFCorpus, BEIR/SciFact (via ir_datasets).

Requires: pip install ir_datasets
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import bm25s
import numpy as np
import pytrec_eval

from bayesian_bm25.probability import sigmoid
from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.calibration import minmax_normalize, platt_scaling_fit
from benchmarks.metrics import (
    best_f1_threshold,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
    threshold_f1,
)
from benchmarks.ranking_eval import ranked_run
from benchmarks.utils import IRDataset, load_beir_dataset


def _ranking_metrics(scores, dataset, qid, matched, k):
    """Evaluate a top-1000 run against all of the query's judgments."""
    positions = np.flatnonzero(matched)
    run = ranked_run(
        [dataset.doc_ids[i] for i in positions], np.asarray(scores)[positions], depth=1000,
    )
    if not run:
        return 0.0, 0.0, 0.0
    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: dataset.qrels[qid]}, {f"ndcg_cut_{k}", "map", f"P_{k}"},
    )
    row = evaluator.evaluate({qid: run})[qid]
    return row[f"ndcg_cut_{k}"], row["map"], row[f"P_{k}"]


def evaluate_bm25_raw(
    bm25_model: bm25s.BM25,
    dataset: IRDataset,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate raw BM25 ranking (no calibration, so no ECE)."""
    ndcgs, aps, precisions = [], [], []

    for qid, qtokens in dataset.queries:
        scores = bm25_model.get_scores(qtokens)
        ndcg, ap, precision = _ranking_metrics(scores, dataset, qid, scores > 0, k)
        ndcgs.append(ndcg)
        aps.append(ap)
        precisions.append(precision)

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"P@{k}": float(np.mean(precisions)),
        "MAP": float(np.mean(aps)),
    }


def evaluate_bayesian(
    scorer: BayesianBM25Scorer,
    dataset: IRDataset,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate Bayesian BM25 with ranking + calibration metrics."""
    ndcgs, aps, precisions = [], [], []
    all_probs, all_labels = [], []

    for qid, qtokens in dataset.queries:
        qrel = dataset.qrels[qid]
        probs = scorer.get_probabilities(qtokens)
        nonzero = scorer._bm25.get_scores(qtokens) > 0
        ndcg, ap, precision = _ranking_metrics(
            scorer.get_log_odds(qtokens), dataset, qid, nonzero, k,
        )
        ndcgs.append(ndcg)
        aps.append(ap)
        precisions.append(precision)
        if np.any(nonzero):
            nonzero_dids = [dataset.doc_ids[i] for i, nz in enumerate(nonzero) if nz]
            nonzero_labels = [float(qrel.get(did, 0) >= 1) for did in nonzero_dids]
            all_probs.extend(probs[nonzero].tolist())
            all_labels.extend(nonzero_labels)

    probs_arr = np.array(all_probs) if all_probs else np.array([])
    labels_arr = np.array(all_labels) if all_labels else np.array([])

    ece = expected_calibration_error(probs_arr, labels_arr) if len(probs_arr) > 0 else 1.0
    brier = brier_score(probs_arr, labels_arr) if len(probs_arr) > 0 else 1.0

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"P@{k}": float(np.mean(precisions)),
        "MAP": float(np.mean(aps)),
        "ECE": ece,
        "Brier": brier,
    }


def collect_training_data(
    bm25_model: bm25s.BM25,
    dataset: IRDataset,
    query_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (score, label) pairs for parameter fitting."""
    qid_to_tokens = {qid: qt for qid, qt in dataset.queries}
    all_scores, all_labels = [], []

    for qid in query_ids:
        qtokens = qid_to_tokens[qid]
        qrel = dataset.qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        nonzero = scores > 0
        if not np.any(nonzero):
            continue
        nonzero_scores = scores[nonzero]
        nonzero_dids = [dataset.doc_ids[i] for i, nz in enumerate(nonzero) if nz]
        nonzero_labels = np.array([float(qrel.get(did, 0) >= 1) for did in nonzero_dids])
        all_scores.extend(nonzero_scores.tolist())
        all_labels.extend(nonzero_labels.tolist())

    return np.array(all_scores), np.array(all_labels)


# ---------------------------------------------------------------------------
# Threshold transfer test
# ---------------------------------------------------------------------------

def evaluate_threshold_transfer(
    scorer: BayesianBM25Scorer,
    dataset: IRDataset,
    train_qids: list[str],
    test_qids: list[str],
) -> tuple[float, float, float]:
    """Find best threshold on train queries, evaluate on test queries.

    Returns (train_f1, test_f1, gap).
    """
    qid_to_tokens = {qid: qt for qid, qt in dataset.queries}

    def collect(qids):
        vals, labs = [], []
        for qid in qids:
            qt = qid_to_tokens[qid]
            qrel = dataset.qrels[qid]
            scores = scorer.get_log_odds(qt)
            nz = scorer._bm25.get_scores(qt) > 0
            if not np.any(nz):
                continue
            nz_dids = [dataset.doc_ids[i] for i, n in enumerate(nz) if n]
            nz_labs = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
            vals.extend(scores[nz].tolist())
            labs.extend(nz_labs)
        return np.array(vals), np.array(labs)

    train_v, train_l = collect(train_qids)
    test_v, test_l = collect(test_qids)
    t, train_f1 = best_f1_threshold(train_v, train_l)
    _, _, test_f1 = threshold_f1(test_v, test_l, t)

    return train_f1, test_f1, train_f1 - test_f1


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_base_rate_comparison(dataset: IRDataset, k: int = 10) -> dict:
    """Compare base rate prior configurations on a single dataset."""
    print(f"\n{'=' * 78}")
    print(f"Dataset: {dataset.name.upper()}")
    print(f"{'=' * 78}")

    # Split queries: first half for training, second for evaluation
    all_qids = [qid for qid, _ in dataset.queries]
    rng = np.random.default_rng(42)
    rng.shuffle(all_qids)
    mid = len(all_qids) // 2
    train_qids = all_qids[:mid]
    eval_qids = all_qids[mid:]

    eval_queries = [(qid, qt) for qid, qt in dataset.queries if qid in set(eval_qids)]
    eval_dataset = IRDataset(
        name=dataset.name,
        corpus_tokens=dataset.corpus_tokens,
        doc_ids=dataset.doc_ids,
        queries=eval_queries,
        qrels=dataset.qrels,
    )

    print(f"  Train: {len(train_qids)} queries, Eval: {len(eval_qids)} queries")

    # Build shared BM25 index for raw baseline and training data collection
    bm25_model = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25_model.index(dataset.corpus_tokens, show_progress=False)

    # Collect training data for batch fit
    train_scores, train_labels = collect_training_data(bm25_model, dataset, train_qids)
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    actual_base_rate = n_pos / len(train_labels) if len(train_labels) > 0 else 0.0
    print(f"  Training data: {len(train_scores)} samples (pos={n_pos}, neg={n_neg})")
    print(f"  Actual relevance rate in training data: {actual_base_rate:.6f}")

    # -----------------------------------------------------------------------
    # Evaluate all configurations
    # -----------------------------------------------------------------------
    results: list[tuple[str, dict[str, float]]] = []
    threshold_results: list[tuple[str, float, float, float]] = []

    # 1. Raw BM25 (baseline)
    print(f"\n{'--' * 39}")
    print("1. Raw BM25 (baseline)")
    raw = evaluate_bm25_raw(bm25_model, eval_dataset, k=k)
    results.append(("Raw BM25", raw))

    # 2. Bayesian -- no base rate
    print("2. Bayesian (auto, no base rate)")
    scorer_no_br = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=None)
    scorer_no_br.index(dataset.corpus_tokens, show_progress=False)
    print(f"   alpha={scorer_no_br._transform.alpha:.4f}, "
          f"beta={scorer_no_br._transform.beta:.4f}, "
          f"base_rate=None")
    r = evaluate_bayesian(scorer_no_br, eval_dataset, k=k)
    results.append(("Bayesian (no base rate)", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(scorer_no_br, dataset, train_qids, eval_qids)
    threshold_results.append(("Bayesian (no base rate)", tr_f1, te_f1, gap))

    # 3. Bayesian -- base_rate="auto"
    print("3. Bayesian (auto, base_rate=auto)")
    scorer_auto_br = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate="auto")
    scorer_auto_br.index(dataset.corpus_tokens, show_progress=False)
    print(f"   alpha={scorer_auto_br._transform.alpha:.4f}, "
          f"beta={scorer_auto_br._transform.beta:.4f}, "
          f"base_rate={scorer_auto_br.base_rate:.6f}")
    r = evaluate_bayesian(scorer_auto_br, eval_dataset, k=k)
    results.append(("Bayesian (base_rate=auto)", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(scorer_auto_br, dataset, train_qids, eval_qids)
    threshold_results.append(("Bayesian (base_rate=auto)", tr_f1, te_f1, gap))

    # 4. Bayesian -- explicit base rates
    explicit_rates = [0.001, 0.01, 0.05, 0.1]
    for br_val in explicit_rates:
        label = f"Bayesian (base_rate={br_val})"
        print(f"4. {label}")
        scorer_br = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=br_val)
        scorer_br.index(dataset.corpus_tokens, show_progress=False)
        r = evaluate_bayesian(scorer_br, eval_dataset, k=k)
        results.append((label, r))
        tr_f1, te_f1, gap = evaluate_threshold_transfer(scorer_br, dataset, train_qids, eval_qids)
        threshold_results.append((label, tr_f1, te_f1, gap))

    # 5. Bayesian (batch fit) -- no base rate
    print("5. Bayesian (batch fit, no base rate)")
    scorer_fit_no_br = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=None)
    scorer_fit_no_br.index(dataset.corpus_tokens, show_progress=False)
    scorer_fit_no_br._transform.fit(
        train_scores, train_labels,
        learning_rate=0.05, max_iterations=3000,
    )
    print(f"   alpha={scorer_fit_no_br._transform.alpha:.4f}, "
          f"beta={scorer_fit_no_br._transform.beta:.4f}, "
          f"base_rate=None")
    r = evaluate_bayesian(scorer_fit_no_br, eval_dataset, k=k)
    results.append(("Batch fit (no base rate)", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(
        scorer_fit_no_br, dataset, train_qids, eval_qids
    )
    threshold_results.append(("Batch fit (no base rate)", tr_f1, te_f1, gap))

    # 6. Bayesian (batch fit) -- base_rate="auto"
    print("6. Bayesian (batch fit, base_rate=auto)")
    scorer_fit_auto_br = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate="auto")
    scorer_fit_auto_br.index(dataset.corpus_tokens, show_progress=False)
    scorer_fit_auto_br._transform.fit(
        train_scores, train_labels,
        learning_rate=0.05, max_iterations=3000,
    )
    print(f"   alpha={scorer_fit_auto_br._transform.alpha:.4f}, "
          f"beta={scorer_fit_auto_br._transform.beta:.4f}, "
          f"base_rate={scorer_fit_auto_br.base_rate:.6f}")
    r = evaluate_bayesian(scorer_fit_auto_br, eval_dataset, k=k)
    results.append(("Batch fit (base_rate=auto)", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(
        scorer_fit_auto_br, dataset, train_qids, eval_qids
    )
    threshold_results.append(("Batch fit (base_rate=auto)", tr_f1, te_f1, gap))

    # 7. Bayesian (batch fit) -- base_rate = actual relevance rate
    print("7. Bayesian (batch fit, base_rate=actual)")
    br_actual = max(1e-6, min(actual_base_rate, 0.5))
    scorer_fit_actual = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=br_actual)
    scorer_fit_actual.index(dataset.corpus_tokens, show_progress=False)
    scorer_fit_actual._transform.fit(
        train_scores, train_labels,
        learning_rate=0.05, max_iterations=3000,
    )
    print(f"   alpha={scorer_fit_actual._transform.alpha:.4f}, "
          f"beta={scorer_fit_actual._transform.beta:.4f}, "
          f"base_rate={scorer_fit_actual.base_rate:.6f} (actual)")
    r = evaluate_bayesian(scorer_fit_actual, eval_dataset, k=k)
    results.append((f"Batch fit (base_rate={br_actual:.4f})", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(
        scorer_fit_actual, dataset, train_qids, eval_qids
    )
    threshold_results.append((f"Batch fit (base_rate={br_actual:.4f})", tr_f1, te_f1, gap))

    # 8. Platt scaling baseline
    print("8. Platt scaling (baseline)")
    platt_A, platt_B = platt_scaling_fit(train_scores, train_labels)
    print(f"   Platt A={platt_A:.4f}, B={platt_B:.4f}")

    platt_all_probs, platt_all_labels, platt_all_logits = [], [], []
    platt_ndcgs, platt_aps, platt_precisions = [], [], []
    for qid, qtokens in eval_dataset.queries:
        qrel = dataset.qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        platt_probs = np.array(sigmoid(platt_A * scores + platt_B))
        ndcg, ap, precision = _ranking_metrics(
            platt_A * scores + platt_B, dataset, qid, scores > 0, k,
        )
        platt_ndcgs.append(ndcg)
        platt_aps.append(ap)
        platt_precisions.append(precision)
        nonzero = scores > 0
        if np.any(nonzero):
            nz_dids = [dataset.doc_ids[i] for i, nz in enumerate(nonzero) if nz]
            nz_labels = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
            platt_all_probs.extend(platt_probs[nonzero].tolist())
            platt_all_labels.extend(nz_labels)
            platt_all_logits.extend((platt_A * scores[nonzero] + platt_B).tolist())

    platt_p_arr = np.array(platt_all_probs) if platt_all_probs else np.array([])
    platt_l_arr = np.array(platt_all_labels) if platt_all_labels else np.array([])
    platt_ece = (
        expected_calibration_error(platt_p_arr, platt_l_arr)
        if len(platt_p_arr) > 0
        else 1.0
    )
    platt_brier = brier_score(platt_p_arr, platt_l_arr) if len(platt_p_arr) > 0 else 1.0
    r_platt = {
        f"NDCG@{k}": float(np.mean(platt_ndcgs)),
        f"P@{k}": float(np.mean(platt_precisions)),
        "MAP": float(np.mean(platt_aps)),
        "ECE": platt_ece,
        "Brier": platt_brier,
    }
    results.append(("Platt scaling", r_platt))

    # Platt threshold transfer
    platt_train_logits = platt_A * train_scores + platt_B
    platt_t, platt_train_f1 = best_f1_threshold(platt_train_logits, train_labels)
    _, _, platt_test_f1 = threshold_f1(np.array(platt_all_logits), platt_l_arr, platt_t)
    threshold_results.append((
        "Platt scaling", platt_train_f1, platt_test_f1, platt_train_f1 - platt_test_f1
    ))

    # 9. Min-max normalization baseline
    print("9. Min-max normalization (baseline)")
    minmax_all_probs, minmax_all_labels, minmax_all_scores = [], [], []
    minmax_ndcgs, minmax_aps, minmax_precisions = [], [], []
    for qid, qtokens in eval_dataset.queries:
        qrel = dataset.qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        minmax_probs = minmax_normalize(scores, train_scores)
        ndcg, ap, precision = _ranking_metrics(scores, dataset, qid, scores > 0, k)
        minmax_ndcgs.append(ndcg)
        minmax_aps.append(ap)
        minmax_precisions.append(precision)
        nonzero = scores > 0
        if np.any(nonzero):
            nz_dids = [dataset.doc_ids[i] for i, nz in enumerate(nonzero) if nz]
            nz_labels = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
            minmax_all_probs.extend(minmax_probs[nonzero].tolist())
            minmax_all_labels.extend(nz_labels)
            minmax_all_scores.extend(scores[nonzero].tolist())

    minmax_p_arr = np.array(minmax_all_probs) if minmax_all_probs else np.array([])
    minmax_l_arr = np.array(minmax_all_labels) if minmax_all_labels else np.array([])
    minmax_ece = (
        expected_calibration_error(minmax_p_arr, minmax_l_arr)
        if len(minmax_p_arr) > 0
        else 1.0
    )
    minmax_brier = brier_score(minmax_p_arr, minmax_l_arr) if len(minmax_p_arr) > 0 else 1.0
    r_minmax = {
        f"NDCG@{k}": float(np.mean(minmax_ndcgs)),
        f"P@{k}": float(np.mean(minmax_precisions)),
        "MAP": float(np.mean(minmax_aps)),
        "ECE": minmax_ece,
        "Brier": minmax_brier,
    }
    results.append(("Min-max normalization", r_minmax))

    # Min-max threshold transfer
    minmax_t, minmax_train_f1 = best_f1_threshold(train_scores, train_labels)
    _, _, minmax_test_f1 = threshold_f1(np.array(minmax_all_scores), minmax_l_arr, minmax_t)
    threshold_results.append((
        "Min-max normalization",
        minmax_train_f1,
        minmax_test_f1,
        minmax_train_f1 - minmax_test_f1,
    ))

    # 10. Batch fit -- prior_aware mode (C2)
    print("10. Batch fit (prior_aware, C2)")
    scorer_fit_pa = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=None)
    scorer_fit_pa.index(dataset.corpus_tokens, show_progress=False)
    # Collect TF and doc_len_ratio for training data
    qid_to_tokens_map = {qid: qt for qid, qt in dataset.queries}
    train_tfs, train_dlrs = [], []
    for qid in train_qids:
        qtokens = qid_to_tokens_map[qid]
        qrel = dataset.qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        nonzero = scores > 0
        if not np.any(nonzero):
            continue
        for i, nz in enumerate(nonzero):
            if nz:
                doc_tokens = dataset.corpus_tokens[i]
                query_set = set(qtokens)
                tf_val = float(sum(1 for t in doc_tokens if t in query_set))
                dl = len(doc_tokens)
                avgdl = scorer_fit_pa.avgdl
                train_tfs.append(tf_val)
                train_dlrs.append(dl / avgdl)

    train_tfs_arr = np.array(train_tfs[:len(train_scores)])
    train_dlrs_arr = np.array(train_dlrs[:len(train_scores)])
    scorer_fit_pa._transform.fit(
        train_scores, train_labels,
        learning_rate=0.01, max_iterations=3000,
        mode="prior_aware",
        tfs=train_tfs_arr, doc_len_ratios=train_dlrs_arr,
    )
    print(f"   alpha={scorer_fit_pa._transform.alpha:.4f}, "
          f"beta={scorer_fit_pa._transform.beta:.4f}, mode=prior_aware")
    r = evaluate_bayesian(scorer_fit_pa, eval_dataset, k=k)
    results.append(("Batch fit (prior_aware)", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(scorer_fit_pa, dataset, train_qids, eval_qids)
    threshold_results.append(("Batch fit (prior_aware)", tr_f1, te_f1, gap))

    # 11. Batch fit -- prior_free mode (C3)
    print("11. Batch fit (prior_free, C3)")
    scorer_fit_pf = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=None)
    scorer_fit_pf.index(dataset.corpus_tokens, show_progress=False)
    scorer_fit_pf._transform.fit(
        train_scores, train_labels,
        learning_rate=0.05, max_iterations=3000,
        mode="prior_free",
    )
    print(f"   alpha={scorer_fit_pf._transform.alpha:.4f}, "
          f"beta={scorer_fit_pf._transform.beta:.4f}, mode=prior_free")
    r = evaluate_bayesian(scorer_fit_pf, eval_dataset, k=k)
    results.append(("Batch fit (prior_free)", r))
    tr_f1, te_f1, gap = evaluate_threshold_transfer(
        scorer_fit_pf, dataset, train_qids, eval_qids
    )
    threshold_results.append(("Batch fit (prior_free)", tr_f1, te_f1, gap))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"SUMMARY -- {dataset.name.upper()}")
    print(f"{'=' * 78}")

    col_w = 36
    hdr = (
        f"  {'Method':<{col_w}}  {'NDCG@10':>8}  {'MAP':>8}"
        f"  {'P@10':>8}  {'ECE':>8}  {'Brier':>8}"
    )
    print(f"\n{hdr}")
    print(f"  {'---' * 12}  {'---' * 3}  {'---' * 3}  {'---' * 3}  {'---' * 3}  {'---' * 3}"
          )

    for name, r in results:
        ndcg_str = f"{r[f'NDCG@{k}']:>8.4f}"
        map_str = f"{r['MAP']:>8.4f}"
        pk_str = f"{r[f'P@{k}']:>8.4f}"
        ece_str = f"{r['ECE']:>8.4f}" if "ECE" in r else "     n/a"
        brier_str = f"{r['Brier']:>8.4f}" if "Brier" in r else "     n/a"
        print(f"  {name:<{col_w}}  {ndcg_str}  {map_str}  {pk_str}  {ece_str}  {brier_str}")

    # -----------------------------------------------------------------------
    # Deltas vs "no base rate" baseline
    # -----------------------------------------------------------------------
    no_br = results[1][1]  # "Bayesian (no base rate)"
    print("\n  Calibration improvement vs 'no base rate':")
    delta_hdr = (
        f"  {'Method':<{col_w}}  {'ECE delta':>10}  {'ECE %':>8}"
        f"  {'Brier delta':>12}  {'Brier %':>8}"
    )
    print(delta_hdr)
    print(f"  {'---' * 12}  {'---' * 4}  {'---' * 3}  {'---' * 4}     {'---' * 3}")

    for name, r in results[2:]:
        if "ECE" not in r:
            continue
        ece_delta = r["ECE"] - no_br["ECE"]
        ece_pct = (ece_delta / no_br["ECE"]) * 100 if no_br["ECE"] != 0 else 0
        brier_delta = r["Brier"] - no_br["Brier"]
        brier_pct = (brier_delta / no_br["Brier"]) * 100 if no_br["Brier"] != 0 else 0
        sign_e = "+" if ece_delta >= 0 else ""
        sign_b = "+" if brier_delta >= 0 else ""
        print(f"  {name:<{col_w}}  {sign_e}{ece_delta:>9.4f}  {ece_pct:>+7.1f}%"
              f"  {sign_b}{brier_delta:>11.4f}  {brier_pct:>+7.1f}%")

    # -----------------------------------------------------------------------
    # Threshold transfer
    # -----------------------------------------------------------------------
    print("\n  Threshold transfer (train threshold -> test queries):")
    print(f"  {'Method':<{col_w}}  {'Train F1':>8}  {'Test F1':>8}  {'Gap':>8}")
    print(f"  {'---' * 12}  {'---' * 3}  {'---' * 3}  {'---' * 3}")

    for name, tr_f1, te_f1, gap in threshold_results:
        print(f"  {name:<{col_w}}  {tr_f1:>8.4f}  {te_f1:>8.4f}  {gap:>+8.4f}")
    print("  (Smaller gap = threshold generalises better)")

    # -----------------------------------------------------------------------
    # Reliability diagram
    # -----------------------------------------------------------------------
    print("\n  Reliability diagram (selected configurations):")
    selected = [
        ("Bayesian (no base rate)", scorer_no_br),
        ("Bayesian (base_rate=auto)", scorer_auto_br),
        ("Batch fit (no base rate)", scorer_fit_no_br),
        ("Batch fit (base_rate=auto)", scorer_fit_auto_br),
    ]

    for label, scorer in selected:
        all_probs, all_labels = [], []
        for qid, qtokens in eval_dataset.queries:
            qrel = dataset.qrels[qid]
            probs = scorer.get_probabilities(qtokens)
            nonzero = scorer._bm25.get_scores(qtokens) > 0
            if np.any(nonzero):
                nonzero_dids = [dataset.doc_ids[i] for i, nz in enumerate(nonzero) if nz]
                nonzero_labels = [float(qrel.get(did, 0) >= 1) for did in nonzero_dids]
                all_probs.extend(probs[nonzero].tolist())
                all_labels.extend(nonzero_labels)

        if not all_probs:
            continue

        bins = reliability_diagram(np.array(all_probs), np.array(all_labels), n_bins=10)
        print(f"\n  [{label}]")
        print(f"    {'Bin':>8}  {'Predicted':>10}  {'Actual':>10}  {'Count':>8}  {'|Error|':>8}")
        for avg_pred, avg_actual, count in bins:
            err = abs(avg_pred - avg_actual)
            print(
                f"    {avg_pred:>8.3f}  {avg_pred:>10.4f}"
                f"  {avg_actual:>10.4f}  {count:>8}  {err:>8.4f}"
            )

    recorded = {name: r for name, r in results}
    for name, train_f1, test_f1, gap in threshold_results:
        recorded[name].update({"Train F1": train_f1, "Test F1": test_f1, "F1 Gap": gap})
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base rate prior comparison benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("Base Rate Prior Comparison Benchmark")
    print("=" * 78)

    datasets_to_run = [
        ("nfcorpus", "test"),
        ("scifact", "test"),
    ]

    all_results = {}
    for ds_name, split in datasets_to_run:
        print(f"\nLoading {ds_name}...")
        dataset = load_beir_dataset(ds_name, split)
        all_results[ds_name] = run_base_rate_comparison(dataset, k=10)

    if args.output:
        output = {
            "benchmark": "base_rate",
            "protocol": {
                "query_split": "50/50, seed 42",
                "ranking_depth": 1000,
                "map": "full-run AP with all judged relevant documents in the denominator",
                "calibration_population": "matching documents, shared across methods",
                "threshold_selection": "training queries only, unsaturated scores",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
