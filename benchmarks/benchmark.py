#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Bayesian BM25 vs raw BM25 on standard IR datasets.

Compares ranking quality and probability calibration across:
  1. Raw BM25 (baseline)
  2. Bayesian BM25 with auto-estimated parameters
  3. Bayesian BM25 with batch-fitted parameters
  4. Bayesian BM25 with online-learned parameters

Datasets: BEIR/NFCorpus, BEIR/SciFact (via ir_datasets).

Requires: pip install ir_datasets
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import bm25s
import ir_datasets
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from bayesian_bm25.probability import BayesianProbabilityTransform, sigmoid
from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.metrics import (
    average_precision,
    expected_calibration_error,
    ndcg_at_k,
    precision_at_k,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class IRDataset:
    name: str
    corpus_tokens: list[list[str]]
    doc_ids: list[str]
    queries: list[tuple[str, list[str]]]   # (query_id, tokens)
    qrels: dict[str, dict[str, int]]       # query_id -> {doc_id -> relevance}


def load_beir_dataset(dataset_name: str, split: str = "test") -> IRDataset:
    """Load a BEIR dataset via ir_datasets."""
    ds_id = f"beir/{dataset_name}/{split}"
    print(f"  Loading {ds_id}...")
    ds = ir_datasets.load(ds_id)

    # Documents
    doc_id_list = []
    corpus_tokens = []
    for doc in ds.docs_iter():
        doc_id_list.append(doc.doc_id)
        text = doc.text
        if hasattr(doc, "title") and doc.title:
            text = doc.title + " " + text
        corpus_tokens.append(text.lower().split())

    # Queries
    queries = []
    for q in ds.queries_iter():
        queries.append((q.query_id, q.text.lower().split()))

    # Relevance judgments
    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    # Filter queries that have qrels
    queries = [(qid, qtokens) for qid, qtokens in queries if qid in qrels]

    print(f"    {len(corpus_tokens)} docs, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} qrels")

    return IRDataset(
        name=dataset_name,
        corpus_tokens=corpus_tokens,
        doc_ids=doc_id_list,
        queries=queries,
        qrels=qrels,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def doc_id_to_idx(doc_ids: list[str]) -> dict[str, int]:
    return {did: i for i, did in enumerate(doc_ids)}


def get_relevance_vector(
    ranked_doc_ids: list[str],
    qrel: dict[str, int],
    binary_threshold: int = 1,
) -> np.ndarray:
    """Convert ranked doc IDs to a relevance vector."""
    return np.array([
        float(qrel.get(did, 0) >= binary_threshold) for did in ranked_doc_ids
    ])


def get_graded_relevance_vector(
    ranked_doc_ids: list[str],
    qrel: dict[str, int],
) -> np.ndarray:
    """Convert ranked doc IDs to graded relevance vector."""
    return np.array([float(qrel.get(did, 0)) for did in ranked_doc_ids])


def evaluate_bm25_raw(
    bm25_model: bm25s.BM25,
    dataset: IRDataset,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate raw BM25 ranking."""
    did_to_idx = doc_id_to_idx(dataset.doc_ids)
    ndcgs, aps, precisions = [], [], []

    for qid, qtokens in dataset.queries:
        qrel = dataset.qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        top_k_idx = np.argsort(-scores)[:k]
        ranked_dids = [dataset.doc_ids[i] for i in top_k_idx]

        graded = get_graded_relevance_vector(ranked_dids, qrel)
        binary = get_relevance_vector(ranked_dids, qrel)

        ndcgs.append(ndcg_at_k(graded, k))
        aps.append(average_precision(binary))
        precisions.append(precision_at_k(binary, k))

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
    """Evaluate Bayesian BM25 ranking and calibration."""
    ndcgs, aps, precisions = [], [], []
    all_probs, all_labels = [], []

    for qid, qtokens in dataset.queries:
        qrel = dataset.qrels[qid]
        probs = scorer.get_probabilities(qtokens)
        top_k_idx = np.argsort(-probs)[:k]
        ranked_dids = [dataset.doc_ids[i] for i in top_k_idx]

        graded = get_graded_relevance_vector(ranked_dids, qrel)
        binary = get_relevance_vector(ranked_dids, qrel)

        ndcgs.append(ndcg_at_k(graded, k))
        aps.append(average_precision(binary))
        precisions.append(precision_at_k(binary, k))

        # Calibration: score all nonzero-probability documents
        nonzero = probs > 0
        if np.any(nonzero):
            nonzero_dids = [dataset.doc_ids[i] for i, nz in enumerate(nonzero) if nz]
            nonzero_labels = [float(qrel.get(did, 0) >= 1) for did in nonzero_dids]
            all_probs.extend(probs[nonzero].tolist())
            all_labels.extend(nonzero_labels)

    ece = expected_calibration_error(
        np.array(all_probs), np.array(all_labels)
    ) if all_probs else 1.0

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"P@{k}": float(np.mean(precisions)),
        "MAP": float(np.mean(aps)),
        "ECE": ece,
    }


def collect_training_data(
    bm25_model: bm25s.BM25,
    dataset: IRDataset,
    query_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (score, label) pairs from queries for parameter fitting."""
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
# Threshold-based evaluation
# ---------------------------------------------------------------------------

def threshold_f1(
    values: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 at a given threshold."""
    predicted = values >= threshold
    tp = np.sum(predicted & (labels == 1))
    fp = np.sum(predicted & (labels == 0))
    fn = np.sum(~predicted & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def find_best_threshold(
    values: np.ndarray,
    labels: np.ndarray,
    n_candidates: int = 200,
) -> tuple[float, float]:
    """Find the threshold that maximises F1."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        return lo, 0.0
    candidates = np.linspace(lo, hi, n_candidates)
    best_f1, best_t = 0.0, candidates[0]
    for t in candidates:
        _, _, f1 = threshold_f1(values, labels, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def evaluate_threshold_transfer(
    bm25_model: bm25s.BM25,
    scorer: BayesianBM25Scorer,
    dataset: IRDataset,
    train_qids: list[str],
    test_qids: list[str],
    label: str,
) -> tuple[float, float, float, float]:
    """Find best threshold on train queries, evaluate on test queries.

    Returns (bm25_test_f1, bayesian_test_f1, bm25_gap, bayesian_gap).
    """
    qid_to_tokens = {qid: qt for qid, qt in dataset.queries}

    def collect(qids, use_probs: bool):
        vals, labs = [], []
        for qid in qids:
            qt = qid_to_tokens[qid]
            qrel = dataset.qrels[qid]
            if use_probs:
                scores = scorer.get_probabilities(qt)
            else:
                scores = bm25_model.get_scores(qt)
            nz = scores > 0
            if not np.any(nz):
                continue
            nz_dids = [dataset.doc_ids[i] for i, n in enumerate(nz) if n]
            nz_labs = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
            vals.extend(scores[nz].tolist())
            labs.extend(nz_labs)
        return np.array(vals), np.array(labs)

    # BM25
    bm25_train_v, bm25_train_l = collect(train_qids, False)
    bm25_test_v, bm25_test_l = collect(test_qids, False)
    bm25_t, bm25_train_f1 = find_best_threshold(bm25_train_v, bm25_train_l)
    _, _, bm25_test_f1 = threshold_f1(bm25_test_v, bm25_test_l, bm25_t)

    # Bayesian
    bay_train_v, bay_train_l = collect(train_qids, True)
    bay_test_v, bay_test_l = collect(test_qids, True)
    bay_t, bay_train_f1 = find_best_threshold(bay_train_v, bay_train_l)
    _, _, bay_test_f1 = threshold_f1(bay_test_v, bay_test_l, bay_t)

    return bm25_train_f1, bm25_test_f1, bay_train_f1, bay_test_f1


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_single_dataset(dataset: IRDataset, k: int = 10) -> None:
    """Run the full benchmark on one dataset."""
    print(f"\n{'=' * 72}")
    print(f"Dataset: {dataset.name.upper()}")
    print(f"{'=' * 72}")

    # Split queries: first half for training, second for evaluation
    all_qids = [qid for qid, _ in dataset.queries]
    rng = np.random.default_rng(42)
    rng.shuffle(all_qids)
    mid = len(all_qids) // 2
    train_qids = all_qids[:mid]
    eval_qids = all_qids[mid:]

    eval_queries_filtered = [(qid, qt) for qid, qt in dataset.queries if qid in set(eval_qids)]
    train_queries_filtered = [(qid, qt) for qid, qt in dataset.queries if qid in set(train_qids)]

    eval_dataset = IRDataset(
        name=dataset.name,
        corpus_tokens=dataset.corpus_tokens,
        doc_ids=dataset.doc_ids,
        queries=eval_queries_filtered,
        qrels=dataset.qrels,
    )
    train_dataset = IRDataset(
        name=dataset.name,
        corpus_tokens=dataset.corpus_tokens,
        doc_ids=dataset.doc_ids,
        queries=train_queries_filtered,
        qrels=dataset.qrels,
    )

    print(f"  Train: {len(train_qids)} queries, Eval: {len(eval_qids)} queries")

    # --- 1. Raw BM25 ---
    print(f"\n{'--' * 36}")
    print("1. Raw BM25 (baseline)")
    bm25_model = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25_model.index(dataset.corpus_tokens, show_progress=False)
    raw_results = evaluate_bm25_raw(bm25_model, eval_dataset, k=k)
    print_results(raw_results)

    # --- 2. Bayesian BM25 (auto) ---
    print(f"\n{'--' * 36}")
    print("2. Bayesian BM25 (auto-estimated)")
    scorer_auto = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer_auto.index(dataset.corpus_tokens, show_progress=False)
    print(f"  alpha={scorer_auto._transform.alpha:.4f}, beta={scorer_auto._transform.beta:.4f}")
    auto_results = evaluate_bayesian(scorer_auto, eval_dataset, k=k)
    print_results(auto_results)

    # --- 3. Bayesian BM25 (batch fit) ---
    print(f"\n{'--' * 36}")
    print("3. Bayesian BM25 (batch fit)")
    train_scores, train_labels = collect_training_data(bm25_model, dataset, train_qids)
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    print(f"  Training: {len(train_scores)} samples (pos={n_pos}, neg={n_neg})")

    scorer_fit = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer_fit.index(dataset.corpus_tokens, show_progress=False)
    scorer_fit._transform.fit(
        train_scores, train_labels,
        learning_rate=0.05, max_iterations=3000,
    )
    print(f"  alpha={scorer_fit._transform.alpha:.4f}, beta={scorer_fit._transform.beta:.4f}")
    fit_results = evaluate_bayesian(scorer_fit, eval_dataset, k=k)
    print_results(fit_results)

    # --- 4. Bayesian BM25 (online) ---
    print(f"\n{'--' * 36}")
    print("4. Bayesian BM25 (online learning)")
    scorer_online = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer_online.index(dataset.corpus_tokens, show_progress=False)
    print(f"  Initial: alpha={scorer_online._transform.alpha:.4f}, "
          f"beta={scorer_online._transform.beta:.4f}")

    qid_to_tokens = {qid: qt for qid, qt in dataset.queries}
    snapshots = []
    n_updates = 0
    snapshot_at = {0, 10, 25, 50, 75, mid}

    snapshot_results = evaluate_bayesian(scorer_online, eval_dataset, k=k)
    snapshots.append((0, snapshot_results))

    for qid in train_qids:
        qtokens = qid_to_tokens[qid]
        qrel = dataset.qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        nz = scores > 0
        if not np.any(nz):
            continue
        nz_scores = scores[nz]
        nz_dids = [dataset.doc_ids[i] for i, n in enumerate(nz) if n]
        nz_labels = np.array([float(qrel.get(did, 0) >= 1) for did in nz_dids])

        scorer_online._transform.update(
            nz_scores, nz_labels, learning_rate=0.05, momentum=0.9,
        )
        n_updates += 1

        if n_updates in snapshot_at:
            snap = evaluate_bayesian(scorer_online, eval_dataset, k=k)
            snapshots.append((n_updates, snap))

    if n_updates not in snapshot_at:
        snap = evaluate_bayesian(scorer_online, eval_dataset, k=k)
        snapshots.append((n_updates, snap))

    print(f"  Final:   alpha={scorer_online._transform.alpha:.4f}, "
          f"beta={scorer_online._transform.beta:.4f}")
    online_results = evaluate_bayesian(scorer_online, eval_dataset, k=k)
    print_results(online_results)

    # --- Online progression ---
    print(f"\n{'--' * 36}")
    print("Online learning progression")
    print(f"  {'Queries':>7}  {'NDCG@10':>8}  {'MAP':>8}  {'ECE':>8}")
    for n, snap in snapshots:
        print(f"  {n:>7}  {snap[f'NDCG@{k}']:>8.4f}  {snap['MAP']:>8.4f}  {snap['ECE']:>8.4f}")

    # --- Summary table ---
    print(f"\n{'--' * 36}")
    print("Summary")
    print(f"  {'Method':<35}  {'NDCG@10':>8}  {'MAP':>8}  {'ECE':>8}")
    print(f"  {'─' * 35}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    rows = [
        ("Raw BM25", raw_results),
        ("Bayesian (auto)", auto_results),
        ("Bayesian (batch fit)", fit_results),
        ("Bayesian (online)", online_results),
    ]
    for name, r in rows:
        ece = f"{r['ECE']:>8.4f}" if "ECE" in r else "     n/a"
        print(f"  {name:<35}  {r[f'NDCG@{k}']:>8.4f}  {r['MAP']:>8.4f}  {ece}")

    print(f"\n  vs Raw BM25:")
    for name, r in rows[1:]:
        dn = r[f"NDCG@{k}"] - raw_results[f"NDCG@{k}"]
        dm = r["MAP"] - raw_results["MAP"]
        sn = "+" if dn >= 0 else ""
        sm = "+" if dm >= 0 else ""
        print(f"    {name:<33}  NDCG {sn}{dn:.4f}  MAP {sm}{dm:.4f}")

    # --- Threshold transfer ---
    print(f"\n{'--' * 36}")
    print("Threshold transfer (train threshold -> test queries)")
    bm25_tr_f1, bm25_te_f1, bay_tr_f1, bay_te_f1 = evaluate_threshold_transfer(
        bm25_model, scorer_online, dataset, train_qids, eval_qids, "online",
    )
    print(f"  {'Method':<20}  {'Train F1':>8}  {'Test F1':>8}  {'Gap':>8}")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    print(f"  {'Raw BM25':<20}  {bm25_tr_f1:>8.4f}  {bm25_te_f1:>8.4f}  "
          f"{bm25_tr_f1 - bm25_te_f1:>+8.4f}")
    print(f"  {'Bayesian (online)':<20}  {bay_tr_f1:>8.4f}  {bay_te_f1:>8.4f}  "
          f"{bay_tr_f1 - bay_te_f1:>+8.4f}")
    print(f"  (Smaller gap = threshold generalises better across queries)")


def print_results(results: dict[str, float]) -> None:
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")


def main() -> None:
    print("=" * 72)
    print("Bayesian BM25 Benchmark -- Standard IR Datasets")
    print("=" * 72)

    datasets_to_run = [
        ("nfcorpus", "test"),
        ("scifact", "test"),
    ]

    for ds_name, split in datasets_to_run:
        print(f"\nLoading {ds_name}...")
        dataset = load_beir_dataset(ds_name, split)
        run_single_dataset(dataset, k=10)


if __name__ == "__main__":
    main()
