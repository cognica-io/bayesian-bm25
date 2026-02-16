#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Calibration verification: Are Bayesian BM25 probabilities meaningful?

Three verification axes:
  1. Calibration -- reliability diagram + ECE + Brier score
  2. Threshold transfer -- does the same threshold work across queries?
  3. Baseline comparison -- Bayesian vs min-max normalization vs Platt scaling

Requires: pip install ir_datasets
"""

from __future__ import annotations

import sys

import bm25s
import ir_datasets
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from bayesian_bm25.probability import BayesianProbabilityTransform, sigmoid
from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(name: str, split: str = "test"):
    """Load a BEIR dataset and return components."""
    ds = ir_datasets.load(f"beir/{name}/{split}")

    doc_ids = []
    corpus_tokens = []
    for doc in ds.docs_iter():
        doc_ids.append(doc.doc_id)
        text = doc.text
        if hasattr(doc, "title") and doc.title:
            text = doc.title + " " + text
        corpus_tokens.append(text.lower().split())

    queries = []
    for q in ds.queries_iter():
        queries.append((q.query_id, q.text.lower().split()))

    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    queries = [(qid, qt) for qid, qt in queries if qid in qrels]
    return corpus_tokens, doc_ids, queries, qrels


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------


def collect_scores_and_labels(
    bm25_model: bm25s.BM25,
    query_ids: list[str],
    qid_to_tokens: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (score, label) pairs from given queries."""
    all_scores, all_labels = [], []
    for qid in query_ids:
        qtokens = qid_to_tokens[qid]
        qrel = qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        nz = scores > 0
        if not np.any(nz):
            continue
        nz_dids = [doc_ids[i] for i, n in enumerate(nz) if n]
        nz_labels = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
        all_scores.extend(scores[nz].tolist())
        all_labels.extend(nz_labels)
    return np.array(all_scores), np.array(all_labels)


def collect_probabilities_and_labels(
    scorer: BayesianBM25Scorer,
    query_ids: list[str],
    qid_to_tokens: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (probability, label) pairs from given queries."""
    all_probs, all_labels = [], []
    for qid in query_ids:
        qtokens = qid_to_tokens[qid]
        qrel = qrels[qid]
        probs = scorer.get_probabilities(qtokens)
        nz = probs > 0
        if not np.any(nz):
            continue
        nz_dids = [doc_ids[i] for i, n in enumerate(nz) if n]
        nz_labels = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
        all_probs.extend(probs[nz].tolist())
        all_labels.extend(nz_labels)
    return np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Baseline: min-max normalization
# ---------------------------------------------------------------------------


def minmax_normalize(
    scores: np.ndarray,
    train_scores: np.ndarray,
) -> np.ndarray:
    """Normalize scores to [0, 1] using train set min/max."""
    lo = float(np.min(train_scores))
    hi = float(np.max(train_scores))
    if hi == lo:
        return np.full_like(scores, 0.5)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Baseline: Platt scaling
# ---------------------------------------------------------------------------


def platt_scaling_fit(
    scores: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 5000,
    tol: float = 1e-7,
) -> tuple[float, float]:
    """Fit Platt scaling parameters A, B: P = sigmoid(A*score + B).

    This is the standard method from Platt (1999) for converting
    SVM/classifier outputs into probabilities.
    """
    A = 0.0
    B = 0.0
    for _ in range(max_iter):
        z = A * scores + B
        p = sigmoid(z)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        error = p - labels
        grad_A = float(np.mean(error * scores))
        grad_B = float(np.mean(error))
        new_A = A - lr * grad_A
        new_B = B - lr * grad_B
        if abs(new_A - A) < tol and abs(new_B - B) < tol:
            A, B = new_A, new_B
            break
        A, B = new_A, new_B
    return A, B


# ---------------------------------------------------------------------------
# Threshold transfer
# ---------------------------------------------------------------------------


def find_best_threshold(
    values: np.ndarray,
    labels: np.ndarray,
    n_candidates: int = 200,
) -> tuple[float, float]:
    """Find threshold maximizing F1."""
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        return lo, 0.0
    candidates = np.linspace(lo, hi, n_candidates)
    best_f1, best_t = 0.0, candidates[0]
    for t in candidates:
        predicted = values >= t
        tp = np.sum(predicted & (labels == 1))
        fp = np.sum(predicted & (labels == 0))
        fn = np.sum(~predicted & (labels == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def threshold_f1(values, labels, t):
    predicted = values >= t
    tp = np.sum(predicted & (labels == 1))
    fp = np.sum(predicted & (labels == 0))
    fn = np.sum(~predicted & (labels == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_verification(dataset_name: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"Calibration Verification: {dataset_name.upper()}")
    print(f"{'=' * 72}")

    # Load
    print(f"\nLoading {dataset_name}...")
    corpus_tokens, doc_ids, queries, qrels = load_dataset(dataset_name)
    qid_to_tokens = {qid: qt for qid, qt in queries}
    print(f"  {len(corpus_tokens)} docs, {len(queries)} queries")

    # Split train/test
    rng = np.random.default_rng(42)
    all_qids = [qid for qid, _ in queries]
    rng.shuffle(all_qids)
    mid = len(all_qids) // 2
    train_qids = all_qids[:mid]
    test_qids = all_qids[mid:]
    print(f"  Train: {len(train_qids)} queries, Test: {len(test_qids)} queries")

    # Index
    bm25_model = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25_model.index(corpus_tokens, show_progress=False)

    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer.index(corpus_tokens, show_progress=False)

    # Collect data
    train_scores, train_labels = collect_scores_and_labels(
        bm25_model, train_qids, qid_to_tokens, doc_ids, qrels
    )
    test_scores, test_labels = collect_scores_and_labels(
        bm25_model, test_qids, qid_to_tokens, doc_ids, qrels
    )
    test_probs, test_prob_labels = collect_probabilities_and_labels(
        scorer, test_qids, qid_to_tokens, doc_ids, qrels
    )

    n_pos = int(test_labels.sum())
    n_neg = len(test_labels) - n_pos
    base_rate = n_pos / len(test_labels)
    print(f"\n  Test set: {len(test_labels)} scored docs "
          f"(pos={n_pos}, neg={n_neg}, base_rate={base_rate:.4f})")

    # === 1. Calibration ===
    print(f"\n{'--' * 36}")
    print("1. CALIBRATION (Reliability Diagram)")
    print(f"{'--' * 36}")

    # --- Bayesian BM25 ---
    bayesian_ece = expected_calibration_error(test_probs, test_prob_labels)
    bayesian_brier = brier_score(test_probs, test_prob_labels)
    bayesian_bins = reliability_diagram(test_probs, test_prob_labels)

    print(f"\n  Bayesian BM25 (auto-estimated):")
    print(f"    ECE   = {bayesian_ece:.4f}")
    print(f"    Brier = {bayesian_brier:.4f}")
    print(f"\n    {'Predicted':>10}  {'Actual':>8}  {'Count':>7}  {'Gap':>8}")
    for pred, actual, count in bayesian_bins:
        gap = pred - actual
        bar = "+" * int(abs(gap) * 100) if gap > 0 else "-" * int(abs(gap) * 100)
        print(f"    {pred:10.4f}  {actual:8.4f}  {count:7d}  {gap:+8.4f}  {bar}")

    # --- Bayesian BM25 (batch fit) ---
    scorer_fit = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer_fit.index(corpus_tokens, show_progress=False)
    scorer_fit._transform.fit(
        train_scores, train_labels, learning_rate=0.05, max_iterations=5000
    )
    fit_probs, fit_labels = collect_probabilities_and_labels(
        scorer_fit, test_qids, qid_to_tokens, doc_ids, qrels
    )
    fit_ece = expected_calibration_error(fit_probs, fit_labels)
    fit_brier = brier_score(fit_probs, fit_labels)
    fit_bins = reliability_diagram(fit_probs, fit_labels)

    print(f"\n  Bayesian BM25 (batch fit on train queries):")
    print(f"    ECE   = {fit_ece:.4f}")
    print(f"    Brier = {fit_brier:.4f}")
    print(f"\n    {'Predicted':>10}  {'Actual':>8}  {'Count':>7}  {'Gap':>8}")
    for pred, actual, count in fit_bins:
        gap = pred - actual
        bar = "+" * int(abs(gap) * 100) if gap > 0 else "-" * int(abs(gap) * 100)
        print(f"    {pred:10.4f}  {actual:8.4f}  {count:7d}  {gap:+8.4f}  {bar}")

    # --- Min-max normalization ---
    minmax_probs = minmax_normalize(test_scores, train_scores)
    minmax_ece = expected_calibration_error(minmax_probs, test_labels)
    minmax_brier = brier_score(minmax_probs, test_labels)
    minmax_bins = reliability_diagram(minmax_probs, test_labels)

    print(f"\n  Min-max normalization (baseline):")
    print(f"    ECE   = {minmax_ece:.4f}")
    print(f"    Brier = {minmax_brier:.4f}")
    print(f"\n    {'Predicted':>10}  {'Actual':>8}  {'Count':>7}  {'Gap':>8}")
    for pred, actual, count in minmax_bins:
        gap = pred - actual
        bar = "+" * int(abs(gap) * 100) if gap > 0 else "-" * int(abs(gap) * 100)
        print(f"    {pred:10.4f}  {actual:8.4f}  {count:7d}  {gap:+8.4f}  {bar}")

    # --- Platt scaling ---
    A, B = platt_scaling_fit(train_scores, train_labels)
    platt_probs = sigmoid(A * test_scores + B)
    platt_ece = expected_calibration_error(platt_probs, test_labels)
    platt_brier = brier_score(platt_probs, test_labels)
    platt_bins = reliability_diagram(platt_probs, test_labels)

    print(f"\n  Platt scaling (A={A:.4f}, B={B:.4f}):")
    print(f"    ECE   = {platt_ece:.4f}")
    print(f"    Brier = {platt_brier:.4f}")
    print(f"\n    {'Predicted':>10}  {'Actual':>8}  {'Count':>7}  {'Gap':>8}")
    for pred, actual, count in platt_bins:
        gap = pred - actual
        bar = "+" * int(abs(gap) * 100) if gap > 0 else "-" * int(abs(gap) * 100)
        print(f"    {pred:10.4f}  {actual:8.4f}  {count:7d}  {gap:+8.4f}  {bar}")

    # --- Constant baseline (always predict base rate) ---
    constant_probs = np.full_like(test_labels, base_rate)
    constant_brier = brier_score(constant_probs, test_labels)
    print(f"\n  Constant baseline (always predict base_rate={base_rate:.4f}):")
    print(f"    Brier = {constant_brier:.4f}  (reference: no discrimination)")

    # === 2. Threshold Transfer ===
    print(f"\n{'--' * 36}")
    print("2. THRESHOLD TRANSFER")
    print(f"{'--' * 36}")
    print("  (Train a threshold on train queries, evaluate on test queries)")
    print("  (Smaller gap = threshold generalizes better = more meaningful)")

    methods = []

    # Raw BM25
    bm25_train_scores, bm25_train_labels = train_scores, train_labels
    bm25_test_scores, bm25_test_labels = test_scores, test_labels
    bm25_t, bm25_train_f1 = find_best_threshold(bm25_train_scores, bm25_train_labels)
    _, _, bm25_test_f1 = threshold_f1(bm25_test_scores, bm25_test_labels, bm25_t)
    methods.append(("Raw BM25", bm25_t, bm25_train_f1, bm25_test_f1))

    # Min-max
    minmax_train = minmax_normalize(train_scores, train_scores)
    minmax_t, minmax_train_f1 = find_best_threshold(minmax_train, train_labels)
    _, _, minmax_test_f1 = threshold_f1(minmax_probs, test_labels, minmax_t)
    methods.append(("Min-max norm", minmax_t, minmax_train_f1, minmax_test_f1))

    # Platt
    platt_train = sigmoid(A * train_scores + B)
    platt_t, platt_train_f1 = find_best_threshold(platt_train, train_labels)
    _, _, platt_test_f1 = threshold_f1(platt_probs, test_labels, platt_t)
    methods.append(("Platt scaling", platt_t, platt_train_f1, platt_test_f1))

    # Bayesian (auto)
    bay_train_probs, bay_train_labels = collect_probabilities_and_labels(
        scorer, train_qids, qid_to_tokens, doc_ids, qrels
    )
    bay_t, bay_train_f1 = find_best_threshold(bay_train_probs, bay_train_labels)
    _, _, bay_test_f1 = threshold_f1(test_probs, test_prob_labels, bay_t)
    methods.append(("Bayesian (auto)", bay_t, bay_train_f1, bay_test_f1))

    # Bayesian (fit)
    fit_train_probs, fit_train_labels = collect_probabilities_and_labels(
        scorer_fit, train_qids, qid_to_tokens, doc_ids, qrels
    )
    fit_t, fit_train_f1 = find_best_threshold(fit_train_probs, fit_train_labels)
    _, _, fit_test_f1_result = threshold_f1(fit_probs, fit_labels, fit_t)
    methods.append(("Bayesian (fit)", fit_t, fit_train_f1, fit_test_f1_result))

    print(f"\n  {'Method':<20}  {'Threshold':>9}  {'Train F1':>8}  "
          f"{'Test F1':>8}  {'Gap':>8}")
    print(f"  {'---' * 20}")
    for name, t, tr_f1, te_f1 in methods:
        gap = tr_f1 - te_f1
        print(f"  {name:<20}  {t:9.4f}  {tr_f1:8.4f}  {te_f1:8.4f}  {gap:+8.4f}")

    # === 3. Summary ===
    print(f"\n{'--' * 36}")
    print("3. SUMMARY")
    print(f"{'--' * 36}")

    print(f"\n  {'Method':<20}  {'ECE':>8}  {'Brier':>8}  {'Thr.Gap':>8}")
    print(f"  {'---' * 20}")
    summaries = [
        ("Raw BM25", None, None, bm25_train_f1 - bm25_test_f1),
        ("Min-max norm", minmax_ece, minmax_brier, minmax_train_f1 - minmax_test_f1),
        ("Platt scaling", platt_ece, platt_brier, platt_train_f1 - platt_test_f1),
        ("Bayesian (auto)", bayesian_ece, bayesian_brier, bay_train_f1 - bay_test_f1),
        ("Bayesian (fit)", fit_ece, fit_brier, fit_train_f1 - fit_test_f1_result),
    ]
    for name, ece, brier, thr_gap in summaries:
        ece_s = f"{ece:8.4f}" if ece is not None else "     n/a"
        brier_s = f"{brier:8.4f}" if brier is not None else "     n/a"
        print(f"  {name:<20}  {ece_s}  {brier_s}  {thr_gap:+8.4f}")

    print(f"\n  Interpretation:")
    print(f"    ECE: lower = better calibrated (predicted prob matches actual rate)")
    print(f"    Brier: lower = better (calibration + discrimination combined)")
    print(f"    Thr.Gap: lower = threshold generalizes across queries")
    print(f"    Brier reference (no discrimination): {constant_brier:.4f}")


def main():
    print("=" * 72)
    print("Bayesian BM25 -- Calibration Verification")
    print("=" * 72)
    print("Are the probabilities meaningful?")

    for ds_name in ["nfcorpus", "scifact"]:
        run_verification(ds_name)


if __name__ == "__main__":
    main()
