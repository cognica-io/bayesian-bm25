#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Ranking evaluation that is exact in double precision.

``pytrec_eval`` (like ``trec_eval``) stores run scores in single
precision and breaks ties by document id in descending order.  Two
log-odds values near 30 that differ in the seventh significant digit
are therefore the same score to the evaluator, and their order is
decided by the document ids.  That is the saturation problem of
probability-valued scores in a third guise.

The helpers here rank in double precision first and hand the evaluator
a score that encodes the rank, so that the evaluated order is the order
of the scores.  Exact double-precision ties are broken by document id
in descending order, which is what the evaluator itself would do.
"""

from __future__ import annotations

import numpy as np

METRIC_KEYS = ("ndcg_cut", "map_cut", "recall")


def ranked_run(
    doc_ids: list[str] | np.ndarray,
    scores: np.ndarray,
    depth: int = 100,
) -> dict[str, float]:
    """Top-``depth`` documents as a run whose scores encode the rank.

    Parameters
    ----------
    doc_ids : sequence of str
        Document ids of the candidates.
    scores : array
        Scores of the candidates; higher is better.  Compared in double
        precision.
    depth : int
        Number of documents to keep.  Metrics with a cutoff at or below
        ``depth`` are unaffected by the truncation.

    Returns
    -------
    dict
        ``{doc_id: depth - rank}`` for the top ``depth`` documents, so
        that consecutive ranks differ by exactly one, which single
        precision represents exactly.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return {}
    k = min(depth, n)

    if k < n:
        # Keep every document tied with the k-th score so that the tie
        # is broken below by the full rule, not by argpartition.
        threshold = np.partition(scores, n - k)[n - k]
        keep = np.flatnonzero(scores >= threshold)
    else:
        keep = np.arange(n)

    kept_ids = [str(doc_ids[i]) for i in keep]
    kept_scores = scores[keep]
    # Sort by score descending, then document id descending.
    order = sorted(
        range(len(keep)),
        key=lambda i: (-kept_scores[i], _descending_key(kept_ids[i])),
    )[:k]
    return {kept_ids[i]: float(k - rank) for rank, i in enumerate(order)}


def raw_run(
    doc_ids: list[str] | np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """All candidates with their raw scores, as the first version did.

    The evaluator then compares the scores in single precision.  Kept
    only to reproduce the first-version protocol.
    """
    return {str(doc_ids[i]): float(scores[i]) for i in range(len(scores))}


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Per-query NDCG@k, MAP@k and Recall@k via pytrec_eval.

    Queries that appear in ``qrels`` but not in ``run`` are scored as
    empty result lists, so every method is averaged over the same
    queries.

    Returns
    -------
    dict
        ``{metric_name: {query_id: value}}`` with metric names
        ``"NDCG@k"``, ``"MAP@k"`` and ``"Recall@k"``.
    """
    import pytrec_eval

    measures = {f"ndcg_cut_{k}", f"map_cut_{k}", f"recall_{k}"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    complete_run = {qid: run.get(qid, {}) for qid in qrels}
    # pytrec_eval drops queries with an empty result list; score them 0.
    results = evaluator.evaluate(
        {qid: docs for qid, docs in complete_run.items() if docs}
    )

    names = {
        f"ndcg_cut_{k}": f"NDCG@{k}",
        f"map_cut_{k}": f"MAP@{k}",
        f"recall_{k}": f"Recall@{k}",
    }
    per_query: dict[str, dict[str, float]] = {name: {} for name in names.values()}
    for qid in qrels:
        row = results.get(qid)
        for key, name in names.items():
            per_query[name][qid] = float(row[key]) if row is not None else 0.0
    return per_query


def mean_metrics(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean of each per-query metric."""
    return {
        name: float(np.mean(list(values.values()))) if values else 0.0
        for name, values in per_query.items()
    }


def _descending_key(doc_id: str) -> tuple[int, ...]:
    """Sort key that orders strings in descending lexicographic order.

    Negating the code points reverses the order of the characters; the
    trailing sentinel makes a string sort after its own prefixes, which
    is their descending order.
    """
    return tuple(-ord(ch) for ch in doc_id) + (1,)
