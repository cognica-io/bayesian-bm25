#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Per-query view of a prepared dataset: population, labels, candidates.

All arrays of :class:`QuerySignals` are indexed by *population*
position.  The population is the corpus minus the document whose id
equals the query's (when identical ids are excluded), so the excluded
document is absent from every statistic, from calibration and from
scoring alike -- not merely filtered from the final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.prepared_dataset import PreparedDataset


@dataclass
class QuerySignals:
    """Signals of one query over its reference population.

    Attributes
    ----------
    index : int
        Position of the query in the dataset.
    qid : str
        Query id.
    doc_index : ndarray of shape (P,)
        Corpus index of each population document.
    bm25 : ndarray of shape (P,), float64
        BM25 scores (0 for documents that match no query term).
    dense : ndarray of shape (P,), float64
        Cosine similarities.
    labels : ndarray of shape (P,), float64
        Binary relevance (1 for a judged-relevant document).
    candidates : ndarray of shape (M,)
        Population positions of the union of each signal's top results.
    bm25_rank, dense_rank : ndarray of shape (M,)
        1-based rank of each candidate in the lexical and dense result
        lists, 0 when the candidate is not in that list.
    """

    index: int
    qid: str
    doc_index: np.ndarray
    bm25: np.ndarray
    dense: np.ndarray
    labels: np.ndarray
    candidates: np.ndarray
    bm25_rank: np.ndarray
    dense_rank: np.ndarray

    @property
    def matched(self) -> np.ndarray:
        """Mask of population documents that match at least one query term."""
        return self.bm25 > 0.0

    @property
    def n_relevant(self) -> int:
        """Number of relevant documents in the population."""
        return int(np.sum(self.labels))


def _top_positions(scores: np.ndarray, k: int, eligible: np.ndarray | None) -> np.ndarray:
    """Positions of the ``k`` highest scores, best first.

    Ties are broken by position, so the order is deterministic.  When
    ``eligible`` is given, only those positions can be returned.
    """
    pool = np.arange(len(scores)) if eligible is None else np.flatnonzero(eligible)
    if len(pool) == 0:
        return pool
    pool_scores = scores[pool]
    if len(pool) > k:
        threshold = np.partition(pool_scores, len(pool) - k)[len(pool) - k]
        keep = pool_scores >= threshold
        pool = pool[keep]
        pool_scores = pool_scores[keep]
    order = np.argsort(-pool_scores, kind="stable")[:k]
    return pool[order]


def query_signals(
    prepared: PreparedDataset,
    q_idx: int,
    retrieve_k: int = 1000,
) -> QuerySignals:
    """Build the per-query view for one query.

    The lexical result list holds only documents with a non-zero BM25
    score: a document that matches no query term is not retrieved by the
    lexical signal, however short the list of matches is.

    Parameters
    ----------
    prepared : PreparedDataset
        The dataset.
    q_idx : int
        Position of the query.
    retrieve_k : int
        Depth of each signal's result list.

    Returns
    -------
    QuerySignals
    """
    n_docs = prepared.n_docs
    excluded = prepared.excluded_for_query(q_idx)

    if excluded.size > 0:
        doc_index = np.delete(np.arange(n_docs, dtype=np.int64), excluded)
        bm25 = np.delete(prepared.bm25_scores[q_idx], excluded).astype(np.float64)
        dense = np.delete(prepared.dense_scores[q_idx], excluded).astype(np.float64)
    else:
        doc_index = np.arange(n_docs, dtype=np.int64)
        bm25 = prepared.bm25_scores[q_idx].astype(np.float64)
        dense = prepared.dense_scores[q_idx].astype(np.float64)

    labels_corpus = np.zeros(n_docs, dtype=np.float64)
    labels_corpus[prepared.relevant[q_idx]] = 1.0
    labels = labels_corpus[doc_index]

    bm25_top = _top_positions(bm25, retrieve_k, bm25 > 0.0)
    dense_top = _top_positions(dense, retrieve_k, None)
    candidates = np.union1d(bm25_top, dense_top)

    rank = np.zeros(len(bm25), dtype=np.int64)
    rank[bm25_top] = np.arange(1, len(bm25_top) + 1)
    bm25_rank = rank[candidates].copy()
    rank[bm25_top] = 0
    rank[dense_top] = np.arange(1, len(dense_top) + 1)
    dense_rank = rank[candidates].copy()

    return QuerySignals(
        index=q_idx,
        qid=prepared.query_ids[q_idx],
        doc_index=doc_index,
        bm25=bm25,
        dense=dense,
        labels=labels,
        candidates=candidates,
        bm25_rank=bm25_rank,
        dense_rank=dense_rank,
    )
