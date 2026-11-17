#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""A BEIR dataset with both retrieval signals scored for every document.

The re-evaluation needs each query's score against the whole corpus,
not just a retrieved list: the count constraint of per-query calibration
is a statement about the reference population.  :class:`PreparedDataset`
therefore holds the full query-by-document matrices of BM25 scores and
cosine similarities, together with the corpus-level parameters that the
first version of the benchmark estimated once per corpus.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.hybrid_beir import encode_dense, load_beir_dataset, tokenize_texts


@dataclass
class PreparedDataset:
    """Signals and labels of one BEIR dataset.

    Attributes
    ----------
    name : str
        Dataset name.
    corpus_ids : list of str
        Document ids, in corpus order.
    query_ids : list of str
        Ids of the test queries, in evaluation order.
    query_tokens : list of list of str
        Tokenized queries.
    bm25_scores : ndarray of shape (n_queries, n_docs), float32
        BM25 score of every document for every query (0 for a document
        that matches no query term).
    dense_scores : ndarray of shape (n_queries, n_docs), float32
        Cosine similarity of every document for every query.
    excluded : ndarray of shape (n_queries,), int64
        Corpus index of the document whose id equals the query's id, or
        -1 when there is none or identical ids are not excluded.
    relevant : list of ndarray
        Corpus indices of the documents judged relevant, per query.
    qrels : dict
        Relevance judgments ``{query_id: {doc_id: grade}}``.
    scorer : BayesianBM25Scorer
        Corpus-level scorer (slope and midpoint from pseudo-queries).
    corpus_alpha, corpus_beta : float
        Corpus-level lexical slope and midpoint.
    corpus_base_rate : float
        Corpus-level base rate from the percentile rule.
    dense_kappa, dense_beta : float
        Corpus-level scale and location of the cosine distance, from a
        sample of query-document pairs.
    """

    name: str
    corpus_ids: list[str]
    query_ids: list[str]
    query_tokens: list[list[str]]
    bm25_scores: np.ndarray
    dense_scores: np.ndarray
    excluded: np.ndarray
    relevant: list[np.ndarray]
    qrels: dict[str, dict[str, int]]
    scorer: BayesianBM25Scorer
    corpus_alpha: float
    corpus_beta: float
    corpus_base_rate: float
    dense_kappa: float
    dense_beta: float
    identity: str = "id"
    dataset_fingerprint: str = ""
    excluded_documents: list[np.ndarray] | None = None
    preparation_seconds: float = 0.0
    model_name: str = "unspecified"

    def excluded_for_query(self, query_index: int) -> np.ndarray:
        """Every self-document excluded from this query's population."""
        if self.excluded_documents is not None:
            return self.excluded_documents[query_index]
        position = int(self.excluded[query_index])
        if position >= 0:
            return np.array([position], dtype=np.int64)
        return np.array([], dtype=np.int64)

    @property
    def n_docs(self) -> int:
        """Number of documents in the corpus."""
        return len(self.corpus_ids)

    @property
    def n_queries(self) -> int:
        """Number of test queries."""
        return len(self.query_ids)


def prepare_dataset(
    dataset_dir: str,
    name: str,
    model_name: str,
    cache_dir: str | None,
    *,
    ignore_identical_ids: bool = True,
    identity: str = "text",
) -> PreparedDataset:
    """Load a BEIR dataset and score both signals against the whole corpus.

    Parameters
    ----------
    dataset_dir : str
        Directory with ``corpus.jsonl``, ``queries.jsonl`` and
        ``qrels/test.tsv``.
    name : str
        Dataset name (used for logging and the embedding cache).
    model_name : str
        Sentence-transformers model for the dense signal.
    cache_dir : str or None
        Root of the embedding cache; None disables caching.
    ignore_identical_ids : bool
        Exclude, for each query, the document whose id equals the
        query's id.  This is the default of the BEIR evaluator and what
        MTEB applies to ArguAna, whose queries are drawn from the
        corpus: the query's own text is otherwise a candidate, ranks
        first for every signal, and is not relevant.

    Returns
    -------
    PreparedDataset
    """
    t0 = time.time()
    data = load_beir_dataset(dataset_dir, identity=identity)
    corpus_ids = data["corpus_ids"]
    query_ids = data["query_ids"]
    qrels = data["qrels"]
    n_docs = len(corpus_ids)
    n_queries = len(query_ids)

    corpus_tokens = tokenize_texts(data["corpus_texts"])
    query_tokens = tokenize_texts(data["query_texts"])

    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer.index(corpus_tokens, show_progress=False)
    pseudo_query_scores = scorer._sample_pseudo_query_scores(corpus_tokens)
    base_rate = scorer._estimate_base_rate(pseudo_query_scores, n_docs)

    bm25_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
    for q_idx, tokens in enumerate(query_tokens):
        if tokens:
            bm25_scores[q_idx] = scorer._bm25.get_scores(tokens)

    cache_path: str | None = None
    if cache_dir is not None:
        safe_model = model_name.replace("/", "_")
        cache_path = os.path.join(cache_dir, name, f"{safe_model}-{identity}.npz")
    cache_context = {"identity": identity, "identity_version": 1}
    corpus_emb = encode_dense(
        data["corpus_texts"], model_name, cache_path, cache_key="corpus",
        cache_context=cache_context,
    )
    query_emb = encode_dense(
        data["query_texts"], model_name, cache_path, cache_key="queries",
        cache_context=cache_context,
    )
    # One matrix-vector product per query, the same expression the hybrid
    # benchmark evaluates, so that both produce bit-identical similarities
    # (a single matrix-matrix product can differ in the last place, which
    # is enough to reorder near-duplicate documents).
    dense_scores = np.empty((n_queries, n_docs), dtype=np.float32)
    for q_idx in range(n_queries):
        dense_scores[q_idx] = query_emb[q_idx] @ corpus_emb.T

    # Corpus-level location and scale of the cosine distance, estimated
    # exactly as the first version did (same sample, same seed).
    rng = np.random.default_rng(42)
    bg_docs = rng.choice(n_docs, size=min(1000, n_docs), replace=False)
    bg_queries = rng.choice(n_queries, size=min(50, n_queries), replace=False)
    doc_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    background_parts = []
    for q_idx in bg_queries:
        excluded_ids = data["excluded_doc_ids"][query_ids[q_idx]] if ignore_identical_ids else []
        excluded_positions = [doc_index[doc_id] for doc_id in excluded_ids]
        sample = bg_docs[~np.isin(bg_docs, excluded_positions)]
        background_parts.append(1.0 - np.asarray(query_emb[q_idx] @ corpus_emb[sample].T))
    bg_distances = np.concatenate(background_parts).astype(np.float64)
    dense_beta = float(np.median(bg_distances))
    bg_std = float(np.std(bg_distances))
    dense_kappa = 1.0 / bg_std if bg_std > 1e-12 else 1.0

    excluded = np.full(n_queries, -1, dtype=np.int64)
    excluded_documents = [np.array([], dtype=np.int64) for _ in query_ids]
    if ignore_identical_ids:
        for q_idx, qid in enumerate(query_ids):
            positions = [doc_index[doc_id] for doc_id in data["excluded_doc_ids"][qid]]
            excluded_documents[q_idx] = np.array(positions, dtype=np.int64)
            excluded[q_idx] = positions[0] if positions else -1

    relevant: list[np.ndarray] = []
    for qid in query_ids:
        indices = [
            doc_index[doc_id]
            for doc_id, grade in qrels[qid].items()
            if grade > 0 and doc_id in doc_index
        ]
        relevant.append(np.asarray(sorted(indices), dtype=np.int64))

    n_excluded = int(np.sum(excluded >= 0))
    print(
        f"  {name}: {n_docs} docs, {n_queries} queries, "
        f"alpha={scorer._transform.alpha:.4f}, beta={scorer._transform.beta:.4f}, "
        f"base_rate={base_rate:.6f}, identical ids excluded for "
        f"{n_excluded} queries ({time.time() - t0:.1f}s)",
        flush=True,
    )

    return PreparedDataset(
        name=name,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        query_tokens=query_tokens,
        bm25_scores=bm25_scores,
        dense_scores=dense_scores,
        excluded=excluded,
        relevant=relevant,
        qrels=qrels,
        scorer=scorer,
        corpus_alpha=float(scorer._transform.alpha),
        corpus_beta=float(scorer._transform.beta),
        corpus_base_rate=float(base_rate),
        dense_kappa=dense_kappa,
        dense_beta=dense_beta,
        identity=identity, dataset_fingerprint=data["dataset_fingerprint"],
        excluded_documents=excluded_documents, preparation_seconds=time.time() - t0,
        model_name=model_name,
    )
