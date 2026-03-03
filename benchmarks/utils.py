#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Shared benchmark utilities: dataset loading and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import ir_datasets
import numpy as np


@dataclass
class IRDataset:
    """Container for an IR evaluation dataset."""

    name: str
    corpus_tokens: list[list[str]]
    doc_ids: list[str]
    queries: list[tuple[str, list[str]]]  # (query_id, tokens)
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id -> relevance}


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
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    # Filter queries that have qrels
    queries = [(qid, qtokens) for qid, qtokens in queries if qid in qrels]

    print(
        f"    {len(corpus_tokens)} docs, {len(queries)} queries, "
        f"{sum(len(v) for v in qrels.values())} qrels"
    )

    return IRDataset(
        name=dataset_name,
        corpus_tokens=corpus_tokens,
        doc_ids=doc_id_list,
        queries=queries,
        qrels=qrels,
    )


def doc_id_to_idx(doc_ids: list[str]) -> dict[str, int]:
    """Build a mapping from document ID to corpus index."""
    return {did: i for i, did in enumerate(doc_ids)}


def get_relevance_vector(
    ranked_doc_ids: list[str],
    qrel: dict[str, int],
    binary_threshold: int = 1,
) -> np.ndarray:
    """Convert ranked doc IDs to a binary relevance vector."""
    return np.array(
        [float(qrel.get(did, 0) >= binary_threshold) for did in ranked_doc_ids]
    )


def get_graded_relevance_vector(
    ranked_doc_ids: list[str],
    qrel: dict[str, int],
) -> np.ndarray:
    """Convert ranked doc IDs to a graded relevance vector."""
    return np.array([float(qrel.get(did, 0)) for did in ranked_doc_ids])
