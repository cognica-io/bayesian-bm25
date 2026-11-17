#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Exact text identity, alias judgments, and self-document exclusions."""

from __future__ import annotations

import hashlib
import json


def canonicalize_qrels(
    qrels: dict, query_alias_to_id: dict, corpus_alias_to_id: dict,
) -> dict[str, dict[str, int]]:
    """Merge any split's query and document aliases at the maximum grade."""
    result: dict[str, dict[str, int]] = {}
    for qid, judgments in qrels.items():
        canonical_query = query_alias_to_id.get(qid, qid)
        merged = result.setdefault(canonical_query, {})
        for doc_id, grade in judgments.items():
            canonical_doc = corpus_alias_to_id.get(doc_id, doc_id)
            merged[canonical_doc] = max(merged.get(canonical_doc, grade), grade)
    return result


def canonicalize_dataset(data: dict, identity: str = "text") -> dict:
    """Apply Section 6.1 identity before tokenization or calibration.

    Document identity uses the exact (title, body) pair; a shared abstract
    under different titles does not merge documents. Query identity uses
    its exact text. Alias judgments use their highest grade. The id mode
    retains the standard evaluator protocol for controlled comparisons.
    """
    if identity not in ("id", "text"):
        raise ValueError("identity must be 'id' or 'text'")
    result = dict(data)
    raw_docs = data["corpus_ids"]
    doc_groups: dict[object, str] = {}
    doc_alias_to_id: dict[str, str] = {}
    doc_aliases: dict[str, list[str]] = {}
    doc_positions: list[int] = []
    for position, doc_id in enumerate(raw_docs):
        key = (
            (data["corpus_titles"][position], data["corpus_bodies"][position])
            if identity == "text" else doc_id
        )
        if key not in doc_groups:
            doc_groups[key] = doc_id
            doc_positions.append(position)
            doc_aliases[doc_id] = []
        canonical = doc_groups[key]
        doc_alias_to_id[doc_id] = canonical
        doc_aliases[canonical].append(doc_id)
    for key in ("corpus_ids", "corpus_texts", "corpus_titles", "corpus_bodies"):
        result[key] = [data[key][position] for position in doc_positions]

    all_ids = data["all_query_ids"]
    all_texts = data["all_query_texts"]
    text_by_id = dict(zip(all_ids, all_texts, strict=True))
    # Prefer an evaluated id as representative of each text, so the test
    # population remains explicit even if an unevaluated alias appears first.
    query_groups: dict[object, str] = {}
    for qid in list(data["query_ids"]) + list(all_ids):
        key = text_by_id[qid] if identity == "text" else qid
        query_groups.setdefault(key, qid)
    query_alias_to_id: dict[str, str] = {}
    query_aliases: dict[str, list[str]] = {}
    for qid in all_ids:
        key = text_by_id[qid] if identity == "text" else qid
        canonical = query_groups[key]
        query_alias_to_id[qid] = canonical
        query_aliases.setdefault(canonical, []).append(qid)

    def queries(ids: list[str]) -> tuple[list[str], list[str]]:
        canonical_ids = list(dict.fromkeys(query_alias_to_id[qid] for qid in ids))
        return canonical_ids, [text_by_id[qid] for qid in canonical_ids]

    result["query_ids"], result["query_texts"] = queries(data["query_ids"])
    result["all_query_ids"], result["all_query_texts"] = queries(all_ids)
    qrels = canonicalize_qrels(data["qrels"], query_alias_to_id, doc_alias_to_id)
    result["qrels"] = qrels

    doc_index = {doc_id: position for position, doc_id in enumerate(result["corpus_ids"])}
    bodies: dict[str, list[str]] = {}
    for doc_id, body in zip(result["corpus_ids"], result["corpus_bodies"], strict=True):
        bodies.setdefault(body, []).append(doc_id)
    self_documents: dict[str, list[str]] = {}
    for qid in result["all_query_ids"]:
        excluded = {
            doc_alias_to_id[alias]
            for alias in query_aliases[qid] if alias in doc_alias_to_id
        }
        if identity == "text":
            excluded.update(bodies.get(text_by_id[qid], []))
        self_documents[qid] = sorted(excluded, key=doc_index.__getitem__)

    result.update(
        identity=identity, identity_version=1,
        corpus_aliases=doc_aliases, query_aliases=query_aliases,
        corpus_alias_to_id=doc_alias_to_id, query_alias_to_id=query_alias_to_id,
        excluded_doc_ids=self_documents,
        source_document_count=len(raw_docs), source_query_count=len(data["query_ids"]),
    )
    fingerprint = hashlib.sha256()
    for value in (
        identity, "1", result["corpus_ids"], result["corpus_titles"],
        result["corpus_bodies"], result["query_ids"], result["query_texts"], qrels,
    ):
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("ascii")
        fingerprint.update(len(encoded).to_bytes(8, "big"))
        fingerprint.update(encoded)
    result["dataset_fingerprint"] = fingerprint.hexdigest()
    return result
