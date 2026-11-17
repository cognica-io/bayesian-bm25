#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Canonical texts, alias judgments, query copies, and cache identity."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import hybrid_beir
from benchmarks.dataset_identity import canonicalize_dataset


def _dataset():
    ids = ["query_alias", "answer1", "answer2", "notice1", "notice2", "empty1", "empty2"]
    titles = ["", "Answer", "Answer", "First title", "Second title", "", ""]
    bodies = ["query text", "answer", "answer", "notice", "notice", "", ""]
    return {
        "corpus_ids": ids, "corpus_titles": titles, "corpus_bodies": bodies,
        "corpus_texts": [
            title + " " + body if title else body
            for title, body in zip(titles, bodies, strict=True)
        ],
        "query_ids": ["q", "query_alias", "notice_query"],
        "query_texts": ["query text", "query text", "notice"],
        "all_query_ids": ["q", "query_alias", "notice_query"],
        "all_query_texts": ["query text", "query text", "notice"],
        "qrels": {
            "q": {"answer1": 1}, "query_alias": {"answer2": 2}, "notice_query": {"notice1": 1},
        },
    }


class TestDatasetIdentity:
    def test_training_qrels_merge_aliases_and_exclude_all_test_texts(self):
        raw = _dataset()
        raw["all_query_ids"] += ["train", "train_alias"]
        raw["all_query_texts"] += ["independent query", "independent query"]
        data = canonicalize_dataset(raw)
        judgments = hybrid_beir.training_qrels_for_dataset(data, {
            "query_alias": {"answer2": 1},
            "train": {"answer1": 1}, "train_alias": {"answer2": 2},
        })
        assert judgments == {"train": {"answer1": 2}}

    def test_text_identity_merges_alias_judgments_at_the_highest_grade(self):
        data = canonicalize_dataset(_dataset())
        assert data["query_ids"] == ["q", "notice_query"]
        assert data["qrels"]["q"] == {"answer1": 2}
        assert data["corpus_aliases"]["answer1"] == ["answer1", "answer2"]
        assert len(data["corpus_ids"]) == 5

    def test_a_body_without_its_title_does_not_identify_a_document(self):
        data = canonicalize_dataset(_dataset())
        assert "notice1" in data["corpus_ids"] and "notice2" in data["corpus_ids"]
        assert data["excluded_doc_ids"]["notice_query"] == ["notice1", "notice2"]

    def test_a_query_alias_is_excluded_even_with_a_different_representative_id(self):
        data = canonicalize_dataset(_dataset())
        assert data["excluded_doc_ids"]["q"] == ["query_alias"]
        assert data["query_aliases"]["q"] == ["q", "query_alias"]

    def test_id_identity_preserves_the_standard_evaluator_population(self):
        data = canonicalize_dataset(_dataset(), identity="id")
        assert len(data["query_ids"]) == 3
        assert len(data["corpus_ids"]) == 7
        assert data["excluded_doc_ids"]["q"] == []
        assert data["excluded_doc_ids"]["query_alias"] == ["query_alias"]

    def test_fingerprint_changes_when_identity_or_labels_change(self):
        source = _dataset()
        original = canonicalize_dataset(source)["dataset_fingerprint"]
        assert original != canonicalize_dataset(source, "id")["dataset_fingerprint"]
        source["qrels"]["q"]["answer1"] = 3
        assert original != canonicalize_dataset(source)["dataset_fingerprint"]

    def test_cache_rejects_same_size_different_texts_and_identity(self, tmp_path, monkeypatch):
        calls = []

        def encode(texts, **kwargs):
            calls.append(list(texts))
            return np.array([[len(text), sum(map(ord, text))] for text in texts], dtype=np.float32)

        monkeypatch.setattr(
            hybrid_beir, "_dense_encoder", lambda model: SimpleNamespace(encode=encode),
        )
        path = str(tmp_path / "embeddings.npz")
        original = hybrid_beir.encode_dense(
            ["a", "b"], "encoder", path, cache_context={"identity": "id"},
        )
        cached = hybrid_beir.encode_dense(
            ["a", "b"], "encoder", path, cache_context={"identity": "id"},
        )
        np.testing.assert_array_equal(cached, original)
        assert len(calls) == 1
        changed = hybrid_beir.encode_dense(
            ["a", "c"], "encoder", path, cache_context={"identity": "id"},
        )
        assert not np.array_equal(changed, original)
        hybrid_beir.encode_dense(["a", "c"], "encoder", path, cache_context={"identity": "text"})
        assert len(calls) == 3

    def test_legacy_shape_only_cache_is_not_trusted(self, tmp_path, monkeypatch):
        path = str(tmp_path / "embeddings.npz")
        np.savez(path, embeddings=np.zeros((2, 2)))
        monkeypatch.setattr(hybrid_beir, "_dense_encoder", lambda model: SimpleNamespace(
            encode=lambda texts, **kwargs: np.ones((2, 2)),
        ))
        actual = hybrid_beir.encode_dense(["a", "b"], "encoder", path)
        assert np.all(actual == 1.0)

    def test_invalid_identity_is_rejected(self):
        with pytest.raises(ValueError, match="identity"):
            canonicalize_dataset(_dataset(), "approximate")
