#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.multi_field module."""

import numpy as np
import pytest

from bayesian_bm25.multi_field import MultiFieldScorer
from bayesian_bm25.scorer import BayesianBM25Scorer


@pytest.fixture
def two_field_docs():
    """A small corpus with title and body fields."""
    return [
        {
            "title": ["cat", "sat", "mat"],
            "body": ["the", "cat", "sat", "on", "the", "mat"],
        },
        {
            "title": ["dog", "chased", "cat"],
            "body": ["the", "dog", "chased", "the", "cat", "around"],
        },
        {
            "title": ["quick", "brown", "fox"],
            "body": ["a", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        },
        {
            "title": ["hello", "world"],
            "body": ["hello", "world", "program"],
        },
        {
            "title": ["machine", "learning"],
            "body": ["machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence"],
        },
    ]


@pytest.fixture
def multi_scorer(two_field_docs):
    """A MultiFieldScorer indexed on the two-field corpus."""
    scorer = MultiFieldScorer(
        fields=["title", "body"],
        k1=1.2,
        b=0.75,
        method="lucene",
    )
    scorer.index(two_field_docs, show_progress=False)
    return scorer


class TestIndexAndRetrieve:
    def test_index_and_retrieve(self, multi_scorer):
        """Basic 2-field index/retrieve workflow returns results."""
        doc_ids, probs = multi_scorer.retrieve(["cat"], k=3)
        assert len(doc_ids) == 3
        assert len(probs) == 3
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        # Cat documents (0, 1) should be in top results
        assert 0 in doc_ids or 1 in doc_ids

    def test_get_probabilities_shape(self, multi_scorer):
        """get_probabilities returns (num_docs,) array."""
        probs = multi_scorer.get_probabilities(["cat"])
        assert probs.shape == (multi_scorer.num_docs,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_custom_field_weights(self, two_field_docs):
        """Non-uniform weights change the ranking."""
        scorer_title = MultiFieldScorer(
            fields=["title", "body"],
            field_weights={"title": 0.9, "body": 0.1},
            method="lucene",
        )
        scorer_body = MultiFieldScorer(
            fields=["title", "body"],
            field_weights={"title": 0.1, "body": 0.9},
            method="lucene",
        )
        scorer_title.index(two_field_docs, show_progress=False)
        scorer_body.index(two_field_docs, show_progress=False)

        probs_title = scorer_title.get_probabilities(["cat"])
        probs_body = scorer_body.get_probabilities(["cat"])

        # Different weights should produce different probability distributions
        assert not np.allclose(probs_title, probs_body)

    def test_missing_field_raises(self):
        """Document missing a field raises ValueError."""
        scorer = MultiFieldScorer(fields=["title", "body"], method="lucene")
        docs = [
            {"title": ["hello"]},  # missing "body"
        ]
        with pytest.raises(ValueError, match="missing field"):
            scorer.index(docs, show_progress=False)

    def test_single_field_equivalent(self, two_field_docs):
        """1-field MultiFieldScorer matches BayesianBM25Scorer."""
        # Single-field multi scorer
        multi = MultiFieldScorer(
            fields=["body"],
            k1=1.2,
            b=0.75,
            method="lucene",
        )
        multi.index(two_field_docs, show_progress=False)

        # Direct single scorer
        single = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        body_corpus = [doc["body"] for doc in two_field_docs]
        single.index(body_corpus, show_progress=False)

        query = ["cat"]
        probs_multi = multi.get_probabilities(query)
        probs_single = single.get_probabilities(query)

        # With a single field and equal weights (1.0), the log-odds conjunction
        # with alpha resolves to a single-signal identity (Proposition 4.3.2)
        # up to floating point, results should be very close
        np.testing.assert_allclose(probs_multi, probs_single, atol=1e-6)


class TestProperties:
    def test_num_docs(self, multi_scorer, two_field_docs):
        assert multi_scorer.num_docs == len(two_field_docs)

    def test_fields(self, multi_scorer):
        assert multi_scorer.fields == ["title", "body"]

    def test_field_weights_default(self, multi_scorer):
        weights = multi_scorer.field_weights
        assert weights == pytest.approx({"title": 0.5, "body": 0.5})

    def test_field_weights_custom(self, two_field_docs):
        scorer = MultiFieldScorer(
            fields=["title", "body"],
            field_weights={"title": 0.7, "body": 0.3},
            method="lucene",
        )
        scorer.index(two_field_docs, show_progress=False)
        assert scorer.field_weights == pytest.approx(
            {"title": 0.7, "body": 0.3}
        )


class TestAddDocuments:
    def test_add_documents(self, multi_scorer, two_field_docs):
        """Incremental add_documents works."""
        original_count = multi_scorer.num_docs
        multi_scorer.add_documents(
            [{"title": ["new", "cat"], "body": ["brand", "new", "cat", "doc"]}],
            show_progress=False,
        )
        assert multi_scorer.num_docs == original_count + 1

        # New document should be findable
        probs = multi_scorer.get_probabilities(["cat"])
        new_doc_id = original_count
        assert probs[new_doc_id] > 0

    def test_add_documents_before_index_raises(self):
        """RuntimeError if add_documents called before index()."""
        scorer = MultiFieldScorer(fields=["title", "body"])
        with pytest.raises(RuntimeError, match="index"):
            scorer.add_documents(
                [{"title": ["a"], "body": ["b"]}]
            )

    def test_add_documents_missing_field_raises(self, multi_scorer):
        """add_documents validates field presence."""
        with pytest.raises(ValueError, match="missing field"):
            multi_scorer.add_documents(
                [{"title": ["only", "title"]}],
                show_progress=False,
            )


class TestValidation:
    def test_empty_fields_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            MultiFieldScorer(fields=[])

    def test_duplicate_fields_raises(self):
        with pytest.raises(ValueError, match="duplicates"):
            MultiFieldScorer(fields=["title", "title"])

    def test_weights_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing key"):
            MultiFieldScorer(
                fields=["title", "body"],
                field_weights={"title": 1.0},
            )

    def test_weights_bad_sum_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            MultiFieldScorer(
                fields=["title", "body"],
                field_weights={"title": 0.5, "body": 0.6},
            )
