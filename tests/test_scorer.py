#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.scorer module."""

import numpy as np
import pytest

from bayesian_bm25.scorer import BayesianBM25Scorer


@pytest.fixture
def small_corpus():
    """A small corpus for end-to-end testing."""
    return [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["a", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["hello", "world"],
        ["machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence"],
        ["the", "cat", "and", "the", "dog", "are", "friends"],
    ]


@pytest.fixture
def scorer(small_corpus):
    """A scorer indexed on the small corpus."""
    s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    s.index(small_corpus, show_progress=False)
    return s


class TestIndexing:
    def test_doc_lengths(self, scorer, small_corpus):
        assert len(scorer.doc_lengths) == len(small_corpus)
        expected_lengths = [len(doc) for doc in small_corpus]
        np.testing.assert_array_equal(scorer.doc_lengths, expected_lengths)

    def test_avgdl(self, scorer, small_corpus):
        expected = np.mean([len(doc) for doc in small_corpus])
        assert scorer.avgdl == pytest.approx(expected)

    def test_num_docs(self, scorer, small_corpus):
        assert scorer.num_docs == len(small_corpus)


class TestRetrieve:
    def test_returns_correct_shape(self, scorer):
        doc_ids, probs = scorer.retrieve([["cat"]], k=3)
        assert doc_ids.shape == (1, 3)
        assert probs.shape == (1, 3)

    def test_probabilities_in_bounds(self, scorer):
        doc_ids, probs = scorer.retrieve([["cat"]], k=6)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_ranking_preserved(self, scorer):
        """Default auto-estimated alpha/beta should preserve BM25 ranking.

        With alpha = 1/std(scores) and beta = median(scores), the
        likelihood signal dominates over per-document prior differences,
        so BM25 ranking order is preserved in the probabilities.
        """
        doc_ids, probs = scorer.retrieve([["cat"]], k=6)
        nonzero_mask = probs[0] > 0
        nonzero_probs = probs[0][nonzero_mask]
        if len(nonzero_probs) > 1:
            # bm25s returns descending by score; probabilities should follow
            assert np.all(np.diff(nonzero_probs) <= 0), (
                f"Ranking not preserved: {nonzero_probs}"
            )

    def test_monotonicity_fixed_prior(self):
        """Higher BM25 scores give higher probabilities when priors are equal."""
        from bayesian_bm25.probability import BayesianProbabilityTransform

        t = BayesianProbabilityTransform(alpha=1.0, beta=0.5)
        scores = np.array([0.2, 0.5, 1.0, 2.0, 3.0])
        fixed_tf = 5.0
        fixed_ratio = 0.5
        probs = t.score_to_probability(scores, fixed_tf, fixed_ratio)
        assert np.all(np.diff(probs) > 0)

    def test_multiple_queries(self, scorer):
        queries = [["cat"], ["dog"], ["machine", "learning"]]
        doc_ids, probs = scorer.retrieve(queries, k=3)
        assert doc_ids.shape == (3, 3)
        assert probs.shape == (3, 3)

    def test_relevant_docs_ranked_high(self, scorer):
        """Documents containing query terms should appear in results."""
        doc_ids, probs = scorer.retrieve([["cat"]], k=3)
        # Documents 0, 1, 5 contain "cat"
        top_docs = set(doc_ids[0].tolist())
        cat_docs = {0, 1, 5}
        assert len(top_docs & cat_docs) >= 2  # At least 2 of the 3 cat docs


class TestGetProbabilities:
    def test_returns_all_docs(self, scorer, small_corpus):
        probs = scorer.get_probabilities(["cat"])
        assert probs.shape == (len(small_corpus),)

    def test_bounds(self, scorer):
        probs = scorer.get_probabilities(["cat"])
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_nonzero_for_matching_docs(self, scorer):
        probs = scorer.get_probabilities(["cat"])
        # Documents 0, 1, 5 contain "cat" and should have nonzero probability
        for doc_id in [0, 1, 5]:
            assert probs[doc_id] > 0, f"Document {doc_id} should have nonzero probability"

    def test_zero_for_nonmatching_docs(self, scorer):
        probs = scorer.get_probabilities(["cat"])
        # Document 3 ("hello world") should have zero probability
        assert probs[3] == 0.0


class TestNoIndexError:
    def test_retrieve_before_index(self):
        s = BayesianBM25Scorer()
        with pytest.raises(RuntimeError, match="index"):
            s.retrieve([["cat"]])

    def test_get_probabilities_before_index(self):
        s = BayesianBM25Scorer()
        with pytest.raises(RuntimeError, match="index"):
            s.get_probabilities(["cat"])

    def test_doc_lengths_before_index(self):
        s = BayesianBM25Scorer()
        with pytest.raises(RuntimeError, match="index"):
            _ = s.doc_lengths

    def test_avgdl_before_index(self):
        s = BayesianBM25Scorer()
        with pytest.raises(RuntimeError, match="index"):
            _ = s.avgdl
