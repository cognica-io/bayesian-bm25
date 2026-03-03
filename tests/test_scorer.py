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


class TestBaseRateScorer:
    def test_default_no_base_rate(self, scorer):
        """Default scorer has base_rate=None."""
        assert scorer.base_rate is None

    def test_explicit_base_rate(self, small_corpus):
        """base_rate=0.01 is stored and used."""
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=0.01)
        s.index(small_corpus, show_progress=False)
        assert s.base_rate == pytest.approx(0.01)

    def test_auto_base_rate(self, small_corpus):
        """base_rate="auto" produces a float in (0, 1)."""
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate="auto")
        s.index(small_corpus, show_progress=False)
        assert s.base_rate is not None
        assert 0.0 < s.base_rate < 1.0

    def test_base_rate_reduces_probabilities(self, small_corpus):
        """Nonzero probs are lower with base_rate=0.01 vs no base rate."""
        s_none = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        s_none.index(small_corpus, show_progress=False)

        s_low = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=0.01)
        s_low.index(small_corpus, show_progress=False)

        probs_none = s_none.get_probabilities(["cat"])
        probs_low = s_low.get_probabilities(["cat"])

        nonzero = probs_none > 0
        assert np.any(nonzero), "Expected some nonzero probabilities"
        assert np.all(probs_low[nonzero] < probs_none[nonzero])

    def test_ranking_preserved(self, small_corpus):
        """base_rate does not change document ranking order."""
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate=0.01)
        s.index(small_corpus, show_progress=False)

        doc_ids, probs = s.retrieve([["cat"]], k=6)
        nonzero_mask = probs[0] > 0
        nonzero_probs = probs[0][nonzero_mask]
        if len(nonzero_probs) > 1:
            assert np.all(np.diff(nonzero_probs) <= 0), (
                f"Ranking not preserved with base_rate: {nonzero_probs}"
            )


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


class TestEstimateBaseRate:
    """Tests for the _estimate_base_rate private method."""

    def test_empty_scores_returns_minimum(self, scorer):
        """Empty per_query_scores list returns the lower clamp 1e-6."""
        result = scorer._estimate_base_rate([], n_docs=100)
        assert result == pytest.approx(1e-6)

    def test_known_distribution(self, scorer, small_corpus):
        """Synthetic scores with known 95th percentile produce a sensible rate."""
        # Create a single set of scores where 5% are above the 95th pct
        scores = np.concatenate([
            np.ones(95) * 1.0,  # 95 docs scoring 1.0
            np.ones(5) * 10.0,  # 5 docs scoring 10.0
        ])
        per_query_scores = [scores]
        n_docs = len(small_corpus)
        result = scorer._estimate_base_rate(per_query_scores, n_docs)
        assert 0.0 < result <= 0.5

    def test_clamp_upper(self, scorer):
        """Extreme input that would yield high base rate is clamped to 0.5."""
        # All scores above any percentile -- ratio ~= 1.0
        scores = np.ones(100) * 5.0
        per_query_scores = [scores]
        result = scorer._estimate_base_rate(per_query_scores, n_docs=1)
        assert result <= 0.5

    def test_clamp_lower(self, scorer):
        """Tiny fraction above 95th pct is clamped to 1e-6."""
        # Only 1 score above threshold in a huge corpus
        scores = np.concatenate([np.zeros(999), np.array([100.0])])
        per_query_scores = [scores]
        result = scorer._estimate_base_rate(per_query_scores, n_docs=1_000_000)
        assert result >= 1e-6


class TestEmptyAndOOVQueries:
    """Tests for edge cases with empty or out-of-vocabulary queries."""

    def test_retrieve_empty_query(self, scorer):
        """Empty query returns zero probabilities."""
        doc_ids, probs = scorer.retrieve([[]], k=3)
        assert probs.shape == (1, 3)
        assert np.all(probs == 0.0)

    def test_retrieve_oov_query(self, scorer):
        """Query with only out-of-vocabulary tokens returns zero probs."""
        doc_ids, probs = scorer.retrieve([["xyznonexistent"]], k=3)
        assert probs.shape == (1, 3)
        assert np.all(probs == 0.0)

    def test_get_probabilities_oov(self, scorer, small_corpus):
        """get_probabilities with all OOV tokens returns zero probs."""
        probs = scorer.get_probabilities(["xyznonexistent_token"])
        assert probs.shape == (len(small_corpus),)
        assert np.all(probs == 0.0)


class TestSingleDocCorpus:
    """Tests for a single-document corpus edge case."""

    def test_single_doc_index_and_retrieve(self):
        """A 1-document corpus can be indexed and retrieved from."""
        corpus = [["hello", "world"]]
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        s.index(corpus, show_progress=False)
        assert s.num_docs == 1

        doc_ids, probs = s.retrieve([["hello"]], k=1)
        assert doc_ids.shape == (1, 1)
        assert probs.shape == (1, 1)
        assert probs[0, 0] >= 0.0

    def test_single_doc_auto_estimate(self):
        """Single-document corpus with std=0 falls back to alpha=1.0."""
        corpus = [["hello", "world"]]
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        s.index(corpus, show_progress=False)
        # With only one document, std of scores is 0, so alpha defaults to 1.0
        # The scorer should still function without errors
        probs = s.get_probabilities(["hello"])
        assert probs.shape == (1,)
        assert np.all(np.isfinite(probs))


class TestComputeTFBatch:
    """Direct tests for the _compute_tf_batch private method."""

    def test_known_counts(self, scorer):
        """Known corpus + query gives exact TF counts."""
        # Doc 0: ["the", "cat", "sat", "on", "the", "mat"]
        # Query: ["cat", "the"] -> intersection has "cat" and "the" = 2 unique
        doc_ids = np.array([0])
        result = scorer._compute_tf_batch(doc_ids, ["cat", "the"])
        assert result[0] == pytest.approx(2.0)

    def test_no_overlap(self, scorer):
        """All OOV query tokens produce zero TF."""
        doc_ids = np.array([0, 1, 2])
        result = scorer._compute_tf_batch(doc_ids, ["xyznonexistent"])
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_empty_document(self):
        """Document with zero tokens produces TF of 0."""
        corpus = [[], ["hello", "world"]]
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        s.index(corpus, show_progress=False)
        doc_ids = np.array([0])
        result = s._compute_tf_batch(doc_ids, ["hello"])
        assert result[0] == pytest.approx(0.0)
