#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.scorer module."""

import numpy as np
import pytest

from bayesian_bm25.scorer import BayesianBM25Scorer, RetrievalResult


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


class TestBaseRateMethod:
    """Tests for the base_rate_method parameter (percentile, mixture, elbow)."""

    def test_invalid_method_raises(self):
        """Invalid base_rate_method raises ValueError."""
        with pytest.raises(ValueError, match="base_rate_method"):
            BayesianBM25Scorer(base_rate_method="invalid")

    def test_percentile_default(self, small_corpus):
        """Default method is 'percentile' and produces valid base rate."""
        s = BayesianBM25Scorer(method="lucene", base_rate="auto")
        s.index(small_corpus, show_progress=False)
        assert s.base_rate is not None
        assert 0.0 < s.base_rate < 1.0

    def test_mixture_method(self, small_corpus):
        """mixture method produces valid base rate in (0, 1)."""
        s = BayesianBM25Scorer(
            method="lucene", base_rate="auto", base_rate_method="mixture"
        )
        s.index(small_corpus, show_progress=False)
        assert s.base_rate is not None
        assert 0.0 < s.base_rate <= 0.5

    def test_elbow_method(self, small_corpus):
        """elbow method produces valid base rate in (0, 1)."""
        s = BayesianBM25Scorer(
            method="lucene", base_rate="auto", base_rate_method="elbow"
        )
        s.index(small_corpus, show_progress=False)
        assert s.base_rate is not None
        assert 0.0 < s.base_rate <= 0.5

    def test_mixture_bimodal_distribution(self):
        """Mixture EM recovers a sensible base rate from bimodal data."""
        # Low-scoring (non-relevant) population: mean=1, std=0.5
        # High-scoring (relevant) population: mean=5, std=0.5
        rng = np.random.default_rng(42)
        low = rng.normal(1.0, 0.5, size=900)
        high = rng.normal(5.0, 0.5, size=100)
        scores = np.concatenate([low, high])
        scores = scores[scores > 0]  # keep positive
        per_query_scores = [scores]

        result = BayesianBM25Scorer._base_rate_mixture(per_query_scores)
        # True proportion is 100/1000 = 0.1; EM should be in the ballpark
        assert 0.01 < result < 0.5

    def test_elbow_clear_knee(self):
        """Elbow method finds a knee in a clearly kinked distribution."""
        # Sharp drop: 10 high scores, then 90 low scores
        high = np.ones(10) * 10.0
        low = np.linspace(2.0, 0.1, 90)
        scores = np.concatenate([high, low])
        per_query_scores = [scores]

        result = BayesianBM25Scorer._base_rate_elbow(per_query_scores)
        assert 0.01 < result < 0.5

    def test_mixture_empty_returns_minimum(self):
        """Mixture with < 2 scores returns 1e-6."""
        result = BayesianBM25Scorer._base_rate_mixture([np.array([1.0])])
        assert result == pytest.approx(1e-6)

    def test_elbow_too_few_scores(self):
        """Elbow with < 3 scores returns 1e-6."""
        result = BayesianBM25Scorer._base_rate_elbow([np.array([1.0, 2.0])])
        assert result == pytest.approx(1e-6)

    def test_method_only_used_when_auto(self, small_corpus):
        """base_rate_method is irrelevant when base_rate is not 'auto'."""
        s = BayesianBM25Scorer(
            method="lucene", base_rate=0.01, base_rate_method="mixture"
        )
        s.index(small_corpus, show_progress=False)
        # Explicit base_rate wins over method
        assert s.base_rate == pytest.approx(0.01)

    def test_all_methods_produce_bounded_results(self):
        """All three methods produce results in [1e-6, 0.5]."""
        rng = np.random.default_rng(123)
        scores = rng.exponential(2.0, size=500)
        scores = scores[scores > 0]
        per_query_scores = [scores]

        for method_fn in [
            lambda pqs: BayesianBM25Scorer._base_rate_percentile(pqs, n_docs=1000),
            BayesianBM25Scorer._base_rate_mixture,
            BayesianBM25Scorer._base_rate_elbow,
        ]:
            result = method_fn(per_query_scores)
            assert 1e-6 <= result <= 0.5, f"Out of bounds: {result}"


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


class TestRetrievalResult:
    def test_retrieve_default_returns_tuple(self, scorer):
        """Default retrieve() returns a tuple for backward compatibility."""
        result = scorer.retrieve([["cat"]], k=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        doc_ids, probs = result
        assert doc_ids.shape == (1, 3)
        assert probs.shape == (1, 3)

    def test_retrieve_explain_returns_result(self, scorer):
        """explain=True returns a RetrievalResult."""
        result = scorer.retrieve([["cat"]], k=3, explain=True)
        assert isinstance(result, RetrievalResult)
        assert result.doc_ids.shape == (1, 3)
        assert result.probabilities.shape == (1, 3)
        assert result.explanations is not None

    def test_result_has_explanations(self, scorer):
        """explanations list has correct shape: [num_queries][k]."""
        result = scorer.retrieve([["cat"], ["dog"]], k=3, explain=True)
        assert len(result.explanations) == 2
        for q_explanations in result.explanations:
            assert len(q_explanations) == 3

    def test_explanation_traces_match_probabilities(self, scorer):
        """trace posteriors match result probabilities for nonzero docs."""
        result = scorer.retrieve([["cat"]], k=6, explain=True)
        for rank in range(6):
            prob = result.probabilities[0, rank]
            trace = result.explanations[0][rank]
            if prob > 0:
                assert trace is not None
                assert abs(trace.posterior - prob) < 1e-6
            else:
                assert trace is None


class TestAddDocuments:
    def test_add_documents_increases_count(self, scorer, small_corpus):
        """num_docs grows after adding documents."""
        original_count = scorer.num_docs
        scorer.add_documents(
            [["new", "document", "here"]], show_progress=False
        )
        assert scorer.num_docs == original_count + 1

    def test_add_documents_before_index_raises(self):
        """RuntimeError if add_documents is called before index()."""
        s = BayesianBM25Scorer()
        with pytest.raises(RuntimeError, match="index"):
            s.add_documents([["hello"]])

    def test_add_documents_preserves_search(self, small_corpus):
        """Old documents are still retrievable after adding new ones."""
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        s.index(small_corpus, show_progress=False)
        probs_before = s.get_probabilities(["cat"])
        cat_docs_before = set(np.where(probs_before > 0)[0].tolist())

        s.add_documents(
            [["completely", "unrelated", "tokens"]], show_progress=False
        )
        probs_after = s.get_probabilities(["cat"])
        cat_docs_after = set(np.where(probs_after > 0)[0].tolist())

        # Original cat docs (0, 1, 5) should still appear
        assert cat_docs_before.issubset(cat_docs_after)

    def test_add_documents_finds_new_docs(self, small_corpus):
        """Newly added documents appear in results."""
        s = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
        s.index(small_corpus, show_progress=False)
        new_doc_id = len(small_corpus)  # index of the new doc

        s.add_documents(
            [["cat", "cat", "cat", "cat", "cat"]], show_progress=False
        )
        probs = s.get_probabilities(["cat"])
        assert probs[new_doc_id] > 0


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
