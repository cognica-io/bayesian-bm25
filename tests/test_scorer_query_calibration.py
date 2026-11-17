#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for per-query calibration and log-odds output of the scorers."""

import numpy as np
import pytest

from bayesian_bm25.multi_field import MultiFieldScorer
from bayesian_bm25.probability import BayesianProbabilityTransform, logit, sigmoid
from bayesian_bm25.query_calibration import QueryCalibration
from bayesian_bm25.scorer import BayesianBM25Scorer, RetrievalResult


def _corpus(n_docs: int = 400, seed: int = 0) -> tuple[list[list[str]], list[str]]:
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(250)]
    corpus = [
        list(rng.choice(vocab, size=int(rng.integers(20, 80)))) for _ in range(n_docs)
    ]
    return corpus, vocab


@pytest.fixture(scope="module")
def corpus_and_vocab():
    return _corpus()


@pytest.fixture(scope="module")
def query_scorer(corpus_and_vocab):
    corpus, _ = corpus_and_vocab
    scorer = BayesianBM25Scorer(
        method="lucene", calibration="query", kappa=1.0, expected_relevant=2.0,
    )
    scorer.index(corpus, show_progress=False)
    return scorer


@pytest.fixture(scope="module")
def corpus_scorer(corpus_and_vocab):
    corpus, _ = corpus_and_vocab
    scorer = BayesianBM25Scorer(method="lucene", base_rate="auto")
    scorer.index(corpus, show_progress=False)
    return scorer


def _queries(vocab: list[str]) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(1)
    return list(rng.choice(vocab, size=4)), list(rng.choice(vocab, size=120))


class TestQueryCalibrationMode:
    def test_posteriors_sum_to_expected_relevant(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        for query in _queries(vocab):
            probabilities = query_scorer.get_probabilities(query)
            assert float(np.sum(probabilities)) == pytest.approx(2.0, rel=1e-8)

    def test_bm25_ranking_is_preserved(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        for query in _queries(vocab):
            raw = np.asarray(query_scorer._bm25.get_scores(query), dtype=np.float64)
            log_odds = query_scorer.get_log_odds(query)
            order = np.argsort(raw, kind="stable")
            assert np.all(np.diff(log_odds[order]) >= 0.0)

    def test_slope_follows_the_scale_of_the_query(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        short, long = _queries(vocab)
        short_cal = query_scorer.calibrate_query(short)
        long_cal = query_scorer.calibrate_query(long)
        assert isinstance(short_cal, QueryCalibration)
        # A long query has more dispersed scores, hence a smaller slope.
        assert long_cal.scale > short_cal.scale
        assert long_cal.slope < short_cal.slope
        assert short_cal.slope == pytest.approx(1.0 / short_cal.scale)

    def test_bounded_confidence(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        _, long = _queries(vocab)
        probabilities = query_scorer.get_probabilities(long)
        for p in (0.25, 0.5, 0.9):
            assert int(np.sum(probabilities >= p)) <= 2.0 / p

    def test_non_matching_documents_get_a_small_positive_probability(
        self, query_scorer, corpus_and_vocab,
    ):
        _, vocab = corpus_and_vocab
        short, _ = _queries(vocab)
        raw = np.asarray(query_scorer._bm25.get_scores(short))
        probabilities = query_scorer.get_probabilities(short)
        unmatched = raw == 0
        assert np.any(unmatched)
        assert np.all(probabilities[unmatched] > 0.0)
        assert len(np.unique(probabilities[unmatched])) == 1
        assert np.max(probabilities[unmatched]) < np.min(probabilities[~unmatched])

    def test_query_without_any_match(self, query_scorer, corpus_and_vocab):
        corpus, _ = corpus_and_vocab
        probabilities = query_scorer.get_probabilities(["not-in-vocabulary"])
        np.testing.assert_allclose(probabilities, 2.0 / len(corpus))

    def test_retrieve_matches_get_probabilities(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        short, long = _queries(vocab)
        doc_ids, probabilities = query_scorer.retrieve([short, long], k=5)
        assert doc_ids.shape == (2, 5)
        for row, query in enumerate((short, long)):
            full = query_scorer.get_probabilities(query)
            np.testing.assert_allclose(probabilities[row], full[doc_ids[row]], rtol=1e-6)
            assert np.all(np.diff(probabilities[row]) <= 1e-12)

    def test_retrieve_log_odds(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        short, _ = _queries(vocab)
        _, probabilities = query_scorer.retrieve([short], k=5)
        _, log_odds = query_scorer.retrieve([short], k=5, return_log_odds=True)
        np.testing.assert_allclose(sigmoid(log_odds), probabilities, rtol=1e-9)

    def test_explain_reports_the_two_term_posterior(self, query_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        short, _ = _queries(vocab)
        result = query_scorer.retrieve([short], k=3, explain=True)
        assert isinstance(result, RetrievalResult)
        assert result.calibrations is not None
        calibration = result.calibrations[0]
        trace = result.explanations[0][0]
        assert trace.posterior == pytest.approx(result.probabilities[0, 0], rel=1e-6)
        # No document-quality prior enters per-query calibration.
        assert trace.composite_prior == 0.5
        assert trace.logit_prior == pytest.approx(0.0)
        assert trace.alpha == pytest.approx(calibration.slope)
        assert trace.base_rate == pytest.approx(calibration.prevalence)

    def test_long_query_saturates_corpus_level_but_not_per_query(
        self, query_scorer, corpus_scorer, corpus_and_vocab,
    ):
        _, vocab = corpus_and_vocab
        _, long = _queries(vocab)

        corpus_level = corpus_scorer.get_probabilities(long)
        head = np.sort(corpus_level)[::-1][:100]
        assert len(np.unique(head)) < 10

        per_query = query_scorer.get_probabilities(long)
        raw = np.asarray(query_scorer._bm25.get_scores(long), dtype=np.float64)
        top = np.argsort(-raw)[:100]
        # Distinct scores keep distinct probabilities.
        assert len(np.unique(per_query[top])) == len(np.unique(raw[top]))

    def test_properties(self, query_scorer):
        assert query_scorer.calibration == "query"
        assert query_scorer.kappa == 1.0
        assert query_scorer.expected_relevant == 2.0

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="calibration must be one of"):
            BayesianBM25Scorer(calibration="document")
        with pytest.raises(ValueError, match="kappa must be positive"):
            BayesianBM25Scorer(calibration="query", kappa=0.0)
        with pytest.raises(ValueError, match="expected_relevant must be positive"):
            BayesianBM25Scorer(calibration="query", expected_relevant=0.0)

    def test_expected_relevant_must_be_below_the_corpus_size(self):
        corpus, _ = _corpus(n_docs=5)
        scorer = BayesianBM25Scorer(calibration="query", expected_relevant=5.0)
        with pytest.raises(ValueError, match="smaller than the corpus size"):
            scorer.index(corpus, show_progress=False)

    def test_calibrate_query_requires_an_index(self):
        with pytest.raises(RuntimeError, match="Call index"):
            BayesianBM25Scorer(calibration="query").calibrate_query(["a"])


class TestCorpusLevelLogOdds:
    def test_log_odds_agree_with_probabilities(self, corpus_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        short, _ = _queries(vocab)
        probabilities = corpus_scorer.get_probabilities(short)
        log_odds = corpus_scorer.get_log_odds(short)
        matched = probabilities > 0.0
        np.testing.assert_allclose(
            sigmoid(log_odds[matched]), probabilities[matched], rtol=1e-9,
        )
        assert np.all(np.isneginf(log_odds[~matched]))

    def test_log_odds_keep_the_order_that_probabilities_lose(
        self, corpus_scorer, corpus_and_vocab,
    ):
        _, vocab = corpus_and_vocab
        _, long = _queries(vocab)
        probabilities = corpus_scorer.get_probabilities(long)
        log_odds = corpus_scorer.get_log_odds(long)
        top = np.argsort(-log_odds)[:100]
        assert len(np.unique(log_odds[top])) > 90
        assert len(np.unique(probabilities[top])) < 10

    def test_retrieve_log_odds(self, corpus_scorer, corpus_and_vocab):
        _, vocab = corpus_and_vocab
        short, _ = _queries(vocab)
        doc_ids, log_odds = corpus_scorer.retrieve([short], k=5, return_log_odds=True)
        full = corpus_scorer.get_log_odds(short)
        np.testing.assert_allclose(log_odds[0], full[doc_ids[0]], rtol=1e-6)

    def test_calibrate_query_is_available_in_corpus_mode(
        self, corpus_scorer, corpus_and_vocab,
    ):
        _, vocab = corpus_and_vocab
        short, _ = _queries(vocab)
        assert corpus_scorer.calibration == "corpus"
        assert isinstance(corpus_scorer.calibrate_query(short), QueryCalibration)


class TestTransformLogOdds:
    def test_score_to_log_odds_is_the_logit_of_the_probability(self):
        transform = BayesianProbabilityTransform(alpha=0.8, beta=3.0, base_rate=0.02)
        scores = np.array([0.5, 2.0, 4.0, 7.0])
        tfs = np.array([1.0, 3.0, 5.0, 8.0])
        ratios = np.array([0.4, 0.9, 1.3, 0.6])
        np.testing.assert_allclose(
            sigmoid(transform.score_to_log_odds(scores, tfs, ratios)),
            transform.score_to_probability(scores, tfs, ratios),
            rtol=1e-9,
        )

    def test_three_additive_terms(self):
        transform = BayesianProbabilityTransform(alpha=0.8, beta=3.0, base_rate=0.02)
        prior = transform.composite_prior(4.0, 0.7)
        expected = 0.8 * (5.0 - 3.0) + float(logit(prior)) + float(logit(0.02))
        assert transform.score_to_log_odds(5.0, 4.0, 0.7) == pytest.approx(expected)

    def test_log_odds_are_strictly_increasing_where_probabilities_tie(self):
        transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([40.0, 60.0, 300.0])
        log_odds = transform.score_to_log_odds(scores, 5.0, 0.5)
        assert log_odds[0] < log_odds[1] < log_odds[2]
        probabilities = transform.score_to_probability(scores, 5.0, 0.5)
        assert probabilities[0] == probabilities[1] == probabilities[2]

    def test_prior_free_mode_has_no_prior_term(self):
        transform = BayesianProbabilityTransform(alpha=0.5, beta=1.0)
        transform._training_mode = "prior_free"
        assert transform.score_to_log_odds(3.0, 9.0, 0.5) == pytest.approx(1.0)

    def test_custom_prior_function(self):
        transform = BayesianProbabilityTransform(
            alpha=0.5, beta=1.0, prior_fn=lambda score, tf, ratio: 0.8,
        )
        assert transform.score_to_log_odds(3.0, 1.0, 1.0) == pytest.approx(
            1.0 + float(logit(0.8))
        )

    def test_log_likelihood_ratio(self):
        transform = BayesianProbabilityTransform(alpha=0.7, beta=2.0)
        assert transform.log_likelihood_ratio(5.0) == pytest.approx(2.1)
        assert isinstance(transform.log_likelihood_ratio(5.0), float)

    def test_wand_bound_in_log_odds_dominates_every_document(self):
        transform = BayesianProbabilityTransform(alpha=0.9, beta=2.0, base_rate=0.01)
        rng = np.random.default_rng(2)
        scores = rng.uniform(0.0, 12.0, size=200)
        tfs = rng.uniform(0.0, 12.0, size=200)
        ratios = rng.uniform(0.1, 2.0, size=200)
        bound = transform.wand_upper_bound_log_odds(12.0)
        assert np.all(transform.score_to_log_odds(scores, tfs, ratios) <= bound)

    def test_wand_bound_in_log_odds_keeps_discriminating(self):
        transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        assert transform.wand_upper_bound_log_odds(80.0) > (
            transform.wand_upper_bound_log_odds(50.0)
        )
        assert transform.wand_upper_bound(80.0) == transform.wand_upper_bound(50.0)

    def test_wand_bound_rejects_invalid_prior(self):
        transform = BayesianProbabilityTransform()
        with pytest.raises(ValueError, match="p_max must be in"):
            transform.wand_upper_bound_log_odds(3.0, p_max=1.0)


class TestMultiFieldLogOdds:
    @pytest.fixture(scope="class")
    def scorer(self):
        rng = np.random.default_rng(3)
        vocab = [f"w{i}" for i in range(60)]
        documents = [
            {
                "title": list(rng.choice(vocab, size=4)),
                "body": list(rng.choice(vocab, size=30)),
            }
            for _ in range(120)
        ]
        scorer = MultiFieldScorer(fields=["title", "body"], method="lucene")
        scorer.index(documents, show_progress=False)
        return scorer

    def test_log_odds_agree_with_probabilities(self, scorer):
        query = ["w1", "w7", "w20"]
        np.testing.assert_allclose(
            sigmoid(scorer.get_log_odds(query)), scorer.get_probabilities(query),
            rtol=1e-9,
        )

    def test_retrieve_ranks_on_log_odds(self, scorer):
        query = ["w1", "w7", "w20"]
        doc_ids, probabilities = scorer.retrieve(query, k=5)
        _, log_odds = scorer.retrieve(query, k=5, return_log_odds=True)
        full = scorer.get_log_odds(query)
        np.testing.assert_allclose(log_odds, full[doc_ids])
        assert np.all(np.diff(log_odds) <= 0.0)
        np.testing.assert_allclose(sigmoid(log_odds), probabilities)
