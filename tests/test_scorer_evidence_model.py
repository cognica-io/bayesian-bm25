#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""End-to-end retrieval under quality priors and ceiling calibration."""

import numpy as np
import pytest

from bayesian_bm25 import BayesianBM25Scorer
from bayesian_bm25.probability import sigmoid


@pytest.fixture
def corpus():
    return [
        ["foo"] * 5 + ["x"] * 15,
        ["foo"] + ["bar"] * 6,
        ["other"] * 20,
    ]


class TestScorerEvidenceModel:
    def test_quality_prior_can_select_a_document_outside_raw_top_k(self, corpus):
        scorer = BayesianBM25Scorer(method="lucene", alpha=0.1, beta=0.0)
        scorer.index(corpus, show_progress=False)
        raw = np.asarray(scorer._bm25.get_scores(["foo"]))
        assert np.argmax(raw) == 0
        assert np.argmax(scorer.get_log_odds(["foo"])) == 1
        ids, probabilities = scorer.retrieve([["foo"]], k=1)
        assert ids[0, 0] == 1
        assert probabilities[0, 0] == pytest.approx(scorer.get_probabilities(["foo"])[1])

    def test_ceiling_and_count_are_applied_to_the_corpus(self, corpus):
        scorer = BayesianBM25Scorer(
            method="lucene", calibration="query", expected_relevant=0.8, ceiling=0.4,
        )
        scorer.index(corpus, show_progress=False)
        for query in (["foo"], ["unknown"], []):
            probabilities = scorer.get_probabilities(query)
            assert probabilities.sum() == pytest.approx(0.8)
            assert np.max(probabilities) <= 0.4 + 1e-15
            result = scorer.retrieve([query], k=3, explain=True)
            expected_order = (
                np.argsort(-scorer._bm25.get_scores(query), kind="stable")
                if query else np.arange(len(corpus))
            )
            np.testing.assert_array_equal(result.doc_ids[0], expected_order)
            np.testing.assert_allclose(
                result.probabilities[0], probabilities[result.doc_ids[0]],
            )
            np.testing.assert_allclose(sigmoid(result.posterior_log_odds), result.probabilities)
            assert all(trace is not None for trace in result.explanations[0])

    def test_per_query_overrides_preserve_default_parameters(self, corpus):
        scorer = BayesianBM25Scorer(method="lucene", calibration="query")
        scorer.index(corpus, show_progress=False)
        result = scorer.retrieve(
            [["foo"], ["other"]], k=2, explain=True,
            expected_relevant=np.array([0.5, 1.2]),
            ceiling=np.array([0.3, 0.8]), kappa=np.array([1.2, 0.5]),
        )
        for idx, query in enumerate((["foo"], ["other"])):
            model = result.calibrations[idx]
            population = scorer._bm25.get_scores(query)
            assert np.sum(model.probability(population)) == pytest.approx([0.5, 1.2][idx])
            assert model.ceiling == [0.3, 0.8][idx]
            probabilities = scorer.get_probabilities(
                query, expected_relevant=[0.5, 1.2][idx], ceiling=[0.3, 0.8][idx],
                kappa=[1.2, 0.5][idx],
            )
            assert np.sum(probabilities) == pytest.approx([0.5, 1.2][idx])
        assert scorer.expected_relevant == 1.0
        assert scorer.ceiling == 1.0
        assert scorer.kappa == 1.0

    def test_explanations_describe_the_capped_posterior(self, corpus):
        scorer = BayesianBM25Scorer(
            method="lucene", calibration="query", expected_relevant=0.8, ceiling=0.4,
        )
        scorer.index(corpus, show_progress=False)
        result = scorer.retrieve([["foo"]], k=2, explain=True)
        for rank, trace in enumerate(result.explanations[0]):
            assert trace.posterior == pytest.approx(result.probabilities[0, rank])
            assert trace.posterior_log_odds == pytest.approx(result.posterior_log_odds[0, rank])
            assert trace.ranking_score == pytest.approx(result.ranking_scores[0, rank])
            assert trace.ceiling == 0.4

    def test_log_odds_output_keeps_ranking_metadata(self, corpus):
        scorer = BayesianBM25Scorer(
            method="lucene", calibration="query", expected_relevant=0.8, ceiling=0.4,
        )
        scorer.index(corpus, show_progress=False)
        result = scorer.retrieve([["foo"]], k=3, return_log_odds=True, explain=True)
        np.testing.assert_array_equal(result.probabilities, result.posterior_log_odds)
        assert np.all(np.diff(result.ranking_scores[0]) <= 0.0)

    def test_override_shapes_are_checked(self, corpus):
        scorer = BayesianBM25Scorer(method="lucene", calibration="query")
        scorer.index(corpus, show_progress=False)
        with pytest.raises(ValueError, match="one value per query"):
            scorer.retrieve([["foo"]], k=1, expected_relevant=np.array([0.5, 0.8]))

    def test_corpus_mode_rejects_query_overrides(self, corpus):
        scorer = BayesianBM25Scorer(method="lucene")
        scorer.index(corpus, show_progress=False)
        with pytest.raises(ValueError, match="require calibration='query'"):
            scorer.retrieve([["foo"]], k=1, ceiling=0.4)
