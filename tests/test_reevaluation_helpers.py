#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for the helpers of benchmarks.reevaluation."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from bayesian_bm25.consistency import dilute_offset, expected_count  # noqa: E402
from bayesian_bm25.probability import sigmoid  # noqa: E402
from bayesian_bm25.scorer import BayesianBM25Scorer  # noqa: E402
from benchmarks.metrics import best_f1_threshold, threshold_f1  # noqa: E402
from benchmarks.prepared_dataset import PreparedDataset  # noqa: E402
from benchmarks.query_signals import query_signals  # noqa: E402
from benchmarks.reevaluation import (  # noqa: E402
    CORRECTED_VARIANTS,
    DENSITY_RATIO_VARIANTS,
    FUSION_CONFIGS,
    factorized_statistics,
    fit_kappas,
    fuse_candidates,
    fused_log_odds,
    lexical_scale,
    per_query_lexical_log_odds,
    run_dataset,
    score_convex,
    tail_units,
    threshold_transfer,
    training_sample,
    unified_posterior,
)


def _prepared(
    n_docs: int = 300,
    n_queries: int = 6,
    seed: int = 0,
    corpus: list[list[str]] | None = None,
    queries: list[list[str]] | None = None,
) -> PreparedDataset:
    """A small synthetic dataset with both signals scored for every document.

    The corpus and the queries are drawn at random unless given.
    """
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(120)]
    if corpus is None:
        corpus = [
            list(rng.choice(vocab, size=int(rng.integers(15, 60)))) for _ in range(n_docs)
        ]
    if queries is None:
        queries = [
            list(rng.choice(vocab, size=int(rng.integers(2, 8)))) for _ in range(n_queries)
        ]
    n_docs, n_queries = len(corpus), len(queries)

    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer.index(corpus, show_progress=False)
    bm25 = np.stack([scorer._bm25.get_scores(q) for q in queries]).astype(np.float32)
    dense = rng.uniform(-0.2, 0.9, size=(n_queries, n_docs)).astype(np.float32)

    corpus_ids = [f"d{i}" for i in range(n_docs)]
    query_ids = [f"q{i}" for i in range(n_queries)]
    relevant = [np.sort(rng.choice(n_docs, size=3, replace=False)) for _ in range(n_queries)]
    qrels = {
        qid: {corpus_ids[i]: 1 for i in rel}
        for qid, rel in zip(query_ids, relevant, strict=True)
    }
    return PreparedDataset(
        name="synthetic",
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        query_tokens=queries,
        bm25_scores=bm25,
        dense_scores=dense,
        excluded=np.full(n_queries, -1, dtype=np.int64),
        relevant=relevant,
        qrels=qrels,
        scorer=scorer,
        corpus_alpha=float(scorer._transform.alpha),
        corpus_beta=float(scorer._transform.beta),
        corpus_base_rate=0.01,
        dense_kappa=9.0,
        dense_beta=0.9,
    )


@pytest.fixture(scope="module")
def prepared():
    return _prepared()


class TestQuerySignals:
    def test_zero_match_tail_has_explicit_bulk_fallback(self, prepared):
        from dataclasses import replace

        sig = query_signals(prepared, 0, retrieve_k=50)
        sig = replace(sig, bm25=np.zeros_like(sig.bm25))
        lexical, dense = tail_units(sig, fallback_lexical_scale=0.7)
        np.testing.assert_array_equal(lexical, np.zeros_like(lexical))
        assert np.all(np.isfinite(dense))

    def test_lexical_list_holds_only_matching_documents(self, prepared):
        sig = query_signals(prepared, 0, retrieve_k=50)
        in_bm25 = sig.bm25_rank > 0
        assert np.all(sig.bm25[sig.candidates[in_bm25]] > 0.0)

    def test_ranks_follow_the_scores(self, prepared):
        sig = query_signals(prepared, 1, retrieve_k=50)
        in_dense = sig.dense_rank > 0
        order = np.argsort(sig.dense_rank[in_dense])
        scores = sig.dense[sig.candidates[in_dense]][order]
        assert np.all(np.diff(scores) <= 0.0)

    def test_excluded_document_is_absent_from_the_population(self):
        prepared = _prepared(seed=1)
        prepared.excluded[0] = 7
        sig = query_signals(prepared, 0, retrieve_k=50)
        assert len(sig.bm25) == prepared.n_docs - 1
        assert 7 not in sig.doc_index
        # Labels follow the documents, not their positions.
        relevant = set(prepared.relevant[0].tolist()) - {7}
        assert set(sig.doc_index[sig.labels > 0].tolist()) == relevant


class TestFusion:
    def test_range_normalization_without_floor_is_the_convex_combination(self, prepared):
        # An affine map of the score has the same min-max normalization
        # as the score, so the range-normalized fusion of a corpus-level
        # posterior with a consistent non-match is the convex combination.
        sig = query_signals(prepared, 2, retrieve_k=80)
        fused = fuse_candidates(
            prepared, sig, **FUSION_CONFIGS["Balanced-NoPrior-Consistent-Cos"],
        )
        np.testing.assert_allclose(fused, score_convex(sig), atol=1e-12)

    def test_consistent_non_match_makes_per_query_constants_cancel(self):
        # Section 5.6: when every document carries the lexical term, a
        # constant added to the lexical log-odds (here, a different base
        # rate) changes no fused order.  When non-matches are scored by
        # the dense term alone, the constant moves one group against the
        # other.
        low = _prepared(seed=13)
        high = _prepared(seed=13)
        high.corpus_base_rate = 0.4
        sig = query_signals(low, 3, retrieve_k=80)
        assert np.any(sig.bm25[sig.candidates] == 0.0)

        consistent = dict(
            lex_model="corpus", lex_norm="none", unmatched="consistent",
            dense_model="cos", dense_norm="corpus",
        )
        a = fuse_candidates(low, sig, **consistent)
        b = fuse_candidates(high, sig, **consistent)
        assert np.array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable"))

        dense_only = {**consistent, "unmatched": "dense_only"}
        c = fuse_candidates(low, sig, **dense_only)
        d = fuse_candidates(high, sig, **dense_only)
        assert not np.array_equal(np.argsort(c, kind="stable"), np.argsort(d, kind="stable"))

    def test_dense_only_treatment_scores_groups_differently(self, prepared):
        sig = query_signals(prepared, 3, retrieve_k=80)
        consistent = fuse_candidates(prepared, sig, **FUSION_CONFIGS["Local+Consistent"])
        dense_only = fuse_candidates(prepared, sig, **FUSION_CONFIGS["Local"])
        matched = sig.bm25[sig.candidates] > 0.0
        if np.any(~matched):
            assert not np.allclose(consistent[~matched], dense_only[~matched])

    def test_fused_posterior_meets_the_count_constraint(self):
        rng = np.random.default_rng(2)
        z_lex = np.where(rng.random(5000) < 0.3, rng.gamma(2.0, 1.0, 5000), 0.0)
        z_vec = rng.normal(0.0, 1.0, 5000)
        for scale in (0.5, 1.0):
            log_odds = fused_log_odds(z_lex, z_vec, 4.0, scale=scale, consistent=True)
            assert float(np.sum(sigmoid(log_odds))) == pytest.approx(4.0, rel=1e-8)

    def test_summing_correlated_evidence_overcounts(self):
        # Two copies of one signal: each is calibrated to the count, but
        # their sum expects more relevant documents than the count.
        rng = np.random.default_rng(3)
        z = rng.normal(0.0, 1.0, 20000)
        additive = fused_log_odds(z, z, 1.0, consistent=False)
        assert float(np.sum(sigmoid(additive))) > 2.0
        halved = fused_log_odds(z, z, 1.0, scale=0.5, consistent=False)
        assert float(np.sum(sigmoid(halved))) == pytest.approx(1.0, rel=1e-6)

    def test_per_query_lexical_log_odds(self):
        rng = np.random.default_rng(4)
        z = rng.gamma(2.0, 1.0, 3000)
        log_odds = per_query_lexical_log_odds(z, 2.5, kappa=1.3)
        assert expected_count(log_odds) == pytest.approx(2.5, rel=1e-8)
        assert np.array_equal(np.argsort(log_odds), np.argsort(z))


class TestFactorizedStatistics:
    def test_exact_for_a_single_term_query(self, prepared):
        # With one query term there is nothing to factorize.
        single = _prepared(seed=5)
        single.query_tokens[0] = [single.query_tokens[0][0]]
        scores = np.asarray(
            single.scorer._bm25.get_scores(single.query_tokens[0]), dtype=np.float64,
        )
        matched = scores[scores > 0]
        alpha = 1.0 / float(np.std(matched))
        psi, sigma = factorized_statistics(single, 0, len(scores), alpha=alpha)
        exact_psi = dilute_offset(alpha * scores, 1.0) - np.log(len(scores))
        assert psi == pytest.approx(exact_psi, abs=1e-5)
        assert sigma == pytest.approx(float(np.std(matched)), rel=1e-5)

    def test_matches_the_sum_of_term_contributions(self, prepared):
        # The factorized mean is exact for any query: scores are sums of
        # per-term contributions.
        scores = np.asarray(
            prepared.scorer._bm25.get_scores(prepared.query_tokens[1]), dtype=np.float64,
        )
        psi_small, _ = factorized_statistics(prepared, 1, len(scores), alpha=1e-6)
        # psi(alpha) ~ alpha * mean score as alpha -> 0.
        assert psi_small / 1e-6 == pytest.approx(float(np.mean(scores)), rel=1e-3)

    def test_equal_contributions_have_no_dispersion(self):
        # One query term, and every document that contains it has the same
        # term frequency and length, hence the same score.  The variance
        # is then a difference of equal second moments; its cancellation
        # error (2.2e-16 for this corpus) must not be returned as a scale.
        matching = [["x", f"a{i}", f"b{i}", f"c{i}"] for i in range(5)]
        others = [[f"p{i}", f"q{i}", f"r{i}", f"s{i}"] for i in range(40)]
        prepared = _prepared(corpus=matching + others, queries=[["x"]])
        scores = np.asarray(
            prepared.scorer._bm25.get_scores(prepared.query_tokens[0]), dtype=np.float64,
        )
        assert np.count_nonzero(scores) == 5
        assert len(np.unique(scores[scores > 0])) == 1

        _, sigma = factorized_statistics(prepared, 0, len(scores), alpha=1.0)
        assert sigma == 0.0
        # With an explicit slope the generating function stays exact.
        psi, _ = factorized_statistics(prepared, 0, len(scores), alpha=0.5)
        exact_psi = dilute_offset(0.5 * scores, 1.0) - np.log(len(scores))
        assert psi == pytest.approx(exact_psi, abs=1e-9)

    def test_excluded_document_is_left_out(self):
        prepared = _prepared(seed=6)
        scores = np.asarray(
            prepared.scorer._bm25.get_scores(prepared.query_tokens[0]), dtype=np.float64,
        )
        top = int(np.argmax(scores))
        full, _ = factorized_statistics(prepared, 0, len(scores), alpha=1.0)
        prepared.excluded[0] = top
        without, _ = factorized_statistics(prepared, 0, len(scores) - 1, alpha=1.0)
        assert without < full


class TestSupervisedFits:
    def test_training_sample_represents_the_population(self, prepared):
        sig = query_signals(prepared, 0, retrieve_k=40)
        rng = np.random.default_rng(7)
        positions, weights = training_sample(sig, rng, n_tail=50)
        assert float(np.sum(weights)) == pytest.approx(len(sig.bm25))
        assert len(np.unique(positions)) == len(positions)
        assert set(np.flatnonzero(sig.labels > 0)).issubset(set(positions.tolist()))
        assert set(sig.candidates.tolist()).issubset(set(positions.tolist()))

    def test_fit_recovers_a_known_slope(self):
        rng = np.random.default_rng(8)
        features, labels, weights = [], [], []
        for _ in range(60):
            z = rng.normal(0.0, 1.0, size=(2000, 1))
            log_odds = 1.7 * z[:, 0] - 6.0
            y = (rng.random(2000) < sigmoid(log_odds)).astype(np.float64)
            features.append(z)
            labels.append(y)
            weights.append(np.ones(2000))
        kappa = fit_kappas(features, labels, weights)
        assert kappa[0] == pytest.approx(1.7, abs=0.15)

    def test_fit_recovers_two_slopes(self):
        rng = np.random.default_rng(9)
        features, labels, weights = [], [], []
        for _ in range(60):
            z = rng.normal(0.0, 1.0, size=(2000, 2))
            log_odds = 0.6 * z[:, 0] + 2.0 * z[:, 1] - 6.5
            y = (rng.random(2000) < sigmoid(log_odds)).astype(np.float64)
            features.append(z)
            labels.append(y)
            weights.append(np.ones(2000))
        kappa = fit_kappas(features, labels, weights)
        assert kappa[0] == pytest.approx(0.6, abs=0.15)
        assert kappa[1] == pytest.approx(2.0, abs=0.2)


class TestThresholds:
    def test_exact_search_matches_brute_force(self):
        rng = np.random.default_rng(10)
        values = np.round(rng.normal(size=300), 1)
        labels = (rng.random(300) < sigmoid(2.0 * values - 1.0)).astype(np.float64)
        threshold, f1 = best_f1_threshold(values, labels)
        brute = max(threshold_f1(values, labels, t)[2] for t in np.unique(values))
        assert f1 == pytest.approx(brute)
        assert threshold_f1(values, labels, threshold)[2] == pytest.approx(f1)

    def test_missed_relevant_documents_lower_the_f1(self):
        values = np.array([3.0, 2.0, 1.0])
        labels = np.array([1.0, 1.0, 0.0])
        assert best_f1_threshold(values, labels)[1] == pytest.approx(1.0)
        assert best_f1_threshold(values, labels, n_missed=2)[1] == pytest.approx(2 / 3)

    def test_transfer_is_invariant_under_a_query_independent_monotone_map(self):
        # Proposition 6.1.
        rng = np.random.default_rng(11)
        values, labels, missed = [], [], []
        for _ in range(40):
            v = rng.gamma(2.0, 2.0, size=200)
            y = (rng.random(200) < sigmoid(0.8 * v - 6.0)).astype(np.float64)
            values.append(v)
            labels.append(y)
            missed.append(0)
        folds = np.arange(40) % 5
        raw = threshold_transfer(values, labels, missed, folds)
        mapped = threshold_transfer(
            [1.3 * v - 4.0 for v in values], labels, missed, folds,
        )
        cubed = threshold_transfer([v**3 for v in values], labels, missed, folds)
        for key in ("test_micro_f1", "test_macro_f1", "admitted_mean"):
            assert mapped[key] == pytest.approx(raw[key])
            assert cubed[key] == pytest.approx(raw[key])

    def test_transfer_changes_under_a_query_dependent_map(self):
        rng = np.random.default_rng(12)
        values, labels, missed = [], [], []
        for i in range(40):
            scale = 1.0 + 4.0 * (i % 2)
            z = rng.normal(size=300)
            y = (rng.random(300) < sigmoid(2.5 * z - 4.0)).astype(np.float64)
            values.append(scale * z)
            labels.append(y)
            missed.append(0)
        folds = np.arange(40) % 5
        raw = threshold_transfer(values, labels, missed, folds)
        standardized = threshold_transfer(
            [v / np.std(v) for v in values], labels, missed, folds,
        )
        assert standardized["test_micro_f1"] > raw["test_micro_f1"] + 0.02


class TestUnifiedPosterior:
    @pytest.fixture(scope="class")
    def query(self):
        prepared = _prepared(n_docs=400, n_queries=4, seed=5)
        sig = query_signals(prepared, 0, retrieve_k=50)
        z_lex = sig.bm25 / lexical_scale(sig, 1.0 / prepared.corpus_alpha)
        lexical = per_query_lexical_log_odds(z_lex, 1.0)
        prevalence = 1.0 / len(sig.bm25)
        prior = math.log(prevalence) - math.log1p(-prevalence)
        return sig, lexical, prior, prevalence

    @pytest.mark.parametrize("background", ["gaussian", "marginal"])
    def test_fused_log_odds_are_the_sum_of_equation_5_2(self, query, background):
        sig, lexical, prior, _ = query
        fused, vector = unified_posterior(sig, lexical, prior, background=background)
        # Lexical posterior plus vector evidence: the prior is counted once.
        np.testing.assert_allclose(
            fused, lexical[sig.candidates] + (vector - prior), rtol=0.0, atol=1e-12,
        )
        assert fused.shape == sig.candidates.shape

    @pytest.mark.parametrize("background", ["gaussian", "marginal"])
    def test_exact_form_is_the_correction_of_remark_4_5(self, query, background):
        sig, lexical, prior, prevalence = query
        _, approximate = unified_posterior(sig, lexical, prior, background=background)
        _, exact = unified_posterior(
            sig, lexical, prior, posterior="exact", background=background,
        )
        evidence = approximate - prior
        # Where pi * exp(evidence) stays below one the exact log-odds are
        # those of (4.1) plus log((1 - pi) / (1 - pi * exp(evidence))).
        unclipped = prevalence * np.exp(evidence) < 0.999
        assert np.count_nonzero(unclipped) > 10
        correction = np.log1p(-prevalence) - np.log1p(-prevalence * np.exp(evidence[unclipped]))
        np.testing.assert_allclose(
            exact[unclipped], approximate[unclipped] + correction, rtol=0.0, atol=1e-8,
        )
        # The correction has the sign of the evidence: (4.1) understates a
        # posterior that the distance supports and overstates one it opposes.
        supported = evidence[unclipped] > 0.0
        assert np.any(supported) and np.any(~supported)
        assert np.all(exact[unclipped][supported] > approximate[unclipped][supported])
        assert np.all(exact[unclipped][~supported] <= approximate[unclipped][~supported])

    def test_background_changes_the_evidence(self, query):
        sig, lexical, prior, _ = query
        _, gaussian = unified_posterior(sig, lexical, prior, background="gaussian")
        _, marginal = unified_posterior(sig, lexical, prior, background="marginal")
        assert not np.allclose(gaussian, marginal)

    def test_weighting_changes_the_relevant_density(self, query):
        sig, lexical, prior, _ = query
        _, weighted = unified_posterior(sig, lexical, prior)
        _, uniform = unified_posterior(sig, lexical, prior, weighting="uniform")
        _, sharpened = unified_posterior(sig, lexical, prior, weighting="sharpened")
        assert not np.allclose(weighted, uniform)
        assert not np.allclose(weighted, sharpened)

    def test_monotone_projection_orders_the_evidence_by_distance(self, query):
        sig, lexical, prior, _ = query
        _, vector = unified_posterior(sig, lexical, prior, monotone=True)
        distances = 1.0 - sig.dense[sig.candidates]
        order = np.argsort(distances, kind="stable")
        # Closer is never weaker evidence.
        assert np.all(np.diff(vector[order]) <= 1e-12)

    def test_rejects_unknown_options(self, query):
        sig, lexical, prior, _ = query
        with pytest.raises(ValueError, match="unknown background"):
            unified_posterior(sig, lexical, prior, background="residuals")
        with pytest.raises(ValueError, match="unknown weighting"):
            unified_posterior(sig, lexical, prior, weighting="gap")


class TestRunDataset:
    N_QUERIES = 12

    @pytest.fixture(scope="class")
    def result(self):
        prepared = _prepared(n_docs=400, n_queries=self.N_QUERIES, seed=3)
        return run_dataset(prepared, k=10, retrieve_k=50, n_folds=3)

    def test_every_ranking_row_covers_every_query(self, result):
        assert set(FUSION_CONFIGS) <= set(result["ranking"])
        for name, entry in result["ranking"].items():
            for metric in ("NDCG@10", "MAP@10", "Recall@10"):
                assert len(entry["per_query"][metric]) == self.N_QUERIES, name

    def test_posteriors_sum_to_the_convention(self, result):
        # Every synthetic query has three relevant documents, so a
        # posterior calibrated to R expects R / 3 of the observed count.
        lexical = result["calibration"]["lexical"]
        fused = result["calibration"]["fused"]
        assert lexical["Lex-Query-R1"]["all"]["expected_over_observed"] == pytest.approx(
            1.0 / 3.0, rel=1e-6,
        )
        assert lexical["Lex-Query-Roracle"]["all"]["expected_over_observed"] == pytest.approx(
            1.0, rel=1e-6,
        )
        for name in ("Fused-Query-Consistent-rho0", "Fused-Query-Consistent-rho0.5",
                     "Fused-Query-Consistent-rho1"):
            assert fused[name]["all"]["expected_over_observed"] == pytest.approx(
                1.0 / 3.0, rel=1e-6,
            )

    def test_every_density_ratio_variant_is_ranked_and_calibrated(self, result):
        fused = result["calibration"]["fused"]
        vector = result["calibration"]["vector"]
        for suffix, _, _, _ in DENSITY_RATIO_VARIANTS:
            assert f"Additive-Query-DensityRatio{suffix}" in result["ranking"]
            assert set(fused[f"Unified-Query{suffix}"]) == {"pool", "dense_list", "top10"}
            assert set(vector[f"Vec-DensityRatio{suffix}"]) == {"retrieved", "top10"}

    def test_corrections_of_the_unified_posterior(self, result):
        fused = result["calibration"]["fused"]
        # Every synthetic query has three relevant documents, all of them
        # in the population but not necessarily in the candidate pool, so
        # the expected count is checked directly: one per query.
        def expected_count(summary: dict) -> float:
            return summary["expected_over_observed"] * summary["prevalence"] * summary["n"]

        for suffix in CORRECTED_VARIANTS:
            for correction in ("-Consistent", "-Pooled-Consistent"):
                summary = fused[f"Unified-Query{suffix}{correction}"]["pool"]
                assert expected_count(summary) == pytest.approx(self.N_QUERIES, rel=1e-6)
            # Scaling the evidence by a positive constant keeps the order,
            # so the head of the ranking holds the same documents.
            summed = fused[f"Unified-Query{suffix}"]["top10"]
            pooled = fused[f"Unified-Query{suffix}-Pooled"]["top10"]
            assert pooled["n"] == summed["n"]
            assert pooled["prevalence"] == pytest.approx(summed["prevalence"], abs=1e-12)

    def test_pooled_fusion_is_measured_on_the_candidate_pool(self, result):
        fused = result["calibration"]["fused"]
        for name, pools in fused.items():
            if name.startswith("Fused-"):
                assert {"all", "pool", "top10"} <= set(pools), name
        # The candidate pool is part of the population, so a posterior
        # expects no more relevant documents on it than over the whole.
        def expected_count(summary: dict) -> float:
            return summary["expected_over_observed"] * summary["prevalence"] * summary["n"]

        consistent = fused["Fused-Query-Consistent-rho0.5"]
        assert consistent["pool"]["n"] < consistent["all"]["n"]
        assert expected_count(consistent["pool"]) <= expected_count(consistent["all"]) + 1e-9
        # Over the population the constraint holds exactly: one per query.
        assert expected_count(consistent["all"]) == pytest.approx(self.N_QUERIES, rel=1e-6)

    def test_diagnostics_are_finite(self, result):
        for key, value in result["diagnostics"].items():
            assert np.isfinite(value), key

    def test_threshold_rows_obey_proposition_6_1(self, result):
        # The corpus-level log-odds are an increasing function of the raw
        # score that is the same for every query.
        raw = result["threshold"]["Raw BM25"]
        corpus = result["threshold"]["Corpus-level log-odds"]
        for key in ("train_micro_f1", "test_micro_f1", "test_macro_f1", "admitted_mean"):
            assert corpus[key] == pytest.approx(raw[key], abs=1e-12)
