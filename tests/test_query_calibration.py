#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.query_calibration (Procedure 3.14)."""

import numpy as np
import pytest

from bayesian_bm25.consistency import dilute_offset
from bayesian_bm25.probability import logit
from bayesian_bm25.query_calibration import (
    QueryCalibration,
    calibrate_query,
    score_dispersion,
)


def _query(seed: int = 0, n_match: int = 400, n_docs: int = 20000):
    rng = np.random.default_rng(seed)
    matched = rng.gamma(3.0, 1.5, size=n_match)
    population = np.concatenate([matched, np.zeros(n_docs - n_match)])
    return matched, population


class TestScoreDispersion:
    def test_std(self):
        s = np.array([1.0, 2.0, 4.0, 7.0])
        assert score_dispersion(s, "std") == pytest.approx(np.std(s))

    def test_robust_statistics_estimate_gaussian_std(self):
        rng = np.random.default_rng(0)
        s = rng.normal(5.0, 2.0, size=200000)
        assert score_dispersion(s, "mad") == pytest.approx(2.0, rel=0.02)
        assert score_dispersion(s, "iqr") == pytest.approx(2.0, rel=0.02)

    def test_robust_statistics_resist_an_outlier(self):
        rng = np.random.default_rng(1)
        s = np.append(rng.normal(5.0, 1.0, size=500), 1e6)
        assert score_dispersion(s, "std") > 1000.0
        assert score_dispersion(s, "mad") < 2.0

    def test_degenerate_inputs_have_zero_dispersion(self):
        assert score_dispersion(np.array([3.0])) == 0.0
        assert score_dispersion(np.array([])) == 0.0
        assert score_dispersion(np.full(10, 2.5)) == 0.0

    def test_rounding_error_is_not_dispersion(self):
        # The mean of n equal values need not equal them in floating
        # point, which leaves a standard deviation of order 1e-16 times
        # their magnitude.  As a scale it would give a slope of 1e16.
        s = np.full(3, 0.1)
        assert np.std(s) > 0.0
        for method in ("std", "mad", "iqr"):
            assert score_dispersion(s, method) == 0.0
        assert score_dispersion(np.full(12345, 17.77)) == 0.0

    def test_small_genuine_variation_is_kept(self):
        s = np.array([1e6, 1e6 + 1e-3])
        assert score_dispersion(s) == pytest.approx(5e-4, rel=1e-3)

    def test_large_translation_does_not_remove_genuine_variation(self):
        s = np.array([1.0, 2.0, 3.0, 4.0])
        assert score_dispersion(s + 1e12) == pytest.approx(score_dispersion(s))

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            score_dispersion(np.arange(5.0), "range")


class TestCalibrateQuery:
    def test_ceiling_preserves_count_and_changes_midpoint_semantics(self):
        matched, population = _query()
        model = calibrate_query(
            matched, expected_relevant=3.0, population_size=len(population), ceiling=0.4,
        )
        assert np.sum(model.probability(population)) == pytest.approx(3.0, rel=1e-9)
        assert np.max(model.probability(population)) <= 0.4
        assert model.evidence(model.midpoint) == pytest.approx(0.0, abs=1e-12)
        with pytest.raises(ValueError, match="no finite even-odds"):
            _ = model.even_odds_score

    def test_ceiling_even_odds_and_prior_bounds(self):
        scores = np.linspace(-2.0, 4.0, 200)
        priors = np.linspace(-1.0, 1.0, 200)
        model = calibrate_query(
            scores, expected_relevant=5.0, ceiling=0.8, document_prior_log_odds=priors,
        )
        assert np.sum(model.probability(scores, priors)) == pytest.approx(5.0)
        assert model.probability(model.even_odds_score) == pytest.approx(0.5)
        prior_max = 1.0 / (1.0 + np.exp(-1.0))
        assert np.all(model.log_odds(scores, priors) <= model.upper_bound_log_odds(4.0, prior_max))
        assert np.all(
            model.ranking_score(scores, priors) <= model.upper_bound_ranking_score(4.0, prior_max)
        )
        np.testing.assert_allclose(
            model.evidence(scores, priors) + model.prior_log_odds,
            model.log_odds(scores, priors),
        )

    def test_ceiling_is_affine_equivariant(self):
        matched, population = _query()
        base = calibrate_query(matched, population_size=len(population), ceiling=0.4)
        mapped = calibrate_query(
            20.0 * matched + 3.0, population_size=len(population), unscored_score=3.0,
            ceiling=0.4,
        )
        np.testing.assert_allclose(
            mapped.probability(20.0 * matched + 3.0), base.probability(matched), rtol=1e-9,
        )

    def test_posteriors_sum_to_expected_relevant(self):
        matched, population = _query()
        for target in (0.5, 1.0, 3.0, 25.0):
            cal = calibrate_query(
                matched, expected_relevant=target, population_size=len(population),
            )
            assert float(np.sum(cal.probability(population))) == pytest.approx(
                target, rel=1e-8
            )

    def test_slope_is_kappa_over_dispersion(self):
        matched, population = _query()
        cal = calibrate_query(matched, kappa=1.7, population_size=len(population))
        assert cal.slope == pytest.approx(1.7 / np.std(matched))
        assert cal.scale == pytest.approx(np.std(matched))
        assert cal.kappa == 1.7

    def test_prevalence_and_midpoint(self):
        matched, population = _query()
        cal = calibrate_query(
            matched, expected_relevant=4.0, population_size=len(population),
        )
        assert cal.prevalence == pytest.approx(4.0 / len(population))
        assert cal.prior_log_odds == pytest.approx(float(logit(cal.prevalence)))
        # offset = slope * midpoint - logit(prevalence)
        assert cal.offset == pytest.approx(
            cal.slope * cal.midpoint - cal.prior_log_odds
        )
        # The evidence changes sign at the midpoint.
        assert cal.evidence(cal.midpoint) == pytest.approx(0.0, abs=1e-12)
        # A document at the even-odds score is as likely relevant as not.
        assert cal.probability(cal.even_odds_score) == pytest.approx(0.5)

    def test_evidence_is_log_odds_minus_prior(self):
        matched, population = _query()
        cal = calibrate_query(matched, population_size=len(population))
        np.testing.assert_allclose(
            cal.evidence(matched), cal.log_odds(matched) - cal.prior_log_odds,
        )

    def test_rank_preservation(self):
        # Corollary 3.18: the posterior is a monotone transform of the score.
        matched, population = _query()
        cal = calibrate_query(matched, population_size=len(population))
        order = np.argsort(matched)
        assert np.all(np.diff(cal.log_odds(matched)[order]) >= 0.0)
        assert np.array_equal(np.argsort(cal.log_odds(matched)), order)

    def test_scale_equivariance(self):
        # Proposition 3.11: an affine image of the scores, with the
        # non-matches mapped along, gives the same posteriors.
        matched, population = _query()
        a, b = 20.0, 3.0
        base = calibrate_query(matched, population_size=len(population))
        mapped = calibrate_query(
            a * matched + b,
            population_size=len(population),
            unscored_score=b,
        )
        assert mapped.slope == pytest.approx(base.slope / a)
        assert mapped.midpoint == pytest.approx(a * base.midpoint + b)
        np.testing.assert_allclose(
            mapped.probability(a * matched + b), base.probability(matched), rtol=1e-9,
        )

    def test_long_query_does_not_inflate_evidence(self):
        # Scores twenty times larger leave every posterior unchanged,
        # where a query-independent slope multiplies the evidence by twenty.
        matched, population = _query()
        short = calibrate_query(matched, population_size=len(population))
        long = calibrate_query(20.0 * matched, population_size=len(population))
        np.testing.assert_allclose(
            long.probability(20.0 * matched), short.probability(matched), rtol=1e-9,
        )

    def test_bounded_confidence(self):
        # At most R / p documents can have posterior p or more.
        matched, population = _query()
        target = 2.0
        cal = calibrate_query(
            100.0 * matched, expected_relevant=target, population_size=len(population),
        )
        probabilities = cal.probability(100.0 * population)
        for p in (0.1, 0.5, 0.9):
            assert int(np.sum(probabilities >= p)) <= target / p

    def test_dilute_limit(self):
        # Corollary 3.8: when no posterior is large, calibration is R
        # times a softmax over the population.
        rng = np.random.default_rng(3)
        scores = rng.normal(0.0, 1.0, size=200000)
        cal = calibrate_query(scores, kappa=1.0, expected_relevant=1.0)
        softmax = np.exp(cal.slope * scores)
        softmax /= np.sum(softmax)
        np.testing.assert_allclose(cal.probability(scores), softmax, rtol=1e-2)

    def test_dilute_intercept_option(self):
        matched, population = _query()
        cal = calibrate_query(
            matched, population_size=len(population), intercept="dilute",
        )
        assert cal.offset == pytest.approx(
            dilute_offset(cal.slope * population, 1.0)
        )
        exact = calibrate_query(matched, population_size=len(population))
        assert cal.offset >= exact.offset

    def test_gaussian_midpoint_lies_above_the_non_relevant_mean(self):
        # For Gaussian non-relevant scores and small prevalence the
        # consistent midpoint is mu_0 + alpha * sigma_0^2 / 2, not the
        # median: median anchoring is overconfident.
        rng = np.random.default_rng(4)
        mu_0, sigma_0 = 10.0, 2.0
        scores = rng.normal(mu_0, sigma_0, size=400000)
        cal = calibrate_query(scores, kappa=1.0, expected_relevant=1.0)
        expected = mu_0 + cal.slope * sigma_0**2 / 2.0
        assert cal.midpoint == pytest.approx(expected, abs=0.05)
        assert cal.midpoint > np.median(scores) + 0.5

    def test_non_match_evidence_depends_on_coverage(self):
        # Remark 3.10: absence of a match is weak evidence for a selective
        # query and strong evidence for one that most documents match.
        rng = np.random.default_rng(5)
        n_docs = 50000
        selective = rng.gamma(3.0, 1.5, size=50)
        broad = rng.gamma(3.0, 1.5, size=45000)
        e_selective = calibrate_query(
            selective, population_size=n_docs, scale=2.0,
        ).evidence(0.0)
        e_broad = calibrate_query(
            broad, population_size=n_docs, scale=2.0,
        ).evidence(0.0)
        assert e_broad < e_selective < 0.0
        assert abs(e_selective) < 1.0 < abs(e_broad)

    def test_explicit_scale_overrides_dispersion(self):
        matched, population = _query()
        cal = calibrate_query(matched, scale=4.0, population_size=len(population))
        assert cal.slope == pytest.approx(0.25)

    def test_fallback_scale_for_a_single_match(self):
        cal = calibrate_query(
            np.array([7.5]), population_size=1000, fallback_scale=2.0,
        )
        assert cal.slope == pytest.approx(0.5)
        population = np.concatenate([[7.5], np.zeros(999)])
        assert float(np.sum(cal.probability(population))) == pytest.approx(1.0)

    def test_no_dispersion_without_fallback_raises(self):
        with pytest.raises(ValueError, match="no dispersion"):
            calibrate_query(np.array([7.5]), population_size=1000)

    def test_equal_scores_use_the_fallback_scale(self):
        # Three equal scores: numpy reports a standard deviation of
        # 1.4e-17, which must not become the scale of the query.
        scores = np.full(3, 0.1)
        with pytest.raises(ValueError, match="no dispersion"):
            calibrate_query(scores, population_size=1000)
        cal = calibrate_query(scores, population_size=1000, fallback_scale=2.0)
        assert cal.slope == pytest.approx(0.5)
        population = np.concatenate([scores, np.zeros(997)])
        assert float(np.sum(cal.probability(population))) == pytest.approx(1.0)

    def test_match_mask_restricts_the_dispersion(self):
        matched, population = _query()
        mask = population > 0.0
        cal = calibrate_query(population, match_mask=mask)
        reference = calibrate_query(matched, population_size=len(population))
        assert cal.slope == pytest.approx(reference.slope)
        assert cal.offset == pytest.approx(reference.offset, abs=1e-8)

    def test_weights_represent_a_sampled_tail(self):
        # The retrieved list plus a weighted uniform sample of the rest
        # reproduces the offset of the whole population.
        rng = np.random.default_rng(6)
        scores = rng.gamma(3.0, 1.5, size=30000)
        exact = calibrate_query(scores, scale=2.0)
        order = np.argsort(-scores)
        top, rest = order[:1000], order[1000:]
        sample = rng.choice(rest, size=3000, replace=False)
        approx = calibrate_query(
            np.concatenate([scores[top], scores[sample]]),
            scale=2.0,
            weights=np.concatenate(
                [np.ones(1000), np.full(3000, len(rest) / 3000)]
            ),
        )
        assert approx.offset == pytest.approx(exact.offset, abs=0.05)

    def test_document_prior_enters_the_count_constraint(self):
        matched, population = _query()
        rng = np.random.default_rng(7)
        prior = rng.uniform(-1.0, 1.0, size=len(matched))
        cal = calibrate_query(
            matched,
            population_size=len(population),
            document_prior_log_odds=prior,
            expected_relevant=2.0,
        )
        total = float(np.sum(cal.probability(matched, prior)))
        total += (len(population) - len(matched)) * float(cal.probability(0.0))
        assert total == pytest.approx(2.0, rel=1e-8)

    def test_upper_bound_log_odds_is_exact_for_pruning(self):
        # Theorem 3.20: a score upper bound maps to a log-odds upper bound.
        matched, population = _query()
        cal = calibrate_query(matched, population_size=len(population))
        bound = float(np.max(matched)) + 1.0
        assert np.all(cal.log_odds(matched) <= cal.upper_bound_log_odds(bound))
        assert cal.upper_bound_log_odds(np.max(matched)) == pytest.approx(
            np.max(cal.log_odds(matched))
        )

    def test_upper_bound_with_document_prior(self):
        matched, population = _query()
        cal = calibrate_query(matched, population_size=len(population))
        with_prior = cal.upper_bound_log_odds(5.0, document_prior_upper_bound=0.9)
        assert with_prior == pytest.approx(
            cal.upper_bound_log_odds(5.0) + float(logit(0.9))
        )
        with pytest.raises(ValueError, match="must be in"):
            cal.upper_bound_log_odds(5.0, document_prior_upper_bound=1.0)

    def test_scalar_inputs_return_floats(self):
        matched, population = _query()
        cal = calibrate_query(matched, population_size=len(population))
        assert isinstance(cal.log_odds(3.0), float)
        assert isinstance(cal.evidence(3.0), float)
        assert isinstance(cal.probability(3.0), float)

    def test_calibration_is_immutable(self):
        matched, population = _query()
        cal = calibrate_query(matched, population_size=len(population))
        assert isinstance(cal, QueryCalibration)
        with pytest.raises(AttributeError):
            cal.slope = 2.0

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"kappa": 0.0}, "kappa must be positive"),
            ({"expected_relevant": 0.0}, "strictly between"),
            ({"expected_relevant": 1e9}, "strictly between"),
            ({"population_size": 10}, "smaller than the number of scored"),
            ({"intercept": "median"}, "intercept must be one of"),
            ({"scale": -1.0}, "must be positive"),
        ],
    )
    def test_invalid_arguments(self, kwargs, message):
        matched, _ = _query()
        with pytest.raises(ValueError, match=message):
            calibrate_query(matched, **kwargs)

    def test_rejects_empty_scores(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calibrate_query(np.array([]))
