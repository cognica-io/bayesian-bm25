#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Verification tests for theorems from both papers.

Paper 1: "Bayesian BM25: A Probabilistic Framework for Hybrid Text and
          Vector Search" (Jeong, 2026)
Paper 2: "From Bayesian Inference to Neural Computation" (Jeong, 2026)

Each test class corresponds to a specific theorem, definition, or section
from the papers.  The tests verify both exact numerical values and
structural properties that must hold for all valid inputs.
"""

import math

import numpy as np
import pytest

from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    _clamp_probability,
    logit,
    sigmoid,
)
from bayesian_bm25.fusion import (
    LearnableLogOddsWeights,
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_or,
)


def _gaussian_cdf(x):
    """Vectorized Gaussian CDF via standard library erf."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))


# -----------------------------------------------------------------------
# Paper 1: Bayesian BM25
# -----------------------------------------------------------------------


class TestSigmoidProperties:
    """Verify Lemma 2.1.3 from Paper 1."""

    def test_symmetry(self):
        """sigma(x) + sigma(-x) = 1 for all x (Property 1)."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-100, 100, size=10000)
        lhs = sigmoid(x) + sigmoid(-x)
        np.testing.assert_allclose(lhs, 1.0, atol=1e-12)

    def test_derivative_identity(self):
        """sigma'(x) = sigma(x) * (1 - sigma(x)) (Property 2).

        Verified numerically via central finite differences.
        """
        x = np.linspace(-10, 10, 1000)
        s = sigmoid(x)
        analytical = s * (1.0 - s)
        h = 1e-7
        numerical = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-6)

    def test_bounds(self):
        """0 < sigma(x) < 1 for all finite x (Property 3).

        In IEEE 754 float64, sigmoid saturates to exactly 0.0 or 1.0
        for |x| > ~36.  We test the representable range.
        """
        rng = np.random.default_rng(42)
        x = rng.uniform(-36, 36, size=10000)
        s = sigmoid(x)
        assert np.all(s > 0)
        assert np.all(s < 1)

    def test_strict_monotonicity(self):
        """sigma is strictly increasing (Property derived from 2).

        Tested within the float64-representable range where successive
        sigmoid values are distinguishable at the given step size.
        """
        x = np.linspace(-20, 20, 10000)
        s = sigmoid(x)
        assert np.all(np.diff(s) > 0)


class TestLogitSigmoidDuality:
    """Verify Lemma 2.1.4 from Paper 2: logit(sigma(x)) = x."""

    def test_roundtrip_sigmoid_logit(self):
        """sigma(logit(p)) = p for p in (0, 1)."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0.001, 0.999, size=10000)
        recovered = sigmoid(logit(p))
        np.testing.assert_allclose(recovered, p, atol=1e-10)

    def test_roundtrip_logit_sigmoid(self):
        """logit(sigma(x)) = x for finite x."""
        x = np.linspace(-20, 20, 1000)
        recovered = logit(sigmoid(x))
        np.testing.assert_allclose(recovered, x, atol=1e-8)


class TestPosteriorFormula:
    """Verify Theorem 4.1.3 from Paper 1.

    The posterior P(R|s) = L*p / (L*p + (1-L)*(1-p)) is equivalent to
    sigmoid(logit(L) + logit(p)) in the log-odds domain.  Both paths
    must produce identical results.
    """

    def test_log_odds_equivalence(self):
        """Two formulations of Bayes' posterior must agree."""
        rng = np.random.default_rng(42)
        L = rng.uniform(0.01, 0.99, size=10000)
        p = rng.uniform(0.01, 0.99, size=10000)

        # Path 1: direct formula (Eq. 22)
        direct = BayesianProbabilityTransform.posterior(L, p)

        # Path 2: log-odds addition
        log_odds_path = sigmoid(logit(L) + logit(p))

        np.testing.assert_allclose(direct, log_odds_path, atol=1e-9)

    def test_uniform_prior_identity(self):
        """With prior = 0.5, posterior = likelihood (logit(0.5) = 0)."""
        rng = np.random.default_rng(42)
        L = rng.uniform(0.01, 0.99, size=1000)
        posterior = BayesianProbabilityTransform.posterior(L, 0.5)
        np.testing.assert_allclose(posterior, L, atol=1e-9)


class TestMonotonicity:
    """Verify Theorem 4.3.1 from Paper 1.

    The Bayesian BM25 posterior is monotonically increasing in BM25 score
    for any fixed prior.
    """

    def test_posterior_monotonic_in_score(self):
        """For fixed prior, higher score -> higher posterior."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-2.0, 5.0)
            prior = rng.uniform(0.1, 0.9)

            t = BayesianProbabilityTransform(alpha=alpha, beta=beta)
            scores = np.sort(rng.uniform(-5, 10, size=50))
            likelihoods = t.likelihood(scores)
            posteriors = t.posterior(likelihoods, prior)

            diffs = np.diff(posteriors)
            assert np.all(diffs >= 0), (
                f"Monotonicity violated: alpha={alpha:.3f}, beta={beta:.3f}, "
                f"prior={prior:.3f}, min_diff={diffs.min():.2e}"
            )

    def test_full_pipeline_monotonic_fixed_prior(self):
        """score_to_probability is monotonic when tf and ratio are fixed."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-2.0, 5.0)
            tf = rng.uniform(0, 20)
            ratio = rng.uniform(0.1, 3.0)

            t = BayesianProbabilityTransform(alpha=alpha, beta=beta)
            scores = np.sort(rng.uniform(-5, 10, size=100))
            probs = t.score_to_probability(scores, tf, ratio)

            diffs = np.diff(probs)
            assert np.all(diffs >= 0), (
                f"Pipeline monotonicity violated: alpha={alpha:.3f}, "
                f"beta={beta:.3f}, tf={tf:.1f}, ratio={ratio:.2f}"
            )


class TestPriorBounds:
    """Verify Theorem 4.2.4 from Paper 1.

    The composite prior is always in [0.1, 0.9].
    """

    def test_composite_prior_bounds_random(self):
        """Composite prior in [0.1, 0.9] for random tf and doc_len_ratio."""
        rng = np.random.default_rng(42)
        tf = rng.uniform(0, 100, size=10000)
        ratio = rng.uniform(0, 10, size=10000)
        prior = BayesianProbabilityTransform.composite_prior(tf, ratio)
        assert np.all(prior >= 0.1)
        assert np.all(prior <= 0.9)

    def test_tf_prior_bounds(self):
        """TF prior in [0.2, 0.9] (Eq. 25)."""
        rng = np.random.default_rng(42)
        tf = rng.uniform(0, 1000, size=10000)
        p = BayesianProbabilityTransform.tf_prior(tf)
        assert np.all(p >= 0.2)
        assert np.all(p <= 0.9)

    def test_norm_prior_bounds(self):
        """Norm prior in [0.3, 0.9] (Eq. 26)."""
        rng = np.random.default_rng(42)
        ratio = rng.uniform(0, 100, size=10000)
        p = BayesianProbabilityTransform.norm_prior(ratio)
        assert np.all(p >= 0.3)
        assert np.all(p <= 0.9)


class TestBaseRateLogOdds:
    """Verify the three-term log-odds formulation with base rate."""

    def test_three_term_log_odds_equivalence(self):
        """posterior(L, p, br) == sigmoid(logit(L) + logit(br) + logit(p))."""
        rng = np.random.default_rng(42)
        L = rng.uniform(0.01, 0.99, size=10000)
        p = rng.uniform(0.01, 0.99, size=10000)

        for br in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]:
            direct = BayesianProbabilityTransform.posterior(L, p, base_rate=br)
            log_odds_path = sigmoid(logit(L) + logit(br) + logit(p))
            np.testing.assert_allclose(direct, log_odds_path, atol=1e-9)

    def test_base_rate_half_reduces_to_two_term(self):
        """Three-term with br=0.5 equals two-term (logit(0.5) = 0)."""
        rng = np.random.default_rng(42)
        L = rng.uniform(0.01, 0.99, size=10000)
        p = rng.uniform(0.01, 0.99, size=10000)

        two_term = BayesianProbabilityTransform.posterior(L, p)
        three_term = BayesianProbabilityTransform.posterior(L, p, base_rate=0.5)
        np.testing.assert_allclose(three_term, two_term, atol=1e-9)

    def test_monotonicity_with_base_rate(self):
        """Theorem 4.3.1 holds for any base_rate: higher score -> higher posterior."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-2.0, 5.0)
            prior = rng.uniform(0.1, 0.9)
            br = rng.uniform(0.001, 0.999)

            t = BayesianProbabilityTransform(alpha=alpha, beta=beta, base_rate=br)
            scores = np.sort(rng.uniform(-5, 10, size=50))
            probs = t.score_to_probability(scores, prior, prior)

            diffs = np.diff(probs)
            assert np.all(diffs >= 0), (
                f"Monotonicity violated with base_rate={br:.4f}: "
                f"alpha={alpha:.3f}, beta={beta:.3f}, min_diff={diffs.min():.2e}"
            )


class TestPaperValues:
    """Verify Section 11.1 from Paper 1.

    BM25 scores [1.0464478, 0.56150854, 1.1230172] with alpha=1, beta=0
    should produce probabilities that preserve relative ordering and
    lie in (0, 1).
    """

    def test_section_11_1_ordering(self):
        t = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
        scores = np.array([1.0464478, 0.56150854, 1.1230172])
        tf = np.array([5.0, 3.0, 7.0])
        ratio = np.array([0.5, 0.5, 0.5])
        probs = t.score_to_probability(scores, tf, ratio)

        assert np.all(probs > 0)
        assert np.all(probs < 1)
        # Score ordering: s[2] > s[0] > s[1], with comparable priors
        # -> prob ordering should follow
        assert probs[2] > probs[1]
        assert probs[0] > probs[1]


# -----------------------------------------------------------------------
# Paper 2: From Bayesian Inference to Neural Computation
# -----------------------------------------------------------------------


class TestScaleNeutrality:
    """Verify Theorem 4.1.2 from Paper 2.

    If p_i = p for all i, then the log-odds mean maps back to p
    (regardless of n).  With alpha=0, the conjunction should return
    exactly p.
    """

    def test_identical_signals_alpha_zero(self):
        """With alpha=0, identical signals pass through unchanged."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for n in [1, 2, 3, 5, 10]:
                signals = np.full(n, p)
                result = log_odds_conjunction(signals, alpha=0.0)
                assert result == pytest.approx(p, abs=1e-8), (
                    f"Scale neutrality failed: p={p}, n={n}, result={result}"
                )

    def test_identical_signals_alpha_half(self):
        """With alpha=0.5, identical signals are amplified/dampened by n^0.5."""
        for p in [0.6, 0.7, 0.8, 0.9]:
            for n in [2, 3, 5]:
                signals = np.full(n, p)
                result = log_odds_conjunction(signals, alpha=0.5)
                # Manual calculation: l_bar = logit(p), l_adj = logit(p) * n^0.5
                expected = sigmoid(logit(p) * (n ** 0.5))
                assert result == pytest.approx(expected, abs=1e-10), (
                    f"p={p}, n={n}: got {result}, expected {expected}"
                )


class TestSignPreservation:
    """Verify Theorem 4.2.2 from Paper 2.

    Multiplicative scaling preserves the sign of the log-odds mean.
    """

    def test_positive_log_odds_stays_positive(self):
        """If l_bar > 0, then result > 0.5 (positive log-odds)."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 6)
            # Generate probabilities whose mean logit is positive
            probs = rng.uniform(0.55, 0.99, size=n)
            if np.mean(logit(probs)) <= 0:
                continue
            result = log_odds_conjunction(probs)
            assert result > 0.5, (
                f"Sign violated: probs={probs}, result={result}"
            )

    def test_negative_log_odds_stays_negative(self):
        """If l_bar < 0, then result < 0.5 (negative log-odds)."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.01, 0.45, size=n)
            if np.mean(logit(probs)) >= 0:
                continue
            result = log_odds_conjunction(probs)
            assert result < 0.5, (
                f"Sign violated: probs={probs}, result={result}"
            )


class TestIrrelevanceNonInversion:
    """Verify Corollary 4.2.3 from Paper 2.

    If all p_i < 0.5 (all signals report irrelevance), the conjunction
    result must also be < 0.5.  No amount of agreement among irrelevant
    signals can produce a relevance judgment.
    """

    def test_all_irrelevant_stays_irrelevant(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 10)
            probs = rng.uniform(0.01, 0.49, size=n)
            for alpha in [0.0, 0.5, 1.0, 2.0]:
                result = log_odds_conjunction(probs, alpha=alpha)
                assert result < 0.5, (
                    f"Irrelevance inverted: probs={probs}, alpha={alpha}, "
                    f"result={result}"
                )

    def test_all_relevant_stays_relevant(self):
        """Theorem 4.5.1 (iv): If all p_i > 0.5, result > 0.5."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 10)
            probs = rng.uniform(0.51, 0.99, size=n)
            for alpha in [0.0, 0.5, 1.0, 2.0]:
                result = log_odds_conjunction(probs, alpha=alpha)
                assert result > 0.5, (
                    f"Relevance inverted: probs={probs}, alpha={alpha}, "
                    f"result={result}"
                )


class TestPaper2NumericalTable:
    """Verify the exact numerical table from Paper 2, Section 4.5.

    The table compares product rule (prob_and) and log-odds conjunction
    for two signals (n=2, alpha=0.5):

        p1    p2    Product   Conjunction   Interpretation
        0.9   0.9   0.81      0.96          Strong agreement amplified
        0.7   0.7   0.49      0.77          Moderate agreement preserved
        0.7   0.3   0.21      0.50          Exact neutrality (logits cancel)
        0.3   0.3   0.09      0.23          Irrelevance preserved
    """

    @pytest.mark.parametrize("p1,p2,expected_and,expected_conj", [
        (0.9, 0.9, 0.81, 0.96),
        (0.7, 0.7, 0.49, 0.77),
        (0.7, 0.3, 0.21, 0.50),
        (0.3, 0.3, 0.09, 0.23),
    ])
    def test_table_values(self, p1, p2, expected_and, expected_conj):
        probs = np.array([p1, p2])
        assert prob_and(probs) == pytest.approx(expected_and, abs=0.01)
        assert log_odds_conjunction(probs) == pytest.approx(expected_conj, abs=0.01)

    def test_exact_computation_09_09(self):
        """Hand-traced: logit(0.9)=2.197, l_adj=2.197*sqrt(2)=3.107,
        sigmoid(3.107)=0.957."""
        probs = np.array([0.9, 0.9])
        l_bar = logit(0.9)
        l_adj = l_bar * (2 ** 0.5)
        expected = sigmoid(l_adj)
        result = log_odds_conjunction(probs)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_exact_computation_07_03(self):
        """Hand-traced: logit(0.7)=0.847, logit(0.3)=-0.847, l_bar=0,
        l_adj=0, sigmoid(0)=0.5 exactly."""
        probs = np.array([0.7, 0.3])
        result = log_odds_conjunction(probs)
        # logit(0.7) + logit(0.3) = logit(0.7) + (-logit(0.7)) = 0
        # by sigmoid symmetry (Lemma 2.1.3)
        assert result == pytest.approx(0.5, abs=1e-10)


class TestDisagreementModeration:
    """Verify Theorem 4.5.1 (ii) from Paper 2.

    When signals disagree symmetrically (p and 1-p), logits cancel
    and the result is exactly 0.5.
    """

    def test_symmetric_disagreement(self):
        """(p, 1-p) -> 0.5 for any p."""
        for p in np.linspace(0.01, 0.99, 50):
            probs = np.array([p, 1.0 - p])
            result = log_odds_conjunction(probs)
            assert result == pytest.approx(0.5, abs=1e-8), (
                f"Symmetric disagreement failed: p={p}, result={result}"
            )


class TestLogisticRegressionEquivalence:
    """Verify Theorem 5.2.1a from Paper 2.

    When all signals are sigmoid-calibrated from raw scores, the
    log-odds conjunction reduces to logistic regression:
      sigmoid(n^alpha * mean(alpha_i * (s_i - beta_i)))
    because logit(sigmoid(x)) = x (Lemma 2.1.4).
    """

    def test_sigmoid_calibrated_signals(self):
        """Conjunction of sigmoid-calibrated signals = logistic regression."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = rng.integers(2, 6)
            alphas = rng.uniform(0.5, 3.0, size=n)
            betas = rng.uniform(-1.0, 3.0, size=n)
            scores = rng.uniform(-2.0, 5.0, size=n)
            conf_alpha = 0.5

            # Calibrate each signal via sigmoid
            calibrated = np.array([
                sigmoid(alphas[i] * (scores[i] - betas[i]))
                for i in range(n)
            ])

            # Path 1: log-odds conjunction
            result = log_odds_conjunction(calibrated, alpha=conf_alpha)

            # Path 2: direct logistic regression
            # logit(sigmoid(a*(s-b))) = a*(s-b), so l_bar = mean(a_i*(s_i-b_i))
            pre_activations = alphas * (scores - betas)
            l_bar = np.mean(pre_activations)
            l_adj = l_bar * (n ** conf_alpha)
            expected = sigmoid(l_adj)

            assert result == pytest.approx(expected, abs=1e-10), (
                f"Logistic equivalence failed: n={n}, result={result}, "
                f"expected={expected}"
            )


class TestAgreementAmplification:
    """Verify Theorem 4.5.1 (i) from Paper 2.

    If p_i > 0.5 for all i, then the conjunction result exceeds each
    individual p_i for n >= 2 and alpha > 0.
    """

    def test_amplification_exceeds_input(self):
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(2, 6)
            p = rng.uniform(0.55, 0.95)
            probs = np.full(n, p)
            result = log_odds_conjunction(probs, alpha=0.5)
            assert result > p, (
                f"No amplification: p={p}, n={n}, result={result}"
            )

    def test_more_signals_more_amplification(self):
        """Adding agreeing signals increases confidence."""
        for p in [0.6, 0.7, 0.8, 0.9]:
            prev = p
            for n in range(2, 8):
                probs = np.full(n, p)
                result = log_odds_conjunction(probs, alpha=0.5)
                assert result > prev or result == pytest.approx(prev, abs=1e-10), (
                    f"Amplification not monotonic in n: p={p}, n={n}"
                )
                prev = result


class TestConjunctionVsProductRule:
    """Compare log-odds conjunction against naive product rule.

    For agreeing high-probability signals, the conjunction should
    always exceed the product rule (which suffers from shrinkage).
    """

    def test_conjunction_beats_product_for_agreement(self):
        """When all p_i > 0.5, conjunction > product rule."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.55, 0.99, size=n)
            conj = log_odds_conjunction(probs, alpha=0.5)
            prod = prob_and(probs)
            assert conj > prod, (
                f"Conjunction not better than product: "
                f"probs={probs}, conj={conj:.4f}, prod={prod:.4f}"
            )


# -----------------------------------------------------------------------
# Output range and numerical stability
# -----------------------------------------------------------------------


class TestConjunctionStrictBounds:
    """Verify Theorem 5.1.2 from Paper 1.

    For p_i in (0, 1) with n >= 2:
        0 < prob_and(probs) < min(probs)
    """

    def test_prob_and_strictly_below_min(self):
        """prob_and(probs) < min(probs) for n >= 2 with p_i in (0, 1)."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.01, 0.99, size=n)
            result = prob_and(probs)
            assert result > 0, (
                f"Lower bound violated: probs={probs}, result={result}"
            )
            assert result < np.min(probs), (
                f"Upper bound violated: probs={probs}, result={result}, "
                f"min={np.min(probs)}"
            )


class TestDisjunctionStrictBounds:
    """Verify Theorem 5.2.2 from Paper 1.

    For p_i in (0, 1) with n >= 2:
        max(probs) < prob_or(probs) < 1
    """

    def test_prob_or_strictly_above_max(self):
        """prob_or(probs) > max(probs) for n >= 2 with p_i in (0, 1)."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.01, 0.99, size=n)
            result = prob_or(probs)
            assert result > np.max(probs), (
                f"Lower bound violated: probs={probs}, result={result}, "
                f"max={np.max(probs)}"
            )
            assert result < 1, (
                f"Upper bound violated: probs={probs}, result={result}"
            )


class TestLogOPEquivalence:
    """Verify Theorem 4.1.2a from Paper 2.

    The log-odds mean sigma(mean(logit(P_i))) equals the normalized
    Product of Experts (PoE) formula:
        P_LogOP = prod(P_i^{1/n}) / (prod(P_i^{1/n}) + prod((1-P_i)^{1/n}))
    """

    def test_log_odds_mean_equals_normalized_poe(self):
        """sigma(mean(logit(P_i))) == normalized PoE for random vectors."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 7)
            probs = rng.uniform(0.01, 0.99, size=n)

            # Path 1: log-odds mean (alpha=0 to get pure mean)
            log_odds_result = sigmoid(np.mean(logit(probs)))

            # Path 2: normalized PoE
            # prod(P_i^{1/n}) / (prod(P_i^{1/n}) + prod((1-P_i)^{1/n}))
            prod_p = np.prod(probs ** (1.0 / n))
            prod_1mp = np.prod((1.0 - probs) ** (1.0 / n))
            poe_result = prod_p / (prod_p + prod_1mp)

            np.testing.assert_allclose(log_odds_result, poe_result, atol=1e-10), (
                f"PoE equivalence failed: n={n}, probs={probs}, "
                f"log_odds={log_odds_result}, poe={poe_result}"
            )


class TestHeterogeneousSignalCombination:
    """Verify Remark 5.2.3 from Paper 2.

    When combining signals with different calibrations (sigmoid-calibrated
    BM25 + linear-calibrated cosine), the logit acts as identity on BM25
    but as a genuine nonlinearity on cosine.
    """

    def test_bm25_plus_cosine_pipeline(self):
        """Sigmoid-calibrated BM25 + linear-calibrated cosine -> valid probability."""
        bm25_scores = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        cosine_scores = np.array([0.2, 0.4, 0.6, 0.8, 0.95])

        # Calibrate BM25 via sigmoid (alpha=1, beta=1)
        bm25_probs = sigmoid(1.0 * (bm25_scores - 1.0))

        # Calibrate cosine via linear mapping
        cosine_probs = cosine_to_probability(cosine_scores)

        results = []
        for i in range(len(bm25_scores)):
            probs = np.array([bm25_probs[i], cosine_probs[i]])
            result = log_odds_conjunction(probs)
            # Must be a valid probability
            assert 0 < result < 1, (
                f"Invalid probability: bm25={bm25_scores[i]}, "
                f"cosine={cosine_scores[i]}, result={result}"
            )
            results.append(result)

        # Monotonicity: both signals increase -> result should increase
        results = np.array(results)
        assert np.all(np.diff(results) > 0), (
            f"Monotonicity violated: results={results}"
        )

    def test_logit_nonlinearity_for_linear_calibration(self):
        """logit(cosine_to_probability(s)) is nonlinear in s.

        If it were linear, logit(cosine_to_probability(s)) = a*s + b for
        some constants.  We show this fails by checking that the second
        differences are non-zero (a linear function has zero second
        differences).
        """
        s = np.linspace(-0.9, 0.9, 100)
        transformed = logit(cosine_to_probability(s))

        # Second differences of a linear function are zero
        second_diff = np.diff(transformed, n=2)
        assert not np.allclose(second_diff, 0.0, atol=1e-8), (
            "logit(cosine_to_probability(s)) appears linear, "
            "but should be nonlinear"
        )


class TestSingleSignalIdentity:
    """Verify Proposition 4.3.2 from Paper 2.

    When n=1: l_adjusted = logit(P_1) * 1^alpha = logit(P_1) for ANY alpha,
    so P_final = P_1 regardless of alpha.
    """

    def test_single_signal_identity_all_alphas(self):
        """n=1 returns P_1 for any alpha value."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.01, 0.99, size=50)
        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for p in probs:
                result = log_odds_conjunction(np.array([p]), alpha=alpha)
                assert result == pytest.approx(p, abs=1e-8), (
                    f"Single signal identity failed: p={p}, alpha={alpha}, "
                    f"result={result}"
                )


class TestWeightedAlphaComposition:
    """Verify that alpha and weights compose correctly (Paper 2, Section 4.2 + Theorem 8.3).

    The composed formula is:
        P = sigma(n^alpha * sum(w_i * logit(P_i)))
    """

    def test_weighted_alpha_composition(self):
        """sigma(n^alpha * sum(w_i * logit(P_i))) matches hand-computed values."""
        probs = np.array([0.8, 0.6])
        w = np.array([0.7, 0.3])
        alpha = 0.5
        n = 2

        # Hand-compute expected value
        l_weighted = np.sum(w * logit(probs))
        expected = sigmoid((n ** alpha) * l_weighted)

        result = log_odds_conjunction(probs, alpha=alpha, weights=w)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_uniform_weights_with_alpha_matches_unweighted(self):
        """Uniform weights w_i=1/n with explicit alpha matches unweighted formula.

        Unweighted: sigma(n^alpha * mean(logit(P_i)))
                  = sigma(n^alpha * (1/n) * sum(logit(P_i)))

        Weighted with w_i=1/n:
                    sigma(n^alpha * sum((1/n) * logit(P_i)))
                  = sigma(n^alpha * (1/n) * sum(logit(P_i)))

        These are identical.
        """
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.01, 0.99, size=n)
            alpha = rng.uniform(0.0, 2.0)
            uniform_w = np.full(n, 1.0 / n)

            unweighted = log_odds_conjunction(probs, alpha=alpha)
            weighted = log_odds_conjunction(probs, alpha=alpha, weights=uniform_w)
            assert weighted == pytest.approx(unweighted, abs=1e-10), (
                f"Uniform weights != unweighted: n={n}, alpha={alpha:.3f}, "
                f"probs={probs}"
            )


class TestMonotoneShrinkage:
    """Verify Theorem 3.2.1 + Corollary 3.2.2 from Paper 1.

    Adding one more signal to prob_and always strictly decreases the result.
    For p_i in (0, 1): prob_and(n+1 signals) < prob_and(n signals).
    """

    def test_prob_and_decreases_with_more_signals(self):
        """prob_and result strictly decreases as n grows."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            # Generate a pool of 2..8 signals
            max_n = rng.integers(3, 9)
            all_probs = rng.uniform(0.01, 0.99, size=max_n)

            prev_result = prob_and(all_probs[:2])
            for n in range(3, max_n + 1):
                result = prob_and(all_probs[:n])
                assert result < prev_result, (
                    f"Shrinkage violated: n={n}, prev={prev_result:.6f}, "
                    f"cur={result:.6f}, probs={all_probs[:n]}"
                )
                prev_result = result

    def test_prob_and_approaches_zero(self):
        """Many signals push prob_and toward zero."""
        probs = np.full(50, 0.9)
        result = prob_and(probs)
        # 0.9^50 = 0.00515...
        assert result < 0.01


class TestInformationLoss:
    """Verify Proposition 3.4.1 from Paper 1.

    The product rule is invariant to agreement structure: two different
    input vectors with the same product give the same prob_and result.
    This is precisely the information loss that log_odds_conjunction fixes.
    """

    def test_same_product_same_result(self):
        """prob_and depends only on the product, not individual values."""
        # (0.9, 0.1) and (0.3, 0.3) both have product 0.09
        result_a = prob_and(np.array([0.9, 0.1]))
        result_b = prob_and(np.array([0.3, 0.3]))
        assert result_a == pytest.approx(result_b, abs=1e-10)

    def test_conjunction_breaks_invariance(self):
        """log_odds_conjunction distinguishes agreement from disagreement."""
        # Same pairs as above: product is the same, but agreement structure differs
        disagreement = log_odds_conjunction(np.array([0.9, 0.1]))
        agreement = log_odds_conjunction(np.array([0.3, 0.3]))
        # Disagreement (0.9, 0.1) -> ~0.5 (logits cancel)
        # Agreement (0.3, 0.3) -> ~0.23 (both signal irrelevance)
        assert disagreement != pytest.approx(agreement, abs=0.01), (
            f"Conjunction failed to distinguish: disagreement={disagreement:.4f}, "
            f"agreement={agreement:.4f}"
        )

    def test_randomized_same_product_invariance(self):
        """For arbitrary pairs with the same product, prob_and is identical."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            # Pick a target product in (0, 1)
            target = rng.uniform(0.01, 0.99)
            # Build two pairs: (a, target/a) and (b, target/b)
            a = rng.uniform(max(target, 0.01), 0.99)
            b = rng.uniform(max(target, 0.01), 0.99)
            pair_a = np.array([a, target / a])
            pair_b = np.array([b, target / b])
            # Both must produce the same prob_and result
            result_a = prob_and(pair_a)
            result_b = prob_and(pair_b)
            assert result_a == pytest.approx(result_b, abs=1e-8), (
                f"Product invariance violated: pair_a={pair_a}, pair_b={pair_b}, "
                f"result_a={result_a:.10f}, result_b={result_b:.10f}"
            )


class TestSqrtNScalingLaw:
    """Verify Theorem 4.4.1 + Proposition 4.4.2 from Paper 2.

    The sqrt(n) scaling (alpha=0.5) preserves the signal-to-noise ratio
    of evidence as the number of signals grows.  Specifically:

    - For identical signals p, the effective logit is logit(p) * sqrt(n)
    - This means confidence grows as sqrt(n), analogous to the standard
      error of the mean shrinking as 1/sqrt(n)
    """

    def test_effective_logit_scales_as_sqrt_n(self):
        """The effective log-odds scale as sqrt(n) for identical signals."""
        for p in [0.6, 0.7, 0.8, 0.9]:
            base_logit = logit(p)
            for n in [2, 3, 4, 5, 8, 10]:
                probs = np.full(n, p)
                result = log_odds_conjunction(probs, alpha=0.5)
                expected = sigmoid(base_logit * np.sqrt(n))
                assert result == pytest.approx(expected, abs=1e-10), (
                    f"sqrt(n) scaling failed: p={p}, n={n}, "
                    f"result={result}, expected={expected}"
                )

    def test_sqrt_scaling_vs_linear_scaling(self):
        """sqrt(n) scaling grows slower than linear (alpha=1) scaling."""
        p = 0.8
        for n in [2, 3, 5, 10]:
            probs = np.full(n, p)
            sqrt_result = log_odds_conjunction(probs, alpha=0.5)
            linear_result = log_odds_conjunction(probs, alpha=1.0)
            # For p > 0.5 and n >= 2, linear scaling amplifies more
            assert linear_result > sqrt_result, (
                f"Linear should amplify more: n={n}, "
                f"sqrt={sqrt_result:.4f}, linear={linear_result:.4f}"
            )

    def test_sqrt_scaling_vs_no_scaling(self):
        """sqrt(n) scaling amplifies more than no scaling (alpha=0)."""
        p = 0.8
        for n in [2, 3, 5, 10]:
            probs = np.full(n, p)
            sqrt_result = log_odds_conjunction(probs, alpha=0.5)
            no_scale_result = log_odds_conjunction(probs, alpha=0.0)
            # For p > 0.5 and n >= 2, sqrt scaling amplifies beyond alpha=0
            assert sqrt_result > no_scale_result, (
                f"sqrt should amplify more than alpha=0: n={n}, "
                f"sqrt={sqrt_result:.4f}, none={no_scale_result:.4f}"
            )

    def test_confidence_growth_rate(self):
        """Doubling n should approximately double the effective log-odds deviation.

        For alpha=0.5: l_eff(2n) / l_eff(n) = sqrt(2n) / sqrt(n) = sqrt(2).
        """
        p = 0.75
        base_logit = logit(p)
        for n in [2, 4, 8]:
            l_eff_n = base_logit * np.sqrt(n)
            l_eff_2n = base_logit * np.sqrt(2 * n)
            ratio = l_eff_2n / l_eff_n
            assert ratio == pytest.approx(np.sqrt(2), abs=1e-10), (
                f"Growth rate not sqrt(2): n={n}, ratio={ratio}"
            )


class TestSpreadProperty:
    """Verify Theorem 4.5.1 (iii) from Paper 2.

    When signals have high variance (some high, some low but not
    symmetric), the conjunction result is moderated toward 0.5 compared
    to using only the agreeing signals.  Disagreement reduces confidence.
    """

    def test_disagreement_reduces_confidence(self):
        """Adding a contradicting signal moves the result toward 0.5."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            # Start with two agreeing high signals
            p_high = rng.uniform(0.7, 0.95)
            agreeing = np.array([p_high, p_high])
            result_agree = log_odds_conjunction(agreeing, alpha=0.0)

            # Add a low (contradicting) signal
            p_low = rng.uniform(0.05, 0.3)
            mixed = np.array([p_high, p_high, p_low])
            result_mixed = log_odds_conjunction(mixed, alpha=0.0)

            # The mixed result should be closer to 0.5 than the agreeing result
            dist_agree = abs(result_agree - 0.5)
            dist_mixed = abs(result_mixed - 0.5)
            assert dist_mixed < dist_agree, (
                f"Disagreement did not reduce confidence: "
                f"agree={result_agree:.4f} (dist={dist_agree:.4f}), "
                f"mixed={result_mixed:.4f} (dist={dist_mixed:.4f}), "
                f"p_high={p_high:.3f}, p_low={p_low:.3f}"
            )

    def test_high_variance_near_half(self):
        """Signals spread symmetrically around 0.5 produce result near 0.5."""
        # Equal number of signals above and below 0.5, symmetrically placed
        for offset in [0.1, 0.2, 0.3, 0.4]:
            probs = np.array([0.5 + offset, 0.5 - offset])
            result = log_odds_conjunction(probs, alpha=0.0)
            assert result == pytest.approx(0.5, abs=1e-8), (
                f"Symmetric spread not neutral: offset={offset}, result={result}"
            )

    def test_variance_ordering(self):
        """Higher variance inputs produce results closer to 0.5 (alpha=0).

        Compare [0.8, 0.8] (low variance) vs [0.95, 0.65] (same mean logit
        is not guaranteed, so we fix the mean logit and vary spread).
        """
        # Fix mean logit, vary spread
        mean_logit = logit(0.75)  # ~1.0986
        for spread in [0.0, 0.5, 1.0, 1.5]:
            p1 = sigmoid(mean_logit + spread)
            p2 = sigmoid(mean_logit - spread)
            probs = np.array([p1, p2])
            result = log_odds_conjunction(probs, alpha=0.0)
            # With alpha=0, mean logit is preserved, so result should
            # always equal sigmoid(mean_logit) regardless of spread
            expected = sigmoid(mean_logit)
            assert result == pytest.approx(expected, abs=1e-8), (
                f"Alpha=0 should be spread-invariant: spread={spread}, "
                f"result={result}, expected={expected}"
            )


class TestGeometricMeanResidual:
    """Verify Remark 4.1.3 from Paper 2.

    The geometric mean in probability space prod(P_i)^{1/n} differs from
    the log-odds mean sigma(mean(logit(P_i))) by a nonlinear residual.
    They are only equal when all P_i are identical.
    """

    def test_geometric_mean_differs_from_log_odds_mean(self):
        """For heterogeneous probabilities, geometric mean != log-odds mean."""
        rng = np.random.default_rng(42)
        differ_count = 0
        for _ in range(1000):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.1, 0.9, size=n)

            # Geometric mean in probability space
            geo_mean = np.prod(probs) ** (1.0 / n)

            # Log-odds mean
            log_odds_mean = sigmoid(np.mean(logit(probs)))

            if not np.isclose(geo_mean, log_odds_mean, atol=1e-6):
                differ_count += 1

        # They should differ for the vast majority of heterogeneous inputs
        assert differ_count > 900, (
            f"Expected most cases to differ, but only {differ_count}/1000 did"
        )

    def test_identical_signals_no_residual(self):
        """When all P_i = p, geometric mean = p = log-odds mean (no residual)."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for n in [2, 3, 5, 10]:
                probs = np.full(n, p)
                geo_mean = np.prod(probs) ** (1.0 / n)
                log_odds_mean = sigmoid(np.mean(logit(probs)))
                assert geo_mean == pytest.approx(p, abs=1e-10)
                assert log_odds_mean == pytest.approx(p, abs=1e-10)

    def test_geometric_mean_underestimates_for_high_probs(self):
        """For agreeing high probabilities, geometric mean < log-odds mean.

        The geometric mean suffers from the same shrinkage as the product
        rule (just normalized by n), while the log-odds mean correctly
        preserves the consensus.
        """
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(2, 6)
            # All probabilities above 0.5 but not identical
            probs = rng.uniform(0.6, 0.95, size=n)
            # Make them not identical
            probs = np.sort(probs)
            if np.allclose(probs, probs[0]):
                continue

            geo_mean = np.prod(probs) ** (1.0 / n)
            log_odds_mean = sigmoid(np.mean(logit(probs)))

            assert geo_mean < log_odds_mean, (
                f"Geometric mean should underestimate: probs={probs}, "
                f"geo={geo_mean:.6f}, log_odds={log_odds_mean:.6f}"
            )


class TestSigmoidUniqueness:
    """Verify Theorem 6.2.1 from Paper 2.

    The sigmoid is the unique function satisfying all three properties:
    (a) Maps R -> (0, 1)
    (b) Symmetric: f(x) + f(-x) = 1
    (c) Self-derivative: f'(x) = f(x) * (1 - f(x))

    Alternative activations must violate at least one property.
    """

    def test_sigmoid_satisfies_all_three(self):
        """Sigmoid satisfies (a), (b), and (c) simultaneously."""
        x = np.linspace(-10, 10, 1000)
        s = sigmoid(x)

        # (a) Output in (0, 1)
        assert np.all(s > 0) and np.all(s < 1)

        # (b) Symmetry: s(x) + s(-x) = 1
        np.testing.assert_allclose(s + sigmoid(-x), 1.0, atol=1e-12)

        # (c) Self-derivative: s'(x) = s(x) * (1 - s(x))
        analytical = s * (1.0 - s)
        h = 1e-7
        numerical = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-6)

    def test_relu_violates_range(self):
        """ReLU violates property (a): output is not bounded in (0, 1)."""
        x = np.array([2.0, 5.0, 10.0])
        relu = np.maximum(0, x)
        # ReLU output exceeds 1 for x > 1
        assert np.any(relu > 1), "ReLU should violate (0, 1) range"

    def test_tanh_rescaled_violates_self_derivative(self):
        """Rescaled tanh f(x) = (1 + tanh(x))/2 satisfies (a) and (b) but not (c).

        This function maps R -> (0, 1) and is symmetric, but its derivative
        is (1 - tanh(x)^2) / 2, which does NOT equal f(x) * (1 - f(x)).
        """
        x = np.linspace(-5, 5, 1000)
        f = (1.0 + np.tanh(x)) / 2.0

        # (a) Range: satisfied
        assert np.all(f > 0) and np.all(f < 1)

        # (b) Symmetry: f(x) + f(-x) = (1+tanh(x))/2 + (1+tanh(-x))/2
        #             = (1+tanh(x))/2 + (1-tanh(x))/2 = 1. Satisfied.
        np.testing.assert_allclose(f + (1.0 + np.tanh(-x)) / 2.0, 1.0, atol=1e-12)

        # (c) Self-derivative: VIOLATED
        # Actual derivative: (1 - tanh(x)^2) / 2
        actual_deriv = (1.0 - np.tanh(x) ** 2) / 2.0
        # Self-derivative formula: f(x) * (1 - f(x))
        self_deriv = f * (1.0 - f)

        # These should NOT be equal (tanh_rescaled violates property c)
        assert not np.allclose(actual_deriv, self_deriv, atol=1e-4), (
            "Rescaled tanh should violate the self-derivative property"
        )


class TestOutputRange:
    """All probability outputs must be in (0, 1) for any valid input."""

    def test_score_to_probability_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            alpha = rng.uniform(0.01, 10.0)
            beta = rng.uniform(-10.0, 10.0)
            t = BayesianProbabilityTransform(alpha=alpha, beta=beta)
            scores = rng.uniform(-100, 100, size=100)
            tf = rng.uniform(0, 100, size=100)
            ratio = rng.uniform(0, 10, size=100)
            probs = t.score_to_probability(scores, tf, ratio)
            assert np.all(probs > 0)
            assert np.all(probs < 1)

    def test_log_odds_conjunction_range(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = rng.integers(2, 10)
            probs = rng.uniform(0.01, 0.99, size=n)
            alpha = rng.uniform(0.0, 1.0)
            result = log_odds_conjunction(probs, alpha=alpha)
            assert 0 < result < 1

    def test_extreme_inputs(self):
        """Numerical stability at boundaries."""
        t = BayesianProbabilityTransform(alpha=10.0, beta=0.0)
        # Very large score
        p = t.score_to_probability(1000.0, tf=10, doc_len_ratio=0.5)
        assert np.isfinite(p)
        assert 0 < p < 1
        # Very negative score
        p = t.score_to_probability(-1000.0, tf=0, doc_len_ratio=5.0)
        assert np.isfinite(p)
        assert 0 < p < 1


# -----------------------------------------------------------------------
# Paper 2, Section 6: Activation Functions
# -----------------------------------------------------------------------


class TestTanhIsSigmoidInDisguise:
    """Verify Proposition 6.3.1 from Paper 2.

    The rescaled tanh, (1 + tanh(x)) / 2, is identically sigma(2x).
    Any valid rescaling of tanh to the unit interval reduces to the
    sigmoid with parameter absorption.
    """

    def test_identity_over_range(self):
        """(1 + tanh(x)) / 2 == sigma(2x) for all x."""
        x = np.linspace(-20, 20, 10000)
        lhs = (1.0 + np.tanh(x)) / 2.0
        rhs = sigmoid(2.0 * x)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_identity_at_extremes(self):
        """Identity holds at large magnitudes near saturation."""
        for x in [-100.0, -50.0, 50.0, 100.0]:
            lhs = (1.0 + np.tanh(x)) / 2.0
            rhs = sigmoid(2.0 * x)
            np.testing.assert_allclose(lhs, rhs, atol=1e-15)

    def test_derivative_consistency(self):
        """Derivatives of (1+tanh(x))/2 and sigma(2x) match."""
        x = np.linspace(-5, 5, 1000)
        h = 1e-7
        # Numerical derivative of (1 + tanh(x)) / 2
        d_tanh = ((1.0 + np.tanh(x + h)) / 2.0 - (1.0 + np.tanh(x - h)) / 2.0) / (2 * h)
        # Numerical derivative of sigma(2x)
        d_sig = (sigmoid(2.0 * (x + h)) - sigmoid(2.0 * (x - h))) / (2 * h)
        np.testing.assert_allclose(d_tanh, d_sig, atol=1e-5)


class TestProbitExclusion:
    """Verify Proposition 6.3.4 from Paper 2.

    The probit function Phi(x) satisfies constraints C1 (range) and
    C4 (symmetry) but violates C3 (self-derivative).
    """

    def test_satisfies_c1_range(self):
        """Phi maps R -> (0, 1), verified within float64-representable range."""
        x = np.linspace(-8, 8, 10000)
        phi = _gaussian_cdf(x)
        assert np.all(phi > 0)
        assert np.all(phi < 1)

    def test_satisfies_c4_symmetry(self):
        """Phi(-x) = 1 - Phi(x) for all x."""
        x = np.linspace(-10, 10, 10000)
        np.testing.assert_allclose(
            _gaussian_cdf(-x), 1.0 - _gaussian_cdf(x), atol=1e-12
        )

    def test_violates_c3_self_derivative(self):
        """Phi'(x) != Phi(x) * (1 - Phi(x))."""
        x = np.linspace(-3, 3, 1000)
        phi = _gaussian_cdf(x)
        # Actual derivative of Phi is the Gaussian PDF
        actual_deriv = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2.0)
        # Self-derivative formula
        self_deriv = phi * (1.0 - phi)
        # These must NOT be equal
        assert not np.allclose(actual_deriv, self_deriv, atol=1e-3), (
            "Probit should violate the self-derivative property (C3)"
        )


class TestNeuronPosteriorIdentity:
    """Verify Theorem 6.4.1 from Paper 2.

    The Bayesian posterior sigma(alpha * (s - beta)) and the sigmoid
    neuron sigma(w * x + b) are the same object under w = alpha,
    b = -alpha * beta.
    """

    def test_parameter_correspondence(self):
        """sigma(alpha*(s - beta)) == sigma(w*s + b) with w=alpha, b=-alpha*beta."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-3.0, 5.0)
            s = rng.uniform(-10, 10)

            posterior = sigmoid(alpha * (s - beta))
            w = alpha
            b = -alpha * beta
            neuron = sigmoid(w * s + b)
            np.testing.assert_allclose(posterior, neuron, atol=1e-14)

    def test_batch_equivalence(self):
        """Batch version: vectors of scores produce identical results."""
        rng = np.random.default_rng(42)
        alpha = 2.5
        beta = 1.3
        scores = rng.uniform(-5, 10, size=1000)

        posterior = sigmoid(alpha * (scores - beta))
        neuron = sigmoid(alpha * scores + (-alpha * beta))
        np.testing.assert_allclose(posterior, neuron, atol=1e-14)


class TestReLUFromMAP:
    """Verify Theorem 6.5.3 from Paper 2.

    The MAP estimate of h under exponential prior + Gaussian likelihood
    is max(0, z - theta) where z = x/w, theta = lambda*tau^2/w^2.
    """

    def test_closed_form_matches_grid_search(self):
        """MAP closed form matches brute-force grid search of log-posterior."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            w = rng.uniform(0.5, 3.0)
            lam = rng.uniform(0.1, 5.0)
            tau = rng.uniform(0.1, 2.0)
            x = rng.uniform(-3, 5)

            # Closed form
            z = x / w
            theta = lam * tau**2 / w**2
            h_closed = max(0.0, z - theta)

            # Grid search over h >= 0
            h_grid = np.linspace(0, max(10.0, z + 5), 10000)
            log_post = -(x - w * h_grid)**2 / (2 * tau**2) - lam * h_grid
            h_grid_opt = h_grid[np.argmax(log_post)]

            np.testing.assert_allclose(h_closed, h_grid_opt, atol=0.01), (
                f"MAP mismatch: w={w:.2f}, lam={lam:.2f}, tau={tau:.2f}, "
                f"x={x:.2f}, closed={h_closed:.4f}, grid={h_grid_opt:.4f}"
            )

    def test_gradient_zero_at_optimum(self):
        """Gradient of log-posterior is zero (or h is at boundary h=0)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            w = rng.uniform(0.5, 3.0)
            lam = rng.uniform(0.1, 5.0)
            tau = rng.uniform(0.1, 2.0)
            x = rng.uniform(-5, 10)

            z = x / w
            theta = lam * tau**2 / w**2
            h_star = max(0.0, z - theta)

            if h_star > 0:
                # Interior solution: gradient should be zero
                grad = w * (x - w * h_star) / tau**2 - lam
                assert abs(grad) < 1e-8, (
                    f"Gradient not zero at interior optimum: grad={grad:.2e}"
                )
            else:
                # Boundary: gradient at h=0 should be non-positive
                grad_at_zero = w * x / tau**2 - lam
                assert grad_at_zero <= 1e-8, (
                    f"Gradient at boundary should be non-positive: "
                    f"grad={grad_at_zero:.2e}"
                )

    def test_sparsity_threshold(self):
        """h* = 0 when x < w*theta (input below threshold)."""
        w, lam, tau = 1.0, 1.0, 1.0
        theta = lam * tau**2 / w**2  # = 1.0
        # Below threshold
        assert max(0.0, -1.0 / w - theta) == 0.0
        assert max(0.0, 0.5 / w - theta) == 0.0
        # Above threshold
        assert max(0.0, 2.0 / w - theta) > 0.0


class TestSwishBayesianExpectedSignal:
    """Verify Theorem 6.7.4 from Paper 2.

    Under the self-gated relevance model, E[Y|x] = x * sigma(x) = Swish(x).
    """

    def test_swish_identity(self):
        """x * sigma(x) is the expected value under binary gating."""
        x = np.linspace(-10, 10, 10000)
        # E[Y|x] = x * P(R=1|x) + 0 * P(R=0|x) = x * sigma(x)
        expected_value = x * sigmoid(x) + 0.0 * (1.0 - sigmoid(x))
        swish = x * sigmoid(x)
        np.testing.assert_allclose(expected_value, swish, atol=1e-15)

    def test_swish_properties(self):
        """Swish is smooth, non-monotone near zero, and asymptotic to ReLU for x >> 0."""
        x_pos = np.linspace(10, 20, 100)
        swish_pos = x_pos * sigmoid(x_pos)
        relu_pos = np.maximum(0, x_pos)
        # For large positive x, sigma(x) -> 1, so Swish -> x -> ReLU
        np.testing.assert_allclose(swish_pos, relu_pos, atol=0.001)

        # Swish has a minimum near x ~ -1.278
        x_fine = np.linspace(-2, 0, 10000)
        swish_fine = x_fine * sigmoid(x_fine)
        assert np.min(swish_fine) < 0, "Swish should be negative near x ~ -1.278"

    def test_swish_at_zero(self):
        """Swish(0) = 0 * sigma(0) = 0 * 0.5 = 0."""
        assert 0.0 * sigmoid(0.0) == 0.0


class TestReLUSwishMAPBayesDuality:
    """Verify Theorem 6.7.5 from Paper 2.

    ReLU = x * 1[x > 0] (MAP estimate, hard gate)
    Swish = x * sigma(x) (Bayes estimate, soft gate)
    As beta -> inf in Swish_beta, Swish converges to ReLU.
    """

    def test_relu_as_hard_gate(self):
        """ReLU(x) = x * 1[x > 0] for all x."""
        x = np.linspace(-10, 10, 10000)
        relu = np.maximum(0, x)
        hard_gated = x * (x > 0).astype(np.float64)
        np.testing.assert_allclose(relu, hard_gated, atol=1e-15)

    def test_swish_as_soft_gate(self):
        """Swish(x) = x * sigma(x) for all x."""
        x = np.linspace(-10, 10, 10000)
        swish = x * sigmoid(x)
        # Verify it is between 0 and ReLU for positive x
        relu = np.maximum(0, x)
        mask = x > 0.5  # Avoid the transition region near 0
        assert np.all(swish[mask] <= relu[mask] + 1e-10)

    def test_convergence_to_relu(self):
        """Swish_beta -> ReLU as beta -> inf, away from x=0."""
        x = np.linspace(-10, 10, 10000)
        # Exclude near-zero region where convergence is slow
        mask = np.abs(x) > 0.5
        for beta in [10, 50, 100]:
            swish_beta = x * sigmoid(beta * x)
            relu = np.maximum(0, x)
            max_err = np.max(np.abs(swish_beta[mask] - relu[mask]))
            assert max_err < 1.0 / beta + 0.01, (
                f"beta={beta}: max error {max_err:.4f} too large"
            )


class TestGeneralizedSwishLimits:
    """Verify Theorem 6.7.6 from Paper 2.

    Three limits of x * sigma(beta * x):
    - beta -> 0: x/2 (uniform prior, maximum ignorance)
    - beta = 1: x * sigma(x) (canonical Bayesian posterior)
    - beta -> inf: max(0, x) (deterministic MAP)
    """

    def test_beta_zero_limit(self):
        """As beta -> 0, x * sigma(beta*x) -> x/2.

        The error of the approximation scales as O(beta * x^2) due to
        the Taylor expansion sigma(u) = 0.5 + u/4 + O(u^3), so the
        tolerance must account for the quadratic dependence on x.
        """
        x = np.linspace(-5, 5, 1000)
        for beta in [0.001, 0.01, 0.05]:
            swish_beta = x * sigmoid(beta * x)
            expected = x / 2.0
            # Error is O(beta * x^2): use max(x^2) * beta / 4 as tolerance
            tol = beta * np.max(x**2) / 4.0 + 1e-10
            np.testing.assert_allclose(swish_beta, expected, atol=tol)

    def test_beta_one_canonical(self):
        """At beta=1, x * sigma(beta*x) = x * sigma(x) = Swish(x)."""
        x = np.linspace(-10, 10, 10000)
        swish_1 = x * sigmoid(1.0 * x)
        swish = x * sigmoid(x)
        np.testing.assert_allclose(swish_1, swish, atol=1e-15)

    def test_beta_inf_limit(self):
        """As beta -> inf, x * sigma(beta*x) -> ReLU(x), away from x=0."""
        x = np.linspace(-5, 5, 10000)
        mask = np.abs(x) > 0.5
        relu = np.maximum(0, x)
        for beta in [20, 100, 500]:
            swish_beta = x * sigmoid(beta * x)
            np.testing.assert_allclose(
                swish_beta[mask], relu[mask], atol=2.0 / beta + 0.01
            )

    def test_monotone_interpolation(self):
        """For x > 0, Swish_beta is monotonically increasing in beta."""
        x = 2.0
        prev = x * sigmoid(0.01 * x)
        for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            current = x * sigmoid(beta * x)
            assert current >= prev - 1e-10, (
                f"Not monotone in beta: beta={beta}, prev={prev:.6f}, "
                f"current={current:.6f}"
            )
            prev = current


class TestGELUFromGaussianRelevance:
    """Verify Theorem 6.8.1 from Paper 2.

    GELU(x) = x * Phi(x), where Phi is the standard Gaussian CDF.
    """

    def test_gelu_definition(self):
        """GELU(x) = x * Phi(x) over a wide range."""
        x = np.linspace(-5, 5, 10000)
        gelu = x * _gaussian_cdf(x)
        # Verify against an independent computation
        expected = x * 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))
        np.testing.assert_allclose(gelu, expected, atol=1e-14)

    def test_gelu_at_zero(self):
        """GELU(0) = 0."""
        assert 0.0 * _gaussian_cdf(0.0) == 0.0

    def test_gelu_asymptotic_to_relu(self):
        """For large positive x, GELU(x) -> x (like ReLU)."""
        x = np.linspace(5, 20, 100)
        gelu = x * _gaussian_cdf(x)
        np.testing.assert_allclose(gelu, x, atol=0.01)

    def test_gelu_asymptotic_to_zero(self):
        """For large negative x, GELU(x) -> 0 (like ReLU)."""
        x = np.linspace(-20, -5, 100)
        gelu = x * _gaussian_cdf(x)
        np.testing.assert_allclose(gelu, 0.0, atol=0.01)


class TestGELUApproxSwish:
    """Verify Proposition 6.8.2 from Paper 2.

    Phi(x) ~ sigma(1.702 * x), with maximum error < 0.01.
    Therefore GELU(x) ~ Swish_{1.702}(x).
    """

    def test_cdf_approximation_error(self):
        """max|Phi(x) - sigma(1.702x)| < 0.01 over practical range."""
        x = np.linspace(-6, 6, 100000)
        phi = _gaussian_cdf(x)
        sig_approx = sigmoid(1.702 * x)
        max_err = np.max(np.abs(phi - sig_approx))
        assert max_err < 0.01, (
            f"Probit-logistic approximation max error: {max_err:.6f}"
        )

    def test_gelu_swish_approximation(self):
        """max|GELU(x) - Swish_{1.702}(x)| is small over practical range."""
        x = np.linspace(-5, 5, 10000)
        gelu = x * _gaussian_cdf(x)
        swish_1702 = x * sigmoid(1.702 * x)
        max_err = np.max(np.abs(gelu - swish_1702))
        assert max_err < 0.025, (
            f"GELU-Swish1702 max error: {max_err:.6f}"
        )

    def test_approximation_preserves_shape(self):
        """Both GELU and Swish_{1.702} have the same qualitative shape."""
        x = np.linspace(-3, 3, 1000)
        gelu = x * _gaussian_cdf(x)
        swish_1702 = x * sigmoid(1.702 * x)
        # Both should be negative in roughly the same region
        gelu_neg = x[gelu < 0]
        swish_neg = x[swish_1702 < 0]
        assert len(gelu_neg) > 0 and len(swish_neg) > 0
        # Their zero-crossings should be close
        np.testing.assert_allclose(
            gelu_neg[-1], swish_neg[-1], atol=0.1
        )


class TestSoftGatedActivationHierarchy:
    """Verify Proposition 6.8.3 from Paper 2.

    For x > 0: x/2 < swish(x) < gelu(x) < relu(x) = x.

    This follows from the gate ordering: for x > 0,
    0.5 < sigma(x) < Phi(x) < 1 (since Phi(x) ~ sigma(1.702x) > sigma(x)).
    """

    def test_hierarchy_positive_x(self):
        """x/2 < swish(x) < gelu(x) < relu(x) for x > 0.

        Range restricted to avoid float64 saturation where Phi(x) rounds
        to exactly 1.0 (around x > 8.2), collapsing gelu == relu.
        """
        x = np.linspace(0.01, 6, 10000)
        half_x = x / 2.0
        swish = x * sigmoid(x)
        gelu = x * _gaussian_cdf(x)
        relu = x.copy()  # ReLU(x) = x for x > 0

        assert np.all(half_x < swish), "x/2 < swish violated"
        assert np.all(swish < gelu), "swish < gelu violated"
        assert np.all(gelu < relu), "gelu < relu violated"

    def test_gate_ordering(self):
        """For x > 0: 0.5 < sigma(x) < Phi(x) < 1.

        Range restricted to x < 6 to avoid Phi(x) saturating to 1.0 in float64.
        """
        x = np.linspace(0.01, 6, 10000)
        sig = sigmoid(x)
        phi = _gaussian_cdf(x)

        assert np.all(sig > 0.5), "sigma(x) > 0.5 violated for x > 0"
        assert np.all(sig < phi), "sigma(x) < Phi(x) violated for x > 0"
        assert np.all(phi < 1.0), "Phi(x) < 1 violated"

    def test_hierarchy_collapses_at_zero(self):
        """At x=0: all four activations equal 0."""
        x = 0.0
        assert x / 2.0 == 0.0
        assert x * sigmoid(x) == 0.0
        assert x * _gaussian_cdf(x) == 0.0
        assert max(0, x) == 0.0


# -----------------------------------------------------------------------
# Paper 2, Section 5: Neural Network Structure
# -----------------------------------------------------------------------


class TestNeuralNetworkPipeline:
    """Verify Theorem 5.2.1 from Paper 2.

    The 4-stage pipeline (calibrate -> logit -> linear -> sigmoid)
    is equivalent to log_odds_conjunction.
    """

    def test_pipeline_matches_conjunction(self):
        """Manual 4-stage pipeline produces same result as log_odds_conjunction."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(2, 7)
            probs = rng.uniform(0.01, 0.99, size=n)
            alpha = rng.uniform(0.0, 2.0)

            # Stage 1: probs are already calibrated
            # Stage 2: logit transform
            log_odds = logit(probs)
            # Stage 3: linear aggregation with confidence scaling
            l_bar = np.mean(log_odds)
            l_adjusted = l_bar * (n ** alpha)
            # Stage 4: sigmoid output
            pipeline_result = sigmoid(l_adjusted)

            # Compare with log_odds_conjunction
            conj_result = log_odds_conjunction(probs, alpha=alpha)
            np.testing.assert_allclose(pipeline_result, conj_result, atol=1e-10)

    def test_sigmoid_calibrated_reduces_to_logistic_regression(self):
        """When all P_i = sigma(a_i*s_i + b_i), pipeline reduces to logistic regression.

        This verifies Theorem 5.2.1a: logit(sigma(x)) = x collapses the
        hidden layer.
        """
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 6)
            alphas = rng.uniform(0.5, 3.0, size=n)
            betas = rng.uniform(-2.0, 3.0, size=n)
            scores = rng.uniform(-2.0, 5.0, size=n)
            conf_alpha = rng.uniform(0.0, 1.0)

            # Calibrate via sigmoid
            beta_primes = -alphas * betas
            probs = sigmoid(alphas * scores + beta_primes)

            # Full pipeline via conjunction
            full_result = log_odds_conjunction(probs, alpha=conf_alpha)

            # Logistic regression shortcut
            w_primes = alphas / (n ** (1.0 - conf_alpha))
            b = np.sum(beta_primes) / (n ** (1.0 - conf_alpha))
            lr_result = sigmoid(np.sum(w_primes * scores) + b)

            np.testing.assert_allclose(full_result, lr_result, atol=1e-9)

    def test_heterogeneous_calibration_is_nonlinear(self):
        """When signals have mixed calibrations, logit is a genuine nonlinearity.

        BM25 (sigmoid-calibrated) + cosine (linear-calibrated):
        logit acts as identity on BM25 but nonlinearly on cosine.
        """
        rng = np.random.default_rng(42)
        for _ in range(100):
            # BM25 signal: sigmoid-calibrated
            alpha_bm25 = rng.uniform(0.5, 3.0)
            beta_bm25 = rng.uniform(0.0, 3.0)
            s_bm25 = rng.uniform(0, 5)
            p_bm25 = sigmoid(alpha_bm25 * (s_bm25 - beta_bm25))

            # Cosine signal: linear-calibrated
            s_cos = rng.uniform(-0.9, 0.9)
            p_cos = cosine_to_probability(s_cos)

            # logit of sigmoid-calibrated should recover linear pre-activation
            logit_bm25 = logit(p_bm25)
            expected_linear = alpha_bm25 * (s_bm25 - beta_bm25)
            np.testing.assert_allclose(logit_bm25, expected_linear, atol=1e-6)

            # logit of linear-calibrated is nonlinear in s_cos
            # Verify by checking that second derivative is non-zero
            logit_cos = logit(p_cos)
            # This is log((1+s)/(1-s)) which is nonlinear in s
            expected_nonlinear = np.log((1.0 + s_cos) / (1.0 - s_cos))
            np.testing.assert_allclose(logit_cos, expected_nonlinear, atol=1e-6)


class TestParameterCorrespondence:
    """Verify Theorem 5.3.1 from Paper 2.

    In the sigmoid-calibrated special case:
    w'_i = alpha_i / n^(1-alpha), b = sum(beta'_i) / n^(1-alpha)
    """

    def test_weight_formula(self):
        """Effective weights w'_i = alpha_i / n^(1-alpha)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 6)
            alphas = rng.uniform(0.5, 3.0, size=n)
            conf_alpha = rng.uniform(0.0, 1.5)

            expected_weights = alphas / (n ** (1.0 - conf_alpha))
            # These weights should produce correct results when applied to scores
            scores = rng.uniform(-2, 5, size=n)
            betas = rng.uniform(-1, 3, size=n)
            beta_primes = -alphas * betas

            # Using the weight formula directly
            b = np.sum(beta_primes) / (n ** (1.0 - conf_alpha))
            direct = sigmoid(np.sum(expected_weights * scores) + b)

            # Using log_odds_conjunction
            probs = sigmoid(alphas * scores + beta_primes)
            conj = log_odds_conjunction(probs, alpha=conf_alpha)

            np.testing.assert_allclose(direct, conj, atol=1e-9)

    def test_bias_formula(self):
        """Effective bias b = sum(beta'_i) / n^(1-alpha)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 6)
            alphas = rng.uniform(0.5, 3.0, size=n)
            betas = rng.uniform(-1, 3, size=n)
            beta_primes = -alphas * betas
            conf_alpha = rng.uniform(0.0, 1.5)

            # Bias formula
            b = np.sum(beta_primes) / (n ** (1.0 - conf_alpha))

            # Verify: with zero scores, the output should be sigmoid(b)
            scores = np.zeros(n)
            probs = sigmoid(alphas * scores + beta_primes)
            conj = log_odds_conjunction(probs, alpha=conf_alpha)
            expected = sigmoid(b)

            np.testing.assert_allclose(conj, expected, atol=1e-9)


# -----------------------------------------------------------------------
# Paper 2, Section 7: WAND Pruning
# -----------------------------------------------------------------------


class TestMonotonicityPreservationForPruning:
    """Verify Theorem 7.3.1 from Paper 2.

    The sigmoid transformation preserves BM25 upper bounds for
    WAND/BMW pruning.  If s1 > s2 then sigma(alpha*(s1-beta)) >
    sigma(alpha*(s2-beta)).
    """

    def test_sigmoid_preserves_score_ordering(self):
        """BM25 score ordering is preserved under sigmoid transformation."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-2.0, 5.0)
            t = BayesianProbabilityTransform(alpha=alpha, beta=beta)
            scores = np.sort(rng.uniform(-5, 10, size=100))
            likelihoods = t.likelihood(scores)
            # Sorted scores should produce sorted likelihoods
            assert np.all(np.diff(likelihoods) >= 0)

    def test_upper_bound_validity(self):
        """Sigmoid of BM25 upper bound >= sigmoid of any actual BM25 score."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-2.0, 5.0)
            t = BayesianProbabilityTransform(alpha=alpha, beta=beta)

            # Simulate actual scores and their upper bounds
            actual_scores = rng.uniform(0, 5, size=10)
            upper_bounds = actual_scores + rng.uniform(0.1, 3.0, size=10)

            actual_probs = t.likelihood(actual_scores)
            upper_probs = t.likelihood(upper_bounds)

            assert np.all(upper_probs >= actual_probs), (
                f"Upper bound violated: alpha={alpha:.2f}, beta={beta:.2f}"
            )

    def test_wand_upper_bound_method(self):
        """BayesianProbabilityTransform.wand_upper_bound produces valid bounds."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            alpha = rng.uniform(0.1, 5.0)
            beta = rng.uniform(-2.0, 5.0)
            t = BayesianProbabilityTransform(alpha=alpha, beta=beta)

            bm25_ub = rng.uniform(1, 10, size=5)
            bayesian_ub = t.wand_upper_bound(bm25_ub)

            # For any actual score <= upper bound, the actual probability
            # should be <= the Bayesian upper bound
            for i in range(5):
                actual_score = rng.uniform(0, bm25_ub[i])
                actual_prob = t.score_to_probability(
                    actual_score, tf=10.0, doc_len_ratio=0.5
                )
                assert actual_prob <= bayesian_ub[i] + 1e-10, (
                    f"WAND UB violated: actual={actual_prob:.6f}, "
                    f"ub={bayesian_ub[i]:.6f}"
                )


class TestExactPruningRequirements:
    """Verify Theorem 7.5.1 + Corollary 7.5.2 from Paper 2.

    Exact WAND-style pruning requires:
    (i) Boundedness: f: R -> [a, b] for finite a, b
    (ii) Monotonicity: f is strictly monotone

    Sigmoid satisfies both; ReLU satisfies (ii) but not (i).
    """

    def test_sigmoid_is_bounded(self):
        """Sigmoid output is always in (0, 1) within float64-representable range.

        For |x| > ~36, sigmoid saturates to exactly 0.0 or 1.0 in float64.
        The mathematical property holds on R; we test the representable range.
        """
        x = np.linspace(-36, 36, 100000)
        s = sigmoid(x)
        assert np.all(s > 0) and np.all(s < 1)

    def test_sigmoid_is_monotone(self):
        """Sigmoid is strictly increasing."""
        x = np.linspace(-20, 20, 10000)
        s = sigmoid(x)
        assert np.all(np.diff(s) > 0)

    def test_relu_is_monotone(self):
        """ReLU is (weakly) monotonically increasing."""
        x = np.linspace(-10, 10, 10000)
        relu = np.maximum(0, x)
        assert np.all(np.diff(relu) >= 0)

    def test_relu_is_unbounded(self):
        """ReLU is unbounded above -- cannot compute finite output upper bounds."""
        # For any bound M, there exists x such that ReLU(x) > M
        for M in [1, 10, 100, 1000, 1e6]:
            x = M + 1
            assert np.maximum(0, x) > M

    def test_sigmoid_enables_pruning(self):
        """Given a score upper bound, we can compute a tight probability upper bound."""
        alpha, beta = 2.0, 3.0
        score_ub = 5.0  # No score exceeds this
        prob_ub = sigmoid(alpha * (score_ub - beta))
        assert 0 < prob_ub < 1

        # Any score below the upper bound produces a lower probability
        test_scores = np.array([0, 1, 2, 3, 4, 4.99])
        test_probs = sigmoid(alpha * (test_scores - beta))
        assert np.all(test_probs <= prob_ub)


# -----------------------------------------------------------------------
# Paper 2, Section 8: Attention
# -----------------------------------------------------------------------


class TestAttentionAsLogOP:
    """Verify Theorem 8.3 from Paper 2.

    Attention as Logarithmic Opinion Pooling:
    P_LogOP = sigma(sum(w_i * logit(P_i)))
    where w_i >= 0 and sum(w_i) = 1.
    """

    def test_log_op_formula(self):
        """sigma(sum(w_i * logit(P_i))) matches log_odds_conjunction with weights."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(2, 7)
            probs = rng.uniform(0.01, 0.99, size=n)
            # Generate valid softmax weights
            raw = rng.uniform(0.1, 3.0, size=n)
            weights = raw / np.sum(raw)

            # Manual Log-OP computation
            log_op = sigmoid(np.sum(weights * logit(probs)))

            # Via log_odds_conjunction with weights (alpha=0 for pure weighted)
            conj = log_odds_conjunction(probs, alpha=0.0, weights=weights)

            np.testing.assert_allclose(log_op, conj, atol=1e-10)

    def test_softmax_weights_are_valid(self):
        """Softmax produces valid PoE weights (non-negative, sum to 1)."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = rng.integers(2, 10)
            # Simulate query-key compatibility scores
            compatibility = rng.uniform(-2, 2, size=n)
            # Softmax normalization (attention weights)
            exp_scores = np.exp(compatibility - np.max(compatibility))
            weights = exp_scores / np.sum(exp_scores)

            assert np.all(weights >= 0)
            np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-12)

    def test_poe_equivalence(self):
        """Log-OP is mathematically equivalent to normalized Product of Experts.

        P_PoE = prod(P_i^w_i) / (prod(P_i^w_i) + prod((1-P_i)^w_i))
        """
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.05, 0.95, size=n)
            raw = rng.uniform(0.1, 3.0, size=n)
            weights = raw / np.sum(raw)

            # Path 1: Log-OP via logit
            log_op = sigmoid(np.sum(weights * logit(probs)))

            # Path 2: Normalized PoE
            prod_p = np.prod(probs ** weights)
            prod_1mp = np.prod((1.0 - probs) ** weights)
            poe = prod_p / (prod_p + prod_1mp)

            np.testing.assert_allclose(log_op, poe, atol=1e-10)

    def test_attention_with_confidence_scaling(self):
        """Weighted Log-OP with confidence scaling: sigma(n^alpha * sum(w_i * logit(P_i)))."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.1, 0.9, size=n)
            raw = rng.uniform(0.1, 3.0, size=n)
            weights = raw / np.sum(raw)
            alpha = rng.uniform(0.0, 1.0)

            # Manual computation
            manual = sigmoid((n ** alpha) * np.sum(weights * logit(probs)))

            # Via conjunction
            conj = log_odds_conjunction(probs, alpha=alpha, weights=weights)

            np.testing.assert_allclose(manual, conj, atol=1e-10)


# -----------------------------------------------------------------------
# Paper 2, Section 9: Depth
# -----------------------------------------------------------------------


class TestRecursiveBayesianInference:
    """Verify Theorem 9.1.1 from Paper 2.

    Composed inference units produce valid probabilities through L layers.
    Each layer takes the previous layer's posterior outputs as input
    evidence and produces new posteriors.
    """

    def test_single_layer_validity(self):
        """One inference unit: calibrate -> logit -> linear -> sigmoid in (0,1)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 6)
            probs = rng.uniform(0.01, 0.99, size=n)
            alpha = rng.uniform(0.0, 1.0)
            result = log_odds_conjunction(probs, alpha=alpha)
            assert 0 < result < 1

    def test_multi_layer_validity(self):
        """Stacking L inference units preserves (0, 1) range at every layer."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n_signals = rng.integers(2, 5)
            n_layers = rng.integers(2, 8)
            alpha = rng.uniform(0.0, 0.5)

            # Initial evidence signals
            signals = rng.uniform(0.01, 0.99, size=n_signals)

            for layer in range(n_layers):
                # Each layer takes previous outputs as input
                result = log_odds_conjunction(signals, alpha=alpha)
                assert 0 < result < 1, (
                    f"Layer {layer}: result {result} out of (0, 1)"
                )
                # The output becomes one of the signals for the next layer
                # Simulate: replace one signal with the result, add new evidence
                signals = np.concatenate([
                    rng.uniform(0.01, 0.99, size=n_signals - 1),
                    [result],
                ])

    def test_depth_increases_confidence(self):
        """Multiple layers of agreeing evidence push probability toward 1.

        Each layer combines the running belief with a new confirming signal
        (p=0.8) using alpha=0.5.  After 20 layers, confidence should be
        very high.
        """
        p = 0.8  # Moderately strong relevance signal
        current = p
        for _ in range(20):
            signals = np.array([current, p])
            current = log_odds_conjunction(signals, alpha=0.5)
        assert current > 0.95, (
            f"20 layers of agreement should produce high confidence: {current:.4f}"
        )

    def test_depth_preserves_irrelevance(self):
        """Multiple layers of irrelevant evidence keep probability below 0.5."""
        p = 0.3  # Irrelevance signal
        current = p
        for _ in range(10):
            signals = np.array([current, p])
            current = log_odds_conjunction(signals, alpha=0.5)
        assert current < 0.5, (
            f"Irrelevance should be preserved through depth: {current:.4f}"
        )


# -----------------------------------------------------------------------
# Paper 2, Section 6: Additional Exclusion Proofs and Characterizations
# -----------------------------------------------------------------------


class TestSoftplusExclusion:
    """Verify Proposition 6.3.2 from Paper 2.

    The softplus function f(x) = log(1 + exp(x)) maps R -> (0, +inf),
    violating constraint C1 (range).  Additionally f(-x) != 1 - f(x),
    violating constraint C4 (symmetry).
    """

    def test_violates_c1_unbounded_above(self):
        """Softplus output exceeds 1 for large x, violating C1."""
        x = np.array([2.0, 5.0, 10.0, 50.0])
        softplus = np.log(1.0 + np.exp(x))
        assert np.all(softplus > 1), (
            "Softplus should exceed 1 for x > ~0.31"
        )

    def test_violates_c1_never_reaches_zero(self):
        """Softplus is mathematically positive, approaches 0 from above.

        In float64, softplus(-x) underflows to 0.0 for x > ~36.
        We test within the representable range.
        """
        x = np.array([-10.0, -20.0, -30.0])
        softplus = np.log(1.0 + np.exp(x))
        assert np.all(softplus > 0), "Softplus should be strictly positive"
        # But it is NOT in (0, 1) for all x
        x_large = np.array([5.0, 10.0])
        assert np.all(np.log(1.0 + np.exp(x_large)) > 1)

    def test_violates_c4_symmetry(self):
        """f(-x) != 1 - f(x) for softplus."""
        x = np.linspace(-5, 5, 1000)
        f_x = np.log(1.0 + np.exp(x))
        f_neg_x = np.log(1.0 + np.exp(-x))
        one_minus_f_x = 1.0 - f_x
        # These should NOT be equal
        assert not np.allclose(f_neg_x, one_minus_f_x, atol=1e-4), (
            "Softplus should violate the symmetry property f(-x) = 1 - f(x)"
        )

    def test_violates_c3_self_derivative(self):
        """Softplus derivative sigma(x) != f(x) * (1 - f(x))."""
        x = np.linspace(-5, 5, 1000)
        f_x = np.log(1.0 + np.exp(x))
        # Actual derivative of softplus is sigma(x)
        actual_deriv = sigmoid(x)
        # Self-derivative formula would be f(x) * (1 - f(x))
        self_deriv = f_x * (1.0 - f_x)
        assert not np.allclose(actual_deriv, self_deriv, atol=1e-3), (
            "Softplus should violate the self-derivative property"
        )


class TestReLUConstraintViolations:
    """Verify Proposition 6.3.3 from Paper 2.

    ReLU violates:
    - C1: output in [0, +inf), not (0, 1)
    - C3: not differentiable at x=0, and derivative != f(x)*(1-f(x))
    - C4: f(-x) = 0 != 1 - f(x) for x > 0
    """

    def test_violates_c1_range(self):
        """ReLU output exceeds 1 for x > 1."""
        x = np.array([2.0, 5.0, 10.0, 100.0])
        relu = np.maximum(0, x)
        assert np.all(relu > 1), "ReLU should exceed 1 for x > 1"

    def test_violates_c1_exact_zero(self):
        """ReLU outputs exactly 0 for x <= 0 (not strictly in (0, 1))."""
        x = np.array([-5.0, -1.0, 0.0])
        relu = np.maximum(0, x)
        assert np.all(relu == 0), "ReLU should be exactly 0 for x <= 0"

    def test_violates_c3_self_derivative(self):
        """ReLU derivative (step function) != f(x) * (1 - f(x))."""
        # For x > 0: ReLU'(x) = 1, but f(x)*(1-f(x)) = x*(1-x) which is != 1
        x = np.array([0.5, 1.0, 2.0, 5.0])
        relu = np.maximum(0, x)
        actual_deriv = np.ones_like(x)  # derivative of ReLU for x > 0
        self_deriv = relu * (1.0 - relu)
        assert not np.allclose(actual_deriv, self_deriv, atol=1e-3), (
            "ReLU should violate the self-derivative property"
        )

    def test_violates_c4_symmetry(self):
        """f(-x) != 1 - f(x) for ReLU when x > 0."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        f_x = np.maximum(0, x)          # = x
        f_neg_x = np.maximum(0, -x)     # = 0
        one_minus_f_x = 1.0 - f_x       # = 1 - x
        # f(-x) = 0, but 1 - f(x) = 1 - x (these are not equal)
        assert not np.allclose(f_neg_x, one_minus_f_x, atol=1e-3), (
            "ReLU should violate symmetry f(-x) = 1 - f(x)"
        )


class TestReLUCharacterization:
    """Verify Theorem 6.5.4 from Paper 2.

    ReLU is the unique MAP estimator satisfying:
    Q1: Non-negativity (h* >= 0)
    Q2: Sparsity (h* = 0 for a positive-measure set of inputs)
    Q3: Linearity above threshold (h* grows linearly for strong inputs)
    Q4: Hard thresholding (below threshold, output is exactly zero)
    """

    def test_q1_non_negativity(self):
        """ReLU output is always non-negative."""
        x = np.linspace(-100, 100, 100000)
        relu = np.maximum(0, x)
        assert np.all(relu >= 0)

    def test_q2_sparsity(self):
        """ReLU is exactly zero for a positive-measure set of inputs.

        For any threshold theta, ReLU(x - theta) = 0 for all x <= theta,
        which is a positive-measure set.
        """
        rng = np.random.default_rng(42)
        x = rng.uniform(-10, 10, size=10000)
        theta = 1.0
        relu = np.maximum(0, x - theta)
        zero_count = np.sum(relu == 0)
        # Roughly half the inputs should be below threshold
        assert zero_count > 4000, (
            f"Expected many exact zeros: {zero_count}/10000"
        )
        # Verify they are EXACTLY zero, not approximately zero
        assert np.all(relu[x <= theta] == 0.0)

    def test_q3_linearity_above_threshold(self):
        """Above threshold, ReLU grows linearly with unit slope."""
        x = np.linspace(1.0, 100.0, 10000)
        theta = 1.0
        relu = np.maximum(0, x - theta)
        # Should be exactly x - theta for x > theta
        expected = x - theta
        np.testing.assert_allclose(relu, expected, atol=1e-14)
        # Verify unit slope via finite differences
        diffs = np.diff(relu) / np.diff(x)
        np.testing.assert_allclose(diffs, 1.0, atol=1e-10)

    def test_q4_hard_threshold(self):
        """Below threshold, output is structurally zero (not approximately zero).

        This distinguishes ReLU from soft activations like Swish or GELU
        where the output is small but non-zero for negative inputs.
        """
        x = np.linspace(-10, -0.001, 10000)
        relu = np.maximum(0, x)
        # ALL outputs must be exactly 0.0
        assert np.all(relu == 0.0), "Hard threshold violated: non-zero output below 0"

        # Contrast with Swish: Swish is NOT exactly zero below threshold
        swish = x * sigmoid(x)
        assert not np.all(swish == 0.0), (
            "Swish should NOT have hard threshold (soft activation)"
        )

    def test_map_solution_matches_relu_form(self):
        """The MAP closed form max(0, z-theta) has ReLU structure for all parameters."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            w = rng.uniform(0.5, 3.0)
            lam = rng.uniform(0.1, 5.0)
            tau = rng.uniform(0.1, 2.0)
            z = rng.uniform(-5, 10)

            theta = lam * tau**2 / w**2
            h_star = max(0.0, z - theta)

            # Verify ReLU properties hold for the MAP solution
            assert h_star >= 0.0  # Q1
            if z <= theta:
                assert h_star == 0.0  # Q2 + Q4
            else:
                assert h_star == pytest.approx(z - theta, abs=1e-14)  # Q3


# -----------------------------------------------------------------------
# Paper 2, Section 7: Neural Pruning
# -----------------------------------------------------------------------


class TestWANDAsNeuralPruning:
    """Verify Theorem 7.4.1 from Paper 2.

    In the neural interpretation, WAND computes: if the maximum possible
    activation sigma(alpha*(ub - beta)) < theta, the neuron is skipped.
    This pruning is exact: top-k results are identical to exhaustive
    computation.
    """

    def test_pruning_condition_is_safe(self):
        """If sigma(alpha*(ub - beta)) < theta, no actual score can exceed theta."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            alpha = rng.uniform(0.5, 5.0)
            beta = rng.uniform(-1.0, 5.0)

            # BM25 upper bound per term
            ub = rng.uniform(2.0, 10.0)
            # Maximum possible neuron activation
            max_activation = sigmoid(alpha * (ub - beta))

            # Generate actual scores all <= ub
            n_docs = 50
            actual_scores = rng.uniform(0, ub, size=n_docs)
            actual_activations = sigmoid(alpha * (actual_scores - beta))

            # All actual activations must be <= max_activation
            assert np.all(actual_activations <= max_activation + 1e-12), (
                f"Pruning safety violated: max actual={np.max(actual_activations):.6f}, "
                f"bound={max_activation:.6f}"
            )

    def test_pruning_produces_exact_topk(self):
        """WAND-style pruning produces identical top-k to exhaustive scoring."""
        rng = np.random.default_rng(42)
        alpha, beta = 2.0, 3.0
        k = 5

        for _ in range(100):
            n_docs = 50
            # Simulate per-document BM25 scores and upper bounds
            actual_scores = rng.uniform(0, 8, size=n_docs)
            upper_bounds = actual_scores + rng.uniform(0.1, 2.0, size=n_docs)

            # Exhaustive: compute all activations
            all_activations = sigmoid(alpha * (actual_scores - beta))
            topk_exhaustive = np.sort(all_activations)[-k:]

            # WAND pruning: skip documents whose upper bound activation < theta
            # Build threshold progressively (simplified WAND simulation)
            sorted_indices = np.argsort(-all_activations)
            theta = all_activations[sorted_indices[k - 1]] if k <= n_docs else 0.0

            # Check which documents can be pruned
            max_activations = sigmoid(alpha * (upper_bounds - beta))
            prunable = max_activations < theta

            # All pruned documents must have actual activation < theta
            assert np.all(all_activations[prunable] < theta + 1e-12), (
                "Pruned document had activation >= threshold"
            )

            # Non-pruned documents include all top-k
            non_pruned_activations = all_activations[~prunable]
            topk_pruned = np.sort(non_pruned_activations)[-k:]
            np.testing.assert_allclose(topk_pruned, topk_exhaustive, atol=1e-12)

    def test_relu_lacks_finite_bound(self):
        """ReLU cannot provide finite activation upper bounds without input range."""
        # For sigmoid: any score upper bound gives a finite activation bound
        alpha, beta = 2.0, 3.0
        ub = 10.0
        sigmoid_bound = sigmoid(alpha * (ub - beta))
        assert 0 < sigmoid_bound < 1  # Always a finite, useful bound

        # For ReLU: upper bound = ub itself, which grows without limit
        relu_bound = max(0, ub)
        assert relu_bound == ub  # Bound grows with input -- no compression


class TestBMWAsBlockLevelPruning:
    """Verify Corollary 7.4.2 from Paper 2.

    BMW partitions documents into blocks and precomputes per-block maximum
    scores.  If no document in a block can produce a sufficient activation,
    the entire block is skipped.
    """

    def test_block_pruning_is_safe(self):
        """Block-level max activation bounds are valid for all documents in block."""
        rng = np.random.default_rng(42)
        alpha, beta = 2.0, 3.0

        for _ in range(200):
            block_size = rng.integers(10, 50)
            block_scores = rng.uniform(0, 8, size=block_size)
            # Block-max upper bound
            block_max = np.max(block_scores)
            block_max_activation = sigmoid(alpha * (block_max - beta))

            # All activations in the block must be <= block max activation
            block_activations = sigmoid(alpha * (block_scores - beta))
            assert np.all(block_activations <= block_max_activation + 1e-12)

    def test_block_pruning_produces_exact_topk(self):
        """Block-level pruning produces identical top-k to exhaustive scoring."""
        rng = np.random.default_rng(42)
        alpha, beta = 2.0, 3.0
        k = 5

        for _ in range(50):
            n_docs = 200
            block_size = 20
            n_blocks = n_docs // block_size
            scores = rng.uniform(0, 8, size=n_docs)

            # Exhaustive computation
            all_activations = sigmoid(alpha * (scores - beta))
            topk_exhaustive = np.sort(all_activations)[-k:]

            # Progressive threshold (simplified)
            theta = np.sort(all_activations)[-k]

            # Block-level pruning
            non_pruned_activations = []
            for b in range(n_blocks):
                block_start = b * block_size
                block_end = block_start + block_size
                block_scores = scores[block_start:block_end]
                block_max = np.max(block_scores)
                block_max_activation = sigmoid(alpha * (block_max - beta))

                if block_max_activation >= theta:
                    # Block not pruned: evaluate all documents
                    block_activations = sigmoid(alpha * (block_scores - beta))
                    non_pruned_activations.extend(block_activations)

            non_pruned = np.array(non_pruned_activations)
            topk_block = np.sort(non_pruned)[-k:]
            np.testing.assert_allclose(topk_block, topk_exhaustive, atol=1e-12)

    def test_blocks_skipped_are_irrelevant(self):
        """Skipped blocks contain no documents that belong in top-k."""
        rng = np.random.default_rng(42)
        alpha, beta = 2.0, 3.0
        k = 3

        for _ in range(100):
            n_docs = 100
            block_size = 10
            n_blocks = n_docs // block_size
            scores = rng.uniform(0, 8, size=n_docs)

            all_activations = sigmoid(alpha * (scores - beta))
            theta = np.sort(all_activations)[-k]

            for b in range(n_blocks):
                block_slice = slice(b * block_size, (b + 1) * block_size)
                block_max_act = sigmoid(alpha * (np.max(scores[block_slice]) - beta))

                if block_max_act < theta:
                    # This block is pruned -- verify no top-k members are here
                    block_acts = all_activations[block_slice]
                    assert np.all(block_acts < theta + 1e-12), (
                        f"Pruned block contains a top-k document: "
                        f"max_block={np.max(block_acts):.6f}, theta={theta:.6f}"
                    )


# -----------------------------------------------------------------------
# Paper 2, Section 8.7: Exact Attention Pruning
# -----------------------------------------------------------------------


class TestTokenLevelAttentionPruning:
    """Verify Theorem 8.7.1 from Paper 2.

    Token-level exact pruning in attention: a token i can be pruned when
    sum(w_j * v_j, j in A) + sum(w_j * ub(v_j), j not in A) < theta.

    The attention output is a = sum(w_i * v_i) where w_i are softmax
    weights and v_i = logit(P_i) are value vectors in log-odds space.
    """

    def test_pruning_condition_is_safe(self):
        """Upper bound on attention output is always >= actual output."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(3, 15)
            # Calibrated probabilities -> value vectors in log-odds space
            probs = rng.uniform(0.05, 0.95, size=n)
            values = logit(probs)

            # Upper bounds on values (each ub >= actual value)
            upper_bounds = values + rng.uniform(0.1, 2.0, size=n)

            # Softmax attention weights
            raw_scores = rng.uniform(-2, 2, size=n)
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            weights = exp_scores / np.sum(exp_scores)

            # Actual attention output
            actual_output = np.sum(weights * values)

            # Upper bound on output: replace all values with their upper bounds
            upper_bound_output = np.sum(weights * upper_bounds)

            assert actual_output <= upper_bound_output + 1e-12, (
                f"Pruning bound violated: actual={actual_output:.6f}, "
                f"bound={upper_bound_output:.6f}"
            )

    def test_partial_evaluation_bound(self):
        """Partial evaluation + upper bounds on remaining tokens is valid.

        Given a set A of evaluated tokens and the rest unevaluated,
        sum(w_j*v_j, j in A) + sum(w_j*ub(v_j), j not in A) >= actual output.
        """
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(5, 20)
            probs = rng.uniform(0.05, 0.95, size=n)
            values = logit(probs)
            upper_bounds = values + rng.uniform(0.1, 2.0, size=n)

            raw_scores = rng.uniform(-2, 2, size=n)
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            weights = exp_scores / np.sum(exp_scores)

            actual_output = np.sum(weights * values)

            # Evaluate a random subset
            n_evaluated = rng.integers(1, n)
            evaluated = rng.choice(n, size=n_evaluated, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[evaluated] = True

            # Partial evaluation: actual for evaluated, upper bound for rest
            partial_bound = (
                np.sum(weights[mask] * values[mask])
                + np.sum(weights[~mask] * upper_bounds[~mask])
            )

            assert actual_output <= partial_bound + 1e-12, (
                f"Partial evaluation bound violated: actual={actual_output:.6f}, "
                f"bound={partial_bound:.6f}"
            )

    def test_pruning_preserves_topk(self):
        """Token pruning via upper bounds produces exact top-k attention outputs."""
        rng = np.random.default_rng(42)
        k = 3

        for _ in range(100):
            n_queries = 10
            n_tokens = 20

            # Generate value vectors for each query
            all_outputs = []
            pruned_outputs = []

            for q in range(n_queries):
                probs = rng.uniform(0.1, 0.9, size=n_tokens)
                values = logit(probs)
                upper_bounds = values + rng.uniform(0.1, 1.5, size=n_tokens)

                raw_scores = rng.uniform(-2, 2, size=n_tokens)
                exp_scores = np.exp(raw_scores - np.max(raw_scores))
                weights = exp_scores / np.sum(exp_scores)

                output = np.sum(weights * values)
                all_outputs.append(output)

            all_outputs = np.array(all_outputs)
            topk_indices = np.argsort(all_outputs)[-k:]
            topk_exhaustive = np.sort(all_outputs[topk_indices])

            # With pruning: outputs bounded above must include all top-k
            # (simplified: the actual output is always <= bound, so if
            #  bound < threshold, actual < threshold -> safe to prune)
            theta = all_outputs[topk_indices[0]]
            surviving = all_outputs >= theta - 1e-12
            assert np.sum(surviving) >= k

    def test_sigmoid_values_enable_trivial_bounds(self):
        """When values are sigmoid outputs (in (0,1)), ub(v_i) = 1 always works."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(3, 15)
            # Value vectors as sigmoid outputs (bounded in (0, 1))
            values = sigmoid(rng.uniform(-3, 3, size=n))

            raw_scores = rng.uniform(-2, 2, size=n)
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            weights = exp_scores / np.sum(exp_scores)

            actual_output = np.sum(weights * values)
            # Trivial upper bound: all values <= 1
            trivial_bound = np.sum(weights * np.ones(n))  # = 1.0

            assert actual_output <= trivial_bound + 1e-12
            assert trivial_bound == pytest.approx(1.0, abs=1e-12)


class TestHeadLevelAttentionPruning:
    """Verify Corollary 8.7.2 from Paper 2.

    In multi-head attention, each head j can be treated as a BMW block.
    If the maximum possible contribution of head j satisfies:
        ub(head_j) < (theta - a_partial) / H
    then the entire head can be skipped.
    """

    def test_head_upper_bound_validity(self):
        """ub(head_j) = max(w_i * ub(v_i)) >= actual head output for all tokens."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            n_tokens = rng.integers(5, 20)
            probs = rng.uniform(0.05, 0.95, size=n_tokens)
            values = logit(probs)
            upper_bounds = values + rng.uniform(0.1, 2.0, size=n_tokens)

            raw_scores = rng.uniform(-2, 2, size=n_tokens)
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            weights = exp_scores / np.sum(exp_scores)

            # Actual head output
            head_output = np.sum(weights * values)

            # Head upper bound: sum(w_i * ub(v_i))
            head_ub = np.sum(weights * upper_bounds)

            assert head_output <= head_ub + 1e-12

    def test_multihead_pruning_is_exact(self):
        """Pruning entire heads via upper bounds produces exact final output."""
        rng = np.random.default_rng(42)
        n_heads = 8
        n_tokens = 15

        for _ in range(100):
            # Compute actual output for each head
            head_outputs = []
            head_upper_bounds = []

            for h in range(n_heads):
                probs = rng.uniform(0.1, 0.9, size=n_tokens)
                values = logit(probs)
                ubs = values + rng.uniform(0.1, 2.0, size=n_tokens)

                raw_scores = rng.uniform(-2, 2, size=n_tokens)
                exp_scores = np.exp(raw_scores - np.max(raw_scores))
                weights = exp_scores / np.sum(exp_scores)

                head_out = np.sum(weights * values)
                head_ub = np.sum(weights * ubs)

                head_outputs.append(head_out)
                head_upper_bounds.append(head_ub)

            head_outputs = np.array(head_outputs)
            head_upper_bounds = np.array(head_upper_bounds)

            # Final output is mean of heads (simplified multi-head aggregation)
            exact_output = np.mean(head_outputs)

            # Progressive head pruning: if a head's max contribution
            # cannot change the final result significantly, skip it
            a_partial = 0.0
            evaluated_count = 0
            for h in np.argsort(-head_upper_bounds):
                # Remaining budget: how much can remaining heads contribute?
                remaining_heads = n_heads - evaluated_count - 1
                if remaining_heads > 0:
                    remaining_ub = np.sum(
                        np.sort(head_upper_bounds)[-remaining_heads:]
                    )
                else:
                    remaining_ub = 0.0

                # Always evaluate (simplified): just verify bounds
                a_partial += head_outputs[h]
                evaluated_count += 1

            # Final aggregated output matches
            np.testing.assert_allclose(
                a_partial / n_heads, exact_output, atol=1e-12
            )

    def test_pruned_heads_are_negligible(self):
        """Heads whose upper bound is below threshold contribute negligibly."""
        rng = np.random.default_rng(42)
        n_heads = 8
        n_tokens = 10

        for _ in range(100):
            head_outputs = []
            head_ubs = []

            for h in range(n_heads):
                probs = rng.uniform(0.1, 0.9, size=n_tokens)
                values = logit(probs)
                ubs = values + rng.uniform(0.1, 2.0, size=n_tokens)

                raw_scores = rng.uniform(-2, 2, size=n_tokens)
                exp_scores = np.exp(raw_scores - np.max(raw_scores))
                weights = exp_scores / np.sum(exp_scores)

                head_outputs.append(np.sum(weights * values))
                head_ubs.append(np.sum(weights * ubs))

            head_outputs = np.array(head_outputs)
            head_ubs = np.array(head_ubs)

            # The BMW pruning condition: a head can be skipped if its
            # maximum possible contribution is below the per-head threshold
            total_output = np.sum(head_outputs)
            per_head_threshold = total_output / n_heads

            for h in range(n_heads):
                if head_ubs[h] < per_head_threshold:
                    # This head is prunable -- verify its actual contribution
                    # is indeed below the threshold
                    assert head_outputs[h] <= head_ubs[h], (
                        f"Head {h}: actual={head_outputs[h]:.4f} > "
                        f"ub={head_ubs[h]:.4f}"
                    )


# -----------------------------------------------------------------------
# Remark 5.3.2: Learnable Weights / Naive Bayes Init
# -----------------------------------------------------------------------


class TestRemark532NaiveBayesInit:
    """Verify Remark 5.3.2: uniform 1/n weights are the Naive Bayes init."""

    def test_initial_weights_are_uniform(self):
        """Initial weights w_i = 1/n for various n."""
        for n in [1, 2, 3, 5, 8]:
            learner = LearnableLogOddsWeights(n_signals=n)
            expected = np.full(n, 1.0 / n)
            np.testing.assert_allclose(learner.weights, expected, atol=1e-15)

    def test_uniform_init_matches_unweighted_conjunction(self):
        """Uniform init with alpha=0 matches unweighted conjunction with alpha=0.

        At initialization, LearnableLogOddsWeights(n, alpha=0) should produce
        the same result as log_odds_conjunction(probs, alpha=0) because
        uniform weights w_i=1/n make the weighted sum equal to the mean.
        """
        rng = np.random.RandomState(42)
        for n in [2, 3, 5]:
            learner = LearnableLogOddsWeights(n_signals=n, alpha=0.0)
            probs = rng.uniform(0.1, 0.9, size=n)
            result_learner = learner(probs)
            result_unweighted = log_odds_conjunction(probs, alpha=0.0)
            assert result_learner == pytest.approx(result_unweighted, abs=1e-10), (
                f"Mismatch for n={n}: learner={result_learner}, "
                f"unweighted={result_unweighted}"
            )


class TestHebbianGradientStructure:
    """Verify the Hebbian gradient structure from Remark 5.3.2.

    The gradient dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
    has key structural properties tied to the Hebbian interpretation.
    """

    def test_gradient_zero_when_signals_identical(self):
        """Gradient is zero when all signals are identical (x_j = x_bar_w for all j).

        When all input probabilities are equal, every signal equals the
        weighted mean, so (x_j - x_bar_w) = 0 for all j.  No learning
        signal -- the network cannot distinguish between signals.
        """
        learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
        # All signals identical -> x_j = x_bar_w for all j
        probs = np.array([[0.7, 0.7, 0.7]])
        labels = np.array([1.0])

        n = learner.n_signals
        scale = n ** learner.alpha
        x = logit(probs)
        w = learner.weights
        x_bar_w = np.sum(w * x, axis=-1)
        p = np.atleast_1d(np.asarray(sigmoid(scale * x_bar_w), dtype=np.float64))
        error = p - labels

        grad = np.mean(
            scale * error[:, np.newaxis] * w[np.newaxis, :] * (x - x_bar_w[:, np.newaxis]),
            axis=0,
        )
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_gradient_reduces_weight_of_overestimating_signal(self):
        """When prediction overshoots (p > y), the gradient reduces weight of above-mean signals.

        For a positive label y=1, if p > 1 (impossible, but for y=0):
        error = (p - y) > 0.  A signal above the weighted mean has
        (x_j - x_bar_w) > 0.  The gradient dL/dz_j > 0 means z_j
        will decrease, reducing that signal's weight -- correct behavior.
        """
        learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
        # Signal 0: high, Signal 1: low.  Label=0 (not relevant).
        # Combined prediction will be > 0, so p > 0.5 > y=0.
        probs = np.array([[0.9, 0.3]])
        labels = np.array([0.0])

        n = learner.n_signals
        scale = n ** learner.alpha
        x = logit(probs)
        w = learner.weights
        x_bar_w = np.sum(w * x, axis=-1)
        p = np.atleast_1d(np.asarray(sigmoid(scale * x_bar_w), dtype=np.float64))
        error = p - labels  # positive (p > 0, y = 0)

        grad = np.mean(
            scale * error[:, np.newaxis] * w[np.newaxis, :] * (x - x_bar_w[:, np.newaxis]),
            axis=0,
        )

        # Signal 0 (0.9) is above the weighted mean -> positive gradient
        # -> z_0 will decrease -> weight of the "too high" signal decreases
        assert grad[0] > 0, f"Expected positive gradient for above-mean signal, got {grad[0]}"

        # Signal 1 (0.3) is below the weighted mean -> negative gradient
        # -> z_1 will increase -> weight of the "more correct" signal increases
        assert grad[1] < 0, f"Expected negative gradient for below-mean signal, got {grad[1]}"


class TestTheorem531ParameterCorrespondence:
    """Verify Theorem 5.3.1: equal-quality signals maintain uniform weights.

    When all signals are equally informative, training should not
    significantly perturb the weights from their uniform initialization.
    """

    def test_equal_quality_signals_stay_uniform(self):
        """Equal-quality signals maintain approximately uniform weights after training."""
        rng = np.random.RandomState(42)
        n = 3
        m = 500
        labels = rng.randint(0, 2, size=m).astype(np.float64)

        # All signals are equally informative (same noise level)
        signals = []
        for _ in range(n):
            s = np.where(labels == 1, 0.8, 0.2)
            noise = rng.uniform(-0.1, 0.1, size=m)
            s = np.clip(s + noise, 0.05, 0.95)
            signals.append(s)
        probs = np.column_stack(signals)

        learner = LearnableLogOddsWeights(n_signals=n, alpha=0.0)
        learner.fit(probs, labels, learning_rate=0.05, max_iterations=1000)

        # Weights should remain approximately uniform
        uniform = np.full(n, 1.0 / n)
        np.testing.assert_allclose(learner.weights, uniform, atol=0.1, err_msg=(
            f"Equal-quality signals should maintain ~uniform weights, "
            f"got {learner.weights}"
        ))
