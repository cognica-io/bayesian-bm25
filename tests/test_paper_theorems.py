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

import numpy as np
import pytest

from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    _clamp_probability,
    logit,
    sigmoid,
)
from bayesian_bm25.fusion import (
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_or,
)


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
