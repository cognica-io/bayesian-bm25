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
from bayesian_bm25.fusion import log_odds_conjunction, prob_and, prob_or


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
