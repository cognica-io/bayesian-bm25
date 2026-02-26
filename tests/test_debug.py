#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for bayesian_bm25.debug module."""

import numpy as np
import pytest

from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    logit,
    sigmoid,
)
from bayesian_bm25.fusion import (
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
    prob_not,
    prob_or,
)
from bayesian_bm25.debug import (
    BM25SignalTrace,
    ComparisonResult,
    DocumentTrace,
    FusionDebugger,
    FusionTrace,
    NotTrace,
    VectorSignalTrace,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def transform():
    """A typical BayesianProbabilityTransform without base_rate."""
    return BayesianProbabilityTransform(alpha=0.45, beta=6.10)


@pytest.fixture
def transform_with_base_rate():
    """A BayesianProbabilityTransform with base_rate set."""
    return BayesianProbabilityTransform(alpha=0.45, beta=6.10, base_rate=0.02)


@pytest.fixture
def debugger(transform):
    return FusionDebugger(transform)


@pytest.fixture
def debugger_br(transform_with_base_rate):
    return FusionDebugger(transform_with_base_rate)


# ---------------------------------------------------------------------------
# trace_bm25
# ---------------------------------------------------------------------------

class TestTraceBM25:
    def test_intermediate_values_match_transform(self, debugger, transform):
        """All intermediate values must match direct function calls."""
        score, tf, dl = 8.42, 5.0, 0.60
        trace = debugger.trace_bm25(score, tf, dl)

        assert trace.raw_score == score
        assert trace.tf == tf
        assert trace.doc_len_ratio == dl

        assert trace.likelihood == pytest.approx(
            float(transform.likelihood(score))
        )
        assert trace.tf_prior == pytest.approx(
            float(transform.tf_prior(tf))
        )
        assert trace.norm_prior == pytest.approx(
            float(transform.norm_prior(dl))
        )
        assert trace.composite_prior == pytest.approx(
            float(transform.composite_prior(tf, dl))
        )
        assert trace.posterior == pytest.approx(
            float(transform.posterior(trace.likelihood, trace.composite_prior))
        )

    def test_logit_values(self, debugger):
        trace = debugger.trace_bm25(8.42, 5.0, 0.60)
        assert trace.logit_likelihood == pytest.approx(
            float(logit(trace.likelihood))
        )
        assert trace.logit_prior == pytest.approx(
            float(logit(trace.composite_prior))
        )
        assert trace.logit_base_rate is None

    def test_params_snapshot(self, debugger, transform):
        trace = debugger.trace_bm25(5.0, 3.0, 0.8)
        assert trace.alpha == transform.alpha
        assert trace.beta == transform.beta
        assert trace.base_rate is None

    def test_with_base_rate(self, debugger_br, transform_with_base_rate):
        trace = debugger_br.trace_bm25(8.42, 5.0, 0.60)
        t = transform_with_base_rate
        expected_posterior = float(
            t.posterior(trace.likelihood, trace.composite_prior, base_rate=t.base_rate)
        )
        assert trace.posterior == pytest.approx(expected_posterior)
        assert trace.base_rate == 0.02
        assert trace.logit_base_rate == pytest.approx(float(logit(0.02)))

    def test_returns_bm25_signal_trace(self, debugger):
        trace = debugger.trace_bm25(5.0, 3.0, 0.8)
        assert isinstance(trace, BM25SignalTrace)

    def test_extreme_score(self, debugger):
        """Very high score should produce likelihood near 1."""
        trace = debugger.trace_bm25(100.0, 10.0, 0.5)
        assert trace.likelihood == pytest.approx(1.0, abs=1e-5)

    def test_zero_score(self, debugger):
        """Zero score should still produce valid results."""
        trace = debugger.trace_bm25(0.0, 1.0, 1.0)
        assert 0.0 < trace.likelihood < 1.0
        assert 0.0 < trace.posterior < 1.0


# ---------------------------------------------------------------------------
# trace_vector
# ---------------------------------------------------------------------------

class TestTraceVector:
    def test_matches_cosine_to_probability(self, debugger):
        trace = debugger.trace_vector(0.74)
        assert trace.cosine_score == 0.74
        assert trace.probability == pytest.approx(
            float(cosine_to_probability(0.74))
        )
        assert trace.logit_probability == pytest.approx(
            float(logit(trace.probability))
        )

    def test_zero_cosine(self, debugger):
        trace = debugger.trace_vector(0.0)
        assert trace.probability == pytest.approx(0.5)

    def test_high_cosine(self, debugger):
        trace = debugger.trace_vector(0.99)
        assert trace.probability > 0.9

    def test_negative_cosine(self, debugger):
        trace = debugger.trace_vector(-0.5)
        assert trace.probability < 0.5

    def test_returns_vector_signal_trace(self, debugger):
        trace = debugger.trace_vector(0.5)
        assert isinstance(trace, VectorSignalTrace)


# ---------------------------------------------------------------------------
# trace_fusion
# ---------------------------------------------------------------------------

class TestTraceFusion:
    def test_log_odds_default_alpha(self, debugger):
        """Default alpha for unweighted log_odds is 0.5."""
        probs = [0.8, 0.7]
        trace = debugger.trace_fusion(probs)
        assert trace.method == "log_odds"
        assert trace.alpha == 0.5
        assert trace.fused_probability == pytest.approx(
            float(log_odds_conjunction(np.array(probs))), abs=1e-9
        )

    def test_log_odds_captures_intermediates(self, debugger):
        probs = [0.8, 0.7]
        trace = debugger.trace_fusion(probs, method="log_odds")
        assert trace.logits is not None
        assert len(trace.logits) == 2
        assert trace.mean_logit is not None
        assert trace.n_alpha_scale is not None
        assert trace.scaled_logit is not None

        # Verify math: mean of logits
        expected_mean = (float(logit(0.8)) + float(logit(0.7))) / 2.0
        assert trace.mean_logit == pytest.approx(expected_mean)

        # n^alpha = 2^0.5 = sqrt(2)
        assert trace.n_alpha_scale == pytest.approx(2 ** 0.5)

        # scaled = mean * n^alpha
        assert trace.scaled_logit == pytest.approx(
            expected_mean * (2 ** 0.5)
        )

    def test_log_odds_explicit_alpha(self, debugger):
        probs = [0.8, 0.7]
        trace = debugger.trace_fusion(probs, alpha=0.3)
        assert trace.alpha == 0.3
        expected = float(log_odds_conjunction(np.array(probs), alpha=0.3))
        assert trace.fused_probability == pytest.approx(expected, abs=1e-9)

    def test_log_odds_weighted(self, debugger):
        probs = [0.8, 0.7]
        weights = [0.6, 0.4]
        trace = debugger.trace_fusion(probs, weights=weights)
        assert trace.weights is not None
        assert trace.weights == pytest.approx(weights)
        # Default alpha for weighted is 0.0
        assert trace.alpha == 0.0
        expected = float(
            log_odds_conjunction(np.array(probs), weights=np.array(weights))
        )
        assert trace.fused_probability == pytest.approx(expected, abs=1e-9)

    def test_log_odds_weighted_with_alpha(self, debugger):
        probs = [0.8, 0.7]
        weights = [0.6, 0.4]
        trace = debugger.trace_fusion(probs, weights=weights, alpha=0.5)
        assert trace.alpha == 0.5
        expected = float(
            log_odds_conjunction(
                np.array(probs), alpha=0.5, weights=np.array(weights)
            )
        )
        assert trace.fused_probability == pytest.approx(expected, abs=1e-9)

    def test_prob_and(self, debugger):
        probs = [0.8, 0.9]
        trace = debugger.trace_fusion(probs, method="prob_and")
        assert trace.method == "prob_and"
        assert trace.fused_probability == pytest.approx(
            float(prob_and(np.array(probs)))
        )
        # No log_odds intermediates for prob_and
        assert trace.logits is None
        assert trace.mean_logit is None

    def test_prob_or(self, debugger):
        probs = [0.8, 0.9]
        trace = debugger.trace_fusion(probs, method="prob_or")
        assert trace.method == "prob_or"
        assert trace.fused_probability == pytest.approx(
            float(prob_or(np.array(probs)))
        )

    def test_default_signal_names(self, debugger):
        trace = debugger.trace_fusion([0.5, 0.6, 0.7])
        assert trace.signal_names == ["signal_0", "signal_1", "signal_2"]

    def test_custom_signal_names(self, debugger):
        trace = debugger.trace_fusion([0.5, 0.6], names=["BM25", "Vec"])
        assert trace.signal_names == ["BM25", "Vec"]

    def test_invalid_method(self, debugger):
        with pytest.raises(ValueError, match="method must be"):
            debugger.trace_fusion([0.5], method="invalid")

    def test_single_signal(self, debugger):
        trace = debugger.trace_fusion([0.8], alpha=0.0)
        assert trace.fused_probability == pytest.approx(0.8, abs=1e-5)

    def test_returns_fusion_trace(self, debugger):
        trace = debugger.trace_fusion([0.5, 0.6])
        assert isinstance(trace, FusionTrace)

    def test_prob_and_intermediates(self, debugger):
        """prob_and should capture log-space intermediates."""
        probs = [0.8, 0.7, 0.6]
        trace = debugger.trace_fusion(probs, method="prob_and")
        assert trace.log_probs is not None
        assert len(trace.log_probs) == 3
        assert trace.log_prob_sum is not None

        # Verify math: ln(p_i)
        for p, lp in zip(probs, trace.log_probs):
            assert lp == pytest.approx(float(np.log(p)))

        # sum(ln(p_i))
        assert trace.log_prob_sum == pytest.approx(sum(trace.log_probs))

        # exp(sum) = fused
        assert trace.fused_probability == pytest.approx(
            float(np.exp(trace.log_prob_sum)), abs=1e-9
        )

        # log_odds fields should be None
        assert trace.logits is None
        assert trace.complements is None

    def test_prob_or_intermediates(self, debugger):
        """prob_or should capture complement log-space intermediates."""
        probs = [0.8, 0.7, 0.6]
        trace = debugger.trace_fusion(probs, method="prob_or")
        assert trace.complements is not None
        assert trace.log_complements is not None
        assert trace.log_complement_sum is not None
        assert len(trace.complements) == 3
        assert len(trace.log_complements) == 3

        # Verify math: 1-p_i
        for p, c in zip(probs, trace.complements):
            assert c == pytest.approx(1.0 - p, abs=1e-9)

        # ln(1-p_i)
        for c, lc in zip(trace.complements, trace.log_complements):
            assert lc == pytest.approx(float(np.log(c)))

        # sum(ln(1-p_i))
        assert trace.log_complement_sum == pytest.approx(
            sum(trace.log_complements)
        )

        # 1 - exp(sum) = fused
        assert trace.fused_probability == pytest.approx(
            1.0 - float(np.exp(trace.log_complement_sum)), abs=1e-9
        )

        # log_odds / prob_and fields should be None
        assert trace.logits is None
        assert trace.log_probs is None

    def test_prob_not(self, debugger):
        probs = [0.8, 0.9]
        trace = debugger.trace_fusion(probs, method="prob_not")
        assert trace.method == "prob_not"
        # prob_not = prod(1-p_i) = complement of prob_or
        expected_or = float(prob_or(np.array(probs)))
        assert trace.fused_probability == pytest.approx(
            1.0 - expected_or, abs=1e-9
        )

    def test_prob_not_intermediates(self, debugger):
        """prob_not should capture complement log-space intermediates."""
        probs = [0.8, 0.7, 0.6]
        trace = debugger.trace_fusion(probs, method="prob_not")
        assert trace.complements is not None
        assert trace.log_complements is not None
        assert trace.log_complement_sum is not None

        # Verify math: 1-p_i
        for p, c in zip(probs, trace.complements):
            assert c == pytest.approx(1.0 - p, abs=1e-9)

        # ln(1-p_i)
        for c, lc in zip(trace.complements, trace.log_complements):
            assert lc == pytest.approx(float(np.log(c)))

        # sum(ln(1-p_i))
        assert trace.log_complement_sum == pytest.approx(
            sum(trace.log_complements)
        )

        # fused = exp(sum) (NOT exp, not 1-exp like OR)
        assert trace.fused_probability == pytest.approx(
            float(np.exp(trace.log_complement_sum)), abs=1e-9
        )

    def test_prob_not_is_complement_of_prob_or(self, debugger):
        """prob_not + prob_or = 1 for the same inputs."""
        probs = [0.8, 0.7, 0.6]
        not_trace = debugger.trace_fusion(probs, method="prob_not")
        or_trace = debugger.trace_fusion(probs, method="prob_or")
        assert not_trace.fused_probability + or_trace.fused_probability == pytest.approx(
            1.0, abs=1e-9
        )

    def test_prob_not_single_signal(self, debugger):
        """Single signal: prob_not = 1-p."""
        trace = debugger.trace_fusion([0.8], method="prob_not")
        assert trace.fused_probability == pytest.approx(0.2, abs=1e-5)


# ---------------------------------------------------------------------------
# trace_not
# ---------------------------------------------------------------------------

class TestTraceNot:
    def test_complement_matches_prob_not(self, debugger):
        trace = debugger.trace_not(0.7, name="snake")
        assert trace.input_probability == 0.7
        assert trace.input_name == "snake"
        assert trace.complement == pytest.approx(float(prob_not(0.7)))

    def test_logit_sign_flip(self, debugger):
        """logit(NOT p) = -logit(p)."""
        trace = debugger.trace_not(0.8)
        assert trace.logit_complement == pytest.approx(
            -trace.logit_input, abs=1e-9
        )

    def test_logit_values(self, debugger):
        trace = debugger.trace_not(0.7)
        assert trace.logit_input == pytest.approx(float(logit(0.7)))
        assert trace.logit_complement == pytest.approx(float(logit(0.3)))

    def test_half(self, debugger):
        """NOT 0.5 = 0.5 (uncertainty is self-complementary)."""
        trace = debugger.trace_not(0.5)
        assert trace.complement == pytest.approx(0.5)
        assert trace.logit_input == pytest.approx(0.0, abs=1e-9)
        assert trace.logit_complement == pytest.approx(0.0, abs=1e-9)

    def test_near_zero(self, debugger):
        trace = debugger.trace_not(0.01)
        assert trace.complement == pytest.approx(0.99)

    def test_near_one(self, debugger):
        trace = debugger.trace_not(0.99)
        assert trace.complement == pytest.approx(0.01)

    def test_returns_not_trace(self, debugger):
        trace = debugger.trace_not(0.5)
        assert isinstance(trace, NotTrace)

    def test_default_name(self, debugger):
        trace = debugger.trace_not(0.5)
        assert trace.input_name == "signal"

    def test_involution(self, debugger):
        """NOT(NOT(p)) = p."""
        trace1 = debugger.trace_not(0.7)
        trace2 = debugger.trace_not(trace1.complement)
        assert trace2.complement == pytest.approx(0.7, abs=1e-9)


# ---------------------------------------------------------------------------
# format_not
# ---------------------------------------------------------------------------

class TestFormatNot:
    def test_contains_input_and_complement(self, debugger):
        trace = debugger.trace_not(0.7, name="snake")
        output = debugger.format_not(trace)
        assert "NOT snake" in output
        assert "0.700" in output
        assert "0.300" in output

    def test_contains_logit_sign_flip(self, debugger):
        trace = debugger.trace_not(0.7, name="snake")
        output = debugger.format_not(trace)
        assert "sign flipped" in output

    def test_contains_logit_values(self, debugger):
        trace = debugger.trace_not(0.8, name="topic")
        output = debugger.format_not(trace)
        assert "logit" in output


# ---------------------------------------------------------------------------
# trace_document
# ---------------------------------------------------------------------------

class TestTraceDocument:
    def test_bm25_only(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60, doc_id="doc-1",
        )
        assert trace.doc_id == "doc-1"
        assert "BM25" in trace.signals
        assert "Vector" not in trace.signals
        assert isinstance(trace.signals["BM25"], BM25SignalTrace)
        assert isinstance(trace.fusion, FusionTrace)
        assert trace.final_probability == trace.fusion.fused_probability

    def test_vector_only(self, debugger):
        trace = debugger.trace_document(cosine_score=0.74, doc_id="doc-2")
        assert "Vector" in trace.signals
        assert "BM25" not in trace.signals
        assert isinstance(trace.signals["Vector"], VectorSignalTrace)

    def test_both_signals(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60,
            cosine_score=0.74, doc_id="doc-3",
        )
        assert "BM25" in trace.signals
        assert "Vector" in trace.signals
        assert len(trace.fusion.signal_probabilities) == 2
        assert trace.fusion.signal_names == ["BM25", "Vector"]

    def test_fusion_uses_correct_method(self, debugger):
        trace = debugger.trace_document(
            bm25_score=5.0, tf=3.0, doc_len_ratio=0.8,
            cosine_score=0.5, method="prob_and",
        )
        assert trace.fusion.method == "prob_and"

    def test_no_signals_raises(self, debugger):
        with pytest.raises(ValueError, match="At least one"):
            debugger.trace_document()

    def test_bm25_missing_tf_raises(self, debugger):
        with pytest.raises(ValueError, match="tf and doc_len_ratio"):
            debugger.trace_document(bm25_score=5.0)

    def test_bm25_missing_doc_len_ratio_raises(self, debugger):
        with pytest.raises(ValueError, match="tf and doc_len_ratio"):
            debugger.trace_document(bm25_score=5.0, tf=3.0)

    def test_returns_document_trace(self, debugger):
        trace = debugger.trace_document(cosine_score=0.5)
        assert isinstance(trace, DocumentTrace)

    def test_final_probability_matches_fusion(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.0, tf=4.0, doc_len_ratio=0.5, cosine_score=0.7,
        )
        assert trace.final_probability == trace.fusion.fused_probability

    def test_consistency_with_individual_traces(self, debugger):
        """trace_document should produce same values as calling trace_bm25 + trace_vector + trace_fusion."""
        score, tf, dl, cos = 8.42, 5.0, 0.60, 0.74
        doc_trace = debugger.trace_document(
            bm25_score=score, tf=tf, doc_len_ratio=dl,
            cosine_score=cos, doc_id="doc-x",
        )

        bm25_trace = debugger.trace_bm25(score, tf, dl)
        vec_trace = debugger.trace_vector(cos)
        fusion_trace = debugger.trace_fusion(
            [bm25_trace.posterior, vec_trace.probability],
            names=["BM25", "Vector"],
        )

        assert doc_trace.signals["BM25"].posterior == pytest.approx(
            bm25_trace.posterior
        )
        assert doc_trace.signals["Vector"].probability == pytest.approx(
            vec_trace.probability
        )
        assert doc_trace.final_probability == pytest.approx(
            fusion_trace.fused_probability, abs=1e-9
        )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_signal_deltas(self, debugger):
        a = debugger.trace_document(
            bm25_score=8.0, tf=5.0, doc_len_ratio=0.6,
            cosine_score=0.74, doc_id="doc-a",
        )
        b = debugger.trace_document(
            bm25_score=6.0, tf=3.0, doc_len_ratio=0.8,
            cosine_score=0.50, doc_id="doc-b",
        )
        result = debugger.compare(a, b)

        assert "BM25" in result.signal_deltas
        assert "Vector" in result.signal_deltas
        assert result.signal_deltas["BM25"] == pytest.approx(
            a.signals["BM25"].posterior - b.signals["BM25"].posterior
        )
        assert result.signal_deltas["Vector"] == pytest.approx(
            a.signals["Vector"].probability - b.signals["Vector"].probability
        )

    def test_dominant_signal(self, debugger):
        """Dominant signal should have the largest absolute delta."""
        a = debugger.trace_document(
            bm25_score=8.0, tf=5.0, doc_len_ratio=0.6,
            cosine_score=0.74, doc_id="doc-a",
        )
        b = debugger.trace_document(
            bm25_score=6.0, tf=3.0, doc_len_ratio=0.8,
            cosine_score=0.50, doc_id="doc-b",
        )
        result = debugger.compare(a, b)

        max_delta_name = max(
            result.signal_deltas, key=lambda k: abs(result.signal_deltas[k])
        )
        assert result.dominant_signal == max_delta_name

    def test_crossover_detection(self, debugger):
        """Detect when one signal favors the opposite document from the fused result."""
        # doc-a: low BM25 but high vector
        a = debugger.trace_document(
            bm25_score=3.0, tf=1.0, doc_len_ratio=1.5,
            cosine_score=0.95, doc_id="doc-a",
        )
        # doc-b: high BM25 but low vector
        b = debugger.trace_document(
            bm25_score=10.0, tf=8.0, doc_len_ratio=0.5,
            cosine_score=0.10, doc_id="doc-b",
        )
        result = debugger.compare(a, b)

        # Signals should point in opposite directions
        bm25_delta = result.signal_deltas["BM25"]
        vec_delta = result.signal_deltas["Vector"]
        assert bm25_delta * vec_delta < 0  # opposite signs

        # crossover_stage should be set
        assert result.crossover_stage is not None

    def test_no_crossover_when_signals_agree(self, debugger):
        """No crossover when all signals favor the same document."""
        a = debugger.trace_document(
            bm25_score=9.0, tf=7.0, doc_len_ratio=0.5,
            cosine_score=0.90, doc_id="doc-a",
        )
        b = debugger.trace_document(
            bm25_score=3.0, tf=1.0, doc_len_ratio=1.5,
            cosine_score=0.20, doc_id="doc-b",
        )
        result = debugger.compare(a, b)
        assert result.crossover_stage is None

    def test_returns_comparison_result(self, debugger):
        a = debugger.trace_document(cosine_score=0.8, doc_id="a")
        b = debugger.trace_document(cosine_score=0.3, doc_id="b")
        result = debugger.compare(a, b)
        assert isinstance(result, ComparisonResult)
        assert result.doc_a is a
        assert result.doc_b is b

    def test_single_signal_comparison(self, debugger):
        """Comparison works with only one signal type."""
        a = debugger.trace_document(cosine_score=0.9, doc_id="a")
        b = debugger.trace_document(cosine_score=0.3, doc_id="b")
        result = debugger.compare(a, b)
        assert result.dominant_signal == "Vector"
        assert result.signal_deltas["Vector"] > 0


# ---------------------------------------------------------------------------
# format_trace
# ---------------------------------------------------------------------------

class TestFormatTrace:
    def test_contains_document_id(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60, doc_id="doc-42",
        )
        output = debugger.format_trace(trace)
        assert "Document: doc-42" in output

    def test_contains_bm25_fields(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60, doc_id="doc-42",
        )
        output = debugger.format_trace(trace)
        assert "[BM25]" in output
        assert "raw=8.42" in output
        assert "likelihood=" in output
        assert "tf_prior=" in output
        assert "norm_prior=" in output
        assert "composite_prior=" in output
        assert "posterior=" in output

    def test_contains_vector_fields(self, debugger):
        trace = debugger.trace_document(
            cosine_score=0.74, doc_id="doc-42",
        )
        output = debugger.format_trace(trace)
        assert "[Vector]" in output
        assert "cosine=0.740" in output
        assert "prob=" in output

    def test_contains_fusion_fields(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60,
            cosine_score=0.74, doc_id="doc-42",
        )
        output = debugger.format_trace(trace)
        assert "[Fusion]" in output
        assert "method=log_odds" in output
        assert "-> final=" in output

    def test_verbose_includes_logits(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60,
            cosine_score=0.74, doc_id="doc-42",
        )
        verbose_output = debugger.format_trace(trace, verbose=True)
        non_verbose = debugger.format_trace(trace, verbose=False)
        assert "logit(" in verbose_output
        assert "logit(" not in non_verbose

    def test_base_rate_formatting(self, debugger_br):
        trace = debugger_br.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60, doc_id="doc-42",
        )
        output = debugger_br.format_trace(trace)
        assert "base_rate=" in output

    def test_unknown_doc_id(self, debugger):
        trace = debugger.trace_document(cosine_score=0.5)
        output = debugger.format_trace(trace)
        assert "Document: unknown" in output

    def test_prob_and_shows_log_probs(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.0, tf=4.0, doc_len_ratio=0.5,
            cosine_score=0.7, method="prob_and",
        )
        output = debugger.format_trace(trace)
        assert "method=prob_and" in output
        assert "ln(P)=" in output
        assert "sum(ln(P))=" in output

    def test_prob_or_shows_complements(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.0, tf=4.0, doc_len_ratio=0.5,
            cosine_score=0.7, method="prob_or",
        )
        output = debugger.format_trace(trace)
        assert "method=prob_or" in output
        assert "1-P=" in output
        assert "ln(1-P)=" in output
        assert "sum(ln(1-P))=" in output

    def test_prob_and_non_verbose_hides_intermediates(self, debugger):
        trace = debugger.trace_document(
            cosine_score=0.7, method="prob_and",
        )
        output = debugger.format_trace(trace, verbose=False)
        assert "ln(P)=" not in output
        assert "-> final=" in output

    def test_prob_or_non_verbose_hides_intermediates(self, debugger):
        trace = debugger.trace_document(
            cosine_score=0.7, method="prob_or",
        )
        output = debugger.format_trace(trace, verbose=False)
        assert "1-P=" not in output
        assert "-> final=" in output

    def test_prob_not_shows_complements(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.0, tf=4.0, doc_len_ratio=0.5,
            cosine_score=0.7, method="prob_not",
        )
        output = debugger.format_trace(trace)
        assert "method=prob_not" in output
        assert "1-P=" in output
        assert "ln(1-P)=" in output
        assert "sum(ln(1-P))=" in output

    def test_prob_not_non_verbose_hides_intermediates(self, debugger):
        trace = debugger.trace_document(
            cosine_score=0.7, method="prob_not",
        )
        output = debugger.format_trace(trace, verbose=False)
        assert "1-P=" not in output
        assert "-> final=" in output


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_one_line(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60,
            cosine_score=0.74, doc_id="doc-42",
        )
        summary = debugger.format_summary(trace)
        assert "\n" not in summary

    def test_contains_doc_id(self, debugger):
        trace = debugger.trace_document(
            cosine_score=0.74, doc_id="doc-42",
        )
        summary = debugger.format_summary(trace)
        assert summary.startswith("doc-42:")

    def test_contains_signal_values(self, debugger):
        trace = debugger.trace_document(
            bm25_score=8.42, tf=5.0, doc_len_ratio=0.60,
            cosine_score=0.74, doc_id="doc-42",
        )
        summary = debugger.format_summary(trace)
        assert "BM25=" in summary
        assert "Vec=" in summary
        assert "Fused=" in summary

    def test_contains_method(self, debugger):
        trace = debugger.trace_document(
            cosine_score=0.5, doc_id="doc-1",
        )
        summary = debugger.format_summary(trace)
        assert "log_odds" in summary

    def test_unknown_doc_id(self, debugger):
        trace = debugger.trace_document(cosine_score=0.5)
        summary = debugger.format_summary(trace)
        assert summary.startswith("unknown:")


# ---------------------------------------------------------------------------
# format_comparison
# ---------------------------------------------------------------------------

class TestFormatComparison:
    def test_contains_both_doc_ids(self, debugger):
        a = debugger.trace_document(
            bm25_score=8.0, tf=5.0, doc_len_ratio=0.6,
            cosine_score=0.74, doc_id="doc-42",
        )
        b = debugger.trace_document(
            bm25_score=6.0, tf=3.0, doc_len_ratio=0.8,
            cosine_score=0.50, doc_id="doc-77",
        )
        result = debugger.compare(a, b)
        output = debugger.format_comparison(result)
        assert "doc-42" in output
        assert "doc-77" in output

    def test_contains_signal_table(self, debugger):
        a = debugger.trace_document(
            bm25_score=8.0, tf=5.0, doc_len_ratio=0.6,
            cosine_score=0.74, doc_id="doc-42",
        )
        b = debugger.trace_document(
            bm25_score=6.0, tf=3.0, doc_len_ratio=0.8,
            cosine_score=0.50, doc_id="doc-77",
        )
        result = debugger.compare(a, b)
        output = debugger.format_comparison(result)
        assert "Signal" in output
        assert "BM25" in output
        assert "Vector" in output
        assert "Fused" in output
        assert "delta" in output

    def test_contains_rank_order(self, debugger):
        a = debugger.trace_document(cosine_score=0.9, doc_id="a")
        b = debugger.trace_document(cosine_score=0.3, doc_id="b")
        result = debugger.compare(a, b)
        output = debugger.format_comparison(result)
        assert "Rank order:" in output

    def test_contains_dominant_signal(self, debugger):
        a = debugger.trace_document(
            bm25_score=8.0, tf=5.0, doc_len_ratio=0.6,
            cosine_score=0.74, doc_id="a",
        )
        b = debugger.trace_document(
            bm25_score=6.0, tf=3.0, doc_len_ratio=0.8,
            cosine_score=0.50, doc_id="b",
        )
        result = debugger.compare(a, b)
        output = debugger.format_comparison(result)
        assert "Dominant signal:" in output

    def test_crossover_note(self, debugger):
        """When signals disagree, format_comparison should include a note."""
        a = debugger.trace_document(
            bm25_score=3.0, tf=1.0, doc_len_ratio=1.5,
            cosine_score=0.95, doc_id="a",
        )
        b = debugger.trace_document(
            bm25_score=10.0, tf=8.0, doc_len_ratio=0.5,
            cosine_score=0.10, doc_id="b",
        )
        result = debugger.compare(a, b)
        if result.crossover_stage is not None:
            output = debugger.format_comparison(result)
            assert "Note:" in output

    def test_default_doc_labels(self, debugger):
        """When doc_id is None, should use 'doc_a' / 'doc_b'."""
        a = debugger.trace_document(cosine_score=0.9)
        b = debugger.trace_document(cosine_score=0.3)
        result = debugger.compare(a, b)
        output = debugger.format_comparison(result)
        assert "doc_a" in output
        assert "doc_b" in output


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

class TestLazyImport:
    def test_fusion_debugger_importable(self):
        import bayesian_bm25
        assert hasattr(bayesian_bm25, "FusionDebugger")
        assert bayesian_bm25.FusionDebugger is FusionDebugger

    def test_nonexistent_attribute_raises(self):
        import bayesian_bm25
        with pytest.raises(AttributeError, match="no attribute"):
            _ = bayesian_bm25.NoSuchThing


# ---------------------------------------------------------------------------
# End-to-end: full pipeline verification
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_no_base_rate(self, debugger, transform):
        """End-to-end: debugger trace matches manual computation."""
        score, tf, dl, cos = 8.42, 5.0, 0.60, 0.74

        # Manual computation
        likelihood = float(transform.likelihood(score))
        prior = float(transform.composite_prior(tf, dl))
        posterior = float(transform.posterior(likelihood, prior))
        vec_prob = float(cosine_to_probability(cos))
        fused = float(
            log_odds_conjunction(np.array([posterior, vec_prob]))
        )

        # Debugger trace
        doc_trace = debugger.trace_document(
            bm25_score=score, tf=tf, doc_len_ratio=dl,
            cosine_score=cos, doc_id="e2e",
        )

        assert doc_trace.signals["BM25"].posterior == pytest.approx(posterior)
        assert doc_trace.signals["Vector"].probability == pytest.approx(vec_prob)
        assert doc_trace.final_probability == pytest.approx(fused, abs=1e-9)

    def test_full_pipeline_with_base_rate(self, debugger_br, transform_with_base_rate):
        """End-to-end with base_rate."""
        score, tf, dl, cos = 8.42, 5.0, 0.60, 0.74
        t = transform_with_base_rate

        posterior = float(
            t.posterior(
                float(t.likelihood(score)),
                float(t.composite_prior(tf, dl)),
                base_rate=t.base_rate,
            )
        )
        vec_prob = float(cosine_to_probability(cos))
        fused = float(
            log_odds_conjunction(np.array([posterior, vec_prob]))
        )

        doc_trace = debugger_br.trace_document(
            bm25_score=score, tf=tf, doc_len_ratio=dl,
            cosine_score=cos, doc_id="e2e-br",
        )

        assert doc_trace.signals["BM25"].posterior == pytest.approx(posterior)
        assert doc_trace.final_probability == pytest.approx(fused, abs=1e-9)

    def test_prob_and_fusion(self, debugger):
        """trace_document with prob_and method matches direct prob_and."""
        score, tf, dl, cos = 5.0, 3.0, 0.8, 0.6
        doc_trace = debugger.trace_document(
            bm25_score=score, tf=tf, doc_len_ratio=dl,
            cosine_score=cos, method="prob_and",
        )
        bm25_post = doc_trace.signals["BM25"].posterior
        vec_prob = doc_trace.signals["Vector"].probability
        expected = float(prob_and(np.array([bm25_post, vec_prob])))
        assert doc_trace.final_probability == pytest.approx(expected, abs=1e-9)

    def test_prob_or_fusion(self, debugger):
        """trace_document with prob_or method matches direct prob_or."""
        score, tf, dl, cos = 5.0, 3.0, 0.8, 0.6
        doc_trace = debugger.trace_document(
            bm25_score=score, tf=tf, doc_len_ratio=dl,
            cosine_score=cos, method="prob_or",
        )
        bm25_post = doc_trace.signals["BM25"].posterior
        vec_prob = doc_trace.signals["Vector"].probability
        expected = float(prob_or(np.array([bm25_post, vec_prob])))
        assert doc_trace.final_probability == pytest.approx(expected, abs=1e-9)

    def test_weighted_fusion(self, debugger):
        """trace_document with weights matches direct weighted log_odds."""
        score, tf, dl, cos = 5.0, 3.0, 0.8, 0.6
        weights = [0.7, 0.3]
        doc_trace = debugger.trace_document(
            bm25_score=score, tf=tf, doc_len_ratio=dl,
            cosine_score=cos, weights=weights,
        )
        bm25_post = doc_trace.signals["BM25"].posterior
        vec_prob = doc_trace.signals["Vector"].probability
        expected = float(
            log_odds_conjunction(
                np.array([bm25_post, vec_prob]),
                weights=np.array(weights),
            )
        )
        assert doc_trace.final_probability == pytest.approx(expected, abs=1e-9)

    def test_hierarchical_and_or_not(self, debugger):
        """Hierarchical fusion: AND(OR(title, body), vector, NOT(spam))."""
        p_title = 0.85
        p_body = 0.70
        p_vector = 0.80
        p_spam = 0.90

        # Step 1: OR(title, body)
        step1 = debugger.trace_fusion(
            [p_title, p_body], names=["title", "body"], method="prob_or",
        )
        expected_or = float(prob_or(np.array([p_title, p_body])))
        assert step1.fused_probability == pytest.approx(expected_or, abs=1e-9)

        # Step 2: NOT(spam)
        step2 = debugger.trace_not(p_spam, name="spam")
        expected_not = float(prob_not(p_spam))
        assert step2.complement == pytest.approx(expected_not, abs=1e-9)

        # Step 3: AND(step1, vector, NOT(spam))
        step3 = debugger.trace_fusion(
            [step1.fused_probability, p_vector, step2.complement],
            names=["OR(title,body)", "vector", "NOT(spam)"],
            method="prob_and",
        )
        expected_and = float(prob_and(np.array([
            expected_or, p_vector, expected_not,
        ])))
        assert step3.fused_probability == pytest.approx(expected_and, abs=1e-9)

    def test_hierarchical_nested_or_of_ands(self, debugger):
        """Hierarchical: OR(AND(a, b), AND(c, d))."""
        a, b, c, d = 0.9, 0.8, 0.6, 0.7

        left = debugger.trace_fusion([a, b], method="prob_and")
        right = debugger.trace_fusion([c, d], method="prob_and")
        final = debugger.trace_fusion(
            [left.fused_probability, right.fused_probability],
            names=["AND(a,b)", "AND(c,d)"],
            method="prob_or",
        )

        expected_left = float(prob_and(np.array([a, b])))
        expected_right = float(prob_and(np.array([c, d])))
        expected = float(prob_or(np.array([expected_left, expected_right])))
        assert final.fused_probability == pytest.approx(expected, abs=1e-9)

    def test_hierarchical_not_into_log_odds(self, debugger):
        """NOT signal fed into log_odds fusion."""
        p_match = 0.85
        p_exclude = 0.70

        not_trace = debugger.trace_not(p_exclude)
        fused = debugger.trace_fusion(
            [p_match, not_trace.complement],
            names=["match", "NOT(exclude)"],
            method="log_odds",
        )

        expected = float(log_odds_conjunction(
            np.array([p_match, float(prob_not(p_exclude))])
        ))
        assert fused.fused_probability == pytest.approx(expected, abs=1e-9)
