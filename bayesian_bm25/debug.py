#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Fusion debugger for tracing intermediate values through the Bayesian BM25 pipeline.

Records every intermediate value -- likelihood, prior, posterior, logits,
fusion -- so that the final fused probability can be fully explained.

The module is scorer-independent: it works with raw values (scores, tf,
doc_len_ratio, cosine similarity) and a ``BayesianProbabilityTransform``
instance.  No dependency on ``bm25s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    _clamp_probability,
    logit,
    sigmoid,
)
from bayesian_bm25.fusion import (
    cosine_to_probability,
    prob_and,
    prob_not,
    prob_or,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BM25SignalTrace:
    """Trace of a single BM25 signal through the full probability pipeline."""

    # Input
    raw_score: float
    tf: float
    doc_len_ratio: float

    # Intermediate
    likelihood: float
    tf_prior: float
    norm_prior: float
    composite_prior: float

    # Logit-space
    logit_likelihood: float
    logit_prior: float
    logit_base_rate: float | None

    # Output
    posterior: float

    # Transform params snapshot
    alpha: float
    beta: float
    base_rate: float | None


@dataclass
class VectorSignalTrace:
    """Trace of a cosine similarity through probability conversion."""

    # Input
    cosine_score: float

    # Output
    probability: float

    # Logit-space
    logit_probability: float


@dataclass
class NotTrace:
    """Trace of a probabilistic NOT (complement) operation."""

    # Input
    input_probability: float
    input_name: str

    # Output
    complement: float

    # Logit-space (sign flip)
    logit_input: float
    logit_complement: float


@dataclass
class FusionTrace:
    """Trace of the combination step for multiple probability signals."""

    # Input
    signal_probabilities: list[float]
    signal_names: list[str]

    # Method
    method: str  # "log_odds", "prob_and", "prob_or", "prob_not"

    # Intermediate (for log_odds)
    logits: list[float] | None
    mean_logit: float | None
    alpha: float | None
    n_alpha_scale: float | None
    scaled_logit: float | None
    weights: list[float] | None

    # Output
    fused_probability: float

    # Intermediate (for prob_and): log-space product
    log_probs: list[float] | None = None
    log_prob_sum: float | None = None

    # Intermediate (for prob_or): complement log-space product
    complements: list[float] | None = None
    log_complements: list[float] | None = None
    log_complement_sum: float | None = None


@dataclass
class DocumentTrace:
    """Complete trace for one document across all signals and fusion."""

    doc_id: str | int | None
    signals: dict[str, BM25SignalTrace | VectorSignalTrace]
    fusion: FusionTrace
    final_probability: float


@dataclass
class ComparisonResult:
    """Comparison of two document traces explaining rank differences."""

    doc_a: DocumentTrace
    doc_b: DocumentTrace
    signal_deltas: dict[str, float]
    dominant_signal: str
    crossover_stage: str | None


# ---------------------------------------------------------------------------
# FusionDebugger
# ---------------------------------------------------------------------------

class FusionDebugger:
    """Traces intermediate values through the Bayesian BM25 fusion pipeline.

    Parameters
    ----------
    transform : BayesianProbabilityTransform
        A configured transform instance whose parameters (alpha, beta,
        base_rate) define the BM25-to-probability mapping.
    """

    def __init__(self, transform: BayesianProbabilityTransform) -> None:
        self._transform = transform

    def trace_bm25(
        self,
        score: float,
        tf: float,
        doc_len_ratio: float,
        *,
        doc_id: str | int | None = None,
    ) -> BM25SignalTrace:
        """Trace a single BM25 score through the full probability pipeline."""
        t = self._transform

        likelihood_val = float(t.likelihood(score))
        tf_prior_val = float(t.tf_prior(tf))
        norm_prior_val = float(t.norm_prior(doc_len_ratio))
        composite_prior_val = float(t.composite_prior(tf, doc_len_ratio))
        posterior_val = float(
            t.posterior(likelihood_val, composite_prior_val, base_rate=t.base_rate)
        )

        logit_likelihood_val = float(logit(likelihood_val))
        logit_prior_val = float(logit(composite_prior_val))
        logit_base_rate_val = (
            float(logit(t.base_rate)) if t.base_rate is not None else None
        )

        return BM25SignalTrace(
            raw_score=score,
            tf=tf,
            doc_len_ratio=doc_len_ratio,
            likelihood=likelihood_val,
            tf_prior=tf_prior_val,
            norm_prior=norm_prior_val,
            composite_prior=composite_prior_val,
            logit_likelihood=logit_likelihood_val,
            logit_prior=logit_prior_val,
            logit_base_rate=logit_base_rate_val,
            posterior=posterior_val,
            alpha=t.alpha,
            beta=t.beta,
            base_rate=t.base_rate,
        )

    def trace_vector(
        self,
        cosine_score: float,
        *,
        doc_id: str | int | None = None,
    ) -> VectorSignalTrace:
        """Trace a cosine similarity through probability conversion."""
        prob_val = float(cosine_to_probability(cosine_score))
        logit_val = float(logit(prob_val))

        return VectorSignalTrace(
            cosine_score=cosine_score,
            probability=prob_val,
            logit_probability=logit_val,
        )

    def trace_not(
        self,
        probability: float,
        *,
        name: str = "signal",
    ) -> NotTrace:
        """Trace a probabilistic NOT (complement) operation.

        In log-odds space, NOT is a sign flip: logit(1-p) = -logit(p).
        """
        complement = float(prob_not(probability))
        logit_in = float(logit(probability))
        logit_out = float(logit(complement))

        return NotTrace(
            input_probability=probability,
            input_name=name,
            complement=complement,
            logit_input=logit_in,
            logit_complement=logit_out,
        )

    def format_not(self, trace: NotTrace) -> str:
        """Format a NOT trace as human-readable text."""
        lines = [
            f"  [NOT {trace.input_name}]",
            f"    P({trace.input_name}) = {trace.input_probability:.3f}",
            f"    P(NOT {trace.input_name}) = 1 - {trace.input_probability:.3f}"
            f" = {trace.complement:.3f}",
            f"    logit({trace.input_probability:.3f}) = {trace.logit_input:+.3f}",
            f"    logit({trace.complement:.3f}) = {trace.logit_complement:+.3f}"
            f"  (sign flipped)",
        ]
        return "\n".join(lines)

    def trace_fusion(
        self,
        probabilities: list[float] | np.ndarray,
        *,
        names: list[str] | None = None,
        method: str = "log_odds",
        alpha: float | None = None,
        weights: list[float] | np.ndarray | None = None,
    ) -> FusionTrace:
        """Trace the fusion of multiple probability signals.

        Parameters
        ----------
        probabilities : list of float
            Probability values to fuse.
        names : list of str or None
            Human-readable names for each signal.  Defaults to
            ``["signal_0", "signal_1", ...]``.
        method : str
            Fusion method: ``"log_odds"``, ``"prob_and"``, ``"prob_or"``,
            or ``"prob_not"``.
        alpha : float or None
            Confidence scaling exponent for log_odds method.
        weights : list of float or None
            Per-signal weights for weighted log_odds.
        """
        probs = list(probabilities)
        n = len(probs)
        if names is None:
            names = [f"signal_{i}" for i in range(n)]

        if method == "log_odds":
            return self._trace_log_odds(probs, names, alpha, weights)
        elif method == "prob_and":
            return self._trace_prob_and(probs, names)
        elif method == "prob_or":
            return self._trace_prob_or(probs, names)
        elif method == "prob_not":
            return self._trace_prob_not(probs, names)
        else:
            raise ValueError(
                f"method must be 'log_odds', 'prob_and', 'prob_or',"
                f" or 'prob_not', got {method!r}"
            )

    def _trace_log_odds(
        self,
        probs: list[float],
        names: list[str],
        alpha: float | None,
        weights: list[float] | np.ndarray | None,
    ) -> FusionTrace:
        """Unroll log_odds_conjunction to capture all intermediates."""
        n = len(probs)
        probs_arr = _clamp_probability(np.array(probs, dtype=np.float64))
        logits_arr = [float(logit(p)) for p in probs_arr]

        weights_list: list[float] | None = None

        if weights is not None:
            weights_arr = np.array(weights, dtype=np.float64)
            weights_list = [float(w) for w in weights_arr]
            effective_alpha = 0.0 if alpha is None else alpha
            n_alpha_scale = float(n ** effective_alpha)
            # Weighted sum of logits
            weighted_logit = float(np.sum(weights_arr * np.array(logits_arr)))
            scaled = n_alpha_scale * weighted_logit
            fused = float(sigmoid(scaled))
            return FusionTrace(
                signal_probabilities=list(probs_arr),
                signal_names=names,
                method="log_odds",
                logits=logits_arr,
                mean_logit=weighted_logit,
                alpha=effective_alpha,
                n_alpha_scale=n_alpha_scale,
                scaled_logit=scaled,
                weights=weights_list,
                fused_probability=fused,
            )

        effective_alpha = 0.5 if alpha is None else alpha
        mean_logit_val = float(np.mean(logits_arr))
        n_alpha_scale = float(n ** effective_alpha)
        scaled = mean_logit_val * n_alpha_scale
        fused = float(sigmoid(scaled))

        return FusionTrace(
            signal_probabilities=list(probs_arr),
            signal_names=names,
            method="log_odds",
            logits=logits_arr,
            mean_logit=mean_logit_val,
            alpha=effective_alpha,
            n_alpha_scale=n_alpha_scale,
            scaled_logit=scaled,
            weights=None,
            fused_probability=fused,
        )

    def _trace_prob_and(
        self,
        probs: list[float],
        names: list[str],
    ) -> FusionTrace:
        """Capture log-space product intermediates for prob_and."""
        probs_arr = _clamp_probability(np.array(probs, dtype=np.float64))
        log_probs = [float(np.log(p)) for p in probs_arr]
        log_sum = float(np.sum(log_probs))
        fused = float(np.exp(log_sum))

        return FusionTrace(
            signal_probabilities=list(probs_arr),
            signal_names=names,
            method="prob_and",
            logits=None,
            mean_logit=None,
            alpha=None,
            n_alpha_scale=None,
            scaled_logit=None,
            weights=None,
            fused_probability=fused,
            log_probs=log_probs,
            log_prob_sum=log_sum,
        )

    def _trace_prob_or(
        self,
        probs: list[float],
        names: list[str],
    ) -> FusionTrace:
        """Capture complement log-space intermediates for prob_or."""
        probs_arr = _clamp_probability(np.array(probs, dtype=np.float64))
        comps = [float(1.0 - p) for p in probs_arr]
        log_comps = [float(np.log(c)) for c in comps]
        log_sum = float(np.sum(log_comps))
        fused = float(1.0 - np.exp(log_sum))

        return FusionTrace(
            signal_probabilities=list(probs_arr),
            signal_names=names,
            method="prob_or",
            logits=None,
            mean_logit=None,
            alpha=None,
            n_alpha_scale=None,
            scaled_logit=None,
            weights=None,
            fused_probability=fused,
            complements=comps,
            log_complements=log_comps,
            log_complement_sum=log_sum,
        )

    def _trace_prob_not(
        self,
        probs: list[float],
        names: list[str],
    ) -> FusionTrace:
        """Capture complement log-space intermediates for prob_not.

        prob_not computes prod(1 - p_i): the probability that NONE of
        the signals are relevant.  This is the complement of prob_or.
        """
        probs_arr = _clamp_probability(np.array(probs, dtype=np.float64))
        comps = [float(1.0 - p) for p in probs_arr]
        log_comps = [float(np.log(c)) for c in comps]
        log_sum = float(np.sum(log_comps))
        fused = float(np.exp(log_sum))

        return FusionTrace(
            signal_probabilities=list(probs_arr),
            signal_names=names,
            method="prob_not",
            logits=None,
            mean_logit=None,
            alpha=None,
            n_alpha_scale=None,
            scaled_logit=None,
            weights=None,
            fused_probability=fused,
            complements=comps,
            log_complements=log_comps,
            log_complement_sum=log_sum,
        )

    def trace_document(
        self,
        *,
        bm25_score: float | None = None,
        tf: float | None = None,
        doc_len_ratio: float | None = None,
        cosine_score: float | None = None,
        method: str = "log_odds",
        alpha: float | None = None,
        weights: list[float] | np.ndarray | None = None,
        doc_id: str | int | None = None,
    ) -> DocumentTrace:
        """Full pipeline trace for one document (convenience method).

        Traces whichever signals are provided (BM25, vector, or both),
        then fuses them.
        """
        signals: dict[str, BM25SignalTrace | VectorSignalTrace] = {}
        probs: list[float] = []
        names: list[str] = []

        has_bm25 = bm25_score is not None
        has_vector = cosine_score is not None

        if has_bm25:
            if tf is None or doc_len_ratio is None:
                raise ValueError(
                    "tf and doc_len_ratio are required when bm25_score is provided"
                )
            bm25_trace = self.trace_bm25(bm25_score, tf, doc_len_ratio, doc_id=doc_id)
            signals["BM25"] = bm25_trace
            probs.append(bm25_trace.posterior)
            names.append("BM25")

        if has_vector:
            vec_trace = self.trace_vector(cosine_score, doc_id=doc_id)
            signals["Vector"] = vec_trace
            probs.append(vec_trace.probability)
            names.append("Vector")

        if not probs:
            raise ValueError(
                "At least one of bm25_score or cosine_score must be provided"
            )

        fusion_trace = self.trace_fusion(
            probs, names=names, method=method, alpha=alpha, weights=weights,
        )

        return DocumentTrace(
            doc_id=doc_id,
            signals=signals,
            fusion=fusion_trace,
            final_probability=fusion_trace.fused_probability,
        )

    def compare(
        self,
        trace_a: DocumentTrace,
        trace_b: DocumentTrace,
    ) -> ComparisonResult:
        """Compare two document traces to explain rank differences.

        Computes per-signal deltas (a - b) at the final probability
        stage of each signal, identifies the dominant signal (largest
        absolute delta), and detects crossover -- when a signal favors
        the opposite document from the fused result.
        """
        all_names = list(
            dict.fromkeys(
                list(trace_a.signals.keys()) + list(trace_b.signals.keys())
            )
        )

        signal_deltas: dict[str, float] = {}
        for name in all_names:
            prob_a = self._signal_probability(trace_a, name)
            prob_b = self._signal_probability(trace_b, name)
            signal_deltas[name] = prob_a - prob_b

        # Dominant signal: largest absolute delta
        dominant = max(signal_deltas, key=lambda k: abs(signal_deltas[k]))

        # Crossover detection: does any signal favor the opposite document?
        fused_delta = trace_a.final_probability - trace_b.final_probability
        crossover_stage: str | None = None
        for name, delta in signal_deltas.items():
            if name == dominant:
                continue
            # A signal "crosses over" when it favors the opposite document
            if fused_delta != 0.0 and delta != 0.0:
                if (fused_delta > 0 and delta < 0) or (fused_delta < 0 and delta > 0):
                    crossover_stage = name
                    break

        return ComparisonResult(
            doc_a=trace_a,
            doc_b=trace_b,
            signal_deltas=signal_deltas,
            dominant_signal=dominant,
            crossover_stage=crossover_stage,
        )

    @staticmethod
    def _signal_probability(trace: DocumentTrace, name: str) -> float:
        """Extract the final probability from a signal within a document trace."""
        sig = trace.signals.get(name)
        if sig is None:
            return 0.5  # neutral if signal missing
        if isinstance(sig, BM25SignalTrace):
            return sig.posterior
        if isinstance(sig, VectorSignalTrace):
            return sig.probability
        return 0.5

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_trace(self, trace: DocumentTrace, *, verbose: bool = True) -> str:
        """Format a document trace as human-readable text."""
        lines: list[str] = []
        doc_label = trace.doc_id if trace.doc_id is not None else "unknown"
        lines.append(f"Document: {doc_label}")

        for name, sig in trace.signals.items():
            if isinstance(sig, BM25SignalTrace):
                lines.append(
                    f"  [{name}] raw={sig.raw_score:.2f}"
                    f" -> likelihood={sig.likelihood:.3f}"
                    f" (alpha={sig.alpha:.2f}, beta={sig.beta:.2f})"
                )
                lines.append(
                    f"         tf={sig.tf:.0f} -> tf_prior={sig.tf_prior:.3f}"
                )
                lines.append(
                    f"         dl_ratio={sig.doc_len_ratio:.2f}"
                    f" -> norm_prior={sig.norm_prior:.3f}"
                )
                lines.append(
                    f"         composite_prior={sig.composite_prior:.3f}"
                )
                if sig.base_rate is not None:
                    # Show posterior without base_rate first, then with
                    posterior_no_br = float(
                        self._transform.posterior(
                            sig.likelihood, sig.composite_prior, base_rate=None
                        )
                    )
                    lines.append(
                        f"         posterior={posterior_no_br:.3f}"
                    )
                    lines.append(
                        f"         with base_rate={sig.base_rate:.3f}:"
                        f" posterior={sig.posterior:.3f}"
                    )
                else:
                    lines.append(
                        f"         posterior={sig.posterior:.3f}"
                    )
                if verbose:
                    lines.append(
                        f"         logit(posterior)={float(logit(sig.posterior)):.3f}"
                    )
                lines.append("")  # blank line
            elif isinstance(sig, VectorSignalTrace):
                lines.append(
                    f"  [{name}] cosine={sig.cosine_score:.3f}"
                    f" -> prob={sig.probability:.3f}"
                )
                if verbose:
                    lines.append(
                        f"           logit(prob)={sig.logit_probability:.3f}"
                    )
                lines.append("")

        # Fusion
        f = trace.fusion
        alpha_str = f", alpha={f.alpha}" if f.alpha is not None else ""
        n_str = f", n={len(f.signal_probabilities)}"
        lines.append(f"  [Fusion] method={f.method}{alpha_str}{n_str}")
        if verbose:
            # log_odds intermediates
            if f.logits is not None:
                logits_str = "[" + ", ".join(f"{v:.3f}" for v in f.logits) + "]"
                lines.append(f"           logits={logits_str}")
            if f.mean_logit is not None:
                lines.append(f"           mean_logit={f.mean_logit:.3f}")
            if f.n_alpha_scale is not None:
                lines.append(
                    f"           n^alpha={f.n_alpha_scale:.3f},"
                    f" scaled={f.scaled_logit:.3f}"
                )
            if f.weights is not None:
                weights_str = "[" + ", ".join(f"{w:.3f}" for w in f.weights) + "]"
                lines.append(f"           weights={weights_str}")
            # prob_and intermediates
            if f.log_probs is not None:
                lp_str = "[" + ", ".join(f"{v:.3f}" for v in f.log_probs) + "]"
                lines.append(f"           ln(P)={lp_str}")
                lines.append(f"           sum(ln(P))={f.log_prob_sum:.3f}")
            # prob_or intermediates
            if f.complements is not None:
                cp_str = "[" + ", ".join(f"{v:.3f}" for v in f.complements) + "]"
                lines.append(f"           1-P={cp_str}")
            if f.log_complements is not None:
                lc_str = "[" + ", ".join(f"{v:.3f}" for v in f.log_complements) + "]"
                lines.append(f"           ln(1-P)={lc_str}")
                lines.append(f"           sum(ln(1-P))={f.log_complement_sum:.3f}")
        lines.append(f"           -> final={f.fused_probability:.3f}")

        return "\n".join(lines)

    def format_summary(self, trace: DocumentTrace) -> str:
        """Compact one-line summary of a document trace."""
        doc_label = trace.doc_id if trace.doc_id is not None else "unknown"
        parts: list[str] = []
        for name, sig in trace.signals.items():
            if isinstance(sig, BM25SignalTrace):
                parts.append(f"BM25={sig.posterior:.3f}")
            elif isinstance(sig, VectorSignalTrace):
                parts.append(f"Vec={sig.probability:.3f}")

        f = trace.fusion
        alpha_str = f", alpha={f.alpha}" if f.alpha is not None else ""
        signal_str = " ".join(parts)
        return (
            f"{doc_label}: {signal_str}"
            f" -> Fused={f.fused_probability:.3f}"
            f" ({f.method}{alpha_str})"
        )

    def format_comparison(self, comparison: ComparisonResult) -> str:
        """Format a comparison result as human-readable text."""
        a = comparison.doc_a
        b = comparison.doc_b
        a_label = a.doc_id if a.doc_id is not None else "doc_a"
        b_label = b.doc_id if b.doc_id is not None else "doc_b"

        lines: list[str] = []
        lines.append(f"Comparison: {a_label} vs {b_label}")

        # Header
        lines.append(
            f"  {'Signal':<12} {str(a_label):>8}  {str(b_label):>8}"
            f"  {'delta':>8}   dominant"
        )

        all_names = list(comparison.signal_deltas.keys())
        for name in all_names:
            prob_a = self._signal_probability(a, name)
            prob_b = self._signal_probability(b, name)
            delta = comparison.signal_deltas[name]
            dominant_marker = (
                "   <-- largest" if name == comparison.dominant_signal else ""
            )
            lines.append(
                f"  {name:<12} {prob_a:>8.3f}  {prob_b:>8.3f}"
                f"  {delta:>+8.3f}{dominant_marker}"
            )

        # Fused row
        fused_delta = a.final_probability - b.final_probability
        lines.append(
            f"  {'Fused':<12} {a.final_probability:>8.3f}"
            f"  {b.final_probability:>8.3f}"
            f"  {fused_delta:>+8.3f}"
        )
        lines.append("")

        # Rank order
        if fused_delta > 0:
            lines.append(
                f"  Rank order: {a_label} > {b_label} (by {fused_delta:+.3f})"
            )
        elif fused_delta < 0:
            lines.append(
                f"  Rank order: {b_label} > {a_label} (by +{abs(fused_delta):.3f})"
            )
        else:
            lines.append(f"  Rank order: tied")

        # Dominant signal
        dom = comparison.dominant_signal
        dom_delta = comparison.signal_deltas[dom]
        if dom_delta >= 0:
            favored = a_label
        else:
            favored = b_label
        lines.append(
            f"  Dominant signal: {dom}"
            f" ({dom_delta:+.3f} in {favored}'s favor)"
        )

        # Crossover note
        if comparison.crossover_stage is not None:
            cross = comparison.crossover_stage
            cross_delta = comparison.signal_deltas[cross]
            if cross_delta >= 0:
                cross_favored = a_label
            else:
                cross_favored = b_label
            lines.append(
                f"  Note: {cross} favored {cross_favored},"
                f" but {dom} signal outweighed it"
            )

        return "\n".join(lines)
