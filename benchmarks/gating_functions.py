#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Sparse signal gating functions (Paper 2, Theorems 6.5-6.8, Remark 6.5.4).

Evaluates the impact of gating functions on fusion quality:

  1. Gating comparison: none vs relu vs swish vs gelu vs softplus across noise levels
  2. Generalized swish: beta sensitivity analysis
  3. Fusion quality: BCE and MSE under heterogeneous signal quality
  4. Timing: overhead of gating in log_odds_conjunction
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from bayesian_bm25.fusion import log_odds_conjunction
from bayesian_bm25.probability import sigmoid


def generate_signals(
    n_docs: int,
    noise_levels: list[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic signals with controlled noise.

    Returns (labels, probs) where probs has shape (n_docs, n_signals).
    """
    labels = rng.integers(0, 2, size=n_docs).astype(np.float64)
    true_logits = np.where(labels == 1, 1.5, -1.5)

    signals = []
    for noise in noise_levels:
        noisy_logits = true_logits + rng.normal(0, noise, size=n_docs)
        signals.append(np.asarray(sigmoid(noisy_logits), dtype=np.float64))

    return labels, np.column_stack(signals)


def evaluate_fusion(
    fused: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate fusion quality via BCE and MSE."""
    fused_c = np.clip(fused, 1e-10, 1.0 - 1e-10)
    bce = -float(np.mean(
        labels * np.log(fused_c) + (1 - labels) * np.log(1 - fused_c)
    ))
    mse = float(np.mean((fused - labels) ** 2))
    return {"BCE": bce, "MSE": mse}


def run_gating_comparison(rng: np.random.Generator) -> dict:
    """Compare gating functions across noise scenarios."""
    print("=" * 72)
    print("Gating Function Comparison Across Noise Scenarios")
    print("=" * 72)

    # Scenarios: (noise_levels, description)
    scenarios = [
        ([0.5, 0.5, 0.5], "All reliable (noise=0.5)"),
        ([0.5, 1.0, 2.0], "Mixed quality"),
        ([0.5, 2.0, 3.0], "One reliable, two noisy"),
        ([0.5, 3.0, 5.0], "One reliable, two very noisy"),
    ]
    gating_modes = ["none", "relu", "swish", "gelu", "softplus"]
    n_docs = 5000

    results = []
    col_s = 35
    print(f"\n  {'Scenario':<{col_s}}  {'Gating':<8}  {'BCE':>7}  {'MSE':>7}")
    print(f"  {'-' * col_s}  {'-' * 8}  {'-' * 7}  {'-' * 7}")

    for noise_levels, desc in scenarios:
        labels, probs = generate_signals(n_docs, noise_levels, rng)
        scenario_results = {"scenario": desc}

        for i, gating in enumerate(gating_modes):
            fused = np.asarray(
                log_odds_conjunction(probs, gating=gating), dtype=np.float64
            )
            metrics = evaluate_fusion(fused, labels)

            scenario_label = desc if i == 0 else ""
            print(
                f"  {scenario_label:<{col_s}}  {gating:<8}  "
                f"{metrics['BCE']:>7.4f}  {metrics['MSE']:>7.4f}"
            )
            scenario_results[gating] = metrics

        results.append(scenario_results)
        print()

    return {"scenarios": results}


def run_beta_sensitivity(rng: np.random.Generator) -> dict:
    """Evaluate generalized swish beta parameter sensitivity."""
    print("=" * 72)
    print("Generalized Swish Beta Sensitivity")
    print("=" * 72)

    n_docs = 5000
    noise_levels = [0.5, 2.0, 3.0]
    labels, probs = generate_signals(n_docs, noise_levels, rng)

    betas = [0.01, 0.1, 0.5, 1.0, 1.702, 2.0, 5.0, 10.0, 50.0]

    results = []
    print(f"\n  Noise levels: {noise_levels}")
    print(f"\n  {'Beta':>8}  {'BCE':>7}  {'MSE':>7}  {'Note':>20}")
    print(f"  {'-' * 50}")

    for beta in betas:
        fused = np.asarray(
            log_odds_conjunction(probs, gating="swish", gating_beta=beta),
            dtype=np.float64,
        )
        metrics = evaluate_fusion(fused, labels)

        note = ""
        if beta == 1.0:
            note = "standard swish"
        elif beta == 1.702:
            note = "= GELU"
        elif beta < 0.1:
            note = "-> x/2 limit"
        elif beta >= 50.0:
            note = "-> ReLU limit"

        print(
            f"  {beta:>8.3f}  {metrics['BCE']:>7.4f}  {metrics['MSE']:>7.4f}  "
            f"{note:>20}"
        )
        results.append({
            "beta": beta,
            **metrics,
            "note": note,
        })

    # Verify GELU equivalence
    fused_gelu = np.asarray(
        log_odds_conjunction(probs, gating="gelu"), dtype=np.float64
    )
    fused_swish_1702 = np.asarray(
        log_odds_conjunction(probs, gating="swish", gating_beta=1.702),
        dtype=np.float64,
    )
    max_diff = float(np.max(np.abs(fused_gelu - fused_swish_1702)))
    print(f"\n  GELU vs Swish(1.702) max |diff|: {max_diff:.2e}")

    return {"betas": results, "gelu_swish_max_diff": max_diff}


def run_timing(rng: np.random.Generator) -> dict:
    """Measure gating overhead in log_odds_conjunction."""
    print("\n" + "=" * 72)
    print("Gating Overhead Timing")
    print("=" * 72)

    n_docs = 10000
    noise_levels = [0.5, 1.0, 2.0]
    _, probs = generate_signals(n_docs, noise_levels, rng)

    gating_modes = ["none", "relu", "swish", "gelu", "softplus"]
    n_repeats = 100

    results = []
    print(f"\n  n_docs={n_docs}, n_signals={len(noise_levels)}, repeats={n_repeats}")
    print(f"\n  {'Gating':<8}  {'Total ms':>10}  {'Per-call us':>12}  {'Overhead':>10}")
    print(f"  {'-' * 48}")

    baseline_ms = None
    for gating in gating_modes:
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            log_odds_conjunction(probs, gating=gating)
        total_ms = (time.perf_counter() - t0) * 1000
        per_call_us = total_ms / n_repeats * 1000

        if baseline_ms is None:
            baseline_ms = total_ms
            overhead = "baseline"
        else:
            overhead = f"{total_ms / baseline_ms:.2f}x"

        print(
            f"  {gating:<8}  {total_ms:>10.1f}  {per_call_us:>12.1f}  {overhead:>10}"
        )
        results.append({
            "gating": gating,
            "total_ms": total_ms,
            "per_call_us": per_call_us,
        })

    return {"timings": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse signal gating functions benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    comparison = run_gating_comparison(rng)
    sensitivity = run_beta_sensitivity(rng)
    timing = run_timing(rng)

    if args.output:
        output = {
            "benchmark": "gating_functions",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {
                "gating_comparison": comparison,
                "beta_sensitivity": sensitivity,
                "timing": timing,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
