#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: WAND upper bound tightness evaluation.

Evaluates the Bayesian WAND upper bound (Theorem 6.1.2) for safe
document pruning:

  1. Computes Bayesian upper bounds for synthetic BM25 term upper bounds
  2. Compares with actual maximum probabilities across many documents
  3. Reports potential skip rate at various probability thresholds
  4. Measures tightness: ratio of actual_max / upper_bound

Tighter upper bounds lead to more effective pruning (more documents
safely skipped), improving query processing throughput.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from bayesian_bm25.probability import BayesianProbabilityTransform


def compute_actual_max_probability(
    transform: BayesianProbabilityTransform,
    bm25_upper_bound: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Estimate the actual maximum probability by sampling many documents."""
    scores = rng.uniform(0, bm25_upper_bound, size=n_samples)
    tfs = rng.uniform(0, 20, size=n_samples)
    doc_len_ratios = rng.uniform(0.1, 3.0, size=n_samples)

    probs = np.array([
        transform.score_to_probability(scores[i], tfs[i], doc_len_ratios[i])
        for i in range(n_samples)
    ])
    return float(np.max(probs))


def run_tightness_evaluation(
    transform: BayesianProbabilityTransform,
    label: str,
) -> None:
    """Evaluate upper bound tightness for various BM25 score ranges."""
    rng = np.random.default_rng(42)
    n_samples = 5000

    bm25_bounds = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

    print(f"\n  {'BM25 UB':>8}  {'Bayes UB':>9}  {'Actual Max':>10}  "
          f"{'Tightness':>10}  {'Safe':>5}")
    print(f"  {'---' * 20}")

    for bm25_ub in bm25_bounds:
        bayesian_ub = transform.wand_upper_bound(bm25_ub)
        actual_max = compute_actual_max_probability(
            transform, bm25_ub, n_samples, rng
        )

        tightness = actual_max / bayesian_ub if bayesian_ub > 0 else 0.0
        safe = "OK" if bayesian_ub >= actual_max - 1e-9 else "FAIL"

        print(
            f"  {bm25_ub:>8.1f}  {bayesian_ub:>9.6f}  {actual_max:>10.6f}  "
            f"{tightness:>10.4f}  {safe:>5}"
        )


def run_skip_rate_analysis(
    transform: BayesianProbabilityTransform,
    label: str,
) -> None:
    """Estimate document skip rates at various probability thresholds."""
    rng = np.random.default_rng(42)

    # Simulate a vocabulary with different BM25 upper bounds per term
    n_terms = 100
    bm25_upper_bounds = rng.exponential(scale=3.0, size=n_terms)
    bm25_upper_bounds = np.clip(bm25_upper_bounds, 0.5, 20.0)

    bayesian_upper_bounds = np.array([
        transform.wand_upper_bound(ub) for ub in bm25_upper_bounds
    ])

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n  Skip rate analysis ({n_terms} simulated terms):")
    print(f"  {'Threshold':>10}  {'Terms Skipped':>14}  {'Skip Rate':>10}")
    print(f"  {'---' * 15}")

    for threshold in thresholds:
        # Terms whose upper bound is below the threshold can be skipped
        skippable = np.sum(bayesian_upper_bounds < threshold)
        skip_rate = skippable / n_terms
        print(f"  {threshold:>10.2f}  {skippable:>14d}  {skip_rate:>10.1%}")


def main() -> None:
    print("=" * 72)
    print("WAND Upper Bound Tightness Benchmark")
    print("=" * 72)

    configs = [
        (
            BayesianProbabilityTransform(alpha=1.0, beta=2.0),
            "Default (alpha=1.0, beta=2.0, no base_rate)",
        ),
        (
            BayesianProbabilityTransform(alpha=1.5, beta=1.0),
            "Steep sigmoid (alpha=1.5, beta=1.0)",
        ),
        (
            BayesianProbabilityTransform(alpha=1.0, beta=2.0, base_rate=0.01),
            "With base_rate=0.01 (alpha=1.0, beta=2.0)",
        ),
        (
            BayesianProbabilityTransform(alpha=1.0, beta=2.0, base_rate=0.1),
            "With base_rate=0.1 (alpha=1.0, beta=2.0)",
        ),
    ]

    for transform, label in configs:
        print(f"\n{'--' * 36}")
        print(f"  Configuration: {label}")
        print(f"{'--' * 36}")

        run_tightness_evaluation(transform, label)
        run_skip_rate_analysis(transform, label)

    # Summary
    print(f"\n{'=' * 72}")
    print("Interpretation:")
    print("  - Tightness close to 1.0 = bound is tight (good for pruning)")
    print("  - Tightness << 1.0 = bound is loose (safe but fewer skips)")
    print("  - Safe = OK means upper bound always >= actual max (required)")
    print("  - Higher skip rate = more documents pruned at each threshold")
    print("  - base_rate < 0.5 tightens bounds by pulling probabilities down")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
