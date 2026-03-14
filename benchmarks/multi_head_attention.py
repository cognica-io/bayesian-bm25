#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Multi-head attention fusion and pruning (Paper 2, Section 8).

Evaluates:

  1. Multi-head vs single-head fusion quality (BCE, MSE)
  2. Attention pruning safety and efficiency
  3. Head diversity after training
  4. Scaling: number of heads vs quality and timing
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
)
from bayesian_bm25.probability import sigmoid


def generate_data(
    n_docs: int,
    n_signals: int,
    n_qf: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic signals with query features.

    Returns (labels, probs, query_features).
    """
    labels = rng.integers(0, 2, size=n_docs).astype(np.float64)

    signals = []
    noise_levels = [0.5 + i * 0.7 for i in range(n_signals)]
    true_logits = np.where(labels == 1, 1.5, -1.5)
    for noise in noise_levels:
        noisy = true_logits + rng.normal(0, noise, size=n_docs)
        signals.append(np.asarray(sigmoid(noisy), dtype=np.float64))

    probs = np.column_stack(signals)
    query_features = rng.standard_normal(size=(n_docs, n_qf))

    return labels, probs, query_features


def evaluate(
    fused: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute BCE and MSE."""
    fused_c = np.clip(fused, 1e-10, 1.0 - 1e-10)
    bce = -float(np.mean(
        labels * np.log(fused_c) + (1 - labels) * np.log(1 - fused_c)
    ))
    mse = float(np.mean((fused - labels) ** 2))
    return {"BCE": bce, "MSE": mse}


def run_head_comparison(rng: np.random.Generator) -> dict:
    """Compare single-head vs multi-head fusion quality."""
    print("=" * 72)
    print("Single-Head vs Multi-Head Fusion Quality")
    print("=" * 72)

    n_signals = 3
    n_qf = 4
    n_train = 2000
    n_test = 1000

    train_labels, train_probs, train_qf = generate_data(n_train, n_signals, n_qf, rng)
    test_labels, test_probs, test_qf = generate_data(n_test, n_signals, n_qf, rng)

    head_counts = [1, 2, 4, 8]
    results = []

    print(f"\n  n_signals={n_signals}, n_qf={n_qf}, train={n_train}, test={n_test}")
    print(f"\n  {'Heads':>5}  {'Train BCE':>10}  {'Test BCE':>9}  {'Test MSE':>9}  {'Time ms':>8}")
    print(f"  {'-' * 50}")

    for n_heads in head_counts:
        if n_heads == 1:
            model = AttentionLogOddsWeights(
                n_signals=n_signals, n_query_features=n_qf, alpha=0.0,
            )
            t0 = time.perf_counter()
            model.fit(
                train_probs, train_labels, train_qf,
                learning_rate=0.1, max_iterations=1000,
            )
            fit_ms = (time.perf_counter() - t0) * 1000

            train_fused = np.asarray(model(train_probs, train_qf), dtype=np.float64)
            test_fused = np.asarray(model(test_probs, test_qf), dtype=np.float64)
        else:
            model = MultiHeadAttentionLogOddsWeights(
                n_heads=n_heads, n_signals=n_signals,
                n_query_features=n_qf, alpha=0.0,
            )
            t0 = time.perf_counter()
            model.fit(
                train_probs, train_labels, train_qf,
                learning_rate=0.1, max_iterations=1000,
            )
            fit_ms = (time.perf_counter() - t0) * 1000

            train_fused = np.asarray(model(train_probs, train_qf), dtype=np.float64)
            test_fused = np.asarray(model(test_probs, test_qf), dtype=np.float64)

        train_metrics = evaluate(train_fused, train_labels)
        test_metrics = evaluate(test_fused, test_labels)

        print(
            f"  {n_heads:>5}  {train_metrics['BCE']:>10.4f}  "
            f"{test_metrics['BCE']:>9.4f}  {test_metrics['MSE']:>9.4f}  "
            f"{fit_ms:>8.1f}"
        )
        results.append({
            "n_heads": n_heads,
            "train_bce": train_metrics["BCE"],
            "test_bce": test_metrics["BCE"],
            "test_mse": test_metrics["MSE"],
            "fit_ms": fit_ms,
        })

    return {"head_counts": results}


def run_pruning_benchmark(rng: np.random.Generator) -> dict:
    """Evaluate attention pruning safety and efficiency."""
    print("\n" + "=" * 72)
    print("Attention Pruning: Safety and Efficiency")
    print("=" * 72)

    n_signals = 3
    n_qf = 3
    n_train = 1000
    n_candidates = 500

    train_labels, train_probs, train_qf = generate_data(n_train, n_signals, n_qf, rng)

    # Train a multi-head model
    model = MultiHeadAttentionLogOddsWeights(
        n_heads=4, n_signals=n_signals, n_query_features=n_qf, alpha=0.0,
    )
    model.fit(
        train_probs, train_labels, train_qf,
        learning_rate=0.1, max_iterations=500,
    )

    # Generate candidates
    _, cand_probs, cand_qf = generate_data(n_candidates, n_signals, n_qf, rng)

    # Compute actual fused probabilities
    actual = np.asarray(model(cand_probs, cand_qf), dtype=np.float64)

    # Use slightly inflated upper bounds for pruning
    upper_probs = np.clip(cand_probs + 0.05, 0.0, 1.0)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    print(f"\n  n_candidates={n_candidates}, n_heads=4")
    print(f"\n  {'Threshold':>9}  {'Surviving':>9}  {'Pruned':>6}  "
          f"{'Prune %':>8}  {'True Pos':>8}  {'Missed':>6}  {'Safe':>5}")
    print(f"  {'-' * 60}")

    for threshold in thresholds:
        surviving_idx, fused = model.prune(
            cand_probs, cand_qf, threshold, upper_bound_probs=upper_probs,
        )

        n_surviving = len(surviving_idx)
        n_pruned = n_candidates - n_surviving
        prune_pct = 100.0 * n_pruned / n_candidates

        # True positives: candidates with actual >= threshold
        truly_above = set(np.where(actual >= threshold)[0])
        surviving_set = set(surviving_idx)
        missed = len(truly_above - surviving_set)
        safe = "OK" if missed == 0 else "FAIL"

        print(
            f"  {threshold:>9.2f}  {n_surviving:>9}  {n_pruned:>6}  "
            f"{prune_pct:>7.1f}%  {len(truly_above):>8}  {missed:>6}  {safe:>5}"
        )
        results.append({
            "threshold": threshold,
            "surviving": n_surviving,
            "pruned": n_pruned,
            "prune_pct": prune_pct,
            "true_positives": len(truly_above),
            "missed": missed,
            "safe": missed == 0,
        })

    return {"thresholds": results}


def run_head_diversity(rng: np.random.Generator) -> dict:
    """Measure weight diversity across trained heads."""
    print("\n" + "=" * 72)
    print("Head Diversity After Training")
    print("=" * 72)

    n_signals = 3
    n_qf = 3
    n_train = 2000

    train_labels, train_probs, train_qf = generate_data(n_train, n_signals, n_qf, rng)

    model = MultiHeadAttentionLogOddsWeights(
        n_heads=4, n_signals=n_signals, n_query_features=n_qf, alpha=0.0,
    )
    model.fit(
        train_probs, train_labels, train_qf,
        learning_rate=0.1, max_iterations=1000,
    )

    # Compute pairwise L2 distances between head weight matrices
    heads = model.heads
    n_heads = len(heads)
    distances = []

    print(f"\n  Pairwise L2 distances between head weight matrices:")
    print(f"\n  {'Head i':>6}  {'Head j':>6}  {'L2 Distance':>12}")
    print(f"  {'-' * 30}")

    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            wi = heads[i].weights_matrix
            wj = heads[j].weights_matrix
            dist = float(np.sqrt(np.sum((wi - wj) ** 2)))
            print(f"  {i:>6}  {j:>6}  {dist:>12.6f}")
            distances.append({
                "head_i": i,
                "head_j": j,
                "l2_distance": dist,
            })

    avg_dist = float(np.mean([d["l2_distance"] for d in distances]))
    print(f"\n  Average pairwise distance: {avg_dist:.6f}")
    print(f"  {'Diverse' if avg_dist > 0.01 else 'Not diverse'}")

    return {"distances": distances, "avg_distance": avg_dist}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-head attention fusion benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    comparison = run_head_comparison(rng)
    pruning = run_pruning_benchmark(rng)
    diversity = run_head_diversity(rng)

    if args.output:
        output = {
            "benchmark": "multi_head_attention",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {
                "head_comparison": comparison,
                "pruning": pruning,
                "head_diversity": diversity,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
