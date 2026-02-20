#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Weighted log-odds fusion and cosine-to-probability conversion.

Demonstrates:
  1. Weighted vs uniform log-odds conjunction for hybrid search
  2. cosine_to_probability conversion in a hybrid BM25 + vector pipeline
  3. Impact of weight allocation on fusion quality

Uses synthetic BM25 + vector signals with controlled noise levels to
show when weighted fusion outperforms uniform fusion.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from bayesian_bm25.fusion import (
    cosine_to_probability,
    log_odds_conjunction,
    prob_and,
)
from bayesian_bm25.probability import sigmoid


def generate_hybrid_signals(
    n_docs: int,
    bm25_noise: float,
    vector_noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic BM25 + vector search signals.

    Returns (true_relevance, bm25_probs, cosine_scores, vector_probs).
    """
    # True relevance probability
    true_relevance = rng.uniform(0.05, 0.95, size=n_docs)

    # BM25 signal: logit(true) + noise -> back to probability
    from bayesian_bm25.probability import logit
    bm25_logits = logit(true_relevance) + rng.normal(0, bm25_noise, size=n_docs)
    bm25_probs = np.array(sigmoid(bm25_logits))

    # Vector signal: cosine similarity with noise
    # True cosine = 2*relevance - 1 (map [0,1] -> [-1,1])
    true_cosine = 2.0 * true_relevance - 1.0
    cosine_scores = np.clip(
        true_cosine + rng.normal(0, vector_noise, size=n_docs), -1.0, 1.0
    )
    vector_probs = np.array(cosine_to_probability(cosine_scores))

    return true_relevance, bm25_probs, cosine_scores, vector_probs


def evaluate_fusion(
    fused_probs: np.ndarray,
    true_relevance: np.ndarray,
) -> dict[str, float]:
    """Evaluate fusion quality via MSE and rank correlation."""
    mse = float(np.mean((fused_probs - true_relevance) ** 2))

    # Spearman rank correlation
    rank_fused = np.argsort(np.argsort(-fused_probs))
    rank_true = np.argsort(np.argsort(-true_relevance))
    n = len(fused_probs)
    d_sq = np.sum((rank_fused - rank_true) ** 2)
    spearman = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))

    return {"MSE": mse, "Spearman": spearman}


def run_weighted_fusion_benchmark() -> None:
    """Compare weighted vs uniform fusion across noise scenarios."""
    print("=" * 72)
    print("Weighted Log-Odds Fusion Benchmark")
    print("=" * 72)

    rng = np.random.default_rng(42)
    n_docs = 1000

    # Scenarios: (bm25_noise, vector_noise, description)
    scenarios = [
        (0.5, 2.0, "BM25 reliable, vector noisy"),
        (2.0, 0.5, "BM25 noisy, vector reliable"),
        (1.0, 1.0, "Equal reliability"),
        (0.3, 0.3, "Both reliable"),
        (2.0, 2.0, "Both noisy"),
    ]

    col_w = 32
    print(f"\n  {'Scenario':<{col_w}}  {'Method':<20}  {'MSE':>8}  {'Spearman':>8}")
    print(f"  {'---' * 11}  {'---' * 7}  {'---' * 3}  {'---' * 3}")

    for bm25_noise, vector_noise, desc in scenarios:
        true_rel, bm25_p, _, vector_p = generate_hybrid_signals(
            n_docs, bm25_noise, vector_noise, rng
        )

        # 1. Naive AND (product rule)
        stacked = np.stack([bm25_p, vector_p], axis=-1)
        naive_and = prob_and(stacked)
        r_and = evaluate_fusion(naive_and, true_rel)

        # 2. Uniform log-odds (alpha=0 for fair comparison with weighted)
        uniform_lo = log_odds_conjunction(stacked, alpha=0.0)
        r_uniform = evaluate_fusion(uniform_lo, true_rel)

        # 3. Weighted log-odds: weight toward more reliable signal
        total_noise = bm25_noise + vector_noise
        w_bm25 = vector_noise / total_noise  # Lower noise -> higher weight
        w_vector = bm25_noise / total_noise
        weights = np.array([w_bm25, w_vector])
        weighted_lo = log_odds_conjunction(stacked, weights=weights)
        r_weighted = evaluate_fusion(weighted_lo, true_rel)

        # 4. Oracle: perfectly weighted (uses true noise ratio)
        # Same as weighted in this synthetic setup

        methods = [
            ("Naive AND", r_and),
            ("Uniform log-odds", r_uniform),
            (f"Weighted (w={weights.round(2)})", r_weighted),
        ]

        for i, (method, r) in enumerate(methods):
            scenario_label = desc if i == 0 else ""
            print(
                f"  {scenario_label:<{col_w}}  {method:<20}  "
                f"{r['MSE']:>8.4f}  {r['Spearman']:>8.4f}"
            )
        print()


def run_cosine_pipeline_demo() -> None:
    """Demonstrate the hybrid search pipeline using cosine_to_probability."""
    print("=" * 72)
    print("Hybrid Search Pipeline: cosine_to_probability + log_odds_conjunction")
    print("=" * 72)

    rng = np.random.default_rng(42)

    # Simulate a query result set
    n_results = 20
    print(f"\n  Simulating {n_results} search results from BM25 + vector search\n")

    # BM25 probabilities (already calibrated via Bayesian BM25)
    bm25_probs = np.sort(rng.uniform(0.1, 0.95, size=n_results))[::-1]

    # Cosine similarities from vector search
    cosine_scores = np.sort(rng.uniform(-0.2, 0.95, size=n_results))[::-1]
    vector_probs = cosine_to_probability(cosine_scores)

    # Fuse with weighted log-odds (BM25 weight=0.6, vector weight=0.4)
    weights = np.array([0.6, 0.4])
    stacked = np.stack([bm25_probs, vector_probs], axis=-1)
    fused = log_odds_conjunction(stacked, weights=weights)

    print(f"  {'Rank':>4}  {'BM25 P':>8}  {'Cosine':>8}  {'Vector P':>8}  {'Fused P':>8}")
    print(f"  {'---' * 18}")

    # Sort by fused probability
    order = np.argsort(-fused)
    for rank, idx in enumerate(order[:10], 1):
        print(
            f"  {rank:>4}  {bm25_probs[idx]:>8.4f}  "
            f"{cosine_scores[idx]:>8.4f}  {vector_probs[idx]:>8.4f}  "
            f"{fused[idx]:>8.4f}"
        )

    print(f"\n  Weights: BM25={weights[0]:.1f}, Vector={weights[1]:.1f}")
    print(f"  cosine_to_probability maps [-1,1] -> (0,1) via P = (1+cos)/2")
    print(f"  log_odds_conjunction fuses in log-odds space with reliability weights")


def main() -> None:
    run_weighted_fusion_benchmark()
    print()
    run_cosine_pipeline_demo()


if __name__ == "__main__":
    main()
