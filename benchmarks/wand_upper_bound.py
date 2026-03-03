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

import argparse
import json
from datetime import datetime, timezone

import bm25s
import numpy as np

from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.utils import load_beir_dataset


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
    rng: np.random.Generator,
) -> dict:
    """Evaluate upper bound tightness for various BM25 score ranges."""
    n_samples = 5000

    bm25_bounds = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    tightness_entries = []

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
        tightness_entries.append({
            "bm25_upper_bound": bm25_ub,
            "bayesian_upper_bound": bayesian_ub,
            "actual_max": actual_max,
            "tightness": tightness,
            "safe": safe == "OK",
        })

    return {"entries": tightness_entries}


def run_skip_rate_analysis(
    transform: BayesianProbabilityTransform,
    label: str,
    rng: np.random.Generator,
) -> dict:
    """Estimate document skip rates at various probability thresholds."""
    # Simulate a vocabulary with different BM25 upper bounds per term
    n_terms = 100
    bm25_upper_bounds = rng.exponential(scale=3.0, size=n_terms)
    bm25_upper_bounds = np.clip(bm25_upper_bounds, 0.5, 20.0)

    bayesian_upper_bounds = np.array([
        transform.wand_upper_bound(ub) for ub in bm25_upper_bounds
    ])

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    skip_rate_entries = {}

    print(f"\n  Skip rate analysis ({n_terms} simulated terms):")
    print(f"  {'Threshold':>10}  {'Terms Skipped':>14}  {'Skip Rate':>10}")
    print(f"  {'---' * 15}")

    for threshold in thresholds:
        # Terms whose upper bound is below the threshold can be skipped
        skippable = np.sum(bayesian_upper_bounds < threshold)
        skip_rate = skippable / n_terms
        print(f"  {threshold:>10.2f}  {skippable:>14d}  {skip_rate:>10.1%}")
        skip_rate_entries[str(threshold)] = skip_rate

    return {"skip_rates": skip_rate_entries}


def run_real_corpus_evaluation(dataset_name: str) -> dict:
    """Evaluate WAND upper bound tightness on a real BEIR corpus."""
    print(f"\n{'=' * 72}")
    print(f"Real-Corpus WAND Evaluation: {dataset_name.upper()}")
    print(f"{'=' * 72}")

    ds = load_beir_dataset(dataset_name)

    # Build BM25 index
    bm25_model = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25_model.index(ds.corpus_tokens, show_progress=False)

    # Fit BayesianProbabilityTransform on real scores
    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer.index(ds.corpus_tokens, show_progress=False)
    transform = scorer._transform

    print(f"  alpha={transform.alpha:.4f}, beta={transform.beta:.4f}")

    # Compute per-query term upper bounds from real BM25 scores
    all_tightness = []
    n_safe = 0
    n_total = 0

    for _qid, qtokens in ds.queries[:50]:  # Limit to 50 queries for speed
        scores = bm25_model.get_scores(qtokens)
        nonzero_scores = scores[scores > 0]
        if len(nonzero_scores) == 0:
            continue

        bm25_ub = float(np.max(nonzero_scores))
        bayesian_ub = transform.wand_upper_bound(bm25_ub)

        # Compute actual max probability
        probs = scorer.get_probabilities(qtokens)
        actual_max = float(np.max(probs)) if np.any(probs > 0) else 0.0

        tightness = actual_max / bayesian_ub if bayesian_ub > 0 else 0.0
        safe = bayesian_ub >= actual_max - 1e-9

        all_tightness.append(tightness)
        if safe:
            n_safe += 1
        n_total += 1

    mean_tightness = float(np.mean(all_tightness)) if all_tightness else 0.0
    std_tightness = float(np.std(all_tightness)) if all_tightness else 0.0
    safety_rate = n_safe / n_total if n_total > 0 else 0.0

    print(f"\n  Queries evaluated: {n_total}")
    print(f"  Mean tightness: {mean_tightness:.4f} +/- {std_tightness:.4f}")
    print(f"  Safety rate: {n_safe}/{n_total} ({safety_rate:.1%})")

    # Skip rate analysis with real term upper bounds
    term_ubs = []
    for _qid, qtokens in ds.queries[:50]:
        scores = bm25_model.get_scores(qtokens)
        nonzero_scores = scores[scores > 0]
        if len(nonzero_scores) > 0:
            term_ubs.append(float(np.max(nonzero_scores)))

    if term_ubs:
        bayesian_ubs = np.array([
            transform.wand_upper_bound(ub) for ub in term_ubs
        ])
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        skip_rates = {}
        print(f"\n  Skip rate analysis ({len(term_ubs)} real queries):")
        print(f"  {'Threshold':>10}  {'Skippable':>10}  {'Skip Rate':>10}")
        print(f"  {'---' * 15}")
        for threshold in thresholds:
            skippable = int(np.sum(bayesian_ubs < threshold))
            skip_rate = skippable / len(bayesian_ubs)
            print(f"  {threshold:>10.2f}  {skippable:>10d}  {skip_rate:>10.1%}")
            skip_rates[str(threshold)] = skip_rate
    else:
        skip_rates = {}

    return {
        "dataset": dataset_name,
        "alpha": transform.alpha,
        "beta": transform.beta,
        "n_queries": n_total,
        "mean_tightness": mean_tightness,
        "std_tightness": std_tightness,
        "safety_rate": safety_rate,
        "skip_rates": skip_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WAND upper bound tightness benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of random seeds for statistical reporting",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["nfcorpus", "scifact"],
        help="Run real-corpus evaluation on a BEIR dataset",
    )
    args = parser.parse_args()

    # Real-corpus evaluation mode
    if args.dataset:
        real_result = run_real_corpus_evaluation(args.dataset)
        if args.output:
            output = {
                "benchmark": "wand_upper_bound_real",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": real_result,
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults written to {args.output}")
        return

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

    all_seed_results = []
    for seed_idx in range(args.seeds):
        seed = 42 + seed_idx
        if args.seeds > 1:
            print(f"\n{'#' * 72}")
            print(f"# Seed {seed} ({seed_idx + 1}/{args.seeds})")
            print(f"{'#' * 72}")

        rng = np.random.default_rng(seed)

        config_results = []
        for transform, label in configs:
            print(f"\n{'--' * 36}")
            print(f"  Configuration: {label}")
            print(f"{'--' * 36}")

            tightness = run_tightness_evaluation(transform, label, rng)
            skip_rates = run_skip_rate_analysis(transform, label, rng)
            config_results.append({
                "label": label,
                "tightness": tightness,
                "skip_rates": skip_rates,
            })

        all_seed_results.append({
            "seed": seed,
            "configs": config_results,
        })

    # Summary
    print(f"\n{'=' * 72}")
    print("Interpretation:")
    print("  - Tightness close to 1.0 = bound is tight (good for pruning)")
    print("  - Tightness << 1.0 = bound is loose (safe but fewer skips)")
    print("  - Safe = OK means upper bound always >= actual max (required)")
    print("  - Higher skip rate = more documents pruned at each threshold")
    print("  - base_rate < 0.5 tightens bounds by pulling probabilities down")
    print(f"{'=' * 72}")

    if args.seeds > 1:
        tightness_values = []
        for sr in all_seed_results:
            for cfg in sr["configs"]:
                for entry in cfg["tightness"]["entries"]:
                    tightness_values.append(entry["tightness"])
        mean_t = float(np.mean(tightness_values))
        std_t = float(np.std(tightness_values))
        print(f"\nAggregate ({args.seeds} seeds):")
        print(f"  Mean tightness: {mean_t:.4f} +/- {std_t:.4f}")

    if args.output:
        results_payload = (
            all_seed_results[0] if args.seeds == 1 else all_seed_results
        )
        output = {
            "benchmark": "wand_upper_bound",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": args.seeds,
            "results": results_payload,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
