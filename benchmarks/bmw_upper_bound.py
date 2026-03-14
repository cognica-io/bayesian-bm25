#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: BMW block-max upper bound tightness evaluation.

Evaluates BlockMaxIndex (Paper 1, Section 6.2; Paper 2, Corollary 7.4.2):

  1. Block-level vs global WAND upper bound tightness
  2. Pruning rate at various probability thresholds
  3. Block size sensitivity: how block_size affects bound tightness
  4. Safety verification: no true positive is ever pruned
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.scorer import BlockMaxIndex


def generate_score_matrix(
    n_terms: int,
    n_docs: int,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic per-term BM25 score matrix.

    Parameters
    ----------
    n_terms : int
        Number of query terms.
    n_docs : int
        Number of documents.
    sparsity : float
        Fraction of zero entries (0.0 = dense, 0.9 = very sparse).
    rng : Generator
        Random number generator.

    Returns
    -------
    score_matrix : array of shape (n_terms, n_docs)
    """
    # BM25-like scores: exponential distribution (most scores are small)
    scores = rng.exponential(scale=2.0, size=(n_terms, n_docs))

    # Apply sparsity: zero out a fraction of entries
    mask = rng.random(size=(n_terms, n_docs)) > sparsity
    scores *= mask

    return scores


def run_tightness_comparison(rng: np.random.Generator) -> dict:
    """Compare block-level vs global WAND upper bound tightness."""
    print("=" * 72)
    print("Block-Level vs Global WAND Upper Bound Tightness")
    print("=" * 72)

    transform = BayesianProbabilityTransform(alpha=0.5, beta=3.0)
    n_terms = 5
    n_docs = 1024
    block_size = 128

    score_matrix = generate_score_matrix(n_terms, n_docs, sparsity=0.7, rng=rng)

    idx = BlockMaxIndex(block_size=block_size)
    idx.build(score_matrix)

    # Global WAND upper bound per term
    global_maxes = np.max(score_matrix, axis=1)

    results = []
    print(f"\n  n_docs={n_docs}, n_terms={n_terms}, block_size={block_size}")
    print(f"  n_blocks={idx.n_blocks}")
    print(f"\n  {'Term':>4}  {'Global Max':>10}  {'Global UB':>10}  "
          f"{'Avg Block Max':>13}  {'Avg Block UB':>12}  {'Tightness':>10}")
    print(f"  {'-' * 70}")

    for t in range(n_terms):
        global_ub = float(transform.wand_upper_bound(global_maxes[t]))

        block_maxes = [idx.block_upper_bound(t, b) for b in range(idx.n_blocks)]
        block_ubs = [
            idx.bayesian_block_upper_bound(t, b, transform)
            for b in range(idx.n_blocks)
        ]

        avg_block_max = float(np.mean(block_maxes))
        avg_block_ub = float(np.mean(block_ubs))
        tightness = avg_block_ub / global_ub if global_ub > 0 else 0.0

        print(
            f"  {t:>4}  {global_maxes[t]:>10.4f}  {global_ub:>10.6f}  "
            f"{avg_block_max:>13.4f}  {avg_block_ub:>12.6f}  {tightness:>10.4f}"
        )
        results.append({
            "term": t,
            "global_max": float(global_maxes[t]),
            "global_ub": global_ub,
            "avg_block_max": avg_block_max,
            "avg_block_ub": avg_block_ub,
            "tightness_ratio": tightness,
        })

    print("\n  Tightness ratio = avg_block_ub / global_ub (lower = tighter)")
    return {"terms": results, "n_docs": n_docs, "block_size": block_size}


def run_pruning_rate(rng: np.random.Generator) -> dict:
    """Measure potential pruning rate at various thresholds."""
    print("\n" + "=" * 72)
    print("Pruning Rate at Various Probability Thresholds")
    print("=" * 72)

    transform = BayesianProbabilityTransform(alpha=0.5, beta=3.0)
    n_terms = 3
    n_docs = 2048
    block_size = 128

    score_matrix = generate_score_matrix(n_terms, n_docs, sparsity=0.7, rng=rng)

    idx = BlockMaxIndex(block_size=block_size)
    idx.build(score_matrix)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    results = []
    print(f"\n  n_docs={n_docs}, n_terms={n_terms}, block_size={block_size}")
    print(f"\n  {'Threshold':>9}  {'Blocks Pruned':>13}  {'Block Prune %':>13}  "
          f"{'Docs Pruned':>11}  {'Doc Prune %':>11}  {'Safety':>6}")
    print(f"  {'-' * 70}")

    for threshold in thresholds:
        blocks_pruned = 0
        docs_pruned = 0
        safety_ok = True

        for b in range(idx.n_blocks):
            # A block can be pruned if ALL term upper bounds are below threshold
            # (simplified: use max across terms)
            max_block_ub = max(
                idx.bayesian_block_upper_bound(0, b, transform),
                *[idx.bayesian_block_upper_bound(t, b, transform)
                  for t in range(1, n_terms)]
            )

            if max_block_ub < threshold:
                blocks_pruned += 1
                start = b * block_size
                end = min(start + block_size, n_docs)
                docs_pruned += end - start

                # Safety check: verify no doc in this block actually exceeds threshold
                for d in range(start, end):
                    total_score = float(np.sum(score_matrix[:, d]))
                    actual_prob = float(
                        transform.score_to_probability(total_score, tf=3.0, doc_len_ratio=0.5)
                    )
                    if actual_prob >= threshold:
                        safety_ok = False

        block_pct = 100.0 * blocks_pruned / idx.n_blocks
        doc_pct = 100.0 * docs_pruned / n_docs
        safety_str = "OK" if safety_ok else "FAIL"

        print(
            f"  {threshold:>9.2f}  {blocks_pruned:>13}  {block_pct:>12.1f}%  "
            f"{docs_pruned:>11}  {doc_pct:>10.1f}%  {safety_str:>6}"
        )
        results.append({
            "threshold": threshold,
            "blocks_pruned": blocks_pruned,
            "block_prune_pct": block_pct,
            "docs_pruned": docs_pruned,
            "doc_prune_pct": doc_pct,
            "safety": safety_ok,
        })

    return {"thresholds": results}


def run_block_size_sensitivity(rng: np.random.Generator) -> dict:
    """Evaluate how block_size affects bound tightness."""
    print("\n" + "=" * 72)
    print("Block Size Sensitivity")
    print("=" * 72)

    transform = BayesianProbabilityTransform(alpha=0.5, beta=3.0)
    n_terms = 3
    n_docs = 2048

    score_matrix = generate_score_matrix(n_terms, n_docs, sparsity=0.7, rng=rng)
    global_max = float(np.max(score_matrix))
    global_ub = float(transform.wand_upper_bound(global_max))

    block_sizes = [32, 64, 128, 256, 512, 1024]

    results = []
    print(f"\n  n_docs={n_docs}, n_terms={n_terms}")
    print(f"  Global WAND UB (max score={global_max:.4f}): {global_ub:.6f}")
    print(f"\n  {'Block Size':>10}  {'N Blocks':>8}  {'Avg Block UB':>12}  "
          f"{'Max Block UB':>12}  {'Tightness':>10}")
    print(f"  {'-' * 60}")

    for bs in block_sizes:
        idx = BlockMaxIndex(block_size=bs)
        idx.build(score_matrix)

        all_block_ubs = []
        for t in range(n_terms):
            for b in range(idx.n_blocks):
                all_block_ubs.append(
                    idx.bayesian_block_upper_bound(t, b, transform)
                )

        avg_ub = float(np.mean(all_block_ubs))
        max_ub = float(np.max(all_block_ubs))
        tightness = avg_ub / global_ub if global_ub > 0 else 0.0

        print(
            f"  {bs:>10}  {idx.n_blocks:>8}  {avg_ub:>12.6f}  "
            f"{max_ub:>12.6f}  {tightness:>10.4f}"
        )
        results.append({
            "block_size": bs,
            "n_blocks": idx.n_blocks,
            "avg_block_ub": avg_ub,
            "max_block_ub": max_ub,
            "tightness_ratio": tightness,
        })

    print("\n  Smaller blocks = tighter bounds = more pruning potential")
    return {"block_sizes": results, "global_ub": global_ub}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BMW block-max upper bound benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    tightness = run_tightness_comparison(rng)
    pruning = run_pruning_rate(rng)
    sensitivity = run_block_size_sensitivity(rng)

    if args.output:
        output = {
            "benchmark": "bmw_upper_bound",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {
                "tightness": tightness,
                "pruning_rate": pruning,
                "block_size_sensitivity": sensitivity,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
