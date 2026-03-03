#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Scalability of Bayesian BM25 across corpus sizes.

Measures wall time and peak memory for core operations:
  - index() wall time + peak memory
  - retrieve(k=10) wall time per query
  - get_probabilities() wall time per query
  - _compute_tf_batch() wall time per query

Uses a synthetic corpus with Zipf-distributed vocabulary to simulate
realistic term frequency patterns.

Supports --output for JSON and --large for 1M-document corpus.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from datetime import datetime, timezone

import numpy as np

from bayesian_bm25.scorer import BayesianBM25Scorer


def generate_synthetic_corpus(
    n_docs: int,
    vocab_size: int,
    avg_doc_len: int,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[list[str]]]:
    """Generate a synthetic corpus with Zipf-distributed vocabulary.

    Returns (corpus_tokens, queries) where queries are 3-5 term samples.
    """
    # Build vocabulary
    vocab = [f"term_{i}" for i in range(vocab_size)]

    # Zipf distribution for term frequencies
    zipf_weights = 1.0 / np.arange(1, vocab_size + 1)
    zipf_weights /= zipf_weights.sum()

    # Generate documents
    corpus_tokens = []
    for _ in range(n_docs):
        doc_len = max(5, int(rng.normal(avg_doc_len, avg_doc_len * 0.3)))
        term_indices = rng.choice(vocab_size, size=doc_len, p=zipf_weights)
        tokens = [vocab[i] for i in term_indices]
        corpus_tokens.append(tokens)

    # Generate queries (3-5 terms each, sampled from the vocabulary)
    n_queries = min(100, n_docs // 10)
    queries = []
    for _ in range(n_queries):
        q_len = rng.integers(3, 6)
        term_indices = rng.choice(vocab_size, size=q_len, p=zipf_weights)
        queries.append([vocab[i] for i in term_indices])

    return corpus_tokens, queries


def measure_index(
    scorer: BayesianBM25Scorer,
    corpus_tokens: list[list[str]],
) -> dict:
    """Measure index() wall time and peak memory."""
    tracemalloc.start()
    t0 = time.perf_counter()
    scorer.index(corpus_tokens, show_progress=False)
    wall_ms = (time.perf_counter() - t0) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "wall_ms": wall_ms,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
    }


def measure_retrieve(
    scorer: BayesianBM25Scorer,
    queries: list[list[str]],
    k: int = 10,
) -> dict:
    """Measure retrieve(k) wall time per query."""
    times = []
    for qtokens in queries:
        t0 = time.perf_counter()
        scorer.retrieve([qtokens], k=k)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def measure_get_probabilities(
    scorer: BayesianBM25Scorer,
    queries: list[list[str]],
) -> dict:
    """Measure get_probabilities() wall time per query."""
    times = []
    for qtokens in queries:
        t0 = time.perf_counter()
        scorer.get_probabilities(qtokens)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def measure_compute_tf_batch(
    scorer: BayesianBM25Scorer,
    queries: list[list[str]],
) -> dict:
    """Measure _compute_tf_batch() wall time per query."""
    times = []
    for qtokens in queries:
        scores = scorer._bm25.get_scores(qtokens)
        nonzero_mask = scores > 0
        if not np.any(nonzero_mask):
            continue
        nonzero_indices = np.where(nonzero_mask)[0]

        t0 = time.perf_counter()
        scorer._compute_tf_batch(nonzero_indices, qtokens)
        times.append((time.perf_counter() - t0) * 1000)

    if not times:
        return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0}

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def run_scalability_for_size(
    n_docs: int,
    rng: np.random.Generator,
) -> dict:
    """Run all scalability measurements for a given corpus size."""
    vocab_size = min(10000, n_docs)
    avg_doc_len = 100

    print(f"\n  Generating {n_docs:,} documents (vocab={vocab_size:,})...")
    corpus_tokens, queries = generate_synthetic_corpus(
        n_docs, vocab_size, avg_doc_len, rng,
    )
    print(f"  Generated {len(queries)} queries")

    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")

    # Index
    print("  Indexing...", end="", flush=True)
    index_result = measure_index(scorer, corpus_tokens)
    print(f" {index_result['wall_ms']:.0f} ms, "
          f"{index_result['peak_memory_mb']:.1f} MB peak")

    # Retrieve
    print("  Retrieving...", end="", flush=True)
    retrieve_result = measure_retrieve(scorer, queries, k=10)
    print(f" {retrieve_result['mean_ms']:.2f} ms/query (mean)")

    # Get probabilities
    print("  get_probabilities...", end="", flush=True)
    probs_result = measure_get_probabilities(scorer, queries)
    print(f" {probs_result['mean_ms']:.2f} ms/query (mean)")

    # Compute TF batch
    print("  _compute_tf_batch...", end="", flush=True)
    tf_result = measure_compute_tf_batch(scorer, queries)
    print(f" {tf_result['mean_ms']:.2f} ms/query (mean)")

    return {
        "n_docs": n_docs,
        "vocab_size": vocab_size,
        "n_queries": len(queries),
        "index": index_result,
        "retrieve_k10": retrieve_result,
        "get_probabilities": probs_result,
        "compute_tf_batch": tf_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalability benchmark for Bayesian BM25"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--large", action="store_true",
        help="Include 1M-document corpus (slow)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Bayesian BM25 Scalability Benchmark")
    print("=" * 72)

    rng = np.random.default_rng(42)

    corpus_sizes = [1_000, 10_000, 100_000]
    if args.large:
        corpus_sizes.append(1_000_000)

    all_results = []
    for n_docs in corpus_sizes:
        print(f"\n{'--' * 36}")
        print(f"Corpus size: {n_docs:,}")
        print(f"{'--' * 36}")
        result = run_scalability_for_size(n_docs, rng)
        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 72}")
    print("Summary")
    print(f"{'=' * 72}")
    print(f"\n  {'Docs':>10}  {'Index ms':>10}  {'Mem MB':>8}  "
          f"{'Retr ms':>8}  {'Probs ms':>9}  {'TF ms':>8}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 9}  {'-' * 8}")

    for r in all_results:
        print(
            f"  {r['n_docs']:>10,}  "
            f"{r['index']['wall_ms']:>10.0f}  "
            f"{r['index']['peak_memory_mb']:>8.1f}  "
            f"{r['retrieve_k10']['mean_ms']:>8.2f}  "
            f"{r['get_probabilities']['mean_ms']:>9.2f}  "
            f"{r['compute_tf_batch']['mean_ms']:>8.2f}"
        )

    if args.output:
        output = {
            "benchmark": "scalability",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
