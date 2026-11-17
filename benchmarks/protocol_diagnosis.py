#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Decomposition of the two defects of the first evaluation.

The first version of the hybrid benchmark differed from the corrected
protocol in two ways:

1. Identical ids.  The document whose id equals the query's was kept as
   a candidate.  On ArguAna, whose queries are arguments drawn from the
   corpus, that document is the query's own text: it ranks first for
   every signal and is not relevant.
2. Probability ranking.  Fused log-odds were converted to probabilities
   (through a sigmoid, or after clamping to [1e-10, 1 - 1e-10]) before
   ranking, and the evaluator compared them in single precision.  Once
   the evidence exceeds what a probability can resolve, documents tie
   and their order is whatever the tie-breaking rule gives.

This script evaluates a few configurations in all four combinations of
the two factors, so that each defect's contribution can be read off:

    identical ids kept      x  ranked on probability   (first version)
    identical ids kept      x  ranked on log-odds
    identical ids excluded  x  ranked on probability
    identical ids excluded  x  ranked on log-odds      (corrected)

It also counts, for every configuration, how many of the 100 best
candidates carry a score that is exactly tied with another one under
probability ranking.

In all four cells the lexical result list holds only documents with a
non-zero BM25 score; the first version also admitted zero-score
documents, which matters only when fewer than the retrieval depth
match.

Usage:
    python -m benchmarks.protocol_diagnosis -d <beir-dir> --datasets arguana
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from bayesian_bm25.fusion import balanced_log_odds_fusion, cosine_to_probability
from bayesian_bm25.probability import BayesianProbabilityTransform, sigmoid
from benchmarks.hybrid_beir import (
    UNMATCHED_LOG_ODDS,
    _compute_dense_calibration,
    fusion_bayesian_bm25_logodds,
    fusion_bayesian_bm25_or,
    fusion_convex,
    fusion_logodds_local,
    fusion_rrf,
)
from benchmarks.prepared_dataset import PreparedDataset, prepare_dataset
from benchmarks.query_signals import query_signals
from benchmarks.ranking_eval import evaluate_run, mean_metrics, ranked_run, raw_run
from benchmarks.reevaluation import document_prior_features

BASELINES = ("BM25", "Dense", "RRF", "Convex")
BAYESIAN = (
    "Standardized (Local)",
    "Range-normalized (Balanced)",
    "Corpus-level conjunction",
    "Probabilistic OR",
)


def _tied_in_head(scores: np.ndarray, depth: int = 100) -> int:
    """Number of the ``depth`` best scores that are not unique."""
    head = np.sort(scores)[::-1][:depth]
    return len(head) - len(np.unique(head))


def evaluate_protocol(
    prepared: PreparedDataset,
    *,
    rank_on_probability: bool,
    k: int = 10,
    retrieve_k: int = 1000,
) -> tuple[dict[str, float], dict[str, float]]:
    """NDCG@k of every configuration under one ranking mode.

    Returns the mean NDCG@k per configuration and, for the Bayesian
    configurations, the mean number of tied scores among the 100 best
    candidates.
    """
    corpus_ids = np.asarray(prepared.corpus_ids)
    transform = prepared.scorer._transform
    # The range-normalized fusion of the first version starts from the
    # lexical posterior with the percentile base rate.
    transform_br = BayesianProbabilityTransform(
        alpha=transform.alpha, beta=transform.beta,
        base_rate=prepared.corpus_base_rate,
    )
    runs: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in BASELINES + BAYESIAN
    }
    ties: dict[str, list[int]] = {name: [] for name in BAYESIAN}

    def add(name: str, qid: str, doc_ids: np.ndarray, scores: np.ndarray) -> None:
        if rank_on_probability and name in BAYESIAN:
            # First version: raw scores go to the evaluator, which
            # compares them in single precision.
            runs[name][qid] = raw_run(doc_ids, scores)
            ties[name].append(_tied_in_head(scores.astype(np.float32)))
        else:
            runs[name][qid] = ranked_run(doc_ids, scores)
            if name in BAYESIAN:
                ties[name].append(_tied_in_head(scores))

    for q_idx in range(prepared.n_queries):
        sig = query_signals(prepared, q_idx, retrieve_k)
        cand = sig.candidates
        doc_ids = corpus_ids[sig.doc_index[cand]]
        s = sig.bm25[cand]
        c = sig.dense[cand]
        matched = s > 0.0
        in_bm25 = sig.bm25_rank > 0
        in_dense = sig.dense_rank > 0

        add("BM25", sig.qid, doc_ids[in_bm25], s[in_bm25])
        add("Dense", sig.qid, doc_ids[in_dense], c[in_dense])
        add(
            "RRF", sig.qid, doc_ids,
            fusion_rrf(sig.bm25_rank.astype(np.float64), sig.dense_rank.astype(np.float64)),
        )
        add("Convex", sig.qid, doc_ids, fusion_convex(s, c))

        tfs, doc_len_ratios = document_prior_features(prepared, sig, cand)
        dense_median, dense_alpha = _compute_dense_calibration(c)
        bm25_median, bm25_alpha = _compute_dense_calibration(s)

        # Lexical posterior of the first version: corpus-level slope and
        # midpoint with the composite document prior.
        log_odds = np.full(len(cand), -np.inf, dtype=np.float64)
        probabilities = np.zeros(len(cand), dtype=np.float64)
        log_odds_br = np.full(len(cand), UNMATCHED_LOG_ODDS, dtype=np.float64)
        probabilities_br = np.zeros(len(cand), dtype=np.float64)
        if np.any(matched):
            features = (s[matched], tfs[matched], doc_len_ratios[matched])
            log_odds[matched] = transform.score_to_log_odds(*features)
            probabilities[matched] = transform.score_to_probability(*features)
            log_odds_br[matched] = transform_br.score_to_log_odds(*features)
            probabilities_br[matched] = transform_br.score_to_probability(*features)

        local = fusion_logodds_local(s, c, bm25_median, bm25_alpha, dense_median, dense_alpha)
        conjunction = fusion_bayesian_bm25_logodds(
            s, transform.alpha, transform.beta, tfs, doc_len_ratios,
            c, dense_median, dense_alpha,
        )

        if rank_on_probability:
            dense_probabilities = np.asarray(cosine_to_probability(c), dtype=np.float64)
            add("Standardized (Local)", sig.qid, doc_ids, np.asarray(sigmoid(local)))
            add(
                "Corpus-level conjunction", sig.qid, doc_ids,
                np.asarray(sigmoid(conjunction)),
            )
            add(
                "Probabilistic OR", sig.qid, doc_ids,
                1.0 - (1.0 - probabilities) * (1.0 - dense_probabilities),
            )
            add(
                "Range-normalized (Balanced)", sig.qid, doc_ids,
                np.asarray(balanced_log_odds_fusion(probabilities_br, c), dtype=np.float64),
            )
        else:
            add("Standardized (Local)", sig.qid, doc_ids, local)
            add("Corpus-level conjunction", sig.qid, doc_ids, conjunction)
            add(
                "Probabilistic OR", sig.qid, doc_ids,
                fusion_bayesian_bm25_or(log_odds, c),
            )
            add(
                "Range-normalized (Balanced)", sig.qid, doc_ids,
                np.asarray(
                    balanced_log_odds_fusion(log_odds_br, c, from_log_odds=True),
                    dtype=np.float64,
                ),
            )

    ndcg = {
        name: mean_metrics(evaluate_run(prepared.qrels, run, k=k))[f"NDCG@{k}"]
        for name, run in runs.items()
    }
    tied = {name: float(np.mean(values)) for name, values in ties.items()}
    return ndcg, tied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose the first evaluation's protocol error and saturation",
    )
    parser.add_argument("-d", "--beir-dir", required=True, help="Path to BEIR data directory")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"],
        help="BEIR datasets to evaluate (default: all 5)",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Dense encoder")
    parser.add_argument("--identity", choices=["id", "text", "both"], default="both")
    parser.add_argument("--cache-dir", default=None, help="Embedding cache directory")
    parser.add_argument("-o", "--output", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.beir_dir, ".cache", "embeddings")
    all_results: dict[str, dict] = {}
    identities = ("id", "text") if args.identity == "both" else (args.identity,)
    cells = [
        (identity, ranking)
        for identity in identities for ranking in ("probability", "log-odds")
    ]

    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        results: dict[str, dict] = {}
        metadata = {}
        for identity in identities:
            prepared = prepare_dataset(
                os.path.join(args.beir_dir, name), name, args.model, cache_dir,
                ignore_identical_ids=True, identity=identity,
            )
            metadata[identity] = {
                "n_docs": prepared.n_docs, "n_queries": prepared.n_queries,
                "dataset_fingerprint": prepared.dataset_fingerprint,
                "reference_population": "query_excluded_corpus",
            }
            for ranking in ("probability", "log-odds"):
                ndcg, tied = evaluate_protocol(
                    prepared, rank_on_probability=ranking == "probability",
                )
                results[f"{identity}/{ranking}"] = {"ndcg": ndcg, "tied_top100": tied}
        results["metadata"] = metadata
        all_results[name] = results

        header = "".join(f"{identity + '/' + ranking:>22}" for identity, ranking in cells)
        print(f"\n  NDCG@10 (identity / ranking)\n  {'':<30}{header}")
        for method in BASELINES + BAYESIAN:
            row = "".join(
                f"{100 * results[f'{identity}/{ranking}']['ndcg'][method]:>22.2f}"
                for identity, ranking in cells
            )
            print(f"  {method:<30}{row}")
        print(f"\n  Tied scores among the 100 best candidates (mean)\n  {'':<30}{header}")
        for method in BAYESIAN:
            row = "".join(
                f"{results[f'{identity}/{ranking}']['tied_top100'][method]:>22.1f}"
                for identity, ranking in cells
            )
            print(f"  {method:<30}{row}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=1)

    if args.output:
        print(f"\nresults saved to {args.output}")


if __name__ == "__main__":
    main()
