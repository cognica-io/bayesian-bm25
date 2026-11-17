#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Additional jointly fitted slopes in tail units, separate from paper rows."""

from __future__ import annotations

import numpy as np

from benchmarks.evidence_protocol import TAIL_DEPTH
from benchmarks.prepared_dataset import PreparedDataset
from benchmarks.query_signals import query_signals
from benchmarks.ranking_eval import evaluate_run, mean_metrics, ranked_run
from benchmarks.reevaluation import _top_k, fit_kappas, lexical_scale, tail_units, training_sample


def run_tail_fit(
    prepared: PreparedDataset,
    *,
    k: int = 10,
    retrieve_k: int = 1000,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Replay the paper's CV samples but fit the two tail-unit slopes jointly."""
    rng = np.random.default_rng(seed)
    folds = np.empty(prepared.n_queries, dtype=int)
    folds[rng.permutation(prepared.n_queries)] = np.arange(prepared.n_queries) % n_folds
    features, labels, weights = [], [], []
    for qi in range(prepared.n_queries):
        sig = query_signals(prepared, qi, retrieve_k)
        lexical, dense = tail_units(
            sig,
            TAIL_DEPTH,
            fallback_lexical_scale=lexical_scale(sig, 1.0 / prepared.corpus_alpha),
        )
        # Replay the paper run's preceding sampled-offset diagnostic.
        matched = np.flatnonzero(sig.matched)
        if len(matched):
            top = matched[_top_k(sig.bm25[matched], retrieve_k)]
            if len(top) < len(sig.bm25):
                rest = np.flatnonzero(~np.isin(np.arange(len(sig.bm25)), top))
                rng.choice(rest, size=min(1000, len(rest)), replace=False)
        positions, multiplicities = training_sample(sig, rng)
        features.append(np.column_stack([lexical[positions], dense[positions]]))
        labels.append(sig.labels[positions])
        weights.append(multiplicities)
    parameters = []
    for fold in range(n_folds):
        train = np.flatnonzero(folds != fold)
        parameters.append(
            fit_kappas(
                [features[i] for i in train],
                [labels[i] for i in train],
                [weights[i] for i in train],
            )
        )
    run = {}
    corpus_ids = np.asarray(prepared.corpus_ids)
    for qi in range(prepared.n_queries):
        sig = query_signals(prepared, qi, retrieve_k)
        lexical, dense = tail_units(
            sig,
            TAIL_DEPTH,
            fallback_lexical_scale=lexical_scale(sig, 1.0 / prepared.corpus_alpha),
        )
        kl, kv = parameters[int(folds[qi])]
        scores = kl * lexical[sig.candidates] + kv * dense[sig.candidates]
        run[sig.qid] = ranked_run(corpus_ids[sig.doc_index[sig.candidates]], scores)
    metrics = evaluate_run(prepared.qrels, run, k=k)
    return {
        "experiment": "additional_joint_tail_slopes",
        "identity": prepared.identity,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "query_ids": prepared.query_ids,
        "folds": folds.tolist(),
        "parameters": [values.tolist() for values in parameters],
        "ranking": {
            "mean": mean_metrics(metrics),
            "per_query": {
                metric: [values[qid] for qid in prepared.query_ids]
                for metric, values in metrics.items()
            },
        },
    }
