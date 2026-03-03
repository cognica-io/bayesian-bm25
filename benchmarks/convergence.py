#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Experiment: How many online epochs to match batch fit ECE?

Runs online learning for multiple epochs over the training data
and tracks ECE convergence, comparing raw vs Polyak-averaged
parameters against the batch fit target.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import bm25s
import numpy as np

from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.scorer import BayesianBM25Scorer
from benchmarks.metrics import expected_calibration_error
from benchmarks.utils import load_beir_dataset


def compute_ece(
    alpha: float,
    beta: float,
    bm25_model: bm25s.BM25,
    eval_queries, doc_ids, qrels, corpus_tokens,
) -> float:
    """Compute ECE for given alpha/beta over eval queries."""
    all_probs, all_labels = [], []
    transform = BayesianProbabilityTransform(alpha=alpha, beta=beta)
    scorer_tmp = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer_tmp._bm25 = bm25_model
    scorer_tmp._doc_lengths = np.array([len(d) for d in corpus_tokens], dtype=np.float64)
    scorer_tmp._avgdl = float(np.mean(scorer_tmp._doc_lengths))
    scorer_tmp._corpus_tokens = corpus_tokens
    scorer_tmp._transform = transform

    for qid, qtokens in eval_queries:
        qrel = qrels[qid]
        probs = scorer_tmp.get_probabilities(qtokens)
        nz = probs > 0
        if not np.any(nz):
            continue
        nz_dids = [doc_ids[i] for i, n in enumerate(nz) if n]
        nz_labels = [float(qrel.get(did, 0) >= 1) for did in nz_dids]
        all_probs.extend(probs[nz].tolist())
        all_labels.extend(nz_labels)

    if not all_probs:
        return 1.0
    return expected_calibration_error(np.array(all_probs), np.array(all_labels))


def auto_estimate_params(train_scores: np.ndarray) -> tuple[float, float]:
    """Estimate alpha and beta from training score statistics."""
    score_std = float(np.std(train_scores))
    alpha = 1.0 / score_std if score_std > 0 else 1.0
    beta = float(np.median(train_scores))
    return alpha, beta


def run_online_experiment(
    label: str,
    init_alpha: float,
    init_beta: float,
    lr: float,
    mom: float,
    decay_tau: float,
    max_grad_norm: float,
    avg_decay: float,
    train_batches: list,
    n_samples: int,
    batch_ece: float,
    bm25_model,
    eval_queries,
    doc_ids,
    qrels,
    corpus_tokens,
    rng,
    max_epochs: int = 30,
) -> dict:
    """Run one online learning config, comparing raw vs averaged ECE."""
    print(f"\n{'--' * 30}")
    print(f"Config: {label}")
    print(f"  init: alpha={init_alpha:.4f}, beta={init_beta:.4f}")
    print(f"  lr={lr}, momentum={mom}, decay_tau={decay_tau}, "
          f"max_grad={max_grad_norm}, avg_decay={avg_decay}")
    print(f"{'--' * 30}")

    transform = BayesianProbabilityTransform(alpha=init_alpha, beta=init_beta)
    total_updates = 0

    header = (f"  {'Epoch':>5}  {'Updates':>8}  "
              f"{'Alpha':>7}  {'Beta':>7}  {'ECE_raw':>8}  "
              f"{'AvgAlpha':>8}  {'AvgBeta':>8}  {'ECE_avg':>8}  {'vs Batch':>8}")
    print(header)

    ece_raw = compute_ece(
        transform.alpha, transform.beta,
        bm25_model, eval_queries, doc_ids, qrels, corpus_tokens,
    )
    print(f"  {0:>5}  {0:>8}  "
          f"{transform.alpha:>7.4f}  {transform.beta:>7.4f}  {ece_raw:>8.4f}  "
          f"{transform.averaged_alpha:>8.4f}  {transform.averaged_beta:>8.4f}  "
          f"{ece_raw:>8.4f}  {ece_raw - batch_ece:>+8.4f}")

    raw_converged = None
    avg_converged = None

    for epoch in range(1, max_epochs + 1):
        batch_order = list(range(len(train_batches)))
        rng.shuffle(batch_order)

        for idx in batch_order:
            scores_batch, labels_batch = train_batches[idx]
            transform.update(
                scores_batch, labels_batch,
                learning_rate=lr, momentum=mom,
                decay_tau=decay_tau, max_grad_norm=max_grad_norm,
                avg_decay=avg_decay,
            )
            total_updates += 1

        ece_raw = compute_ece(
            transform.alpha, transform.beta,
            bm25_model, eval_queries, doc_ids, qrels, corpus_tokens,
        )
        ece_avg = compute_ece(
            transform.averaged_alpha, transform.averaged_beta,
            bm25_model, eval_queries, doc_ids, qrels, corpus_tokens,
        )

        if abs(ece_raw - batch_ece) < 0.01 and raw_converged is None:
            raw_converged = epoch
        if abs(ece_avg - batch_ece) < 0.01 and avg_converged is None:
            avg_converged = epoch

        marker = ""
        if avg_converged == epoch:
            marker = " <-- avg matched"
        elif raw_converged == epoch:
            marker = " <-- raw matched"

        if epoch <= 5 or epoch % 5 == 0 or marker:
            print(f"  {epoch:>5}  {total_updates:>8}  "
                  f"{transform.alpha:>7.4f}  {transform.beta:>7.4f}  {ece_raw:>8.4f}  "
                  f"{transform.averaged_alpha:>8.4f}  {transform.averaged_beta:>8.4f}  "
                  f"{ece_avg:>8.4f}  {ece_avg - batch_ece:>+8.4f}{marker}")

    epochs_data = []
    for epoch_num in range(1, max_epochs + 1):
        ece_raw_val = compute_ece(
            transform.alpha, transform.beta,
            bm25_model, eval_queries, doc_ids, qrels, corpus_tokens,
        )
        ece_avg_val = compute_ece(
            transform.averaged_alpha, transform.averaged_beta,
            bm25_model, eval_queries, doc_ids, qrels, corpus_tokens,
        )
        epochs_data.append({
            "epoch": epoch_num,
            "raw_ece": ece_raw_val,
            "avg_ece": ece_avg_val,
        })

    print()
    if raw_converged:
        print(f"  Raw converged at epoch {raw_converged} "
              f"({raw_converged * n_samples:,} samples)")
    else:
        print(f"  Raw did not converge within {max_epochs} epochs")

    if avg_converged:
        print(f"  Avg converged at epoch {avg_converged} "
              f"({avg_converged * n_samples:,} samples)")
    else:
        print(f"  Avg did not converge within {max_epochs} epochs")

    return {
        "label": label,
        "raw_converged_epoch": raw_converged,
        "avg_converged_epoch": avg_converged,
        "epochs": epochs_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Online convergence benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of random seeds for statistical reporting",
    )
    args = parser.parse_args()

    print("Loading NFCorpus...")
    ds = load_beir_dataset("nfcorpus", "test")
    corpus_tokens = ds.corpus_tokens
    doc_ids = ds.doc_ids
    queries = ds.queries
    qrels = ds.qrels

    # Split queries (deterministic split, seed only affects online training)
    split_rng = np.random.default_rng(42)
    all_qids = [qid for qid, _ in queries]
    split_rng.shuffle(all_qids)
    mid = len(all_qids) // 2
    train_qids = set(all_qids[:mid])
    eval_qids = set(all_qids[mid:])

    train_queries = [(qid, qt) for qid, qt in queries if qid in train_qids]
    eval_queries = [(qid, qt) for qid, qt in queries if qid in eval_qids]

    # Index
    bm25_model = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    bm25_model.index(corpus_tokens, show_progress=False)

    # Collect training data
    train_scores, train_labels = [], []
    train_batches = []
    for qid, qtokens in train_queries:
        qrel = qrels[qid]
        scores = bm25_model.get_scores(qtokens)
        nz = scores > 0
        if not np.any(nz):
            continue
        nz_scores = scores[nz]
        nz_dids = [doc_ids[i] for i, n in enumerate(nz) if n]
        nz_labels = np.array([float(qrel.get(did, 0) >= 1) for did in nz_dids])
        train_scores.extend(nz_scores.tolist())
        train_labels.extend(nz_labels.tolist())
        train_batches.append((nz_scores, nz_labels))

    train_scores = np.array(train_scores)
    train_labels = np.array(train_labels)
    n_samples = len(train_scores)
    print(f"Training: {len(train_batches)} queries, {n_samples} samples")
    print(f"Eval: {len(eval_queries)} queries")

    # Auto-estimated initial parameters
    auto_alpha, auto_beta = auto_estimate_params(train_scores)
    print(f"\nAuto-estimated init: alpha={auto_alpha:.4f}, beta={auto_beta:.4f}")

    # --- Batch fit baseline ---
    batch_transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
    batch_transform.fit(train_scores, train_labels, learning_rate=0.05, max_iterations=5000)
    batch_ece = compute_ece(
        batch_transform.alpha, batch_transform.beta,
        bm25_model, eval_queries, doc_ids, qrels, corpus_tokens,
    )
    print(f"Batch fit target: alpha={batch_transform.alpha:.4f}, "
          f"beta={batch_transform.beta:.4f}, ECE={batch_ece:.4f}")

    configs = [
        ("lr=0.10, tau=1000, avg=0.99",
         auto_alpha, auto_beta, 0.10, 0.9, 1000.0, 1.0, 0.99),
        ("lr=0.50, tau=2000, avg=0.99",
         auto_alpha, auto_beta, 0.50, 0.9, 2000.0, 1.0, 0.99),
        ("lr=1.00, tau=2000, avg=0.99",
         auto_alpha, auto_beta, 1.00, 0.9, 2000.0, 1.0, 0.99),
        ("lr=1.00, tau=2000, avg=0.995",
         auto_alpha, auto_beta, 1.00, 0.9, 2000.0, 1.0, 0.995),
    ]

    all_seed_results = []
    for seed_idx in range(args.seeds):
        seed = 42 + seed_idx
        if args.seeds > 1:
            print(f"\n{'#' * 72}")
            print(f"# Seed {seed} ({seed_idx + 1}/{args.seeds})")
            print(f"{'#' * 72}")

        rng = np.random.default_rng(seed)

        common_args = dict(
            train_batches=train_batches,
            n_samples=n_samples,
            batch_ece=batch_ece,
            bm25_model=bm25_model,
            eval_queries=eval_queries,
            doc_ids=doc_ids,
            qrels=qrels,
            corpus_tokens=corpus_tokens,
            rng=rng,
            max_epochs=30,
        )

        # ================================================================
        # Warm start with Polyak averaging (the realistic scenario)
        # ================================================================
        print(f"\n{'=' * 60}")
        print("Warm start + Polyak averaging: raw vs averaged ECE")
        print(f"{'=' * 60}")

        config_results = []
        for label, a, b, lr, mom, tau, mg, ad in configs:
            result = run_online_experiment(
                label, a, b, lr, mom, tau, mg, ad, **common_args,
            )
            config_results.append(result)

        all_seed_results.append({
            "seed": seed,
            "configs": config_results,
        })

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Batch fit: {n_samples:,} samples x 5000 iters "
          f"= {n_samples * 5000:,} gradient steps, ECE = {batch_ece:.4f}")
    print(f"  Online: {len(train_batches)} query batches per epoch, "
          f"{n_samples:,} samples per epoch")

    if args.output:
        results_payload = (
            all_seed_results[0] if args.seeds == 1 else all_seed_results
        )
        output = {
            "benchmark": "convergence",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": args.seeds,
            "results": results_payload,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
