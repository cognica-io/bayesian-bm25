#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""BEIR hybrid search benchmark: Bayesian BM25 fusion vs baselines.

Follows the BEIR evaluation protocol: retrieve top-R candidates from each
signal, fuse the union, evaluate with pytrec_eval.

Tokenization uses bm25s.tokenize with Snowball English stemmer and stop
word removal, matching the BEIR official BM25 baseline (Lucene
EnglishAnalyzer with Porter stemmer).

Methods compared:

  1. BM25         -- Sparse retrieval via bm25s
  2. Dense        -- Cosine similarity via sentence-transformers
  3. Convex       -- w * dense_norm + (1-w) * bm25_norm, w=0.5
  4. RRF          -- sum(1/(k + rank_i)), k=60
  5. Bayesian-OR      -- Bayesian BM25 probs + probabilistic OR
  6. Bayesian-LogOdds -- Per-query calibrated log-odds fusion

Dependencies:
    pip install bayesian-bm25[scorer] sentence-transformers pytrec-eval-0.5
    pip install PyStemmer

Usage:
    python hybrid_beir.py -d ../../beir
    python hybrid_beir.py -d ../../beir --datasets arguana scifact
    python hybrid_beir.py -d ../../beir --model all-MiniLM-L6-v2 -o results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bm25s
import Stemmer

from bayesian_bm25.fusion import balanced_log_odds_fusion, cosine_to_probability
from bayesian_bm25.probability import logit
from bayesian_bm25.scorer import BayesianBM25Scorer

STEMMER = Stemmer.Stemmer("english")


# ---------------------------------------------------------------------------
# BEIR data loading
# ---------------------------------------------------------------------------

def load_beir_dataset(dataset_dir: str) -> dict:
    """Load a BEIR dataset from local jsonl/tsv files.

    Expected layout:
        dataset_dir/corpus.jsonl
        dataset_dir/queries.jsonl
        dataset_dir/qrels/test.tsv
    """
    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", "test.tsv")

    corpus_ids = []
    corpus_texts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus_ids.append(str(doc["_id"]))
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                corpus_texts.append(title + " " + text)
            else:
                corpus_texts.append(text)

    query_ids = []
    query_texts = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            query_ids.append(str(q["_id"]))
            query_texts.append(q["text"])

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    filtered_qids = []
    filtered_qtexts = []
    qid_set = set(qrels.keys())
    for qid, qtext in zip(query_ids, query_texts):
        if qid in qid_set:
            filtered_qids.append(qid)
            filtered_qtexts.append(qtext)

    return {
        "corpus_ids": corpus_ids,
        "corpus_texts": corpus_texts,
        "query_ids": filtered_qids,
        "query_texts": filtered_qtexts,
        "qrels": qrels,
    }


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_texts(texts: list[str]) -> list[list[str]]:
    """Tokenize with Snowball English stemmer + stop word removal."""
    return bm25s.tokenize(
        texts,
        stemmer=STEMMER,
        stopwords="english",
        return_ids=False,
        show_progress=False,
    )


# ---------------------------------------------------------------------------
# Dense encoding with embedding cache
# ---------------------------------------------------------------------------

def encode_dense(
    texts: list[str],
    model_name: str,
    cache_path: str | None = None,
    cache_key: str = "embeddings",
    batch_size: int = 128,
) -> np.ndarray:
    """Encode texts with sentence-transformers, with .npz caching.

    Returns L2-normalized embeddings of shape (len(texts), dim).
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        if cache_key in data:
            emb = data[cache_key]
            if emb.shape[0] == len(texts):
                return emb

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.exists(cache_path):
            existing = dict(np.load(cache_path))
            existing[cache_key] = emb
            np.savez(cache_path, **existing)
        else:
            np.savez(cache_path, **{cache_key: emb})

    return emb


# ---------------------------------------------------------------------------
# Fusion methods (operate on candidate-level arrays)
# ---------------------------------------------------------------------------

def fusion_convex(
    bm25_scores: np.ndarray,
    dense_sim: np.ndarray,
    weight: float = 0.5,
) -> np.ndarray:
    """Convex combination with min-max normalization within candidate set."""
    bm25_norm = _min_max_normalize(bm25_scores)
    dense_norm = _min_max_normalize(dense_sim)
    return weight * dense_norm + (1.0 - weight) * bm25_norm


def fusion_rrf(
    bm25_ranks: np.ndarray,
    dense_ranks: np.ndarray,
    k: int = 60,
) -> np.ndarray:
    """RRF with retrieval ranks. Rank 0 = not retrieved (no contribution)."""
    scores = np.zeros(len(bm25_ranks), dtype=np.float64)
    active_bm25 = bm25_ranks > 0
    active_dense = dense_ranks > 0
    scores[active_bm25] += 1.0 / (k + bm25_ranks[active_bm25])
    scores[active_dense] += 1.0 / (k + dense_ranks[active_dense])
    return scores


def fusion_bayesian_bm25_or(
    bayesian_probs: np.ndarray,
    dense_sim: np.ndarray,
) -> np.ndarray:
    """Probabilistic OR: 1 - (1-p_sparse)*(1-p_dense)."""
    dense_probs = np.asarray(cosine_to_probability(dense_sim), dtype=np.float64)
    sparse = np.asarray(bayesian_probs, dtype=np.float64)
    return 1.0 - (1.0 - sparse) * (1.0 - dense_probs)


def fusion_bayesian_bm25_logodds(
    bayesian_probs: np.ndarray,
    dense_sim: np.ndarray,
    dense_median: float,
    dense_alpha: float,
) -> np.ndarray:
    """Per-query calibrated log-odds fusion.

    Documents with only one signal active use single-signal logit
    (not penalized for missing the other).
    """
    n_signals = 2
    alpha = 0.5
    scale = n_signals ** alpha
    w_dense = 0.5
    w_sparse = 0.5
    epsilon = 1e-10

    logit_d = np.clip(dense_alpha * (dense_sim - dense_median), -500.0, 500.0)

    has_sparse = bayesian_probs > epsilon
    p_s = np.clip(bayesian_probs, epsilon, 1.0 - epsilon)
    logit_s = np.asarray(logit(p_s), dtype=np.float64)

    l_bar = w_dense * logit_d + w_sparse * logit_s
    scores_both = l_bar * scale
    scores_dense_only = logit_d * w_dense

    return np.where(has_sparse, scores_both, scores_dense_only)


def fusion_logodds_local(
    bm25_scores: np.ndarray,
    dense_sim: np.ndarray,
    bm25_median: float,
    bm25_alpha: float,
    dense_median: float,
    dense_alpha: float,
) -> np.ndarray:
    """Symmetric log-odds fusion: both signals calibrated at candidate level.

    Instead of converting BM25 -> Bayesian BM25 probability -> logit (which
    saturates), calibrate raw BM25 scores the same way as dense:
    logit = alpha*(s - median).
    """
    n_signals = 2
    alpha = 0.5
    scale = n_signals ** alpha
    w_dense = 0.5
    w_sparse = 0.5

    logit_d = np.clip(dense_alpha * (dense_sim - dense_median), -500.0, 500.0)
    logit_s = np.clip(bm25_alpha * (bm25_scores - bm25_median), -500.0, 500.0)

    has_sparse = bm25_scores > 0

    l_bar = w_dense * logit_d + w_sparse * logit_s
    scores_both = l_bar * scale
    scores_dense_only = logit_d * w_dense

    return np.where(has_sparse, scores_both, scores_dense_only)


def fusion_bayesian_bm25_logodds_br(
    bayesian_probs_br: np.ndarray,
    dense_sim: np.ndarray,
    dense_median: float,
    dense_alpha: float,
) -> np.ndarray:
    """Bayesian BM25 log-odds with base rate prior (dampens probability saturation)."""
    n_signals = 2
    alpha = 0.5
    scale = n_signals ** alpha
    w_dense = 0.5
    w_sparse = 0.5
    epsilon = 1e-10

    logit_d = np.clip(dense_alpha * (dense_sim - dense_median), -500.0, 500.0)

    has_sparse = bayesian_probs_br > epsilon
    p_s = np.clip(bayesian_probs_br, epsilon, 1.0 - epsilon)
    logit_s = np.asarray(logit(p_s), dtype=np.float64)

    l_bar = w_dense * logit_d + w_sparse * logit_s
    scores_both = l_bar * scale
    scores_dense_only = logit_d * w_dense

    return np.where(has_sparse, scores_both, scores_dense_only)


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _compute_dense_calibration(dense_sim: np.ndarray) -> tuple[float, float]:
    """Per-query calibration: beta=median, alpha_eff=1/std of positive scores."""
    positive = dense_sim[dense_sim > 0]
    if len(positive) == 0:
        return 0.0, 1.0
    median = float(np.median(positive))
    std = float(np.std(positive))
    alpha_eff = 1.0 / std if std > 0 else 1.0
    return median, alpha_eff


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pytrec(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, float]:
    import pytrec_eval

    measures = {f"ndcg_cut_{k}", f"map_cut_{k}", f"recall_{k}"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    results = evaluator.evaluate(run)

    ndcg_scores = []
    map_scores = []
    recall_scores = []
    for qid in results:
        ndcg_scores.append(results[qid][f"ndcg_cut_{k}"])
        map_scores.append(results[qid][f"map_cut_{k}"])
        recall_scores.append(results[qid][f"recall_{k}"])

    return {
        f"NDCG@{k}": float(np.mean(ndcg_scores)),
        f"MAP@{k}": float(np.mean(map_scores)),
        f"Recall@{k}": float(np.mean(recall_scores)),
    }


# ---------------------------------------------------------------------------
# Main benchmark pipeline
# ---------------------------------------------------------------------------

METHODS = [
    "BM25", "Dense", "Convex", "RRF",
    "Bayesian-OR", "Bayesian-LogOdds", "LO-Local", "Bayesian-LO-BR", "Bayesian-Balanced",
]


def run_dataset(
    dataset_dir: str,
    dataset_name: str,
    model_name: str,
    k: int = 10,
    retrieve_k: int = 1000,
) -> dict[str, dict[str, float]]:
    """Run all fusion methods on a single BEIR dataset.

    Follows retrieve-then-evaluate: retrieve top-R from each signal,
    fuse union candidates, evaluate top-k.
    """
    print(f"\n{'=' * 70}")
    print(f"  {dataset_name}")
    print(f"{'=' * 70}")

    # 1. Load data
    t0 = time.time()
    data = load_beir_dataset(dataset_dir)
    corpus_ids = data["corpus_ids"]
    corpus_texts = data["corpus_texts"]
    query_ids = data["query_ids"]
    query_texts = data["query_texts"]
    qrels = data["qrels"]
    n_docs = len(corpus_ids)
    n_queries = len(query_ids)
    print(f"  Loaded: {n_docs} docs, {n_queries} queries, "
          f"{sum(len(v) for v in qrels.values())} qrels "
          f"({time.time() - t0:.1f}s)")

    # 2. Tokenize with stemmer
    t0 = time.time()
    corpus_tokens = tokenize_texts(corpus_texts)
    query_tokens_list = tokenize_texts(query_texts)
    print(f"  Tokenized with Snowball English stemmer ({time.time() - t0:.1f}s)")

    # 3. Build BM25 index (two scorers: default + base_rate="auto")
    t0 = time.time()
    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer.index(corpus_tokens, show_progress=False)
    scorer_br = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate="auto")
    scorer_br.index(corpus_tokens, show_progress=False)
    print(f"  BM25 indexed (base_rate={scorer_br.base_rate:.6f}) ({time.time() - t0:.1f}s)")

    # 4. Encode dense embeddings (with cache)
    cache_dir = os.path.join(dataset_dir, "embedding_cache")
    safe_model = model_name.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe_model}.npz")

    t0 = time.time()
    corpus_emb = encode_dense(
        corpus_texts, model_name, cache_path, cache_key="corpus",
    )
    query_emb = encode_dense(
        query_texts, model_name, cache_path, cache_key="queries",
    )
    print(f"  Dense encoded: corpus {corpus_emb.shape}, queries {query_emb.shape} "
          f"({time.time() - t0:.1f}s)")

    # 5. Retrieve-then-evaluate
    t0 = time.time()
    qrels_pytrec: dict[str, dict[str, int]] = {}
    for qid, rels in qrels.items():
        qrels_pytrec[qid] = {did: score for did, score in rels.items()}

    runs: dict[str, dict[str, dict[str, float]]] = {m: {} for m in METHODS}

    # Pre-allocate rank arrays (reused per query, reset after each)
    bm25_rank_full = np.zeros(n_docs, dtype=np.float64)
    dense_rank_full = np.zeros(n_docs, dtype=np.float64)

    effective_R = min(retrieve_k, n_docs)

    for q_idx in range(n_queries):
        qid = query_ids[q_idx]
        qtokens = query_tokens_list[q_idx]

        # -- Full-array scores (fast, needed to find top-R) --
        raw_bm25 = scorer._bm25.get_scores(qtokens)
        dense_sim = (query_emb[q_idx] @ corpus_emb.T).astype(np.float64)

        # -- Retrieve top-R from each signal --
        bm25_topR = np.argsort(-raw_bm25)[:effective_R]
        dense_topR = np.argsort(-dense_sim)[:effective_R]

        # BM25 run: top-R BM25 results
        runs["BM25"][qid] = {
            corpus_ids[i]: float(raw_bm25[i]) for i in bm25_topR
        }

        # Dense run: top-R dense results
        runs["Dense"][qid] = {
            corpus_ids[i]: float(dense_sim[i]) for i in dense_topR
        }

        # -- Union candidates for hybrid methods --
        union_set = set(bm25_topR.tolist()) | set(dense_topR.tolist())
        union_idx = np.array(sorted(union_set))

        cand_bm25 = raw_bm25[union_idx]
        cand_dense = dense_sim[union_idx]

        # Bayesian BM25 probs for union candidates only (fast: only TF for candidates)
        cand_bayesian_probs = np.zeros(len(union_idx), dtype=np.float64)
        active = cand_bm25 > 0
        if np.any(active):
            active_doc_ids = union_idx[active]
            active_scores = cand_bm25[active]
            doc_len_ratios = scorer._doc_lengths[active_doc_ids] / scorer._avgdl
            tfs = scorer._compute_tf_batch(active_doc_ids, qtokens)
            cand_bayesian_probs[active] = scorer._transform.score_to_probability(
                active_scores, tfs, doc_len_ratios,
            )

        # Bayesian BM25 probs with base rate prior for union candidates
        cand_bayesian_probs_br = np.zeros(len(union_idx), dtype=np.float64)
        if np.any(active):
            cand_bayesian_probs_br[active] = scorer_br._transform.score_to_probability(
                active_scores, tfs, doc_len_ratios,
            )

        # Calibration from candidate set
        dense_median, dense_alpha = _compute_dense_calibration(cand_dense)
        bm25_median, bm25_alpha = _compute_dense_calibration(cand_bm25)

        # Retrieval ranks for RRF (rank 0 = not retrieved)
        bm25_rank_full[bm25_topR] = np.arange(1, len(bm25_topR) + 1, dtype=np.float64)
        dense_rank_full[dense_topR] = np.arange(1, len(dense_topR) + 1, dtype=np.float64)
        cand_bm25_rank = bm25_rank_full[union_idx]
        cand_dense_rank = dense_rank_full[union_idx]
        bm25_rank_full[bm25_topR] = 0
        dense_rank_full[dense_topR] = 0

        # Fusion
        hybrid_scores = {
            "Convex": fusion_convex(cand_bm25, cand_dense),
            "RRF": fusion_rrf(cand_bm25_rank, cand_dense_rank),
            "Bayesian-OR": fusion_bayesian_bm25_or(
                cand_bayesian_probs, cand_dense,
            ),
            "Bayesian-LogOdds": fusion_bayesian_bm25_logodds(
                cand_bayesian_probs, cand_dense, dense_median, dense_alpha,
            ),
            "LO-Local": fusion_logodds_local(
                cand_bm25, cand_dense,
                bm25_median, bm25_alpha, dense_median, dense_alpha,
            ),
            "Bayesian-LO-BR": fusion_bayesian_bm25_logodds_br(
                cand_bayesian_probs_br, cand_dense, dense_median, dense_alpha,
            ),
            "Bayesian-Balanced": balanced_log_odds_fusion(
                cand_bayesian_probs_br, cand_dense,
            ),
        }

        for method_name, scores in hybrid_scores.items():
            runs[method_name][qid] = {
                corpus_ids[union_idx[j]]: float(scores[j])
                for j in range(len(union_idx))
            }

    print(f"  Scored {n_queries} queries x {len(METHODS)} methods, "
          f"R={effective_R} ({time.time() - t0:.1f}s)")

    # 6. Evaluate
    results: dict[str, dict[str, float]] = {}
    for method_name in METHODS:
        results[method_name] = evaluate_pytrec(qrels_pytrec, runs[method_name], k=k)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(
    all_results: dict[str, dict[str, dict[str, float]]],
    k: int = 10,
) -> None:
    datasets = list(all_results.keys())
    metrics = [f"NDCG@{k}", f"MAP@{k}", f"Recall@{k}"]

    col_w = max(len(m) for m in METHODS) + 2

    for metric in metrics:
        line_w = 14 + len(METHODS) * (2 + col_w)
        print(f"\n{metric} Results")
        print("=" * line_w)

        header = f"{'Dataset':<14}"
        for method in METHODS:
            header += f"  {method:>{col_w}}"
        print(header)
        print("-" * line_w)

        averages: dict[str, list[float]] = {m: [] for m in METHODS}
        for ds_name in datasets:
            row = f"{ds_name:<14}"
            for method in METHODS:
                val = all_results[ds_name][method][metric] * 100
                row += f"  {val:>{col_w - 1}.2f}%"
                averages[method].append(val)
            print(row)

        print("-" * line_w)
        row = f"{'Average':<14}"
        for method in METHODS:
            avg = float(np.mean(averages[method]))
            row += f"  {avg:>{col_w - 1}.2f}%"
        print(row)

    print(f"\nDelta vs BM25 (NDCG@{k}):")
    bm25_metric = f"NDCG@{k}"
    for method in METHODS[1:]:
        deltas = []
        for ds_name in datasets:
            bm25_val = all_results[ds_name]["BM25"][bm25_metric]
            method_val = all_results[ds_name][method][bm25_metric]
            deltas.append((method_val - bm25_val) * 100)
        avg_delta = float(np.mean(deltas))
        sign = "+" if avg_delta >= 0 else ""
        print(f"  {method:<14}  avg: {sign}{avg_delta:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEIR hybrid search benchmark for Bayesian BM25"
    )
    parser.add_argument(
        "-d", "--beir-dir", required=True,
        help="Path to BEIR data directory",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"],
        help="BEIR datasets to benchmark (default: all 5)",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=10,
        help="Evaluation depth (default: 10)",
    )
    parser.add_argument(
        "-R", "--retrieve-k", type=int, default=1000,
        help="Retrieval depth per signal (default: 1000)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  BEIR Hybrid Search Benchmark -- Bayesian BM25")
    print(f"  Model: {args.model}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  k={args.top_k}, R={args.retrieve_k}")
    print("=" * 70)

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for ds_name in args.datasets:
        ds_dir = os.path.join(args.beir_dir, ds_name)
        if not os.path.isdir(ds_dir):
            print(f"\nWARNING: Dataset directory not found: {ds_dir}, skipping")
            continue
        all_results[ds_name] = run_dataset(
            ds_dir, ds_name, args.model,
            k=args.top_k, retrieve_k=args.retrieve_k,
        )

    print_results_table(all_results, k=args.top_k)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
