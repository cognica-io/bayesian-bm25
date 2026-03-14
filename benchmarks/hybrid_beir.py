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

  1. BM25             -- Sparse retrieval via bm25s
  2. Dense            -- Cosine similarity via sentence-transformers
  3. Convex           -- w * dense_norm + (1-w) * bm25_norm, w=0.5
  4. RRF              -- sum(1/(k + rank_i)), k=60
  5. Bayesian-OR      -- Bayesian BM25 probs + probabilistic OR
  6. Bayesian-LogOdds -- Per-query calibrated log-odds fusion
  7. Balanced-Mix     -- balanced fusion with mixture base_rate estimation
  8. Balanced-Elbow   -- balanced fusion with elbow base_rate estimation
  9. Gated-ReLU       -- log-odds conjunction with ReLU gating (Theorem 6.5.3)
 10. Gated-Swish      -- log-odds conjunction with Swish gating (Theorem 6.7.4)
 11. Gated-GELU       -- log-odds conjunction with GELU gating (Theorem 6.8.1)
 12. Gated-Swish-B2   -- generalized swish with beta=2.0 (Theorem 6.7.6)
 13. Attention        -- query-dependent attention weights (Paper 2, Section 8)
 14. Attn-NR          -- Attention + logit normalization + 7 features (dense+cross)
 15. Attn-NR-CV       -- 5-fold cross-validated Attn-NR
 16. Multi-Head       -- 4-head attention fusion (Remark 8.6)
 17. MH-NR            -- Multi-head + normalize + rich features (Corollary 8.7.2)
 18. MultiField       -- MultiFieldScorer (title + body), sparse-only
 19. MF-Balanced      -- MultiField probs + dense via balanced_log_odds_fusion

With --tune, additional tuned methods are evaluated:

 20. Bayesian-Tuned     -- BayesianBM25 with tuned alpha, beta, base_rate
 21. Balanced-Tuned     -- balanced_log_odds_fusion with tuned weight
 22. Hybrid-AND-Tuned   -- log-odds conjunction with tuned alpha

Dependencies:
    pip install bayesian-bm25[scorer] sentence-transformers pytrec-eval-0.5
    pip install PyStemmer

Usage:
    python hybrid_beir.py -d ../../beir
    python hybrid_beir.py -d ../../beir --datasets arguana scifact
    python hybrid_beir.py -d ../../beir --model all-MiniLM-L6-v2 -o results.json
    python hybrid_beir.py -d ../../beir --datasets scifact --tune
    python hybrid_beir.py -d ../../beir --download --datasets scifact
    python hybrid_beir.py -d ../../beir --cache-dir /tmp/emb_cache
    python hybrid_beir.py -d ../../beir --no-cache
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import zipfile

import bm25s
import numpy as np
import Stemmer

from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    balanced_log_odds_fusion,
    cosine_to_probability,
    log_odds_conjunction,
)
from bayesian_bm25.metrics import calibration_report
from bayesian_bm25.multi_field import MultiFieldScorer
from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    logit,
)
from bayesian_bm25.scorer import BayesianBM25Scorer

STEMMER = Stemmer.Stemmer("english")

BEIR_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
)


# ---------------------------------------------------------------------------
# BEIR data download
# ---------------------------------------------------------------------------

def download_beir_dataset(dataset_name: str, beir_dir: str) -> str:
    """Download and extract a BEIR dataset from the official CDN.

    Downloads ``{dataset_name}.zip`` from the BEIR public repository and
    extracts it into *beir_dir*.  If the dataset directory already exists,
    this is a no-op.

    Parameters
    ----------
    dataset_name : str
        Name of the BEIR dataset (e.g. "scifact", "arguana").
    beir_dir : str
        Root directory where datasets are stored.  The dataset will be
        extracted to ``{beir_dir}/{dataset_name}/``.

    Returns
    -------
    str
        Path to the extracted dataset directory.
    """
    dataset_dir = os.path.join(beir_dir, dataset_name)
    if os.path.isdir(dataset_dir):
        print(f"  {dataset_name}: already exists at {dataset_dir}")
        return dataset_dir

    url = f"{BEIR_BASE_URL}/{dataset_name}.zip"
    os.makedirs(beir_dir, exist_ok=True)
    zip_path = os.path.join(beir_dir, f"{dataset_name}.zip")

    print(f"  Downloading {dataset_name} from {url} ...")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\r  [{pct:3d}%] {mb_down:.1f} / {mb_total:.1f} MB",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    except urllib.error.HTTPError as exc:
        print(f"\n  ERROR: failed to download {url} ({exc})")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise SystemExit(1) from exc
    print()  # newline after progress

    print(f"  Extracting to {beir_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(beir_dir)
    os.remove(zip_path)

    if not os.path.isdir(dataset_dir):
        print(f"  WARNING: expected {dataset_dir} after extraction, not found")
    else:
        print(f"  Done: {dataset_dir}")

    return dataset_dir


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
    corpus_titles: list[str] = []
    corpus_bodies: list[str] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus_ids.append(str(doc["_id"]))
            title = doc.get("title", "")
            text = doc.get("text", "")
            corpus_titles.append(title)
            corpus_bodies.append(text)
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
    for qid, qtext in zip(query_ids, query_texts, strict=True):
        if qid in qid_set:
            filtered_qids.append(qid)
            filtered_qtexts.append(qtext)

    return {
        "corpus_ids": corpus_ids,
        "corpus_texts": corpus_texts,
        "corpus_titles": corpus_titles,
        "corpus_bodies": corpus_bodies,
        "query_ids": filtered_qids,
        "query_texts": filtered_qtexts,
        "qrels": qrels,
        "all_query_ids": query_ids,
        "all_query_texts": query_texts,
    }


def load_qrels_from_tsv(path: str) -> dict[str, dict[str, int]]:
    """Load qrels from a TSV file (query_id, doc_id, relevance)."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score
    return qrels


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

    When *cache_path* is given, the function first checks for a cached
    array under *cache_key*.  A cache hit requires both the key to exist
    and the row count to match ``len(texts)``.

    Parameters
    ----------
    texts : list of str
        Texts to encode.
    model_name : str
        Sentence-transformers model identifier.
    cache_path : str or None
        Path to an ``.npz`` file for reading/writing cached embeddings.
        Multiple keys (e.g. ``"corpus"`` and ``"queries"``) coexist in
        the same file.  ``None`` disables caching.
    cache_key : str
        Key inside the ``.npz`` archive (default ``"embeddings"``).
    batch_size : int
        Encoding batch size passed to sentence-transformers.

    Returns
    -------
    np.ndarray
        L2-normalized embeddings of shape ``(len(texts), dim)``.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        if cache_key in data:
            emb = data[cache_key]
            if emb.shape[0] == len(texts):
                print(f"    {cache_key}: cache hit "
                      f"({emb.shape[0]} x {emb.shape[1]}, {cache_path})")
                return emb
            print(f"    {cache_key}: cache stale "
                  f"(cached {emb.shape[0]} != current {len(texts)}), re-encoding")

    from sentence_transformers import SentenceTransformer

    print(f"    {cache_key}: encoding {len(texts)} texts with {model_name} ...")
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
        print(f"    {cache_key}: saved to cache ({cache_path})")

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
# Auto-tuning functions
# ---------------------------------------------------------------------------

def _compute_bayesian_run_from_cache(
    tune_cache: dict[str, dict],
    corpus_ids: list[str],
    transform: BayesianProbabilityTransform,
) -> dict[str, dict[str, float]]:
    """Build a pytrec_eval run dict from cached per-query data."""
    run: dict[str, dict[str, float]] = {}
    for qid, cache in tune_cache.items():
        union_idx = cache["union_idx"]
        active = cache["active"]
        probs = np.zeros(len(union_idx), dtype=np.float64)
        if np.any(active):
            probs[active] = transform.score_to_probability(
                cache["active_scores"], cache["tfs"], cache["doc_len_ratios"],
            )
        run[qid] = {
            corpus_ids[union_idx[j]]: float(probs[j])
            for j in range(len(union_idx))
        }
    return run


def _compute_balanced_run_from_cache(
    tune_cache: dict[str, dict],
    corpus_ids: list[str],
    transform: BayesianProbabilityTransform,
    weight: float,
) -> dict[str, dict[str, float]]:
    """Build a balanced fusion run dict from cached per-query data."""
    run: dict[str, dict[str, float]] = {}
    for qid, cache in tune_cache.items():
        union_idx = cache["union_idx"]
        active = cache["active"]
        cand_dense = cache["cand_dense"]
        probs = np.zeros(len(union_idx), dtype=np.float64)
        if np.any(active):
            probs[active] = transform.score_to_probability(
                cache["active_scores"], cache["tfs"], cache["doc_len_ratios"],
            )
        fused = np.asarray(
            balanced_log_odds_fusion(probs, cand_dense, weight),
            dtype=np.float64,
        )
        run[qid] = {
            corpus_ids[union_idx[j]]: float(fused[j])
            for j in range(len(union_idx))
        }
    return run


def _compute_hybrid_and_run_from_cache(
    tune_cache: dict[str, dict],
    corpus_ids: list[str],
    transform: BayesianProbabilityTransform,
    hybrid_alpha: float,
) -> dict[str, dict[str, float]]:
    """Build a log-odds conjunction run dict from cached per-query data."""
    run: dict[str, dict[str, float]] = {}
    for qid, cache in tune_cache.items():
        union_idx = cache["union_idx"]
        active = cache["active"]
        cand_dense = cache["cand_dense"]
        probs = np.zeros(len(union_idx), dtype=np.float64)
        if np.any(active):
            probs[active] = transform.score_to_probability(
                cache["active_scores"], cache["tfs"], cache["doc_len_ratios"],
            )
        dense_probs = np.asarray(
            cosine_to_probability(cand_dense), dtype=np.float64,
        )
        n_cand = len(union_idx)
        combined = np.zeros(n_cand, dtype=np.float64)
        for d in range(n_cand):
            pair = np.array([probs[d], dense_probs[d]])
            combined[d] = float(log_odds_conjunction(pair, alpha=hybrid_alpha))
        run[qid] = {
            corpus_ids[union_idx[j]]: float(combined[j])
            for j in range(len(union_idx))
        }
    return run


def learn_parameters_from_qrels(
    scorer: BayesianBM25Scorer,
    corpus_ids: list[str],
    all_query_ids: list[str],
    all_query_texts: list[str],
    train_qrels: dict[str, dict[str, int]],
) -> tuple[float, float]:
    """Learn alpha and beta via supervised gradient descent on train qrels.

    Collects (BM25_score, label) pairs. Positive docs from qrels get
    label=1; sampled absent docs get label=0 (up to 20 per query).
    """
    rng = np.random.default_rng(42)
    doc_id_to_idx = {did: i for i, did in enumerate(corpus_ids)}
    n_docs = len(corpus_ids)

    # Find train queries
    train_qid_set = set(train_qrels.keys())
    train_indices = [
        i for i, qid in enumerate(all_query_ids)
        if qid in train_qid_set
    ]

    if not train_indices:
        return (1.0, 0.5)

    train_texts = [all_query_texts[i] for i in train_indices]
    train_qids = [all_query_ids[i] for i in train_indices]
    train_tokens = tokenize_texts(train_texts)

    all_scores: list[float] = []
    all_labels: list[float] = []

    for qid, qtokens in zip(train_qids, train_tokens, strict=True):
        rel_map = train_qrels.get(qid)
        if not rel_map:
            continue
        if not qtokens:
            continue
        bm25_scores = scorer._bm25.get_scores(qtokens)

        qrel_indices: set[int] = set()
        for did, rel in rel_map.items():
            idx = doc_id_to_idx.get(did)
            if idx is None:
                continue
            all_scores.append(float(bm25_scores[idx]))
            all_labels.append(1.0 if rel > 0 else 0.0)
            qrel_indices.add(idx)

        # Negative sampling
        neg_pool = [i for i in range(n_docs) if i not in qrel_indices]
        neg_count = min(20, len(neg_pool))
        if neg_count > 0:
            neg_sample = rng.choice(neg_pool, size=neg_count, replace=False)
            for idx in neg_sample:
                all_scores.append(float(bm25_scores[idx]))
                all_labels.append(0.0)

    if len(all_scores) < 2:
        return (1.0, 0.5)

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.float64)

    transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
    transform.fit(scores_arr, labels_arr, learning_rate=0.1, max_iterations=2000)
    return (transform.alpha, transform.beta)


def grid_search_tuned(
    scorer: BayesianBM25Scorer,
    corpus_ids: list[str],
    tune_cache: dict[str, dict],
    qrels: dict[str, dict[str, int]],
    alpha: float,
    beta: float,
    auto_base_rate: float,
    k: int,
) -> dict[str, float | None]:
    """Grid search over base_rate, fusion_weight, and hybrid_alpha.

    Uses cached per-query data from the main scoring loop.
    """
    # Filter cache to queries with qrels
    eval_cache = {qid: c for qid, c in tune_cache.items() if qid in qrels}
    qrels_pytrec = {qid: {d: s for d, s in rels.items()} for qid, rels in qrels.items()}

    has_dense = any("cand_dense" in c and c["cand_dense"] is not None for c in eval_cache.values())

    # -- Phase B: base_rate grid search --
    base_rate_candidates: list[float | None] = [
        None, 0.001, 0.005, 0.01, 0.05, 0.1, auto_base_rate,
    ]
    seen: set = set()
    unique_candidates: list[float | None] = []
    for c in base_rate_candidates:
        key = round(c, 10) if c is not None else None
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    base_rate_candidates = unique_candidates

    best_base_rate: float | None = None
    best_base_rate_ndcg = -1.0

    print(f"  Grid search: base_rate ({len(base_rate_candidates)} candidates)...")
    for br in base_rate_candidates:
        transform = BayesianProbabilityTransform(alpha=alpha, beta=beta, base_rate=br)
        run = _compute_bayesian_run_from_cache(eval_cache, corpus_ids, transform)
        result = evaluate_pytrec(qrels_pytrec, run, k=k)
        ndcg = result[f"NDCG@{k}"]
        label = str(br) if br is not None else "None"
        print(f"    base_rate={label:>12s}  NDCG@{k}={ndcg:.4f}")
        if ndcg > best_base_rate_ndcg:
            best_base_rate_ndcg = ndcg
            best_base_rate = br

    # -- Phase C: fusion_weight grid search --
    best_fusion_weight: float | None = None
    if has_dense:
        fusion_candidates = [round(w * 0.1, 1) for w in range(11)]
        best_fusion_ndcg = -1.0

        print(f"  Grid search: fusion_weight ({len(fusion_candidates)} candidates)...")
        for fw in fusion_candidates:
            transform = BayesianProbabilityTransform(
                alpha=alpha, beta=beta, base_rate=best_base_rate,
            )
            run = _compute_balanced_run_from_cache(eval_cache, corpus_ids, transform, fw)
            result = evaluate_pytrec(qrels_pytrec, run, k=k)
            ndcg = result[f"NDCG@{k}"]
            print(f"    fusion_weight={fw:.1f}  NDCG@{k}={ndcg:.4f}")
            if ndcg > best_fusion_ndcg:
                best_fusion_ndcg = ndcg
                best_fusion_weight = fw

    # -- Phase D: hybrid_alpha grid search --
    best_hybrid_alpha: float | None = None
    if has_dense:
        hybrid_alpha_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_hybrid_ndcg = -1.0

        print(f"  Grid search: hybrid_alpha ({len(hybrid_alpha_candidates)} candidates)...")
        for ha in hybrid_alpha_candidates:
            transform = BayesianProbabilityTransform(
                alpha=alpha, beta=beta, base_rate=best_base_rate,
            )
            run = _compute_hybrid_and_run_from_cache(eval_cache, corpus_ids, transform, ha)
            result = evaluate_pytrec(qrels_pytrec, run, k=k)
            ndcg = result[f"NDCG@{k}"]
            print(f"    hybrid_alpha={ha:.2f}  NDCG@{k}={ndcg:.4f}")
            if ndcg > best_hybrid_ndcg:
                best_hybrid_ndcg = ndcg
                best_hybrid_alpha = ha

    return {
        "alpha": alpha,
        "beta": beta,
        "base_rate": best_base_rate,
        "fusion_weight": best_fusion_weight,
        "hybrid_alpha": best_hybrid_alpha,
    }


def print_tuned_summary(
    tuned: dict[str, float | None],
    auto_alpha: float,
    auto_beta: float,
    auto_base_rate: float,
    learned: bool,
    has_dense: bool,
) -> None:
    """Print a summary of all tuned parameters."""
    alpha = tuned["alpha"]
    beta = tuned["beta"]
    base_rate = tuned["base_rate"]
    fusion_weight = tuned.get("fusion_weight")
    hybrid_alpha = tuned.get("hybrid_alpha")

    alpha_source = "(learned)" if learned else "(auto-estimated)"
    beta_source = "(learned)" if learned else "(auto-estimated)"

    print()
    print("--- Tuned Parameters ---")
    if learned:
        print(f"  alpha:         {alpha:.2f} {alpha_source:16s} [auto-estimated: {auto_alpha:.2f}]")
        print(f"  beta:          {beta:.2f} {beta_source:16s} [auto-estimated: {auto_beta:.2f}]")
    else:
        print(f"  alpha:         {alpha:.2f} {alpha_source}")
        print(f"  beta:          {beta:.2f} {beta_source}")

    if base_rate is not None:
        print(f"  base_rate:     {base_rate:.4f} (grid search best, 7 candidates)")
    else:
        print("  base_rate:     None (grid search best, 7 candidates)")

    if has_dense:
        if fusion_weight is not None:
            print(f"  fusion_weight: {fusion_weight:.2f}  (grid search best, 11 candidates)")
        if hybrid_alpha is not None:
            print(f"  hybrid_alpha:  {hybrid_alpha:.2f}  (grid search best, 5 candidates)")

    print("------------------------")
    print()


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
# Attention variant helpers
# ---------------------------------------------------------------------------


def _prepare_attn_probs(
    probs_br: np.ndarray,
    cand_dense: np.ndarray,
    probs_no_br: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Prepare raw probability signals for attention.

    Normalization is handled by the model when ``normalize=True`` is set
    on the ``AttentionLogOddsWeights`` constructor.

    When probs_no_br is provided, returns 3 signals (BM25-BR, dense,
    BM25-no-BR).  Otherwise returns 2 signals.
    """
    dense_probs = np.asarray(
        cosine_to_probability(cand_dense), dtype=np.float64,
    )
    if probs_no_br is not None:
        return probs_br, dense_probs, probs_no_br
    return probs_br, dense_probs


def _collect_attn_training_data(
    attn_cache: dict,
    corpus_ids: list,
    qrels: dict,
    feature_key: str,
    seed: int = 42,
    exclude_qids: set | None = None,
    n_signals: int = 2,
) -> tuple[list, list, list, list]:
    """Collect (probs, labels, features, query_ids) for attention training.

    When exclude_qids is set, those queries are skipped (for CV).
    n_signals=3 includes BM25 probs without base rate as a third signal.
    Returns query_ids so fit() can normalize logits per-query when normalize=True.
    """
    rng = np.random.default_rng(seed)
    train_probs: list[list[float]] = []
    train_labels: list[float] = []
    train_features: list[np.ndarray] = []
    train_query_ids: list[str] = []

    for qid, cache in attn_cache.items():
        if exclude_qids and qid in exclude_qids:
            continue
        qrel_map = qrels.get(qid)
        if not qrel_map:
            continue

        ui = cache["union_idx"]
        probs_no_br = cache.get("cand_probs") if n_signals == 3 else None
        signals = _prepare_attn_probs(
            cache["cand_probs_br"], cache["cand_dense"],
            probs_no_br=probs_no_br,
        )
        feats = cache[feature_key]

        pos_count = 0
        neg_indices: list[int] = []
        for j in range(len(ui)):
            doc_id = corpus_ids[ui[j]]
            if doc_id in qrel_map:
                train_probs.append([s[j] for s in signals])
                train_labels.append(1.0 if qrel_map[doc_id] > 0 else 0.0)
                train_features.append(feats)
                train_query_ids.append(qid)
                if qrel_map[doc_id] > 0:
                    pos_count += 1
            else:
                neg_indices.append(j)

        n_neg = min(pos_count, len(neg_indices))
        if n_neg > 0:
            sampled = rng.choice(neg_indices, size=n_neg, replace=False)
            for j in sampled:
                train_probs.append([s[j] for s in signals])
                train_labels.append(0.0)
                train_features.append(feats)
                train_query_ids.append(qid)

    return train_probs, train_labels, train_features, train_query_ids


def _score_attn_variant(
    model: AttentionLogOddsWeights,
    attn_cache: dict,
    corpus_ids: list,
    feature_key: str,
    only_qids: set | None = None,
    n_signals: int = 2,
) -> dict[str, dict[str, float]]:
    """Score queries with a trained attention model. Returns run dict."""
    run: dict[str, dict[str, float]] = {}
    for qid, cache in attn_cache.items():
        if only_qids is not None and qid not in only_qids:
            continue
        ui = cache["union_idx"]
        probs_no_br = cache.get("cand_probs") if n_signals == 3 else None
        signals = _prepare_attn_probs(
            cache["cand_probs_br"], cache["cand_dense"],
            probs_no_br=probs_no_br,
        )
        feats = cache[feature_key]
        probs_matrix = np.column_stack(signals)

        scores = model(probs_matrix, feats, use_averaged=True)
        run[qid] = {
            corpus_ids[ui[j]]: float(scores[j])
            for j in range(len(ui))
        }
    return run


def _train_and_score_attn(
    variant_name: str,
    attn_cache: dict,
    corpus_ids: list,
    qrels: dict,
    feature_key: str,
    n_features: int,
    normalize: bool,
    methods: list[str],
    runs: dict,
    alpha: float = 0.5,
    n_signals: int = 2,
    lr: float = 0.01,
    max_iter: int = 500,
) -> bool:
    """Train one attention variant and populate its run. Returns success."""
    tp, tl, tf, tq = _collect_attn_training_data(
        attn_cache, corpus_ids, qrels, feature_key,
        n_signals=n_signals,
    )
    n_train = len(tp)
    labels_arr = np.array(tl, dtype=np.float64)
    has_both = (
        n_train >= 10
        and float(np.sum(labels_arr)) > 0
        and float(np.sum(1.0 - labels_arr)) > 0
    )

    if not has_both:
        if variant_name in methods:
            methods.remove(variant_name)
        runs.pop(variant_name, None)
        print(f"  {variant_name} skipped (insufficient data: {n_train} pairs)")
        return False

    model = AttentionLogOddsWeights(
        n_signals=n_signals, n_query_features=n_features, alpha=alpha,
        normalize=normalize,
    )
    model.fit(
        np.array(tp, dtype=np.float64),
        labels_arr,
        np.array(tf, dtype=np.float64),
        learning_rate=lr,
        max_iterations=max_iter,
        query_ids=np.array(tq) if normalize else None,
    )
    runs[variant_name] = _score_attn_variant(
        model, attn_cache, corpus_ids, feature_key,
        n_signals=n_signals,
    )
    print(f"  {variant_name} trained ({n_train} pairs)")
    return True


def _train_attn_cv(
    attn_cache: dict,
    corpus_ids: list,
    qrels: dict,
    feature_key: str,
    n_features: int,
    normalize: bool,
    methods: list[str],
    runs: dict,
    n_folds: int = 5,
    alpha: float = 0.5,
    n_signals: int = 2,
    lr: float = 0.01,
    max_iter: int = 500,
) -> bool:
    """Train Attn-NR-CV via k-fold cross-validation."""
    variant_name = "Attn-NR-CV"
    cv_qids = [qid for qid in attn_cache if qrels.get(qid)]
    n_cv = len(cv_qids)

    if n_cv < 10:
        if variant_name in methods:
            methods.remove(variant_name)
        runs.pop(variant_name, None)
        print(f"  {variant_name} skipped (insufficient queries: {n_cv})")
        return False

    rng_cv = np.random.default_rng(42)
    perm = rng_cv.permutation(n_cv)
    fold_size = n_cv // n_folds
    cv_run: dict[str, dict[str, float]] = {}
    total_train = 0

    for fold_i in range(n_folds):
        start = fold_i * fold_size
        end = n_cv if fold_i == n_folds - 1 else start + fold_size
        test_qids = {cv_qids[perm[j]] for j in range(start, end)}

        tp, tl, tf, tq = _collect_attn_training_data(
            attn_cache, corpus_ids, qrels,
            feature_key, exclude_qids=test_qids,
            n_signals=n_signals,
        )
        labels_arr = np.array(tl, dtype=np.float64)
        has_both = (
            len(tp) >= 10
            and float(np.sum(labels_arr)) > 0
            and float(np.sum(1.0 - labels_arr)) > 0
        )
        if not has_both:
            continue

        model = AttentionLogOddsWeights(
            n_signals=n_signals, n_query_features=n_features, alpha=alpha,
            normalize=normalize,
        )
        model.fit(
            np.array(tp, dtype=np.float64),
            labels_arr,
            np.array(tf, dtype=np.float64),
            learning_rate=lr,
            max_iterations=max_iter,
            query_ids=np.array(tq) if normalize else None,
        )
        fold_run = _score_attn_variant(
            model, attn_cache, corpus_ids, feature_key,
            only_qids=test_qids,
            n_signals=n_signals,
        )
        cv_run.update(fold_run)
        total_train += len(tp)

    if cv_run:
        runs[variant_name] = cv_run
        print(
            f"  {variant_name} trained ({n_folds}-fold CV, "
            f"{n_cv} queries, {total_train} total train pairs)"
        )
        return True

    if variant_name in methods:
        methods.remove(variant_name)
    runs.pop(variant_name, None)
    print(f"  {variant_name} skipped (all CV folds had insufficient data)")
    return False


# ---------------------------------------------------------------------------
# Main benchmark pipeline
# ---------------------------------------------------------------------------

BASELINE_METHODS = [
    "BM25", "Dense", "Convex", "RRF",
    "Bayesian-OR", "Bayesian-LogOdds", "LO-Local", "Bayesian-LO-BR", "Bayesian-Balanced",
    "Balanced-Mix", "Balanced-Elbow",
    "Gated-ReLU", "Gated-Swish", "Gated-GELU", "Gated-Swish-B2",
    "Attention", "Attn-NR", "Attn-NR-CV",
    "Multi-Head", "MH-NR",
    "MultiField", "MF-Balanced",
]

TUNED_METHODS = [
    "Bayesian-Tuned", "Balanced-Tuned", "Hybrid-AND-Tuned",
]


def run_dataset(
    dataset_dir: str,
    dataset_name: str,
    model_name: str,
    k: int = 10,
    retrieve_k: int = 1000,
    tune: bool = False,
    cache_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    """Run all fusion methods on a single BEIR dataset.

    Follows retrieve-then-evaluate: retrieve top-R from each signal,
    fuse union candidates, evaluate top-k.

    When tune=True, additionally runs auto-estimation, supervised learning
    (if train qrels exist), grid search, and evaluates tuned configurations.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory (containing corpus.jsonl, etc.).
    dataset_name : str
        Human-readable name for logging and cache key.
    model_name : str
        Sentence-transformers model identifier.
    k : int
        Evaluation depth.
    retrieve_k : int
        Number of candidates per signal.
    tune : bool
        Whether to run auto-tuning after baseline evaluation.
    cache_dir : str or None
        Root directory for embedding cache.  Embeddings are stored at
        ``{cache_dir}/{dataset_name}/{model}.npz``.  ``None`` disables
        caching entirely.
    """
    print(f"\n{'=' * 70}")
    print(f"  {dataset_name}")
    print(f"{'=' * 70}")

    # 1. Load data
    t0 = time.time()
    data = load_beir_dataset(dataset_dir)
    corpus_ids = data["corpus_ids"]
    corpus_texts = data["corpus_texts"]
    corpus_titles = data["corpus_titles"]
    corpus_bodies = data["corpus_bodies"]
    query_ids = data["query_ids"]
    query_texts = data["query_texts"]
    qrels = data["qrels"]
    all_query_ids = data["all_query_ids"]
    all_query_texts = data["all_query_texts"]
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
    scorer_br_mix = BayesianBM25Scorer(
        k1=1.2, b=0.75, method="lucene",
        base_rate="auto", base_rate_method="mixture",
    )
    scorer_br_mix.index(corpus_tokens, show_progress=False)
    scorer_br_elbow = BayesianBM25Scorer(
        k1=1.2, b=0.75, method="lucene",
        base_rate="auto", base_rate_method="elbow",
    )
    scorer_br_elbow.index(corpus_tokens, show_progress=False)
    print(
        f"  BM25 indexed -- base_rate: percentile={scorer_br.base_rate:.6f}, "
        f"mixture={scorer_br_mix.base_rate:.6f}, "
        f"elbow={scorer_br_elbow.base_rate:.6f} ({time.time() - t0:.1f}s)"
    )

    # 3b. Build multi-field BM25 index (title + body as separate fields)
    t0 = time.time()
    title_tokens = tokenize_texts(corpus_titles)
    body_tokens = tokenize_texts(corpus_bodies)

    # Check whether the title field has any actual tokens.  Some BEIR
    # datasets (e.g. FiQA) have empty titles for most/all documents,
    # which causes bm25s to fail on an empty vocabulary.
    has_title_tokens = any(len(t) > 0 for t in title_tokens)

    mf_scorer: MultiFieldScorer | None = None
    if has_title_tokens:
        mf_docs = [
            {"title": title_tokens[i], "body": body_tokens[i]}
            for i in range(len(corpus_titles))
        ]
        mf_scorer = MultiFieldScorer(
            fields=["title", "body"],
            k1=1.2, b=0.75, method="lucene",
        )
        mf_scorer.index(mf_docs, show_progress=False)
        print(f"  MultiField indexed (title + body) ({time.time() - t0:.1f}s)")
    else:
        print(f"  MultiField skipped (no title tokens in corpus) ({time.time() - t0:.1f}s)")

    # 4. Encode dense embeddings
    cache_path: str | None = None
    if cache_dir is not None:
        safe_model = model_name.replace("/", "_")
        cache_path = os.path.join(cache_dir, dataset_name, f"{safe_model}.npz")

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

    methods = list(BASELINE_METHODS)
    if mf_scorer is None:
        methods = [m for m in methods if m not in ("MultiField", "MF-Balanced")]
    runs: dict[str, dict[str, dict[str, float]]] = {m: {} for m in methods}

    # Pre-allocate rank arrays (reused per query, reset after each)
    bm25_rank_full = np.zeros(n_docs, dtype=np.float64)
    dense_rank_full = np.zeros(n_docs, dtype=np.float64)

    effective_R = min(retrieve_k, n_docs)

    # Per-query cache for tuning
    tune_cache: dict[str, dict] = {}

    # Per-query cache for attention model training
    attn_cache: dict[str, dict] = {}

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
        active_doc_ids = np.array([], dtype=int)
        active_scores = np.array([], dtype=np.float64)
        tfs = np.array([], dtype=np.float64)
        doc_len_ratios = np.array([], dtype=np.float64)

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
        cand_bayesian_probs_mix = np.zeros(len(union_idx), dtype=np.float64)
        cand_bayesian_probs_elbow = np.zeros(len(union_idx), dtype=np.float64)
        if np.any(active):
            cand_bayesian_probs_br[active] = scorer_br._transform.score_to_probability(
                active_scores, tfs, doc_len_ratios,
            )
            cand_bayesian_probs_mix[active] = scorer_br_mix._transform.score_to_probability(
                active_scores, tfs, doc_len_ratios,
            )
            cand_bayesian_probs_elbow[active] = scorer_br_elbow._transform.score_to_probability(
                active_scores, tfs, doc_len_ratios,
            )

        # Cache for tuning
        if tune:
            tune_cache[qid] = {
                "union_idx": union_idx,
                "cand_dense": cand_dense,
                "active": active.copy(),
                "active_scores": active_scores.copy(),
                "tfs": tfs.copy(),
                "doc_len_ratios": doc_len_ratios.copy(),
            }

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
            "Balanced-Mix": balanced_log_odds_fusion(
                cand_bayesian_probs_mix, cand_dense,
            ),
            "Balanced-Elbow": balanced_log_odds_fusion(
                cand_bayesian_probs_elbow, cand_dense,
            ),
        }

        # Gated log-odds conjunction: BM25 probs + dense probs (Phase 5.3)
        # Stacks two probability signals and applies gating in logit space
        dense_probs_cand = np.asarray(
            cosine_to_probability(cand_dense), dtype=np.float64,
        )
        gated_input = np.column_stack([cand_bayesian_probs_br, dense_probs_cand])
        hybrid_scores["Gated-ReLU"] = log_odds_conjunction(
            gated_input, gating="relu",
        )
        hybrid_scores["Gated-Swish"] = log_odds_conjunction(
            gated_input, gating="swish",
        )
        hybrid_scores["Gated-GELU"] = log_odds_conjunction(
            gated_input, gating="gelu",
        )
        hybrid_scores["Gated-Swish-B2"] = log_odds_conjunction(
            gated_input, gating="swish", gating_beta=2.0,
        )

        for method_name, scores in hybrid_scores.items():
            runs[method_name][qid] = {
                corpus_ids[union_idx[j]]: float(scores[j])
                for j in range(len(union_idx))
            }

        # MultiField: dense probabilities over all docs, take top-R
        if mf_scorer is not None:
            mf_probs_full = mf_scorer.get_probabilities(qtokens)
            mf_topR = np.argsort(-mf_probs_full)[:effective_R]
            runs["MultiField"][qid] = {
                corpus_ids[i]: float(mf_probs_full[i]) for i in mf_topR
            }

            # MF-Balanced: MultiField probs + dense via balanced_log_odds_fusion
            # Union of MultiField top-R and Dense top-R
            mf_union_set = set(mf_topR.tolist()) | set(dense_topR.tolist())
            mf_union_idx = np.array(sorted(mf_union_set))
            mf_cand_probs = mf_probs_full[mf_union_idx]
            mf_cand_dense = dense_sim[mf_union_idx]
            mf_balanced_scores = balanced_log_odds_fusion(mf_cand_probs, mf_cand_dense)
            runs["MF-Balanced"][qid] = {
                corpus_ids[mf_union_idx[j]]: float(mf_balanced_scores[j])
                for j in range(len(mf_union_idx))
            }

        # Cache for attention model: query features + candidate signals
        qlen = len(qtokens)
        bm25_hit_ratio = float(np.count_nonzero(raw_bm25)) / n_docs
        max_bm25_log = float(np.log1p(np.max(raw_bm25))) if np.any(raw_bm25 > 0) else 0.0

        # Dense-side features (per-query, computed from full score arrays)
        top10_k = min(10, n_docs)
        dense_top_scores = np.sort(dense_sim)[-top10_k:]
        dense_top10_mean = float(np.mean(dense_top_scores))
        dense_top10_std = float(np.std(dense_top_scores)) if top10_k > 1 else 0.0
        max_dense_log = float(np.log1p(max(0.0, float(np.max(dense_sim)))))

        # Cross-signal feature: top-100 retrieval overlap (Jaccard)
        top100_k = min(100, n_docs)
        bm25_top100 = set(np.argsort(-raw_bm25)[:top100_k].tolist())
        dense_top100 = set(np.argsort(-dense_sim)[:top100_k].tolist())
        union_sz = len(bm25_top100 | dense_top100)
        overlap_ratio = float(len(bm25_top100 & dense_top100)) / union_sz if union_sz > 0 else 0.0

        attn_cache[qid] = {
            "union_idx": union_idx,
            "cand_probs": cand_bayesian_probs.copy(),
            "cand_probs_br": cand_bayesian_probs_br.copy(),
            "cand_dense": cand_dense.copy(),
            "features": np.array(
                [np.log1p(qlen), bm25_hit_ratio, max_bm25_log],
                dtype=np.float64,
            ),
            "features_rich": np.array(
                [np.log1p(qlen), bm25_hit_ratio, max_bm25_log,
                 dense_top10_mean, dense_top10_std, max_dense_log,
                 overlap_ratio],
                dtype=np.float64,
            ),
        }

    print(f"  Scored {n_queries} queries x {len(methods)} methods, "
          f"R={effective_R} ({time.time() - t0:.1f}s)")

    # 5b. Train and score AttentionLogOddsWeights (Paper 2, Section 8)
    # Collect (probability_pair, label, query_features) from qrels.
    # BEIR qrels typically only list relevant documents, so we also sample
    # negative candidates (unjudged docs from the retrieval union set) to
    # provide the model with both classes.
    t0 = time.time()
    attn_train_probs: list[list[float]] = []
    attn_train_labels: list[float] = []
    attn_train_features: list[np.ndarray] = []
    attn_rng = np.random.default_rng(42)

    for qid, cache in attn_cache.items():
        qrel_map = qrels.get(qid)
        if not qrel_map:
            continue
        ui = cache["union_idx"]
        probs_br = cache["cand_probs_br"]
        dense_probs_arr = np.asarray(
            cosine_to_probability(cache["cand_dense"]), dtype=np.float64,
        )
        feats = cache["features"]

        # Positive examples: judged relevant documents in the candidate set
        pos_count = 0
        neg_indices: list[int] = []
        for j in range(len(ui)):
            doc_id = corpus_ids[ui[j]]
            if doc_id in qrel_map:
                attn_train_probs.append([probs_br[j], dense_probs_arr[j]])
                attn_train_labels.append(1.0 if qrel_map[doc_id] > 0 else 0.0)
                attn_train_features.append(feats)
                if qrel_map[doc_id] > 0:
                    pos_count += 1
            else:
                neg_indices.append(j)

        # Negative sampling: sample up to pos_count unjudged candidates
        n_neg = min(pos_count, len(neg_indices))
        if n_neg > 0:
            sampled = attn_rng.choice(neg_indices, size=n_neg, replace=False)
            for j in sampled:
                attn_train_probs.append([probs_br[j], dense_probs_arr[j]])
                attn_train_labels.append(0.0)
                attn_train_features.append(feats)

    n_attn_train = len(attn_train_probs)
    attn_labels_arr = np.array(attn_train_labels, dtype=np.float64)
    has_both_classes = (
        n_attn_train >= 10
        and float(np.sum(attn_labels_arr)) > 0
        and float(np.sum(1.0 - attn_labels_arr)) > 0
    )

    if has_both_classes:
        attn_model = AttentionLogOddsWeights(
            n_signals=2, n_query_features=3, alpha=0.5,
        )
        attn_model.fit(
            np.array(attn_train_probs, dtype=np.float64),
            attn_labels_arr,
            np.array(attn_train_features, dtype=np.float64),
            learning_rate=0.01,
            max_iterations=500,
        )

        for qid, cache in attn_cache.items():
            ui = cache["union_idx"]
            probs_br = cache["cand_probs_br"]
            dense_probs_arr = np.asarray(
                cosine_to_probability(cache["cand_dense"]), dtype=np.float64,
            )
            probs_matrix = np.column_stack([probs_br, dense_probs_arr])
            feats = cache["features"]

            # Compute query-dependent attention weights
            w = attn_model._compute_weights(
                feats.reshape(1, -1), use_averaged=True,
            ).squeeze(0)
            attn_scores = log_odds_conjunction(
                probs_matrix, alpha=attn_model.alpha, weights=w,
            )
            runs["Attention"][qid] = {
                corpus_ids[ui[j]]: float(attn_scores[j])
                for j in range(len(ui))
            }

        print(f"  Attention trained on {n_attn_train} judged pairs ({time.time() - t0:.1f}s)")
    else:
        # Not enough training data -- remove Attention from methods
        methods = [m for m in methods if m != "Attention"]
        if "Attention" in runs:
            del runs["Attention"]
        print(f"  Attention skipped (insufficient training data: {n_attn_train} pairs)")

    # 5c. Improved attention: normalization + richer features
    #
    #   Attn-NR    : logit-space min-max normalization + 7 features
    #                (3 BM25 + 3 dense + 1 cross-signal overlap)
    #   Attn-NR-CV : 5-fold cross-validation of Attn-NR
    #
    # normalize=True on AttentionLogOddsWeights applies per-signal
    # column min-max normalization in logit space (same scaling as
    # balanced_log_odds_fusion).  Rich features add dense-side
    # statistics (top-10 mean/std, max) and retrieval overlap.
    t0 = time.time()
    _train_and_score_attn(
        "Attn-NR", attn_cache, corpus_ids, qrels,
        "features_rich", 7, True, methods, runs,
    )
    _train_attn_cv(
        attn_cache, corpus_ids, qrels,
        "features_rich", 7, True, methods, runs,
    )
    print(f"  Attention variants done ({time.time() - t0:.1f}s)")

    # 5d. Multi-head attention fusion (Paper 2, Remark 8.6)
    t0 = time.time()

    # Multi-Head: basic features, 4 heads
    mh_tp, mh_tl, mh_tf, mh_tq = _collect_attn_training_data(
        attn_cache, corpus_ids, qrels, "features",
        n_signals=2,
    )
    mh_n_train = len(mh_tp)
    mh_labels = np.array(mh_tl, dtype=np.float64)
    mh_has_both = (
        mh_n_train >= 10
        and float(np.sum(mh_labels)) > 0
        and float(np.sum(1.0 - mh_labels)) > 0
    )

    if mh_has_both:
        mh_model = MultiHeadAttentionLogOddsWeights(
            n_heads=4, n_signals=2, n_query_features=3, alpha=0.5,
        )
        mh_model.fit(
            np.array(mh_tp, dtype=np.float64),
            mh_labels,
            np.array(mh_tf, dtype=np.float64),
            learning_rate=0.01,
            max_iterations=500,
        )

        for qid, cache in attn_cache.items():
            ui = cache["union_idx"]
            signals = _prepare_attn_probs(cache["cand_probs_br"], cache["cand_dense"])
            probs_matrix = np.column_stack(signals)
            feats = cache["features"]
            scores = mh_model(probs_matrix, feats, use_averaged=True)
            runs["Multi-Head"][qid] = {
                corpus_ids[ui[j]]: float(scores[j])
                for j in range(len(ui))
            }
        print(f"  Multi-Head trained ({mh_n_train} pairs, 4 heads)")
    else:
        methods = [m for m in methods if m != "Multi-Head"]
        if "Multi-Head" in runs:
            del runs["Multi-Head"]
        print(f"  Multi-Head skipped (insufficient data: {mh_n_train} pairs)")

    # MH-NR: rich features + normalize, 4 heads
    mh_nr_tp, mh_nr_tl, mh_nr_tf, mh_nr_tq = _collect_attn_training_data(
        attn_cache, corpus_ids, qrels, "features_rich",
        n_signals=2,
    )
    mh_nr_n_train = len(mh_nr_tp)
    mh_nr_labels = np.array(mh_nr_tl, dtype=np.float64)
    mh_nr_has_both = (
        mh_nr_n_train >= 10
        and float(np.sum(mh_nr_labels)) > 0
        and float(np.sum(1.0 - mh_nr_labels)) > 0
    )

    if mh_nr_has_both:
        mh_nr_model = MultiHeadAttentionLogOddsWeights(
            n_heads=4, n_signals=2, n_query_features=7, alpha=0.5,
            normalize=True,
        )
        mh_nr_model.fit(
            np.array(mh_nr_tp, dtype=np.float64),
            mh_nr_labels,
            np.array(mh_nr_tf, dtype=np.float64),
            learning_rate=0.01,
            max_iterations=500,
            query_ids=np.array(mh_nr_tq),
        )

        for qid, cache in attn_cache.items():
            ui = cache["union_idx"]
            signals = _prepare_attn_probs(cache["cand_probs_br"], cache["cand_dense"])
            probs_matrix = np.column_stack(signals)
            feats = cache["features_rich"]
            scores = mh_nr_model(probs_matrix, feats, use_averaged=True)
            runs["MH-NR"][qid] = {
                corpus_ids[ui[j]]: float(scores[j])
                for j in range(len(ui))
            }
        print(f"  MH-NR trained ({mh_nr_n_train} pairs, 4 heads, normalize)")
    else:
        methods = [m for m in methods if m != "MH-NR"]
        if "MH-NR" in runs:
            del runs["MH-NR"]
        print(f"  MH-NR skipped (insufficient data: {mh_nr_n_train} pairs)")

    print(f"  Multi-head variants done ({time.time() - t0:.1f}s)")

    # 6. Evaluate baselines
    results: dict[str, dict[str, float]] = {}
    for method_name in methods:
        results[method_name] = evaluate_pytrec(qrels_pytrec, runs[method_name], k=k)

    # 6b. Calibration diagnostics for probability-producing methods
    calib_methods = [m for m in CALIBRATION_METHODS if m in runs]
    if calib_methods:
        print_calibration_section(runs, qrels, calib_methods)

    # 7. Tuning (if enabled)
    if tune:
        print("\n  --- Auto-tuning ---")

        # Step 1: Auto-estimate alpha, beta
        print("  [Tune] Step 1: Auto-estimating alpha, beta from corpus...")
        per_query_scores = scorer._sample_pseudo_query_scores(corpus_tokens)
        auto_alpha = scorer._transform.alpha
        auto_beta = scorer._transform.beta
        print(f"    auto alpha={auto_alpha:.4f}, beta={auto_beta:.4f}")

        # Step 2: Auto-estimate base_rate
        print("  [Tune] Step 2: Auto-estimating base_rate...")
        auto_base_rate = scorer._estimate_base_rate(per_query_scores, n_docs)
        print(f"    auto base_rate={auto_base_rate:.6f}")

        # Step 3: Supervised learning (if train qrels exist)
        tuned_alpha = auto_alpha
        tuned_beta = auto_beta
        learned = False

        train_qrels_path = os.path.join(dataset_dir, "qrels", "train.tsv")
        train_qrels: dict[str, dict[str, int]] | None = None
        if os.path.exists(train_qrels_path):
            print(f"  [Tune] Step 3: Supervised learning from {train_qrels_path}...")
            train_qrels = load_qrels_from_tsv(train_qrels_path)
            print(f"    {len(train_qrels)} train queries loaded")
            tuned_alpha, tuned_beta = learn_parameters_from_qrels(
                scorer, corpus_ids,
                all_query_ids, all_query_texts,
                train_qrels,
            )
            learned = True
            print(f"    learned alpha={tuned_alpha:.4f}, beta={tuned_beta:.4f}")
        else:
            print("  [Tune] Step 3: Skipped (no train qrels)")

        # Step 4: Grid search
        print(f"  [Tune] Step 4: Grid search (optimizing NDCG@{k})...")

        # Use train qrels for grid search if available, otherwise test qrels
        if train_qrels is not None:
            # Build tune_cache for train queries
            train_qid_set = set(train_qrels.keys())
            train_indices = [
                i for i, qid in enumerate(all_query_ids) if qid in train_qid_set
            ]
            train_texts = [all_query_texts[i] for i in train_indices]
            train_qids = [all_query_ids[i] for i in train_indices]
            train_tokens = tokenize_texts(train_texts)
            train_emb = encode_dense(
                train_texts, model_name,
                cache_path=cache_path,
                cache_key="train_queries",
                batch_size=128,
            )

            train_tune_cache: dict[str, dict] = {}
            for tq_idx, (tqid, tqtokens) in enumerate(zip(train_qids, train_tokens, strict=True)):
                if not tqtokens:
                    continue
                raw_bm25 = scorer._bm25.get_scores(tqtokens)
                dense_sim = (train_emb[tq_idx] @ corpus_emb.T).astype(np.float64)
                bm25_topR = np.argsort(-raw_bm25)[:effective_R]
                dense_topR = np.argsort(-dense_sim)[:effective_R]
                union_set = set(bm25_topR.tolist()) | set(dense_topR.tolist())
                union_idx = np.array(sorted(union_set))
                cand_bm25 = raw_bm25[union_idx]
                cand_dense = dense_sim[union_idx]
                active = cand_bm25 > 0
                a_scores = np.array([], dtype=np.float64)
                a_tfs = np.array([], dtype=np.float64)
                a_dlr = np.array([], dtype=np.float64)
                if np.any(active):
                    a_doc_ids = union_idx[active]
                    a_scores = cand_bm25[active]
                    a_dlr = scorer._doc_lengths[a_doc_ids] / scorer._avgdl
                    a_tfs = scorer._compute_tf_batch(a_doc_ids, tqtokens)
                train_tune_cache[tqid] = {
                    "union_idx": union_idx,
                    "cand_dense": cand_dense,
                    "active": active.copy(),
                    "active_scores": a_scores.copy(),
                    "tfs": a_tfs.copy(),
                    "doc_len_ratios": a_dlr.copy(),
                }

            grid_qrels = train_qrels
            grid_cache = train_tune_cache
        else:
            grid_qrels = qrels
            grid_cache = tune_cache

        tuned = grid_search_tuned(
            scorer, corpus_ids, grid_cache, grid_qrels,
            tuned_alpha, tuned_beta, auto_base_rate, k,
        )

        # Step 5: Print summary
        has_dense = True  # hybrid_beir always has dense
        print_tuned_summary(
            tuned, auto_alpha, auto_beta, auto_base_rate, learned, has_dense,
        )

        # Step 6: Evaluate tuned configurations on test qrels
        print("  [Tune] Evaluating tuned methods on test qrels...")

        tuned_transform = BayesianProbabilityTransform(
            alpha=tuned["alpha"], beta=tuned["beta"],
            base_rate=tuned["base_rate"],
        )

        # Bayesian-Tuned
        run_tuned = _compute_bayesian_run_from_cache(
            tune_cache, corpus_ids, tuned_transform,
        )
        runs["Bayesian-Tuned"] = run_tuned
        results["Bayesian-Tuned"] = evaluate_pytrec(qrels_pytrec, run_tuned, k=k)

        # Balanced-Tuned
        if tuned["fusion_weight"] is not None:
            run_balanced = _compute_balanced_run_from_cache(
                tune_cache, corpus_ids, tuned_transform, tuned["fusion_weight"],
            )
            runs["Balanced-Tuned"] = run_balanced
            results["Balanced-Tuned"] = evaluate_pytrec(qrels_pytrec, run_balanced, k=k)

        # Hybrid-AND-Tuned
        if tuned["hybrid_alpha"] is not None:
            run_hybrid = _compute_hybrid_and_run_from_cache(
                tune_cache, corpus_ids, tuned_transform, tuned["hybrid_alpha"],
            )
            runs["Hybrid-AND-Tuned"] = run_hybrid
            results["Hybrid-AND-Tuned"] = evaluate_pytrec(qrels_pytrec, run_hybrid, k=k)

    return results


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

CALIBRATION_METHODS = [
    "Bayesian-OR", "Bayesian-LO-BR", "Bayesian-Balanced",
    "Balanced-Mix", "Balanced-Elbow",
    "Gated-ReLU", "Gated-Swish", "Gated-GELU", "Gated-Swish-B2",
    "Attention",
    "Multi-Head", "MH-NR",
    "MultiField", "MF-Balanced",
]


def print_calibration_section(
    runs: dict[str, dict[str, dict[str, float]]],
    qrels: dict[str, dict[str, int]],
    methods: list[str],
) -> None:
    """Print calibration diagnostics for probability-producing methods."""
    print("\n  --- Calibration Diagnostics ---")
    print(f"  {'Method':<22}  {'ECE':>10}  {'Brier':>10}  {'Samples':>8}")
    print(f"  {'-' * 22}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    for method in methods:
        if method not in runs:
            continue
        method_run = runs[method]

        probs_list: list[float] = []
        labels_list: list[float] = []

        for qid, doc_scores in method_run.items():
            qrel_map = qrels.get(qid)
            if qrel_map is None:
                continue
            # Only include doc_ids that appear in qrels for this query
            # to avoid negative bias from unjudged documents
            for doc_id, score in doc_scores.items():
                if doc_id in qrel_map:
                    probs_list.append(score)
                    labels_list.append(1.0 if qrel_map[doc_id] > 0 else 0.0)

        if len(probs_list) < 2:
            print(f"  {method:<22}  {'n/a':>10}  {'n/a':>10}  {len(probs_list):>8}")
            continue

        probs_arr = np.array(probs_list, dtype=np.float64)
        labels_arr = np.array(labels_list, dtype=np.float64)
        report = calibration_report(probs_arr, labels_arr)
        print(
            f"  {method:<22}  {report.ece:>10.6f}  {report.brier:>10.6f}"
            f"  {report.n_samples:>8}"
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(
    all_results: dict[str, dict[str, dict[str, float]]],
    k: int = 10,
) -> None:
    # Discover all methods across all datasets
    all_methods: list[str] = []
    seen: set[str] = set()
    method_order = BASELINE_METHODS + TUNED_METHODS
    for m in method_order:
        for ds_results in all_results.values():
            if m in ds_results and m not in seen:
                all_methods.append(m)
                seen.add(m)
                break

    datasets = list(all_results.keys())
    metrics = [f"NDCG@{k}", f"MAP@{k}", f"Recall@{k}"]

    col_w = max(len(m) for m in all_methods) + 2

    for metric in metrics:
        line_w = 14 + len(all_methods) * (2 + col_w)
        print(f"\n{metric} Results")
        print("=" * line_w)

        header = f"{'Dataset':<14}"
        for method in all_methods:
            header += f"  {method:>{col_w}}"
        print(header)
        print("-" * line_w)

        averages: dict[str, list[float]] = {m: [] for m in all_methods}
        for ds_name in datasets:
            row = f"{ds_name:<14}"
            for method in all_methods:
                if method in all_results[ds_name]:
                    val = all_results[ds_name][method][metric] * 100
                    row += f"  {val:>{col_w - 1}.2f}%"
                    averages[method].append(val)
                else:
                    row += f"  {'n/a':>{col_w}}"
            print(row)

        print("-" * line_w)
        row = f"{'Average':<14}"
        for method in all_methods:
            if averages[method]:
                avg = float(np.mean(averages[method]))
                row += f"  {avg:>{col_w - 1}.2f}%"
            else:
                row += f"  {'n/a':>{col_w}}"
        print(row)

    print(f"\nDelta vs BM25 (NDCG@{k}):")
    bm25_metric = f"NDCG@{k}"
    for method in all_methods:
        if method == "BM25":
            continue
        deltas = []
        for ds_name in datasets:
            if method not in all_results[ds_name]:
                continue
            bm25_val = all_results[ds_name]["BM25"][bm25_metric]
            method_val = all_results[ds_name][method][bm25_metric]
            deltas.append((method_val - bm25_val) * 100)
        if deltas:
            avg_delta = float(np.mean(deltas))
            sign = "+" if avg_delta >= 0 else ""
            print(f"  {method:<20}  avg: {sign}{avg_delta:.2f}")


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
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable auto-tuning of parameters (base_rate, fusion_weight, hybrid_alpha)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing BEIR datasets automatically from the official CDN",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help=(
            "Directory for embedding cache.  Defaults to {beir-dir}/.cache/embeddings. "
            "Pass --no-cache to disable caching entirely."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable embedding cache (always re-encode)",
    )
    args = parser.parse_args()

    # Resolve cache directory
    if args.no_cache:
        effective_cache_dir: str | None = None
    elif args.cache_dir is not None:
        effective_cache_dir = args.cache_dir
    else:
        effective_cache_dir = os.path.join(args.beir_dir, ".cache", "embeddings")

    tune_label = " [TUNE]" if args.tune else ""
    cache_label = "disabled" if effective_cache_dir is None else effective_cache_dir
    print("=" * 70)
    print(f"  BEIR Hybrid Search Benchmark -- Bayesian BM25{tune_label}")
    print(f"  Model: {args.model}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  k={args.top_k}, R={args.retrieve_k}")
    print(f"  Cache: {cache_label}")
    if args.download:
        print("  Download: enabled")
    print("=" * 70)

    # Download datasets if requested
    if args.download:
        print("\n--- Checking / downloading datasets ---")
        for ds_name in args.datasets:
            download_beir_dataset(ds_name, args.beir_dir)

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for ds_name in args.datasets:
        ds_dir = os.path.join(args.beir_dir, ds_name)
        if not os.path.isdir(ds_dir):
            print(f"\nWARNING: Dataset directory not found: {ds_dir}, skipping")
            print("  Hint: use --download to fetch missing datasets automatically")
            continue
        all_results[ds_name] = run_dataset(
            ds_dir, ds_name, args.model,
            k=args.top_k, retrieve_k=args.retrieve_k,
            tune=args.tune,
            cache_dir=effective_cache_dir,
        )

    print_results_table(all_results, k=args.top_k)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
