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

Evaluation protocol:

  * The document whose id equals the query's id is excluded before
    retrieval, before calibration and before scoring (the default of the
    BEIR evaluator, and what MTEB applies to ArguAna, whose queries are
    arguments drawn from the corpus).  Pass --keep-identical-ids to
    disable.
  * The lexical result list holds only documents with a non-zero BM25
    score.
  * Every Bayesian method is ranked on log-odds, never on a probability:
    a probability rounds to 1 above about 36.7 in log-odds and is
    clamped at about 23.0, after which documents tie and the order is
    decided by tie-breaking.
  * Rankings are computed in double precision and passed to pytrec_eval
    as rank-encoded scores, because the evaluator compares scores in
    single precision.
  * Every method with weights learned from relevance labels is
    cross-validated over queries: the weights applied to a query were
    fitted on queries of the other folds.

Methods compared:

  1. BM25                       -- sparse retrieval via bm25s
  2. Dense                      -- cosine similarity via sentence-transformers
  3. Convex                     -- w * dense_norm + (1-w) * bm25_norm, w=0.5
  4. RRF                        -- sum(1/(k + rank_i)), k=60
  5. Bayesian-OR                -- probabilistic OR fusion
  6. Bayesian-LogOdds           -- per-query calibrated log-odds fusion
  7. Bayesian-LogOdds-Local     -- symmetric log-odds, locally calibrated
  8. Bayesian-LogOdds-BR        -- log-odds fusion with base rate prior
  9. Bayesian-Balanced          -- balanced log-odds (percentile base rate)
 10. Bayesian-Balanced-Mix      -- balanced fusion (mixture base rate)
 11. Bayesian-Balanced-Elbow    -- balanced fusion (elbow base rate)
 12. Bayesian-Gated-ReLU        -- gated log-odds, ReLU
 13. Bayesian-Gated-Swish       -- gated log-odds, Swish
 14. Bayesian-Gated-GELU        -- gated log-odds, GELU
 15. Bayesian-Gated-Swish-B2    -- gated log-odds, Swish beta=2.0
 16. Bayesian-Gated-Softplus    -- gated log-odds, softplus
 17. Bayesian-Attention         -- query-dependent attention weights (5-fold CV)
 18. Bayesian-Attn-Norm         -- attention + logit normalization + 7 features (5-fold CV)
 19. Bayesian-MultiHead         -- 4-head attention fusion (5-fold CV)
 20. Bayesian-MultiHead-Norm    -- MultiHead + normalize + rich features (5-fold CV)
 21. Bayesian-MultiField        -- MultiFieldScorer (title + body), sparse-only
 22. Bayesian-MultiField-Bal    -- MultiField + dense via balanced fusion
 23. Bayesian-Vector-Balanced   -- VPT-calibrated dense + balanced fusion
 24. Bayesian-Vector-Softplus   -- VPT-calibrated dense + softplus gating
 25. Bayesian-Vector-Attn       -- VPT-calibrated dense + attention fusion (5-fold CV)
 26. Dense-Kappa                -- global sigmoid calibration (Paper 3, Stage 1)
 27. Dense-Arctan               -- arctangent normalization baseline
 28. Dense-Platt                -- supervised Platt scaling baseline (5-fold CV)
 29. VPT-DensityPrior           -- VPT with IVF density prior only (Stage 6a)
 30. VPT-BM25Weights            -- VPT with BM25 weights only (Stage 6b)
 31. VPT-BW-0.2                 -- VPT bandwidth ablation c=0.2 (Stage 7)
 32. VPT-BW-0.5                 -- VPT bandwidth ablation c=0.5
 33. VPT-BW-1.0                 -- VPT bandwidth ablation c=1.0
 34. VPT-BW-2.0                 -- VPT bandwidth ablation c=2.0

With --tune, additional tuned methods are evaluated on datasets that
ship training qrels (parameters are tuned on the training queries only):

 35. Bayesian-Tuned             -- tuned alpha, beta, base_rate
 36. Bayesian-Balanced-Tuned    -- balanced fusion with tuned weight
 37. Bayesian-Hybrid-AND-Tuned  -- log-odds conjunction with tuned alpha

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
    python hybrid_beir.py -d ../../beir --keep-identical-ids
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from functools import lru_cache

import bm25s
import numpy as np
import Stemmer

from bayesian_bm25.calibration import PlattCalibrator
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.fusion import (
    AttentionLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
    balanced_log_odds_fusion,
    cosine_to_probability,
    log_odds_fusion,
    log_odds_or,
)
from bayesian_bm25.multi_field import MultiFieldScorer
from bayesian_bm25.probability import (
    BayesianProbabilityTransform,
    _clamp_probability,
    logit,
    sigmoid,
)
from bayesian_bm25.scorer import BayesianBM25Scorer
from bayesian_bm25.vector_probability import (
    VectorProbabilityTransform,
    ivf_density_prior,
)
from benchmarks.calibration_accumulator import CalibrationAccumulator
from benchmarks.dataset_identity import canonicalize_qrels
from benchmarks.ranking_eval import evaluate_run, mean_metrics, ranked_run
from benchmarks.reference_vector import reference_vector_log_odds
from benchmarks.search_diagnostics import (
    build_exact_search_diagnostics,
    build_ivf_search_diagnostics,
    separability_gate,
)
from benchmarks.simple_ivf import IVFSearchResult, SimpleIVF

STEMMER = Stemmer.Stemmer("english")
DEFAULT_DATASETS = ("arguana", "fiqa", "nfcorpus", "scidocs", "scifact")
DEFAULT_DENSE_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BEIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "beir")
REQUIRED_DATASET_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")


def validate_dataset_inputs(beir_dir: str | os.PathLike[str], datasets: list[str]) -> None:
    """Require every requested collection before running any of them."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    if len(set(datasets)) != len(datasets):
        raise ValueError("dataset names must be unique")
    missing = [
        os.path.join(beir_dir, name, relative)
        for name in datasets
        for relative in REQUIRED_DATASET_FILES
        if not os.path.isfile(os.path.join(beir_dir, name, relative))
    ]
    if missing:
        paths = "\n".join(missing)
        raise FileNotFoundError(f"Required BEIR inputs are missing:\n{paths}")

# Log-odds given to a candidate that matches no query term wherever a
# fusion needs a finite lexical value: the logit of the probability
# clamp, 1e-10.  The first version reached the same value by clamping a
# probability of 0.
UNMATCHED_LOG_ODDS = float(logit(1e-10))

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

def load_beir_dataset(dataset_dir: str, *, identity: str = "text") -> dict:
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

    from benchmarks.dataset_identity import canonicalize_dataset

    return canonicalize_dataset({
        "corpus_ids": corpus_ids,
        "corpus_texts": corpus_texts,
        "corpus_titles": corpus_titles,
        "corpus_bodies": corpus_bodies,
        "query_ids": filtered_qids,
        "query_texts": filtered_qtexts,
        "qrels": qrels,
        "all_query_ids": query_ids,
        "all_query_texts": query_texts,
    }, identity=identity)


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


def training_qrels_for_dataset(data: dict, qrels: dict) -> dict:
    """Canonicalize train judgments and exclude every evaluated query alias."""
    merged = canonicalize_qrels(qrels, data["query_alias_to_id"], data["corpus_alias_to_id"])
    evaluated = set(data["query_ids"])
    available = set(data["all_query_ids"])
    return {qid: judgments for qid, judgments in merged.items()
            if qid in available and qid not in evaluated}


def posterior_context(method: str) -> CalibrationContext:
    """The actual evaluation population of each historical posterior variant."""
    return CalibrationContext(
        reference_population=(
            "query_excluded_multifield_list" if method == "Bayesian-MultiField"
            else "query_excluded_lexical_dense_union"
        ),
        count_source="not_count_constrained",
        slope_source=("training_folds" if method == "Dense-Platt" or "Att" in method
                      or "Head" in method else "score_shape_assumption"),
        ceiling_source="not_applicable", relevance_target="judged_relevance",
    )


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

@lru_cache(maxsize=1)
def _dense_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_dense(
    texts: list[str],
    model_name: str,
    cache_path: str | None = None,
    cache_key: str = "embeddings",
    batch_size: int = 128,
    cache_context: dict | None = None,
) -> np.ndarray:
    """Encode texts with sentence-transformers, with .npz caching.

    Returns L2-normalized embeddings of shape (len(texts), dim).

    When *cache_path* is given, the function first checks for a cached
    array under *cache_key*. A hit requires the exact text order, model
    name, and cache context to match a recorded fingerprint. An archive
    without fingerprints is re-encoded rather than trusted by row count.

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
    digest = hashlib.sha256(json.dumps(
        {"model": model_name, "context": cache_context}, sort_keys=True,
    ).encode("utf-8"))
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    fingerprint = digest.hexdigest()
    fingerprint_key = f"{cache_key}_fingerprint"
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            if cache_key in data and fingerprint_key in data:
                emb = data[cache_key]
                if str(data[fingerprint_key]) == fingerprint and emb.shape[0] == len(texts):
                    print(f"    {cache_key}: cache hit "
                          f"({emb.shape[0]} x {emb.shape[1]}, {cache_path})")
                    return emb
            print(f"    {cache_key}: cache fingerprint mismatch, re-encoding")

    print(f"    {cache_key}: encoding {len(texts)} texts with {model_name} ...")
    model = _dense_encoder(model_name)
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
            with np.load(cache_path) as data:
                existing = dict(data)
        else:
            existing = {}
        existing[cache_key] = emb
        existing[fingerprint_key] = np.asarray(fingerprint)
        existing[f"{cache_key}_context"] = np.asarray(json.dumps(cache_context, sort_keys=True))
        np.savez(cache_path, **existing)
        print(f"    {cache_key}: saved to cache ({cache_path})")

    return emb


def _compute_bm25_features_for_docs(
    scorer: BayesianBM25Scorer,
    raw_bm25: np.ndarray,
    doc_indices: np.ndarray,
    qtokens: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-doc BM25 features for an arbitrary subset."""
    doc_indices = np.asarray(doc_indices, dtype=np.int32)
    scores = np.asarray(raw_bm25[doc_indices], dtype=np.float64)
    active = scores > 0.0

    tfs = np.zeros(len(doc_indices), dtype=np.float64)
    doc_len_ratios = np.ones(len(doc_indices), dtype=np.float64)

    if np.any(active):
        active_doc_ids = doc_indices[active]
        doc_len_ratios[active] = (
            scorer._doc_lengths[active_doc_ids] / scorer._avgdl
        )
        tfs[active] = scorer._compute_tf_batch(active_doc_ids, qtokens)

    return scores, active, tfs, doc_len_ratios


def _apply_bm25_transform(
    transform: BayesianProbabilityTransform,
    scores: np.ndarray,
    active: np.ndarray,
    tfs: np.ndarray,
    doc_len_ratios: np.ndarray,
) -> np.ndarray:
    """Apply a Bayesian BM25 transform to a precomputed subset."""
    probs = np.zeros(len(scores), dtype=np.float64)
    if np.any(active):
        probs[active] = transform.score_to_probability(
            scores[active], tfs[active], doc_len_ratios[active],
        )
    return probs


def _apply_bm25_log_odds(
    transform: BayesianProbabilityTransform,
    scores: np.ndarray,
    active: np.ndarray,
    tfs: np.ndarray,
    doc_len_ratios: np.ndarray,
) -> np.ndarray:
    """Posterior log-odds of a Bayesian BM25 transform on a precomputed subset.

    The log-odds are assembled directly, without passing through a
    probability, so they are neither rounded nor clamped.  A candidate
    that matches no query term has probability 0, hence log-odds -inf.
    """
    log_odds = np.full(len(scores), -np.inf, dtype=np.float64)
    if np.any(active):
        log_odds[active] = transform.score_to_log_odds(
            scores[active], tfs[active], doc_len_ratios[active],
        )
    return log_odds


def _with_unmatched_floor(log_odds: np.ndarray) -> np.ndarray:
    """Replace the -inf of lexical non-matches by a finite floor."""
    return np.where(np.isfinite(log_odds), log_odds, UNMATCHED_LOG_ODDS)


def _dense_log_odds(dense_sim: np.ndarray) -> np.ndarray:
    """Log-odds of the linear cosine-to-probability map, logit((1 + cos) / 2)."""
    return np.asarray(logit(cosine_to_probability(dense_sim)), dtype=np.float64)


def _retrieve_dense_candidates(
    query_vec: np.ndarray,
    corpus_emb: np.ndarray,
    retrieve_k: int,
    *,
    dense_backend: str,
    dense_index: SimpleIVF | None = None,
    ivf_nprobe: int | None = None,
    exclude_idx: int | np.ndarray | None = None,
) -> dict[str, np.ndarray | IVFSearchResult | None]:
    """Retrieve dense candidates using either exact scan or SimpleIVF.

    When ``exclude_idx`` is given, that document is removed from the
    result list and from the scores returned for background estimation,
    so it can enter neither the candidates nor any calibration
    statistic.
    """
    if dense_backend == "exact":
        dense_scores_full = np.asarray(
            query_vec @ corpus_emb.T, dtype=np.float64,
        )
        ranking_scores = dense_scores_full
        background_scores = dense_scores_full
        if exclude_idx is not None:
            ranking_scores = dense_scores_full.copy()
            ranking_scores[exclude_idx] = -np.inf
            background_scores = np.delete(dense_scores_full, exclude_idx)
        # Stable sort: documents with equal scores keep corpus order, so
        # the ranks that reciprocal rank fusion consumes are deterministic.
        top_idx = np.argsort(-ranking_scores, kind="stable")[:retrieve_k]
        if exclude_idx is not None:
            top_idx = top_idx[~np.isin(top_idx, exclude_idx)]
        top_scores = dense_scores_full[top_idx]
        return {
            "top_idx": np.asarray(top_idx, dtype=np.int32),
            "top_scores": np.asarray(top_scores, dtype=np.float64),
            "sample_idx": np.asarray(top_idx, dtype=np.int32),
            "sample_scores": np.asarray(top_scores, dtype=np.float64),
            "sample_cell_populations": None,
            "full_scores": background_scores,
            "all_scores": dense_scores_full,
            "search_result": None,
        }

    if dense_backend != "ivf" or dense_index is None:
        raise ValueError(
            f"dense_backend must be 'exact' or 'ivf', got {dense_backend!r}",
        )

    extra = 0 if exclude_idx is None else np.atleast_1d(exclude_idx).size
    search = dense_index.search(query_vec, retrieve_k + extra, nprobe=ivf_nprobe)
    if exclude_idx is not None:
        keep = ~np.isin(search.indices, exclude_idx)
        candidate_keep = ~np.isin(search.candidate_indices, exclude_idx)
        search = IVFSearchResult(
            indices=search.indices[keep][:retrieve_k],
            scores=search.scores[keep][:retrieve_k],
            cell_ids=search.cell_ids[keep][:retrieve_k],
            cell_populations=search.cell_populations[keep][:retrieve_k],
            candidate_indices=search.candidate_indices[candidate_keep],
            candidate_scores=search.candidate_scores[candidate_keep],
            candidate_cell_ids=search.candidate_cell_ids[candidate_keep],
            candidate_cell_populations=search.candidate_cell_populations[candidate_keep],
            probed_cell_ids=search.probed_cell_ids,
            probed_cell_scores=search.probed_cell_scores,
            centroid_scores=search.centroid_scores,
        )
    return {
        "top_idx": search.indices,
        "top_scores": search.scores,
        "sample_idx": search.indices,
        "sample_scores": search.scores,
        "sample_cell_populations": search.cell_populations,
        "full_scores": None,
        "all_scores": None,
        "search_result": search,
    }


def _resolve_ivf_nprobe(
    dense_index: SimpleIVF,
    retrieve_k: int,
    ivf_nprobe: int | None,
    oversample_factor: float = 2.0,
) -> int:
    """Choose nprobe so IVF sees enough candidates for top-k quality."""
    if ivf_nprobe is not None:
        return max(1, min(int(ivf_nprobe), dense_index.n_cells))

    target_candidates = max(
        int(np.ceil(retrieve_k * oversample_factor)),
        int(np.ceil(retrieve_k + dense_index.avg_population)),
    )
    needed = int(np.ceil(target_candidates / max(dense_index.avg_population, 1.0)))
    return max(
        dense_index.default_nprobe,
        min(needed, dense_index.n_cells),
    )


def _combine_vpt_sample_guidance(
    lexical_probs: np.ndarray,
    lexical_active: np.ndarray,
    density_prior: np.ndarray | None = None,
    *,
    neutral_prob: float = 0.5,
    lexical_floor: float = 0.5,
    min_lexical_mix: float = 0.35,
    max_lexical_mix: float = 0.85,
    max_logit: float = 10.0,
) -> np.ndarray:
    """Blend lexical and density hints into one benchmark-local VPT weight.

    Missing lexical evidence is treated as neutral instead of negative.
    When a density prior is available (IVF), combine both hints in logit
    space so the package calibrator still receives a single generic weight
    vector without index-specific logic.
    """
    lexical_probs = np.asarray(lexical_probs, dtype=np.float64)
    lexical_active = np.asarray(lexical_active, dtype=bool)
    guidance = np.full(len(lexical_probs), neutral_prob, dtype=np.float64)

    if np.any(lexical_active):
        guidance[lexical_active] = np.maximum(
            lexical_probs[lexical_active], lexical_floor,
        )

    if density_prior is None:
        return guidance

    density_prior = np.asarray(density_prior, dtype=np.float64)
    if density_prior.shape != guidance.shape:
        raise ValueError(
            "density_prior must have the same shape as lexical_probs",
        )

    active_ratio = float(np.mean(lexical_active)) if len(guidance) > 0 else 0.0
    lexical_mix = float(
        np.clip(
            min_lexical_mix + 0.5 * active_ratio,
            min_lexical_mix,
            max_lexical_mix,
        )
    )
    blended_logit = (
        lexical_mix * np.asarray(logit(_clamp_probability(guidance)))
        + (1.0 - lexical_mix)
        * np.asarray(logit(_clamp_probability(density_prior)))
    )
    return np.asarray(
        sigmoid(np.clip(blended_logit, -max_logit, max_logit)),
        dtype=np.float64,
    )


def _blend_log_odds_signal(
    base_log_odds: np.ndarray,
    refined_log_odds: np.ndarray,
    gate: float,
) -> np.ndarray:
    """Blend refined log-odds back toward a safer base signal."""
    gate = float(np.clip(gate, 0.0, 1.0))
    return (1.0 - gate) * np.asarray(base_log_odds, dtype=np.float64) + gate * np.asarray(
        refined_log_odds, dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Fusion methods (operate on candidate-level arrays)
#
# Every Bayesian fusion returns log-odds (or a fusion score), never a
# probability: the final sigmoid of the first version rounded every
# strong candidate to exactly 1 and left their order to tie-breaking.
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
    bayesian_log_odds: np.ndarray,
    dense_sim: np.ndarray,
) -> np.ndarray:
    """Probabilistic OR, 1 - (1-p_sparse)*(1-p_dense), as log-odds.

    A candidate with no lexical match has sparse log-odds -inf
    (probability 0), for which the disjunction reduces to the dense
    term.
    """
    return np.asarray(
        log_odds_or(np.column_stack([bayesian_log_odds, _dense_log_odds(dense_sim)])),
        dtype=np.float64,
    )


def _two_signal_conjunction(
    logit_dense: np.ndarray,
    logit_sparse: np.ndarray,
    has_sparse: np.ndarray,
) -> np.ndarray:
    """Log-odds conjunction of two signals with square-root-of-n scaling.

    Candidates with a lexical match get the conjunction of both signals,
    n^0.5 times the mean log-odds.  Candidates without one are scored by
    the dense term alone at half weight, as in the first version.
    """
    scale = 2.0 ** 0.5
    both = (0.5 * logit_dense + 0.5 * logit_sparse) * scale
    dense_only = 0.5 * logit_dense
    return np.where(has_sparse, both, dense_only)


def fusion_bayesian_bm25_logodds(
    bm25_scores: np.ndarray,
    bm25_alpha: float,
    bm25_beta: float,
    tfs: np.ndarray,
    doc_len_ratios: np.ndarray,
    dense_sim: np.ndarray,
    dense_median: float,
    dense_alpha: float,
    base_rate: float | None = None,
) -> np.ndarray:
    """Per-query calibrated log-odds fusion (Theorem 7.1.1).

    Full BM25 posterior is assembled directly in logit space:
        logit(L) + logit(prior) [+ logit(base_rate)]
    and the fused log-odds are returned as they are.
    """
    logit_d = dense_alpha * (dense_sim - dense_median)

    # BM25 evidence: logit(L) + logit(prior) in logit space directly
    logit_likelihood = bm25_alpha * (bm25_scores - bm25_beta)
    prior = BayesianProbabilityTransform.composite_prior(
        tfs, doc_len_ratios,
    )
    logit_s = logit_likelihood + np.asarray(logit(prior), dtype=np.float64)
    if base_rate is not None:
        logit_s = logit_s + float(logit(base_rate))

    return _two_signal_conjunction(logit_d, logit_s, bm25_scores > 0)


def fusion_logodds_local(
    bm25_scores: np.ndarray,
    dense_sim: np.ndarray,
    bm25_median: float,
    bm25_alpha: float,
    dense_median: float,
    dense_alpha: float,
) -> np.ndarray:
    """Symmetric log-odds fusion: both signals calibrated at candidate level.

    Raw BM25 scores are calibrated the same way as dense similarities,
    logit = alpha*(s - median) with the statistics of the candidate
    pool.  Returns fused log-odds.
    """
    logit_d = dense_alpha * (dense_sim - dense_median)
    logit_s = bm25_alpha * (bm25_scores - bm25_median)
    return _two_signal_conjunction(logit_d, logit_s, bm25_scores > 0)


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _arctan_log_odds(scores: np.ndarray, alpha: float = 5.0) -> np.ndarray:
    """Arctangent normalization, p = (2/pi) * arctan(alpha * s), as log-odds.

    Paper 3, Section 8.2 baseline.  Maps similarity scores to [0, 1]
    via the arctan function.  The scaling parameter alpha controls the
    steepness of the transition.  A non-positive similarity maps to a
    non-positive value, which is clamped to the smallest probability.
    """
    probabilities = (2.0 / np.pi) * np.arctan(alpha * np.asarray(scores, dtype=np.float64))
    return np.asarray(logit(_clamp_probability(probabilities)), dtype=np.float64)


def _global_kappa_log_odds(
    distances: np.ndarray,
    *,
    kappa: float,
    beta: float,
) -> np.ndarray:
    """Global sigmoid calibration, P = sigmoid(kappa * (beta - d)), as log-odds.

    Paper 3, Section 8.4 Stage 1 baseline.  Applies a corpus-level
    location and scale to cosine distances.

    Parameters
    ----------
    distances : array
        Cosine distances d = 1 - s.
    kappa : float
        Sigmoid steepness, 1 / std of the background distances.
    beta : float
        Sigmoid midpoint, the median background distance.
    """
    d = np.asarray(distances, dtype=np.float64)
    # Negative sign: smaller distance = higher log-odds
    return kappa * (beta - d)


def _compute_dense_calibration(dense_sim: np.ndarray) -> tuple[float, float]:
    """Per-query calibration: beta=median, alpha_eff=1/std of positive scores."""
    positive = dense_sim[dense_sim > 0]
    if len(positive) == 0:
        return 0.0, 1.0
    median = float(np.median(positive))
    std = float(np.std(positive))
    alpha_eff = 1.0 / std if std > 0 else 1.0
    return median, alpha_eff


def fusion_vpt_balanced(
    sparse_log_odds: np.ndarray,
    vpt_dense_log_odds: np.ndarray,
    *,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> np.ndarray:
    """Additive log-odds fusion with VPT-calibrated dense log-odds.

    This keeps the benchmark closer to the paper's posterior view:
    lexical log-odds evidence plus vector log-density-ratio evidence.
    The dense term is scaled down when it is more dispersed than the
    lexical one.  Returns fused log-odds; nothing is clipped.
    """
    logit_sparse = np.asarray(sparse_log_odds, dtype=np.float64)
    logit_dense = np.asarray(vpt_dense_log_odds, dtype=np.float64)

    sparse_std = max(float(np.std(logit_sparse)), 1e-6)
    dense_std = max(float(np.std(logit_dense)), 1e-6)
    dense_scale = dense_weight * min(1.0, sparse_std / dense_std)

    return sparse_weight * logit_sparse + dense_scale * logit_dense


# ---------------------------------------------------------------------------
# Auto-tuning functions
# ---------------------------------------------------------------------------

def _cached_log_odds(
    cache: dict,
    transform: BayesianProbabilityTransform,
) -> np.ndarray:
    """Lexical posterior log-odds of a cached candidate set (-inf if unmatched)."""
    active = cache["active"]
    log_odds = np.full(len(cache["union_idx"]), -np.inf, dtype=np.float64)
    if np.any(active):
        log_odds[active] = transform.score_to_log_odds(
            cache["active_scores"], cache["tfs"], cache["doc_len_ratios"],
        )
    return log_odds


def _compute_bayesian_run_from_cache(
    tune_cache: dict[str, dict],
    corpus_ids: list[str],
    transform: BayesianProbabilityTransform,
) -> dict[str, dict[str, float]]:
    """Build a pytrec_eval run dict from cached per-query data."""
    run: dict[str, dict[str, float]] = {}
    for qid, cache in tune_cache.items():
        run[qid] = ranked_run(
            [corpus_ids[i] for i in cache["union_idx"]],
            _cached_log_odds(cache, transform),
        )
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
        fused = np.asarray(
            balanced_log_odds_fusion(
                _with_unmatched_floor(_cached_log_odds(cache, transform)),
                cache["cand_dense"],
                weight,
                from_log_odds=True,
            ),
            dtype=np.float64,
        )
        run[qid] = ranked_run([corpus_ids[i] for i in cache["union_idx"]], fused)
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
        signals = np.column_stack([
            _with_unmatched_floor(_cached_log_odds(cache, transform)),
            _dense_log_odds(cache["cand_dense"]),
        ])
        fused = np.asarray(
            log_odds_fusion(signals, alpha=hybrid_alpha), dtype=np.float64,
        )
        run[qid] = ranked_run([corpus_ids[i] for i in cache["union_idx"]], fused)
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
    """Mean NDCG@k, MAP@k and Recall@k of a run over the queries of ``qrels``."""
    return mean_metrics(evaluate_run(qrels, run, k=k))


# ---------------------------------------------------------------------------
# Learned fusion weights (cross-validated over queries)
# ---------------------------------------------------------------------------


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    """Min-max normalize each column over the rows (one query's candidates)."""
    result = np.empty_like(x)
    for col in range(x.shape[1]):
        result[:, col] = _min_max_normalize(x[:, col])
    return result


def _attention_signals(
    cache: dict,
    dense_key: str,
    normalize: bool,
) -> np.ndarray:
    """Signal matrix of one query's candidates for the attention models.

    Columns are the lexical posterior log-odds (with base rate) and the
    dense log-odds stored under ``dense_key``.  With ``normalize`` each
    column is min-max normalized over the query's full candidate set.
    Normalizing here, before any rows are selected for training, makes
    the training rows see exactly the values the model is later scored
    on; normalizing the handful of sampled training rows among
    themselves would not.
    """
    signals = np.column_stack([cache["cand_log_odds_br"], cache[dense_key]])
    return _normalize_columns(signals) if normalize else signals


def _collect_attention_training_data(
    attn_cache: dict,
    train_qids: list[str],
    feature_key: str,
    dense_key: str,
    normalize: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (signals, labels, query features) from the training queries.

    Positives are the judged candidates of each query.  BEIR qrels list
    almost only relevant documents, so an equal number of unjudged
    candidates is sampled per query as negatives.
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    features: list[np.ndarray] = []

    for qid in train_qids:
        cache = attn_cache[qid]
        signals = _attention_signals(cache, dense_key, normalize)
        judged = np.flatnonzero(cache["cand_judged"])
        unjudged = np.flatnonzero(~cache["cand_judged"])
        relevant = cache["cand_labels"]

        n_neg = min(int(np.sum(relevant[judged] > 0)), len(unjudged))
        sampled = (
            rng.choice(unjudged, size=n_neg, replace=False)
            if n_neg > 0
            else np.empty(0, dtype=np.int64)
        )
        for j in np.concatenate([judged, sampled]):
            rows.append(signals[j])
            labels.append(float(relevant[j]))
            features.append(cache[feature_key])

    if not rows:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(features, dtype=np.float64),
    )


def cross_validated_attention_run(
    variant_name: str,
    attn_cache: dict,
    corpus_ids: list[str],
    feature_key: str,
    make_model,
    *,
    normalize: bool,
    dense_key: str = "cand_dense_log_odds",
    calibration: CalibrationAccumulator | None = None,
    n_folds: int = 5,
    learning_rate: float = 0.01,
    max_iterations: int = 500,
    seed: int = 42,
) -> dict[str, dict[str, float]] | None:
    """Train a learned fusion on k-1 folds of queries, score the held-out fold.

    Every query is scored by a model that never saw its relevance
    labels.  The models are trained and scored on log-odds
    (``from_log_odds=True``) and ranked on the fused log-odds.

    Parameters
    ----------
    variant_name : str
        Name used in log messages.
    attn_cache : dict
        Per-query candidate signals, labels and query features.
    corpus_ids : list of str
        Document ids, in corpus order.
    feature_key : str
        Which query feature vector of the cache to use.
    make_model : callable
        Zero-argument factory returning a fresh model with ``fit`` and
        ``__call__`` (``AttentionLogOddsWeights`` or
        ``MultiHeadAttentionLogOddsWeights``).
    normalize : bool
        Min-max normalize each signal over the query's candidate set
        before it reaches the model.
    dense_key : str
        Cache key of the dense log-odds column.
    calibration : CalibrationAccumulator or None
        When given and ``normalize`` is False, accumulates the held-out
        fused log-odds against the candidate labels.  Normalized inputs
        do not yield a posterior, so they are not accumulated.

    Returns
    -------
    dict or None
        The run, or None when there are too few queries or a fold has
        no usable training data.
    """
    # Candidate judgments select training rows, never the held-out queries.
    qids = list(attn_cache)
    if len(qids) < 2 * n_folds:
        print(f"  {variant_name} skipped (insufficient queries: {len(qids)})")
        return None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qids))
    run: dict[str, dict[str, float]] = {}
    total_train = 0

    for fold in range(n_folds):
        test_qids = [qids[i] for i in perm[fold::n_folds]]
        test_set = set(test_qids)
        train_qids = [qid for qid in qids if qid not in test_set]

        signals, labels, features = _collect_attention_training_data(
            attn_cache, train_qids, feature_key, dense_key, normalize, rng,
        )
        if (
            len(labels) < 10
            or float(np.sum(labels)) == 0.0
            or float(np.sum(1.0 - labels)) == 0.0
        ):
            print(f"  {variant_name} skipped (fold {fold} has insufficient data)")
            return None

        model = make_model()
        model.fit(
            signals, labels, features,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            from_log_odds=True,
        )
        total_train += len(labels)

        for qid in test_qids:
            cache = attn_cache[qid]
            fused = np.asarray(
                model(
                    _attention_signals(cache, dense_key, normalize),
                    cache[feature_key],
                    use_averaged=True,
                    from_log_odds=True,
                    return_log_odds=True,
                ),
                dtype=np.float64,
            )
            run[qid] = ranked_run(
                [corpus_ids[i] for i in cache["union_idx"]], fused,
            )
            if calibration is not None and not normalize:
                calibration.update(fused, cache["cand_labels"])

    print(
        f"  {variant_name} trained ({n_folds}-fold CV, {len(qids)} queries, "
        f"{total_train} total train pairs)"
    )
    return run


def cross_validated_platt_run(
    attn_cache: dict,
    corpus_ids: list[str],
    *,
    calibration: CalibrationAccumulator | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, float]] | None:
    """Platt scaling of the cosine similarity, cross-validated over queries.

    Each fold's calibrator is fitted on the dense result lists of the
    other folds: a retrieved document is a positive when it is judged
    relevant and a negative otherwise.  The fit uses the convergent
    Newton solver.  Platt scaling is a monotone map of the similarity,
    so its ranking is the dense ranking; what it changes is the
    calibration of the output.
    """
    qids = list(attn_cache)
    if len(qids) < 2 * n_folds:
        print(f"  Dense-Platt skipped (insufficient queries: {len(qids)})")
        return None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qids))
    run: dict[str, dict[str, float]] = {}

    for fold in range(n_folds):
        test_qids = [qids[i] for i in perm[fold::n_folds]]
        test_set = set(test_qids)
        train_scores = np.concatenate(
            [attn_cache[q]["dense_list_scores"] for q in qids if q not in test_set]
        )
        train_labels = np.concatenate(
            [attn_cache[q]["dense_list_labels"] for q in qids if q not in test_set]
        )
        if float(np.sum(train_labels)) == 0.0:
            print(f"  Dense-Platt skipped (fold {fold} has no relevant document)")
            return None

        calibrator = PlattCalibrator()
        calibrator.fit(train_scores, train_labels)

        for qid in test_qids:
            cache = attn_cache[qid]
            log_odds = np.asarray(
                calibrator.log_odds(cache["cand_dense"]), dtype=np.float64,
            )
            run[qid] = ranked_run(
                [corpus_ids[i] for i in cache["union_idx"]], log_odds,
            )
            if calibration is not None:
                calibration.update(log_odds, cache["cand_labels"])

    print(f"  Dense-Platt fitted ({n_folds}-fold CV, {len(qids)} queries)")
    return run


# ---------------------------------------------------------------------------
# Main benchmark pipeline
# ---------------------------------------------------------------------------

BASELINE_METHODS = [
    "BM25", "Dense", "Convex", "RRF",
    "Dense-Kappa", "Dense-Arctan", "Dense-Platt",
    "Bayesian-OR", "Bayesian-LogOdds", "Bayesian-LogOdds-Local",
    "Bayesian-LogOdds-BR", "Bayesian-Balanced",
    "Bayesian-Balanced-Mix", "Bayesian-Balanced-Elbow",
    "Bayesian-Gated-ReLU", "Bayesian-Gated-Swish", "Bayesian-Gated-GELU",
    "Bayesian-Gated-Swish-B2", "Bayesian-Gated-Softplus",
    "Bayesian-Attention", "Bayesian-Attn-Norm",
    "Bayesian-MultiHead", "Bayesian-MultiHead-Norm",
    "Bayesian-MultiField", "Bayesian-MultiField-Bal",
    "Bayesian-Vector-Balanced", "Bayesian-Vector-Softplus", "Bayesian-Vector-Attn",
    "VPT-DensityPrior", "VPT-BM25Weights",
    "VPT-BW-0.2", "VPT-BW-0.5", "VPT-BW-1.0", "VPT-BW-2.0",
]

TUNED_METHODS = [
    "Bayesian-Tuned", "Bayesian-Balanced-Tuned", "Bayesian-Hybrid-AND-Tuned",
]

# Methods whose fused output is a posterior.  Their log-odds are checked
# for calibration against the labels of the candidate pool.  Methods
# that min-max normalize their inputs output a fusion score, not a
# posterior, and are left out.
POSTERIOR_METHODS = [
    "Dense-Kappa", "Dense-Arctan", "Dense-Platt",
    "Bayesian-OR",
    "Bayesian-LogOdds", "Bayesian-LogOdds-Local", "Bayesian-LogOdds-BR",
    "Bayesian-Gated-ReLU", "Bayesian-Gated-Swish", "Bayesian-Gated-GELU",
    "Bayesian-Gated-Swish-B2", "Bayesian-Gated-Softplus",
    "Bayesian-Attention", "Bayesian-MultiHead",
    "Bayesian-MultiField",
    "Bayesian-Vector-Balanced", "Bayesian-Vector-Softplus",
    "VPT-DensityPrior", "VPT-BM25Weights",
    "VPT-BW-0.2", "VPT-BW-0.5", "VPT-BW-1.0", "VPT-BW-2.0",
]

GATED_VARIANTS = [
    ("Bayesian-Gated-ReLU", "relu", 1.0),
    ("Bayesian-Gated-Swish", "swish", 1.0),
    ("Bayesian-Gated-GELU", "gelu", 1.0),
    ("Bayesian-Gated-Swish-B2", "swish", 2.0),
    ("Bayesian-Gated-Softplus", "softplus", 1.0),
]


def _top_matched(raw_bm25: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-scoring documents with a non-zero BM25 score.

    A document that matches no query term is not part of the lexical
    result list, however few documents match.
    """
    matched_idx = np.flatnonzero(raw_bm25 > 0.0)
    order = np.argsort(-raw_bm25[matched_idx], kind="stable")[:k]
    return matched_idx[order].astype(np.int32)


def _labels_for(
    doc_ids: list[str],
    qrel_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Binary relevance and judged flags of a list of documents."""
    labels = np.fromiter(
        (1.0 if qrel_map.get(doc_id, 0) > 0 else 0.0 for doc_id in doc_ids),
        dtype=np.float64, count=len(doc_ids),
    )
    judged = np.fromiter(
        (doc_id in qrel_map for doc_id in doc_ids),
        dtype=bool, count=len(doc_ids),
    )
    return labels, judged


def run_dataset(
    dataset_dir: str,
    dataset_name: str,
    model_name: str,
    k: int = 10,
    retrieve_k: int = 1000,
    tune: bool = False,
    cache_dir: str | None = None,
    dense_backend: str = "exact",
    ivf_cells: int | None = None,
    ivf_nprobe: int | None = None,
    ivf_iterations: int = 10,
    ivf_seed: int = 42,
    ivf_background: str = "query",
    vpt_query_gating: bool = False,
    ignore_identical_ids: bool = True,
    identity: str = "text",
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Run all fusion methods on a single BEIR dataset.

    Follows retrieve-then-evaluate: retrieve top-R from each signal,
    fuse union candidates, evaluate top-k.

    When tune=True and the dataset ships training qrels, additionally
    runs auto-estimation, supervised learning and grid search on the
    training queries, and evaluates the tuned configurations on the
    test queries.

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
    dense_backend : str
        Dense retrieval backend: ``"exact"`` or ``"ivf"``.
    ivf_cells : int or None
        Number of IVF cells.  ``None`` uses an automatic heuristic.
    ivf_nprobe : int or None
        Number of IVF cells to probe at search time.
    ivf_iterations : int
        Lloyd iterations for the SimpleIVF k-means fit.
    ivf_seed : int
        RNG seed for the SimpleIVF build.
    ivf_background : str
        Source of the background density under the IVF backend.
        ``"query"`` (default) estimates, for each query,
        the mean and variance of its distance to the corpus from the
        centroid similarities and two stored sums per cell
        (``SimpleIVF.query_background``, Proposition 4.17). ``"residuals"``
        uses corpus-wide intra-cell residuals for the misspecified
        background ablation; these are a different random variable.
    vpt_query_gating : bool
        Blend the VPT-calibrated dense log-odds toward the plain cosine
        log-odds by an unsupervised separability gate.
    ignore_identical_ids : bool
        Exclude, for each query, the document whose id equals the
        query's id before retrieval, calibration and scoring.

    Returns
    -------
    (results, per_query, calibration, diagnostics)
        ``results[method][metric]`` is the mean over queries;
        ``per_query[method][metric]`` lists the per-query values in the
        order of the dataset's test queries; ``calibration[method]``
        holds the calibration summary of each posterior-producing method
        over the candidate pool; ``diagnostics["ivf_background"]``, under
        the IVF backend, compares the residuals recorded at construction
        with the distances of the queries to the corpus.
    """
    print(f"\n{'=' * 70}")
    print(f"  {dataset_name}")
    print(f"{'=' * 70}")

    dataset_start = time.perf_counter()
    # 1. Load data
    t0 = time.time()
    data = load_beir_dataset(dataset_dir, identity=identity)
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
    doc_position = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    n_identical = (
        sum(bool(data["excluded_doc_ids"][qid]) for qid in query_ids)
        if ignore_identical_ids else 0
    )
    print(f"  Loaded: {n_docs} docs, {n_queries} queries, "
          f"{sum(len(v) for v in qrels.values())} qrels, "
          f"identical ids excluded for {n_identical} queries "
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
        cache_path = os.path.join(cache_dir, dataset_name, f"{safe_model}-{identity}.npz")

    t0 = time.time()
    corpus_emb = encode_dense(
        corpus_texts, model_name, cache_path, cache_key="corpus",
        cache_context={"identity": identity, "identity_version": 1},
    )
    query_emb = encode_dense(
        query_texts, model_name, cache_path, cache_key="queries",
        cache_context={"identity": identity, "identity_version": 1},
    )
    print(f"  Dense encoded: corpus {corpus_emb.shape}, queries {query_emb.shape} "
          f"({time.time() - t0:.1f}s)")

    dense_index: SimpleIVF | None = None
    ivf_nprobe_eff: int | None = None
    if dense_backend == "ivf":
        t0 = time.time()
        dense_index = SimpleIVF.build(
            corpus_emb,
            n_cells=ivf_cells,
            max_iterations=ivf_iterations,
            seed=ivf_seed,
        )
        ivf_nprobe_eff = _resolve_ivf_nprobe(
            dense_index,
            retrieve_k=retrieve_k,
            ivf_nprobe=ivf_nprobe,
        )
        print(
            f"  Dense IVF built -- cells={dense_index.n_cells}, "
            f"nprobe={ivf_nprobe_eff}, avg_pop={dense_index.avg_population:.1f} "
            f"({time.time() - t0:.1f}s)"
        )
    # Distances of each query to the corpus as the index estimates them,
    # for comparison with the residuals recorded at construction.
    query_background_means: list[float] = []
    query_background_stds: list[float] = []

    preparation_seconds = time.perf_counter() - dataset_start
    # 5. Retrieve-then-evaluate
    t0 = time.time()
    qrels_pytrec: dict[str, dict[str, int]] = {}
    for qid, rels in qrels.items():
        qrels_pytrec[qid] = {did: score for did, score in rels.items()}

    methods = list(BASELINE_METHODS)
    if mf_scorer is None:
        methods = [
            m for m in methods
            if m not in ("Bayesian-MultiField", "Bayesian-MultiField-Bal")
        ]
    runs: dict[str, dict[str, dict[str, float]]] = {m: {} for m in methods}
    calibration = {
        m: CalibrationAccumulator(context=posterior_context(m))
        for m in POSTERIOR_METHODS if m in methods
    }

    # Pre-allocate rank arrays (reused per query, reset after each)
    bm25_rank_full = np.zeros(n_docs, dtype=np.float64)
    dense_rank_full = np.zeros(n_docs, dtype=np.float64)

    effective_R = min(retrieve_k, n_docs)

    # Per-query cache for tuning
    tune_cache: dict[str, dict] = {}

    # Per-query cache for the learned fusion models
    attn_cache: dict[str, dict] = {}
    vpt_gate_values: list[float] = []

    # Global kappa background: estimate corpus-level cosine distance stats
    # (Paper 3, Section 8.4 Stage 1).  Sample random query-document pairs.
    rng = np.random.default_rng(42)
    n_bg_sample = min(1000, n_docs)
    bg_sample_idx = rng.choice(n_docs, size=n_bg_sample, replace=False)
    bg_query_idx = rng.choice(n_queries, size=min(50, n_queries), replace=False)
    bg_distances_list: list[float] = []
    for qi in bg_query_idx:
        excluded_positions = (
            [doc_position[doc_id] for doc_id in data["excluded_doc_ids"][query_ids[qi]]]
            if ignore_identical_ids else []
        )
        sample = bg_sample_idx[~np.isin(bg_sample_idx, excluded_positions)]
        sims = query_emb[qi] @ corpus_emb[sample].T
        bg_distances_list.extend((1.0 - sims).tolist())
    bg_distances = np.array(bg_distances_list, dtype=np.float64)
    global_kappa_beta = float(np.median(bg_distances))
    global_kappa_std = float(np.std(bg_distances))
    global_kappa = 1.0 / global_kappa_std if global_kappa_std > 1e-12 else 1.0

    for q_idx in range(n_queries):
        qid = query_ids[q_idx]
        qtokens = query_tokens_list[q_idx]
        query_vec = query_emb[q_idx]
        qrel_map = qrels.get(qid, {})
        exclude_idx = (
            np.array(
                [doc_position[doc_id] for doc_id in data["excluded_doc_ids"][qid]], dtype=np.int64,
            )
            if ignore_identical_ids else None
        )

        # -- BM25 full-array scores --
        if qtokens:
            raw_bm25 = np.array(scorer._bm25.get_scores(qtokens), dtype=np.float64)
        else:
            raw_bm25 = np.zeros(n_docs, dtype=np.float64)
        if exclude_idx is not None:
            # The query's own document is not a candidate for any signal.
            raw_bm25[exclude_idx] = 0.0

        # -- Dense retrieval via exact scan or IVF --
        dense_info = _retrieve_dense_candidates(
            query_vec,
            corpus_emb,
            effective_R,
            dense_backend=dense_backend,
            dense_index=dense_index,
            ivf_nprobe=ivf_nprobe_eff,
            exclude_idx=exclude_idx,
        )
        dense_topR = np.asarray(dense_info["top_idx"], dtype=np.int32)
        dense_top_scores_run = np.asarray(
            dense_info["top_scores"], dtype=np.float64,
        )
        dense_sample_idx = np.asarray(
            dense_info["sample_idx"], dtype=np.int32,
        )
        dense_sample_scores = np.asarray(
            dense_info["sample_scores"], dtype=np.float64,
        )
        dense_sample_cell_populations = dense_info["sample_cell_populations"]
        dense_full_scores = dense_info["full_scores"]
        dense_search_result = dense_info["search_result"]

        # -- Retrieve top-R from each signal --
        bm25_topR = _top_matched(raw_bm25, effective_R)

        runs["BM25"][qid] = ranked_run(
            [corpus_ids[i] for i in bm25_topR], raw_bm25[bm25_topR],
        )
        dense_doc_ids = [corpus_ids[i] for i in dense_topR]
        runs["Dense"][qid] = ranked_run(dense_doc_ids, dense_top_scores_run)

        # -- Union candidates for hybrid methods --
        union_set = set(bm25_topR.tolist()) | set(dense_topR.tolist())
        union_idx = np.array(sorted(union_set), dtype=np.int32)
        cand_doc_ids = [corpus_ids[i] for i in union_idx]
        cand_labels, cand_judged = _labels_for(cand_doc_ids, qrel_map)

        # Similarities of the candidates.  An exact scan has them
        # already; recomputing them on the gathered rows can differ in
        # the last place, which is enough to reorder near-ties.
        dense_all_scores = dense_info["all_scores"]
        if dense_all_scores is not None:
            cand_dense = np.asarray(dense_all_scores[union_idx], dtype=np.float64)
        else:
            cand_dense = np.asarray(
                query_vec @ corpus_emb[union_idx].T, dtype=np.float64,
            )
        cand_bm25, active, cand_tfs, cand_doc_len_ratios = (
            _compute_bm25_features_for_docs(
                scorer, raw_bm25, union_idx, qtokens,
            )
        )
        active_scores = cand_bm25[active]
        tfs = cand_tfs[active]
        doc_len_ratios = cand_doc_len_ratios[active]

        # Lexical posterior log-odds (-inf for a candidate with no match)
        cand_log_odds = _apply_bm25_log_odds(
            scorer._transform, cand_bm25, active, cand_tfs, cand_doc_len_ratios,
        )
        cand_log_odds_br = _with_unmatched_floor(
            _apply_bm25_log_odds(
                scorer_br._transform, cand_bm25, active, cand_tfs, cand_doc_len_ratios,
            )
        )
        cand_log_odds_mix = _with_unmatched_floor(
            _apply_bm25_log_odds(
                scorer_br_mix._transform, cand_bm25, active, cand_tfs,
                cand_doc_len_ratios,
            )
        )
        cand_log_odds_elbow = _with_unmatched_floor(
            _apply_bm25_log_odds(
                scorer_br_elbow._transform, cand_bm25, active, cand_tfs,
                cand_doc_len_ratios,
            )
        )
        cand_dense_log_odds = _dense_log_odds(cand_dense)

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
        cand_dense_dist = 1.0 - cand_dense
        hybrid_scores = {
            "Convex": fusion_convex(cand_bm25, cand_dense),
            "RRF": fusion_rrf(cand_bm25_rank, cand_dense_rank),
            "Bayesian-OR": fusion_bayesian_bm25_or(cand_log_odds, cand_dense),
            "Bayesian-LogOdds": fusion_bayesian_bm25_logodds(
                cand_bm25, scorer._transform.alpha, scorer._transform.beta,
                cand_tfs, cand_doc_len_ratios,
                cand_dense, dense_median, dense_alpha,
            ),
            "Bayesian-LogOdds-Local": fusion_logodds_local(
                cand_bm25, cand_dense,
                bm25_median, bm25_alpha, dense_median, dense_alpha,
            ),
            "Bayesian-LogOdds-BR": fusion_bayesian_bm25_logodds(
                cand_bm25, scorer_br._transform.alpha, scorer_br._transform.beta,
                cand_tfs, cand_doc_len_ratios,
                cand_dense, dense_median, dense_alpha,
                base_rate=scorer_br.base_rate,
            ),
            "Bayesian-Balanced": balanced_log_odds_fusion(
                cand_log_odds_br, cand_dense, from_log_odds=True,
            ),
            "Bayesian-Balanced-Mix": balanced_log_odds_fusion(
                cand_log_odds_mix, cand_dense, from_log_odds=True,
            ),
            "Bayesian-Balanced-Elbow": balanced_log_odds_fusion(
                cand_log_odds_elbow, cand_dense, from_log_odds=True,
            ),
            # Paper 3, Section 8.2/8.4: vector calibration baselines
            "Dense-Kappa": _global_kappa_log_odds(
                cand_dense_dist, kappa=global_kappa, beta=global_kappa_beta,
            ),
            "Dense-Arctan": _arctan_log_odds(cand_dense),
        }

        # Gated log-odds conjunction: lexical + dense log-odds (Phase 5.3)
        gated_input = np.column_stack([cand_log_odds_br, cand_dense_log_odds])
        for name, gating, gating_beta in GATED_VARIANTS:
            hybrid_scores[name] = log_odds_fusion(
                gated_input, gating=gating, gating_beta=gating_beta,
            )

        # VPT-calibrated dense fusion (Paper 3, Theorem 3.1.1)
        # The exact backend fits the background f_G to the query's
        # distances over the corpus.  The IVF backend takes it from index
        # statistics: by default from query-to-centroid similarities
        # and two sums per cell (Section 4.7). The residual background
        # remains an explicitly named ablation.
        if dense_backend == "ivf" and dense_index is not None:
            assert isinstance(dense_search_result, IVFSearchResult)
            query_mu_G, query_sigma_G = dense_index.query_background(
                dense_search_result, query_vector=query_vec, excluded_indices=exclude_idx,
            )
            query_background_means.append(query_mu_G)
            query_background_stds.append(query_sigma_G)
            if ivf_background == "residuals":
                vpt = VectorProbabilityTransform.fit_background(
                    dense_index.background_distances,
                    background_model="gaussian", background_kind="nonrelevant",
                )
            elif ivf_background == "query":
                vpt = VectorProbabilityTransform(mu_G=query_mu_G, sigma_G=query_sigma_G)
            else:
                raise ValueError(
                    f"ivf_background must be 'residuals' or 'query', got {ivf_background!r}"
                )
        else:
            assert dense_full_scores is not None
            vpt = VectorProbabilityTransform.fit_background(
                1.0 - dense_full_scores, background_model="gaussian", background_kind="nonrelevant",
            )

        sample_distances = 1.0 - dense_sample_scores
        sample_bm25_scores, sample_active, sample_tfs, sample_doc_len_ratios = (
            _compute_bm25_features_for_docs(
                scorer, raw_bm25, dense_sample_idx, qtokens,
            )
        )
        # Importance weights are probabilities by definition.
        sample_bm25_probs_br = _apply_bm25_transform(
            scorer_br._transform,
            sample_bm25_scores,
            sample_active,
            sample_tfs,
            sample_doc_len_ratios,
        )
        sample_density_prior = None
        if dense_sample_cell_populations is not None and dense_index is not None:
            sample_density_prior = np.asarray(
                ivf_density_prior(
                    dense_sample_cell_populations,
                    dense_index.avg_population,
                ),
                dtype=np.float64,
            )
        sample_vpt_weights = _combine_vpt_sample_guidance(
            sample_bm25_probs_br,
            sample_active,
            sample_density_prior,
        )
        raw_vpt_dense_log_odds = reference_vector_log_odds(
            vpt, cand_dense_dist, sample_distances, sample_vpt_weights,
        )
        vpt_dense_balanced = raw_vpt_dense_log_odds
        if vpt_query_gating:
            if (
                dense_backend == "ivf"
                and dense_index is not None
                and isinstance(dense_search_result, IVFSearchResult)
            ):
                diagnostics = build_ivf_search_diagnostics(
                    dense_top_scores_run,
                    dense_search_result.cell_ids,
                    dense_search_result,
                    dense_index,
                )
            else:
                diagnostics = build_exact_search_diagnostics(dense_top_scores_run)
            vpt_gate = separability_gate(diagnostics)
            vpt_gate_values.append(vpt_gate)
            vpt_dense_balanced = _blend_log_odds_signal(
                cand_dense_log_odds, raw_vpt_dense_log_odds, vpt_gate,
            )

        hybrid_scores["Bayesian-Vector-Balanced"] = fusion_vpt_balanced(
            cand_log_odds_br, vpt_dense_balanced,
        )
        hybrid_scores["Bayesian-Vector-Softplus"] = log_odds_fusion(
            np.column_stack([cand_log_odds_br, raw_vpt_dense_log_odds]),
            gating="softplus",
        )

        # Section 6.5: compare sources of importance weights. Neither
        # source is established to satisfy conditional independence.
        #
        # VPT-DensityPrior: uses IVF cell density (Strategy 4.6.2) when
        # available, otherwise gap detection (Strategy 4.6.1) or
        # distance-based fallback.  Forces GMM estimation to ensure it
        # takes a different path from BM25-weighted KDE.
        if sample_density_prior is not None:
            density_prior_w = sample_density_prior
        else:
            gap_w = vpt._gap_weights(sample_distances)
            if gap_w is not None:
                density_prior_w = gap_w
            else:
                density_prior_w = VectorProbabilityTransform._distance_density_weights(
                    sample_distances,
                )
        hybrid_scores["VPT-DensityPrior"] = fusion_vpt_balanced(
            cand_log_odds_br,
            vpt.calibrate_with_sample(
                cand_dense_dist, sample_distances,
                weights=density_prior_w,
                method="gmm",
                return_log_odds=True,
            ),
        )

        # VPT-BM25Weights: uses BM25 probabilities as importance weights
        # (Section 4.3).  Forces KDE estimation to isolate the cross-modal
        # weighting contribution.
        bm25_only_weights = np.zeros(len(sample_distances), dtype=np.float64)
        if np.any(sample_active):
            bm25_only_weights[sample_active] = sample_bm25_probs_br[sample_active]
        sharpened_bm25 = VectorProbabilityTransform._sharpen_weights(bm25_only_weights)
        hybrid_scores["VPT-BM25Weights"] = fusion_vpt_balanced(
            cand_log_odds_br,
            vpt.calibrate_with_sample(
                cand_dense_dist, sample_distances,
                weights=sharpened_bm25,
                method="kde",
                return_log_odds=True,
            ),
        )

        # Paper 3, Section 8.4 Stage 7: Bandwidth ablation.
        # Evaluate bandwidth scaling factors c in {0.2, 0.5, 1.0, 2.0}
        # applied to Silverman bandwidth (Remark 4.4.2).
        for bw_factor, label in [
            (0.2, "VPT-BW-0.2"),
            (0.5, "VPT-BW-0.5"),
            (1.0, "VPT-BW-1.0"),
            (2.0, "VPT-BW-2.0"),
        ]:
            hybrid_scores[label] = fusion_vpt_balanced(
                cand_log_odds_br,
                vpt.calibrate_with_sample(
                    cand_dense_dist, sample_distances,
                    weights=sample_vpt_weights,
                    method="kde",
                    bandwidth_factor=bw_factor,
                    return_log_odds=True,
                ),
            )

        for method_name, scores in hybrid_scores.items():
            scores = np.asarray(scores, dtype=np.float64)
            runs[method_name][qid] = ranked_run(cand_doc_ids, scores)
            if method_name in calibration:
                calibration[method_name].update(scores, cand_labels)

        # MultiField: fused log-odds over all docs, take top-R
        if mf_scorer is not None:
            mf_log_odds_full = mf_scorer.get_log_odds(qtokens)
            if exclude_idx is not None:
                mf_log_odds_full[exclude_idx] = -np.inf
            mf_topR = np.argsort(-mf_log_odds_full)[:effective_R]
            if exclude_idx is not None:
                mf_topR = mf_topR[~np.isin(mf_topR, exclude_idx)]
            mf_doc_ids = [corpus_ids[i] for i in mf_topR]
            runs["Bayesian-MultiField"][qid] = ranked_run(
                mf_doc_ids, mf_log_odds_full[mf_topR],
            )
            calibration["Bayesian-MultiField"].update(
                mf_log_odds_full[mf_topR], _labels_for(mf_doc_ids, qrel_map)[0],
            )

            # MultiField-Bal: MultiField log-odds + dense via balanced_log_odds_fusion
            # Union of MultiField top-R and Dense top-R
            mf_union_set = set(mf_topR.tolist()) | set(dense_topR.tolist())
            mf_union_idx = np.array(sorted(mf_union_set), dtype=np.int32)
            mf_cand_dense = np.asarray(
                query_vec @ corpus_emb[mf_union_idx].T, dtype=np.float64,
            )
            runs["Bayesian-MultiField-Bal"][qid] = ranked_run(
                [corpus_ids[i] for i in mf_union_idx],
                np.asarray(
                    balanced_log_odds_fusion(
                        mf_log_odds_full[mf_union_idx], mf_cand_dense,
                        from_log_odds=True,
                    ),
                    dtype=np.float64,
                ),
            )

        # Cache for the learned fusion models: query features + candidate signals
        qlen = len(qtokens)
        bm25_hit_ratio = float(np.count_nonzero(raw_bm25)) / n_docs
        max_bm25_log = float(np.log1p(np.max(raw_bm25))) if np.any(raw_bm25 > 0) else 0.0

        # Dense-side features derived from the backend's retrieved results.
        top10_k = min(10, len(dense_top_scores_run))
        dense_top_scores = dense_top_scores_run[:top10_k]
        dense_top10_mean = float(np.mean(dense_top_scores)) if top10_k > 0 else 0.0
        dense_top10_std = float(np.std(dense_top_scores)) if top10_k > 1 else 0.0
        max_dense_log = (
            float(np.log1p(max(0.0, float(dense_top_scores_run[0]))))
            if len(dense_top_scores_run) > 0
            else 0.0
        )

        # Cross-signal feature: top-100 retrieval overlap (Jaccard)
        top100_k = min(100, n_docs)
        bm25_top100 = set(bm25_topR[:top100_k].tolist())
        dense_top100 = set(dense_topR[:top100_k].tolist())
        union_sz = len(bm25_top100 | dense_top100)
        overlap_ratio = float(len(bm25_top100 & dense_top100)) / union_sz if union_sz > 0 else 0.0

        attn_cache[qid] = {
            "union_idx": union_idx,
            "cand_log_odds_br": cand_log_odds_br,
            "cand_dense": cand_dense,
            "cand_dense_log_odds": cand_dense_log_odds,
            "vpt_dense_log_odds": raw_vpt_dense_log_odds,
            "cand_labels": cand_labels,
            "cand_judged": cand_judged,
            "dense_list_scores": dense_top_scores_run,
            "dense_list_labels": _labels_for(dense_doc_ids, qrel_map)[0],
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

    scoring_seconds = time.time() - t0
    print(f"  Scored {n_queries} queries x {len(methods)} methods, "
          f"R={effective_R} ({time.time() - t0:.1f}s)")
    if vpt_query_gating and vpt_gate_values:
        gate_arr = np.asarray(vpt_gate_values, dtype=np.float64)
        print(
            "  VPT query gate -- "
            f"mean={np.mean(gate_arr):.3f}, "
            f"p10={np.percentile(gate_arr, 10):.3f}, "
            f"p50={np.percentile(gate_arr, 50):.3f}, "
            f"p90={np.percentile(gate_arr, 90):.3f}"
        )

    # 5b. Methods fitted to relevance labels, cross-validated over queries.
    #
    #   Dense-Platt          : Platt scaling of the cosine similarity
    #   Bayesian-Attention   : query-dependent weights from 3 lexical features
    #   Bayesian-Attn-Norm   : + per-signal min-max normalization + 7 features
    #                          (3 BM25 + 3 dense + 1 cross-signal overlap)
    #   Bayesian-Vector-Attn : Attn-Norm with VPT-calibrated dense log-odds
    #   Bayesian-MultiHead   : 4 attention heads, averaged in log-odds
    #   Bayesian-MultiHead-Norm : MultiHead + normalization + 7 features
    t0 = time.time()
    learned_runs: dict[str, dict[str, dict[str, float]] | None] = {
        "Dense-Platt": cross_validated_platt_run(
            attn_cache, corpus_ids, calibration=calibration.get("Dense-Platt"),
        ),
    }
    learned_variants = [
        ("Bayesian-Attention", "features", False, "cand_dense_log_odds",
         lambda: AttentionLogOddsWeights(n_signals=2, n_query_features=3, alpha=0.5)),
        ("Bayesian-Attn-Norm", "features_rich", True, "cand_dense_log_odds",
         lambda: AttentionLogOddsWeights(n_signals=2, n_query_features=7, alpha=0.5)),
        ("Bayesian-Vector-Attn", "features_rich", True, "vpt_dense_log_odds",
         lambda: AttentionLogOddsWeights(n_signals=2, n_query_features=7, alpha=0.5)),
        ("Bayesian-MultiHead", "features", False, "cand_dense_log_odds",
         lambda: MultiHeadAttentionLogOddsWeights(
             n_heads=4, n_signals=2, n_query_features=3, alpha=0.5,
         )),
        ("Bayesian-MultiHead-Norm", "features_rich", True, "cand_dense_log_odds",
         lambda: MultiHeadAttentionLogOddsWeights(
             n_heads=4, n_signals=2, n_query_features=7, alpha=0.5,
         )),
    ]
    for name, feature_key, normalize, dense_key, make_model in learned_variants:
        learned_runs[name] = cross_validated_attention_run(
            name, attn_cache, corpus_ids, feature_key, make_model,
            normalize=normalize, dense_key=dense_key,
            calibration=calibration.get(name),
        )
    for name, run in learned_runs.items():
        if run is None:
            methods = [m for m in methods if m != name]
            runs.pop(name, None)
            calibration.pop(name, None)
        else:
            runs[name] = run
    learned_seconds = time.time() - t0
    print(f"  Learned variants done ({time.time() - t0:.1f}s)")

    # 6. Evaluate baselines
    results: dict[str, dict[str, float]] = {}
    per_query: dict[str, dict[str, list[float]]] = {}

    def record(method_name: str, run: dict[str, dict[str, float]]) -> None:
        values = evaluate_run(qrels_pytrec, run, k=k)
        results[method_name] = mean_metrics(values)
        per_query[method_name] = {
            metric: [by_query[qid] for qid in query_ids]
            for metric, by_query in values.items()
        }

    for method_name in methods:
        record(method_name, runs[method_name])

    # 6b. Calibration diagnostics for posterior-producing methods
    calibration_summaries = {
        method: accumulator.summary()
        for method, accumulator in calibration.items()
        if accumulator.count > 0
    }
    if calibration_summaries:
        print_calibration_section(calibration_summaries)

    # 7. Tuning (if enabled)
    train_qrels_path = os.path.join(dataset_dir, "qrels", "train.tsv")
    train_qrels = (
        training_qrels_for_dataset(data, load_qrels_from_tsv(train_qrels_path))
        if tune and os.path.exists(train_qrels_path) else {}
    )
    tuning_diagnostics = {
        "enabled": tune,
        "status": "not_requested",
        "training_split": "qrels/train.tsv",
        "train_query_ids": sorted(train_qrels),
        "test_query_ids": query_ids,
    }
    if set(train_qrels).intersection(query_ids):
        raise ValueError("tuning training queries overlap evaluated queries")
    if tune and not train_qrels:
        tuning_diagnostics.update(
            status="unavailable", reason="no_disjoint_official_training_queries",
        )
        # Tuning on the test queries and then reporting test performance
        # would evaluate the parameters on the labels they were fitted to.
        print("\n  --- Auto-tuning skipped: no disjoint canonical train queries ---")
    elif tune:
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

        # Step 3: Supervised learning on the train qrels
        print(f"  [Tune] Step 3: Supervised learning from {train_qrels_path}...")
        print(f"    {len(train_qrels)} train queries loaded")
        tuned_alpha, tuned_beta = learn_parameters_from_qrels(
            scorer, corpus_ids,
            all_query_ids, all_query_texts,
            train_qrels,
        )
        print(f"    learned alpha={tuned_alpha:.4f}, beta={tuned_beta:.4f}")

        # Step 4: Grid search on the train queries
        print(f"  [Tune] Step 4: Grid search (optimizing NDCG@{k})...")
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
            cache_context={"identity": identity, "identity_version": 1, "split": "train"},
        )

        train_tune_cache: dict[str, dict] = {}
        for tq_idx, (tqid, tqtokens) in enumerate(zip(train_qids, train_tokens, strict=True)):
            if not tqtokens:
                continue
            canonical_query = data["query_alias_to_id"].get(tqid, tqid)
            train_exclude = (
                np.array([
                    doc_position[doc_id]
                    for doc_id in data["excluded_doc_ids"].get(canonical_query, [])
                ], dtype=np.int64) if ignore_identical_ids else None
            )
            raw_bm25 = np.array(scorer._bm25.get_scores(tqtokens), dtype=np.float64)
            if train_exclude is not None:
                raw_bm25[train_exclude] = 0.0
            query_vec = train_emb[tq_idx]
            dense_info = _retrieve_dense_candidates(
                query_vec,
                corpus_emb,
                effective_R,
                dense_backend=dense_backend,
                dense_index=dense_index,
                ivf_nprobe=ivf_nprobe_eff,
                exclude_idx=train_exclude,
            )
            bm25_topR = _top_matched(raw_bm25, effective_R)
            dense_topR = np.asarray(dense_info["top_idx"], dtype=np.int32)
            union_set = set(bm25_topR.tolist()) | set(dense_topR.tolist())
            union_idx = np.array(sorted(union_set), dtype=np.int32)
            cand_dense = np.asarray(
                query_vec @ corpus_emb[union_idx].T, dtype=np.float64,
            )
            cand_bm25, active, cand_tfs, cand_doc_len_ratios = (
                _compute_bm25_features_for_docs(
                    scorer, raw_bm25, union_idx, tqtokens,
                )
            )
            train_tune_cache[tqid] = {
                "union_idx": union_idx,
                "cand_dense": cand_dense,
                "active": active.copy(),
                "active_scores": cand_bm25[active].copy(),
                "tfs": cand_tfs[active].copy(),
                "doc_len_ratios": cand_doc_len_ratios[active].copy(),
            }

        tuned = grid_search_tuned(
            scorer, corpus_ids, train_tune_cache, train_qrels,
            tuned_alpha, tuned_beta, auto_base_rate, k,
        )

        # Step 5: Print summary
        print_tuned_summary(
            tuned, auto_alpha, auto_beta, auto_base_rate,
            learned=True, has_dense=True,
        )

        # Step 6: Evaluate tuned configurations on test qrels
        print("  [Tune] Evaluating tuned methods on test qrels...")

        tuned_transform = BayesianProbabilityTransform(
            alpha=tuned["alpha"], beta=tuned["beta"],
            base_rate=tuned["base_rate"],
        )

        # Bayesian-Tuned
        runs["Bayesian-Tuned"] = _compute_bayesian_run_from_cache(
            tune_cache, corpus_ids, tuned_transform,
        )
        record("Bayesian-Tuned", runs["Bayesian-Tuned"])

        # Balanced-Tuned
        if tuned["fusion_weight"] is not None:
            runs["Bayesian-Balanced-Tuned"] = _compute_balanced_run_from_cache(
                tune_cache, corpus_ids, tuned_transform, tuned["fusion_weight"],
            )
            record("Bayesian-Balanced-Tuned", runs["Bayesian-Balanced-Tuned"])

        # Hybrid-AND-Tuned
        if tuned["hybrid_alpha"] is not None:
            runs["Bayesian-Hybrid-AND-Tuned"] = _compute_hybrid_and_run_from_cache(
                tune_cache, corpus_ids, tuned_transform, tuned["hybrid_alpha"],
            )
            record("Bayesian-Hybrid-AND-Tuned", runs["Bayesian-Hybrid-AND-Tuned"])

        tuning_diagnostics.update(status="complete", parameters=tuned)

    diagnostics: dict = {
        "tuning": tuning_diagnostics,
        "identity": identity, "identity_version": data["identity_version"],
        "dataset_fingerprint": data["dataset_fingerprint"],
        "source_document_count": data["source_document_count"],
        "source_query_count": data["source_query_count"],
        "canonical_document_count": n_docs, "canonical_query_count": n_queries,
        "reference_population": "query_excluded_corpus",
        "relevance_target": "judged_relevance",
        "reference_vector_configuration": {
            "scope": "Table 1 automatic reference configuration",
            "weights": "gap_then_explicitly_sharpened_external_guidance",
            "bandwidth_factor": 2.0,
            "background_kind": "nonrelevant",
        },
        "timing": {
            "preparation_seconds": preparation_seconds,
            "all_label_free_variants_seconds": scoring_seconds,
            "all_learned_variants_seconds": learned_seconds,
            "label_free_variants_per_query_ms": 1000.0 * scoring_seconds / n_queries,
            "total_seconds": time.perf_counter() - dataset_start,
            "scope": "whole_benchmark_with_embedding_cache_state_recorded_by_logs",
        },
    }
    if dense_index is not None and query_background_means:
        residuals = np.asarray(dense_index.background_distances, dtype=np.float64)
        diagnostics["ivf_background"] = {
            # Document-to-centroid distances recorded at construction.
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            # Query-to-document distances over the corpus, per query.
            "query_distance_mean": float(np.mean(query_background_means)),
            "query_distance_std": float(np.mean(query_background_stds)),
        }
        print(
            "  IVF background -- residuals: "
            f"mean={diagnostics['ivf_background']['residual_mean']:.3f}, "
            f"std={diagnostics['ivf_background']['residual_std']:.3f}; "
            "query-to-document distances: "
            f"mean={diagnostics['ivf_background']['query_distance_mean']:.3f}, "
            f"std={diagnostics['ivf_background']['query_distance_std']:.3f}"
        )

    return results, per_query, calibration_summaries, diagnostics


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def print_calibration_section(
    calibration: dict[str, dict[str, float]],
) -> None:
    """Print calibration diagnostics for posterior-producing methods.

    Each method's fused log-odds are compared with the relevance labels
    of the whole candidate pool, where an unjudged document counts as
    not relevant.  Restricting the comparison to judged documents, as
    the first version did, leaves a pool that is almost entirely
    relevant for datasets whose qrels list only relevant documents; on
    such a pool any predictor that outputs values near 1 looks
    calibrated.

    Besides ECE and the Brier score, the table reports log loss, the
    skill of log loss relative to the constant-prevalence predictor
    (negative: worse than knowing the base rate), and the ratio of
    expected to observed relevant documents (1 is calibration in the
    large).
    """
    print("\n  --- Calibration Diagnostics (candidate pool, unjudged = not relevant) ---")
    print(
        f"  {'Method':<26}  {'ECE':>8}  {'Brier':>8}  {'LogLoss':>9}"
        f"  {'LL skill':>9}  {'Exp/Obs':>10}  {'Samples':>9}"
    )
    print(
        f"  {'-' * 26}  {'-' * 8}  {'-' * 8}  {'-' * 9}"
        f"  {'-' * 9}  {'-' * 10}  {'-' * 9}"
    )

    for method, summary in calibration.items():
        print(
            f"  {method:<26}  {summary['ece']:>8.4f}  {summary['brier']:>8.4f}"
            f"  {summary['log_loss']:>9.4f}  {summary['log_loss_skill']:>9.3f}"
            f"  {summary['expected_over_observed']:>10.3g}  {int(summary['n']):>9}"
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
            if len(averages[method]) == len(datasets):
                avg = float(np.mean(averages[method]))
                row += f"  {avg:>{col_w - 1}.2f}%"
            else:
                coverage = f"n/a ({len(averages[method])}/{len(datasets)})"
                row += f"  {coverage:>{col_w}}"
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
        if len(deltas) == len(datasets):
            avg_delta = float(np.mean(deltas))
            sign = "+" if avg_delta >= 0 else ""
            print(f"  {method:<20}  avg: {sign}{avg_delta:.2f}")
        else:
            print(f"  {method:<20}  n/a ({len(deltas)}/{len(datasets)} datasets)")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-signal BEIR benchmark: BM25 plus one dense encoder"
    )
    parser.add_argument("--identity", choices=["id", "text"], default="text")
    parser.add_argument(
        "-d", "--beir-dir", default=str(DEFAULT_BEIR_DIR),
        help="Path to BEIR data directory",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DEFAULT_DATASETS),
        help="BEIR datasets to benchmark (default: all 5)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_DENSE_MODEL,
        help="Sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--dense-backend",
        choices=["exact", "ivf"],
        default="exact",
        help="Dense retrieval backend (default: exact)",
    )
    parser.add_argument(
        "--ivf-cells", type=int, default=None,
        help="Number of IVF cells when --dense-backend ivf (default: auto)",
    )
    parser.add_argument(
        "--ivf-nprobe", type=int, default=None,
        help="Number of IVF cells to probe at query time (default: auto)",
    )
    parser.add_argument(
        "--ivf-iterations", type=int, default=10,
        help="K-means iterations for SimpleIVF (default: 10)",
    )
    parser.add_argument(
        "--ivf-seed", type=int, default=42,
        help="Random seed for SimpleIVF build (default: 42)",
    )
    parser.add_argument(
        "--ivf-background", choices=["residuals", "query"], default="query",
        help=(
            "Background density under the IVF backend: 'query' (per-query "
            "estimate from centroid similarities; default) or 'residuals' "
            "(intra-cell residuals, a misspecified-background ablation)"
        ),
    )
    parser.add_argument(
        "--vpt-query-gating", action="store_true",
        help="Apply unsupervised VPT gating to Bayesian-Vector-Balanced",
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
        "--keep-identical-ids", action="store_true",
        help=(
            "Keep the document whose id equals the query's id as a candidate. "
            "By default it is excluded before retrieval, calibration and scoring."
        ),
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.top_k < 1 or args.retrieve_k < 1:
        raise ValueError("evaluation and retrieval depths must be positive")

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
    print(f"  Dense backend: {args.dense_backend}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  k={args.top_k}, R={args.retrieve_k}")
    if args.dense_backend == "ivf":
        ivf_cells_label = "auto" if args.ivf_cells is None else str(args.ivf_cells)
        ivf_nprobe_label = "auto" if args.ivf_nprobe is None else str(args.ivf_nprobe)
        print(
            f"  IVF: cells={ivf_cells_label}, nprobe={ivf_nprobe_label}, "
            f"iters={args.ivf_iterations}, seed={args.ivf_seed}, "
            f"background={args.ivf_background}"
        )
    if args.vpt_query_gating:
        print("  VPT query gating: enabled")
    print(
        "  Identical ids: "
        + ("kept as candidates" if args.keep_identical_ids else "excluded")
    )
    print(f"  Cache: {cache_label}")
    if args.download:
        print("  Download: enabled")
    print("=" * 70)

    # Download datasets if requested
    if args.download:
        print("\n--- Checking / downloading datasets ---")
        for ds_name in args.datasets:
            download_beir_dataset(ds_name, args.beir_dir)

    validate_dataset_inputs(args.beir_dir, args.datasets)

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    all_per_query: dict[str, dict[str, dict[str, list[float]]]] = {}
    all_calibration: dict[str, dict[str, dict[str, float]]] = {}
    all_diagnostics: dict[str, dict[str, dict[str, float]]] = {}

    for ds_name in args.datasets:
        ds_dir = os.path.join(args.beir_dir, ds_name)
        (
            all_results[ds_name], all_per_query[ds_name],
            all_calibration[ds_name], all_diagnostics[ds_name],
        ) = run_dataset(
            ds_dir, ds_name, args.model,
            k=args.top_k, retrieve_k=args.retrieve_k,
            tune=args.tune,
            cache_dir=effective_cache_dir,
            dense_backend=args.dense_backend,
            ivf_cells=args.ivf_cells,
            ivf_nprobe=args.ivf_nprobe,
            ivf_iterations=args.ivf_iterations,
            ivf_seed=args.ivf_seed,
            ivf_background=args.ivf_background,
            vpt_query_gating=args.vpt_query_gating,
            ignore_identical_ids=not args.keep_identical_ids,
            identity=args.identity,
        )

        if args.output:
            # Written after every dataset so that a long run can be inspected
            # (and is not lost) before it finishes.
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "protocol": {
                            "signal_count": 2,
                            "signals": ["BM25", args.model],
                            "model": args.model,
                            "datasets_requested": args.datasets,
                            "datasets_completed": list(all_results),
                            "status": (
                                "complete" if len(all_results) == len(args.datasets) else "running"
                            ),
                            "identity": args.identity,
                            "identity_version": 1,
                            "reference_population": "query_excluded_corpus",
                            "relevance_target": "judged_relevance",
                            "dense_backend": args.dense_backend,
                            "ivf_background": (
                                args.ivf_background if args.dense_backend == "ivf" else None
                            ),
                            "k": args.top_k,
                            "retrieve_k": args.retrieve_k,
                            "identical_ids_excluded": not args.keep_identical_ids,
                            "ranking": "log-odds, double precision",
                            "learned_weights": "5-fold cross-validation over queries",
                            "tuning_enabled": args.tune,
                        },
                        "results": all_results,
                        "per_query": all_per_query,
                        "calibration": all_calibration,
                        "diagnostics": all_diagnostics,
                    },
                    f,
                )

    print_results_table(all_results, k=args.top_k)

    if args.output:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
