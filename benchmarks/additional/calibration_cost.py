#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Exhaustive query-to-posterior cost, with score and calibration stages split.

This measures the implemented NumPy pipeline, not pruned WAND or an ANN
service. Corpus embedding and indexing are setup costs. Query encoding
is fresh by default; --query-encoding cached measures its exclusion
explicitly. All queries are measured by default; --query-limit selects
a recorded uniform sample after model warmup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.consistency import dilute_offset, solve_offset
from bayesian_bm25.count_calibration import calibrate_count
from bayesian_bm25.query_calibration import calibrate_query
from bayesian_bm25.scorer import BayesianBM25Scorer
from bayesian_bm25.tail_scale import tail_scale
from benchmarks.evidence_protocol import TAIL_DEPTH
from benchmarks.hybrid_beir import _dense_encoder, encode_dense, load_beir_dataset, tokenize_texts
from benchmarks.query_signals import QuerySignals, _top_positions
from benchmarks.reevaluation import unified_posterior


def measure_scores(
    scorer: BayesianBM25Scorer,
    tokens: list[str],
    corpus_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    excluded: np.ndarray,
    *,
    count: float = 1.0,
    ceiling: float = 0.6,
    retrieve_k: int = 1000,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """Time literal scoring and calibration; return numerical accuracy too."""
    timings = {}
    begin = time.perf_counter()
    raw = (
        np.asarray(scorer._bm25.get_scores(tokens), dtype=np.float64)
        if tokens
        else np.zeros(
            corpus_embeddings.shape[0],
            dtype=np.float64,
        )
    )
    checkpoint = time.perf_counter()
    timings["bm25_scoring_ms"] = 1000.0 * (checkpoint - begin)
    dense = np.asarray(corpus_embeddings @ query_embedding, dtype=np.float64)
    after_dense = time.perf_counter()
    timings["dense_scoring_ms"] = 1000.0 * (after_dense - checkpoint)
    doc_index = np.delete(np.arange(len(raw)), excluded)
    raw, dense = raw[doc_index], dense[doc_index]
    lexical_top = _top_positions(raw, retrieve_k, raw > 0.0)
    dense_top = _top_positions(dense, retrieve_k, None)
    candidates = np.union1d(lexical_top, dense_top)
    lexical_rank, dense_rank = np.zeros(len(raw), dtype=int), np.zeros(len(raw), dtype=int)
    lexical_rank[lexical_top] = np.arange(1, len(lexical_top) + 1)
    dense_rank[dense_top] = np.arange(1, len(dense_top) + 1)
    after_retrieval = time.perf_counter()
    timings["exclusion_and_top_lists_ms"] = 1000.0 * (after_retrieval - after_dense)
    context = CalibrationContext(
        reference_population="query_excluded_corpus",
        count_source="cost_experiment_convention",
        slope_source="convention",
        ceiling_source="cost_experiment_convention",
    )
    lexical = calibrate_query(
        raw,
        match_mask=raw > 0.0,
        expected_relevant=count,
        fallback_scale=1.0 / scorer._transform.alpha,
        context=context,
    )
    after_lexical = time.perf_counter()
    timings["lexical_scale_and_count_ms"] = 1000.0 * (after_lexical - after_retrieval)
    unit = tail_scale(dense[dense_top], k=TAIL_DEPTH)
    if unit <= 0.0:
        raise ValueError("the dense result list has no tail scale")
    evidence = (lexical.slope * raw + dense / unit) / np.sqrt(2.0)
    after_units = time.perf_counter()
    timings["dense_tail_and_pooling_ms"] = 1000.0 * (after_units - after_lexical)
    model = calibrate_count(evidence, count, ceiling=ceiling, context=context)
    logits = model.log_odds(evidence)
    probabilities = expit(logits)
    head = candidates[np.argsort(-model.ranking_score(evidence[candidates]), kind="stable")[:10]]
    after_count = time.perf_counter()
    timings["fused_count_and_output_ms"] = 1000.0 * (after_count - after_units)
    timings["scoring_to_capped_output_ms"] = 1000.0 * (after_count - begin)

    sig = QuerySignals(
        0,
        "cost_query",
        doc_index,
        raw,
        dense,
        np.zeros(len(raw)),
        candidates,
        lexical_rank[candidates],
        dense_rank[candidates],
    )
    prior = math.log(count) - math.log(len(raw) - count)
    density_begin = time.perf_counter()
    density_scores, _ = unified_posterior(sig, lexical.log_odds(raw), prior)
    candidates[np.argsort(-density_scores, kind="stable")[:10]]
    density_ms = 1000.0 * (time.perf_counter() - density_begin)
    timings["gaussian_ratio_kde_and_fusion_ms"] = density_ms
    timings["scoring_to_density_ratio_output_ms"] = 1000.0 * (after_lexical - begin) + density_ms
    marginal_begin = time.perf_counter()
    marginal_scores, _ = unified_posterior(
        sig,
        lexical.log_odds(raw),
        prior,
        posterior="exact",
        background="marginal",
    )
    candidates[np.argsort(-marginal_scores, kind="stable")[:10]]
    marginal_ms = 1000.0 * (time.perf_counter() - marginal_begin)
    timings["marginal_kde_and_fusion_ms"] = marginal_ms
    timings["scoring_to_marginal_output_ms"] = 1000.0 * (after_lexical - begin) + marginal_ms

    # These count-only comparisons already have all evidence scores.
    # Their times must not be presented as a pruned retrieval pipeline.
    approximation_begin = time.perf_counter()
    exact = solve_offset(evidence, count / ceiling)
    timings["exact_offset_only_ms"] = 1000.0 * (time.perf_counter() - approximation_begin)
    approximation_begin = time.perf_counter()
    dilute = dilute_offset(evidence, count / ceiling)
    timings["dilute_offset_only_ms"] = 1000.0 * (time.perf_counter() - approximation_begin)
    approximation_begin = time.perf_counter()
    top = _top_positions(evidence, retrieve_k, None)
    rest = np.flatnonzero(~np.isin(np.arange(len(raw)), top))
    sample = rng.choice(rest, size=min(1000, len(rest)), replace=False)
    sampled_scores = np.concatenate([evidence[top], evidence[sample]])
    weights = np.concatenate(
        [
            np.ones(len(top)),
            np.full(len(sample), len(rest) / len(sample)) if len(sample) else [],
        ]
    )
    sampled = solve_offset(sampled_scores, count / ceiling, weights=weights)
    timings["top_and_tail_sample_offset_ms"] = 1000.0 * (time.perf_counter() - approximation_begin)
    return timings, {
        "relative_count_error": abs(float(np.sum(probabilities)) / count - 1.0),
        "sample_offset_error": sampled - exact,
        "sample_relative_count_error": ceiling * float(np.sum(expit(evidence - sampled))) / count
        - 1.0,
        "dilute_offset_error": dilute - exact,
        "n_candidates": len(candidates),
        "head_count": len(head),
    }


def _distribution(values: list[float], *, uncertainty: bool = False) -> dict:
    array = np.asarray(values)
    result = {
        "mean": float(np.mean(array)),
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "total": float(np.sum(array)),
    }
    if uncertainty:
        rng = np.random.default_rng(42)
        resampled = array[rng.integers(0, len(array), size=(1000, len(array)))]
        quantiles = np.percentile(resampled, [50.0, 95.0], axis=1)
        result["p50_95ci"] = np.percentile(quantiles[0], [2.5, 97.5]).tolist()
        result["p95_95ci"] = np.percentile(quantiles[1], [2.5, 97.5]).tolist()
    return result


def run_dataset(
    dataset_dir: str,
    name: str,
    model_name: str,
    cache_dir: str,
    *,
    query_encoding: str = "online",
    retrieve_k: int = 1000,
    identity: str = "text",
    query_limit: int | None = None,
) -> dict:
    started = time.perf_counter()
    data = load_beir_dataset(dataset_dir, identity=identity)
    corpus_tokens = tokenize_texts(data["corpus_texts"])
    scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
    scorer.index(corpus_tokens, show_progress=False)
    cache = os.path.join(cache_dir, name, f"{model_name.replace('/', '_')}-{identity}.npz")
    context = {"identity": identity, "identity_version": 1}
    corpus_embeddings = encode_dense(
        data["corpus_texts"],
        model_name,
        cache,
        cache_key="corpus",
        cache_context=context,
    )
    if query_encoding == "cached":
        query_embeddings = encode_dense(
            data["query_texts"],
            model_name,
            cache,
            cache_key="queries",
            cache_context=context,
        )
        device = "cached_query_vectors"
        encoder = None
    else:
        encoder = _dense_encoder(model_name)
        device = str(encoder.device)
        # Warmup is a setup cost, explicitly outside the per-query samples.
        encoder.encode([data["query_texts"][0]], normalize_embeddings=True, show_progress_bar=False)
    setup_seconds = time.perf_counter() - started
    stages: dict[str, list[float]] = {}
    errors: dict[str, list[float]] = {}
    doc_position = {doc_id: i for i, doc_id in enumerate(data["corpus_ids"])}
    rng = np.random.default_rng(42)
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive")
    selected = np.arange(len(data["query_ids"]))
    if query_limit is not None and query_limit < len(selected):
        selected = np.sort(np.random.default_rng(42).choice(selected, query_limit, replace=False))
    for number, qi in enumerate(selected):
        qid, text = data["query_ids"][qi], data["query_texts"][qi]
        query_begin = time.perf_counter()
        tokens = tokenize_texts([text])[0]
        tokenized = time.perf_counter()
        if encoder is None:
            vector = query_embeddings[qi]
        else:
            vector = np.asarray(
                encoder.encode(
                    [text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )[0]
        encoded = time.perf_counter()
        excluded = np.array(
            [doc_position[d] for d in data["excluded_doc_ids"][qid]], dtype=np.int64
        )
        scoring_begin = time.perf_counter()
        timing, accuracy = measure_scores(
            scorer,
            tokens,
            corpus_embeddings,
            vector,
            excluded,
            retrieve_k=retrieve_k,
            rng=rng,
        )
        timing["query_tokenization_ms"] = 1000.0 * (tokenized - query_begin)
        timing["query_encoding_ms"] = 1000.0 * (encoded - tokenized)
        preprocessing_ms = 1000.0 * (scoring_begin - query_begin)
        timing["end_to_end_log_linear_ms"] = (
            preprocessing_ms + timing["scoring_to_capped_output_ms"]
        )
        timing["end_to_end_density_ratio_ms"] = (
            preprocessing_ms + timing["scoring_to_density_ratio_output_ms"]
        )
        timing["end_to_end_marginal_ms"] = (
            preprocessing_ms + timing["scoring_to_marginal_output_ms"]
        )
        for key, value in timing.items():
            stages.setdefault(key, []).append(value)
        for key, value in accuracy.items():
            errors.setdefault(key, []).append(value)
        if (number + 1) % 25 == 0:
            print(f"  {name}: cost {number + 1}/{len(selected)}", flush=True)
    return {
        "n_docs": len(data["corpus_ids"]),
        "n_queries": len(data["query_ids"]),
        "n_evaluated_queries": len(selected),
        "evaluated_query_ids": [data["query_ids"][qi] for qi in selected],
        "query_sampling": "all" if query_limit is None else "uniform_without_replacement_seed42",
        "identity": identity,
        "dataset_fingerprint": data["dataset_fingerprint"],
        "reference_population": "query_excluded_corpus",
        "query_encoding": query_encoding,
        "encoder_device": device,
        "setup_seconds": setup_seconds,
        "expected_relevant": 1.0,
        "ceiling": 0.6,
        "retrieve_k": retrieve_k,
        "tail_depth": TAIL_DEPTH,
        "timings_ms": {key: _distribution(v, uncertainty=True) for key, v in stages.items()},
        "diagnostics": {key: _distribution(v) for key, v in errors.items()},
        "max_relative_count_error": max(errors["relative_count_error"]),
        "per_query_timings_ms": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--beir-dir", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--datasets", nargs="+", default=["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"]
    )
    parser.add_argument("--identity", choices=["id", "text"], default="text")
    parser.add_argument("--query-encoding", choices=["online", "cached"], default="online")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Uniform timing sample; quality experiments remain exhaustive",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir or os.path.join(args.beir_dir, ".cache", "embeddings")
    payload = {
        "model": args.model,
        "identity": args.identity,
        "identity_version": 1,
        "scope": "exhaustive_query_processing_after_corpus_index_and_model_warmup",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "results": {},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        payload["results"][name] = run_dataset(
            os.path.join(args.beir_dir, name),
            name,
            args.model,
            cache_dir,
            query_encoding=args.query_encoding,
            identity=args.identity,
            query_limit=args.query_limit,
        )
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")


if __name__ == "__main__":
    main()
