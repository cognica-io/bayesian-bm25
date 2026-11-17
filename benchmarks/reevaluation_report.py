#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tables for the re-evaluation, with confidence intervals and paired tests.

Reads the JSON written by ``benchmarks/reevaluation.py`` and, when
given, the JSON written by ``benchmarks/hybrid_beir.py``, and prints
Markdown tables:

* ranking quality per dataset with the macro-average;
* 95% bootstrap confidence intervals (queries are the resampling unit);
* paired comparisons: difference, bootstrap interval, randomization
  test p-value, and the Holm-adjusted p-value over the family of
  comparisons in the table;
* calibration of the lexical, fused and dense posteriors;
* threshold transfer and diagnostics.

Usage:
    python -m benchmarks.reevaluation_report \\
        --reevaluation benchmarks/results/reevaluation.json \\
        --hybrid benchmarks/results/hybrid_beir.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from benchmarks.significance import (
    bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    paired_randomization_test,
)

DATASET_ORDER = ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"]
DATASET_LABELS = {
    "arguana": "ArguAna",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "scidocs": "SciDocs",
    "scifact": "SciFact",
}

PerQuery = dict[str, dict[str, dict[str, np.ndarray]]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reevaluation(path: str) -> tuple[dict, PerQuery]:
    """Load re-evaluation results and their per-query ranking metrics."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    results = payload["results"]
    per_query: PerQuery = {}
    for dataset, result in results.items():
        per_query[dataset] = {
            method: {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in entry["per_query"].items()
            }
            for method, entry in result["ranking"].items()
        }
    return results, per_query


def load_hybrid(path: str) -> PerQuery:
    """Load the per-query ranking metrics of the hybrid BEIR benchmark."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {
        dataset: {
            method: {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in metrics.items()
            }
            for method, metrics in methods.items()
        }
        for dataset, methods in payload["per_query"].items()
    }


def load_hybrid_diagnostics(path: str) -> dict[str, dict]:
    """Load the per-dataset diagnostics of the hybrid BEIR benchmark."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["diagnostics"]


def _datasets(per_query: PerQuery) -> list[str]:
    return [d for d in DATASET_ORDER if d in per_query]


# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

def ranking_table(
    per_query: PerQuery,
    methods: list[tuple[str, ...]],
    metric: str,
    *,
    bold_among: list[str] | None = None,
    leading: tuple[str, ...] = ("Configuration",),
) -> str:
    """Markdown table of a ranking metric per dataset and its macro-average.

    ``methods`` lists ``(method key, cell, ...)`` tuples: the key selects
    the per-query values and the cells fill the columns named by
    ``leading``.  The best value of each column among ``bold_among``
    (default: all rows) is in bold.
    """
    datasets = _datasets(per_query)
    keys = [row[0] for row in methods]
    contenders = bold_among if bold_among is not None else keys

    values: dict[str, list[float | None]] = {}
    for key in keys:
        row: list[float | None] = []
        for dataset in datasets:
            entry = per_query[dataset].get(key)
            row.append(100.0 * float(np.mean(entry[metric])) if entry else None)
        present = [v for v in row if v is not None]
        row.append(
            float(np.mean([round(v, 2) for v in present]))
            if len(present) == len(datasets) else None
        )
        values[key] = row

    best = []
    for col in range(len(datasets) + 1):
        column = [values[k][col] for k in contenders if values[k][col] is not None]
        best.append(max(column) if column else None)

    header = list(leading) + [DATASET_LABELS[d] for d in datasets] + ["Average"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in methods:
        key = row[0]
        if len(row) - 1 != len(leading):
            raise ValueError(
                f"row {key!r} has {len(row) - 1} cells for {len(leading)} leading columns"
            )
        cells = list(row[1:])
        for col, value in enumerate(values[key]):
            if value is None:
                cells.append("--")
                continue
            text = f"{value:.2f}"
            if (
                key in contenders
                and best[col] is not None
                and round(value, 2) == round(best[col], 2)
            ):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confidence_table(
    per_query: PerQuery,
    methods: list[tuple[str, str]],
    metric: str,
    *,
    n_resamples: int = 10000,
) -> str:
    """Markdown table of means with 95% bootstrap confidence intervals."""
    datasets = _datasets(per_query)
    header = ["Configuration"] + [DATASET_LABELS[d] for d in datasets]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key, label in methods:
        cells = [label]
        for dataset in datasets:
            entry = per_query[dataset].get(key)
            if not entry:
                cells.append("--")
                continue
            mean, lower, upper = bootstrap_ci(entry[metric], n_resamples=n_resamples)
            cells.append(f"{100 * mean:.2f} [{100 * lower:.2f}, {100 * upper:.2f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def macro_difference(
    per_query: PerQuery,
    method_a: str,
    method_b: str,
    metric: str,
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Macro-averaged difference a - b with a stratified bootstrap interval.

    Each dataset's queries are resampled separately, the difference of
    means is taken within each dataset, and the differences are averaged
    over datasets.
    """
    datasets = _datasets(per_query)
    rng = np.random.default_rng(seed)
    total = np.zeros(n_resamples, dtype=np.float64)
    observed = 0.0
    for dataset in datasets:
        diff = per_query[dataset][method_a][metric] - per_query[dataset][method_b][metric]
        observed += float(np.mean(diff))
        n = len(diff)
        chunk = max(1, 2_000_000 // n)
        done = 0
        while done < n_resamples:
            size = min(chunk, n_resamples - done)
            idx = rng.integers(0, n, size=(size, n))
            total[done:done + size] += np.mean(diff[idx], axis=1)
            done += size
    total /= len(datasets)
    lower, upper = np.quantile(total, [0.025, 0.975])
    return observed / len(datasets), float(lower), float(upper)


def comparison_table(
    per_query: PerQuery,
    comparisons: list[tuple[str, str, str]],
    metric: str,
    *,
    n_resamples: int = 10000,
    n_permutations: int = 100000,
    significance_level: float | None = None,
) -> str:
    """Markdown table of paired comparisons.

    ``comparisons`` lists ``(method a, method b, row label)``.  For each
    dataset the cell gives the difference a - b in points, its 95%
    bootstrap interval, and the Holm-adjusted randomization-test
    p-value; the family for the adjustment is every cell of the table.

    With ``significance_level`` the p-value is not printed: a difference
    whose adjusted p-value is below the level is set in bold instead.
    """
    datasets = _datasets(per_query)
    cells: dict[tuple[int, str], tuple[float, float, float]] = {}
    raw_p: list[float] = []
    keys: list[tuple[int, str]] = []
    for row, (method_a, method_b, _) in enumerate(comparisons):
        for dataset in datasets:
            a = per_query[dataset][method_a][metric]
            b = per_query[dataset][method_b][metric]
            cells[(row, dataset)] = paired_bootstrap_ci(a, b, n_resamples=n_resamples)
            raw_p.append(paired_randomization_test(a, b, n_permutations=n_permutations))
            keys.append((row, dataset))
    adjusted = dict(zip(keys, holm_correction(raw_p), strict=True))

    header = ["Comparison"] + [DATASET_LABELS[d] for d in datasets] + ["Average"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row, (method_a, method_b, label) in enumerate(comparisons):
        line = [label]
        for dataset in datasets:
            diff, lower, upper = cells[(row, dataset)]
            p_value = adjusted[(row, dataset)]
            interval = f"[{100 * lower:+.2f}, {100 * upper:+.2f}]"
            if significance_level is None:
                line.append(f"{100 * diff:+.2f} {interval}, {_format_p(p_value)}")
            elif p_value < significance_level:
                line.append(f"**{100 * diff:+.2f}** {interval}")
            else:
                line.append(f"{100 * diff:+.2f} {interval}")
        diff, lower, upper = macro_difference(
            per_query, method_a, method_b, metric, n_resamples=n_resamples,
        )
        line.append(f"{100 * diff:+.2f} [{100 * lower:+.2f}, {100 * upper:+.2f}]")
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def _format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# Calibration tables
# ---------------------------------------------------------------------------

CALIBRATION_COLUMNS = [
    ("ece", "ECE", "{:.4f}"),
    ("log_loss_skill", "LL skill", "{:+.3f}"),
    ("brier_skill", "Brier skill", "{:+.3f}"),
    ("expected_over_observed", "Exp/Obs", "{:.3g}"),
    ("calibration_slope", "Slope", "{:.2f}"),
]


def calibration_table(
    results: dict,
    group: str,
    rows: list[tuple[str, str]],
    pool: str,
    *,
    columns: list[tuple[str, str, str]] | None = None,
) -> str:
    """Markdown table of calibration metrics, one block of columns per dataset."""
    columns = columns or CALIBRATION_COLUMNS
    datasets = [d for d in DATASET_ORDER if d in results]
    header = ["Method"]
    for dataset in datasets:
        header += [f"{DATASET_LABELS[dataset]} {label}" for _, label, _ in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key, label in rows:
        line = [label]
        for dataset in datasets:
            summary = results[dataset]["calibration"][group].get(key, {}).get(pool)
            for metric, _, fmt in columns:
                if summary is None or summary.get(metric) is None:
                    line.append("--")
                else:
                    value = summary[metric]
                    line.append(
                        "-inf" if value == float("-inf")
                        else "inf" if value == float("inf")
                        else "n/a" if value != value
                        else fmt.format(value)
                    )
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def simple_table(
    results: dict,
    section: str,
    rows: list[tuple[str, str]],
    columns: list[tuple[str, str, str]],
) -> str:
    """Markdown table of ``results[dataset][section][row][column]``."""
    datasets = [d for d in DATASET_ORDER if d in results]
    header = ["Method"]
    for dataset in datasets:
        header += [f"{DATASET_LABELS[dataset]} {label}" for _, label, _ in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key, label in rows:
        line = [label]
        for dataset in datasets:
            entry = results[dataset][section].get(key)
            for metric, _, fmt in columns:
                line.append("--" if entry is None else fmt.format(entry[metric]))
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def diagnostics_table(results: dict, rows: list[tuple[str, str, str]]) -> str:
    """Markdown table of per-dataset diagnostics (one row per quantity)."""
    datasets = [d for d in DATASET_ORDER if d in results]
    header = ["Quantity"] + [DATASET_LABELS[d] for d in datasets]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key, label, fmt in rows:
        line = [label]
        for dataset in datasets:
            value = results[dataset]["diagnostics"].get(key)
            line.append("--" if value is None else fmt.format(value))
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def fits_table(results: dict) -> str:
    """Markdown table of the fitted slopes, averaged over folds."""
    datasets = [d for d in DATASET_ORDER if d in results]
    header = ["Quantity"] + [DATASET_LABELS[d] for d in datasets]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    def row(label: str, getter) -> None:
        line = [label]
        for dataset in datasets:
            fits = results[dataset]["fits"]
            values = np.asarray([getter(fit) for fit in fits], dtype=np.float64)
            line.append(f"{np.mean(values):.2f} ({np.min(values):.2f}-{np.max(values):.2f})")
        lines.append("| " + " | ".join(line) + " |")

    row("Lexical slope alone", lambda f: f["kappa_lex"])
    row("Dense slope alone", lambda f: f["kappa_vec"])
    row("Lexical slope, fitted jointly", lambda f: f["kappas_fused"][0])
    row("Dense slope, fitted jointly", lambda f: f["kappas_fused"][1])
    row("Lexical slope, predicted by Theorem 5.8", lambda f: f["kappas_fused_predicted"][0])
    row("Dense slope, predicted by Theorem 5.8", lambda f: f["kappas_fused_predicted"][1])
    row("Mean relevant documents per query", lambda f: f["relevant_mean"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report layout
# ---------------------------------------------------------------------------

MAIN_ROWS = [
    ("BM25", "BM25"),
    ("Dense", "Dense"),
    ("RRF", "RRF (k = 60)"),
    ("Convex", "Convex (min-max, w = 0.5)"),
    ("Additive-Corpus", "Log-linear dense: corpus-level calibration"),
    ("Additive-Query", "Log-linear dense: per-query calibration"),
    ("Additive-Query-Fit", "Log-linear dense: per-query, slopes fitted (CV)"),
    ("Additive-Query-FitPool", "Log-linear dense: per-query, slopes fitted on the pool (CV)"),
    ("Additive-Corpus-DensityRatio", "Density-ratio dense: corpus-level lexical slope"),
    ("Additive-Query-DensityRatio", "Density-ratio dense: per-query lexical calibration"),
    ("Additive-QueryFit-DensityRatio", "Density-ratio dense: per-query, lexical slope fitted (CV)"),
    ("Local", "Log-odds, per-query standardized over the pool"),
    ("Balanced", "Log-odds, per-query range-normalized"),
]

SUPERVISED_ROWS = [
    "Additive-Query-Fit", "Additive-Query-FitPool", "Additive-QueryFit-DensityRatio",
]

SLOPE_ROWS = [
    ("Additive-Corpus", "Lexical corpus-level, dense corpus-level"),
    ("Slope-LexQuery-DenseCorpus", "Lexical per-query, dense corpus-level"),
    ("Slope-LexCorpus-DenseQuery", "Lexical corpus-level, dense per-query"),
    ("Additive-Query", "Lexical per-query, dense per-query"),
]

ABLATION_ROWS = [
    ("Local", "Standardized over the pool; non-match: dense only"),
    ("Local+Consistent", "+ non-match carries the evidence of a zero score"),
    ("Additive-Query", "+ statistics over the population (additive posterior)"),
    ("Balanced", "Range-normalized; document prior; non-match: floor"),
    ("Balanced-NoPrior", "- document prior"),
    ("Balanced-NoPrior-Consistent", "- floor (non-match at a zero score)"),
    ("Balanced-NoPrior-Consistent-Cos", "- arctanh of the similarity (plain cosine)"),
    ("Balanced-NoPrior-Consistent-Cos-Std", "- range (standardized over the pool)"),
    ("Additive-Query-MAD", "Additive posterior, robust dispersion (MAD)"),
    ("Additive-Query-Atanh", "Additive posterior, arctanh of the similarity"),
    ("Additive-Query-Floor", "Additive posterior, non-match at the floor"),
    ("Additive-Query-DenseOnly", "Additive posterior, non-match: dense only"),
    ("Additive-Query+Prior", "Additive posterior + document prior"),
    ("Additive-Query-w0.3", "Additive posterior, dense weight 0.3"),
    ("Additive-Query-w0.4", "Additive posterior, dense weight 0.4"),
    ("Additive-Query-w0.6", "Additive posterior, dense weight 0.6"),
    ("Additive-Query-w0.7", "Additive posterior, dense weight 0.7"),
]

VECTOR_RANK_ROWS = [
    ("Dense", "Dense alone"),
    ("Additive-Query", "Log-linear similarity (Section 4.8)"),
    ("Additive-Query-DensityRatio", "Density ratio, approximate posterior"),
    ("Additive-Query-DensityRatio-Monotone", "Density ratio, approximate, monotone"),
    ("Additive-Query-DensityRatio-Bandwidth2", "Density ratio, approximate, bandwidth x 2"),
    ("Additive-Query-DensityRatio-Uniform", "Density ratio, approximate, uniform weights"),
    ("Additive-Query-DensityRatio-Exact", "Density ratio, exact posterior"),
    ("Additive-Query-DensityRatio-Exact-Monotone", "Density ratio, exact, monotone"),
]

COMPARISONS = [
    ("Additive-Query", "RRF", "Log-linear per-query vs RRF"),
    ("Additive-Query", "Convex", "Log-linear per-query vs Convex"),
    ("Additive-Query", "Additive-Corpus", "Log-linear: per-query vs corpus-level"),
    ("Additive-Query-Fit", "Additive-Query", "Log-linear: fitted slopes vs conventions"),
    ("Additive-Query-DensityRatio", "RRF", "Density-ratio per-query vs RRF"),
    ("Additive-Query-DensityRatio", "Convex", "Density-ratio per-query vs Convex"),
    ("Additive-Query-DensityRatio", "Additive-Corpus-DensityRatio",
     "Density-ratio: per-query vs corpus-level lexical"),
    ("Additive-Query-DensityRatio", "Additive-Query", "Density-ratio vs log-linear dense"),
    ("Additive-Query-DensityRatio", "Balanced", "Density-ratio per-query vs range-normalized"),
    ("Balanced", "Convex", "Range-normalized vs Convex"),
    ("Balanced", "RRF", "Range-normalized vs RRF"),
    ("Balanced", "Additive-Query", "Range-normalized vs log-linear per-query"),
    ("Convex", "RRF", "Convex vs RRF"),
]

LEXICAL_ROWS = [
    ("Lex-Corpus-NoBaseRate", "Corpus-level, no base rate"),
    ("Lex-Corpus-Percentile", "Corpus-level, percentile base rate"),
    ("Lex-Corpus-0.001", "Corpus-level, base rate 0.001"),
    ("Lex-Query-R0.5", "Per-query, R = 0.5"),
    ("Lex-Query-R1", "Per-query, R = 1"),
    ("Lex-Query-R2", "Per-query, R = 2"),
    ("Lex-Query-R5", "Per-query, R = 5"),
    ("Lex-Query-R10", "Per-query, R = 10"),
    ("Lex-Query-R20", "Per-query, R = 20"),
    ("Lex-Query-R50", "Per-query, R = 50"),
    ("Lex-Query-Rmean", "Per-query, R = training mean (CV)"),
    ("Lex-Query-Roracle", "Per-query, R = true count"),
    ("Lex-Query-Fit", "Per-query, slope fitted, R = training mean (CV)"),
    ("Lex-Query-Fit-Roracle", "Per-query, slope fitted, R = true count"),
    ("Lex-Platt-Raw", "Platt scaling of the raw score (CV)"),
    ("Lex-Platt-Standardized", "Platt scaling after per-query standardization (CV)"),
    ("Lex-MinMax", "Global min-max rescaling (CV)"),
]

FUSED_ROWS = [
    ("Fused-Corpus", "Corpus-level additive posterior"),
    ("Fused-Query-Additive-rho1", "Per-query additive (5.2), rho = 1"),
    ("Fused-Query-Additive-rho0.5", "Per-query pooled, rho = 0.5"),
    ("Fused-Query-Additive-rhoMeasured", "Per-query pooled, rho from the measured correlation"),
    ("Fused-Query-Additive-rho0", "Per-query pooled, rho = 0"),
    ("Fused-Query-Consistent-rho1", "+ post-fusion consistency, rho = 1"),
    ("Fused-Query-Consistent-rho0.5", "+ post-fusion consistency, rho = 0.5"),
    ("Fused-Query-Consistent-rhoMeasured", "+ post-fusion consistency, measured rho"),
    ("Fused-Query-Consistent-rho0", "+ post-fusion consistency, rho = 0"),
    ("Fused-Query-Consistent-Roracle", "+ post-fusion consistency, rho = 1, R = true count"),
    ("Fused-Fit-Rmean", "Slopes fitted, R = training mean (CV)"),
    ("Fused-Fit-Roracle", "Slopes fitted, R = true count"),
    ("Fused-Query-DensityRatio", "Per-query lexical + density ratio (5.2)"),
]

VECTOR_ROWS = [
    ("Vec-GlobalSigmoid", "Global sigmoid on distance"),
    ("Vec-Arctan", "Arctangent normalization"),
    ("Vec-Platt-Raw", "Platt scaling of the similarity (CV)"),
    ("Vec-Platt-Standardized", "Platt scaling after per-query standardization (CV)"),
    ("Vec-Query-R1", "Per-query log-linear, R = 1"),
    ("Vec-Query-Roracle", "Per-query log-linear, R = true count"),
    ("Vec-DensityRatio", "Density ratio, approximate posterior"),
    ("Vec-DensityRatio-Monotone", "Density ratio, approximate, monotone"),
    ("Vec-DensityRatio-Bandwidth2", "Density ratio, approximate, bandwidth x 2"),
    ("Vec-DensityRatio-Uniform", "Density ratio, approximate, uniform weights"),
    ("Vec-DensityRatio-Exact", "Density ratio, exact posterior"),
    ("Vec-DensityRatio-Exact-Monotone", "Density ratio, exact, monotone"),
]

THRESHOLD_ROWS = [
    ("Raw BM25", "Raw BM25 score"),
    ("Corpus-level log-odds", "Corpus-level log-odds"),
    ("Corpus-level log-odds + prior", "Corpus-level log-odds + document prior"),
    ("Per-query z-score", "Per-query standardized score"),
    ("Per-query min-max", "Per-query min-max"),
    ("Per-query log-odds (R=1)", "Per-query posterior, R = 1"),
    ("Per-query log-odds (R oracle)", "Per-query posterior, R = true count"),
]

THRESHOLD_COLUMNS = [
    ("test_micro_f1", "micro F1", "{:.4f}"),
    ("test_macro_f1", "macro F1", "{:.4f}"),
]

DIAGNOSTIC_ROWS = [
    ("query_tokens_mean", "Query length (tokens, mean)", "{:.1f}"),
    ("matched_fraction_mean", "Fraction of corpus matched", "{:.3f}"),
    ("relevant_mean", "Relevant documents per query", "{:.2f}"),
    ("sigma_lex_cv", "Coefficient of variation of sigma_q", "{:.2f}"),
    ("sigma_lex_ratio_p95_p5", "sigma_q, 95th / 5th percentile", "{:.1f}"),
    ("sigma_vec_mean", "Dispersion of the similarities over the corpus (mean)", "{:.3f}"),
    ("sigma_vec_cv", "Coefficient of variation of that dispersion", "{:.2f}"),
    ("relevant_unmatched_fraction", "Relevant documents that match no query term", "{:.3f}"),
    ("queries_with_unmatched_relevant", "Queries with such a relevant document", "{:.3f}"),
    ("pool_dense_weight_ratio_gmean",
     "Dense weight relative to lexical: pool over population statistics", "{:.2f}"),
    ("floor_lexical_weight_ratio_mean",
     "Lexical differences among matched candidates under the floor", "{:.2f}"),
    ("max_evidence_corpus_mean", "Corpus-level max evidence (mean)", "{:.1f}"),
    ("max_evidence_corpus_p95", "Corpus-level max evidence (95th pct)", "{:.1f}"),
    ("queries_with_likelihood_one", "Queries with a likelihood rounded to 1", "{:.3f}"),
    ("tied_top100_mean", "Tied among top-100 probabilities (mean)", "{:.1f}"),
    ("tied_top100_log_odds_mean", "Tied among top-100 log-odds (mean)", "{:.2f}"),
    ("queries_with_rounding_ties_top10", "Queries with rounding ties in top 10", "{:.3f}"),
    ("queries_with_rounding_ties_top100", "Queries with rounding ties in top 100", "{:.3f}"),
    ("corr_all", "Signal correlation, population", "{:.3f}"),
    ("corr_matched", "Signal correlation, matched documents", "{:.3f}"),
    ("corr_pool", "Signal correlation, candidate pool", "{:.3f}"),
    ("rho_from_corr_all", "Confidence exponent implied (Theorem 5.8)", "{:.3f}"),
    ("additive_count_ratio_median", "Additive posterior: expected count / R (median)", "{:.2f}"),
    ("offset_dilute_error_mean", "Dilute offset minus exact (mean)", "{:+.3f}"),
    ("offset_dilute_abs_error_p95", "Dilute offset error (95th pct)", "{:.3f}"),
    ("offset_topk_abs_error_mean", "Top-1000 only: offset error (mean)", "{:.4f}"),
    ("offset_topk_bound_abs_error_mean", "Top-1000 + bounded tail: error (mean)", "{:.4f}"),
    ("offset_sampled_abs_error_mean", "Top-1000 + sampled tail: error (mean)", "{:.4f}"),
    ("offset_factorized_error_mean", "Per-term statistics: offset minus exact (mean)", "{:+.3f}"),
    ("offset_factorized_abs_error_mean", "Per-term statistics: offset error (mean)", "{:.3f}"),
    ("psi_gap_mean", "Factorized minus exact log-partition (mean)", "{:+.3f}"),
    ("psi_gap_negative_fraction", "Queries with the factorized value a lower bound", "{:.3f}"),
    ("sigma_factorized_ratio_mean", "Factorized dispersion / exact (mean)", "{:.3f}"),
    ("sigma_factorized_below_fraction", "Queries with the factorized dispersion too small",
     "{:.3f}"),
    ("top_error_factorized_full_mean", "Index statistics only: top document log-odds error",
     "{:+.3f}"),
    ("density_ratio_clipped_mean", "Exact density ratio: fraction clipped", "{:.4f}"),
]


def build_report(
    results: dict,
    per_query: PerQuery,
    hybrid: PerQuery | None,
    *,
    n_resamples: int,
    n_permutations: int,
) -> str:
    """Assemble the full Markdown report."""
    sections: list[str] = []

    def section(title: str, body: str) -> None:
        sections.append(f"## {title}\n\n{body}")

    label_free = [key for key, _ in MAIN_ROWS if key not in SUPERVISED_ROWS]
    for metric in ("NDCG@10", "MAP@10", "Recall@10"):
        section(
            f"Ranking: {metric}",
            ranking_table(per_query, MAIN_ROWS, metric, bold_among=label_free),
        )
    section(
        "Ranking: NDCG@10 with 95% bootstrap intervals",
        confidence_table(per_query, MAIN_ROWS, "NDCG@10", n_resamples=n_resamples),
    )
    section(
        "Ranking: paired comparisons (NDCG@10)",
        comparison_table(
            per_query, COMPARISONS, "NDCG@10",
            n_resamples=n_resamples, n_permutations=n_permutations,
        ),
    )
    section("Slope isolation (NDCG@10)", ranking_table(per_query, SLOPE_ROWS, "NDCG@10"))
    section("Ablation (NDCG@10)", ranking_table(per_query, ABLATION_ROWS, "NDCG@10"))
    section(
        "Vector evidence in the additive posterior (NDCG@10)",
        ranking_table(per_query, VECTOR_RANK_ROWS, "NDCG@10"),
    )
    section("Fitted slopes", fits_table(results))

    for pool in ("all", "matched", "top10"):
        section(
            f"Lexical calibration, pool = {pool}",
            calibration_table(results, "lexical", LEXICAL_ROWS, pool),
        )
    for pool in ("all", "pool", "top10"):
        section(
            f"Fused calibration, pool = {pool}",
            calibration_table(results, "fused", FUSED_ROWS, pool),
        )
    for pool in ("retrieved", "top10"):
        section(
            f"Dense calibration, pool = {pool}",
            calibration_table(results, "vector", VECTOR_ROWS, pool),
        )
    section(
        "Threshold transfer",
        simple_table(results, "threshold", THRESHOLD_ROWS, THRESHOLD_COLUMNS),
    )
    section("Diagnostics", diagnostics_table(results, DIAGNOSTIC_ROWS))

    if hybrid is not None:
        methods = sorted({m for d in hybrid.values() for m in d})
        order = [m for m in _hybrid_order() if m in methods]
        rows = [(m, m) for m in order]
        for metric in ("NDCG@10", "MAP@10", "Recall@10"):
            section(
                f"First-version configurations, corrected protocol: {metric}",
                ranking_table(hybrid, rows, metric),
            )
        section(
            "First-version configurations: NDCG@10 with 95% bootstrap intervals",
            confidence_table(hybrid, rows, "NDCG@10", n_resamples=n_resamples),
        )
        section(
            "First-version configurations: paired comparisons (NDCG@10)",
            comparison_table(
                hybrid,
                [
                    ("Bayesian-Balanced", "Convex", "Range-normalized vs Convex"),
                    ("Bayesian-Balanced", "RRF", "Range-normalized vs RRF"),
                    ("Bayesian-LogOdds-Local", "Convex", "Standardized vs Convex"),
                    ("Bayesian-LogOdds-Local", "RRF", "Standardized vs RRF"),
                    ("Bayesian-LogOdds", "RRF", "Corpus-level vs RRF"),
                    ("Bayesian-Attn-Norm", "Bayesian-Balanced",
                     "Learned weights (CV) vs uniform"),
                    ("Bayesian-Vector-Balanced", "Dense", "Density ratio vs Dense"),
                    ("Bayesian-Vector-Balanced", "Bayesian-Balanced",
                     "Density ratio vs range-normalized"),
                ],
                "NDCG@10",
                n_resamples=n_resamples, n_permutations=n_permutations,
            ),
        )

    return "\n\n".join(sections) + "\n"


def _hybrid_order() -> list[str]:
    return [
        "BM25", "Dense", "RRF", "Convex",
        "Bayesian-LogOdds-Local", "Bayesian-Balanced",
        "Bayesian-Balanced-Mix", "Bayesian-Balanced-Elbow",
        "Bayesian-LogOdds", "Bayesian-LogOdds-BR",
        "Bayesian-Vector-Balanced", "Bayesian-Vector-Softplus",
        "Bayesian-Gated-ReLU", "Bayesian-Gated-Swish", "Bayesian-Gated-GELU",
        "Bayesian-Gated-Swish-B2", "Bayesian-Gated-Softplus",
        "Bayesian-OR",
        "Bayesian-Attention", "Bayesian-Attn-Norm",
        "Bayesian-MultiHead", "Bayesian-MultiHead-Norm", "Bayesian-Vector-Attn",
        "Bayesian-MultiField", "Bayesian-MultiField-Bal",
        "Dense-Kappa", "Dense-Arctan", "Dense-Platt",
        "VPT-DensityPrior", "VPT-BM25Weights",
        "VPT-BW-0.2", "VPT-BW-0.5", "VPT-BW-1.0", "VPT-BW-2.0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tables for the re-evaluation")
    parser.add_argument(
        "--reevaluation", required=True,
        help="JSON written by benchmarks/reevaluation.py",
    )
    parser.add_argument(
        "--hybrid", default=None,
        help="JSON written by benchmarks/hybrid_beir.py (optional)",
    )
    parser.add_argument("--resamples", type=int, default=10000, help="Bootstrap resamples")
    parser.add_argument(
        "--permutations", type=int, default=100000,
        help="Sign-flip samples of the randomization test",
    )
    parser.add_argument("-o", "--output", default=None, help="Write the report to a file")
    args = parser.parse_args()

    results, per_query = load_reevaluation(args.reevaluation)
    hybrid = load_hybrid(args.hybrid) if args.hybrid else None
    report = build_report(
        results, per_query, hybrid,
        n_resamples=args.resamples, n_permutations=args.permutations,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
