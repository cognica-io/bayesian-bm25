#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Additional judgment-budget comparisons and experiment stage timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.paper_tables import (
    CEILING_ROWS,
    SKILL_COUNT_SLOPE,
    _ceiling_methods,
    _dataset_header,
    _datasets,
    calibration_table,
)
from benchmarks.reevaluation_report import PerQuery, load_reevaluation, ranking_table


def budget_table(results: dict) -> str:
    rows = (
        CEILING_ROWS[:1]
        + [CEILING_ROWS[4], CEILING_ROWS[5]]
        + [
            (
                "baseline/logistic_sure",
                "Logistic calibration/fusion, same sure judgments",
                "count + <=30",
            ),
            (
                "baseline/convex_sure",
                "Tuned convex + logistic, same sure judgments",
                "count + <=30",
            ),
            (
                "baseline/logistic_stratified",
                "Logistic calibration/fusion, same strata",
                "count + <=90",
            ),
            ("baseline/convex_stratified", "Tuned convex + logistic, same strata", "count + <=90"),
        ]
    )
    proxy = {}
    for dataset, result in results.items():
        channel = result["channels"]["fused"]
        methods = _ceiling_methods(channel)
        methods.update(
            {"baseline/" + key: value for key, value in channel["baselines"]["calibration"].items()}
        )
        proxy[dataset] = {"calibration": {"fused": methods}}
    return calibration_table(proxy, "fused", rows, "top10", SKILL_COUNT_SLOPE)


def budget_ranking_table(results: dict, per_query: PerQuery) -> str:
    values = {}
    for dataset in _datasets(results):
        values[dataset] = {
            key: {metric: np.asarray(v) for metric, v in entry["per_query"].items()}
            for key, entry in results[dataset]["channels"]["fused"]["baselines"]["ranking"].items()
        }
        values[dataset]["conventional"] = per_query[dataset]["Additive-Query-Tail"]
    return ranking_table(
        values,
        [
            ("conventional", "Fixed pooled evidence", "count (CV)"),
            ("logistic_sure", "Logistic fusion", "same <=30"),
            ("convex_sure", "Tuned convex combination", "same <=30"),
            ("logistic_stratified", "Logistic fusion", "same <=90"),
            ("convex_stratified", "Tuned convex combination", "same <=90"),
        ],
        "NDCG@10",
        leading=("Configuration", "Judgments"),
    )


def timing_table(results: dict, ceiling: dict, hybrid_diagnostics: dict) -> str:
    datasets = _datasets(results)
    lines = _dataset_header(["Measured stage (seconds)"], datasets)
    rows = [
        (
            "Reevaluation preparation, with recorded embedding cache",
            lambda d: results[d]["preparation_seconds"],
        ),
        ("All reevaluation variants", lambda d: results[d]["evaluation_seconds"]),
        (
            "Hybrid: all label-free variants",
            lambda d: hybrid_diagnostics[d]["timing"]["all_label_free_variants_seconds"],
        ),
        (
            "Hybrid: learned variants and CV",
            lambda d: hybrid_diagnostics[d]["timing"]["all_learned_variants_seconds"],
        ),
        (
            "Ceiling: full grid, budgets, CV, and quarters",
            lambda d: ceiling[d]["timing"]["experiments_seconds"],
        ),
    ]
    for label, getter in rows:
        cells = [f"{getter(d):.2f}" for d in datasets]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def cost_table(payload: dict) -> str:
    results = payload["results"]
    datasets = _datasets(results)
    lines = _dataset_header(["Pipeline (p50 / p95, ms)"], datasets)
    for key, label in (
        ("end_to_end_log_linear_ms", "Log-linear with count and ceiling"),
        ("end_to_end_density_ratio_ms", "Gaussian-background density ratio"),
        ("end_to_end_marginal_ms", "Marginal-background density ratio"),
        ("query_encoding_ms", "Query encoding alone"),
        ("fused_count_and_output_ms", "Fused count solve and output alone"),
    ):
        values = [results[d]["timings_ms"][key] for d in datasets]
        cells = [f"{v['p50']:.2f} / {v['p95']:.2f}" for v in values]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "_mpnet"):
        paper = json.loads((root / f"ceiling_slope{suffix}.json").read_text(encoding="ascii"))
        additional = json.loads(
            (root / "additional" / f"same_budget_{paper['model']}.json").read_text(encoding="ascii")
        )
        if (paper["model"], paper["identity"]) != (additional["model"], additional["identity"]):
            raise ValueError("additional and paper results require the same encoder and identity")
        combined = {
            name: dict(
                result,
                channels={
                    channel: dict(entry, baselines=additional["results"][name][channel])
                    for channel, entry in result["channels"].items()
                },
            )
            for name, result in paper["results"].items()
        }
        _, per_query = load_reevaluation(str(root / f"reevaluation{suffix}.json"))
        cost = json.loads(
            (root / "additional" / f"calibration_cost{suffix}.json").read_text(encoding="ascii")
        )
        tables = {
            f"same_budget_head{suffix}": budget_table(combined),
            f"same_budget_ranking{suffix}": budget_ranking_table(combined, per_query),
            f"query_cost{suffix}": cost_table(cost),
        }
        for name, table in tables.items():
            (out / f"{name}.md").write_text(table + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
