#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for the table generators of the re-evaluation."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from benchmarks.paper_tables import (  # noqa: E402
    BANDWIDTH_ROWS,
    BASE_RATE_ROWS,
    MAIN_ROWS,
    PROTOCOL_CELLS,
    PROTOCOL_ROWS,
    WEIGHT_ROWS,
    build_scores_to_evidence_tables,
    build_tables,
    calibration_table,
    format_count_ratio,
)
from benchmarks.reevaluation import run_dataset  # noqa: E402
from benchmarks.reevaluation_report import (  # noqa: E402
    comparison_table,
    load_reevaluation,
    ranking_table,
)
from tests.test_reevaluation_helpers import _prepared  # noqa: E402


def _per_query(values: dict[str, list[float]]) -> dict:
    """Per-query metrics of one dataset from NDCG@10 values per method."""
    return {
        "scifact": {
            method: {"NDCG@10": np.asarray(v, dtype=np.float64)} for method, v in values.items()
        }
    }


class TestFormatCountRatio:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (8222.9, "8223"),
            (999.6, "1000"),
            (90.8, "90.8"),
            (153.0, "153"),
            (5.61, "5.61"),
            (1.0, "1"),
            (0.0131, "0.0131"),
        ],
    )
    def test_three_significant_digits_without_exponent(self, value, expected):
        assert format_count_ratio(value) == expected


class TestRankingTable:
    def test_leading_columns_and_bold(self):
        per_query = _per_query({"A": [0.5, 0.7], "B": [0.9, 0.9], "C": [1.0, 1.0]})
        table = ranking_table(
            per_query,
            [("A", "first", "none"), ("B", "second", "none"), ("C", "third", "CV")],
            "NDCG@10",
            bold_among=["A", "B"],
            leading=("Method", "Labels"),
        )
        lines = table.splitlines()
        assert lines[0] == "| Method | Labels | SciFact | Average |"
        assert lines[2] == "| first | none | 60.00 | 60.00 |"
        # The best value among the contenders is bold; C is not a contender.
        assert lines[3] == "| second | none | **90.00** | **90.00** |"
        assert lines[4] == "| third | CV | 100.00 | 100.00 |"

    def test_rejects_a_row_with_the_wrong_number_of_cells(self):
        per_query = _per_query({"A": [0.5, 0.7]})
        with pytest.raises(ValueError, match="leading columns"):
            ranking_table(per_query, [("A", "first", "extra")], "NDCG@10")


class TestComparisonTable:
    @pytest.fixture(scope="class")
    def per_query(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.2, 0.8, size=200)
        return _per_query(
            {
                "base": base,
                "better": base + 0.05 + rng.normal(0.0, 0.01, size=200),
                "same": base + rng.normal(0.0, 0.05, size=200),
            }
        )

    def test_significant_differences_are_bold(self, per_query):
        table = comparison_table(
            per_query,
            [("better", "base", "better vs base"), ("same", "base", "same vs base")],
            "NDCG@10",
            n_resamples=500,
            n_permutations=2000,
            significance_level=0.05,
        )
        better, same = table.splitlines()[2:4]
        # The first cell is the difference in points, about +5, set in bold.
        assert better.startswith("| better vs base | **+")
        assert 4.5 < float(better.split("**")[1]) < 5.5
        assert "**" not in same
        assert "p =" not in table and "p <" not in table

    def test_p_values_are_printed_without_a_level(self, per_query):
        table = comparison_table(
            per_query,
            [("better", "base", "better vs base")],
            "NDCG@10",
            n_resamples=500,
            n_permutations=2000,
        )
        assert "p < 0.001" in table


class TestCalibrationTable:
    def test_cells_join_the_metrics_and_mark_infinities(self):
        results = {
            "scifact": {
                "calibration": {
                    "lexical": {
                        "row": {
                            "all": {
                                "log_loss_skill": float("-inf"),
                                "expected_over_observed": 1234.5,
                            }
                        }
                    }
                }
            }
        }
        table = calibration_table(
            results,
            "lexical",
            [("row", "Row", "none"), ("missing", "Missing", "none")],
            "all",
            [("log_loss_skill", "{:+.2f}"), ("expected_over_observed", format_count_ratio)],
        )
        lines = table.splitlines()
        assert lines[2] == "| Row | none | $-\\infty$ / 1234 |"
        assert lines[3] == "| Missing | none | -- |"


class TestBuildTables:
    @pytest.fixture(scope="class")
    def tables(self, tmp_path_factory):
        prepared = _prepared(n_docs=400, n_queries=12, seed=3)
        result = run_dataset(prepared, k=10, retrieve_k=50, n_folds=3)
        # Through JSON, as the benchmark script writes it and the report reads it.
        path = tmp_path_factory.mktemp("results") / "reevaluation.json"
        path.write_text(json.dumps({"results": {"scifact": result}}), encoding="utf-8")
        results, per_query = load_reevaluation(str(path))

        # The hybrid benchmark supplies the rows the re-evaluation does not.
        stand_in = per_query["scifact"]["Balanced"]
        hybrid_keys = (
            {row[0] for row in MAIN_ROWS}
            | {row[0] for row in WEIGHT_ROWS}
            | {row[0] for row in BANDWIDTH_ROWS}
            | {"Bayesian-Balanced-Mix", "Bayesian-Balanced-Elbow"}
        )
        hybrid = {"scifact": {key: stand_in for key in hybrid_keys}}
        protocol = {
            "scifact": {
                cell: {
                    "ndcg": {key: 0.5 for key, _ in PROTOCOL_ROWS},
                    "tied_top100": {key: 1.0 for key, _ in PROTOCOL_ROWS[4:]},
                }
                for cell in PROTOCOL_CELLS
            }
        }
        protocol["arguana"] = protocol["scifact"]
        base_rate = {
            "results": {
                dataset: {key: {"ECE": 0.01, "Brier": 0.02} for key, _, _ in BASE_RATE_ROWS}
                for dataset in ("nfcorpus", "scifact")
            }
        }
        ivf_diagnostics = {
            "scifact": {
                "ivf_background": {
                    "residual_mean": 0.38,
                    "residual_std": 0.09,
                    "query_distance_mean": 0.90,
                    "query_distance_std": 0.10,
                }
            }
        }
        return build_tables(
            results,
            per_query,
            hybrid,
            hybrid,
            hybrid,
            ivf_diagnostics,
            protocol,
            base_rate,
            results,
            per_query,
            n_resamples=200,
            n_permutations=500,
        )

    def test_every_table_of_the_paper_is_built(self, tables):
        expected = {f"table{n:02d}" for n in range(1, 20)}
        assert expected <= set(tables)

    def test_tables_are_well_formed(self, tables):
        for name, table in tables.items():
            lines = table.splitlines()
            assert len(lines) >= 3, name
            columns = lines[0].count("|")
            for line in lines:
                assert line.startswith("| ") and line.endswith(" |"), name
            # Cells never contain a pipe, so every row has the header's columns.
            assert {line.count("|") for line in lines} == {columns}, name

    def test_no_row_of_the_harness_is_missing(self, tables):
        # A placeholder cell means a row key that the result files do not hold.
        for name in (
            "table01",
            "table03",
            "table04",
            "table12",
            "table13",
            "table15",
            "table16",
            "table18",
            "table19",
            "supplement_unified_dense_list",
            "supplement_unified_exact_pool",
            "supplement_pooled_corpus",
            "supplement_unified_variants_pool",
            "supplement_unified_variants_dense_list",
            "supplement_ivf_background_statistics",
        ):
            assert "--" not in tables[name].replace("| --- ", ""), name


class TestEvidencePaperTables:
    def test_23_tables_use_new_populations_and_keep_ceiling_rows_separate(self, monkeypatch):
        from benchmarks import ceiling_slope

        monkeypatch.setattr(
            ceiling_slope,
            "GRID",
            np.array(
                [
                    [0.5, 0.5],
                    [0.5, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ]
            ),
        )
        prepared = _prepared(n_docs=200, n_queries=8, seed=15)
        prepared.identity = "text"
        result = run_dataset(prepared, retrieve_k=30, n_folds=2)
        ceiling = ceiling_slope.run_dataset(
            prepared,
            retrieve_k=30,
            n_folds=2,
            n_draws=2,
        )
        values = {
            key: {metric: np.asarray(v) for metric, v in entry["per_query"].items()}
            for key, entry in result["ranking"].items()
        }
        hybrid_keys = {r[0] for r in MAIN_ROWS + WEIGHT_ROWS + BANDWIDTH_ROWS}
        hybrid = {"scifact": {key: values["Balanced"] for key in hybrid_keys}}
        results, per_query, ceilings = (
            {"scifact": result},
            {"scifact": values},
            {"scifact": ceiling},
        )
        protocol = {
            "arguana": {
                cell: {"ndcg": {key: 0.5 for key, _ in PROTOCOL_ROWS}, "tied_top100": {}}
                for cell in ("id/probability", "id/log-odds", "text/probability", "text/log-odds")
            }
        }
        protocol["scifact"] = protocol["arguana"]
        diagnostics = {
            "scifact": {
                "ivf_background": {
                    "residual_mean": 0.4,
                    "residual_std": 0.1,
                    "query_distance_mean": 0.9,
                    "query_distance_std": 0.1,
                },
                "timing": {
                    "all_label_free_variants_seconds": 1.0,
                    "all_learned_variants_seconds": 2.0,
                },
            }
        }
        tables = build_scores_to_evidence_tables(
            results,
            per_query,
            hybrid,
            hybrid,
            hybrid,
            diagnostics,
            protocol,
            results,
            per_query,
            ceilings,
            ceilings,
            hybrid_diagnostics=diagnostics,
            n_resamples=100,
            n_permutations=100,
        )
        assert {f"table{n:02d}" for n in range(1, 24)} <= set(tables)
        assert "Ceiling from 30" in tables["table19"]
        assert "Documents kept" in tables["table22"]
        assert "Text identity" in tables["table05"]
        assert "Log-linear model, slope $1/m_q$" in tables["table15"]
        assert "Same" not in tables["table19"]
        for name, table in tables.items():
            assert len({line.count("|") for line in table.splitlines()}) == 1, name
        result["identity"] = "id"
        with pytest.raises(ValueError, match="text identity"):
            build_scores_to_evidence_tables(
                results,
                per_query,
                hybrid,
                hybrid,
                hybrid,
                diagnostics,
                protocol,
                results,
                per_query,
                ceilings,
                ceilings,
                hybrid_diagnostics=diagnostics,
                n_resamples=100,
                n_permutations=100,
            )
