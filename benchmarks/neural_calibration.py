#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Neural score calibration (Paper 1, Section 12.2 #5).

Evaluates PlattCalibrator and IsotonicCalibrator:

  1. Calibration accuracy: how well each method recovers true probabilities
  2. Monotonicity preservation
  3. Integration with log_odds_conjunction for hybrid fusion
  4. Timing: fit() and calibrate() performance
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from bayesian_bm25.calibration import IsotonicCalibrator, PlattCalibrator
from bayesian_bm25.fusion import log_odds_conjunction
from bayesian_bm25.probability import sigmoid


def generate_calibration_data(
    n_samples: int,
    true_a: float,
    true_b: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic neural scores with known calibration function.

    Returns (scores, labels, true_probs).
    """
    scores = rng.uniform(-3, 5, size=n_samples)
    true_probs = np.asarray(sigmoid(true_a * scores + true_b), dtype=np.float64)
    labels = (rng.random(n_samples) < true_probs).astype(np.float64)
    return scores, labels, true_probs


def run_calibration_accuracy(rng: np.random.Generator) -> dict:
    """Compare Platt and isotonic calibration accuracy."""
    print("=" * 72)
    print("Calibration Accuracy: Platt vs Isotonic")
    print("=" * 72)

    configs = [
        (2.0, -1.0, "Linear: a=2, b=-1"),
        (1.0, 0.0, "Simple: a=1, b=0"),
        (3.0, -3.0, "Steep: a=3, b=-3"),
    ]

    n_train = 2000
    n_test = 500

    results = []
    col_s = 25
    print(f"\n  {'Scenario':<{col_s}}  {'Method':<12}  {'MSE':>7}  {'Max Err':>8}  {'Monotone':>8}")
    print(f"  {'-' * col_s}  {'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 8}")

    for true_a, true_b, desc in configs:
        train_scores, train_labels, _ = generate_calibration_data(
            n_train, true_a, true_b, rng
        )
        test_scores, _, test_true_probs = generate_calibration_data(
            n_test, true_a, true_b, rng
        )

        # Sort test scores for monotonicity check
        order = np.argsort(test_scores)
        test_scores_sorted = test_scores[order]
        test_true_sorted = test_true_probs[order]

        scenario_results = {"scenario": desc}

        # Platt
        platt = PlattCalibrator()
        platt.fit(train_scores, train_labels, learning_rate=0.05, max_iterations=3000)
        platt_cal = np.asarray(platt.calibrate(test_scores_sorted), dtype=np.float64)
        platt_mse = float(np.mean((platt_cal - test_true_sorted) ** 2))
        platt_max = float(np.max(np.abs(platt_cal - test_true_sorted)))
        platt_mono = bool(np.all(np.diff(platt_cal) >= -1e-10))

        print(
            f"  {desc:<{col_s}}  {'Platt':<12}  {platt_mse:>7.5f}  "
            f"{platt_max:>8.5f}  {'Yes' if platt_mono else 'No':>8}"
        )
        scenario_results["platt"] = {
            "mse": platt_mse, "max_err": platt_max,
            "monotone": platt_mono, "a": platt.a, "b": platt.b,
        }

        # Isotonic
        iso = IsotonicCalibrator()
        iso.fit(train_scores, train_labels)
        iso_cal = np.asarray(iso.calibrate(test_scores_sorted), dtype=np.float64)
        iso_mse = float(np.mean((iso_cal - test_true_sorted) ** 2))
        iso_max = float(np.max(np.abs(iso_cal - test_true_sorted)))
        iso_mono = bool(np.all(np.diff(iso_cal) >= -1e-10))

        print(
            f"  {'':<{col_s}}  {'Isotonic':<12}  {iso_mse:>7.5f}  "
            f"{iso_max:>8.5f}  {'Yes' if iso_mono else 'No':>8}"
        )
        scenario_results["isotonic"] = {
            "mse": iso_mse, "max_err": iso_max, "monotone": iso_mono,
        }

        # Uncalibrated (raw sigmoid)
        raw_cal = np.asarray(sigmoid(test_scores_sorted), dtype=np.float64)
        raw_mse = float(np.mean((raw_cal - test_true_sorted) ** 2))
        raw_max = float(np.max(np.abs(raw_cal - test_true_sorted)))

        print(
            f"  {'':<{col_s}}  {'Raw sigmoid':<12}  {raw_mse:>7.5f}  "
            f"{raw_max:>8.5f}  {'Yes':>8}"
        )
        scenario_results["raw"] = {"mse": raw_mse, "max_err": raw_max}

        results.append(scenario_results)
        print()

    return {"scenarios": results}


def run_fusion_benchmark(rng: np.random.Generator) -> dict:
    """Evaluate calibrated neural scores in hybrid fusion."""
    print("=" * 72)
    print("Hybrid Fusion: Calibrated Neural + BM25 Probabilities")
    print("=" * 72)

    n_train = 2000
    n_test = 500

    # Generate neural scores
    true_a, true_b = 2.0, -1.0
    train_scores, train_labels, _ = generate_calibration_data(
        n_train, true_a, true_b, rng
    )
    test_scores, test_labels, _ = generate_calibration_data(
        n_test, true_a, true_b, rng
    )

    # Simulate BM25 probabilities (correlated with labels but noisy)
    true_logits = np.where(test_labels == 1, 1.5, -1.5)
    bm25_probs = np.asarray(
        sigmoid(true_logits + rng.normal(0, 0.8, size=n_test)),
        dtype=np.float64,
    )

    # Calibrate neural scores
    platt = PlattCalibrator()
    platt.fit(train_scores, train_labels, learning_rate=0.05, max_iterations=3000)
    neural_platt = np.asarray(platt.calibrate(test_scores), dtype=np.float64)

    iso = IsotonicCalibrator()
    iso.fit(train_scores, train_labels)
    neural_iso = np.asarray(iso.calibrate(test_scores), dtype=np.float64)

    # Raw sigmoid as baseline
    neural_raw = np.asarray(sigmoid(test_scores), dtype=np.float64)

    # Fusion methods
    methods = [
        ("BM25 only", bm25_probs),
        ("Neural only (raw)", neural_raw),
        ("Neural only (Platt)", neural_platt),
        ("Neural only (Iso)", neural_iso),
    ]

    # Fused methods
    for name, neural in [("Platt", neural_platt), ("Iso", neural_iso), ("Raw", neural_raw)]:
        fused_probs = np.column_stack([bm25_probs, neural])
        fused = np.asarray(log_odds_conjunction(fused_probs), dtype=np.float64)
        methods.append((f"BM25 + Neural ({name})", fused))

    results = []
    print(f"\n  {'Method':<30}  {'BCE':>7}  {'MSE':>7}")
    print(f"  {'-' * 30}  {'-' * 7}  {'-' * 7}")

    for name, preds in methods:
        preds_c = np.clip(preds, 1e-10, 1.0 - 1e-10)
        bce = -float(np.mean(
            test_labels * np.log(preds_c) + (1 - test_labels) * np.log(1 - preds_c)
        ))
        mse = float(np.mean((preds - test_labels) ** 2))

        print(f"  {name:<30}  {bce:>7.4f}  {mse:>7.4f}")
        results.append({"method": name, "bce": bce, "mse": mse})

    return {"methods": results}


def run_timing(rng: np.random.Generator) -> dict:
    """Measure calibrator fit() and calibrate() performance."""
    print("\n" + "=" * 72)
    print("Timing: fit() and calibrate() Performance")
    print("=" * 72)

    sizes = [100, 500, 1000, 5000, 10000]
    results = []

    print(f"\n  {'N':>6}  {'Platt fit ms':>13}  {'Platt cal ms':>13}  "
          f"{'Iso fit ms':>11}  {'Iso cal ms':>11}")
    print(f"  {'-' * 60}")

    for n in sizes:
        scores, labels, _ = generate_calibration_data(n, 2.0, -1.0, rng)
        test_scores = rng.uniform(-3, 5, size=500)

        # Platt timing
        platt = PlattCalibrator()
        t0 = time.perf_counter()
        platt.fit(scores, labels, max_iterations=1000)
        platt_fit_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(100):
            platt.calibrate(test_scores)
        platt_cal_ms = (time.perf_counter() - t0) * 1000 / 100

        # Isotonic timing
        iso = IsotonicCalibrator()
        t0 = time.perf_counter()
        iso.fit(scores, labels)
        iso_fit_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(100):
            iso.calibrate(test_scores)
        iso_cal_ms = (time.perf_counter() - t0) * 1000 / 100

        print(
            f"  {n:>6}  {platt_fit_ms:>13.2f}  {platt_cal_ms:>13.3f}  "
            f"{iso_fit_ms:>11.2f}  {iso_cal_ms:>11.3f}"
        )
        results.append({
            "n": n,
            "platt_fit_ms": platt_fit_ms,
            "platt_cal_ms": platt_cal_ms,
            "iso_fit_ms": iso_fit_ms,
            "iso_cal_ms": iso_cal_ms,
        })

    return {"sizes": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural score calibration benchmark"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    accuracy = run_calibration_accuracy(rng)
    fusion = run_fusion_benchmark(rng)
    timing = run_timing(rng)

    if args.output:
        output = {
            "benchmark": "neural_calibration",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {
                "calibration_accuracy": accuracy,
                "fusion": fusion,
                "timing": timing,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
