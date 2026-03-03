#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Calibration metrics for evaluating probability quality.

Provides Expected Calibration Error (ECE), Brier score, and reliability
diagram data for assessing how well predicted probabilities match actual
relevance rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _bin_mask(probabilities: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Create a boolean mask for probabilities falling in a bin (lo, hi].

    The first bin is inclusive on both sides: [0, hi].
    All other bins are left-exclusive, right-inclusive: (lo, hi].
    """
    if lo == 0:
        return (probabilities >= lo) & (probabilities <= hi)
    return (probabilities > lo) & (probabilities <= hi)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual relevance
    rates.  Lower is better.  Perfect calibration = 0.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probabilities)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = _bin_mask(probabilities, lo, hi)
        count = np.sum(mask)
        if count == 0:
            continue
        avg_prob = np.mean(probabilities[mask])
        avg_label = np.mean(labels[mask])
        ece += (count / total) * abs(avg_prob - avg_label)

    return float(ece)


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Brier score: mean squared error between probabilities and labels.

    Decomposes into calibration + discrimination.  Lower is better.
    A constant prediction of base rate achieves the reference score.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))


def reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Compute reliability diagram data: (avg_predicted, avg_actual, count) per bin.

    Perfect calibration means avg_predicted == avg_actual for every bin.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = _bin_mask(probabilities, lo, hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_pred = float(np.mean(probabilities[mask]))
        avg_actual = float(np.mean(labels[mask]))
        bins.append((avg_pred, avg_actual, count))
    return bins


@dataclass
class CalibrationReport:
    """One-call calibration diagnostic report.

    Bundles ECE, Brier score, and reliability diagram data into a single
    object with a human-readable ``summary()`` method.
    """

    ece: float
    brier: float
    reliability: list[tuple[float, float, int]]
    n_samples: int
    n_bins: int

    def summary(self) -> str:
        """Formatted text summary of calibration metrics."""
        lines = [
            "Calibration Report",
            "==================",
            f"  Samples : {self.n_samples}",
            f"  Bins    : {self.n_bins}",
            f"  ECE     : {self.ece:.6f}",
            f"  Brier   : {self.brier:.6f}",
            "",
            "  Reliability Diagram",
            "  -------------------",
            f"  {'Predicted':>10}  {'Actual':>10}  {'Count':>6}",
        ]
        for avg_pred, avg_actual, count in self.reliability:
            lines.append(f"  {avg_pred:>10.4f}  {avg_actual:>10.4f}  {count:>6}")
        return "\n".join(lines)


def calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute a full calibration diagnostic report in one call.

    Parameters
    ----------
    probabilities : ndarray
        Predicted probabilities.
    labels : ndarray
        Binary relevance labels (0 or 1).
    n_bins : int
        Number of bins for ECE and reliability diagram.

    Returns
    -------
    CalibrationReport with ECE, Brier score, and reliability diagram data.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    ece = expected_calibration_error(probabilities, labels, n_bins=n_bins)
    brier = brier_score(probabilities, labels)
    reliability = reliability_diagram(probabilities, labels, n_bins=n_bins)

    return CalibrationReport(
        ece=ece,
        brier=brier,
        reliability=reliability,
        n_samples=len(probabilities),
        n_bins=n_bins,
    )
