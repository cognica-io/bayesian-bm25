#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Streaming calibration statistics over very large evaluation pools.

A calibration pool that covers every document of every query has tens of
millions of (prediction, label) pairs, almost all of them non-relevant.
:class:`CalibrationAccumulator` consumes them in batches of posterior
log-odds and keeps only sufficient statistics, so that the pool never
has to be held in memory.

It reports the reliability-only measure that the first version of the
paper used (ECE) together with measures that also reward resolution,
because at a prevalence near 0.001 a predictor with no resolution at all
can score well on ECE and on the raw Brier score:

* log loss and Brier score, and their skill relative to the constant
  predictor that outputs the pooled prevalence (skill 0 means "no better
  than knowing the base rate", skill 1 is perfect);
* calibration in the large, the ratio of expected to observed relevant
  documents;
* the calibration slope and intercept of a logistic recalibration of the
  labels on the predicted log-odds (1 and 0 for a calibrated predictor;
  a slope below 1 means the predictions are too extreme).
"""

from __future__ import annotations

import math

import numpy as np

from bayesian_bm25.calibration import PlattCalibrator
from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.probability import sigmoid


class CalibrationAccumulator:
    """Accumulates calibration statistics from batches of log-odds.

    Parameters
    ----------
    n_bins : int
        Number of equal-width probability bins for ECE.
    logit_limit : float
        Log-odds are clipped to ``[-logit_limit, logit_limit]`` for the
        logistic recalibration only; every other statistic uses the
        exact values.
    n_logit_bins : int
        Number of log-odds bins for the logistic recalibration.
    """

    def __init__(
        self,
        n_bins: int = 10,
        logit_limit: float = 40.0,
        n_logit_bins: int = 1600,
        context: CalibrationContext | None = None,
    ) -> None:
        if not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        if not isinstance(n_logit_bins, int) or n_logit_bins < 2:
            raise ValueError("n_logit_bins must be an integer of at least 2")
        if not math.isfinite(logit_limit) or logit_limit <= 0.0:
            raise ValueError("logit_limit must be positive and finite")
        self.context = (
            CalibrationContext(
                reference_population="provided", count_source="judgments",
                slope_source="diagnostic", ceiling_source="not_applicable",
            ) if context is None else context
        )
        if not isinstance(self.context, CalibrationContext):
            raise ValueError("context must be a CalibrationContext")
        self._n_bins = n_bins
        self._logit_limit = logit_limit
        self._n_logit_bins = n_logit_bins

        self._count = 0.0
        self._positives = 0.0
        self._sum_p = 0.0
        self._sum_sq_error = 0.0
        self._sum_log_loss = 0.0

        self._bin_count = np.zeros(n_bins, dtype=np.float64)
        self._bin_sum_p = np.zeros(n_bins, dtype=np.float64)
        self._bin_sum_y = np.zeros(n_bins, dtype=np.float64)

        self._logit_count = np.zeros(n_logit_bins, dtype=np.float64)
        self._logit_sum = np.zeros(n_logit_bins, dtype=np.float64)
        self._logit_sum_y = np.zeros(n_logit_bins, dtype=np.float64)

    @property
    def count(self) -> float:
        """Number of (prediction, label) pairs consumed so far."""
        return self._count

    def update(
        self, log_odds: np.ndarray, labels: np.ndarray,
        *, sample_weight: np.ndarray | None = None,
    ) -> None:
        """Consume a batch of posterior log-odds and binary labels.

        Parameters
        ----------
        log_odds : array
            Posterior log-odds.  ``-inf`` is accepted and means
            probability 0.
        labels : array
            Binary relevance labels (0 or 1), same shape.
        """
        x = np.asarray(log_odds, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError(
                f"log_odds shape {x.shape} does not match labels shape {y.shape}"
            )
        if x.size == 0:
            return
        if np.any(np.isnan(x)) or not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("log_odds must not contain NaN and labels must be binary")
        w = np.ones_like(x) if sample_weight is None else np.asarray(
            sample_weight, dtype=np.float64,
        ).ravel()
        if w.shape != x.shape or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("sample_weight must match the inputs and be finite and non-negative")
        active = w > 0.0
        x, y, w = x[active], y[active], w[active]
        if x.size == 0:
            return

        p = np.asarray(sigmoid(x), dtype=np.float64)

        self._count += float(np.sum(w))
        self._positives += float(np.dot(w, y))
        self._sum_p += float(np.dot(w, p))
        self._sum_sq_error += float(np.dot(w, (p - y) ** 2))

        # Log loss from log-odds: softplus(x) - y * x, exact for any
        # finite magnitude.  The limits x -> -inf are 0 for y = 0 and
        # +inf for y = 1.
        loss = np.where(y == 1.0, np.logaddexp(0.0, -x), np.logaddexp(0.0, x))
        self._sum_log_loss += float(np.dot(w, loss))

        # Equal-width probability bins, (lo, hi], first bin closed.
        bin_idx = np.clip(
            np.ceil(p * self._n_bins).astype(np.int64) - 1, 0, self._n_bins - 1,
        )
        self._bin_count += np.bincount(bin_idx, weights=w, minlength=self._n_bins)
        self._bin_sum_p += np.bincount(bin_idx, weights=w * p, minlength=self._n_bins)
        self._bin_sum_y += np.bincount(bin_idx, weights=w * y, minlength=self._n_bins)

        # Log-odds bins for the logistic recalibration.
        clipped = np.clip(x, -self._logit_limit, self._logit_limit)
        width = 2.0 * self._logit_limit / self._n_logit_bins
        logit_idx = np.clip(
            np.floor((clipped + self._logit_limit) / width).astype(np.int64),
            0,
            self._n_logit_bins - 1,
        )
        self._logit_count += np.bincount(logit_idx, weights=w, minlength=self._n_logit_bins)
        self._logit_sum += np.bincount(
            logit_idx, weights=w * clipped, minlength=self._n_logit_bins,
        )
        self._logit_sum_y += np.bincount(
            logit_idx, weights=w * y, minlength=self._n_logit_bins,
        )

    def reliability(self) -> list[tuple[float, float, int | float]]:
        """Reliability diagram: (mean predicted, mean observed, count) per bin."""
        rows = []
        for b in range(self._n_bins):
            count = self._bin_count[b]
            if count == 0:
                continue
            rows.append(
                (
                    float(self._bin_sum_p[b] / count),
                    float(self._bin_sum_y[b] / count),
                    int(count) if count.is_integer() else float(count),
                )
            )
        return rows

    def summary(self) -> dict[str, float | dict[str, str]]:
        """Calibration metrics of everything consumed so far."""
        n = self._count
        if n == 0:
            raise ValueError("no data has been accumulated")

        prevalence = self._positives / n
        ece = float(
            np.sum(np.abs(self._bin_sum_p - self._bin_sum_y)) / n
        )
        brier = self._sum_sq_error / n
        log_loss = self._sum_log_loss / n

        # Reference: the constant predictor at the pooled prevalence.
        brier_ref = prevalence * (1.0 - prevalence)
        if 0.0 < prevalence < 1.0:
            log_loss_ref = -(
                prevalence * math.log(prevalence)
                + (1.0 - prevalence) * math.log(1.0 - prevalence)
            )
        else:
            log_loss_ref = 0.0

        slope, intercept = self._recalibration()

        return {
            "n": n,
            "prevalence": prevalence,
            "ece": ece,
            "brier": brier,
            "log_loss": log_loss,
            "brier_skill": 1.0 - brier / brier_ref if brier_ref > 0.0 else float("nan"),
            "log_loss_skill": (
                1.0 - log_loss / log_loss_ref if log_loss_ref > 0.0 else float("nan")
            ),
            "expected_over_observed": (
                self._sum_p / self._positives if self._positives > 0.0 else float("nan")
            ),
            "count_ratio": (
                self._sum_p / self._positives if self._positives > 0.0 else float("nan")
            ),
            "expected_relevant": self._sum_p,
            "observed_relevant": self._positives,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "context": self.context.as_dict(),
        }

    def _recalibration(self) -> tuple[float, float]:
        """Slope and intercept of the logistic recalibration.

        Fits ``logit P(y = 1) = intercept + slope * log_odds`` on the
        log-odds histogram: each occupied bin contributes its mean
        log-odds, its observed relevant fraction, and its count.
        """
        occupied = self._logit_count > 0
        if (
            int(np.sum(occupied)) < 2
            or self._positives == 0.0
            or self._positives == self._count
        ):
            return float("nan"), float("nan")

        counts = self._logit_count[occupied]
        x = self._logit_sum[occupied] / counts
        y = self._logit_sum_y[occupied] / counts

        calibrator = PlattCalibrator()
        calibrator.fit(x, y, sample_weight=counts, max_iterations=200, tolerance=1e-9)
        return float(calibrator.a), float(calibrator.b)
