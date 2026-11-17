#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Non-parametric monotone score calibration."""

from __future__ import annotations

import numpy as np

from bayesian_bm25.calibration import _pool_adjacent_violators
from bayesian_bm25.probability import _clamp_probability


class IsotonicCalibrator:
    """Non-parametric monotone calibration via PAVA (numpy-only).

    The Pool Adjacent Violators Algorithm produces a monotonically
    non-decreasing mapping from scores to probabilities.  At inference
    time, binary search with linear interpolation maps new scores to
    calibrated probabilities.
    """

    def __init__(self) -> None:
        self._x: np.ndarray | None = None  # sorted score breakpoints
        self._y: np.ndarray | None = None  # corresponding calibrated values

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression via PAVA.

        Parameters
        ----------
        scores : array
            Raw model scores.
        labels : array
            Binary relevance labels (0 or 1).
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        # Sort by scores
        order = np.argsort(scores)
        x_sorted = scores[order]
        y_sorted = labels[order]

        # PAVA: merge adjacent blocks that violate monotonicity
        block_means, block_lengths = _pool_adjacent_violators(
            y_sorted, np.ones_like(y_sorted),
        )

        # Breakpoints: mean score and mean label of each block
        block_starts = np.concatenate(([0], np.cumsum(block_lengths)[:-1]))
        self._x = np.add.reduceat(x_sorted, block_starts) / block_lengths
        self._y = block_means

    def calibrate(self, scores: np.ndarray | float) -> np.ndarray | float:
        """Apply isotonic calibration via binary search + interpolation.

        Parameters
        ----------
        scores : float or array
            Raw model scores.

        Returns
        -------
        Calibrated probabilities in (0, 1).
        """
        if self._x is None or self._y is None:
            raise RuntimeError("Call fit() before calibrate().")

        scores_arr = np.asarray(scores, dtype=np.float64)
        scalar = scores_arr.ndim == 0
        scores_arr = np.atleast_1d(scores_arr)

        x = self._x
        y = self._y

        # Binary search: find insertion points
        indices = np.searchsorted(x, scores_arr)

        result = np.empty_like(scores_arr)
        for i, (s, idx) in enumerate(zip(scores_arr, indices, strict=True)):
            if idx == 0:
                result[i] = y[0]
            elif idx >= len(x):
                result[i] = y[-1]
            else:
                # Linear interpolation between adjacent breakpoints
                x0, x1 = x[idx - 1], x[idx]
                y0, y1 = y[idx - 1], y[idx]
                if x1 - x0 < 1e-12:
                    result[i] = (y0 + y1) / 2.0
                else:
                    t = (s - x0) / (x1 - x0)
                    result[i] = y0 + t * (y1 - y0)

        result = _clamp_probability(result)
        return float(result[0]) if scalar else result

    def __call__(self, scores: np.ndarray | float) -> np.ndarray | float:
        return self.calibrate(scores)
