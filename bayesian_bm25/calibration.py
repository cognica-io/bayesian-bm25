#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Neural score calibration for integrating external model scores.

Provides sigmoid (Platt) and isotonic (PAVA) calibrators that convert
raw scores from neural models or other sources into calibrated
probabilities suitable for Bayesian fusion via ``log_odds_conjunction``.

Paper ref: Paper 1, Section 12.2 #5; Paper 2, Section 5.1
"""

from __future__ import annotations

import numpy as np

from bayesian_bm25.probability import _clamp_probability, sigmoid


class PlattCalibrator:
    """Sigmoid calibration: P = sigmoid(a * score + b).

    Learns parameters ``a`` and ``b`` via BCE gradient descent so that
    ``sigmoid(a * score + b)`` produces well-calibrated probabilities.

    Parameters
    ----------
    a : float
        Initial slope (scale) parameter.
    b : float
        Initial intercept (shift) parameter.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = a
        self.b = b

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """Learn a and b via gradient descent on BCE loss.

        Parameters
        ----------
        scores : array
            Raw model scores.
        labels : array
            Binary relevance labels (0 or 1).
        learning_rate : float
            Step size for gradient descent.
        max_iterations : int
            Maximum number of gradient descent iterations.
        tolerance : float
            Convergence threshold on maximum absolute parameter change.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        a = self.a
        b = self.b

        for _ in range(max_iterations):
            predicted = _clamp_probability(sigmoid(a * scores + b))
            error = predicted - labels

            grad_a = float(np.mean(error * scores))
            grad_b = float(np.mean(error))

            new_a = a - learning_rate * grad_a
            new_b = b - learning_rate * grad_b

            if abs(new_a - a) < tolerance and abs(new_b - b) < tolerance:
                a = new_a
                b = new_b
                break

            a = new_a
            b = new_b

        self.a = a
        self.b = b

    def calibrate(self, scores: np.ndarray | float) -> np.ndarray | float:
        """Apply sigmoid calibration: sigmoid(a * scores + b).

        Parameters
        ----------
        scores : float or array
            Raw model scores.

        Returns
        -------
        Calibrated probabilities in (0, 1).
        """
        scores = np.asarray(scores, dtype=np.float64)
        result = sigmoid(self.a * scores + self.b)
        return float(result) if np.ndim(result) == 0 else result

    def __call__(self, scores: np.ndarray | float) -> np.ndarray | float:
        return self.calibrate(scores)


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
        n = len(x_sorted)
        block_sums = y_sorted.copy()
        block_counts = np.ones(n, dtype=np.float64)
        block_x_sums = x_sorted.copy()
        active = list(range(n))

        merged = True
        while merged:
            merged = False
            new_active = [active[0]]
            for j in range(1, len(active)):
                prev = new_active[-1]
                curr = active[j]
                val_prev = block_sums[prev] / block_counts[prev]
                val_curr = block_sums[curr] / block_counts[curr]
                if val_prev > val_curr:
                    # Merge curr into prev
                    block_sums[prev] += block_sums[curr]
                    block_counts[prev] += block_counts[curr]
                    block_x_sums[prev] += block_x_sums[curr]
                    merged = True
                else:
                    new_active.append(curr)
            active = new_active

        # Extract breakpoints: mean score and mean label for each block
        result_x = np.array(
            [block_x_sums[i] / block_counts[i] for i in active],
            dtype=np.float64,
        )
        result_y = np.array(
            [block_sums[i] / block_counts[i] for i in active],
            dtype=np.float64,
        )

        self._x = result_x
        self._y = result_y

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
