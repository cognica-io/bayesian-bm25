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

_VALID_PLATT_METHODS = ("newton", "gradient")


def isotonic_regression(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    increasing: bool = True,
) -> np.ndarray:
    """Weighted least-squares isotonic fit of a sequence (PAVA).

    Returns the sequence closest to ``values`` in weighted squared error
    among all monotone sequences.  The Pool Adjacent Violators
    Algorithm runs in a single pass with a stack of blocks, so the cost
    is linear in the length of the input.

    Parameters
    ----------
    values : array of shape (n,)
        Values to fit, in the order along which monotonicity is
        required.
    weights : array of shape (n,) or None
        Positive weights.  None means unit weights.
    increasing : bool
        Fit a non-decreasing sequence (default) or a non-increasing one.

    Returns
    -------
    array of shape (n,)
        The monotone fit.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != y.shape:
            raise ValueError(
                f"weights shape {w.shape} does not match values shape {y.shape}"
            )
        if np.any(w <= 0.0):
            raise ValueError("weights must be positive")

    signed = y if increasing else -y
    means, lengths = _pool_adjacent_violators(signed, w)
    fitted = np.repeat(means, lengths)
    return fitted if increasing else -fitted


def _pool_adjacent_violators(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocks of the non-decreasing isotonic fit.

    Returns the weighted mean and the length of each block, in order.
    Adjacent blocks are merged only when the earlier mean is strictly
    larger than the later one, so equal neighbours stay separate.
    """
    n = len(values)
    block_sums = np.empty(n, dtype=np.float64)
    block_weights = np.empty(n, dtype=np.float64)
    block_lengths = np.empty(n, dtype=np.int64)
    top = 0

    for i in range(n):
        current_sum = weights[i] * values[i]
        current_weight = weights[i]
        current_length = 1
        # Compare block means by cross-multiplication (weights > 0).
        while (
            top > 0
            and block_sums[top - 1] * current_weight
            > current_sum * block_weights[top - 1]
        ):
            top -= 1
            current_sum += block_sums[top]
            current_weight += block_weights[top]
            current_length += block_lengths[top]
        block_sums[top] = current_sum
        block_weights[top] = current_weight
        block_lengths[top] = current_length
        top += 1

    means = block_sums[:top] / block_weights[:top]
    return means, block_lengths[:top]


class PlattCalibrator:
    """Sigmoid calibration: P = sigmoid(a * score + b).

    Learns parameters ``a`` and ``b`` by minimising binary cross-entropy
    so that ``sigmoid(a * score + b)`` produces well-calibrated
    probabilities.

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
        method: str = "newton",
        sample_weight: np.ndarray | None = None,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """Learn a and b by minimising BCE loss.

        Parameters
        ----------
        scores : array
            Raw model scores.
        labels : array
            Binary relevance labels (0 or 1).
        method : str
            ``"newton"`` (default) runs Newton's method with backtracking
            on internally standardized scores.  The loss is convex, so
            the iteration converges to the maximum-likelihood fit
            whatever the scale or the class balance of the scores.

            ``"gradient"`` runs fixed-step gradient descent on the raw
            scores from the current ``(a, b)``.  Its conditioning
            depends on the scale of the scores: on unstandardized,
            heavily imbalanced data it can stop far from the optimum,
            even with a slope of the wrong sign.
        sample_weight : array or None
            Non-negative weight of each sample.  None means unit weights.
        learning_rate : float
            Step size for ``method="gradient"``; not used by
            ``method="newton"``.
        max_iterations : int
            Maximum number of iterations.
        tolerance : float
            Convergence threshold on the maximum absolute parameter
            change (measured on the standardized parameters for
            ``method="newton"``).
        """
        if method not in _VALID_PLATT_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_PLATT_METHODS}, got {method!r}"
            )

        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if scores.shape != labels.shape:
            raise ValueError(
                f"scores shape {scores.shape} does not match labels shape {labels.shape}"
            )
        if scores.size == 0:
            raise ValueError("scores must not be empty")

        if sample_weight is None:
            weights = np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64).ravel()
            if weights.shape != scores.shape:
                raise ValueError(
                    f"sample_weight shape {weights.shape} does not match "
                    f"scores shape {scores.shape}"
                )
            if np.any(weights < 0.0) or not (float(np.sum(weights)) > 0.0):
                raise ValueError(
                    "sample_weight must be non-negative with a positive sum"
                )
            weights = weights / float(np.sum(weights))

        if method == "gradient":
            self.a, self.b = self._fit_gradient(
                scores, labels, weights, self.a, self.b,
                learning_rate, max_iterations, tolerance,
            )
        else:
            self.a, self.b = self._fit_newton(
                scores, labels, weights, max_iterations, tolerance,
            )

    @staticmethod
    def _fit_gradient(
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        a: float,
        b: float,
        learning_rate: float,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[float, float]:
        """Fixed-step gradient descent on the raw scores."""
        for _ in range(max_iterations):
            predicted = _clamp_probability(sigmoid(a * scores + b))
            error = weights * (predicted - labels)

            grad_a = float(np.sum(error * scores))
            grad_b = float(np.sum(error))

            new_a = a - learning_rate * grad_a
            new_b = b - learning_rate * grad_b

            converged = (
                abs(new_a - a) < tolerance and abs(new_b - b) < tolerance
            )
            a, b = new_a, new_b
            if converged:
                break
        return a, b

    @staticmethod
    def _fit_newton(
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[float, float]:
        """Newton's method with backtracking on standardized scores.

        The model is fit as ``sigmoid(u * z + v)`` with
        ``z = (score - mean) / std`` and mapped back to ``(a, b)``.
        Standardizing decouples the slope from the intercept, which is
        what makes the Hessian well conditioned.
        """
        mean = float(np.sum(weights * scores))
        std = float(np.sqrt(np.sum(weights * (scores - mean) ** 2)))
        positive_rate = float(np.sum(weights * labels))

        if not (std > 0.0):
            # A constant score carries no information: the fit is the
            # base rate.
            rate = float(_clamp_probability(positive_rate))
            return 0.0, float(np.log(rate / (1.0 - rate)))

        z = (scores - mean) / std

        def loss(u: float, v: float) -> float:
            logits = u * z + v
            # BCE = softplus(logit) - label * logit, stable for any logit.
            return float(
                np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits))
            )

        rate = float(_clamp_probability(positive_rate))
        u = 0.0
        v = float(np.log(rate / (1.0 - rate)))
        current_loss = loss(u, v)

        for _ in range(max_iterations):
            predicted = np.asarray(sigmoid(u * z + v), dtype=np.float64)
            error = weights * (predicted - labels)
            curvature = weights * predicted * (1.0 - predicted)

            grad_u = float(np.sum(error * z))
            grad_v = float(np.sum(error))
            h_uu = float(np.sum(curvature * z * z))
            h_uv = float(np.sum(curvature * z))
            h_vv = float(np.sum(curvature))

            determinant = h_uu * h_vv - h_uv * h_uv
            if determinant > 1e-300:
                step_u = (h_vv * grad_u - h_uv * grad_v) / determinant
                step_v = (h_uu * grad_v - h_uv * grad_u) / determinant
            else:
                # The Hessian is numerically singular (every prediction
                # has saturated): fall back to the gradient direction.
                step_u, step_v = grad_u, grad_v

            # Backtracking line search on the convex loss.
            slope = grad_u * step_u + grad_v * step_v
            step_size = 1.0
            new_u, new_v = u - step_u, v - step_v
            new_loss = loss(new_u, new_v)
            while new_loss > current_loss - 1e-4 * step_size * slope and step_size > 1e-12:
                step_size *= 0.5
                new_u = u - step_size * step_u
                new_v = v - step_size * step_v
                new_loss = loss(new_u, new_v)

            change = max(abs(new_u - u), abs(new_v - v))
            if new_loss <= current_loss:
                u, v, current_loss = new_u, new_v, new_loss
            if change < tolerance:
                break
            if current_loss < 1e-12:
                # Perfectly separable data: the likelihood has no finite
                # maximiser, and the fit is already exact to rounding.
                break

        return u / std, v - u * mean / std

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

    def log_odds(self, scores: np.ndarray | float) -> np.ndarray | float:
        """Calibrated log-odds: a * scores + b.

        The logit of :meth:`calibrate`, without rounding.  Rank and fuse
        on this value; a probability saturates for large scores.
        """
        result = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return float(result) if np.ndim(result) == 0 else result

    def __call__(self, scores: np.ndarray | float) -> np.ndarray | float:
        return self.calibrate(scores)


def __getattr__(name: str):
    """Preserve the public import after splitting the main classes."""
    if name == "IsotonicCalibrator":
        from bayesian_bm25.isotonic_calibrator import IsotonicCalibrator

        globals()[name] = IsotonicCalibrator
        return IsotonicCalibrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
