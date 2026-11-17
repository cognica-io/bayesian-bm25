#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Calibration and convex-fusion baselines using exactly the sampled labels."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> dict:
    """Bounded maximum likelihood; no validation labels select a penalty.

    Bounds make all-positive samples and separated samples explicit finite
    estimators. They are fixed across datasets and budgets. Features are
    standardized on these same observations, and the coefficients returned
    here operate on the original feature units.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) == 0 or y.shape != (len(x),):
        raise ValueError("a nonempty matching labeled sample is required")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-12] = 1.0
    normalized = (x - mean) / std

    def objective(parameters):
        logits = normalized @ parameters[:-1] + parameters[-1]
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        residual = (expit(logits) - y) / len(y)
        gradient = np.append(normalized.T @ residual, np.sum(residual))
        return float(loss), gradient

    prevalence = (float(np.sum(y)) + 0.5) / (len(y) + 1.0)
    initial = np.append(np.zeros(x.shape[1]), np.log(prevalence / (1.0 - prevalence)))
    bounds = [(-20.0, 20.0)] * x.shape[1] + [(-30.0, 30.0)]
    fit = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    gradient = fit.jac.copy()
    for i, (lower, upper) in enumerate(bounds):
        if (fit.x[i] <= lower + 1e-8 and gradient[i] > 0.0) or (
            fit.x[i] >= upper - 1e-8 and gradient[i] < 0.0
        ):
            gradient[i] = 0.0
    projected_gradient = float(np.max(np.abs(gradient)))
    if projected_gradient > 1e-5:
        raise RuntimeError(f"logistic baseline fit failed: {fit.message}")
    coefficients = fit.x[:-1] / std
    return {
        "coefficients": coefficients.tolist(),
        "intercept": float(fit.x[-1] - np.dot(coefficients, mean)),
        "n_judgments": len(y),
        "n_relevant": int(np.sum(y)),
        "training_log_loss": float(fit.fun),
        "estimator": "bounded_maximum_likelihood",
        "standardized_coefficient_bound": 20.0,
        "intercept_bound": 30.0,
        "projected_gradient_max": projected_gradient,
    }


def predict_logistic(fit: dict, features: np.ndarray) -> np.ndarray:
    """Log-odds from a model fitted on the supplied judgment budget."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x @ np.asarray(fit["coefficients"]) + fit["intercept"]


def fit_convex(features: np.ndarray, labels: np.ndarray) -> dict:
    """Tune a convex ranking weight on the identical document judgments.

    The two columns are per-query min-max lexical and dense scores. For
    each weight, fit a monotone logistic map to those same labels; select
    by training log loss. No additional qrels or test queries are used.
    The raw convex score remains a ranking score; only its fitted logistic
    map is assessed as a probability.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("two features and a nonempty matching labeled sample are required")
    best = None
    for weight in np.linspace(0.0, 1.0, 21):
        score = (1.0 - weight) * x[:, 0] + weight * x[:, 1]

        def profile(slope, score=score):
            # For fixed slope the intercept derivative is monotone. Its
            # bracketed root (or the appropriate boundary) is the exact
            # constrained optimum. The resulting profile is convex in
            # slope, so a second bracketed derivative root solves the
            # complete bounded maximum-likelihood problem.
            def intercept_gradient(intercept):
                return float(np.mean(expit(slope * score + intercept) - y))

            if intercept_gradient(-50.0) >= 0.0:
                intercept = -50.0
            elif intercept_gradient(50.0) <= 0.0:
                intercept = 50.0
            else:
                intercept = brentq(intercept_gradient, -50.0, 50.0, xtol=1e-12)
            logits = slope * score + intercept
            residual = expit(logits) - y
            loss = np.where(y == 1.0, np.logaddexp(0.0, -logits), np.logaddexp(0.0, logits))
            return float(np.mean(loss)), float(np.mean(score * residual)), intercept

        if profile(0.0)[1] >= -1e-12:
            slope = 0.0
        elif profile(100.0)[1] <= 1e-12:
            slope = 100.0
        else:
            slope = brentq(lambda a: profile(a)[1], 0.0, 100.0, xtol=1e-10)
        loss, gradient, intercept = profile(slope)
        if best is None or loss < best["training_log_loss"]:
            best = {
                "dense_weight": float(weight),
                "slope": float(slope),
                "intercept": float(intercept),
                "training_log_loss": loss,
                "n_judgments": len(y),
                "n_relevant": int(np.sum(y)),
                "weight_grid": "0:0.05:1",
                "slope_bound": [0.0, 100.0],
                "intercept_bound": [-50.0, 50.0],
            }
    return best


def predict_convex(fit: dict, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ranking scores separately from fitted posterior log-odds."""
    weight = fit["dense_weight"]
    scores = (1.0 - weight) * features[:, 0] + weight * features[:, 1]
    return scores, fit["slope"] * scores + fit["intercept"]
