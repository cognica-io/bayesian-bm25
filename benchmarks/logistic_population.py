#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""A full empirical population for repeated count-constrained grid fits.

The quadrature is a numerical acceleration, not a sampled background.
Every score contributes. Within bins, discrete Gaussian quadrature
matches moments through degree 2 * order - 1. A Cauchy remainder bound
for the logistic function certifies each solved count. If that bound
exceeds the requested tolerance, the ordinary full-array solver is used.
Selected models are evaluated with the ordinary solver independently.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from bayesian_bm25.consistency import solve_offset


class LogisticPopulation:
    """Positive discrete quadrature of a finite score population."""

    def __init__(
        self,
        scores: np.ndarray,
        *,
        max_slope: float = 2.0**1.5,
        order: int = 10,
        tolerance: float = 1e-10,
    ) -> None:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("scores must be a nonempty finite vector")
        if max_slope <= 0 or order < 2 or tolerance <= 0:
            raise ValueError("max_slope, order and tolerance must be positive")
        self.scores = values
        self.population = len(values)
        self.max_slope = float(max_slope)
        self.order = order
        self.tolerance = tolerance
        self.fallback_count = 0
        self.max_relative_count_bound = 0.0
        self._build()

    def _build(self) -> None:
        # At the largest slope, the distance from a bin's center is at
        # most 0.1. Logistic is analytic and bounded by 1 on every disk
        # of radius 1 around a real argument (|Im z| < pi/2).
        self.half_width = 0.1 / self.max_slope
        width = 2.0 * self.half_width
        bins, group = np.unique(np.floor(self.scores / width), return_inverse=True)
        centers = (bins + 0.5) * width
        mass = np.bincount(group).astype(np.float64)
        x = (self.scores - centers[group]) / self.half_width
        n_bins = len(bins)

        q = 1.0 / np.sqrt(mass[group])
        previous = np.zeros_like(q)
        beta_previous = np.zeros(n_bins)
        jacobi = np.zeros((n_bins, self.order, self.order))
        for step in range(self.order):
            residual = x * q - beta_previous[group] * previous
            alpha = np.bincount(group, weights=q * residual, minlength=n_bins)
            residual -= alpha[group] * q
            beta = np.sqrt(np.bincount(group, weights=residual**2, minlength=n_bins))
            jacobi[:, step, step] = alpha
            if step + 1 < self.order:
                jacobi[:, step, step + 1] = beta
                jacobi[:, step + 1, step] = beta
            previous, q = (
                q,
                np.divide(
                    residual,
                    beta[group],
                    out=np.zeros_like(residual),
                    where=beta[group] > 1e-14,
                ),
            )
            beta_previous = beta

        nodes, vectors = np.linalg.eigh(jacobi)
        weights = mass[:, None] * vectors[:, 0, :] ** 2
        weights *= mass[:, None] / np.sum(weights, axis=1, keepdims=True)

        # Small bins retain their literal points; quadrature would cost
        # more than direct evaluation and could amplify a Lanczos breakdown.
        compressed = mass > self.order
        keep_original = ~compressed[group]
        self.nodes = np.concatenate(
            [
                self.scores[keep_original],
                (centers[:, None] + self.half_width * nodes)[compressed].ravel(),
            ]
        )
        self.weights = np.concatenate(
            [
                np.ones(int(np.sum(keep_original))),
                weights[compressed].ravel(),
            ]
        )
        active = self.weights > 0.0
        self.nodes, self.weights = self.nodes[active], self.weights[active]
        self.centers = centers[compressed]
        self.mass = mass[compressed]

        # Bound the floating-point moment mismatch as well as truncation.
        # For degree d, both true and quadrature measures have mass n and
        # support within the bin. Their Taylor tails are bounded by
        # 2*n*a^(d+1)/(1-a). Moment residuals cover finite arithmetic.
        degree = 2 * self.order - 1
        residuals = []
        power = np.ones_like(x)
        node_power = np.ones_like(nodes)
        for _exponent in range(degree + 1):
            empirical = np.bincount(group, weights=power, minlength=n_bins)
            quadrature = np.sum(weights * node_power, axis=1)
            residuals.append(np.abs(empirical - quadrature)[compressed])
            power *= x
            node_power *= nodes
        self.moment_errors = np.asarray(residuals)
        # Eigensolver roundoff can move nodes a few ulps outside [-1, 1].
        self.support_radius = max(1.0, float(np.max(np.abs(nodes))))

    def count(self, slope: float, offset: float) -> float:
        """Numerical sum of sigmoid(slope * score - offset)."""
        return float(np.dot(self.weights, expit(slope * self.nodes - offset)))

    def count_error_bound(self, slope: float, offset: float) -> float:
        """Absolute count error bound including moment roundoff."""
        if self.mass.size == 0:
            return 0.0
        a = slope * self.half_width * self.support_radius
        if a >= 1.0:
            return math.inf
        powers = (slope * self.half_width) ** np.arange(2 * self.order)
        per_bin = powers @ self.moment_errors
        per_bin += 2.0 * self.mass * a ** (2 * self.order) / (1.0 - a)
        # A tighter Cauchy bound in dilute bins: |sigmoid(z)| <= exp(Re z).
        magnitude = np.exp(np.minimum(slope * self.centers - offset + 1.0, 0.0))
        return float(np.dot(magnitude, per_bin))

    def solve(self, slope: float, expected_latent: float) -> float:
        """Solve on all documents, to the full-array solver's count tolerance."""
        if not 0.0 < slope <= self.max_slope:
            raise ValueError("slope must lie in (0, max_slope]")
        if not 0.0 < expected_latent < self.population:
            raise ValueError("expected_latent must lie in (0, population)")
        prior = math.log(expected_latent) - math.log(self.population - expected_latent)
        lo = slope * float(np.min(self.scores)) - prior
        hi = slope * float(np.max(self.scores)) - prior
        if hi == lo:
            return lo
        root = brentq(
            lambda c: self.count(slope, c) - expected_latent,
            lo,
            hi,
            xtol=1e-12,
            rtol=1e-14,
        )
        error = self.count_error_bound(slope, root)
        error += abs(self.count(slope, root) - expected_latent)
        self.max_relative_count_bound = max(
            self.max_relative_count_bound,
            error / expected_latent,
        )
        if error > self.tolerance * expected_latent:
            self.fallback_count += 1
            return solve_offset(slope * self.scores, expected_latent)
        return float(root)
