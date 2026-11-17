#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Vector score models with explicit background and retrieval-selection roles.

For a non-relevant background, posterior log-odds add the log density
ratio and prior log-odds. For a marginal background, Bayes' rule instead
gives P(R|d) = P_base * relevance_recall * f_R_retrieved(d) / f_G(d).
The latter ratio before posterior conversion is not prior-free evidence.
Estimated densities, external weights and Gaussian tail assumptions must
be validated against judgments; none of them identifies relevance alone.

The core class ``VectorProbabilityTransform`` is fully parameterized
(index-free). Index-specific helpers ``ivf_density_prior`` and
``knn_density_prior`` are standalone utility functions that users call
externally.

Paper ref: From Scores to Evidence, Section 4.
"""

from __future__ import annotations

import math

import numpy as np

from bayesian_bm25.calibration import isotonic_regression
from bayesian_bm25.probability import _EPSILON, _clamp_probability, sigmoid

_VALID_POSTERIORS = ("approximate", "exact")
_VALID_BACKGROUND_KINDS = ("marginal", "nonrelevant")

# Largest posterior the exact form may return; its logit is about 23.0.
_LOG_MAX_POSTERIOR = math.log1p(-_EPSILON)

# Number of bins used to evaluate an empirical background density.
_BACKGROUND_BINS = 4096

# ---------------------------------------------------------------------------
# Private helpers (module-level)
# ---------------------------------------------------------------------------


def _log_gaussian_pdf(
    x: np.ndarray | float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Log of the Gaussian density, exact in the tails.

    log f(x) = -0.5 * ((x - mu) / sigma)^2 - log(sigma * sqrt(2*pi))

    The density itself underflows about 38 standard deviations from the
    mean; its logarithm does not.
    """
    x = np.asarray(x, dtype=np.float64)
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2.0 * math.pi))


def _log_kernel_density(
    eval_points: np.ndarray,
    sample_points: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Log of the weighted Gaussian KDE, evaluated by log-sum-exp.

    log f(x) = log sum_i w_i * K_h(x - x_i) - log sum_i w_i

    An evaluation point far from every kernel centre has a density that
    underflows to zero; the logarithm stays finite and keeps decreasing
    with the distance to the nearest centre.
    """
    eval_points = np.asarray(eval_points, dtype=np.float64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    w_sum = float(np.sum(weights))
    if w_sum < _EPSILON:
        return np.full(eval_points.shape, math.log(_EPSILON), dtype=np.float64)

    active = weights > 0.0
    centres = sample_points[active]
    log_w = np.log(weights[active])
    log_norm = math.log(w_sum) + math.log(bandwidth * math.sqrt(2.0 * math.pi))

    result = np.empty(eval_points.shape, dtype=np.float64)
    for start in range(0, len(eval_points), chunk_size):
        block = eval_points[start:start + chunk_size]
        # Shape: (n_block, n_centres)
        exponent = (
            -0.5 * ((block[:, np.newaxis] - centres[np.newaxis, :]) / bandwidth) ** 2
            + log_w[np.newaxis, :]
        )
        peak = np.max(exponent, axis=1)
        result[start:start + chunk_size] = (
            peak + np.log(np.sum(np.exp(exponent - peak[:, np.newaxis]), axis=1))
        )
    return result - log_norm


def _log_binned_kernel_density(
    eval_points: np.ndarray,
    sample_points: np.ndarray,
    bandwidth: float,
    n_bins: int = _BACKGROUND_BINS,
) -> np.ndarray:
    """Log of an unweighted Gaussian KDE over a large sample.

    The sample is first reduced to a histogram, whose bin centres and
    counts stand in for the individual points.  With many more points
    than bins this makes the cost independent of the sample size; the
    bins must be narrow relative to the bandwidth, which the bin count
    ensures for any bandwidth of practical interest.
    """
    sample_points = np.asarray(sample_points, dtype=np.float64)
    if len(sample_points) <= n_bins:
        return _log_kernel_density(
            eval_points, sample_points, np.ones_like(sample_points), bandwidth,
        )

    lo = float(np.min(sample_points))
    hi = float(np.max(sample_points))
    if hi - lo < _EPSILON:
        return _log_kernel_density(
            eval_points, np.array([lo]), np.array([1.0]), bandwidth,
        )
    counts, edges = np.histogram(sample_points, bins=n_bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return _log_kernel_density(
        eval_points, centres, counts.astype(np.float64), bandwidth,
    )


def _gaussian_pdf(
    x: np.ndarray | float,
    mu: float,
    sigma: float,
) -> np.ndarray | float:
    """Gaussian density without scipy dependency.

    f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    x = np.asarray(x, dtype=np.float64)
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    z = (x - mu) / sigma
    result = coeff * np.exp(-0.5 * z * z)
    return float(result) if result.ndim == 0 else result


def _silverman_bandwidth(
    distances: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted Silverman bandwidth (Definition 4.8 of "From Scores to Evidence").

        h = (4 / (3 n)) ** (1/5) * sigma_w

    where ``n`` is the number of distances and ``sigma_w`` their weighted
    standard deviation.  The weights enter through the dispersion only.
    """
    distances = np.asarray(distances, dtype=np.float64)

    weights = np.ones_like(distances) if weights is None else np.asarray(weights, dtype=np.float64)

    w_sum = float(np.sum(weights))
    n = distances.size

    if w_sum < _EPSILON or n == 0:
        return _EPSILON

    w_mean = float(np.sum(weights * distances) / w_sum)
    w_var = float(np.sum(weights * (distances - w_mean) ** 2) / w_sum)
    sigma_w = math.sqrt(max(w_var, 0.0))

    if sigma_w < _EPSILON:
        return _EPSILON

    h = (4.0 / (3.0 * n)) ** 0.2 * sigma_w
    return max(h, _EPSILON)


def _kernel_density(
    eval_points: np.ndarray,
    sample_points: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Weighted Gaussian KDE (Definition 4.3.1).

    f(x) = sum(w_i * K_h(x - x_i)) / sum(w_i)

    where K_h is a Gaussian kernel with bandwidth h.
    """
    eval_points = np.asarray(eval_points, dtype=np.float64)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    w_sum = float(np.sum(weights))
    if w_sum < _EPSILON:
        return np.full_like(eval_points, _EPSILON)

    # Shape: (n_eval, n_sample)
    diff = eval_points[:, np.newaxis] - sample_points[np.newaxis, :]

    kernel_vals = (
        np.exp(-0.5 * (diff / bandwidth) ** 2)
        / (bandwidth * math.sqrt(2.0 * math.pi))
    )

    density = np.sum(weights[np.newaxis, :] * kernel_vals, axis=1) / w_sum
    return np.maximum(density, _EPSILON)


# ---------------------------------------------------------------------------
# VectorProbabilityTransform
# ---------------------------------------------------------------------------


class VectorProbabilityTransform:
    """Model vector relevance using a declared background density.

    Uses a likelihood ratio framework (Theorem 3.1.1):

        P(R|d) = sigmoid(log(f_R(d) / f_G(d)) + logit(P_base))

    This form is exact for a non-relevant background. For a marginal
    background, the posterior is ``P_base * f_R / f_G`` instead.
    Weighted retrieved samples need not recover the relevant density,
    and a Gaussian background's moments do not identify its tail.

    Parameters
    ----------
    mu_G : float
        Mean of the background Gaussian density.
    sigma_G : float
        Standard deviation of the background Gaussian density.
    base_rate : float or None
        Corpus-level base rate of relevance P_base, in (0, 1).
        None defaults to 0.5 (neutral logit = 0).
    background_kind : str or None
        ``"nonrelevant"`` for a class-conditional negative density or
        ``"marginal"`` for the scored population. Defaults to marginal
        with a background sample and nonrelevant with Gaussian moments.
    background_distances : array or None
        Background sample used for KDE. Without one, use the Gaussian
        specified by ``mu_G`` and ``sigma_G`` as an explicit shape model.
    """

    def __init__(
        self,
        mu_G: float,
        sigma_G: float,
        base_rate: float | None = None,
        *,
        background_kind: str | None = None,
        background_distances: np.ndarray | None = None,
    ) -> None:
        if not math.isfinite(mu_G):
            raise ValueError("mu_G must be finite")
        if not math.isfinite(sigma_G) or sigma_G <= 0.0:
            raise ValueError(f"sigma_G must be positive, got {sigma_G}")
        if base_rate is not None and not (0.0 < base_rate < 1.0):
            raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")

        self.mu_G = float(mu_G)
        self.sigma_G = float(sigma_G)
        self.base_rate = base_rate
        self._logit_base_rate = (
            math.log(base_rate) - math.log1p(-base_rate) if base_rate is not None else 0.0
        )
        self.background_kind = (
            ("marginal" if background_distances is not None else "nonrelevant")
            if background_kind is None else background_kind
        )
        if self.background_kind not in _VALID_BACKGROUND_KINDS:
            raise ValueError("background_kind must be 'marginal' or 'nonrelevant'")
        self._background_distances = None
        if background_distances is not None:
            sample = np.asarray(background_distances, dtype=np.float64).ravel()
            if sample.size == 0 or not np.all(np.isfinite(sample)):
                raise ValueError("background_distances must be non-empty and finite")
            self._background_distances = sample.copy()
        self.background_model = "gaussian" if self._background_distances is None else "kde"

    @classmethod
    def fit_background(
        cls,
        distances: np.ndarray,
        *,
        base_rate: float | None = None,
        background_kind: str = "marginal",
        background_model: str = "kde",
    ) -> VectorProbabilityTransform:
        """Fit a background model to a representative population sample.

        The default is a marginal KDE, which retains the sample for
        density evaluation. ``background_model="gaussian"`` retains
        only its moments and is a separate tail-shape assumption.

        Parameters
        ----------
        distances : array
            Distance/similarity values from a representative corpus sample.
        base_rate : float or None
            Corpus-level base rate of relevance.

        Returns
        -------
        VectorProbabilityTransform
            A new instance with estimated mu_G, sigma_G.
        """
        distances = np.asarray(distances, dtype=np.float64).ravel()
        if distances.size == 0 or not np.all(np.isfinite(distances)):
            raise ValueError("background distances must be non-empty and finite")
        if background_model not in ("kde", "gaussian"):
            raise ValueError("background_model must be 'kde' or 'gaussian'")
        mu_G = float(np.mean(distances))
        sigma_G = float(np.std(distances))
        if sigma_G < _EPSILON:
            sigma_G = _EPSILON
        return cls(
            mu_G=mu_G, sigma_G=sigma_G, base_rate=base_rate,
            background_kind=background_kind,
            background_distances=distances if background_model == "kde" else None,
        )

    def _detect_gap(
        self,
        distances: np.ndarray,
        threshold_ratio: float = 0.15,
    ) -> int | None:
        """Strategy 4.6.1: find semantic cliff in sorted distances.

        Returns the index (in sorted order) of the first element AFTER
        the gap, or None if no significant gap is detected.

        Dual threshold:
        - Primary: span ratio >= threshold_ratio (gap / total_span)
        - Fallback: z-score > 2.0 (gap size relative to mean gap)
        """
        distances = np.asarray(distances, dtype=np.float64)
        if len(distances) < 3:
            return None

        sorted_d = np.sort(distances)
        gaps = np.diff(sorted_d)

        if len(gaps) == 0:
            return None

        total_span = sorted_d[-1] - sorted_d[0]
        if total_span < _EPSILON:
            return None

        # Primary: span ratio threshold
        gap_ratios = gaps / total_span
        max_ratio_idx = int(np.argmax(gap_ratios))

        if gap_ratios[max_ratio_idx] >= threshold_ratio:
            return max_ratio_idx + 1

        # Fallback: z-score threshold
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        if std_gap > _EPSILON:
            z_scores = (gaps - mean_gap) / std_gap
            max_z_idx = int(np.argmax(z_scores))
            if z_scores[max_z_idx] > 2.0:
                return max_z_idx + 1

        return None

    def _gap_weights(self, distances: np.ndarray) -> np.ndarray | None:
        """Binary weights from gap partition (Strategy 4.6.1).

        Returns 1.0 for distances below the gap, 0.0 above.
        Returns None if no gap is detected.
        """
        distances = np.asarray(distances, dtype=np.float64)
        sorted_d = np.sort(distances)
        gap_idx = self._detect_gap(distances)

        if gap_idx is None:
            return None

        threshold = sorted_d[gap_idx]
        return np.where(distances < threshold, 1.0, 0.0)

    @staticmethod
    def _sharpen_weights(
        weights: np.ndarray,
        temperature: float = 0.05,
    ) -> np.ndarray:
        """Softmax temperature sharpening for BM25 probability weights.

        Applies exp((w - max(w)) / T) to create contrast from flat
        probability vectors, then rescales to preserve total mass.

        Parameters
        ----------
        weights : array
            Input weights (e.g., BM25 probabilities).
        temperature : float
            Softmax temperature. Lower values create sharper contrast.
        """
        weights = np.asarray(weights, dtype=np.float64)
        total_mass = float(np.sum(weights))

        w_max = float(np.max(weights))
        sharpened = np.exp((weights - w_max) / temperature)

        sharp_sum = float(np.sum(sharpened))
        if sharp_sum > _EPSILON:
            sharpened = sharpened * (total_mass / sharp_sum)

        return sharpened

    @staticmethod
    def _distance_density_weights(distances: np.ndarray) -> np.ndarray:
        """Ultimate fallback: density weights from distances.

        w_i = sigmoid(median(d) / d_i - 1)

        Closer distances (smaller d_i) get higher weight.
        """
        distances = np.asarray(distances, dtype=np.float64)
        median_d = float(np.median(distances))
        safe_d = np.maximum(distances, _EPSILON)
        raw = median_d / safe_d - 1.0
        return np.asarray(sigmoid(raw))

    def estimate_kde(
        self,
        distances: np.ndarray,
        weights: np.ndarray,
        bandwidth_factor: float = 1.0,
        *,
        eval_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Section 4.3: weighted KDE for relevant-document density f_R.

        f_R(d) = sum(w_i * K_h(d - d_i)) / sum(w_i)

        Parameters
        ----------
        distances : array
            Distance values to evaluate and use as kernel centers.
        weights : array
            Per-sample weights for the KDE.
        bandwidth_factor : float
            Multiplicative factor applied to the Silverman bandwidth.

        Returns
        -------
        array
            Density values f_R at each distance.
        """
        log_density, _ = self._estimate_log_kde(
            distances, weights, bandwidth_factor, eval_points=eval_points,
        )
        return np.maximum(np.exp(log_density), _EPSILON)

    def _estimate_log_kde(
        self,
        distances: np.ndarray,
        weights: np.ndarray,
        bandwidth_factor: float = 1.0,
        *,
        eval_points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Log of the weighted KDE of Section 4.3, and its bandwidth."""
        distances = np.asarray(distances, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if eval_points is None:
            eval_points = distances
        eval_points = np.asarray(eval_points, dtype=np.float64)

        h = _silverman_bandwidth(distances, weights) * bandwidth_factor

        return _log_kernel_density(eval_points, distances, weights, h), h

    def estimate_gmm(
        self,
        distances: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        eval_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Algorithm 5.3.1: GMM-EM for relevant-document density f_R.

        Two-component GMM where the background component (G) parameters
        are fixed (Remark 5.3.2). Only the relevant component (R)
        parameters (mu_R, sigma_R, pi_R) are updated during EM.

        Parameters
        ----------
        distances : array
            Distance values.
        weights : array or None
            Optional per-sample weights for informed initialization.
            When provided, mu_R is initialized to the weighted mean.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence threshold on log-likelihood change.

        Returns
        -------
        array
            Density values f_R at each distance.
        """
        log_density = self._estimate_log_gmm(
            distances, weights, max_iter=max_iter, tol=tol, eval_points=eval_points,
        )
        return np.maximum(np.exp(log_density), _EPSILON)

    def _estimate_log_gmm(
        self,
        distances: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        eval_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Log of the relevant component fitted by Algorithm 5.3.1."""
        distances = np.asarray(distances, dtype=np.float64)
        if eval_points is None:
            eval_points = distances
        eval_points = np.asarray(eval_points, dtype=np.float64)
        n = len(distances)

        # Initialize R component
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            w_sum = float(np.sum(weights))
            if w_sum > _EPSILON:
                mu_R = float(np.sum(weights * distances) / w_sum)
                sigma_R = float(
                    np.sqrt(
                        np.sum(weights * (distances - mu_R) ** 2) / w_sum
                    )
                )
                pi_R = float(np.clip(w_sum / n, 0.1, 0.9))
            else:
                mu_R = float(np.mean(distances))
                sigma_R = float(np.std(distances))
                pi_R = 0.5
        else:
            mu_R = self.mu_G - 0.5 * self.sigma_G
            sigma_R = self.sigma_G * 0.5
            pi_R = 0.3

        if sigma_R < _EPSILON:
            sigma_R = self.sigma_G * 0.5

        prev_ll = -np.inf

        for _ in range(max_iter):
            # E-step: responsibilities
            f_R_vals = pi_R * np.asarray(
                _gaussian_pdf(distances, mu_R, sigma_R)
            )
            f_G_vals = (1.0 - pi_R) * np.asarray(
                _gaussian_pdf(distances, self.mu_G, self.sigma_G)
            )

            total = np.maximum(f_R_vals + f_G_vals, _EPSILON)
            gamma = f_R_vals / total

            # Log-likelihood for convergence check
            ll = float(np.sum(np.log(total)))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # M-step: update R only (G is fixed per Remark 5.3.2)
            gamma_sum = float(np.sum(gamma))
            if gamma_sum < _EPSILON:
                break

            mu_R = float(np.sum(gamma * distances) / gamma_sum)
            sigma_R = float(
                np.sqrt(
                    np.sum(gamma * (distances - mu_R) ** 2) / gamma_sum
                )
            )
            if sigma_R < _EPSILON:
                sigma_R = self.sigma_G * 0.1

            pi_R = float(np.clip(gamma_sum / n, 0.01, 0.99))

        return _log_gaussian_pdf(eval_points, mu_R, sigma_R)

    @staticmethod
    def _signal_mass(weights: np.ndarray | None) -> float:
        if weights is None:
            return 0.0
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size == 0:
            return 0.0
        return float(np.sum(np.maximum(weights, 0.0)))

    def _estimate_relevant_density(
        self,
        eval_points: np.ndarray,
        sample_distances: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        method: str = "auto",
        bandwidth_factor: float = 1.0,
        density_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        log_density, _ = self._estimate_log_relevant_density(
            eval_points,
            sample_distances,
            weights=weights,
            method=method,
            bandwidth_factor=bandwidth_factor,
            density_prior=density_prior,
        )
        return np.maximum(np.exp(log_density), _EPSILON)

    def _estimate_log_relevant_density(
        self,
        eval_points: np.ndarray,
        sample_distances: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        method: str = "auto",
        bandwidth_factor: float = 1.0,
        density_prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float | None]:
        """Log of the relevant-document density at the evaluation points.

        Returns the log density and the kernel bandwidth, or None for the
        bandwidth when the estimate is the parametric mixture component.
        """
        eval_points = np.asarray(eval_points, dtype=np.float64)
        sample_distances = np.asarray(sample_distances, dtype=np.float64)
        if len(sample_distances) == 0:
            return np.full(eval_points.shape, math.log(_EPSILON)), None

        K = len(sample_distances)
        weight_mass = self._signal_mass(weights)
        density_mass = self._signal_mass(density_prior)

        if method == "auto":
            # External guidance must not be replaced by a gap in the same
            # distances being calibrated (Sections 4.3-4.5).
            if weights is not None and weight_mass > _EPSILON:
                return self._estimate_log_kde(
                    sample_distances,
                    weights,
                    bandwidth_factor,
                    eval_points=eval_points,
                )

            if density_prior is not None and density_mass > _EPSILON:
                return self._estimate_log_gmm(
                    sample_distances,
                    density_prior,
                    eval_points=eval_points,
                ), None

            gap_w = self._gap_weights(sample_distances)
            has_gap = gap_w is not None

            if has_gap:
                if K >= 50:
                    return self._estimate_log_kde(
                        sample_distances,
                        gap_w,
                        bandwidth_factor,
                        eval_points=eval_points,
                    )
                return self._estimate_log_gmm(
                    sample_distances,
                    gap_w,
                    eval_points=eval_points,
                ), None

            fallback_w = self._distance_density_weights(sample_distances)
            return self._estimate_log_gmm(
                sample_distances,
                fallback_w,
                eval_points=eval_points,
            ), None

        if method == "kde":
            if weights is not None and weight_mass > _EPSILON:
                effective_w = np.asarray(weights, dtype=np.float64)
            elif density_prior is not None and density_mass > _EPSILON:
                effective_w = np.asarray(density_prior, dtype=np.float64)
            else:
                gap_w = self._gap_weights(sample_distances)
                if gap_w is not None:
                    effective_w = gap_w
                else:
                    effective_w = self._distance_density_weights(sample_distances)
            return self._estimate_log_kde(
                sample_distances,
                effective_w,
                bandwidth_factor,
                eval_points=eval_points,
            )

        if method == "gmm":
            if weights is not None and weight_mass > _EPSILON:
                effective_w = np.asarray(weights, dtype=np.float64)
            elif density_prior is not None and density_mass > _EPSILON:
                effective_w = np.asarray(density_prior, dtype=np.float64)
            else:
                effective_w = None
            return self._estimate_log_gmm(
                sample_distances,
                effective_w,
                eval_points=eval_points,
            ), None

        raise ValueError(
            f"method must be 'auto', 'kde', or 'gmm', got {method!r}"
        )

    def log_density_ratio(
        self,
        distances: np.ndarray | float,
        f_R_values: np.ndarray | float,
        *,
        background_distances: np.ndarray | None = None,
        bandwidth: float | None = None,
    ) -> np.ndarray | float:
        """Log density ratio before posterior conversion.

        log(f_R(d) / f_G(d))

        This is likelihood-ratio evidence only for a non-relevant
        background. Against a marginal background it is pointwise
        mutual information. Use ``evidence`` or ``evidence_with_sample``
        for prior-free evidence consistent with the selected posterior.

        The background term is evaluated in log space, so the ratio
        keeps growing as a distance moves into the tail of the
        background instead of stalling where the density underflows.

        Parameters
        ----------
        distances : float or array
            Distance values.
        f_R_values : float or array
            Relevant-document density values at the distances.

        Returns
        -------
        float or array
            Log density ratios.
        """
        scalar = np.ndim(distances) == 0
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        f_R_values = np.asarray(f_R_values, dtype=np.float64)

        if np.any(f_R_values < 0.0) or not np.all(np.isfinite(f_R_values)):
            raise ValueError("f_R_values must be finite and non-negative")
        with np.errstate(divide="ignore"):
            log_f_R = np.log(f_R_values)
        result = log_f_R - self._log_background_density(
            distances, background_distances, bandwidth,
        )
        return float(result[0]) if scalar and result.size == 1 else result

    def _log_background_density(
        self,
        eval_points: np.ndarray,
        background_distances: np.ndarray | None,
        bandwidth: float | None,
    ) -> np.ndarray:
        """Log of the background density f_G at the evaluation points.

        Without a stored or supplied sample this is the explicit Gaussian
        shape model at construction. With one it is a KDE of the sample; the
        kernel bandwidth of the relevant density is reused when there is
        one, so that numerator and denominator of the density ratio
        smooth the same way.
        """
        if background_distances is None:
            background_distances = self._background_distances
        if background_distances is None:
            return _log_gaussian_pdf(eval_points, self.mu_G, self.sigma_G)

        background = np.asarray(background_distances, dtype=np.float64).ravel()
        if background.size == 0 or not np.all(np.isfinite(background)):
            raise ValueError("background_distances must be non-empty and finite")
        h = bandwidth if bandwidth is not None else _silverman_bandwidth(background)
        return _log_binned_kernel_density(eval_points, background, h)

    def _posterior_log_odds(
        self,
        log_ratio: np.ndarray,
        posterior: str,
    ) -> np.ndarray:
        """Posterior log-odds from a recall-adjusted density ratio.

        ``"approximate"`` is Eq. 4.1, ``log_ratio + logit(P_base)``,
        which treats the background as the non-relevant density.

        ``"exact"`` is Bayes' rule when the background is the marginal
        density of the scored population, relevant documents included:
        ``P = P_base * f_R / f_G``.  Estimated densities can make that
        product exceed one; it is clipped just below one.
        """
        if posterior == "approximate":
            return log_ratio + self._logit_base_rate

        prevalence = self.base_rate if self.base_rate is not None else 0.5
        log_p = np.minimum(log_ratio + math.log(prevalence), _LOG_MAX_POSTERIOR)
        # logit(P) = log P - log(1 - P)
        return log_p - np.log(-np.expm1(log_p))

    def _calibrated_log_odds(
        self,
        eval_points: np.ndarray,
        sample_distances: np.ndarray,
        *,
        weights: np.ndarray | None,
        method: str,
        bandwidth_factor: float,
        density_prior: np.ndarray | None,
        posterior: str | None,
        monotone: bool,
        duplicate_threshold: float | None,
        background_distances: np.ndarray | None,
        weight_temperature: float | None,
        background_kind: str | None,
        relevance_recall: float,
    ) -> np.ndarray:
        """Shared implementation of ``calibrate`` and ``calibrate_with_sample``."""
        if eval_points.ndim != 1 or sample_distances.ndim != 1:
            raise ValueError("evaluation and sample distances must be one-dimensional")
        kind = background_kind
        if kind is None:
            kind = "marginal" if background_distances is not None else self.background_kind
        if kind not in _VALID_BACKGROUND_KINDS:
            raise ValueError("background_kind must be 'marginal' or 'nonrelevant'")
        if posterior is None:
            posterior = "exact" if kind == "marginal" else "approximate"
        if posterior not in _VALID_POSTERIORS:
            raise ValueError(
                f"posterior must be one of {_VALID_POSTERIORS}, got {posterior!r}"
            )
        if posterior == "exact" and kind != "marginal":
            raise ValueError("posterior='exact' requires a marginal background")
        if not math.isfinite(relevance_recall) or not (0.0 < relevance_recall <= 1.0):
            raise ValueError("relevance_recall must be finite and in (0, 1]")
        if not np.all(np.isfinite(eval_points)) or not np.all(np.isfinite(sample_distances)):
            raise ValueError("evaluation and sample distances must be finite")
        if not math.isfinite(bandwidth_factor) or bandwidth_factor <= 0.0:
            raise ValueError("bandwidth_factor must be positive and finite")
        for name, values in (("weights", weights), ("density_prior", density_prior)):
            if values is not None:
                array = np.asarray(values, dtype=np.float64)
                if array.shape != sample_distances.shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must match sample distances and be finite")
                if np.any(array < 0.0):
                    raise ValueError(f"{name} must be non-negative")

        if weight_temperature is not None:
            if not math.isfinite(weight_temperature) or weight_temperature <= 0.0:
                raise ValueError("weight_temperature must be positive and finite")
            if weights is None:
                raise ValueError("weight_temperature requires external weights")

        if duplicate_threshold is not None:
            keep = sample_distances > duplicate_threshold
            if not np.all(keep):
                sample_distances = sample_distances[keep]
                if weights is not None:
                    weights = np.asarray(weights, dtype=np.float64)[keep]
                if density_prior is not None:
                    density_prior = np.asarray(density_prior, dtype=np.float64)[keep]

        if weight_temperature is not None and len(sample_distances) > 0:
            weights = self._sharpen_weights(weights, temperature=weight_temperature)

        log_f_R, bandwidth = self._estimate_log_relevant_density(
            eval_points,
            sample_distances,
            weights=weights,
            method=method,
            bandwidth_factor=bandwidth_factor,
            density_prior=density_prior,
        )
        log_ratio = log_f_R - self._log_background_density(
            eval_points, background_distances, bandwidth,
        ) + math.log(relevance_recall)

        if monotone and len(eval_points) > 1:
            # Project the evidence onto non-increasing functions of the
            # distance (Remark 4.5): closer must never be weaker evidence.
            order = np.argsort(eval_points, kind="stable")
            projected = np.empty_like(log_ratio)
            projected[order] = isotonic_regression(
                log_ratio[order], increasing=False,
            )
            log_ratio = projected

        return self._posterior_log_odds(log_ratio, posterior)

    def calibrate(
        self,
        distances: np.ndarray | float,
        *,
        weights: np.ndarray | None = None,
        method: str = "auto",
        bandwidth_factor: float = 1.0,
        density_prior: np.ndarray | None = None,
        posterior: str | None = None,
        monotone: bool = False,
        duplicate_threshold: float | None = None,
        background_distances: np.ndarray | None = None,
        return_log_odds: bool = False,
        weight_temperature: float | None = None,
        background_kind: str | None = None,
        relevance_recall: float = 1.0,
    ) -> np.ndarray | float:
        """Full calibration pipeline (Theorem 3.1.1).

        The background's declared role selects the posterior formula.

        Auto-routing logic:
        - External weights: KDE with the supplied weights
        - Density prior: GMM initialized from the supplied prior
        - Gap detected, K >= 50: KDE + gap weights + bandwidth_factor
        - Gap detected, K < 50:  GMM + gap-informed initialization
        - Smooth, nothing:        GMM + distance-based density fallback

        Parameters
        ----------
        distances : float or array
            Vector distance/similarity values.
        weights : array or None
            Per-document weights (e.g., BM25 probabilities).
        method : str
            Estimation method: "auto", "kde", or "gmm".
        bandwidth_factor : float
            Multiplicative factor for KDE bandwidth.
        density_prior : array or None
            External density prior weights (e.g., from ivf_density_prior).
        posterior : str or None
            None chooses the formula from the declared background role:
            a marginal background uses ``P_base * f_R / f_G`` (``exact``),
            a non-relevant background adds the prior log-odds to the
            log density ratio (``approximate``). Explicitly choosing
            approximate with a marginal background is the small-posterior
            approximation and remains available as an ablation.
        background_kind : str or None
            Override the background role. A supplied background sample
            defaults to marginal; otherwise use the fitted model's role.
        relevance_recall : float
            Fraction of corpus relevance in the retrieved density sample's
            range, in (0, 1]. Adds ``log(relevance_recall)`` to the ratio
            before posterior conversion. One is a convention, not a recall
            estimate. Counts used after fusion must refer to the constrained
            population, rather than importing a corpus count into a pool.
        monotone : bool
            Project the log density ratio onto non-increasing functions
            of the distance by isotonic regression, so that the
            calibration preserves the ranking of the distance.  A ratio
            of estimated densities need not be monotone on its own.
        duplicate_threshold : float or None
            Distances at or below this value are treated as copies of
            the query and left out of the density estimate.  A copy sits
            at distance zero and otherwise captures gap detection: the
            largest gap is then the one that separates the copy from
            every genuine candidate.  None (default) keeps every
            distance.
        background_distances : array or None
            Sample of distances from the scored population.  When given,
            the background density is a KDE of this sample instead of
            the Gaussian fitted at construction.
        return_log_odds : bool
            Return posterior log-odds instead of probabilities.  A
            probability is clamped to [epsilon, 1 - epsilon]; rank and
            fuse on the log-odds.
        weight_temperature : float or None
            Explicit softmax sharpening temperature for external weights.
            None preserves the supplied weights. Sharpening can reduce
            support and worsen calibration (Section 6.5).

        Returns
        -------
        float or array
            Calibrated probabilities in (0, 1), or log-odds.
        """
        scalar = np.ndim(distances) == 0
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))

        log_odds = self._calibrated_log_odds(
            distances,
            distances,
            weights=weights,
            method=method,
            bandwidth_factor=bandwidth_factor,
            density_prior=density_prior,
            posterior=posterior,
            monotone=monotone,
            duplicate_threshold=duplicate_threshold,
            background_distances=background_distances,
            weight_temperature=weight_temperature,
            background_kind=background_kind,
            relevance_recall=relevance_recall,
        )
        if return_log_odds:
            return float(log_odds[0]) if scalar else log_odds
        result = _clamp_probability(sigmoid(log_odds))

        return float(result[0]) if scalar else result

    def calibrate_with_sample(
        self,
        eval_distances: np.ndarray | float,
        sample_distances: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        method: str = "auto",
        bandwidth_factor: float = 1.0,
        density_prior: np.ndarray | None = None,
        posterior: str | None = None,
        monotone: bool = False,
        duplicate_threshold: float | None = None,
        background_distances: np.ndarray | None = None,
        return_log_odds: bool = False,
        weight_temperature: float | None = None,
        background_kind: str | None = None,
        relevance_recall: float = 1.0,
    ) -> np.ndarray | float:
        """Calibrate eval distances using a separate local sample.

        This is the index-aware path used when the local neighborhood
        sample comes from an ANN structure (e.g. IVF probed cells) while
        probabilities must be produced for an arbitrary evaluation set.

        ``posterior``, ``monotone``, ``duplicate_threshold``,
        ``background_distances`` and ``return_log_odds`` behave as in
        :meth:`calibrate`; ``duplicate_threshold`` applies to the
        sample, not to the evaluation set.
        """
        scalar = np.ndim(eval_distances) == 0
        eval_arr = np.atleast_1d(np.asarray(eval_distances, dtype=np.float64))
        sample_arr = np.atleast_1d(np.asarray(sample_distances, dtype=np.float64))

        log_odds = self._calibrated_log_odds(
            eval_arr,
            sample_arr,
            weights=weights,
            method=method,
            bandwidth_factor=bandwidth_factor,
            density_prior=density_prior,
            posterior=posterior,
            monotone=monotone,
            duplicate_threshold=duplicate_threshold,
            background_distances=background_distances,
            weight_temperature=weight_temperature,
            background_kind=background_kind,
            relevance_recall=relevance_recall,
        )
        if return_log_odds:
            return float(log_odds[0]) if scalar else log_odds
        result = _clamp_probability(sigmoid(log_odds))
        return float(result[0]) if scalar else result

    def evidence(self, distances: np.ndarray | float, **kwargs) -> np.ndarray | float:
        """Prior-free evidence of the selected posterior and background.

        A marginal density ratio is pointwise mutual information, not
        itself a likelihood ratio. This method converts through the
        appropriate posterior in log space and removes its prior.
        """
        if "return_log_odds" in kwargs:
            raise ValueError("evidence already returns prior-free log-odds")
        result = np.asarray(self.calibrate(distances, return_log_odds=True, **kwargs))
        result = result - self._logit_base_rate
        return float(result) if result.ndim == 0 else result

    def evidence_with_sample(
        self, eval_distances: np.ndarray | float, sample_distances: np.ndarray, **kwargs,
    ) -> np.ndarray | float:
        """Prior-free evidence using a separate retrieved density sample."""
        if "return_log_odds" in kwargs:
            raise ValueError("evidence already returns prior-free log-odds")
        result = np.asarray(self.calibrate_with_sample(
            eval_distances, sample_distances, return_log_odds=True, **kwargs,
        )) - self._logit_base_rate
        return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def ivf_density_prior(
    cell_population: int | np.ndarray,
    avg_population: float,
    *,
    gamma: float = 1.0,
) -> float | np.ndarray:
    """IVF cell density prior (Strategy 4.6.2).

    Estimates informativeness from the population of the IVF cell:

        prior = sigmoid(gamma * (avg_population / cell_population - 1))

    Sparse cells (population < average) receive higher prior weight,
    reflecting that vector proximity is more discriminative in sparse
    regions -- analogous to IDF in lexical retrieval.

    Parameters
    ----------
    cell_population : int or array
        Number of documents in the IVF cell(s).
    avg_population : float
        Average population across all cells.
    gamma : float
        Scaling factor controlling sensitivity.

    Returns
    -------
    float or array
        Density prior weight(s) in (0, 1).
    """
    cell_population = np.asarray(cell_population, dtype=np.float64)
    safe_pop = np.maximum(cell_population, _EPSILON)
    ratio = avg_population / safe_pop - 1.0
    result = sigmoid(gamma * ratio)
    return float(result) if np.ndim(result) == 0 else result


def knn_density_prior(
    kth_distance: float | np.ndarray,
    global_median_kth: float,
    *,
    gamma: float = 1.0,
) -> float | np.ndarray:
    """HNSW k-th neighbor distance prior.

        prior = sigmoid(gamma * (kth_distance / global_median_kth - 1))

    A larger k-th-neighbor distance denotes a sparser neighborhood and
    receives a prior above 1/2, as in Section 4.7 of "From Scores to
    Evidence". This is a geometric weighting heuristic, not an identified
    relevance probability.

    Parameters
    ----------
    kth_distance : float or array
        Distance to the k-th nearest neighbor.
    global_median_kth : float
        Median k-th neighbor distance across the corpus.
    gamma : float
        Scaling factor controlling sensitivity.

    Returns
    -------
    float or array
        Density prior weight(s) in (0, 1).
    """
    kth_distance = np.asarray(kth_distance, dtype=np.float64)
    if not np.all(np.isfinite(kth_distance)) or np.any(kth_distance < 0.0):
        raise ValueError("kth_distance must be finite and non-negative")
    if not math.isfinite(global_median_kth) or global_median_kth <= 0.0:
        raise ValueError("global_median_kth must be positive and finite")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    ratio = kth_distance / global_median_kth - 1.0
    result = sigmoid(gamma * ratio)
    return float(result) if np.ndim(result) == 0 else result
