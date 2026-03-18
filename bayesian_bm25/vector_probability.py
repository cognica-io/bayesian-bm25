#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Vector similarity score calibration via likelihood ratio framework.

Implements the Bayesian calibration transform from Paper 3 ("Index-Aware
Bayesian Calibration of Vector Similarity Scores") that replaces naive
(1+cos)/2 conversion with a likelihood ratio approach:

    logit P(R|d) = log(f_R(d) / f_G(d)) + logit(P_base)

The core class ``VectorProbabilityTransform`` is fully parameterized
(index-free). Index-specific helpers ``ivf_density_prior`` and
``knn_density_prior`` are standalone utility functions that users call
externally.

Paper ref: Paper 3, Sections 3-5
"""

from __future__ import annotations

import math

import numpy as np

from bayesian_bm25.probability import _EPSILON, _clamp_probability, logit, sigmoid

# ---------------------------------------------------------------------------
# Private helpers (module-level)
# ---------------------------------------------------------------------------


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
    """Weighted Silverman bandwidth (Definition 4.4.1).

    h = 1.06 * sigma_w * K_eff^(-1/5)

    where sigma_w is the weighted standard deviation and K_eff is the
    effective sample size: K_eff = (sum(w))^2 / sum(w^2).
    """
    distances = np.asarray(distances, dtype=np.float64)

    weights = np.ones_like(distances) if weights is None else np.asarray(weights, dtype=np.float64)

    w_sum = float(np.sum(weights))
    w_sq_sum = float(np.sum(weights * weights))

    if w_sum < _EPSILON or w_sq_sum < _EPSILON:
        return _EPSILON

    k_eff = (w_sum * w_sum) / w_sq_sum

    w_mean = float(np.sum(weights * distances) / w_sum)
    w_var = float(np.sum(weights * (distances - w_mean) ** 2) / w_sum)
    sigma_w = math.sqrt(max(w_var, 0.0))

    if sigma_w < _EPSILON:
        return _EPSILON

    h = 1.06 * sigma_w * k_eff ** (-0.2)
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
    """Calibrates vector similarity distances into probabilities.

    Uses a likelihood ratio framework (Theorem 3.1.1):

        P(R|d) = sigmoid(log(f_R(d) / f_G(d)) + logit(P_base))

    where f_R is the relevant-document density and f_G is the background
    (corpus) Gaussian density. The class is fully parameterized with no
    index coupling.

    Parameters
    ----------
    mu_G : float
        Mean of the background Gaussian density.
    sigma_G : float
        Standard deviation of the background Gaussian density.
    base_rate : float or None
        Corpus-level base rate of relevance P_base, in (0, 1).
        None defaults to 0.5 (neutral logit = 0).
    """

    def __init__(
        self,
        mu_G: float,
        sigma_G: float,
        base_rate: float | None = None,
    ) -> None:
        if sigma_G <= 0.0:
            raise ValueError(f"sigma_G must be positive, got {sigma_G}")
        if base_rate is not None and not (0.0 < base_rate < 1.0):
            raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")

        self.mu_G = float(mu_G)
        self.sigma_G = float(sigma_G)
        self.base_rate = base_rate
        self._logit_base_rate = (
            float(logit(base_rate)) if base_rate is not None else 0.0
        )

    @classmethod
    def fit_background(
        cls,
        distances: np.ndarray,
        *,
        base_rate: float | None = None,
    ) -> VectorProbabilityTransform:
        """Estimate background parameters from a corpus sample.

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
        distances = np.asarray(distances, dtype=np.float64)
        mu_G = float(np.mean(distances))
        sigma_G = float(np.std(distances))
        if sigma_G < _EPSILON:
            sigma_G = _EPSILON
        return cls(mu_G=mu_G, sigma_G=sigma_G, base_rate=base_rate)

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
        bandwidth_factor: float = 2.0,
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
        distances = np.asarray(distances, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        h = _silverman_bandwidth(distances, weights) * bandwidth_factor

        return _kernel_density(distances, distances, weights, h)

    def estimate_gmm(
        self,
        distances: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
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
        distances = np.asarray(distances, dtype=np.float64)
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

        f_R_result = np.asarray(_gaussian_pdf(distances, mu_R, sigma_R))
        return np.maximum(f_R_result, _EPSILON)

    def log_density_ratio(
        self,
        distances: np.ndarray | float,
        f_R_values: np.ndarray | float,
    ) -> np.ndarray | float:
        """Definition 3.2.1: log density ratio (vector evidence).

        log(f_R(d) / f_G(d))

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
        distances = np.asarray(distances, dtype=np.float64)
        f_R_values = np.asarray(f_R_values, dtype=np.float64)

        f_G_values = np.asarray(
            _gaussian_pdf(distances, self.mu_G, self.sigma_G)
        )

        f_R_safe = np.maximum(f_R_values, _EPSILON)
        f_G_safe = np.maximum(f_G_values, _EPSILON)

        result = np.log(f_R_safe / f_G_safe)
        return float(result) if result.ndim == 0 else result

    def calibrate(
        self,
        distances: np.ndarray | float,
        *,
        weights: np.ndarray | None = None,
        method: str = "auto",
        bandwidth_factor: float = 2.0,
        density_prior: np.ndarray | None = None,
    ) -> np.ndarray | float:
        """Full calibration pipeline (Theorem 3.1.1).

        P(R|d) = sigmoid(log(f_R(d) / f_G(d)) + logit(P_base))

        Auto-routing logic:
        - Gap detected, K >= 50: KDE + gap weights + bandwidth_factor
        - Gap detected, K < 50:  GMM + gap-informed initialization
        - Smooth, weights given:  KDE + sharpened weights
        - Smooth, density_prior:  GMM + density prior weights
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

        Returns
        -------
        float or array
            Calibrated probabilities in (0, 1).
        """
        scalar = np.ndim(distances) == 0
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        K = len(distances)

        if method == "auto":
            gap_w = self._gap_weights(distances)
            has_gap = gap_w is not None

            if has_gap:
                if K >= 50:
                    f_R = self.estimate_kde(
                        distances, gap_w, bandwidth_factor
                    )
                else:
                    f_R = self.estimate_gmm(distances, gap_w)
            elif weights is not None:
                sharpened = self._sharpen_weights(weights)
                f_R = self.estimate_kde(
                    distances, sharpened, bandwidth_factor
                )
            elif density_prior is not None:
                f_R = self.estimate_gmm(distances, density_prior)
            else:
                fallback_w = self._distance_density_weights(distances)
                f_R = self.estimate_gmm(distances, fallback_w)

        elif method == "kde":
            if weights is None:
                gap_w = self._gap_weights(distances)
                if gap_w is not None:
                    effective_w = gap_w
                else:
                    effective_w = self._distance_density_weights(distances)
            else:
                effective_w = np.asarray(weights, dtype=np.float64)
            f_R = self.estimate_kde(
                distances, effective_w, bandwidth_factor
            )

        elif method == "gmm":
            if weights is not None:
                effective_w = np.asarray(weights, dtype=np.float64)
            elif density_prior is not None:
                effective_w = np.asarray(density_prior, dtype=np.float64)
            else:
                effective_w = None
            f_R = self.estimate_gmm(distances, effective_w)

        else:
            raise ValueError(
                f"method must be 'auto', 'kde', or 'gmm', got {method!r}"
            )

        log_ratio = self.log_density_ratio(distances, f_R)
        log_odds = log_ratio + self._logit_base_rate
        result = _clamp_probability(sigmoid(log_odds))

        return float(result[0]) if scalar else result


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
    """HNSW k-th neighbor density proxy.

    Estimates informativeness from the k-th nearest neighbor distance:

        prior = sigmoid(gamma * (kth_distance / global_median_kth - 1))

    Sparse neighborhoods (large kth_distance) receive higher prior
    weight, reflecting that vector proximity is more discriminative
    in sparse regions -- analogous to IDF in lexical retrieval.

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
    ratio = kth_distance / max(global_median_kth, _EPSILON) - 1.0
    result = sigmoid(gamma * ratio)
    return float(result) if np.ndim(result) == 0 else result
