#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Time-weighted adaptation of the Bayesian score transform."""

from __future__ import annotations

import numpy as np

from bayesian_bm25.probability import BayesianProbabilityTransform, _clamp_probability, sigmoid


class TemporalBayesianTransform(BayesianProbabilityTransform):
    """BayesianProbabilityTransform with time-weighted parameter adaptation.

    Recent observations receive higher weight in the gradient computation,
    allowing the transform to adapt to non-stationary relevance patterns
    (Paper 1, Section 12.2 #3).

    Parameters
    ----------
    alpha : float
        Steepness of the sigmoid likelihood function.
    beta : float
        Midpoint (shift) of the sigmoid likelihood function.
    base_rate : float or None
        Corpus-level base rate of relevance, in (0, 1).
    decay_half_life : float
        Number of observations after which a sample's weight decays to
        50%.  Must be positive.  Very large values make this behave
        identically to the parent class.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        base_rate: float | None = None,
        decay_half_life: float = 1000.0,
    ) -> None:
        if decay_half_life <= 0.0:
            raise ValueError(
                f"decay_half_life must be positive, got {decay_half_life}"
            )
        super().__init__(alpha=alpha, beta=beta, base_rate=base_rate)
        self._decay_half_life = decay_half_life
        self._decay_rate = np.log(2.0) / decay_half_life
        self._timestamp: int = 0

    @property
    def decay_half_life(self) -> float:
        """Half-life for temporal weighting."""
        return self._decay_half_life

    @property
    def timestamp(self) -> int:
        """Current internal timestamp counter."""
        return self._timestamp

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        timestamps: np.ndarray | None = None,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        mode: str = "balanced",
        tfs: np.ndarray | None = None,
        doc_len_ratios: np.ndarray | None = None,
    ) -> None:
        """Learn alpha and beta with temporal weighting.

        When ``timestamps`` is provided, each sample's gradient is
        weighted by ``exp(-decay_rate * (max_ts - ts_i))``, giving
        recent observations more influence on the learned parameters.

        Parameters
        ----------
        scores : array of BM25 scores
        labels : array of binary relevance labels
        timestamps : array of timestamps or None
            Per-sample timestamps.  When None, all samples are weighted
            equally (equivalent to parent class behavior).
        learning_rate : step size for gradient updates
        max_iterations : maximum number of gradient descent steps
        tolerance : convergence threshold on parameter change
        mode : str
            Training mode: "balanced", "prior_aware", or "prior_free".
        tfs : array or None
            Term frequencies per sample.  Required when mode="prior_aware".
        doc_len_ratios : array or None
            Document length ratios per sample.  Required when mode="prior_aware".
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got {mode!r}"
            )
        if mode == "prior_aware" and (tfs is None or doc_len_ratios is None):
            raise ValueError(
                "tfs and doc_len_ratios are required when mode='prior_aware'"
            )

        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        # Compute temporal weights
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=np.float64)
            max_ts = float(np.max(ts))
            sample_weights = np.exp(-self._decay_rate * (max_ts - ts))
            # Normalize so weights sum to len(scores) for consistent gradient magnitude
            sample_weights = sample_weights * (len(scores) / np.sum(sample_weights))
        else:
            sample_weights = np.ones(len(scores), dtype=np.float64)

        priors: np.ndarray | None = None
        if mode == "prior_aware":
            tfs_arr = np.asarray(tfs, dtype=np.float64)
            dlr_arr = np.asarray(doc_len_ratios, dtype=np.float64)
            priors = np.asarray(
                self.composite_prior(tfs_arr, dlr_arr), dtype=np.float64
            )

        alpha = self.alpha
        beta = self.beta

        for _ in range(max_iterations):
            L = _clamp_probability(sigmoid(alpha * (scores - beta)))

            if mode == "prior_aware":
                p = priors
                denom = L * p + (1.0 - L) * (1.0 - p)
                predicted = _clamp_probability(L * p / denom)

                dP_dL = p * (1.0 - p) / (denom ** 2)
                dL_dalpha = L * (1.0 - L) * (scores - beta)
                dL_dbeta = -L * (1.0 - L) * alpha

                error = predicted - labels
                grad_alpha = np.mean(sample_weights * error * dP_dL * dL_dalpha)
                grad_beta = np.mean(sample_weights * error * dP_dL * dL_dbeta)
            else:
                predicted = L
                error = predicted - labels
                grad_alpha = np.mean(sample_weights * error * (scores - beta))
                grad_beta = np.mean(sample_weights * error * (-alpha))

            new_alpha = alpha - learning_rate * grad_alpha
            new_beta = beta - learning_rate * grad_beta

            if abs(new_alpha - alpha) < tolerance and abs(new_beta - beta) < tolerance:
                alpha = new_alpha
                beta = new_beta
                break

            alpha = new_alpha
            beta = new_beta

        self.alpha = alpha
        self.beta = beta
        self._training_mode = mode
        self._n_updates = 0
        self._grad_alpha_ema = 0.0
        self._grad_beta_ema = 0.0
        self._alpha_avg = alpha
        self._beta_avg = beta

    def update(
        self,
        score: float | np.ndarray,
        label: float | np.ndarray,
        *,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        decay_tau: float = 1000.0,
        max_grad_norm: float = 1.0,
        avg_decay: float = 0.995,
        mode: str | None = None,
        tf: float | np.ndarray | None = None,
        doc_len_ratio: float | np.ndarray | None = None,
    ) -> None:
        """Online update with incrementing internal timestamp.

        Increments the timestamp counter and reduces the Polyak avg_decay
        over time so newer observations dominate the averaged parameters.
        """
        self._timestamp += 1
        # Reduce avg_decay slightly as more observations arrive,
        # so the averaged parameters respond faster to recent data
        effective_avg_decay = avg_decay * (1.0 - 1.0 / (1.0 + self._timestamp))
        super().update(
            score, label,
            learning_rate=learning_rate,
            momentum=momentum,
            decay_tau=decay_tau,
            max_grad_norm=max_grad_norm,
            avg_decay=effective_avg_decay,
            mode=mode,
            tf=tf,
            doc_len_ratio=doc_len_ratio,
        )
