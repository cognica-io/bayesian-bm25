#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""LearnableLogOddsWeights with explicit source and target priors."""

from __future__ import annotations

import math

import numpy as np

from bayesian_bm25.fusion import _resolve_alpha, _to_log_odds, _validated_input_priors
from bayesian_bm25.probability import sigmoid


class LearnableLogOddsWeights:
    """Learnable per-signal reliability weights for log-odds conjunction (Remark 5.3.2).

    Learns weights that map from the Naive Bayes uniform initialization
    (w_i = 1/n) to per-signal reliability weights, completing the
    correspondence to a fully parameterized single-layer network in
    log-odds space: logit -> weighted sum -> sigmoid.

    The gradient dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
    is Hebbian: the product of pre-synaptic activity (signal deviation
    from weighted mean) and post-synaptic error (prediction minus label).

    Source posterior logits can be converted to evidence by supplying
    ``input_prior_log_odds`` and an explicit target ``base_rate``. Without
    source priors the supplied logits are pooled as given. Score
    normalization does not establish probability calibration.

    Parameters
    ----------
    n_signals : int
        Number of probability signals to combine (>= 1).
    alpha : float or str
        Confidence scaling exponent (fixed, not learned).  Default 0.0
        means no count-based scaling.  ``"auto"`` resolves to 0.5
        (sqrt(n) scaling law, Theorem 4.2.1).  Weights (per-signal
        reliability) and alpha (confidence scaling) are orthogonal
        (Paper 2, Section 4.2).
    """

    def __init__(
        self,
        n_signals: int,
        alpha: float | str = 0.0,
        base_rate: float | None = None,
        input_prior_log_odds: np.ndarray | float | None = None,
    ) -> None:
        if n_signals < 1:
            raise ValueError(
                f"n_signals must be >= 1, got {n_signals}"
            )
        if base_rate is not None and not (0.0 < base_rate < 1.0):
            raise ValueError(
                f"base_rate must be in (0, 1), got {base_rate}"
            )
        self._n_signals = n_signals
        self._alpha = _resolve_alpha(alpha, default=0.0)
        self._base_rate = base_rate
        self._input_prior_log_odds = _validated_input_priors(
            input_prior_log_odds, n_signals, base_rate,
        )
        self._logit_base_rate: float | None = (
            math.log(base_rate) - math.log1p(-base_rate) if base_rate is not None else None
        )

        # Softmax parameterization: zeros -> uniform 1/n (Naive Bayes init)
        self._logits = np.zeros(n_signals, dtype=np.float64)

        # Online learning state
        self._n_updates: int = 0
        self._grad_logits_ema = np.zeros(n_signals, dtype=np.float64)

        # Polyak averaging in the simplex
        self._weights_avg = np.full(n_signals, 1.0 / n_signals, dtype=np.float64)

    @property
    def n_signals(self) -> int:
        """Number of probability signals."""
        return self._n_signals

    @property
    def alpha(self) -> float:
        """Confidence scaling exponent (fixed)."""
        return self._alpha

    @property
    def base_rate(self) -> float | None:
        """Corpus-level base rate of relevance, or None if not set."""
        return self._base_rate

    @property
    def weights(self) -> np.ndarray:
        """Current weights: softmax of internal logits."""
        return self._softmax(self._logits)

    @property
    def averaged_weights(self) -> np.ndarray:
        """Polyak-averaged weights for stable inference."""
        return self._weights_avg.copy()

    def __call__(
        self,
        probs: np.ndarray,
        use_averaged: bool = False,
        *,
        from_log_odds: bool = False,
        return_log_odds: bool = False,
    ) -> np.ndarray | float:
        """Combine probability signals via weighted log-odds conjunction.

        Parameters
        ----------
        probs : array of shape (..., n_signals)
            Probability values to combine.
        use_averaged : bool
            If True, use Polyak-averaged weights instead of raw weights.
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities
            and no clamping is applied.
        return_log_odds : bool
            If True, return the fused log-odds instead of a probability.
            Rank and prune on log-odds; a probability saturates.

        Returns
        -------
        Combined probability (or log-odds) after weighted log-odds
        conjunction.
        """
        w = self._weights_avg if use_averaged else self.weights

        n = self._n_signals
        scale = n ** self._alpha
        x = _to_log_odds(probs, from_log_odds, self._input_prior_log_odds)

        l_weighted = scale * np.sum(w * x, axis=-1)
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        if return_log_odds:
            return float(l_weighted) if np.ndim(l_weighted) == 0 else l_weighted
        result = sigmoid(l_weighted)
        return float(result) if np.ndim(result) == 0 else result

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        from_log_odds: bool = False,
    ) -> None:
        """Batch gradient descent on BCE loss to learn weights.

        The gradient for logit z_j is (averaged over samples):
          dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)

        where x_i = logit(P_i), x_bar_w = sum(w_i * x_i), and
        p = sigmoid(n^alpha * x_bar_w).

        Parameters
        ----------
        probs : array of shape (m, n_signals)
            Training probability signals.
        labels : array of shape (m,)
            Binary relevance labels (0 or 1).
        learning_rate : float
            Step size for gradient descent.
        max_iterations : int
            Maximum number of gradient descent iterations.
        tolerance : float
            Convergence threshold on maximum absolute logit change.
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if probs.shape[-1] != self._n_signals:
            raise ValueError(
                f"probs last dimension {probs.shape[-1]} != n_signals {self._n_signals}"
            )

        n = self._n_signals
        scale = n ** self._alpha

        # Log-odds of input signals: shape (m, n)
        x = _to_log_odds(probs, from_log_odds, self._input_prior_log_odds)

        for _ in range(max_iterations):
            w = self._softmax(self._logits)

            # Weighted mean log-odds per sample: shape (m,)
            x_bar_w = np.sum(w * x, axis=-1)

            # Predicted probability: shape (m,)
            l_weighted = scale * x_bar_w
            if self._logit_base_rate is not None:
                l_weighted = l_weighted + self._logit_base_rate
            p = sigmoid(l_weighted)
            p = np.atleast_1d(np.asarray(p, dtype=np.float64))

            # Error: shape (m,)
            error = p - labels

            # Gradient for each logit z_j, averaged over samples:
            # dL/dz_j = scale * (p - y) * w_j * (x_j - x_bar_w)
            # Shape: (m, n) -> mean over m -> (n,)
            grad_logits = np.mean(
                scale * error[:, np.newaxis] * w[np.newaxis, :] * (x - x_bar_w[:, np.newaxis]),
                axis=0,
            )

            self._logits -= learning_rate * grad_logits

            if np.max(np.abs(learning_rate * grad_logits)) < tolerance:
                break

        # Reset online state after batch fit (consistent with BayesianProbabilityTransform.fit())
        self._n_updates = 0
        self._grad_logits_ema = np.zeros(n, dtype=np.float64)
        self._weights_avg = self._softmax(self._logits).copy()

    def update(
        self,
        probs: np.ndarray | float,
        label: np.ndarray | float,
        *,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        decay_tau: float = 1000.0,
        max_grad_norm: float = 1.0,
        avg_decay: float = 0.995,
        from_log_odds: bool = False,
    ) -> None:
        """Online SGD update from a single observation or mini-batch.

        Follows the same patterns as ``BayesianProbabilityTransform.update()``:
        EMA gradient smoothing with bias correction, L2 gradient clipping,
        learning rate decay, and Polyak averaging of weights in the simplex.

        Parameters
        ----------
        probs : array of shape (n_signals,) or (m, n_signals)
            Probability signal(s) for the observed document(s).
        label : float or array of shape (m,)
            Binary relevance label(s).
        learning_rate : float
            Base step size, decayed as lr / (1 + t / tau).
        momentum : float
            EMA decay factor for gradient smoothing.
        decay_tau : float
            Time constant for learning rate decay.
        max_grad_norm : float
            Maximum L2 norm for gradient clipping.
        avg_decay : float
            Decay factor for Polyak weight averaging.
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities.
        """
        probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))
        label = np.atleast_1d(np.asarray(label, dtype=np.float64))

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if probs.shape[-1] != self._n_signals:
            raise ValueError(
                f"probs last dimension {probs.shape[-1]} != n_signals {self._n_signals}"
            )

        n = self._n_signals
        scale = n ** self._alpha
        w = self._softmax(self._logits)

        # Log-odds of input signals: shape (m, n)
        x = _to_log_odds(probs, from_log_odds, self._input_prior_log_odds)

        # Weighted mean log-odds per sample: shape (m,)
        x_bar_w = np.sum(w * x, axis=-1)

        # Predicted probability: shape (m,)
        l_weighted = scale * x_bar_w
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        p = sigmoid(l_weighted)
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))

        # Error: shape (m,)
        error = p - label

        # Gradient for each logit, averaged over mini-batch
        grad_logits = np.mean(
            scale * error[:, np.newaxis] * w[np.newaxis, :] * (x - x_bar_w[:, np.newaxis]),
            axis=0,
        )

        # EMA smoothing of gradients
        self._grad_logits_ema = (
            momentum * self._grad_logits_ema + (1.0 - momentum) * grad_logits
        )

        # Bias correction for early updates
        self._n_updates += 1
        correction = 1.0 - momentum ** self._n_updates
        corrected_grad = self._grad_logits_ema / correction

        # L2 gradient clipping
        grad_norm = float(np.sqrt(np.sum(corrected_grad ** 2)))
        if grad_norm > max_grad_norm:
            corrected_grad = corrected_grad * (max_grad_norm / grad_norm)

        # Learning rate decay: lr / (1 + t / tau)
        effective_lr = learning_rate / (1.0 + self._n_updates / decay_tau)

        self._logits -= effective_lr * corrected_grad

        # Polyak averaging of weights in the simplex
        raw_weights = self._softmax(self._logits)
        self._weights_avg = avg_decay * self._weights_avg + (1.0 - avg_decay) * raw_weights

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax: shift by max to prevent overflow."""
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
