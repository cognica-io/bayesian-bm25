#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Probabilistic score combination functions.

Implements AND, OR, NOT, and log-odds conjunction for combining multiple
probability estimates.  The log-odds conjunction (from "From Bayesian
Inference to Neural Computation") resolves the shrinkage problem of
naive probabilistic AND by using geometric-mean log-odds with an
agreement bonus.

All functions operate on numpy arrays and are batch-friendly.
"""

from __future__ import annotations

import numpy as np

from bayesian_bm25.probability import _EPSILON, _clamp_probability, logit, sigmoid


def cosine_to_probability(
    score: np.ndarray | float,
) -> np.ndarray | float:
    """Convert cosine similarity to probability (Definition 7.1.2).

    Maps cosine similarity in [-1, 1] to probability in (0, 1) via
    P_vector = (1 + score) / 2, with epsilon clamping for numerical
    stability.

    Parameters
    ----------
    score : float or array
        Cosine similarity score(s) in [-1, 1].

    Returns
    -------
    Probability value(s) in (0, 1).
    """
    score = np.asarray(score, dtype=np.float64)
    result = _clamp_probability((1.0 + score) / 2.0)
    return float(result) if result.ndim == 0 else result


def prob_not(prob: np.ndarray | float) -> np.ndarray | float:
    """Probabilistic NOT via complement rule (Eq. 35).

    Computes P(NOT R) = 1 - P(R).  In log-odds space this corresponds
    to negation: logit(1 - p) = -logit(p), so NOT simply flips the
    sign of evidence.

    Parameters
    ----------
    prob : float or array
        Probability value(s) to negate.

    Returns
    -------
    Complement probability: 1 - p, clamped to (epsilon, 1 - epsilon).
    """
    prob = _clamp_probability(np.asarray(prob, dtype=np.float64))
    result = _clamp_probability(1.0 - prob)
    return float(result) if np.ndim(result) == 0 else result


def prob_and(probs: np.ndarray) -> np.ndarray | float:
    """Probabilistic AND via product rule in log-space (Eq. 33-34).

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.

    Returns
    -------
    Combined probability: exp(sum(ln(p_i))) along the last axis.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    result = np.exp(np.sum(np.log(probs), axis=-1))
    return float(result) if np.ndim(result) == 0 else result


def prob_or(probs: np.ndarray) -> np.ndarray | float:
    """Probabilistic OR via complement rule in log-space (Eq. 36-37).

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.

    Returns
    -------
    Combined probability: 1 - exp(sum(ln(1 - p_i))) along the last axis.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    result = 1.0 - np.exp(np.sum(np.log(1.0 - probs), axis=-1))
    return float(result) if np.ndim(result) == 0 else result


def log_odds_conjunction(
    probs: np.ndarray,
    alpha: float | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray | float:
    """Log-odds conjunction with multiplicative confidence scaling (Paper 2, Section 4).

    Resolves the shrinkage problem of naive probabilistic AND by:
      1. Computing the mean log-odds (Eq. 20)
      2. Multiplicative confidence scaling by n^alpha (Eq. 23)
      3. Converting back to probability via sigmoid (Eq. 26)

    When ``weights`` are provided, uses the Log-OP (Log-linear Opinion
    Pool) formulation from Paper 2, Theorem 8.3 / Remark 8.4:
    sigma(n^alpha * sum(w_i * logit(P_i))) where sum(w_i) = 1 and
    w_i >= 0.  Per-signal weights (Theorem 8.3) and confidence scaling
    by signal count (Section 4.2) are orthogonal and compose
    multiplicatively.

    The multiplicative formulation (rather than additive) preserves the
    sign of evidence (Theorem 4.2.2), preventing accidental inversion
    of irrelevance signals (Remark 4.2.4).  Working directly in log-odds
    space avoids the nonlinear residual introduced by geometric-mean
    aggregation in probability space (Remark 4.1.3).

    Parameters
    ----------
    probs : array of shape (..., n)
        Probability values to combine.  The last axis is reduced.
    alpha : float or None
        Confidence scaling exponent.  Higher values amplify the effect
        of multiple agreeing signals.  When None (default), uses 0.5
        in unweighted mode and 0.0 in weighted mode, preserving
        backward compatibility for both calling conventions.
    weights : array of shape (n,) or None
        Per-signal reliability weights for the Log-OP formulation.
        Must be non-negative and sum to 1.  When None (default),
        uses the unweighted mean log-odds with n^alpha scaling.

    Returns
    -------
    Combined probability after log-odds conjunction.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    n = probs.shape[-1]

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if abs(float(np.sum(weights)) - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1, got {float(np.sum(weights))}"
            )

        effective_alpha = 0.0 if alpha is None else alpha

        # Log-OP with confidence scaling:
        # sigma(n^alpha * sum(w_i * logit(P_i)))  (Theorem 8.3 + Section 4.2)
        l_weighted = (n ** effective_alpha) * np.sum(
            weights * logit(probs), axis=-1
        )
        result = sigmoid(l_weighted)
        return float(result) if np.ndim(result) == 0 else result

    effective_alpha = 0.5 if alpha is None else alpha

    # Step 1: mean log-odds (Eq. 20)
    l_bar = np.mean(logit(probs), axis=-1)

    # Step 2: multiplicative confidence scaling (Eq. 23)
    l_adjusted = l_bar * (n ** effective_alpha)

    # Step 3: back to probability (Eq. 26)
    result = sigmoid(l_adjusted)
    return float(result) if np.ndim(result) == 0 else result


def balanced_log_odds_fusion(
    sparse_probs: np.ndarray,
    dense_similarities: np.ndarray,
    weight: float = 0.5,
) -> np.ndarray | float:
    """Balanced log-odds fusion for hybrid sparse-dense retrieval.

    Combines Bayesian BM25 probabilities with dense cosine similarities
    by normalizing both signals in logit space.  Min-max normalization
    ensures each signal contributes equally, preventing the heavy-tailed
    sparse logits (from sigmoid unwrapping) from drowning the dense signal.

    Pipeline:
      1. sparse_probs -> logit(p_sparse)
      2. dense_similarities -> cosine_to_probability -> logit(p_dense)
      3. Min-max normalize each logit array to [0, 1]
      4. Return weight * logit_dense_norm + (1 - weight) * logit_sparse_norm

    Parameters
    ----------
    sparse_probs : array
        Bayesian BM25 probabilities in (0, 1), e.g. from
        ``BayesianBM25Scorer.get_probabilities()``.
    dense_similarities : array
        Cosine similarities in [-1, 1] from a dense encoder.
    weight : float
        Weight for the dense signal (default 0.5 for equal weighting).
        The sparse weight is ``1 - weight``.

    Returns
    -------
    Fusion scores (not probabilities).  Higher is more relevant.
    """
    sparse_probs = np.asarray(sparse_probs, dtype=np.float64)
    dense_similarities = np.asarray(dense_similarities, dtype=np.float64)

    logit_sparse = logit(_clamp_probability(sparse_probs))
    logit_dense = logit(cosine_to_probability(dense_similarities))

    logit_sparse_norm = _min_max_normalize(logit_sparse)
    logit_dense_norm = _min_max_normalize(logit_dense)

    result = weight * logit_dense_norm + (1.0 - weight) * logit_sparse_norm
    return float(result) if np.ndim(result) == 0 else result


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].  Returns zeros if range is negligible."""
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class LearnableLogOddsWeights:
    """Learnable per-signal reliability weights for log-odds conjunction (Remark 5.3.2).

    Learns weights that map from the Naive Bayes uniform initialization
    (w_i = 1/n) to per-signal reliability weights, completing the
    correspondence to a fully parameterized single-layer network in
    log-odds space: logit -> weighted sum -> sigmoid.

    The gradient dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
    is Hebbian: the product of pre-synaptic activity (signal deviation
    from weighted mean) and post-synaptic error (prediction minus label).

    Parameters
    ----------
    n_signals : int
        Number of probability signals to combine (>= 1).
    alpha : float
        Confidence scaling exponent (fixed, not learned).  Default 0.0
        means no count-based scaling.  Weights (per-signal reliability)
        and alpha (confidence scaling) are orthogonal (Paper 2, Section 4.2).
    """

    def __init__(self, n_signals: int, alpha: float = 0.0) -> None:
        if n_signals < 1:
            raise ValueError(
                f"n_signals must be >= 1, got {n_signals}"
            )
        self._n_signals = n_signals
        self._alpha = alpha

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
    ) -> np.ndarray | float:
        """Combine probability signals via weighted log-odds conjunction.

        Parameters
        ----------
        probs : array of shape (..., n_signals)
            Probability values to combine.
        use_averaged : bool
            If True, use Polyak-averaged weights instead of raw weights.

        Returns
        -------
        Combined probability after weighted log-odds conjunction.
        """
        w = self._weights_avg if use_averaged else self.weights
        return log_odds_conjunction(probs, alpha=self._alpha, weights=w)

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
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
        x = logit(_clamp_probability(probs))

        for _ in range(max_iterations):
            w = self._softmax(self._logits)

            # Weighted mean log-odds per sample: shape (m,)
            x_bar_w = np.sum(w * x, axis=-1)

            # Predicted probability: shape (m,)
            p = sigmoid(scale * x_bar_w)
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
        x = logit(_clamp_probability(probs))

        # Weighted mean log-odds per sample: shape (m,)
        x_bar_w = np.sum(w * x, axis=-1)

        # Predicted probability: shape (m,)
        p = sigmoid(scale * x_bar_w)
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
