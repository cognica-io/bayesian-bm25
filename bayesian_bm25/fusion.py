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

from bayesian_bm25.probability import _clamp_probability, logit, sigmoid


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


_SQRT_N_ALPHA = 0.5  # alpha=0.5 implements the sqrt(n) scaling law (Theorem 4.2.1)


def _resolve_alpha(alpha: float | str | None, default: float) -> float:
    """Resolve alpha parameter: "auto" -> sqrt(n) scaling, None -> default."""
    if alpha is None:
        return default
    if isinstance(alpha, str):
        if alpha != "auto":
            raise ValueError(
                f"alpha must be a float, None, or 'auto', got {alpha!r}"
            )
        return _SQRT_N_ALPHA
    return float(alpha)


def _apply_gating(
    logits: np.ndarray, gating: str, beta: float = 1.0,
) -> np.ndarray:
    """Apply sparse-signal gating to logit values before aggregation.

    Parameters
    ----------
    logits : array
        Log-odds values to gate.
    gating : str
        Gating function: "none", "relu", "swish", "gelu", or "softplus".

        - "relu": MAP estimate under sparse prior (Theorem 6.5.3).
          Zeroes out weak/negative evidence: max(0, logit).
        - "swish": Bayes estimate under sparse prior (Theorem 6.7.4).
          Soft gating: logit * sigmoid(beta * logit).  When beta=1.0
          this is the standard swish (Theorem 6.7.6).
        - "gelu": Bayesian expected signal under Gaussian noise model
          (Theorem 6.8.1, Proposition 6.8.2).  Approximated as
          logit * sigmoid(1.702 * logit), which matches Swish_1.702.
          The ``beta`` parameter is ignored for gelu.
        - "softplus": Smooth ReLU that preserves all evidence
          (Remark 6.5.4).  Computes log(1 + exp(beta * logit)) / beta.
          Unlike ReLU, never zeroes out evidence entirely, making it
          suitable for small datasets where discarding any signal is
          costly.  beta=1.0 is the standard softplus; beta -> inf
          approaches ReLU.  Because softplus(x) > x for all finite x,
          it inflates all logits (both positive and negative), producing
          higher fused probabilities than other gating modes.  Consider
          using a lower ``alpha`` to compensate for the increased
          confidence when needed.
    beta : float
        Sharpness parameter for swish and softplus gating.  Controls
        the transition sharpness: beta=1.0 gives the standard form,
        beta -> inf approaches ReLU.  For swish, beta -> 0 approaches
        x/2 (Theorem 6.7.6).  Ignored when gating is "gelu".
    """
    if gating == "none":
        return logits
    if gating == "relu":
        return np.maximum(0.0, logits)
    if gating == "swish":
        return logits * sigmoid(beta * logits)
    if gating == "gelu":
        return logits * sigmoid(1.702 * logits)
    if gating == "softplus":
        return np.logaddexp(0.0, beta * logits) / beta
    raise ValueError(
        f"gating must be 'none', 'relu', 'swish', 'gelu', or 'softplus', "
        f"got {gating!r}"
    )


def log_odds_conjunction(
    probs: np.ndarray,
    alpha: float | str | None = None,
    weights: np.ndarray | None = None,
    gating: str = "none",
    gating_beta: float = 1.0,
    max_logit: float | None = None,
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
    alpha : float, str, or None
        Confidence scaling exponent.  Higher values amplify the effect
        of multiple agreeing signals.  ``"auto"`` resolves to 0.5
        (sqrt(n) scaling law, Theorem 4.2.1).  When None (default),
        uses 0.5 in unweighted mode and 0.0 in weighted mode,
        preserving backward compatibility for both calling conventions.
    weights : array of shape (n,) or None
        Per-signal reliability weights for the Log-OP formulation.
        Must be non-negative and sum to 1.  When None (default),
        uses the unweighted mean log-odds with n^alpha scaling.
    gating : str
        Sparse-signal gating applied to logit values before aggregation.

        - ``"none"`` (default): no gating.
        - ``"relu"``: MAP under sparse prior (Theorem 6.5.3): max(0, logit).
        - ``"swish"``: Bayes under sparse prior (Theorem 6.7.4):
          logit * sigmoid(beta * logit).
        - ``"gelu"``: Bayesian expected signal under Gaussian noise model
          (Theorem 6.8.1): logit * sigmoid(1.702 * logit).
        - ``"softplus"``: smooth ReLU preserving all evidence
          (Remark 6.5.4): log(1 + exp(beta * logit)) / beta.
          Inflates all logits (softplus(x) > x), so consider a lower
          ``alpha`` to temper the increased confidence.
    gating_beta : float
        Sharpness parameter for swish and softplus gating.  Only used
        when ``gating="swish"`` or ``gating="softplus"``.  Default 1.0
        preserves existing behavior.
    max_logit : float or None
        Maximum absolute logit value.  Logits are clipped to
        [-max_logit, +max_logit] after gating and before scaling.
        Prevents sigmoid saturation that destroys ranking
        discrimination when input probabilities are extreme
        (e.g., density-ratio calibrated vector scores near 0 or 1).
        None (default) disables clipping for backward compatibility.

    Returns
    -------
    Combined probability after log-odds conjunction.
    """
    probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
    n = probs.shape[-1]
    raw_logits = logit(probs)
    gated_logits = _apply_gating(raw_logits, gating, beta=gating_beta)

    if max_logit is not None:
        gated_logits = np.clip(gated_logits, -max_logit, max_logit)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if abs(float(np.sum(weights)) - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1, got {float(np.sum(weights))}"
            )

        effective_alpha = _resolve_alpha(alpha, default=0.0)

        # Log-OP with confidence scaling:
        # sigma(n^alpha * sum(w_i * logit(P_i)))  (Theorem 8.3 + Section 4.2)
        l_weighted = (n ** effective_alpha) * np.sum(
            weights * gated_logits, axis=-1
        )
        result = sigmoid(l_weighted)
        return float(result) if np.ndim(result) == 0 else result

    effective_alpha = _resolve_alpha(alpha, default=0.5)

    # Step 1: mean log-odds (Eq. 20)
    l_bar = np.mean(gated_logits, axis=-1)

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

    Notes
    -----
    When all values in a signal are identical (zero variance), the
    ``_min_max_normalize`` step maps that signal to all zeros.  This
    means a zero-variance signal contributes nothing to the fusion
    result -- only the other signal determines the final ranking.
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
        self._logit_base_rate: float | None = (
            float(logit(base_rate)) if base_rate is not None else None
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
        probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
        w = self._weights_avg if use_averaged else self.weights

        n = self._n_signals
        scale = n ** self._alpha
        x = logit(probs)

        l_weighted = scale * np.sum(w * x, axis=-1)
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
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


class AttentionLogOddsWeights:
    """Query-dependent signal weighting via attention (Paper 2, Section 8).

    Computes per-signal softmax attention weights from query features:
    ``w_i(q) = softmax(W @ features + b)[i]``, then combines probability
    signals via weighted log-odds conjunction.  This enables the fusion
    weights to adapt per-query rather than being fixed across all queries.

    The class is feature-agnostic -- it learns a linear projection from
    arbitrary user-provided query features to softmax attention weights.

    Parameters
    ----------
    n_signals : int
        Number of probability signals to combine (>= 1).
    n_query_features : int
        Dimensionality of the query feature vector (>= 1).
    alpha : float or str
        Confidence scaling exponent (fixed, not learned).  ``"auto"``
        resolves to 0.5 (sqrt(n) scaling law, Theorem 4.2.1).
    """

    def __init__(
        self,
        n_signals: int,
        n_query_features: int,
        alpha: float | str = 0.5,
        normalize: bool = False,
        seed: int = 0,
        base_rate: float | None = None,
    ) -> None:
        if n_signals < 1:
            raise ValueError(f"n_signals must be >= 1, got {n_signals}")
        if n_query_features < 1:
            raise ValueError(
                f"n_query_features must be >= 1, got {n_query_features}"
            )
        if base_rate is not None and not (0.0 < base_rate < 1.0):
            raise ValueError(
                f"base_rate must be in (0, 1), got {base_rate}"
            )
        self._n_signals = n_signals
        self._n_query_features = n_query_features
        self._alpha = _resolve_alpha(alpha, default=0.5)
        self._normalize = normalize
        self._base_rate = base_rate
        self._logit_base_rate: float | None = (
            float(logit(base_rate)) if base_rate is not None else None
        )

        # W: (n_signals, n_query_features), b: (n_signals,)
        # Xavier-style initialization scaled for softmax input
        init_scale = 1.0 / np.sqrt(n_query_features)
        rng = np.random.default_rng(seed)
        self._W = rng.normal(0, init_scale, size=(n_signals, n_query_features))
        self._b = np.zeros(n_signals, dtype=np.float64)

        # Online learning state
        self._n_updates: int = 0
        self._grad_W_ema = np.zeros_like(self._W)
        self._grad_b_ema = np.zeros_like(self._b)

        # Polyak averaging
        self._W_avg = self._W.copy()
        self._b_avg = self._b.copy()

    @property
    def n_signals(self) -> int:
        """Number of probability signals."""
        return self._n_signals

    @property
    def n_query_features(self) -> int:
        """Dimensionality of query feature vector."""
        return self._n_query_features

    @property
    def alpha(self) -> float:
        """Confidence scaling exponent (fixed)."""
        return self._alpha

    @property
    def base_rate(self) -> float | None:
        """Corpus-level base rate of relevance, or None if not set."""
        return self._base_rate

    @property
    def normalize(self) -> bool:
        """Whether per-signal logit normalization is enabled."""
        return self._normalize

    @staticmethod
    def _normalize_logits(x: np.ndarray) -> np.ndarray:
        """Per-column min-max normalization on logit array.

        Parameters
        ----------
        x : array of shape (m, n)
            Logit values where each column is a signal.

        Returns
        -------
        Array of same shape with each column independently normalized to [0, 1].
        """
        result = x.copy()
        for col in range(x.shape[-1]):
            result[..., col] = _min_max_normalize(x[..., col])
        return result

    @property
    def weights_matrix(self) -> np.ndarray:
        """Weight matrix W of shape (n_signals, n_query_features)."""
        return self._W.copy()

    def _compute_weights(
        self, query_features: np.ndarray, use_averaged: bool = False
    ) -> np.ndarray:
        """Compute softmax attention weights from query features.

        Parameters
        ----------
        query_features : array of shape (n_query_features,) or (m, n_query_features)
        use_averaged : bool
            Use Polyak-averaged parameters.

        Returns
        -------
        weights : array of shape (n_signals,) or (m, n_signals)
        """
        W = self._W_avg if use_averaged else self._W
        b = self._b_avg if use_averaged else self._b
        # logits: (m, n_signals) = (m, n_qf) @ (n_qf, n_signals) + (n_signals,)
        z = query_features @ W.T + b
        return self._softmax(z)

    def __call__(
        self,
        probs: np.ndarray,
        query_features: np.ndarray,
        use_averaged: bool = False,
    ) -> np.ndarray | float:
        """Combine probability signals via query-dependent weighted log-odds.

        Parameters
        ----------
        probs : array of shape (..., n_signals)
            Probability values to combine.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s) for computing attention weights.
        use_averaged : bool
            If True, use Polyak-averaged parameters.

        Returns
        -------
        Combined probability after attention-weighted log-odds conjunction.
        """
        probs = np.asarray(probs, dtype=np.float64)
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )

        # Per-query attention weights: (m, n_signals)
        w = self._compute_weights(query_features, use_averaged)

        if probs.ndim == 1:
            # Single sample: normalization cannot apply (no candidates to
            # normalize across), fall through to direct computation.
            w_flat = w.squeeze(0)
            x = logit(_clamp_probability(probs))
            n = self._n_signals
            scale = n ** self._alpha
            l_weighted = scale * np.sum(w_flat * x)
            if self._logit_base_rate is not None:
                l_weighted = l_weighted + self._logit_base_rate
            result = sigmoid(l_weighted)
            return float(result)

        # Vectorized batched path (replaces per-row loop)
        x = logit(_clamp_probability(probs))
        if self._normalize:
            x = self._normalize_logits(x)
        n = self._n_signals
        scale = n ** self._alpha
        # w may be (1, n_signals) for single query or (m, n_signals)
        # for per-row queries; broadcast via numpy
        l_weighted = scale * np.sum(w * x, axis=-1)
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        result = sigmoid(l_weighted)
        return np.atleast_1d(np.asarray(result, dtype=np.float64))

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        query_features: np.ndarray,
        *,
        query_ids: np.ndarray | None = None,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """Batch gradient descent on BCE loss to learn W and b.

        Parameters
        ----------
        probs : array of shape (m, n_signals)
            Training probability signals.
        labels : array of shape (m,)
            Binary relevance labels (0 or 1).
        query_features : array of shape (m, n_query_features)
            Query features for each training sample.
        query_ids : array of shape (m,) or None
            Query group identifiers for per-query normalization.  When
            ``normalize=True`` and ``query_ids`` is provided, logit
            normalization is applied within each query group.  When
            ``normalize=True`` and ``query_ids`` is None, the whole batch
            is normalized as a single group.
        learning_rate : float
            Step size for gradient descent.
        max_iterations : int
            Maximum number of gradient descent iterations.
        tolerance : float
            Convergence threshold on maximum absolute parameter change.
        """
        probs = _clamp_probability(np.asarray(probs, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.float64)
        query_features = np.asarray(query_features, dtype=np.float64)

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)

        m = probs.shape[0]
        n = self._n_signals
        scale = n ** self._alpha
        x = logit(probs)  # (m, n)

        if self._normalize:
            if query_ids is not None:
                query_ids = np.asarray(query_ids)
                # Boolean indexing copies, so x[mask] = ... writes back safely
                # without cross-group contamination.
                for qid in np.unique(query_ids):
                    mask = query_ids == qid
                    x[mask] = self._normalize_logits(x[mask])
            else:
                x = self._normalize_logits(x)

        for _ in range(max_iterations):
            # Compute per-sample attention weights
            z = query_features @ self._W.T + self._b  # (m, n)
            w = self._softmax(z)  # (m, n)

            # Weighted log-odds per sample
            x_bar_w = np.sum(w * x, axis=-1)  # (m,)
            l_weighted = scale * x_bar_w
            if self._logit_base_rate is not None:
                l_weighted = l_weighted + self._logit_base_rate
            p = sigmoid(l_weighted)  # (m,)
            p = np.atleast_1d(np.asarray(p, dtype=np.float64))
            error = p - labels  # (m,)

            # Gradient: dL/dz_j = scale * (p - y) * w_j * (x_j - x_bar_w)
            # Then chain through softmax: dz_j/dW_jk = q_k (query feature)
            # and softmax Jacobian: dw_i/dz_j = w_i*(delta_ij - w_j)
            grad_z = (
                scale
                * error[:, np.newaxis]
                * w
                * (x - x_bar_w[:, np.newaxis])
            )  # (m, n)

            # dL/dW = (1/m) * grad_z.T @ query_features  -> (n, n_qf)
            grad_W = grad_z.T @ query_features / m
            grad_b = np.mean(grad_z, axis=0)

            old_W = self._W.copy()
            old_b = self._b.copy()

            self._W -= learning_rate * grad_W
            self._b -= learning_rate * grad_b

            max_change = max(
                float(np.max(np.abs(self._W - old_W))),
                float(np.max(np.abs(self._b - old_b))),
            )
            if max_change < tolerance:
                break

        # Reset online state after batch fit
        self._n_updates = 0
        self._grad_W_ema = np.zeros_like(self._W)
        self._grad_b_ema = np.zeros_like(self._b)
        self._W_avg = self._W.copy()
        self._b_avg = self._b.copy()

    def update(
        self,
        probs: np.ndarray | float,
        label: np.ndarray | float,
        query_features: np.ndarray,
        *,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        decay_tau: float = 1000.0,
        max_grad_norm: float = 1.0,
        avg_decay: float = 0.995,
    ) -> None:
        """Online SGD update from a single observation or mini-batch.

        Parameters
        ----------
        probs : array of shape (n_signals,) or (m, n_signals)
            Probability signal(s).
        label : float or array of shape (m,)
            Binary relevance label(s).
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature(s).
        learning_rate : float
            Base step size, decayed as lr / (1 + t / tau).
        momentum : float
            EMA decay factor for gradient smoothing.
        decay_tau : float
            Time constant for learning rate decay.
        max_grad_norm : float
            Maximum L2 norm for gradient clipping.
        avg_decay : float
            Decay factor for Polyak parameter averaging.
        """
        probs = _clamp_probability(
            np.atleast_1d(np.asarray(probs, dtype=np.float64))
        )
        label = np.atleast_1d(np.asarray(label, dtype=np.float64))
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        n = self._n_signals
        scale = n ** self._alpha
        x = logit(probs)  # (m, n)

        if self._normalize and x.ndim == 2:
            x = self._normalize_logits(x)

        z = query_features @ self._W.T + self._b  # (m, n)
        w = self._softmax(z)  # (m, n)

        x_bar_w = np.sum(w * x, axis=-1)  # (m,)
        l_weighted = scale * x_bar_w
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        p = sigmoid(l_weighted)
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        error = p - label

        grad_z = (
            scale
            * error[:, np.newaxis]
            * w
            * (x - x_bar_w[:, np.newaxis])
        )  # (m, n)

        m = probs.shape[0]
        grad_W = grad_z.T @ query_features / m  # (n, n_qf)
        grad_b = np.mean(grad_z, axis=0)  # (n,)

        # EMA smoothing
        self._grad_W_ema = momentum * self._grad_W_ema + (1.0 - momentum) * grad_W
        self._grad_b_ema = momentum * self._grad_b_ema + (1.0 - momentum) * grad_b

        # Bias correction
        self._n_updates += 1
        correction = 1.0 - momentum ** self._n_updates
        corrected_W = self._grad_W_ema / correction
        corrected_b = self._grad_b_ema / correction

        # L2 gradient clipping (joint norm over W and b)
        grad_norm = float(np.sqrt(
            np.sum(corrected_W ** 2) + np.sum(corrected_b ** 2)
        ))
        if grad_norm > max_grad_norm:
            scale_clip = max_grad_norm / grad_norm
            corrected_W = corrected_W * scale_clip
            corrected_b = corrected_b * scale_clip

        # Learning rate decay
        effective_lr = learning_rate / (1.0 + self._n_updates / decay_tau)

        self._W -= effective_lr * corrected_W
        self._b -= effective_lr * corrected_b

        # Polyak averaging
        self._W_avg = avg_decay * self._W_avg + (1.0 - avg_decay) * self._W
        self._b_avg = avg_decay * self._b_avg + (1.0 - avg_decay) * self._b

    def compute_upper_bounds(
        self,
        upper_bound_probs: np.ndarray,
        query_features: np.ndarray,
        use_averaged: bool = False,
    ) -> np.ndarray:
        """Compute fused probability upper bounds (Theorem 8.7.1).

        Given per-signal probability upper bounds, compute the maximum
        possible fused probability for each candidate.

        Parameters
        ----------
        upper_bound_probs : array of shape (m, n_signals)
            Maximum possible probability per signal per candidate.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s) for computing attention weights.
        use_averaged : bool
            If True, use Polyak-averaged parameters.

        Returns
        -------
        Array of shape (m,) -- upper bound on fused probability per candidate.
        """
        upper_bound_probs = _clamp_probability(
            np.asarray(upper_bound_probs, dtype=np.float64)
        )
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )
        if upper_bound_probs.ndim == 1:
            upper_bound_probs = upper_bound_probs.reshape(1, -1)

        w = self._compute_weights(query_features, use_averaged)
        x = logit(upper_bound_probs)
        if self._normalize:
            x = self._normalize_logits(x)
        n = self._n_signals
        scale = n ** self._alpha
        l_weighted = scale * np.sum(w * x, axis=-1)
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        result = sigmoid(l_weighted)
        return np.atleast_1d(np.asarray(result, dtype=np.float64))

    def prune(
        self,
        probs: np.ndarray,
        query_features: np.ndarray,
        threshold: float,
        upper_bound_probs: np.ndarray | None = None,
        use_averaged: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prune candidates whose upper bound is below threshold (Theorem 8.7.1).

        Parameters
        ----------
        probs : array of shape (m, n_signals)
            Actual probability signals per candidate.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s).
        threshold : float
            Minimum fused probability to survive pruning.
        upper_bound_probs : array of shape (m, n_signals) or None
            Per-signal upper bounds.  If None, uses ``probs`` as its
            own upper bound (no pruning benefit, but safe).
        use_averaged : bool
            If True, use Polyak-averaged parameters.

        Returns
        -------
        (surviving_indices, fused_probabilities) : tuple of arrays
            Indices of candidates that survived pruning and their
            fused probabilities.
        """
        probs = np.asarray(probs, dtype=np.float64)
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if upper_bound_probs is None:
            upper_bound_probs = probs
        upper_bounds = self.compute_upper_bounds(
            upper_bound_probs, query_features, use_averaged
        )
        surviving_mask = upper_bounds >= threshold
        surviving_indices = np.where(surviving_mask)[0]
        if len(surviving_indices) == 0:
            return surviving_indices, np.array([], dtype=np.float64)
        surv_qf = (
            query_features[surviving_indices]
            if query_features.shape[0] > 1
            else query_features
        )
        fused = self(probs[surviving_indices], surv_qf, use_averaged)
        return surviving_indices, np.atleast_1d(fused)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along last axis."""
        z = np.asarray(z, dtype=np.float64)
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class MultiHeadAttentionLogOddsWeights:
    """Multi-head attention fusion (Paper 2, Remark 8.6, Corollary 8.7.2).

    Creates multiple independent ``AttentionLogOddsWeights`` heads, each
    initialized with a different random seed.  At inference time, each
    head produces fused log-odds independently, and the results are
    combined by averaging log-odds across heads before converting back
    to probability via sigmoid.

    Parameters
    ----------
    n_heads : int
        Number of attention heads (>= 1).
    n_signals : int
        Number of probability signals to combine (>= 1).
    n_query_features : int
        Dimensionality of the query feature vector (>= 1).
    alpha : float or str
        Confidence scaling exponent (fixed, not learned).
    normalize : bool
        Whether to apply per-signal logit normalization.
    """

    def __init__(
        self,
        n_heads: int,
        n_signals: int,
        n_query_features: int,
        alpha: float | str = 0.5,
        normalize: bool = False,
    ) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self._n_heads = n_heads
        self._heads = [
            AttentionLogOddsWeights(
                n_signals=n_signals,
                n_query_features=n_query_features,
                alpha=alpha,
                normalize=normalize,
                seed=h,
            )
            for h in range(n_heads)
        ]

    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        return self._n_heads

    @property
    def heads(self) -> list[AttentionLogOddsWeights]:
        """List of attention head instances."""
        return list(self._heads)

    def __call__(
        self,
        probs: np.ndarray,
        query_features: np.ndarray,
        use_averaged: bool = False,
    ) -> np.ndarray | float:
        """Combine probability signals via multi-head attention fusion.

        Each head produces fused log-odds independently.  The final
        result averages the log-odds across heads and applies sigmoid.

        Parameters
        ----------
        probs : array of shape (..., n_signals)
            Probability values to combine.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s).
        use_averaged : bool
            If True, use Polyak-averaged parameters.

        Returns
        -------
        Combined probability after multi-head log-odds averaging.
        """
        probs = np.asarray(probs, dtype=np.float64)
        head_results = []
        for head in self._heads:
            r = head(probs, query_features, use_averaged)
            head_results.append(np.atleast_1d(np.asarray(r, dtype=np.float64)))

        # Average in log-odds space, then sigmoid
        head_logits = [logit(_clamp_probability(r)) for r in head_results]
        avg_logit = np.mean(head_logits, axis=0)
        result = sigmoid(avg_logit)
        if probs.ndim == 1:
            return float(result) if np.ndim(result) == 0 else float(result[0])
        return np.atleast_1d(np.asarray(result, dtype=np.float64))

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        query_features: np.ndarray,
        **kwargs,
    ) -> None:
        """Train all heads on the same data.

        Different random initializations lead to different learned
        solutions, providing diversity across heads.

        Parameters
        ----------
        probs : array of shape (m, n_signals)
            Training probability signals.
        labels : array of shape (m,)
            Binary relevance labels.
        query_features : array of shape (m, n_query_features)
            Query features for each training sample.
        **kwargs
            Additional keyword arguments passed to each head's ``fit()``.
        """
        for head in self._heads:
            head.fit(probs, labels, query_features, **kwargs)

    def update(
        self,
        probs: np.ndarray | float,
        label: np.ndarray | float,
        query_features: np.ndarray,
        **kwargs,
    ) -> None:
        """Online update for all heads.

        Parameters
        ----------
        probs : array of shape (n_signals,) or (m, n_signals)
            Probability signal(s).
        label : float or array of shape (m,)
            Binary relevance label(s).
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature(s).
        **kwargs
            Additional keyword arguments passed to each head's ``update()``.
        """
        for head in self._heads:
            head.update(probs, label, query_features, **kwargs)

    def compute_upper_bounds(
        self,
        upper_bound_probs: np.ndarray,
        query_features: np.ndarray,
        use_averaged: bool = False,
    ) -> np.ndarray:
        """Compute fused upper bounds across heads (Corollary 8.7.2).

        Each head computes its upper bound independently.  The final
        upper bound averages the per-head upper bound log-odds and
        applies sigmoid.

        Parameters
        ----------
        upper_bound_probs : array of shape (m, n_signals)
            Per-signal probability upper bounds.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s).
        use_averaged : bool
            If True, use Polyak-averaged parameters.

        Returns
        -------
        Array of shape (m,) -- upper bound on fused probability per candidate.
        """
        head_bounds = []
        for head in self._heads:
            ub = head.compute_upper_bounds(
                upper_bound_probs, query_features, use_averaged
            )
            head_bounds.append(ub)
        head_logits = [logit(_clamp_probability(b)) for b in head_bounds]
        avg_logit = np.mean(head_logits, axis=0)
        result = sigmoid(avg_logit)
        return np.atleast_1d(np.asarray(result, dtype=np.float64))

    def prune(
        self,
        probs: np.ndarray,
        query_features: np.ndarray,
        threshold: float,
        upper_bound_probs: np.ndarray | None = None,
        use_averaged: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prune candidates using multi-head upper bounds (Corollary 8.7.2).

        Parameters
        ----------
        probs : array of shape (m, n_signals)
            Actual probability signals per candidate.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s).
        threshold : float
            Minimum fused probability to survive pruning.
        upper_bound_probs : array of shape (m, n_signals) or None
            Per-signal upper bounds.  If None, uses ``probs``.
        use_averaged : bool
            If True, use Polyak-averaged parameters.

        Returns
        -------
        (surviving_indices, fused_probabilities) : tuple of arrays
        """
        probs = np.asarray(probs, dtype=np.float64)
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if upper_bound_probs is None:
            upper_bound_probs = probs
        upper_bounds = self.compute_upper_bounds(
            upper_bound_probs, query_features, use_averaged
        )
        surviving_mask = upper_bounds >= threshold
        surviving_indices = np.where(surviving_mask)[0]
        if len(surviving_indices) == 0:
            return surviving_indices, np.array([], dtype=np.float64)
        surv_qf = (
            query_features[surviving_indices]
            if query_features.shape[0] > 1
            else query_features
        )
        fused = self(probs[surviving_indices], surv_qf, use_averaged)
        return surviving_indices, np.atleast_1d(np.asarray(fused, dtype=np.float64))
