#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""AttentionLogOddsWeights with explicit source and target priors."""

from __future__ import annotations

import math

import numpy as np

from bayesian_bm25.fusion import (
    _min_max_normalize,
    _resolve_alpha,
    _to_log_odds,
    _validated_input_priors,
)
from bayesian_bm25.probability import sigmoid


class AttentionLogOddsWeights:
    """Query-dependent signal weighting via attention (Paper 2, Section 8).

    Computes per-signal softmax attention weights from query features:
    ``w_i(q) = softmax(W @ features + b)[i]``, then combines probability
    signals via weighted log-odds conjunction.  This enables the fusion
    weights to adapt per-query rather than being fixed across all queries.

    The class is feature-agnostic -- it learns a linear projection from
    arbitrary user-provided query features to softmax attention weights.

    Source posterior logits can be converted to evidence by supplying
    ``input_prior_log_odds`` and an explicit target ``base_rate``. Without
    source priors the supplied logits are pooled as given. Score
    normalization does not establish probability calibration.

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
        input_prior_log_odds: np.ndarray | float | None = None,
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
        self._input_prior_log_odds = _validated_input_priors(
            input_prior_log_odds, n_signals, base_rate,
        )
        self._logit_base_rate: float | None = (
            math.log(base_rate) - math.log1p(-base_rate) if base_rate is not None else None
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
        *,
        from_log_odds: bool = False,
        return_log_odds: bool = False,
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
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities
            and no clamping is applied.
        return_log_odds : bool
            If True, return the fused log-odds instead of a probability.
            Rank and prune on log-odds; a probability saturates.

        Returns
        -------
        Combined probability (or log-odds) after attention-weighted
        log-odds conjunction.
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
            x = _to_log_odds(probs, from_log_odds, self._input_prior_log_odds)
            n = self._n_signals
            scale = n ** self._alpha
            l_weighted = scale * np.sum(w_flat * x)
            if self._logit_base_rate is not None:
                l_weighted = l_weighted + self._logit_base_rate
            if return_log_odds:
                return float(l_weighted)
            result = sigmoid(l_weighted)
            return float(result)

        # Vectorized batched path (replaces per-row loop)
        x = _to_log_odds(probs, from_log_odds, self._input_prior_log_odds)
        if self._normalize:
            x = self._normalize_logits(x)
        n = self._n_signals
        scale = n ** self._alpha
        # w may be (1, n_signals) for single query or (m, n_signals)
        # for per-row queries; broadcast via numpy
        l_weighted = scale * np.sum(w * x, axis=-1)
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        if return_log_odds:
            return np.atleast_1d(np.asarray(l_weighted, dtype=np.float64))
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
        from_log_odds: bool = False,
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
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        query_features = np.asarray(query_features, dtype=np.float64)

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)

        m = probs.shape[0]
        n = self._n_signals
        scale = n ** self._alpha
        # Copy: per-query normalization below writes into x.
        x = np.array(
            _to_log_odds(probs, from_log_odds, self._input_prior_log_odds), dtype=np.float64,
        )

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
        from_log_odds: bool = False,
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
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities.
        """
        probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))
        label = np.atleast_1d(np.asarray(label, dtype=np.float64))
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        n = self._n_signals
        scale = n ** self._alpha
        x = _to_log_odds(probs, from_log_odds, self._input_prior_log_odds)  # (m, n)

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
        *,
        from_log_odds: bool = False,
        return_log_odds: bool = False,
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
        from_log_odds : bool
            If True, ``upper_bound_probs`` holds log-odds upper bounds.
        return_log_odds : bool
            If True, return upper bounds on the fused log-odds.

        Returns
        -------
        Array of shape (m,) -- upper bound on fused probability (or
        log-odds) per candidate.
        """
        upper_bound_probs = np.asarray(upper_bound_probs, dtype=np.float64)
        query_features = np.atleast_2d(
            np.asarray(query_features, dtype=np.float64)
        )
        if upper_bound_probs.ndim == 1:
            upper_bound_probs = upper_bound_probs.reshape(1, -1)

        w = self._compute_weights(query_features, use_averaged)
        x = _to_log_odds(upper_bound_probs, from_log_odds, self._input_prior_log_odds)
        if self._normalize:
            # Candidate-dependent min/max normalization gives inputs in
            # [0, 1], but normalizing bounds separately is not monotone.
            # Without fixed population normalization statistics the safe
            # bound is the maximum normalized input for every signal.
            x = np.ones_like(x)
        n = self._n_signals
        scale = n ** self._alpha
        l_weighted = scale * np.sum(w * x, axis=-1)
        if self._logit_base_rate is not None:
            l_weighted = l_weighted + self._logit_base_rate
        if return_log_odds:
            return np.atleast_1d(np.asarray(l_weighted, dtype=np.float64))
        result = sigmoid(l_weighted)
        return np.atleast_1d(np.asarray(result, dtype=np.float64))

    def prune(
        self,
        probs: np.ndarray,
        query_features: np.ndarray,
        threshold: float,
        upper_bound_probs: np.ndarray | None = None,
        use_averaged: bool = False,
        *,
        from_log_odds: bool = False,
        return_log_odds: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prune candidates whose upper bound is below threshold (Theorem 8.7.1).

        Parameters
        ----------
        probs : array of shape (m, n_signals)
            Actual probability signals per candidate.
        query_features : array of shape (n_query_features,) or (m, n_query_features)
            Query feature vector(s).
        threshold : float
            Minimum fused probability to survive pruning (a log-odds
            value when ``return_log_odds`` is True).
        upper_bound_probs : array of shape (m, n_signals) or None
            Per-signal upper bounds.  If None, uses ``probs`` as its
            own upper bound (no pruning benefit, but safe).
        use_averaged : bool
            If True, use Polyak-averaged parameters.
        from_log_odds : bool
            If True, ``probs`` and ``upper_bound_probs`` hold log-odds.
        return_log_odds : bool
            If True, ``threshold`` and the returned fused values are
            log-odds.  Pruning on log-odds keeps discriminating between
            candidates whose probability bounds have saturated.

        Returns
        -------
        (surviving_indices, fused_probabilities) : tuple of arrays
            Indices of candidates that survived pruning and their
            fused probabilities (or log-odds).
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
            upper_bound_probs, query_features, use_averaged,
            from_log_odds=from_log_odds, return_log_odds=return_log_odds,
        )
        surviving_mask = upper_bounds >= threshold
        surviving_indices = np.where(surviving_mask)[0]
        if len(surviving_indices) == 0:
            return surviving_indices, np.array([], dtype=np.float64)
        # Normalize once on the original population: pruning must not
        # change the scores of documents that survive.
        fused = np.atleast_1d(self(
            probs, query_features, use_averaged,
            from_log_odds=from_log_odds, return_log_odds=return_log_odds,
        ))[surviving_indices]
        return surviving_indices, np.atleast_1d(fused)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along last axis."""
        z = np.asarray(z, dtype=np.float64)
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
