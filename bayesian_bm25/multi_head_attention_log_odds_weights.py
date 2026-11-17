#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""MultiHeadAttentionLogOddsWeights with explicit source and target priors."""

from __future__ import annotations

import numpy as np

from bayesian_bm25.attention_log_odds_weights import AttentionLogOddsWeights
from bayesian_bm25.probability import sigmoid


class MultiHeadAttentionLogOddsWeights:
    """Multi-head attention fusion (Paper 2, Remark 8.6, Corollary 8.7.2).

    Creates multiple independent ``AttentionLogOddsWeights`` heads, each
    initialized with a different random seed.  At inference time, each
    head produces fused log-odds independently, and the results are
    combined by averaging log-odds across heads before converting back
    to probability via sigmoid.

    Source posterior logits can be converted to evidence by supplying
    ``input_prior_log_odds`` and an explicit target ``base_rate``. Without
    source priors the supplied logits are pooled as given. Score
    normalization does not establish probability calibration.

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
        base_rate: float | None = None,
        input_prior_log_odds: np.ndarray | float | None = None,
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
                base_rate=base_rate,
                input_prior_log_odds=input_prior_log_odds,
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
        *,
        from_log_odds: bool = False,
        return_log_odds: bool = False,
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
        from_log_odds : bool
            If True, ``probs`` holds log-odds instead of probabilities.
        return_log_odds : bool
            If True, return the averaged log-odds instead of a
            probability.

        Returns
        -------
        Combined probability (or log-odds) after multi-head log-odds
        averaging.
        """
        probs = np.asarray(probs, dtype=np.float64)

        # Each head reports log-odds, which are averaged directly: a
        # round trip through a probability would clamp them.
        head_logits = [
            np.atleast_1d(
                np.asarray(
                    head(
                        probs, query_features, use_averaged,
                        from_log_odds=from_log_odds, return_log_odds=True,
                    ),
                    dtype=np.float64,
                )
            )
            for head in self._heads
        ]
        avg_logit = np.mean(head_logits, axis=0)
        result = avg_logit if return_log_odds else sigmoid(avg_logit)
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
        *,
        from_log_odds: bool = False,
        return_log_odds: bool = False,
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
        from_log_odds : bool
            If True, ``upper_bound_probs`` holds log-odds upper bounds.
        return_log_odds : bool
            If True, return upper bounds on the fused log-odds.

        Returns
        -------
        Array of shape (m,) -- upper bound on fused probability (or
        log-odds) per candidate.
        """
        head_logits = [
            head.compute_upper_bounds(
                upper_bound_probs, query_features, use_averaged,
                from_log_odds=from_log_odds, return_log_odds=True,
            )
            for head in self._heads
        ]
        avg_logit = np.mean(head_logits, axis=0)
        result = avg_logit if return_log_odds else sigmoid(avg_logit)
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
        """Prune candidates using multi-head upper bounds (Corollary 8.7.2).

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
            Per-signal upper bounds.  If None, uses ``probs``.
        use_averaged : bool
            If True, use Polyak-averaged parameters.
        from_log_odds : bool
            If True, ``probs`` and ``upper_bound_probs`` hold log-odds.
        return_log_odds : bool
            If True, ``threshold`` and the returned fused values are
            log-odds.

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
            upper_bound_probs, query_features, use_averaged,
            from_log_odds=from_log_odds, return_log_odds=return_log_odds,
        )
        surviving_mask = upper_bounds >= threshold
        surviving_indices = np.where(surviving_mask)[0]
        if len(surviving_indices) == 0:
            return surviving_indices, np.array([], dtype=np.float64)
        fused = np.atleast_1d(self(
            probs, query_features, use_averaged,
            from_log_odds=from_log_odds, return_log_odds=return_log_odds,
        ))[surviving_indices]
        return surviving_indices, np.atleast_1d(np.asarray(fused, dtype=np.float64))
