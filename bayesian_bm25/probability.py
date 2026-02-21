#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Bayesian probability transforms for BM25 scores.

Implements the sigmoid likelihood + composite prior + Bayesian posterior
framework from "Bayesian BM25" for converting raw BM25 retrieval scores
into calibrated probabilities.

All functions operate on numpy arrays (vectorized) and scalars.
"""

from __future__ import annotations

import numpy as np


_EPSILON = 1e-10


def _clamp_probability(p: np.ndarray | float) -> np.ndarray | float:
    """Clamp probability to [epsilon, 1 - epsilon] for numerical stability (Eq. 40)."""
    return np.clip(p, _EPSILON, 1.0 - _EPSILON)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid function.

    Uses the split formulation to avoid overflow:
      x >= 0: 1 / (1 + exp(-x))
      x <  0: exp(x) / (1 + exp(x))
    """
    x = np.asarray(x, dtype=np.float64)
    pos = 1.0 / (1.0 + np.exp(-np.clip(x, 0, None)))
    exp_x = np.exp(np.clip(x, None, 0))
    neg = exp_x / (1.0 + exp_x)
    result = np.where(x >= 0, pos, neg)
    return float(result) if result.ndim == 0 else result


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Logit (inverse sigmoid): log(p / (1 - p))."""
    p = _clamp_probability(np.asarray(p, dtype=np.float64))
    result = np.log(p / (1.0 - p))
    return float(result) if result.ndim == 0 else result


class BayesianProbabilityTransform:
    """Transforms raw BM25 scores into calibrated probabilities.

    Parameters
    ----------
    alpha : float
        Steepness of the sigmoid likelihood function.
    beta : float
        Midpoint (shift) of the sigmoid likelihood function.
    base_rate : float or None
        Corpus-level base rate of relevance, in (0, 1).  When set,
        the posterior includes base_rate via a two-step Bayes update
        (Remark 4.4.5), equivalent to
        sigmoid(logit(L) + logit(base_rate) + logit(prior)).
        None (default) disables base rate correction for backward
        compatibility.  base_rate=0.5 is neutral (logit(0.5) = 0).
    """

    _VALID_MODES = ("balanced", "prior_aware", "prior_free")

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        base_rate: float | None = None,
    ) -> None:
        if base_rate is not None:
            if not (0.0 < base_rate < 1.0):
                raise ValueError(
                    f"base_rate must be in (0, 1), got {base_rate}"
                )
        self.alpha = alpha
        self.beta = beta
        self.base_rate = base_rate
        self._logit_base_rate: float | None = (
            float(logit(base_rate)) if base_rate is not None else None
        )
        self._training_mode: str = "balanced"
        self._n_updates: int = 0
        self._grad_alpha_ema: float = 0.0
        self._grad_beta_ema: float = 0.0
        self._alpha_avg: float = alpha
        self._beta_avg: float = beta

    @property
    def averaged_alpha(self) -> float:
        """EMA-averaged alpha for stable inference after online updates."""
        return self._alpha_avg

    @property
    def averaged_beta(self) -> float:
        """EMA-averaged beta for stable inference after online updates."""
        return self._beta_avg

    def likelihood(self, score: np.ndarray | float) -> np.ndarray | float:
        """Sigmoid likelihood: sigma(alpha * (score - beta))  (Eq. 20)."""
        return sigmoid(self.alpha * (np.asarray(score, dtype=np.float64) - self.beta))

    @staticmethod
    def tf_prior(tf: np.ndarray | float) -> np.ndarray | float:
        """Term-frequency prior: 0.2 + 0.7 * min(1, tf / 10)  (Eq. 25)."""
        tf = np.asarray(tf, dtype=np.float64)
        result = 0.2 + 0.7 * np.minimum(1.0, tf / 10.0)
        return float(result) if result.ndim == 0 else result

    @staticmethod
    def norm_prior(doc_len_ratio: np.ndarray | float) -> np.ndarray | float:
        """Document-length normalisation prior (Eq. 26).

        P_norm = 0.3 + 0.6 * (1 - min(1, |doc_len_ratio - 0.5| * 2))

        where doc_len_ratio = doc_len / avgdl.  The prior peaks at 0.9
        when doc_len_ratio = 0.5 and falls to a floor of 0.3 at
        doc_len_ratio = 0.0 and doc_len_ratio >= 1.0.
        """
        r = np.asarray(doc_len_ratio, dtype=np.float64)
        result = 0.3 + 0.6 * (1.0 - np.minimum(1.0, np.abs(r - 0.5) * 2.0))
        return float(result) if result.ndim == 0 else result

    @staticmethod
    def composite_prior(
        tf: np.ndarray | float,
        doc_len_ratio: np.ndarray | float,
    ) -> np.ndarray | float:
        """Composite prior: clamp(0.7 * P_tf + 0.3 * P_norm, 0.1, 0.9)  (Eq. 27)."""
        p_tf = BayesianProbabilityTransform.tf_prior(tf)
        p_norm = BayesianProbabilityTransform.norm_prior(doc_len_ratio)
        result = np.clip(0.7 * p_tf + 0.3 * p_norm, 0.1, 0.9)
        return float(result) if np.ndim(result) == 0 else result

    @staticmethod
    def posterior(
        likelihood_val: np.ndarray | float,
        prior: np.ndarray | float,
        base_rate: float | None = None,
    ) -> np.ndarray | float:
        """Bayesian posterior via two-step Bayes update (Eq. 22, Remark 4.4.5).

        Without base_rate:
            P = L*p / (L*p + (1-L)*(1-p))

        With base_rate (two-step, avoids expensive logit/sigmoid):
            Step 1: p1 = L*p / (L*p + (1-L)*(1-p))
            Step 2: P  = p1*br / (p1*br + (1-p1)*(1-br))

        Equivalent to sigmoid(logit(L) + logit(prior) + logit(base_rate)).
        """
        l_val = np.asarray(likelihood_val, dtype=np.float64)
        p = np.asarray(prior, dtype=np.float64)
        numerator = l_val * p
        denominator = numerator + (1.0 - l_val) * (1.0 - p)
        result = _clamp_probability(numerator / denominator)
        if base_rate is not None:
            br = np.float64(base_rate)
            numerator_br = result * br
            denominator_br = numerator_br + (1.0 - result) * (1.0 - br)
            result = _clamp_probability(numerator_br / denominator_br)
        return float(result) if np.ndim(result) == 0 else result

    def score_to_probability(
        self,
        score: np.ndarray | float,
        tf: np.ndarray | float,
        doc_len_ratio: np.ndarray | float,
    ) -> np.ndarray | float:
        """Full pipeline: BM25 score -> calibrated probability.

        Computes likelihood from the score, composite prior from tf and
        doc_len_ratio, then applies the Bayesian posterior formula.
        When base_rate is set, the posterior includes base_rate via a
        two-step Bayes update (Remark 4.4.5).

        In prior_free mode (C3), uses prior=0.5 so the posterior equals the
        likelihood, ignoring the composite prior at inference time.
        """
        l_val = self.likelihood(score)

        if self._training_mode == "prior_free":
            prior = np.float64(0.5)
        else:
            prior = self.composite_prior(tf, doc_len_ratio)

        return self.posterior(l_val, prior, base_rate=self.base_rate)

    def wand_upper_bound(
        self,
        bm25_upper_bound: np.ndarray | float,
        p_max: float = 0.9,
    ) -> np.ndarray | float:
        """Compute the Bayesian WAND upper bound for safe document pruning (Theorem 6.1.2).

        Given a standard BM25 upper bound per term, computes the tightest
        safe Bayesian probability upper bound by assuming the maximum
        possible prior (p_max from Theorem 4.2.4).

        The upper bound is:
            L_max = sigmoid(alpha * (bm25_upper_bound - beta))
            P_max = posterior(L_max, p_max [, base_rate])

        Any document's actual Bayesian probability is guaranteed to be
        at most this value, making it safe for WAND-style pruning.

        Parameters
        ----------
        bm25_upper_bound : float or array
            Standard BM25 upper bound score(s) per term.
        p_max : float
            Global prior upper bound from composite_prior clamp (default 0.9,
            which is the maximum of Eq. 27).

        Returns
        -------
        Bayesian probability upper bound(s) for safe pruning.
        """
        l_max = self.likelihood(bm25_upper_bound)
        return self.posterior(l_max, p_max, base_rate=self.base_rate)

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        mode: str = "balanced",
        tfs: np.ndarray | None = None,
        doc_len_ratios: np.ndarray | None = None,
    ) -> None:
        """Learn alpha and beta via gradient descent (Algorithm 8.3.1).

        Minimises binary cross-entropy using simple gradient descent.
        Three training modes are supported (C1/C2/C3 conditions):

        - ``"balanced"`` (C1, default): trains on the sigmoid likelihood
          pred = sigmoid(alpha*(s-beta)).
        - ``"prior_aware"`` (C2): trains on the full Bayesian posterior
          pred = L*p / (L*p + (1-L)*(1-p)) where L is the sigmoid
          likelihood and p is the composite prior.  Requires ``tfs``
          and ``doc_len_ratios``.
        - ``"prior_free"`` (C3): same training as balanced, but at
          inference time ``score_to_probability`` uses prior=0.5
          (posterior = likelihood).

        Parameters
        ----------
        scores : array of BM25 scores
        labels : array of binary relevance labels (0 or 1)
        learning_rate : step size for gradient updates
        max_iterations : maximum number of gradient descent steps
        tolerance : convergence threshold on parameter change
        mode : str
            Training mode: "balanced", "prior_aware", or "prior_free".
        tfs : array or None
            Term frequencies per sample.  Required when mode="prior_aware".
        doc_len_ratios : array or None
            Document length ratios (doc_len / avgdl) per sample.
            Required when mode="prior_aware".
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got {mode!r}"
            )
        if mode == "prior_aware":
            if tfs is None or doc_len_ratios is None:
                raise ValueError(
                    "tfs and doc_len_ratios are required when mode='prior_aware'"
                )

        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

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
                # Posterior: P = L*p / (L*p + (1-L)*(1-p))
                p = priors
                denom = L * p + (1.0 - L) * (1.0 - p)
                predicted = _clamp_probability(L * p / denom)

                # Chain rule: dBCE/dalpha = (P - y) * dP/dL * dL/dalpha
                # dP/dL = p*(1-p) / denom^2
                # dL/dalpha = L*(1-L)*(s-beta)
                # dL/dbeta  = -L*(1-L)*alpha
                dP_dL = p * (1.0 - p) / (denom ** 2)
                dL_dalpha = L * (1.0 - L) * (scores - beta)
                dL_dbeta = -L * (1.0 - L) * alpha

                error = predicted - labels
                grad_alpha = np.mean(error * dP_dL * dL_dalpha)
                grad_beta = np.mean(error * dP_dL * dL_dbeta)
            else:
                # balanced or prior_free: train on sigmoid likelihood
                predicted = L
                error = predicted - labels
                grad_alpha = np.mean(error * (scores - beta))
                grad_beta = np.mean(error * (-alpha))

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
        """Online update of alpha and beta from a single observation or mini-batch.

        Uses SGD with exponential moving average (EMA) of gradients to
        smooth out noise from individual feedback signals.  Alpha is
        constrained to remain positive (a negative steepness would
        invert the score-probability relationship).

        After each parameter step, Polyak-style EMA averaging is applied
        to produce stable ``averaged_alpha`` and ``averaged_beta`` values
        that smooth out per-batch oscillation.

        Parameters
        ----------
        score : float or array
            BM25 score(s) for the observed document(s).
        label : float or array
            Relevance label(s): 1.0 for relevant, 0.0 for not relevant.
        learning_rate : float
            Base step size.  Decayed as lr / (1 + t / decay_tau).
        momentum : float
            EMA decay factor for gradient smoothing.  Higher values
            (closer to 1) smooth more aggressively.
        decay_tau : float
            Time constant for learning rate decay.  Larger values mean
            slower decay.  The effective rate at step t is
            learning_rate / (1 + t / decay_tau).
        max_grad_norm : float
            Maximum L2 norm for the gradient vector (alpha, beta).
            Gradients exceeding this norm are clipped to prevent
            catastrophic parameter changes from imbalanced batches.
        avg_decay : float
            Decay factor for Polyak parameter averaging.  The averaged
            parameters are updated as avg = decay * avg + (1-decay) * raw.
            Higher values (closer to 1) produce smoother averages.
        mode : str or None
            Training mode override.  If None, uses the mode set by the
            last call to ``fit()``, defaulting to ``"balanced"``.
        tf : float, array, or None
            Term frequency(ies).  Required when mode="prior_aware".
        doc_len_ratio : float, array, or None
            Document length ratio(s).  Required when mode="prior_aware".
        """
        effective_mode = mode if mode is not None else self._training_mode
        if effective_mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got {effective_mode!r}"
            )
        if effective_mode == "prior_aware":
            if tf is None or doc_len_ratio is None:
                raise ValueError(
                    "tf and doc_len_ratio are required when mode='prior_aware'"
                )

        score = np.atleast_1d(np.asarray(score, dtype=np.float64))
        label = np.atleast_1d(np.asarray(label, dtype=np.float64))

        L = _clamp_probability(sigmoid(self.alpha * (score - self.beta)))

        if effective_mode == "prior_aware":
            tf_arr = np.atleast_1d(np.asarray(tf, dtype=np.float64))
            dlr_arr = np.atleast_1d(np.asarray(doc_len_ratio, dtype=np.float64))
            p = np.asarray(self.composite_prior(tf_arr, dlr_arr), dtype=np.float64)
            denom = L * p + (1.0 - L) * (1.0 - p)
            predicted = _clamp_probability(L * p / denom)

            dP_dL = p * (1.0 - p) / (denom ** 2)
            dL_dalpha = L * (1.0 - L) * (score - self.beta)
            dL_dbeta = -L * (1.0 - L) * self.alpha

            error = predicted - label
            grad_alpha = float(np.mean(error * dP_dL * dL_dalpha))
            grad_beta = float(np.mean(error * dP_dL * dL_dbeta))
        else:
            predicted = L
            error = predicted - label
            grad_alpha = float(np.mean(error * (score - self.beta)))
            grad_beta = float(np.mean(error * (-self.alpha)))

        if mode is not None:
            self._training_mode = effective_mode

        # EMA smoothing of gradients
        self._grad_alpha_ema = momentum * self._grad_alpha_ema + (1 - momentum) * grad_alpha
        self._grad_beta_ema = momentum * self._grad_beta_ema + (1 - momentum) * grad_beta

        # Bias correction for early updates
        self._n_updates += 1
        correction = 1.0 - momentum ** self._n_updates
        corrected_grad_alpha = self._grad_alpha_ema / correction
        corrected_grad_beta = self._grad_beta_ema / correction

        # Gradient clipping to prevent parameter collapse from imbalanced batches
        grad_norm = np.sqrt(corrected_grad_alpha ** 2 + corrected_grad_beta ** 2)
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / grad_norm
            corrected_grad_alpha *= scale
            corrected_grad_beta *= scale

        # Learning rate decay: lr / (1 + t / tau)
        effective_lr = learning_rate / (1.0 + self._n_updates / decay_tau)

        self.alpha -= effective_lr * corrected_grad_alpha
        self.beta -= effective_lr * corrected_grad_beta

        # Alpha must stay positive (steepness cannot be inverted)
        _ALPHA_MIN = 0.01
        if self.alpha < _ALPHA_MIN:
            self.alpha = _ALPHA_MIN

        # Polyak parameter averaging for stable inference
        self._alpha_avg = avg_decay * self._alpha_avg + (1.0 - avg_decay) * self.alpha
        self._beta_avg = avg_decay * self._beta_avg + (1.0 - avg_decay) * self.beta
