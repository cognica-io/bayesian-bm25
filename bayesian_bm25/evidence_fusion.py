#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Prior-free evidence addition and signed Gaussian fusion bounds."""

from __future__ import annotations

import numpy as np


def posterior_to_evidence(
    posterior_log_odds: np.ndarray | float,
    input_prior_log_odds: np.ndarray | float,
) -> np.ndarray | float:
    """Remove the actual source prior from each posterior logit.

    Priors broadcast over the posterior array, typically one prior per
    signal. A missing observation is not a zero-score observation and
    must be modeled by the caller rather than assigned neutral evidence.
    """
    logits = np.asarray(posterior_log_odds, dtype=np.float64)
    prior = np.asarray(input_prior_log_odds, dtype=np.float64)
    if np.any(np.isnan(logits)) or not np.all(np.isfinite(prior)):
        raise ValueError("posterior logits must not be NaN and source priors must be finite")
    result = logits - prior
    if result.shape != logits.shape:
        raise ValueError("source priors must broadcast to the posterior shape")
    return float(result) if result.ndim == 0 else result


def _coefficients(evidence: np.ndarray, coefficients: np.ndarray | None) -> np.ndarray:
    if evidence.ndim == 0 or evidence.shape[-1] == 0:
        raise ValueError("evidence must have a non-empty signal axis")
    n = evidence.shape[-1]
    values = np.ones(n) if coefficients is None else np.asarray(coefficients, dtype=np.float64)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be a finite vector with one value per signal")
    return values


def fuse_evidence(
    evidence: np.ndarray,
    *,
    prior_logit: np.ndarray | float,
    coefficients: np.ndarray | None = None,
) -> np.ndarray | float:
    """Add prior-free evidence and the target prior exactly once.

    Unit coefficients are exact for true conditionally independent
    likelihood ratios. Gaussian evidence coefficients may be signed and
    need not sum to one; pass ``gaussian_evidence_weights(...)`` directly.
    This linear operation does not establish calibration of estimated
    inputs. Opinion pooling uses ``log_odds_fusion`` instead.
    """
    x = np.asarray(evidence, dtype=np.float64)
    v = _coefficients(x, coefficients)
    if np.any(np.isnan(x)):
        raise ValueError("evidence must not contain NaN")
    prior = np.asarray(prior_logit, dtype=np.float64)
    if not np.all(np.isfinite(prior)):
        raise ValueError("prior_logit must be finite")
    # An inactive expert contributes zero even at an infinite logit.
    contributions = np.zeros_like(x)
    np.multiply(x, v, out=contributions, where=v != 0.0)
    pooled = np.sum(contributions, axis=-1)
    result = pooled + prior
    if result.shape != pooled.shape:
        raise ValueError("prior_logit must broadcast to the document shape")
    if np.any(np.isnan(result)):
        raise ValueError("conflicting infinite evidence has no defined posterior")
    return float(result) if result.ndim == 0 else result


def evidence_upper_bound(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    prior_logit: np.ndarray | float,
    coefficients: np.ndarray | None = None,
) -> np.ndarray | float:
    """Bound a signed linear fusion using both sides of each input range.

    Positive coefficients use upper bounds and negative coefficients use
    lower bounds. A negative coefficient with no finite lower bound gives
    an infinite upper bound, which prevents unsafe pruning.
    """
    lo = np.asarray(lower_bounds, dtype=np.float64)
    hi = np.asarray(upper_bounds, dtype=np.float64)
    if lo.shape != hi.shape or np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds must have the same shape and contain no NaN")
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds")
    v = _coefficients(hi, coefficients)
    extreme = np.where(v >= 0.0, hi, lo)
    return fuse_evidence(extreme, coefficients=v, prior_logit=prior_logit)
