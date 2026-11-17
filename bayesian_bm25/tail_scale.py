#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Mean-excess score units for the retrieved tail (Section 4.8)."""

from __future__ import annotations

import operator

import numpy as np


def tail_scale(scores: np.ndarray, k: int | None = None) -> float:
    """Mean excess of the best ``k-1`` scores over the k-th best score.

    Larger scores must mean closer or stronger matches. ``k=None`` uses
    every supplied score; a larger ``k`` than the sample uses the whole
    sample. Fewer than two scores or a tied head return zero, indicating
    an undefined inverse scale. The caller must choose how to handle
    that case rather than silently creating an arbitrarily steep slope.

    For ``s' = a*s + b`` with positive ``a``, the scale changes by ``a``.
    A slope ``theta / tail_scale(scores)`` is therefore scale equivariant.
    ``theta=1`` is a tail-shape convention, not an estimate of relevance.
    The retrieved scores determine this scale only; a corpus count still
    needs a corpus population or a representative weighted sample.
    """
    if k is not None:
        try:
            depth = operator.index(k)
        except TypeError as exc:
            raise ValueError("k must be an integer of at least 2") from exc
        if isinstance(k, bool) or depth < 2:
            raise ValueError("k must be an integer of at least 2")
    s = np.asarray(scores, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if s.size < 2:
        return 0.0
    depth = s.size if k is None else min(depth, s.size)
    top = np.partition(s, s.size - depth)[-depth:]
    threshold = float(np.min(top))
    excess = float(np.sum(top - threshold) / (depth - 1))
    if excess <= 0.0:
        return 0.0
    return excess
