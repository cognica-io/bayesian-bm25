#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Fixed evidence and result-list populations for the ceiling experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesian_bm25.tail_scale import tail_scale
from benchmarks.evidence_protocol import TAIL_DEPTH
from benchmarks.query_signals import QuerySignals, _top_positions


@dataclass
class CeilingQuery:
    """A query with units held fixed when the corpus is restricted."""

    signals: QuerySignals
    lexical_scale: float
    dense_scale: float

    def evidence(self, channel: str) -> np.ndarray:
        if channel == "lexical":
            return self.signals.bm25 / self.lexical_scale
        if channel == "dense":
            return self.signals.dense / self.dense_scale
        if channel == "fused":
            return (
                self.signals.bm25 / self.lexical_scale + self.signals.dense / self.dense_scale
            ) / np.sqrt(2.0)
        raise ValueError(f"unknown channel {channel!r}")

    def positions(self, channel: str) -> np.ndarray:
        sig = self.signals
        if channel == "lexical":
            return sig.candidates[sig.bm25_rank > 0]
        if channel == "dense":
            return sig.candidates[sig.dense_rank > 0]
        if channel == "fused":
            return sig.candidates
        raise ValueError(f"unknown channel {channel!r}")

    def head(self, channel: str, k: int = 10) -> np.ndarray:
        positions = self.positions(channel)
        scores = self.evidence(channel)[positions]
        return positions[np.argsort(-scores, kind="stable")[:k]]

    def restrict(self, kept: np.ndarray, retrieve_k: int) -> CeilingQuery:
        """Keep original document scores and original units, then retrieve."""
        sig = self.signals
        selected = np.flatnonzero(np.isin(sig.doc_index, kept))
        bm25, dense = sig.bm25[selected], sig.dense[selected]
        lexical_top = _top_positions(bm25, retrieve_k, bm25 > 0.0)
        dense_top = _top_positions(dense, retrieve_k, None)
        candidates = np.union1d(lexical_top, dense_top)
        lexical_rank = np.zeros(len(bm25), dtype=np.int64)
        dense_rank = np.zeros(len(bm25), dtype=np.int64)
        lexical_rank[lexical_top] = np.arange(1, len(lexical_top) + 1)
        dense_rank[dense_top] = np.arange(1, len(dense_top) + 1)
        return CeilingQuery(
            QuerySignals(
                sig.index,
                sig.qid,
                sig.doc_index[selected],
                bm25,
                dense,
                sig.labels[selected],
                candidates,
                lexical_rank[candidates],
                dense_rank[candidates],
            ),
            self.lexical_scale,
            self.dense_scale,
        )


def make_ceiling_query(signals: QuerySignals, corpus_scale: float) -> CeilingQuery:
    """Matching-score standard deviation and top-list mean-excess scale."""
    matching = signals.bm25[signals.matched]
    lexical = float(np.std(matching)) if len(matching) >= 2 else 0.0
    if lexical <= 0.0:
        lexical = corpus_scale
    dense_positions = signals.candidates[signals.dense_rank > 0]
    dense = tail_scale(signals.dense[dense_positions], k=TAIL_DEPTH)
    if dense <= 0.0:
        raise ValueError(f"query {signals.qid!r} has no dense tail scale")
    return CeilingQuery(signals, lexical, dense)
