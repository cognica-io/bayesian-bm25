#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark-local IVF helper for dense retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_EPSILON = 1e-12


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, _EPSILON)


@dataclass
class IVFSearchResult:
    """Result bundle for one IVF query."""

    indices: np.ndarray
    scores: np.ndarray
    cell_ids: np.ndarray
    cell_populations: np.ndarray
    candidate_indices: np.ndarray
    candidate_scores: np.ndarray
    candidate_cell_ids: np.ndarray
    candidate_cell_populations: np.ndarray
    probed_cell_ids: np.ndarray
    probed_cell_scores: np.ndarray


class SimpleIVF:
    """A small cosine-similarity IVF index backed by NumPy."""

    def __init__(
        self,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        sorted_doc_ids: np.ndarray,
        cell_offsets: np.ndarray,
        *,
        default_nprobe: int,
        background_distances: np.ndarray,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.assignments = np.asarray(assignments, dtype=np.int32)
        self.sorted_doc_ids = np.asarray(sorted_doc_ids, dtype=np.int32)
        self.cell_offsets = np.asarray(cell_offsets, dtype=np.int64)
        self.default_nprobe = int(default_nprobe)
        self.background_distances = np.asarray(
            background_distances, dtype=np.float64,
        )

        self.n_docs = int(self.embeddings.shape[0])
        self.dim = int(self.embeddings.shape[1])
        self.n_cells = int(self.centroids.shape[0])
        self.cell_populations = np.diff(self.cell_offsets).astype(np.int32)
        self.avg_population = float(np.mean(self.cell_populations))

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        *,
        n_cells: int | None = None,
        max_iterations: int = 10,
        seed: int = 42,
    ) -> SimpleIVF:
        embeddings = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
        n_docs, dim = embeddings.shape

        if n_docs == 0:
            raise ValueError("embeddings must contain at least one vector")
        if n_cells is None:
            n_cells = max(4, int(round(math.sqrt(n_docs))))
        n_cells = max(1, min(int(n_cells), n_docs))
        if max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {max_iterations}",
            )

        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n_docs, size=n_cells, replace=False)
        centroids = embeddings[init_idx].copy()
        assignments = np.full(n_docs, -1, dtype=np.int32)

        for _ in range(max_iterations):
            sims = embeddings @ centroids.T
            new_assignments = np.argmax(sims, axis=1).astype(np.int32)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            centroid_sums = np.zeros((n_cells, dim), dtype=np.float32)
            np.add.at(centroid_sums, assignments, embeddings)

            counts = np.bincount(assignments, minlength=n_cells).astype(np.int32)
            safe_counts = np.maximum(counts, 1).astype(np.float32)
            centroids = centroid_sums / safe_counts[:, np.newaxis]

            empty = counts == 0
            if np.any(empty):
                refill_idx = rng.choice(
                    n_docs, size=int(np.sum(empty)), replace=False,
                )
                centroids[empty] = embeddings[refill_idx]

            centroids = _l2_normalize_rows(centroids)

        final_sims = embeddings @ centroids.T
        assignments = np.argmax(final_sims, axis=1).astype(np.int32)
        counts = np.bincount(assignments, minlength=n_cells).astype(np.int32)

        order = np.argsort(assignments, kind="stable")
        offsets = np.zeros(n_cells + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

        centroid_scores = np.sum(
            embeddings * centroids[assignments], axis=1, dtype=np.float32,
        )
        background_distances = 1.0 - centroid_scores.astype(np.float64)

        default_nprobe = max(1, int(round(math.sqrt(n_cells))))
        return cls(
            embeddings=embeddings,
            centroids=centroids,
            assignments=assignments,
            sorted_doc_ids=order.astype(np.int32),
            cell_offsets=offsets,
            default_nprobe=default_nprobe,
            background_distances=background_distances,
        )

    def _docs_for_cells(self, cell_ids: np.ndarray) -> np.ndarray:
        doc_groups: list[np.ndarray] = []
        total = 0
        for cell_id in cell_ids:
            start = int(self.cell_offsets[cell_id])
            end = int(self.cell_offsets[cell_id + 1])
            if end > start:
                docs = self.sorted_doc_ids[start:end]
                doc_groups.append(docs)
                total += len(docs)
        if total == 0:
            return np.empty(0, dtype=np.int32)
        if len(doc_groups) == 1:
            return doc_groups[0].copy()
        return np.concatenate(doc_groups).astype(np.int32, copy=False)

    def score_documents(
        self,
        query: np.ndarray,
        doc_indices: np.ndarray,
    ) -> np.ndarray:
        query_vec = np.asarray(query, dtype=np.float32)
        query_vec = query_vec / max(float(np.linalg.norm(query_vec)), _EPSILON)
        doc_indices = np.asarray(doc_indices, dtype=np.int32)
        if len(doc_indices) == 0:
            return np.empty(0, dtype=np.float64)
        scores = self.embeddings[doc_indices] @ query_vec
        return np.asarray(scores, dtype=np.float64)

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        nprobe: int | None = None,
    ) -> IVFSearchResult:
        query_vec = np.asarray(query, dtype=np.float32)
        query_vec = query_vec / max(float(np.linalg.norm(query_vec)), _EPSILON)

        if nprobe is None:
            nprobe = self.default_nprobe
        nprobe = max(1, min(int(nprobe), self.n_cells))

        centroid_scores = self.centroids @ query_vec
        if nprobe >= self.n_cells:
            probed_cell_ids = np.arange(self.n_cells, dtype=np.int32)
        else:
            part = np.argpartition(-centroid_scores, nprobe - 1)[:nprobe]
            probed_cell_ids = part[np.argsort(-centroid_scores[part])].astype(
                np.int32,
            )
        probed_cell_scores = np.asarray(
            centroid_scores[probed_cell_ids], dtype=np.float64,
        )

        candidate_indices = self._docs_for_cells(probed_cell_ids)
        candidate_scores = self.score_documents(query_vec, candidate_indices)
        candidate_cell_ids = self.assignments[candidate_indices]
        candidate_cell_populations = self.cell_populations[candidate_cell_ids]

        k_eff = min(max(int(k), 0), len(candidate_indices))
        if k_eff == 0:
            empty_i = np.empty(0, dtype=np.int32)
            empty_f = np.empty(0, dtype=np.float64)
            return IVFSearchResult(
                indices=empty_i,
                scores=empty_f,
                cell_ids=empty_i,
                cell_populations=empty_i,
                candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                candidate_cell_ids=candidate_cell_ids,
                candidate_cell_populations=candidate_cell_populations,
                probed_cell_ids=probed_cell_ids,
                probed_cell_scores=probed_cell_scores,
            )

        if k_eff == len(candidate_indices):
            top_local = np.argsort(-candidate_scores)
        else:
            top_local = np.argpartition(-candidate_scores, k_eff - 1)[:k_eff]
            top_local = top_local[np.argsort(-candidate_scores[top_local])]

        indices = candidate_indices[top_local]
        scores = candidate_scores[top_local]
        cell_ids = candidate_cell_ids[top_local]
        cell_populations = candidate_cell_populations[top_local]

        return IVFSearchResult(
            indices=np.asarray(indices, dtype=np.int32),
            scores=np.asarray(scores, dtype=np.float64),
            cell_ids=np.asarray(cell_ids, dtype=np.int32),
            cell_populations=np.asarray(cell_populations, dtype=np.int32),
            candidate_indices=np.asarray(candidate_indices, dtype=np.int32),
            candidate_scores=np.asarray(candidate_scores, dtype=np.float64),
            candidate_cell_ids=np.asarray(candidate_cell_ids, dtype=np.int32),
            candidate_cell_populations=np.asarray(
                candidate_cell_populations, dtype=np.int32,
            ),
            probed_cell_ids=probed_cell_ids,
            probed_cell_scores=probed_cell_scores,
        )
