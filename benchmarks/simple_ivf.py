#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark-local IVF helper for dense retrieval experiments."""

from __future__ import annotations

import math

import numpy as np

from benchmarks.ivf_search_result import IVFSearchResult

_EPSILON = 1e-12


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, _EPSILON)


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
        cell_residual_means: np.ndarray,
        cell_residual_q90: np.ndarray,
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
        self.cell_residual_means = np.asarray(
            cell_residual_means, dtype=np.float64,
        )
        self.cell_residual_q90 = np.asarray(cell_residual_q90, dtype=np.float64)

        self.n_docs = int(self.embeddings.shape[0])
        self.dim = int(self.embeddings.shape[1])
        self.n_cells = int(self.centroids.shape[0])
        self.cell_populations = np.diff(self.cell_offsets).astype(np.int32)
        self.avg_population = float(np.mean(self.cell_populations))

        # Per-cell sums of the cosine between each member and its
        # centroid, and of its square.  Two numbers per cell are all the
        # index needs to hold to estimate the background of any query
        # (see query_background).
        member_cosine = 1.0 - self.background_distances
        self.cell_cosine_sums = np.bincount(
            self.assignments, weights=member_cosine, minlength=self.n_cells,
        )
        self.cell_cosine_square_sums = np.bincount(
            self.assignments, weights=member_cosine * member_cosine,
            minlength=self.n_cells,
        )

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
        global_residual_mean = float(np.mean(background_distances))
        global_residual_q90 = float(np.percentile(background_distances, 90))

        cell_residual_means = np.full(n_cells, global_residual_mean, dtype=np.float64)
        cell_residual_q90 = np.full(n_cells, global_residual_q90, dtype=np.float64)
        for cell_id in range(n_cells):
            mask = assignments == cell_id
            if not np.any(mask):
                continue
            cell_res = background_distances[mask]
            cell_residual_means[cell_id] = float(np.mean(cell_res))
            cell_residual_q90[cell_id] = float(np.percentile(cell_res, 90))

        default_nprobe = max(1, int(round(math.sqrt(n_cells))))
        return cls(
            embeddings=embeddings,
            centroids=centroids,
            assignments=assignments,
            sorted_doc_ids=order.astype(np.int32),
            cell_offsets=offsets,
            default_nprobe=default_nprobe,
            background_distances=background_distances,
            cell_residual_means=cell_residual_means,
            cell_residual_q90=cell_residual_q90,
        )

    def query_background(
        self,
        result: IVFSearchResult,
        *,
        query_vector: np.ndarray | None = None,
        excluded_indices: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Mean and standard deviation of a query's cosine distance over the corpus.

        The residuals stored at build time (``background_distances``) are
        distances between a document and the centroid of its own cell.
        They are not distances between a query and a document, and their
        distribution is not the background a calibration of query
        distances needs.  This method estimates that background from what
        the index holds and what a search computes anyway, in time
        proportional to the number of cells.

        Write a member ``x`` of a cell with unit centroid ``c`` as
        ``x = a c + sqrt(1 - a^2) u`` with ``a = <x, c>`` and ``u``
        orthogonal to ``c``.  For a query ``q`` with ``s = <q, c>``,

            <q, x> = a s + sqrt(1 - a^2) <q, u>.

        * First moment.  The centroid is the normalized mean of its
          members, so the members sum to ``(sum of a) * c`` and the
          similarities of a cell sum to ``s * (sum of a)`` exactly.
        * Second moment.  The squared similarities of a cell sum to
          ``s^2 * (sum of a^2)`` plus the orthogonal part, modelled as
          ``k * (1 - s^2) * (N_c - sum of a^2)``.  The constant ``k``
          (the reciprocal of an effective dimension) is estimated on the
          probed cells, where the true similarities are known.

        Probed cells enter with their true similarities.

        Parameters
        ----------
        result : IVFSearchResult
            Result of ``search`` for the query.

        Returns
        -------
        (mean, std)
            Estimated mean and standard deviation of the cosine distance
            ``1 - <q, x>`` over all documents.
        """
        centroid_scores = np.asarray(result.centroid_scores, dtype=np.float64)
        scores = np.asarray(result.candidate_scores, dtype=np.float64)
        cells = np.asarray(result.candidate_cell_ids, dtype=np.int64)
        candidate_indices = np.asarray(result.candidate_indices, dtype=np.int64)
        excluded = np.array([], dtype=np.int64)
        if excluded_indices is not None:
            excluded = np.unique(np.asarray(excluded_indices, dtype=np.int64))
            if np.any(excluded < 0) or np.any(excluded >= self.n_docs):
                raise ValueError("excluded_indices must be valid document positions")
            keep = ~np.isin(candidate_indices, excluded)
            scores, cells, candidate_indices = scores[keep], cells[keep], candidate_indices[keep]

        probed = np.zeros(self.n_cells, dtype=bool)
        probed[np.asarray(result.probed_cell_ids, dtype=np.int64)] = True
        unprobed = ~probed

        # Orthogonal variance per unit of (1 - a^2)(1 - s^2), from the
        # probed candidates.
        orthogonal_scale = 0.0
        if len(scores) > 0:
            member_cosine = 1.0 - self.background_distances[candidate_indices]
            predicted = centroid_scores[cells] * member_cosine
            orthogonal = (1.0 - member_cosine**2) * (1.0 - centroid_scores[cells] ** 2)
            denominator = float(np.sum(orthogonal))
            if denominator > _EPSILON:
                orthogonal_scale = float(np.sum((scores - predicted) ** 2)) / denominator

        s = centroid_scores[unprobed]
        first = float(np.sum(scores)) + float(np.sum(s * self.cell_cosine_sums[unprobed]))
        second = float(np.sum(scores * scores)) + float(
            np.sum(
                s * s * self.cell_cosine_square_sums[unprobed]
                + orthogonal_scale
                * (1.0 - s * s)
                * (self.cell_populations[unprobed] - self.cell_cosine_square_sums[unprobed])
            )
        )
        n_total = len(scores) + int(np.sum(self.cell_populations[unprobed]))
        if excluded_indices is not None:
            # Probed exclusions are already absent from result candidates.
            # For unprobed cells remove the actual held-out contribution
            # from the moment model of the full indexed population.
            unscored = excluded[unprobed[self.assignments[excluded]]]
            if unscored.size:
                if query_vector is None:
                    raise ValueError("query_vector is required to exclude unprobed documents")
                vector = np.asarray(query_vector, dtype=np.float64)
                norm = float(np.linalg.norm(vector))
                if vector.shape != (self.dim,) or not np.all(np.isfinite(vector)) or norm <= 0.0:
                    raise ValueError(
                        "query_vector must be a finite nonzero vector of index dimension"
                    )
                excluded_scores = np.asarray(
                    self.embeddings[unscored] @ (vector / norm), dtype=np.float64,
                )
                first -= float(np.sum(excluded_scores))
                second -= float(np.sum(excluded_scores**2))
                n_total -= unscored.size
        if n_total == 0:
            return 1.0, _EPSILON

        mean_similarity = first / n_total
        variance = max(second / n_total - mean_similarity * mean_similarity, 0.0)
        return 1.0 - mean_similarity, max(math.sqrt(variance), _EPSILON)

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
                centroid_scores=np.asarray(centroid_scores, dtype=np.float64),
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
            centroid_scores=np.asarray(centroid_scores, dtype=np.float64),
        )
