#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Result bundle for one IVF query."""

from dataclasses import dataclass

import numpy as np


@dataclass
class IVFSearchResult:
    """Retrieved documents and scored/probed cells used by background estimation."""

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
    centroid_scores: np.ndarray
