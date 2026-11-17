#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""The automatic reference configuration measured in Table 1 of the paper.

The benchmark selects its weights explicitly: gap weights first, otherwise
sharpened external guidance. It uses twice the Silverman bandwidth. These
are the measured reference configuration, not defaults of Definition 4.8.
The separate weight-source and bandwidth ablations supply their own settings.
"""

import numpy as np

from bayesian_bm25.vector_probability import VectorProbabilityTransform


def reference_vector_log_odds(
    transform: VectorProbabilityTransform,
    eval_distances: np.ndarray,
    sample_distances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    gap = transform._gap_weights(sample_distances)
    if gap is not None:
        weights = gap
        method = "kde" if len(sample_distances) >= 50 else "gmm"
    elif transform._signal_mass(weights) > 1e-10:
        weights = transform._sharpen_weights(weights)
        method = "kde"
    else:
        method = "auto"
    return np.asarray(
        transform.calibrate_with_sample(
            eval_distances,
            sample_distances,
            weights=weights,
            method=method,
            bandwidth_factor=2.0,
            return_log_odds=True,
        ),
        dtype=np.float64,
    )
