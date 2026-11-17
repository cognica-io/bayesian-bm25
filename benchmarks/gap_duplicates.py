#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Exposure of gap detection to a duplicate of the query (synthetic).

Remark 4.12 of "From Scores to Evidence": when the corpus holds a copy
of the query, the copy sits at distance zero,
the largest gap in the sorted distances is the one that separates the
copy from every genuine candidate, and gap detection assigns all its
weight to the copy.  The relevant-distance density then collapses to a
single kernel at zero and every genuine candidate loses its posterior.

The script builds 1,000 candidates, three of them relevant, and reports
the posteriors of the relevant documents

* in the automatic mode of ``VectorProbabilityTransform`` (gap
  detection), without and with a copy of the query;
* for the importance-weighted KDE with externally supplied weights,
  without and with the copy;
* in the automatic mode with the copy present but excluded through
  ``duplicate_threshold``.

No external dataset is required.

Usage:
    python -m benchmarks.gap_duplicates
"""

from __future__ import annotations

import argparse

import numpy as np

from bayesian_bm25.vector_probability import VectorProbabilityTransform

N_RELEVANT = 3


def build_candidates(rng: np.random.Generator) -> np.ndarray:
    """Three relevant documents below a bulk of 997 background candidates."""
    relevant = np.array([0.30, 0.33, 0.40])
    background = np.clip(rng.normal(0.80, 0.07, size=997), 0.58, 1.2)
    return np.concatenate([relevant, background])


def external_weights(distances: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Relevance estimates from a second, imperfect signal.

    The relevant documents receive a high weight; the background a low
    one with a few moderately high outliers, as a lexical signal would
    give.
    """
    weights = rng.beta(1.0, 30.0, size=len(distances))
    weights[:N_RELEVANT] = [0.95, 0.85, 0.60]
    return weights


def run(seed: int = 0) -> dict[str, np.ndarray | int]:
    """Posteriors of the relevant documents in each configuration."""
    rng = np.random.default_rng(seed)
    distances = build_candidates(rng)
    weights = external_weights(distances, rng)
    base_rate = N_RELEVANT / len(distances)

    with_copy = np.concatenate([[0.0], distances])
    # The copy is the query's own text, so the second signal rates it
    # as highly as it can.
    weights_with_copy = np.concatenate([[1.0], weights])

    clean = VectorProbabilityTransform.fit_background(
        distances, base_rate=base_rate, background_model="gaussian", background_kind="nonrelevant",
    )
    copied = VectorProbabilityTransform.fit_background(
        with_copy, base_rate=base_rate, background_model="gaussian", background_kind="nonrelevant",
    )

    auto_clean = clean.calibrate(distances, method="auto")
    auto_copy = copied.calibrate(with_copy, method="auto")
    kde_clean = clean.calibrate(distances, weights=weights, method="kde")
    kde_copy = copied.calibrate(with_copy, weights=weights_with_copy, method="kde")
    auto_excluded = clean.calibrate_with_sample(
        distances, with_copy, method="auto", duplicate_threshold=1e-6,
    )

    floor = float(np.min(auto_copy))
    return {
        "auto_clean": auto_clean[:N_RELEVANT],
        "auto_copy": auto_copy[1:1 + N_RELEVANT],
        "auto_copy_at_floor": int(np.sum(auto_copy[1:] == floor)),
        "kde_clean": kde_clean[:N_RELEVANT],
        "kde_copy": kde_copy[1:1 + N_RELEVANT],
        "auto_excluded": auto_excluded[:N_RELEVANT],
    }


def _format(values: np.ndarray) -> str:
    return ", ".join(f"{v:.4g}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap detection and duplicates of the query",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    result = run(args.seed)
    print("Posterior of the three relevant documents (1,000 candidates)")
    print()
    print("  Automatic mode (gap detection)")
    print(f"    no copy of the query        : {_format(result['auto_clean'])}")
    print(f"    copy at distance zero       : {_format(result['auto_copy'])}")
    print(
        "    genuine candidates tied at the numerical floor: "
        f"{result['auto_copy_at_floor']} of 1000"
    )
    print(f"    copy excluded by threshold  : {_format(result['auto_excluded'])}")
    print()
    print("  Importance-weighted KDE, external weights")
    print(f"    no copy of the query        : {_format(result['kde_clean'])}")
    print(f"    copy at distance zero       : {_format(result['kde_copy'])}")


if __name__ == "__main__":
    main()
