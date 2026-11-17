#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Count assumptions, source priors, capped probabilities and held-out metrics.

Run from the repository root: python -m examples.evidence_calibration
"""

import numpy as np

from bayesian_bm25 import (
    CalibrationContext,
    calibrate_count,
    calibrate_query,
    calibration_report_from_log_odds,
    fuse_evidence,
    posterior_to_evidence,
    tail_scale,
)


def main() -> None:
    # Every row belongs to this query's reference population. The count
    # and constants are illustrative supplied assumptions, not estimates
    # from these scores or from the requested result limit.
    lexical_scores = np.array([8.0, 5.0, 2.0, 0.0, 0.0])
    similarities = np.array([0.91, 0.77, 0.56, 0.45, 0.32])
    count = 1.2
    context = CalibrationContext(
        reference_population="query_excluded_corpus",
        count_source="supplied", slope_source="convention", ceiling_source="supplied",
        relevance_target="judged_relevance",
    )
    lexical = calibrate_query(
        lexical_scores, match_mask=lexical_scores > 0.0,
        expected_relevant=count, context=context,
    )
    dense = calibrate_query(
        similarities, scale=tail_scale(similarities[:3]), expected_relevant=count, context=context,
    )
    posteriors = np.column_stack([
        lexical.log_odds(lexical_scores), dense.log_odds(similarities),
    ])
    evidence = posterior_to_evidence(
        posteriors, np.array([lexical.prior_log_odds, dense.prior_log_odds]),
    )
    additive = fuse_evidence(evidence, prior_logit=lexical.prior_log_odds)

    # Re-solving a total is a separate modeling step. The ceiling bounds
    # each probability while the new offset preserves the supplied count.
    fused = calibrate_count(additive, count, ceiling=0.6, context=context)
    probabilities = fused.probability(additive)
    np.testing.assert_allclose(probabilities.sum(), count, rtol=1e-9)
    assert np.all(probabilities <= 0.6)
    order = np.argsort(-fused.ranking_score(additive), kind="stable")
    print("Document order:", order.tolist())
    print("Modeled probabilities:", np.round(probabilities, 4).tolist())

    # Labels have a defined target and evaluation population. The result
    # is an empirical diagnostic, separate from count consistency.
    labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    report = calibration_report_from_log_odds(
        fused.log_odds(additive), labels, context=context,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
