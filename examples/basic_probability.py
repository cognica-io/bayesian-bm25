#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Converting raw BM25 scores to calibrated probabilities.

You have BM25 scores from any retrieval system and want to interpret
them as probabilities of relevance.  This example uses the core
transform directly, without any specific search backend.
"""

import numpy as np

from bayesian_bm25 import BayesianProbabilityTransform

# --- Setup ---
# alpha controls sigmoid steepness (higher = sharper transition)
# beta controls the midpoint (scores above beta -> probability > 0.5)
transform = BayesianProbabilityTransform(alpha=1.5, beta=1.0)

# --- Example 1: Single document ---
bm25_score = 2.3
tf = 4            # query term appeared 4 times in the document
doc_len_ratio = 0.8  # document is 80% of average length

prob = transform.score_to_probability(bm25_score, tf, doc_len_ratio)
print(f"BM25 score {bm25_score} -> probability {prob:.4f}")

# --- Example 2: Batch of documents ---
scores = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
tfs = np.array([1, 2, 3, 5, 8])
ratios = np.array([0.3, 0.5, 0.8, 1.0, 1.5])

probs = transform.score_to_probability(scores, tfs, ratios)

print("\nBatch conversion:")
print(f"  {'Score':>6}  {'TF':>3}  {'Ratio':>5}  {'Prob':>8}")
for s, t, r, p in zip(scores, tfs, ratios, probs):
    print(f"  {s:6.2f}  {t:3.0f}  {r:5.2f}  {p:8.4f}")

# --- Example 3: Understanding the components ---
print("\nBreaking down the pipeline for score=2.0, tf=5, ratio=1.0:")
score, tf, ratio = 2.0, 5, 1.0

likelihood = transform.likelihood(score)
prior = transform.composite_prior(tf, ratio)
posterior = transform.posterior(likelihood, prior)

print(f"  Likelihood (sigmoid):   {likelihood:.4f}")
print(f"  Composite prior:        {prior:.4f}")
print(f"  Posterior (calibrated):  {posterior:.4f}")
