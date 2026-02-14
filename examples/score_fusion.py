#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Combining relevance signals from multiple queries or fields.

You have multiple probability estimates for the same document (e.g.,
from title match, body match, and metadata match) and want to combine
them into a single relevance probability.
"""

import numpy as np

from bayesian_bm25 import prob_and, prob_or, log_odds_conjunction

# --- Setup: probabilities from different retrieval signals ---
# A document was scored by three different field matchers:
p_title = 0.85    # high title relevance
p_body = 0.70     # moderate body relevance
p_metadata = 0.60 # some metadata relevance

signals = np.array([p_title, p_body, p_metadata])

# --- Probabilistic AND (product rule) ---
# "Document must be relevant in ALL fields"
# Problem: shrinks quickly with more signals
and_result = prob_and(signals)
print(f"Probabilistic AND:        {and_result:.4f}  (shrinkage: {signals.min():.2f} -> {and_result:.4f})")

# --- Probabilistic OR (complement rule) ---
# "Document is relevant in ANY field"
or_result = prob_or(signals)
print(f"Probabilistic OR:         {or_result:.4f}")

# --- Log-odds conjunction (recommended) ---
# "Document is relevant across fields, with agreement bonus"
# Resolves the shrinkage problem of naive AND
conjunction = log_odds_conjunction(signals)
print(f"Log-odds conjunction:     {conjunction:.4f}")

print(f"\nNote: AND shrinks from {signals.min():.2f} to {and_result:.4f}")
print(f"      Conjunction keeps it at {conjunction:.4f} (agreement amplifies)")

# --- Effect of alpha (agreement bonus strength) ---
print("\nAlpha sensitivity:")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    result = log_odds_conjunction(signals, alpha=alpha)
    print(f"  alpha={alpha:.2f} -> {result:.4f}")

# --- Agreement vs. disagreement ---
print("\nAgreement vs. disagreement behavior:")
cases = [
    ("Both agree (high)",    np.array([0.9, 0.9])),
    ("Both agree (low)",     np.array([0.3, 0.3])),
    ("Strong disagreement",  np.array([0.9, 0.1])),
    ("Mild disagreement",    np.array([0.7, 0.3])),
    ("Uncertain",            np.array([0.5, 0.5])),
]

print(f"  {'Case':<25}  {'AND':>6}  {'OR':>6}  {'Conjunction':>11}")
for label, probs in cases:
    a = prob_and(probs)
    o = prob_or(probs)
    c = log_odds_conjunction(probs)
    print(f"  {label:<25}  {a:6.4f}  {o:6.4f}  {c:11.4f}")

# --- Batch processing: multiple documents, same signals ---
print("\nBatch fusion for 4 documents:")
# Shape: (4 documents, 3 signals each)
doc_signals = np.array([
    [0.9, 0.8, 0.7],  # strong across all
    [0.9, 0.2, 0.1],  # strong title, weak elsewhere
    [0.5, 0.5, 0.5],  # uncertain everywhere
    [0.3, 0.4, 0.2],  # weak across all
])

results = log_odds_conjunction(doc_signals)
for i, (sigs, r) in enumerate(zip(doc_signals, results)):
    print(f"  Doc {i}: signals={sigs} -> combined={r:.4f}")
