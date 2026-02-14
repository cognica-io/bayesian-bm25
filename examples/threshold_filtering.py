#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Probability-based relevance filtering and bucketing.

Unlike raw BM25 scores, calibrated probabilities have consistent
meaning across queries.  This enables threshold-based filtering
and confidence bucketing without per-query tuning.
"""

import numpy as np

from bayesian_bm25.scorer import BayesianBM25Scorer

# --- Build index ---
corpus_texts = [
    "the quick brown fox jumps over the lazy dog",
    "a fox in the wild hunts for food",
    "the dog sleeps peacefully by the fireplace",
    "quick sort is a fast sorting algorithm",
    "brown sugar is used in many baking recipes",
    "the lazy programmer automated everything",
    "a wild fox was spotted near the river",
    "dogs are loyal companions to humans",
    "the algorithm runs in linear time",
    "baking bread requires patience and practice",
]

corpus_tokens = [text.lower().split() for text in corpus_texts]
scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
scorer.index(corpus_tokens, show_progress=False)

# --- Query and bucket results by confidence ---
query = "fox"
query_tokens = query.lower().split()
probs = scorer.get_probabilities(query_tokens)

HIGH_THRESHOLD = 0.27
LOW_THRESHOLD = 0.20

high_confidence = []
medium_confidence = []
low_confidence = []

for i, p in enumerate(probs):
    if p <= 0:
        continue
    elif p >= HIGH_THRESHOLD:
        high_confidence.append((i, p))
    elif p >= LOW_THRESHOLD:
        medium_confidence.append((i, p))
    else:
        low_confidence.append((i, p))

print(f"Query: '{query}'\n")

if high_confidence:
    print("High confidence (P >= {:.2f}):".format(HIGH_THRESHOLD))
    for did, p in sorted(high_confidence, key=lambda x: -x[1]):
        print(f"  P={p:.4f}  {corpus_texts[did]}")

if medium_confidence:
    print("\nMedium confidence ({:.2f} <= P < {:.2f}):".format(LOW_THRESHOLD, HIGH_THRESHOLD))
    for did, p in sorted(medium_confidence, key=lambda x: -x[1]):
        print(f"  P={p:.4f}  {corpus_texts[did]}")

if low_confidence:
    print("\nLow confidence (P < {:.2f}):".format(LOW_THRESHOLD))
    for did, p in sorted(low_confidence, key=lambda x: -x[1]):
        print(f"  P={p:.4f}  {corpus_texts[did]}")

non_matching = sum(1 for p in probs if p == 0)
print(f"\nNon-matching documents: {non_matching}")

# --- Cross-query comparison ---
# Because probabilities are calibrated, thresholds work across queries
print(f"\n{'='*60}")
print("Cross-query thresholding (same threshold works for all):\n")

queries = ["fox", "dog", "algorithm"]
for q in queries:
    q_tokens = q.lower().split()
    q_probs = scorer.get_probabilities(q_tokens)
    relevant = [(i, p) for i, p in enumerate(q_probs) if p >= LOW_THRESHOLD]
    relevant.sort(key=lambda x: -x[1])

    print(f"  '{q}' -> {len(relevant)} relevant document(s)")
    for did, p in relevant:
        print(f"    P={p:.4f}  {corpus_texts[did][:50]}")
