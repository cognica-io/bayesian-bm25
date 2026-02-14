#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Multi-field search with probabilistic fusion.

Search across title and body fields separately, then combine
the probabilities using log-odds conjunction to get a unified
relevance score.

Requires: pip install bayesian-bm25[scorer]
"""

import numpy as np

from bayesian_bm25 import log_odds_conjunction
from bayesian_bm25.scorer import BayesianBM25Scorer

# --- Document collection with title and body ---
documents = [
    {
        "title": "introduction to machine learning",
        "body": "machine learning is a branch of artificial intelligence that focuses on building systems that learn from data",
    },
    {
        "title": "deep learning with neural networks",
        "body": "deep learning uses multiple layers of neural networks to model complex patterns in large datasets",
    },
    {
        "title": "python programming basics",
        "body": "python is a versatile programming language used in web development data science and machine learning",
    },
    {
        "title": "data science workflow",
        "body": "a typical data science project involves data collection cleaning analysis visualization and machine learning",
    },
    {
        "title": "natural language processing overview",
        "body": "natural language processing uses machine learning to help computers understand and generate human language",
    },
    {
        "title": "computer vision applications",
        "body": "computer vision applies deep learning to image recognition object detection and video analysis",
    },
]

# --- Build separate indexes for title and body ---
title_tokens = [doc["title"].split() for doc in documents]
body_tokens = [doc["body"].split() for doc in documents]

title_scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
title_scorer.index(title_tokens, show_progress=False)

body_scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
body_scorer.index(body_tokens, show_progress=False)

# --- Query ---
query = "machine learning"
query_tokens = query.split()

title_probs = title_scorer.get_probabilities(query_tokens)
body_probs = body_scorer.get_probabilities(query_tokens)

print(f"Query: '{query}'\n")
print(f"{'Doc':>3}  {'Title P':>8}  {'Body P':>8}  {'Combined':>8}  Title")
print("-" * 70)

combined_probs = np.zeros(len(documents))
for i in range(len(documents)):
    signals = []
    if title_probs[i] > 0:
        signals.append(title_probs[i])
    if body_probs[i] > 0:
        signals.append(body_probs[i])

    if signals:
        combined_probs[i] = log_odds_conjunction(np.array(signals))

    print(
        f"  {i}   {title_probs[i]:7.4f}   {body_probs[i]:7.4f}   "
        f"{combined_probs[i]:7.4f}   {documents[i]['title']}"
    )

# --- Rank by combined probability ---
ranked = np.argsort(-combined_probs)
print(f"\nFinal ranking:")
for rank, did in enumerate(ranked, 1):
    if combined_probs[did] > 0:
        print(f"  {rank}. [{combined_probs[did]:.4f}] {documents[did]['title']}")
