#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: End-to-end document search with calibrated probabilities.

Build a search index, query it, and get back probabilities instead of
opaque BM25 scores.  Uses bm25s as the backend.

Requires: pip install bayesian-bm25[scorer]
"""

import numpy as np

from bayesian_bm25.scorer import BayesianBM25Scorer

# --- Build a corpus ---
corpus_texts = [
    "Python is a popular programming language for data science",
    "Machine learning algorithms learn patterns from data",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing helps computers understand human language",
    "Python libraries like scikit-learn make machine learning accessible",
    "Data visualization tools help explore and present data effectively",
    "Neural networks are inspired by the structure of the human brain",
    "Supervised learning requires labeled training data",
    "Unsupervised learning discovers hidden patterns without labels",
    "Transfer learning reuses pretrained models for new tasks",
]

# Simple whitespace tokenization (in practice, use a proper tokenizer)
corpus_tokens = [text.lower().split() for text in corpus_texts]

# --- Index with default auto-estimated parameters ---
scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
scorer.index(corpus_tokens, show_progress=False)

print(f"Indexed {scorer.num_docs} documents")
print(f"Average document length: {scorer.avgdl:.1f} tokens")

# --- Single query ---
query = "machine learning python"
query_tokens = query.lower().split()

print(f"\nQuery: '{query}'")
print(f"{'Rank':>4}  {'Prob':>8}  Document")
print("-" * 60)

doc_ids, probs = scorer.retrieve([query_tokens], k=5)
for rank, (did, p) in enumerate(zip(doc_ids[0], probs[0]), 1):
    print(f"  {rank:>2}   {p:7.4f}   {corpus_texts[did][:50]}...")

# --- Multiple queries at once ---
queries = [
    "neural networks brain",
    "data visualization",
    "unsupervised learning patterns",
]

print(f"\n{'='*60}")
print("Batch retrieval:")
query_tokens_batch = [q.lower().split() for q in queries]
doc_ids, probs = scorer.retrieve(query_tokens_batch, k=3)

for q, dids, ps in zip(queries, doc_ids, probs):
    print(f"\n  Query: '{q}'")
    for did, p in zip(dids, ps):
        if p > 0:
            print(f"    P={p:.4f}  {corpus_texts[did][:50]}...")

# --- Dense probabilities: score ALL documents ---
print(f"\n{'='*60}")
print("Dense scoring (all documents):")
all_probs = scorer.get_probabilities(query_tokens)

# Sort by probability
ranked = np.argsort(-all_probs)
for did in ranked:
    if all_probs[did] > 0:
        print(f"  Doc {did:>2}: P={all_probs[did]:.4f}  {corpus_texts[did][:50]}...")

# --- Using probabilities for filtering ---
threshold = 0.4
relevant = np.where(all_probs >= threshold)[0]
print(f"\nDocuments with P >= {threshold}: {relevant.tolist()}")
