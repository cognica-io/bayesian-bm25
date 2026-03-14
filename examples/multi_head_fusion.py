#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Multi-head attention fusion with pruning.

When fusing multiple retrieval signals, a single set of attention weights
may not capture all relevant query-signal interactions. Multi-head
attention creates multiple independent weighting patterns and averages
their log-odds, producing more robust fusion.

Attention pruning safely eliminates candidates whose fused probability
cannot exceed a threshold, speeding up re-ranking.

This example shows:
  1. Multi-head vs single-head fusion quality
  2. Attention pruning for efficient re-ranking
  3. Head diversity: different heads learn different patterns
"""

import numpy as np

from bayesian_bm25 import AttentionLogOddsWeights, MultiHeadAttentionLogOddsWeights
from bayesian_bm25.probability import logit, sigmoid

# =====================================================================
# Setup: synthetic 2-signal data (one reliable, one noisy)
# =====================================================================

rng = np.random.RandomState(42)

n_samples = 300
n_query_features = 2
n_signals = 2

# Binary relevance labels
labels = rng.randint(0, 2, size=n_samples).astype(np.float64)

# Query features: 2-dimensional
query_features = rng.randn(n_samples, n_query_features)

# Signal 0: reliable retriever (low noise)
reliable_logits = np.where(labels == 1, 2.0, -2.0) + rng.randn(n_samples) * 0.5
reliable_probs = np.asarray(sigmoid(reliable_logits), dtype=np.float64)

# Signal 1: noisy retriever (high noise)
noisy_logits = np.where(labels == 1, 1.0, -1.0) + rng.randn(n_samples) * 2.0
noisy_probs = np.asarray(sigmoid(noisy_logits), dtype=np.float64)

# Stack into (n_samples, 2) array
all_probs = np.column_stack([reliable_probs, noisy_probs])

# Train/test split: first 200 for training, last 100 for testing
train_probs = all_probs[:200]
train_labels = labels[:200]
train_qf = query_features[:200]

test_probs = all_probs[200:]
test_labels = labels[200:]
test_qf = query_features[200:]

print("Synthetic 2-signal retrieval system:")
print("  Signal 0 (reliable): noise=0.5")
print("  Signal 1 (noisy):    noise=2.0")
print(f"  Samples: {n_samples} (train=200, test=100)")
print(f"  Query features: {n_query_features}-dimensional")
print(f"  Relevant: {int(labels.sum())} / {n_samples}")

# =====================================================================
# 1. Single-head vs multi-head fusion quality
# =====================================================================

print("\n--- Single-Head vs Multi-Head Fusion ---")

# Train single-head attention (1 head)
single_head = AttentionLogOddsWeights(
    n_signals=n_signals,
    n_query_features=n_query_features,
    alpha=0.5,
)
single_head.fit(
    train_probs, train_labels, train_qf,
    learning_rate=0.05, max_iterations=3000,
)

# Train multi-head attention (4 heads)
multi_head = MultiHeadAttentionLogOddsWeights(
    n_heads=4,
    n_signals=n_signals,
    n_query_features=n_query_features,
    alpha=0.5,
)
multi_head.fit(
    train_probs, train_labels, train_qf,
    learning_rate=0.05, max_iterations=3000,
)

# Evaluate BCE loss on test data
single_preds = np.asarray(single_head(test_probs, test_qf), dtype=np.float64)
multi_preds = np.asarray(multi_head(test_probs, test_qf), dtype=np.float64)

eps = 1e-15
single_bce = -float(np.mean(
    test_labels * np.log(np.clip(single_preds, eps, 1.0 - eps))
    + (1.0 - test_labels) * np.log(np.clip(1.0 - single_preds, eps, 1.0 - eps))
))
multi_bce = -float(np.mean(
    test_labels * np.log(np.clip(multi_preds, eps, 1.0 - eps))
    + (1.0 - test_labels) * np.log(np.clip(1.0 - multi_preds, eps, 1.0 - eps))
))

print(f"  Single-head (1 head) BCE loss: {single_bce:.4f}")
print(f"  Multi-head  (4 heads) BCE loss: {multi_bce:.4f}")
if multi_bce < single_bce:
    improvement = (single_bce - multi_bce) / single_bce * 100
    print(f"  -> Multi-head achieves lower loss (improvement: {improvement:.1f}%)")
else:
    print("  -> Single-head achieved lower loss on this run")

# =====================================================================
# 2. Attention pruning for efficient re-ranking
# =====================================================================

print("\n--- Attention Pruning ---")

# Generate 100 candidate documents for pruning demonstration
n_candidates = 100
cand_labels = rng.randint(0, 2, size=n_candidates).astype(np.float64)
cand_reliable = np.where(cand_labels == 1, 2.0, -2.0) + rng.randn(n_candidates) * 0.5
cand_noisy = np.where(cand_labels == 1, 1.0, -1.0) + rng.randn(n_candidates) * 2.0
cand_probs = np.column_stack([
    np.asarray(sigmoid(cand_reliable), dtype=np.float64),
    np.asarray(sigmoid(cand_noisy), dtype=np.float64),
])
cand_qf = rng.randn(1, n_query_features)  # single query

# Upper bounds: assume each signal could be at most its value + margin
upper_margin = 0.15
cand_upper = np.clip(cand_probs + upper_margin, 0.0, 1.0)

# Set a probability threshold
threshold = 0.6

# Prune using multi-head attention
surviving_idx, surviving_fused = multi_head.prune(
    cand_probs, cand_qf, threshold=threshold, upper_bound_probs=cand_upper,
)

# Compute actual fused probabilities for all candidates (ground truth)
all_fused = np.asarray(multi_head(cand_probs, cand_qf), dtype=np.float64)
actual_above = np.where(all_fused >= threshold)[0]

# Check safety: no true positives should be pruned
missed = set(actual_above.tolist()) - set(surviving_idx.tolist())

print(f"  Total candidates:     {n_candidates}")
print(f"  Threshold:            {threshold}")
print(f"  Surviving after prune: {len(surviving_idx)}")
print(f"  Actual above threshold: {len(actual_above)}")
print(f"  Pruned away:          {n_candidates - len(surviving_idx)}")
if len(missed) == 0:
    print("  -> Safety: no true positives were pruned (all retained)")
else:
    print(f"  -> WARNING: {len(missed)} true positive(s) were incorrectly pruned")

# =====================================================================
# 3. Head diversity: different heads learn different patterns
# =====================================================================

print("\n--- Head Diversity ---")
print("Weight matrices (W) learned by each head:")
print(f"  Shape: ({n_signals}, {n_query_features}) per head\n")

weight_matrices = []
for i, head in enumerate(multi_head.heads):
    W = head.weights_matrix
    weight_matrices.append(W)
    print(f"  Head {i}: {np.array2string(W, precision=4, separator=', ')}")

# Measure pairwise differences between heads
print("\nPairwise Frobenius distances between head weight matrices:")
for i in range(len(weight_matrices)):
    for j in range(i + 1, len(weight_matrices)):
        dist = float(np.linalg.norm(weight_matrices[i] - weight_matrices[j]))
        print(f"  Head {i} <-> Head {j}: {dist:.4f}")

# Check that heads are diverse (not all identical)
all_identical = all(
    np.allclose(weight_matrices[0], weight_matrices[k], atol=1e-6)
    for k in range(1, len(weight_matrices))
)
if all_identical:
    print("  -> Heads converged to identical weights (no diversity)")
else:
    print("  -> Heads learned different patterns (diverse)")
