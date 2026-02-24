#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Learning which retrieval signals to trust.

You have multiple retrieval signals (BM25, vector similarity, metadata)
and labeled relevance data.  Rather than hand-tuning weights, use
LearnableLogOddsWeights to learn per-signal reliability from data.

This example shows:
  1. Batch fit: learn weights from a training set
  2. Online update: refine weights from streaming feedback
  3. Averaged weights: use Polyak-averaged weights for stable inference
"""

import numpy as np

from bayesian_bm25 import LearnableLogOddsWeights, log_odds_conjunction
from bayesian_bm25.probability import sigmoid

# =====================================================================
# Setup: simulate a hybrid search system with 3 retrieval signals
# =====================================================================

rng = np.random.default_rng(42)

# True relevance labels for 200 documents
n_docs = 200
labels = rng.integers(0, 2, size=n_docs).astype(np.float64)

# Signal 0: BM25 -- reliable (low noise)
bm25_logits = np.where(labels == 1, 1.5, -1.5) + rng.normal(0, 0.5, n_docs)
bm25_probs = np.asarray(sigmoid(bm25_logits), dtype=np.float64)

# Signal 1: Vector search -- moderate noise
vector_logits = np.where(labels == 1, 1.5, -1.5) + rng.normal(0, 1.0, n_docs)
vector_probs = np.asarray(sigmoid(vector_logits), dtype=np.float64)

# Signal 2: Metadata match -- noisy
meta_logits = np.where(labels == 1, 1.5, -1.5) + rng.normal(0, 2.0, n_docs)
meta_probs = np.asarray(sigmoid(meta_logits), dtype=np.float64)

# Stack into (n_docs, 3) array
all_probs = np.column_stack([bm25_probs, vector_probs, meta_probs])

print("Simulated 3-signal hybrid search system:")
print("  Signal 0 (BM25):     noise=0.5  (reliable)")
print("  Signal 1 (Vector):   noise=1.0  (moderate)")
print("  Signal 2 (Metadata): noise=2.0  (noisy)")
print(f"  Documents: {n_docs}, Relevant: {int(labels.sum())}")

# =====================================================================
# 1. Batch fit: learn weights from training data
# =====================================================================

print("\n--- Batch Learning ---")

learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
print(f"Initial weights (uniform):  {learner.weights.round(4)}")

learner.fit(all_probs, labels, learning_rate=0.1, max_iterations=2000)
print(f"Learned weights (fit):      {learner.weights.round(4)}")
print("  -> BM25 gets highest weight (most reliable signal)")

# Compare fusion quality
fused_uniform = log_odds_conjunction(
    all_probs, alpha=0.0, weights=np.array([1/3, 1/3, 1/3])
)
fused_learned = np.asarray(learner(all_probs), dtype=np.float64)

mse_uniform = float(np.mean((np.asarray(fused_uniform) - labels) ** 2))
mse_learned = float(np.mean((fused_learned - labels) ** 2))
print(f"\n  MSE (uniform weights): {mse_uniform:.4f}")
print(f"  MSE (learned weights): {mse_learned:.4f}")
print(f"  Improvement:           {(mse_uniform - mse_learned) / mse_uniform * 100:.1f}%")

# =====================================================================
# 2. Online update: refine weights from streaming feedback
# =====================================================================

print("\n--- Online Learning ---")

online_learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
print(f"{'Step':>5}  {'w[BM25]':>8}  {'w[Vec]':>8}  {'w[Meta]':>8}")

for step in range(1, 201):
    idx = rng.integers(0, n_docs)
    online_learner.update(
        all_probs[idx], labels[idx],
        learning_rate=0.05, momentum=0.9,
    )
    if step <= 5 or step % 50 == 0:
        w = online_learner.weights
        print(f"  {step:>3}  {w[0]:>8.4f}  {w[1]:>8.4f}  {w[2]:>8.4f}")

print(f"\nFinal raw weights:      {online_learner.weights.round(4)}")
print(f"Final averaged weights: {online_learner.averaged_weights.round(4)}")
print("  -> Averaged weights are smoother (Polyak averaging)")

# =====================================================================
# 3. Inference with averaged weights
# =====================================================================

print("\n--- Inference ---")
print("Top 5 documents ranked by learned fusion:")

# Use averaged weights for stable inference
fused_scores = np.asarray(
    online_learner(all_probs, use_averaged=True), dtype=np.float64
)
ranked = np.argsort(-fused_scores)

print(f"  {'Rank':>4}  {'BM25':>6}  {'Vec':>6}  {'Meta':>6}  {'Fused':>6}  {'Label':>5}")
for rank, idx in enumerate(ranked[:5], 1):
    print(
        f"  {rank:>4}  {bm25_probs[idx]:>6.3f}  {vector_probs[idx]:>6.3f}  "
        f"{meta_probs[idx]:>6.3f}  {fused_scores[idx]:>6.3f}  "
        f"{'Rel' if labels[idx] == 1 else '---':>5}"
    )

# =====================================================================
# 4. Alpha for confidence scaling
# =====================================================================

print("\n--- Alpha: Confidence Scaling ---")
print("Higher alpha amplifies agreement when multiple signals agree:")
print(f"  {'Alpha':>5}  {'Weights':>24}  {'Fused (agree)':>13}  {'Fused (disagree)':>16}")

agree = np.array([0.8, 0.8, 0.8])
disagree = np.array([0.9, 0.5, 0.2])

for alpha in [0.0, 0.25, 0.5]:
    learner_a = LearnableLogOddsWeights(n_signals=3, alpha=alpha)
    learner_a.fit(all_probs, labels, learning_rate=0.1, max_iterations=2000)
    w = learner_a.weights
    f_agree = learner_a(agree)
    f_disagree = learner_a(disagree)
    w_str = "[" + ", ".join(f"{v:.3f}" for v in w) + "]"
    print(f"  {alpha:>5.2f}  {w_str:>24}  {f_agree:>13.4f}  {f_disagree:>16.4f}")
