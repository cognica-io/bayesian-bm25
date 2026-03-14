#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Sparse signal gating for noisy multi-field search.

When some retrieval signals are unreliable (e.g., metadata matches with
many false positives), gating functions suppress weak/noisy evidence
before fusion. This prevents noisy signals from diluting strong ones.

This example shows:
  1. Gating comparison: none vs relu vs swish vs gelu on mixed signals
  2. Generalized swish: beta controls the gate sharpness
  3. Practical scenario: filtering unreliable signals in hybrid search
"""

import numpy as np

from bayesian_bm25 import log_odds_conjunction

# ---------------------------------------------------------------
# 1. Setup and gating comparison
# ---------------------------------------------------------------
print("=== Gating comparison on mixed signals ===\n")

# One strong, one weak/noisy, one moderate signal
signals = np.array([0.9, 0.3, 0.7])
print(f"  Signals: {signals}  (strong=0.9, weak=0.3, moderate=0.7)\n")

gating_modes = ["none", "relu", "swish", "gelu"]

print(f"  {'Gating':<8}  {'Combined':>10}")
print(f"  {'-'*8}  {'-'*10}")
for mode in gating_modes:
    result = log_odds_conjunction(signals, gating=mode)
    print(f"  {mode:<8}  {result:10.4f}")

# With all-positive-logit signals (all > 0.5), gating has minimal effect
print("\n  When all signals are strong (all > 0.5), gating barely matters:")
strong_signals = np.array([0.9, 0.8, 0.7])
print(f"  Signals: {strong_signals}\n")

print(f"  {'Gating':<8}  {'Combined':>10}")
print(f"  {'-'*8}  {'-'*10}")
for mode in gating_modes:
    result = log_odds_conjunction(strong_signals, gating=mode)
    print(f"  {mode:<8}  {result:10.4f}")

# ---------------------------------------------------------------
# 2. Generalized swish: gating_beta controls sharpness
# ---------------------------------------------------------------
print("\n=== Generalized swish beta ===\n")

print("  beta -> 0:   gate approaches x/2  (soft, nearly linear)")
print("  beta  = 1:   standard swish")
print("  beta -> inf: gate approaches ReLU  (hard threshold)\n")

betas = [0.01, 0.1, 0.5, 1.0, 1.702, 5.0, 100.0]

print(f"  {'Beta':>8}  {'Combined':>10}  {'Note'}")
print(f"  {'-'*8}  {'-'*10}  {'-'*25}")
for beta in betas:
    result = log_odds_conjunction(signals, gating="swish", gating_beta=beta)
    note = ""
    if beta == 0.01:
        note = "~ x/2 (beta -> 0)"
    elif beta == 1.0:
        note = "standard swish"
    elif beta == 1.702:
        note = "~ gelu approximation"
    elif beta == 100.0:
        note = "~ relu (beta -> inf)"
    print(f"  {beta:8.3f}  {result:10.4f}  {note}")

# Verify: swish with beta=1.702 matches gelu
swish_1702 = log_odds_conjunction(signals, gating="swish", gating_beta=1.702)
gelu_result = log_odds_conjunction(signals, gating="gelu")
print(f"\n  swish(beta=1.702) = {swish_1702:.6f}")
print(f"  gelu              = {gelu_result:.6f}")
print(f"  Match? {np.isclose(swish_1702, gelu_result)}")

# ---------------------------------------------------------------
# 3. Practical scenario: noisy metadata in hybrid search
# ---------------------------------------------------------------
print("\n=== Practical: noisy metadata in hybrid search ===\n")

documents = [
    "Exact title match, strong content",           # doc 0
    "Good content, metadata miss",                  # doc 1
    "Strong content, metadata false negative",      # doc 2
    "Moderate content, good metadata",              # doc 3
    "Weak everywhere",                              # doc 4
]

# Three signals per document: BM25, vector similarity, metadata
# Metadata is noisy -- it assigns low scores to relevant docs (false negatives).
# Without gating, these spurious low metadata scores drag down good documents.
doc_signals = np.array([
    #  BM25  vector  metadata
    [0.92,  0.88,   0.80],   # doc 0: strong all around, including metadata
    [0.85,  0.80,   0.15],   # doc 1: strong content, metadata false negative
    [0.78,  0.82,   0.10],   # doc 2: strong content, metadata false negative
    [0.55,  0.60,   0.75],   # doc 3: moderate content, good metadata
    [0.20,  0.22,   0.30],   # doc 4: weak everywhere
])

print("  Signal columns: BM25, Vector, Metadata (noisy)")
print(f"  {'Document':<45} {'BM25':>5} {'Vec':>5} {'Meta':>5}")
print(f"  {'-'*45} {'-'*5} {'-'*5} {'-'*5}")
for doc, sigs in zip(documents, doc_signals):
    print(f"  {doc:<45} {sigs[0]:5.2f} {sigs[1]:5.2f} {sigs[2]:5.2f}")

# Rank with different gating strategies
print(f"\n  {'Document':<45} {'none':>8} {'swish':>8} {'gelu':>8}")
print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")

results = {}
for mode in ["none", "swish", "gelu"]:
    results[mode] = log_odds_conjunction(doc_signals, gating=mode)

for i, doc in enumerate(documents):
    print(
        f"  {doc:<45} "
        f"{results['none'][i]:8.4f} "
        f"{results['swish'][i]:8.4f} "
        f"{results['gelu'][i]:8.4f}"
    )

# Show rankings
print("\n  Rankings (best to worst):")
print(f"  {'Rank':>4}  {'No gating':<45} {'Swish gating':<45} {'GELU gating'}")
rank_none  = np.argsort(-results["none"])
rank_swish = np.argsort(-results["swish"])
rank_gelu  = np.argsort(-results["gelu"])

for rank in range(len(documents)):
    i_n = rank_none[rank]
    i_s = rank_swish[rank]
    i_g = rank_gelu[rank]
    print(
        f"  {rank+1:4d}. "
        f"{documents[i_n]:<45} "
        f"{documents[i_s]:<45} "
        f"{documents[i_g]}"
    )

print("\n  Note: without gating, noisy metadata false negatives (docs 1, 2)")
print("  drag down strong content documents. Gating suppresses these weak")
print("  negative-logit signals, letting content quality drive the ranking.")

# ---------------------------------------------------------------
# 4. Batched processing
# ---------------------------------------------------------------
print("\n=== Batched processing with gating ===\n")

# Shape: (4 documents, 3 signals each)
batch_signals = np.array([
    [0.85, 0.90, 0.75],   # all strong
    [0.90, 0.20, 0.80],   # one weak signal
    [0.40, 0.35, 0.30],   # all below 0.5 (negative logits)
    [0.60, 0.10, 0.55],   # mixed: two above, one far below
])

print(f"  {'Signals':<25} {'none':>8} {'relu':>8} {'swish':>8} {'gelu':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

batch_results = {}
for mode in gating_modes:
    batch_results[mode] = log_odds_conjunction(batch_signals, gating=mode)

for i, sigs in enumerate(batch_signals):
    sig_str = f"[{sigs[0]:.2f}, {sigs[1]:.2f}, {sigs[2]:.2f}]"
    print(
        f"  {sig_str:<25} "
        f"{batch_results['none'][i]:8.4f} "
        f"{batch_results['relu'][i]:8.4f} "
        f"{batch_results['swish'][i]:8.4f} "
        f"{batch_results['gelu'][i]:8.4f}"
    )
