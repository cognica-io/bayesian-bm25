#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Exclusion queries using probabilistic NOT.

You want to find documents relevant to one topic but NOT another.
For example, "machine learning NOT neural networks" retrieves ML
documents while penalizing those about neural networks.

prob_not(p) computes 1 - p (the complement probability).  In log-odds
space this is simple negation: logit(1 - p) = -logit(p), which means
NOT flips the sign of evidence and composes naturally with AND, OR,
and log-odds conjunction.
"""

import numpy as np

from bayesian_bm25 import prob_and, prob_not, prob_or, log_odds_conjunction

# ---------------------------------------------------------------
# 1. Basic NOT operation
# ---------------------------------------------------------------
print("=== Basic prob_not ===\n")

p = 0.85
print(f"  P(relevant)     = {p:.4f}")
print(f"  P(NOT relevant) = {prob_not(p):.4f}")
print(f"  P(NOT 0.50)     = {prob_not(0.50):.4f}  (uncertainty is self-complementary)")
print(f"  NOT(NOT({p}))   = {prob_not(prob_not(p)):.4f}  (double negation = identity)")

# ---------------------------------------------------------------
# 2. Exclusion query: "A AND NOT B"
# ---------------------------------------------------------------
print("\n=== Exclusion query: 'python AND NOT java' ===\n")

# Simulated per-document probabilities for two query terms
documents = [
    "Python web framework with Flask",           # python=high, java=low
    "Java Spring Boot microservices",             # python=low,  java=high
    "Python and Java interop with GraalVM",       # python=high, java=high
    "Rust systems programming",                   # python=low,  java=low
    "Python data analysis with pandas",           # python=high, java=low
]

p_python = np.array([0.90, 0.10, 0.80, 0.05, 0.85])
p_java   = np.array([0.05, 0.92, 0.75, 0.03, 0.08])

# "python AND NOT java" = prob_and([p_python, prob_not(p_java)])
p_exclude = np.array([
    prob_and(np.array([pp, prob_not(pj)]))
    for pp, pj in zip(p_python, p_java)
])

print(f"  {'Document':<45} {'P(py)':>6} {'P(java)':>8} {'P(py AND NOT java)':>19}")
print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*19}")
for doc, pp, pj, pe in zip(documents, p_python, p_java, p_exclude):
    print(f"  {doc:<45} {pp:6.4f} {pj:8.4f} {pe:19.4f}")

ranked = np.argsort(-p_exclude)
print(f"\n  Ranked by 'python AND NOT java':")
for rank, i in enumerate(ranked, 1):
    print(f"    {rank}. [{p_exclude[i]:.4f}] {documents[i]}")

# ---------------------------------------------------------------
# 3. Comparison: AND, OR, AND-NOT, OR-NOT
# ---------------------------------------------------------------
print("\n=== Boolean operation comparison ===\n")

a, b = 0.80, 0.70

results = {
    "A AND B":         prob_and(np.array([a, b])),
    "A OR B":          prob_or(np.array([a, b])),
    "A AND NOT B":     prob_and(np.array([a, prob_not(b)])),
    "A OR NOT B":      prob_or(np.array([a, prob_not(b)])),
    "NOT A AND NOT B": prob_and(np.array([prob_not(a), prob_not(b)])),
    "NOT (A AND B)":   prob_not(prob_and(np.array([a, b]))),
    "NOT (A OR B)":    prob_not(prob_or(np.array([a, b]))),
}

print(f"  A = {a}, B = {b}\n")
for label, value in results.items():
    print(f"  {label:<20} = {value:.4f}")

# ---------------------------------------------------------------
# 4. De Morgan's laws hold for probabilistic operations
# ---------------------------------------------------------------
print("\n=== De Morgan's laws verification ===\n")

not_a_and_b = prob_not(prob_and(np.array([a, b])))
or_not_a_not_b = prob_or(np.array([prob_not(a), prob_not(b)]))
print(f"  NOT(A AND B)       = {not_a_and_b:.6f}")
print(f"  OR(NOT A, NOT B)   = {or_not_a_not_b:.6f}")
print(f"  Equal? {np.isclose(not_a_and_b, or_not_a_not_b)}")

not_a_or_b = prob_not(prob_or(np.array([a, b])))
and_not_a_not_b = prob_and(np.array([prob_not(a), prob_not(b)]))
print(f"\n  NOT(A OR B)        = {not_a_or_b:.6f}")
print(f"  AND(NOT A, NOT B)  = {and_not_a_not_b:.6f}")
print(f"  Equal? {np.isclose(not_a_or_b, and_not_a_not_b)}")

# ---------------------------------------------------------------
# 5. NOT with log-odds conjunction (recommended for exclusion)
# ---------------------------------------------------------------
print("\n=== Log-odds conjunction with exclusion ===\n")

# Log-odds conjunction resolves shrinkage; use it instead of
# naive prob_and for better calibration.
# To exclude a signal, negate it before passing to conjunction.
print(f"  Signals: python={p_python[0]:.2f}, java={p_java[0]:.2f}")
print(f"  Include java:  conjunction([py, java])    = "
      f"{log_odds_conjunction(np.array([p_python[0], p_java[0]])):.4f}")
print(f"  Exclude java:  conjunction([py, NOT java]) = "
      f"{log_odds_conjunction(np.array([p_python[0], prob_not(p_java[0])])):.4f}")

# Batch: negate entire java signal, then conjoin
include_signals = np.stack([p_python, p_java], axis=-1)
exclude_signals = np.stack([p_python, prob_not(p_java)], axis=-1)

include_results = log_odds_conjunction(include_signals)
exclude_results = log_odds_conjunction(exclude_signals)

print(f"\n  {'Document':<45} {'Include java':>13} {'Exclude java':>13}")
print(f"  {'-'*45} {'-'*13} {'-'*13}")
for doc, inc, exc in zip(documents, include_results, exclude_results):
    print(f"  {doc:<45} {inc:13.4f} {exc:13.4f}")
