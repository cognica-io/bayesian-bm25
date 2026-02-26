#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Understanding why a document received a particular fused score.

The Bayesian BM25 fusion pipeline transforms raw scores through multiple
stages (likelihood, prior, posterior, fusion) into a single probability.
Each stage collapses information, making it hard to understand *why* a
document ranked where it did.

The FusionDebugger records every intermediate value so you can:
  1. Trace a single document through the full pipeline
  2. Compare two documents to see which signal drove the rank difference
  3. Diagnose cases where signals disagree (crossover)
"""

import numpy as np

from bayesian_bm25 import BayesianProbabilityTransform
from bayesian_bm25.debug import FusionDebugger

# --- Setup ---
# A transform calibrated for a hypothetical corpus.
# base_rate=0.02 means ~2% of documents are relevant to a typical query.
transform = BayesianProbabilityTransform(alpha=0.45, beta=6.10, base_rate=0.02)
debugger = FusionDebugger(transform)

# =========================================================================
# Example 1: Trace a single BM25 signal
# =========================================================================
print("=" * 60)
print("Example 1: Tracing a single BM25 signal")
print("=" * 60)

bm25_trace = debugger.trace_bm25(score=8.42, tf=5, doc_len_ratio=0.60)

print(f"  {'raw_score':<16} = {bm25_trace.raw_score}")
print(f"  {'likelihood':<16} = {bm25_trace.likelihood:.4f}")
print(f"  {'tf_prior':<16} = {bm25_trace.tf_prior:.4f}")
print(f"  {'norm_prior':<16} = {bm25_trace.norm_prior:.4f}")
print(f"  {'composite_prior':<16} = {bm25_trace.composite_prior:.4f}")
print(f"  {'posterior':<16} = {bm25_trace.posterior:.4f}")
print(f"  (alpha={bm25_trace.alpha}, beta={bm25_trace.beta}, base_rate={bm25_trace.base_rate})")

# =========================================================================
# Example 2: Full document trace (BM25 + vector)
# =========================================================================
print()
print("=" * 60)
print("Example 2: Full document trace with format_trace")
print("=" * 60)

doc_trace = debugger.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74,
    doc_id="doc-42",
)
print(debugger.format_trace(doc_trace))

# =========================================================================
# Example 3: Compact summary
# =========================================================================
print()
print("=" * 60)
print("Example 3: One-line summaries for a batch of documents")
print("=" * 60)

documents = [
    {"bm25_score": 8.42, "tf": 5, "doc_len_ratio": 0.60, "cosine_score": 0.74, "doc_id": "doc-42"},
    {"bm25_score": 5.10, "tf": 2, "doc_len_ratio": 1.20, "cosine_score": 0.88, "doc_id": "doc-77"},
    {"bm25_score": 12.0, "tf": 9, "doc_len_ratio": 0.45, "cosine_score": 0.30, "doc_id": "doc-13"},
    {"bm25_score": 3.50, "tf": 1, "doc_len_ratio": 0.90, "cosine_score": 0.55, "doc_id": "doc-99"},
]

traces = []
for doc in documents:
    t = debugger.trace_document(**doc)
    traces.append(t)
    print(debugger.format_summary(t))

# =========================================================================
# Example 4: Compare two documents
# =========================================================================
print()
print("=" * 60)
print("Example 4: Why did doc-42 rank above doc-77?")
print("=" * 60)

comparison = debugger.compare(traces[0], traces[1])
print(debugger.format_comparison(comparison))

# =========================================================================
# Example 5: Crossover -- signals disagree
# =========================================================================
print()
print("=" * 60)
print("Example 5: Crossover detection (BM25 vs Vector disagree)")
print("=" * 60)

# doc-13 has great BM25 but poor vector similarity
# doc-77 has weaker BM25 but strong vector similarity
crossover_cmp = debugger.compare(traces[2], traces[1])
print(debugger.format_comparison(crossover_cmp))

# =========================================================================
# Example 6: Probabilistic AND trace
# =========================================================================
print()
print("=" * 60)
print("Example 6: Probabilistic AND -- all signals must agree")
print("=" * 60)

# "Document must be relevant in BOTH BM25 and vector"
and_trace = debugger.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74, method="prob_and", doc_id="doc-42",
)
print(debugger.format_trace(and_trace))

# =========================================================================
# Example 7: Probabilistic OR trace
# =========================================================================
print()
print("=" * 60)
print("Example 7: Probabilistic OR -- any signal suffices")
print("=" * 60)

# "Document is relevant in EITHER BM25 or vector"
or_trace = debugger.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74, method="prob_or", doc_id="doc-42",
)
print(debugger.format_trace(or_trace))

# =========================================================================
# Example 8: Probabilistic NOT -- P(NONE relevant)
# =========================================================================
print()
print("=" * 60)
print("Example 8: Probabilistic NOT -- P(NONE relevant)")
print("=" * 60)

# prob_not computes prod(1 - p_i): the probability that NONE of the
# signals indicate relevance.  This is the complement of prob_or.
not_trace = debugger.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74, method="prob_not", doc_id="doc-42",
)
print(debugger.format_trace(not_trace))

# Verify: prob_not = 1 - prob_or
print()
print(f"  prob_not  = {not_trace.final_probability:.3f}")
print(f"  prob_or   = {or_trace.final_probability:.3f}")
print(f"  1 - or    = {1 - or_trace.final_probability:.3f}")
print(f"  (prob_not == 1 - prob_or)")

# =========================================================================
# Example 9: Hierarchical fusion -- AND, OR, NOT composed
# =========================================================================
print()
print("=" * 60)
print("Example 9: Hierarchical fusion")
print("=" * 60)

# Scenario: multi-field search with spam filter.
#   - title BM25 match:  P=0.85
#   - body BM25 match:   P=0.70
#   - vector similarity: P=0.80
#   - spam classifier:   P=0.90 (high = likely spam)
#
# Query plan (tree):
#   AND
#   +-- OR(title, body)           "relevant in title OR body"
#   +-- vector                    "semantically similar"
#   +-- NOT(spam)                 "not spam"

p_title = 0.85
p_body = 0.70
p_vector = 0.80
p_spam = 0.90

# --- Step 1: OR(title, body) ---
step1 = debugger.trace_fusion(
    [p_title, p_body],
    names=["title", "body"],
    method="prob_or",
)
print(f"  Step 1: OR(title={p_title}, body={p_body})")
print(f"          = {step1.fused_probability:.3f}")

# --- Step 2: NOT(spam) ---
step2 = debugger.trace_not(p_spam, name="spam")
print(f"  Step 2: NOT(spam={p_spam})")
print(f"          = {step2.complement:.3f}")

# --- Step 3: AND(step1, vector, step2) ---
step3 = debugger.trace_fusion(
    [step1.fused_probability, p_vector, step2.complement],
    names=["OR(title,body)", "vector", "NOT(spam)"],
    method="prob_and",
)
print(f"  Step 3: AND(OR={step1.fused_probability:.3f},"
      f" vec={p_vector}, NOT_spam={step2.complement:.3f})")
print(f"          = {step3.fused_probability:.3f}")
print()

# Full trace of the final fusion step
print("  Final fusion trace:")
print(f"    [Fusion] method={step3.method}, n={len(step3.signal_probabilities)}")
print(f"             P={[f'{p:.3f}' for p in step3.signal_probabilities]}")
lp = step3.log_probs
print(f"             ln(P)=[{', '.join(f'{v:.3f}' for v in lp)}]")
print(f"             sum(ln(P))={step3.log_prob_sum:.3f}")
print(f"             -> final={step3.fused_probability:.3f}")
print()

# Compare: what if we skip the spam filter?
no_filter = debugger.trace_fusion(
    [step1.fused_probability, p_vector],
    names=["OR(title,body)", "vector"],
    method="prob_and",
)
label_a = "Without spam filter: AND(OR, vec)"
label_b = "With spam filter:    AND(OR, vec, NOT spam)"
label_c = "Spam penalty"
w = max(len(label_a), len(label_b), len(label_c))
print(f"  {label_a:<{w}} = {no_filter.fused_probability:.3f}")
print(f"  {label_b:<{w}} = {step3.fused_probability:.3f}")
print(f"  {label_c:<{w}} = {no_filter.fused_probability - step3.fused_probability:+.3f}")

# =========================================================================
# Example 10: Side-by-side comparison of all fusion methods
# =========================================================================
print()
print("=" * 60)
print("Example 10: Same signals, all fusion methods compared")
print("=" * 60)

for method in ["log_odds", "prob_and", "prob_or", "prob_not"]:
    t = debugger.trace_document(
        bm25_score=8.42, tf=5, doc_len_ratio=0.60,
        cosine_score=0.74, method=method, doc_id="doc-42",
    )
    print(debugger.format_summary(t))

# =========================================================================
# Example 11: Weighted fusion
# =========================================================================
print()
print("=" * 60)
print("Example 11: Weighted log-odds fusion (trusting BM25 more)")
print("=" * 60)

weighted_trace = debugger.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74,
    weights=[0.7, 0.3],
    doc_id="doc-42",
)
print(debugger.format_trace(weighted_trace))

# =========================================================================
# Example 12: Without base_rate
# =========================================================================
print()
print("=" * 60)
print("Example 12: Same document without base_rate correction")
print("=" * 60)

transform_no_br = BayesianProbabilityTransform(alpha=0.45, beta=6.10)
debugger_no_br = FusionDebugger(transform_no_br)

trace_no_br = debugger_no_br.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74, doc_id="doc-42",
)

print(f"  With base_rate=0.02: BM25 posterior = {doc_trace.signals['BM25'].posterior:.4f}")
print(f"  Without base_rate:   BM25 posterior = {trace_no_br.signals['BM25'].posterior:.4f}")
print()
print(f"  With base_rate=0.02: fused = {doc_trace.final_probability:.4f}")
print(f"  Without base_rate:   fused = {trace_no_br.final_probability:.4f}")
print()
print("  base_rate pulls BM25 posteriors toward the corpus prior (2%),")
print("  dramatically lowering them when the raw evidence is moderate.")
