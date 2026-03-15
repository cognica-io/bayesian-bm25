# Bayesian BM25

[[Blog](https://www.cognica.io/en/blog/posts/2026-02-01-bayesian-bm25-hybrid-search)] [[Papers](docs/papers)]

The reference implementation of the [Bayesian BM25](https://doi.org/10.5281/zenodo.18414940) and [From Bayesian Inference to Neural Computation](https://doi.org/10.5281/zenodo.18512411) papers, by the original author. Converts raw BM25 retrieval scores into calibrated relevance probabilities using Bayesian inference.

## Overview

Standard BM25 produces unbounded scores that lack consistent meaning across queries, making threshold-based filtering and multi-signal fusion unreliable. Bayesian BM25 addresses this by applying a sigmoid likelihood model with a composite prior (term frequency + document length normalization) and computing Bayesian posteriors that output well-calibrated probabilities in [0, 1]. A corpus-level base rate prior further improves calibration by 68–77% without requiring relevance labels.

Key capabilities:

- **Score-to-probability transform** — convert raw BM25 scores into calibrated relevance probabilities via sigmoid likelihood + composite prior + Bayesian posterior
- **Base rate calibration** — corpus-level base rate prior estimated from score distribution (95th percentile, mixture model, or elbow detection) decomposes the posterior into three additive log-odds terms, reducing expected calibration error by 68–77% without relevance labels
- **Parameter learning** — batch gradient descent or online SGD with EMA-smoothed gradients and Polyak averaging, with three training modes: balanced (C1), prior-aware (C2), and prior-free (C3)
- **Probabilistic fusion** — combine multiple probability signals using AND, OR, NOT, and log-odds conjunction with multiplicative confidence scaling, optional per-signal reliability weights (Log-OP), and sparse signal gating (ReLU/Swish/GELU/Softplus activations from Paper 2, Theorems 6.5.3/6.7.4/6.8.1/Remark 6.5.4) with generalized beta control (Theorem 6.7.6)
- **Learnable fusion weights** — `LearnableLogOddsWeights` learns per-signal reliability from labeled data via a Hebbian gradient that is backprop-free, starting from Naive Bayes uniform initialization (Remark 5.3.2); supports optional `base_rate` additive bias in log-odds space
- **Attention-based fusion** — `AttentionLogOddsWeights` learns query-dependent signal weights via attention mechanism (Paper 2, Section 8), with exact attention pruning via `compute_upper_bounds()` and `prune()` (Theorem 8.7.1); supports optional `base_rate`
- **Multi-head attention** — `MultiHeadAttentionLogOddsWeights` creates multiple independent attention heads with different initializations and averages their log-odds for more robust fusion (Remark 8.6, Corollary 8.7.2)
- **Neural score calibration** — `PlattCalibrator` (sigmoid) and `IsotonicCalibrator` (PAVA) convert raw neural model scores into calibrated probabilities for Bayesian fusion (Section 12.2 #5)
- **External prior features** — `prior_fn` callable on `BayesianProbabilityTransform` allows custom document priors to replace the composite prior, enabling features like recency or popularity weighting (Section 12.2 #6)
- **Temporal adaptation** — `TemporalBayesianTransform` uses exponential decay to weight recent observations more heavily in `fit()`, tracking concept drift in non-stationary relevance patterns (Section 12.2 #3)
- **Hybrid search** — `cosine_to_probability()` converts vector similarity scores to probabilities for fusion with BM25 signals via weighted log-odds conjunction
- **WAND pruning** — `wand_upper_bound()` computes safe Bayesian probability upper bounds for document pruning in top-k retrieval; `BlockMaxIndex` provides tighter block-level bounds for BMW-style pruning (Section 6.2, Corollary 7.4.2)
- **Calibration metrics** — `expected_calibration_error()`, `brier_score()`, `reliability_diagram()`, and `calibration_report()` for evaluating probability quality, with `CalibrationReport` bundling all metrics into a single diagnostic
- **Fusion debugger** — `FusionDebugger` records every intermediate value through the full pipeline (likelihood, prior, posterior, fusion) for transparent inspection, document comparison, and crossover detection; supports hierarchical fusion tracing with AND/OR/NOT composition and gating trace fields
- **Multi-field search** — `MultiFieldScorer` maintains separate BM25 indexes per field and fuses field-level probabilities via log-odds conjunction with configurable per-field weights
- **Search integration** — drop-in scorer wrapping [bm25s](https://github.com/xhluca/bm25s) that returns probabilities instead of raw scores

## Adoption

- [MTEB](https://github.com/embeddings-benchmark/mteb) — included as a baseline retrieval model (`bb25`) for the Massive Text Embedding Benchmark
- [txtai](https://github.com/neuml/txtai) — used for BM25 score normalization in hybrid search (`normalize="bayesian-bm25"`)
- [UQA](https://cognica-io.github.io/uqa/) — used as the scoring operator for probabilistic text retrieval and multi-signal fusion in the unified query algebra

## Installation

```bash
pip install bayesian-bm25
```

To use the integrated search scorer (requires `bm25s`):

```bash
pip install bayesian-bm25[scorer]
```

## Quick Start

### Converting BM25 Scores to Probabilities

```python
import numpy as np
from bayesian_bm25 import BayesianProbabilityTransform

transform = BayesianProbabilityTransform(alpha=1.5, beta=1.0, base_rate=0.01)

scores = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
tfs = np.array([1, 2, 3, 5, 8])
doc_len_ratios = np.array([0.3, 0.5, 0.8, 1.0, 1.5])

probabilities = transform.score_to_probability(scores, tfs, doc_len_ratios)
```

### End-to-End Search with Probabilities

```python
from bayesian_bm25 import BayesianBM25Scorer

corpus_tokens = [
    ["python", "machine", "learning"],
    ["deep", "learning", "neural", "networks"],
    ["data", "visualization", "tools"],
]

scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene", base_rate="auto")
scorer.index(corpus_tokens, show_progress=False)

doc_ids, probabilities = scorer.retrieve([["machine", "learning"]], k=3)
```

### Multi-Field Search

```python
from bayesian_bm25 import MultiFieldScorer

documents = [
    {"title": ["bayesian", "bm25"], "body": ["probabilistic", "framework", "search"]},
    {"title": ["neural", "networks"], "body": ["deep", "learning", "models"]},
    {"title": ["information", "retrieval"], "body": ["search", "ranking", "relevance"]},
]

scorer = MultiFieldScorer(
    fields=["title", "body"],
    field_weights={"title": 0.4, "body": 0.6},
    k1=1.2, b=0.75, method="lucene",
)
scorer.index(documents, show_progress=False)
doc_ids, probabilities = scorer.retrieve(["bayesian", "search"], k=3)
```

### Combining Multiple Signals

```python
import numpy as np
from bayesian_bm25 import log_odds_conjunction, prob_and, prob_not, prob_or

signals = np.array([0.85, 0.70, 0.60])

prob_and(signals)                # 0.357 (shrinkage problem)
log_odds_conjunction(signals)    # 0.773 (agreement-aware)

# Exclusion query: "python AND NOT java"
p_python, p_java = 0.90, 0.75
prob_and(np.array([p_python, prob_not(p_java)]))  # 0.225
```

### Hybrid Text + Vector Search

```python
import numpy as np
from bayesian_bm25 import cosine_to_probability, log_odds_conjunction

# BM25 probabilities (from Bayesian BM25)
bm25_probs = np.array([0.85, 0.60, 0.40])

# Vector search cosine similarities -> probabilities
cosine_scores = np.array([0.92, 0.35, 0.70])
vector_probs = cosine_to_probability(cosine_scores)  # [0.96, 0.675, 0.85]

# Fuse with reliability weights (BM25 weight=0.6, vector weight=0.4)
stacked = np.stack([bm25_probs, vector_probs], axis=-1)
fused = log_odds_conjunction(stacked, weights=np.array([0.6, 0.4]))

# Fuse with weights and confidence scaling (alpha + weights compose)
fused = log_odds_conjunction(stacked, alpha=0.5, weights=np.array([0.6, 0.4]))

# Gated fusion: ReLU/Swish/GELU/Softplus activation in logit space (Paper 2, Theorems 6.5-6.8)
fused_relu = log_odds_conjunction(stacked, gating="relu")         # MAP estimation
fused_swish = log_odds_conjunction(stacked, gating="swish")       # Bayes estimation
fused_gelu = log_odds_conjunction(stacked, gating="gelu")         # Gaussian noise model
fused_softplus = log_odds_conjunction(stacked, gating="softplus") # evidence-preserving

# Generalized beta controls gate sharpness (Theorem 6.7.6)
# beta -> 0: x/2 (maximum ignorance), beta=1: standard form, beta -> inf: ReLU
fused_soft = log_odds_conjunction(stacked, gating="swish", gating_beta=0.5)

# Softplus for small datasets: preserves all evidence (Remark 6.5.4)
# softplus(x) > x for all finite x, so consider lower alpha to compensate
fused_sp = log_odds_conjunction(stacked, gating="softplus", gating_beta=2.0)
```

### Learning Fusion Weights from Data

```python
import numpy as np
from bayesian_bm25 import LearnableLogOddsWeights

# 3 retrieval signals: BM25, vector search, metadata match
learner = LearnableLogOddsWeights(n_signals=3, alpha=0.0)
# Initial weights are uniform: [0.333, 0.333, 0.333]

# Batch fit from labeled data (probs: m x 3, labels: m)
learner.fit(training_probs, training_labels, learning_rate=0.1)
# Learned weights reflect signal reliability: [0.70, 0.19, 0.11]

# Online refinement from streaming feedback
for probs, label in feedback_stream:
    learner.update(probs, label, learning_rate=0.05, momentum=0.9)

# Inference with Polyak-averaged weights for stability
fused = learner(test_probs, use_averaged=True)
```

### Attention-Based Fusion

```python
import numpy as np
from bayesian_bm25 import AttentionLogOddsWeights

# 2 retrieval signals, 3 query features, per-signal logit normalization
attn = AttentionLogOddsWeights(
    n_signals=2, n_query_features=3, alpha=0.5, normalize=True,
)

# Train on labeled data with query features
# training_probs: (m, 2), training_labels: (m,), query_features: (m, 3)
attn.fit(training_probs, training_labels, query_features,
         learning_rate=0.01, max_iterations=500)

# Query-dependent fusion: weights adapt per query
fused = attn(test_probs, test_features, use_averaged=True)
```

### Multi-Head Attention Fusion

```python
import numpy as np
from bayesian_bm25 import MultiHeadAttentionLogOddsWeights

# 4 heads, 2 signals, 3 query features
mh = MultiHeadAttentionLogOddsWeights(
    n_heads=4, n_signals=2, n_query_features=3, alpha=0.5,
)

# Train all heads on the same data (different init -> different solutions)
mh.fit(training_probs, training_labels, query_features,
       learning_rate=0.01, max_iterations=500)

# Inference: heads produce fused log-odds independently, then average + sigmoid
fused = mh(test_probs, test_features, use_averaged=True)

# Attention pruning: safely eliminate candidates below a threshold
surviving_idx, fused_probs = mh.prune(
    candidate_probs, query_features, threshold=0.5,
    upper_bound_probs=candidate_upper_bounds,
)
```

### Neural Score Calibration

```python
from bayesian_bm25.calibration import PlattCalibrator, IsotonicCalibrator
from bayesian_bm25 import log_odds_conjunction

# Platt scaling: P = sigmoid(a * score + b)
platt = PlattCalibrator()
platt.fit(neural_scores, labels, learning_rate=0.01, max_iterations=1000)
calibrated = platt.calibrate(new_scores)  # output in (0, 1)

# Isotonic regression: non-parametric monotone mapping via PAVA
iso = IsotonicCalibrator()
iso.fit(neural_scores, labels)
calibrated = iso.calibrate(new_scores)

# Combine calibrated neural scores with BM25 probabilities
stacked = np.stack([bm25_probs, calibrated], axis=-1)
fused = log_odds_conjunction(stacked)
```

### Temporal Adaptation

```python
from bayesian_bm25.probability import TemporalBayesianTransform

# Short half-life: adapt quickly to changing relevance patterns
transform = TemporalBayesianTransform(
    alpha=1.0, beta=0.0, decay_half_life=100.0,
)

# Batch fit with timestamps: recent data gets more weight
transform.fit(scores, labels, timestamps=timestamps)

# Online update: timestamp auto-increments, Polyak decay reduces over time
for score, label in feedback_stream:
    transform.update(score, label)
```

### WAND Pruning with Bayesian Upper Bounds

```python
from bayesian_bm25 import BayesianProbabilityTransform

transform = BayesianProbabilityTransform(alpha=1.5, beta=2.0, base_rate=0.01)

# Standard BM25 upper bound per query term
bm25_upper_bound = 5.0

# Bayesian upper bound for safe pruning — any document's actual
# probability is guaranteed to be at most this value
bayesian_bound = transform.wand_upper_bound(bm25_upper_bound)
```

### Debugging the Fusion Pipeline

```python
from bayesian_bm25 import BayesianProbabilityTransform
from bayesian_bm25.debug import FusionDebugger

transform = BayesianProbabilityTransform(alpha=0.45, beta=6.10, base_rate=0.02)
debugger = FusionDebugger(transform)

# Trace a single document through the full pipeline
trace = debugger.trace_document(
    bm25_score=8.42, tf=5, doc_len_ratio=0.60,
    cosine_score=0.74, doc_id="doc-42",
)
print(debugger.format_trace(trace))

# Compare two documents to see which signal drove the rank difference
trace_a = debugger.trace_document(bm25_score=8.42, tf=5, doc_len_ratio=0.60, cosine_score=0.74)
trace_b = debugger.trace_document(bm25_score=5.10, tf=2, doc_len_ratio=1.20, cosine_score=0.88)
comparison = debugger.compare(trace_a, trace_b)
print(debugger.format_comparison(comparison))

# Hierarchical fusion: AND(OR(title, body), vector, NOT(spam))
step1 = debugger.trace_fusion([0.85, 0.70], names=["title", "body"], method="prob_or")
step2 = debugger.trace_not(0.90, name="spam")
step3 = debugger.trace_fusion(
    [step1.fused_probability, 0.80, step2.complement],
    names=["OR(title,body)", "vector", "NOT(spam)"],
    method="prob_and",
)
```

### Evaluating Calibration Quality

```python
import numpy as np
from bayesian_bm25 import (
    expected_calibration_error, brier_score, reliability_diagram, calibration_report,
)

probabilities = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.2])
labels = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

ece = expected_calibration_error(probabilities, labels)   # lower is better
bs = brier_score(probabilities, labels)                   # lower is better
bins = reliability_diagram(probabilities, labels, n_bins=5)  # (avg_pred, avg_actual, count)

# One-call diagnostic report
report = calibration_report(probabilities, labels)
print(report.summary())   # formatted text with ECE, Brier, and reliability table
```

### Online Learning from User Feedback

```python
from bayesian_bm25 import BayesianProbabilityTransform

transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)

# Batch warmup on historical data
transform.fit(historical_scores, historical_labels)

# Online refinement from live feedback
for score, label in feedback_stream:
    transform.update(score, label, learning_rate=0.01, momentum=0.95)

# Use Polyak-averaged parameters for stable inference
alpha = transform.averaged_alpha
beta = transform.averaged_beta
```

### Training Modes

```python
from bayesian_bm25 import BayesianProbabilityTransform

transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)

# C1 (balanced, default): train on sigmoid likelihood
transform.fit(scores, labels, mode="balanced")

# C2 (prior-aware): train on full Bayesian posterior
transform.fit(scores, labels, mode="prior_aware", tfs=tfs, doc_len_ratios=ratios)

# C3 (prior-free): train on likelihood, inference uses prior=0.5
transform.fit(scores, labels, mode="prior_free")
```

## Benchmarks

### BEIR Hybrid Search

Evaluated on 5 [BEIR](https://github.com/beir-cellar/beir) datasets using the retrieve-then-evaluate protocol (top-1000 per signal, union candidates, pytrec_eval). Dense encoder: all-MiniLM-L6-v2. BM25: k1=1.2, b=0.75, Lucene variant with Snowball English stemmer.

#### NDCG@10

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| BM25 | 36.16 | 25.32 | 31.85 | 15.65 | 67.91 | 35.38 |
| Dense | 36.98 | 36.87 | 31.59 | 21.64 | 64.51 | 38.32 |
| Convex | 40.03 | 37.10 | 35.61 | 19.65 | 73.38 | 41.15 |
| RRF | 39.61 | 36.85 | 34.43 | 20.09 | 71.43 | 40.48 |
| Bayesian-OR | 0.06 | 25.54 | 33.47 | 15.88 | 66.97 | 28.38 |
| Bayesian-LogOdds | 37.16 | 32.93 | 35.32 | 18.54 | 72.68 | 39.33 |
| LO-Local | 39.63 | 37.20 | 34.10 | 19.50 | 73.81 | 40.85 |
| Bayesian-LO-BR | 37.16 | 32.92 | 30.92 | 18.50 | 72.15 | 38.33 |
| **Bayesian-Balanced** | **37.27** | **40.59** | **35.73** | **21.40** | **72.47** | **41.50** |
| Balanced-Mix | 37.29 | 40.66 | 35.70 | 21.52 | 72.33 | 41.50 |
| Balanced-Elbow | 37.29 | 40.59 | 35.76 | 21.41 | 72.46 | 41.50 |
| Gated-ReLU | 35.17 | 27.52 | 32.42 | 17.07 | 69.02 | 36.24 |
| Gated-Swish | 36.20 | 27.39 | 28.67 | 16.80 | 68.63 | 35.54 |
| Gated-GELU | 36.06 | 27.62 | 30.86 | 17.07 | 69.29 | 36.18 |
| Gated-Swish-B2 | 35.56 | 27.65 | 31.35 | 17.12 | 69.27 | 36.19 |
| Attention | 37.05 | 38.86 | 34.40 | 21.05 | 70.44 | 40.36 |
| **Attn-NR** | **37.21** | **40.43** | **35.43** | **21.91** | **73.22** | **41.64** |
| Attn-NR-CV | 37.23 | 40.45 | 35.33 | 21.97 | 72.32 | 41.46 |
| Multi-Head | 37.04 | 39.27 | 34.32 | 21.17 | 70.28 | 40.42 |
| MH-NR | 37.13 | 39.08 | 35.72 | 21.78 | 70.60 | 40.86 |
| MultiField | 7.40 | -- | 31.16 | 15.68 | 59.93 | 28.55\* |
| MF-Balanced | 38.42 | -- | 34.51 | 20.92 | 66.84 | 40.17\* |

#### MAP@10

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| BM25 | 23.84 | 19.11 | 11.77 | 9.16 | 63.22 | 25.42 |
| Dense | 24.46 | 29.14 | 11.05 | 12.94 | 59.59 | 27.44 |
| Convex | 26.77 | 29.21 | 13.46 | 11.78 | 69.13 | 30.07 |
| RRF | 26.30 | 28.85 | 12.84 | 11.96 | 66.58 | 29.30 |
| Bayesian-OR | 0.03 | 19.10 | 12.41 | 9.19 | 61.72 | 20.49 |
| Bayesian-LogOdds | 24.54 | 25.58 | 13.40 | 11.01 | 68.15 | 28.54 |
| LO-Local | 26.40 | 29.32 | 12.31 | 11.69 | 69.30 | 29.80 |
| Bayesian-LO-BR | 24.54 | 25.58 | 11.47 | 10.98 | 67.66 | 28.05 |
| **Bayesian-Balanced** | **24.61** | **32.74** | **13.80** | **12.83** | **68.03** | **30.40** |
| Balanced-Mix | 24.62 | 32.77 | 13.79 | 12.92 | 67.84 | 30.39 |
| Balanced-Elbow | 24.62 | 32.74 | 13.80 | 12.84 | 68.02 | 30.40 |
| Gated-ReLU | 22.98 | 20.97 | 11.66 | 10.01 | 64.11 | 25.95 |
| Gated-Swish | 23.87 | 20.88 | 10.23 | 9.85 | 63.81 | 25.73 |
| Gated-GELU | 23.78 | 21.04 | 10.87 | 10.02 | 64.44 | 26.03 |
| Gated-Swish-B2 | 23.42 | 21.07 | 11.11 | 10.05 | 64.42 | 26.01 |
| Attention | 24.49 | 30.93 | 12.68 | 12.59 | 65.80 | 29.30 |
| **Attn-NR** | **24.56** | **32.52** | **13.40** | **13.20** | **68.77** | **30.49** |
| Attn-NR-CV | 24.58 | 32.50 | 13.38 | 13.23 | 67.71 | 30.28 |
| Multi-Head | 24.48 | 31.33 | 12.66 | 12.69 | 65.59 | 29.35 |
| MH-NR | 24.53 | 31.19 | 13.79 | 13.08 | 66.14 | 29.75 |
| MultiField | 4.76 | -- | 11.45 | 9.04 | 55.18 | 20.11\* |
| MF-Balanced | 25.47 | -- | 13.04 | 12.57 | 63.22 | 28.57\* |

#### Recall@10

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| BM25 | 75.18 | 31.98 | 14.47 | 16.36 | 80.78 | 43.75 |
| Dense | 76.53 | 44.13 | 15.50 | 23.09 | 78.33 | 47.52 |
| Convex | 81.72 | 45.04 | 17.06 | 20.60 | 84.89 | 49.86 |
| RRF | 81.65 | 45.03 | 16.87 | 21.11 | 84.76 | 49.88 |
| Bayesian-OR | 0.14 | 32.74 | 15.98 | 16.74 | 81.37 | 29.39 |
| Bayesian-LogOdds | 77.03 | 40.67 | 17.24 | 19.36 | 84.96 | 47.85 |
| LO-Local | 81.37 | 45.24 | 16.29 | 20.40 | 86.22 | 49.90 |
| Bayesian-LO-BR | 77.03 | 40.67 | 15.00 | 19.28 | 84.29 | 47.26 |
| **Bayesian-Balanced** | **77.31** | **47.66** | **17.23** | **22.59** | **84.83** | **49.93** |
| Balanced-Mix | 77.38 | 47.61 | 17.26 | 22.71 | 84.83 | 49.96 |
| Balanced-Elbow | 77.38 | 47.66 | 17.24 | 22.61 | 84.83 | 49.95 |
| Gated-ReLU | 74.04 | 34.39 | 16.00 | 17.76 | 82.58 | 44.95 |
| Gated-Swish | 75.32 | 34.21 | 13.88 | 17.40 | 81.91 | 44.55 |
| Gated-GELU | 75.04 | 34.62 | 14.87 | 17.72 | 82.64 | 44.98 |
| Gated-Swish-B2 | 74.11 | 34.65 | 15.16 | 17.80 | 82.64 | 44.87 |
| Attention | 76.74 | 46.76 | 17.10 | 22.23 | 83.04 | 49.17 |
| **Attn-NR** | **77.24** | **47.38** | **17.05** | **23.26** | **85.03** | **49.99** |
| Attn-NR-CV | 77.24 | 47.53 | 16.98 | 23.42 | 84.71 | 49.98 |
| Multi-Head | 76.74 | 47.02 | 17.08 | 22.37 | 83.04 | 49.25 |
| MH-NR | 76.96 | 46.49 | 17.26 | 23.20 | 83.00 | 49.38 |
| MultiField | 16.43 | -- | 14.64 | 16.68 | 72.87 | 30.16\* |
| MF-Balanced | 79.30 | -- | 16.85 | 22.03 | 76.63 | 48.70\* |

\*MultiField/MF-Balanced average over 4 datasets (FiQA corpus lacks title field).

All methods above are zero-shot (no relevance labels required). With `--tune`, additional supervised methods are evaluated:

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | NDCG@10 Avg |
|---|---|---|---|---|---|---|
| **Balanced-Tuned** | **37.29** | **40.49** | **35.65** | **22.03** | **72.70** | **41.63** |
| Hybrid-AND-Tuned | 37.13 | 28.37 | 34.44 | 16.82 | 69.34 | 37.22 |
| Bayesian-Tuned | 0.79 | 24.76 | 32.11 | 15.68 | 67.67 | 28.20 |

#### Delta vs BM25 (NDCG@10)

| Method | Type | Delta |
|---|---|---|
| **Attn-NR** | **zero-shot** | **+6.26** |
| Balanced-Elbow | zero-shot | +6.13 |
| Bayesian-Balanced | zero-shot | +6.12 |
| Balanced-Mix | zero-shot | +6.12 |
| Attn-NR-CV | zero-shot | +6.08 |
| Convex | zero-shot | +5.78 |
| MH-NR | zero-shot | +5.48 |
| LO-Local | zero-shot | +5.47 |
| RRF | zero-shot | +5.10 |
| Multi-Head | zero-shot | +5.04 |
| Attention | zero-shot | +4.98 |
| Bayesian-LogOdds | zero-shot | +3.95 |
| Bayesian-LO-BR | zero-shot | +2.95 |
| Dense | zero-shot | +2.94 |
| MF-Balanced | zero-shot | +2.28\* |
| Gated-ReLU | zero-shot | +0.86 |
| Gated-Swish-B2 | zero-shot | +0.82 |
| Gated-GELU | zero-shot | +0.80 |
| Gated-Swish | zero-shot | +0.16 |

\*MF-Balanced delta computed over 4 datasets (FiQA corpus lacks title field).

**Method descriptions:**

| Method | Description |
|---|---|
| BM25 | Sparse retrieval via bm25s (Lucene variant) |
| Dense | Cosine similarity via sentence-transformers |
| Convex | `w * dense_norm + (1-w) * bm25_norm`, w=0.5 |
| RRF | Reciprocal Rank Fusion, `sum(1/(k + rank))`, k=60 |
| Bayesian-OR | Bayesian BM25 probs + cosine probs via `prob_or` |
| Bayesian-LogOdds | Bayesian BM25 probs to logit, dense calibrated via `logit = alpha * (sim - median)`, combined |
| LO-Local | Both raw BM25 and dense calibrated symmetrically via `logit = alpha * (score - median)`, combined |
| Bayesian-LO-BR | Bayesian-LogOdds with base rate prior |
| **Bayesian-Balanced** | `balanced_log_odds_fusion`: Bayesian BM25 probs and dense sims to logit space, min-max normalize each, combine with equal weights |
| Balanced-Mix | Bayesian-Balanced with mixture-model base rate estimation |
| Balanced-Elbow | Bayesian-Balanced with elbow-detection base rate estimation |
| Gated-ReLU | `log_odds_conjunction` with ReLU gating in logit space (Paper 2, Theorem 6.5.3) |
| Gated-Swish | `log_odds_conjunction` with Swish gating in logit space (Paper 2, Theorem 6.7.4) |
| Gated-GELU | `log_odds_conjunction` with GELU gating (Paper 2, Theorem 6.8.1): `logit * sigmoid(1.702 * logit)` |
| Gated-Swish-B2 | Generalized swish with `gating_beta=2.0` (Paper 2, Theorem 6.7.6) |
| Gated-Softplus | `log_odds_conjunction` with softplus gating (Remark 6.5.4): `log(1 + exp(logit)) / 1`, evidence-preserving smooth ReLU for small datasets |
| Attention | Query-dependent signal weighting via `AttentionLogOddsWeights` (Paper 2, Section 8) |
| **Attn-NR** | Attention with per-signal logit normalization (`normalize=True`) and 7 features (sparse + dense + cross-signal) |
| Attn-NR-CV | Attn-NR with 5-fold cross-validation (train/test split per query) |
| Multi-Head | 4-head `MultiHeadAttentionLogOddsWeights`, averages log-odds across heads (Remark 8.6) |
| MH-NR | Multi-head + logit normalization + 7 features (Corollary 8.7.2) |
| MultiField | `MultiFieldScorer` (title + body) with `log_odds_conjunction`, sparse-only |
| MF-Balanced | MultiField probs + dense via `balanced_log_odds_fusion` |
| Balanced-Tuned | Bayesian-Balanced + supervised `BayesianProbabilityTransform.fit()` + grid search over base_rate and fusion_weight |
| Hybrid-AND-Tuned | `log_odds_conjunction` of Bayesian BM25 and dense probs with tuned alpha |
| Bayesian-Tuned | Sparse-only Bayesian BM25 with tuned alpha, beta, and base_rate (no dense signal) |

**Why include underperforming methods?** The tables above deliberately include methods that underperform BM25. Each failure mode is informative:

- **Bayesian-OR** (NDCG@10 avg 28.38) — Probabilistic OR assumes signal independence and catastrophically fails on ArguAna (0.06%). This demonstrates *why* the log-odds conjunction framework (Paper 2, Section 4) is needed: naive probability combination without logit-space calibration collapses when signal distributions differ.
- **Gated-ReLU / Gated-Swish / Gated-GELU / Gated-Swish-B2 / Gated-Softplus** — Sparse gating (Paper 2, Theorems 6.5.3 / 6.7.4 / 6.8.1 / 6.7.6 / Remark 6.5.4) is too aggressive for the BEIR hybrid fusion task. ReLU zeros out negative logits entirely, discarding useful weak signals; Swish, GELU, and generalized Swish soften the gate but still suppress too much. Softplus preserves all evidence but inflates logits, which is better suited for small datasets with many signals rather than a two-signal setting. These gates are designed for high-dimensional signal spaces where most inputs are noise — in a two-signal (sparse + dense) setting, there is no noise to suppress.
- **MultiField** (28.58 over 4 datasets) — Sparse-only multi-field search loses to concatenated BM25 because field separation fragments term statistics (smaller per-field document frequency, shorter effective document lengths). However, **MF-Balanced** (40.17) recovers most of the gap by fusing with dense embeddings, confirming that field-level BM25 signals are complementary to dense vectors even when they are individually weaker.

Reproduce:
```bash
# Zero-shot (23 methods)
python benchmarks/hybrid_beir.py -d <beir-data-dir>

# With tuning (auto-estimation + supervised learning + grid search)
python benchmarks/hybrid_beir.py -d <beir-data-dir> --tune

# Download BEIR datasets automatically
python benchmarks/hybrid_beir.py -d <beir-data-dir> --download
```

Requires `pip install bayesian-bm25[scorer] sentence-transformers pytrec-eval-0.5 PyStemmer`.

### Sparse Retrieval

Evaluated on [BEIR](https://github.com/beir-cellar/beir) datasets (NFCorpus, SciFact) with k1=1.2, b=0.75, Lucene BM25. Queries are split 50/50 for training and evaluation. "Batch fit" uses gradient descent on training labels; all other Bayesian methods are unsupervised.

#### Ranking Quality

Base rate prior is a monotonic transform — it does not change document ordering.

| Method | NFCorpus NDCG@10 | NFCorpus MAP | SciFact NDCG@10 | SciFact MAP |
|---|---|---|---|---|
| Raw BM25 | 0.5023 | 0.4395 | 0.5900 | 0.5426 |
| Bayesian (auto) | 0.5050 | 0.4403 | 0.5791 | 0.5283 |
| Bayesian (auto) + base rate | 0.5050 | 0.4403 | 0.5791 | 0.5283 |
| Bayesian (batch fit) | 0.5041 | 0.4400 | 0.5826 | 0.5305 |
| Bayesian (batch fit) + base rate | 0.5041 | 0.4400 | 0.5826 | 0.5305 |
| Platt scaling | 0.0229 | 0.0165 | 0.0000 | 0.0000 |
| Min-max normalization | 0.5023 | 0.4395 | 0.5900 | 0.5426 |
| Batch fit (prior-aware, C2) | 0.5066 | 0.4424 | 0.5776 | 0.5236 |
| Batch fit (prior-free, C3) | 0.5023 | 0.4395 | 0.5880 | 0.5389 |

#### Probability Calibration

Expected Calibration Error (ECE) and Brier score. Lower is better.

| Method | NFCorpus ECE | NFCorpus Brier | SciFact ECE | SciFact Brier |
|---|---|---|---|---|
| Bayesian (no base rate) | 0.6519 | 0.4667 | 0.7989 | 0.6635 |
| Bayesian (base_rate=auto) | 0.1461 (-77.6%) | 0.0619 | 0.2577 (-67.7%) | 0.1308 |
| Bayesian (base_rate=0.001) | 0.0081 (-98.8%) | 0.0114 | 0.0354 (-95.6%) | 0.0157 |
| Batch fit (no base rate) | 0.0093 (-98.6%) | 0.0114 | 0.0103 (-98.7%) | 0.0051 |
| Batch fit + base_rate=auto | 0.0085 (-98.7%) | 0.0096 | 0.0021 (-99.7%) | 0.0013 |
| Platt scaling | 0.0186 (-97.1%) | 0.0101 | 0.0188 (-97.7%) | 0.0007 |
| Min-max normalization | 0.0189 (-97.1%) | 0.0105 | 0.0156 (-98.0%) | 0.0009 |
| Batch fit (prior-aware, C2) | 0.0892 (-86.3%) | 0.0439 | 0.1427 (-82.1%) | 0.0802 |
| Batch fit (prior-free, C3) | 0.0029 (-99.6%) | 0.0099 | 0.0058 (-99.3%) | 0.0030 |

#### Threshold Transfer

F1 scores using the best threshold found on training queries, applied to evaluation queries. Smaller gap indicates better generalization.

| Method | NFCorpus Train F1 | NFCorpus Test F1 | SciFact Train F1 | SciFact Test F1 |
|---|---|---|---|---|
| Bayesian (no base rate) | 0.1607 | 0.1511 | 0.3374 | 0.2800 |
| Batch fit (no base rate) | 0.1577 | 0.1405 | 0.2358 | 0.2294 |
| Batch fit + base_rate=auto | 0.1559 | 0.1403 | 0.3316 | 0.3341 |
| Platt scaling | 0.0219 | 0.0193 | 0.0005 | 0.0005 |
| Min-max normalization | 0.1796 | 0.1751 | 0.3526 | 0.3486 |
| Batch fit (prior-aware, C2) | 0.1657 | 0.1539 | 0.3370 | 0.3275 |
| Batch fit (prior-free, C3) | 0.1808 | 0.1758 | 0.2836 | 0.2852 |

Reproduce with `python benchmarks/base_rate.py` (requires `pip install bayesian-bm25[bench]`). The base rate benchmark also includes Platt scaling, min-max normalization, and prior-aware/prior-free training mode comparisons.

Additional benchmarks (no external datasets required):

- `python benchmarks/learnable_weights.py` — learnable weight recovery, fusion quality, online convergence, and timing
- `python benchmarks/weighted_fusion.py` — weighted vs uniform log-odds fusion across noise scenarios
- `python benchmarks/wand_upper_bound.py` — WAND upper bound tightness and skip rate analysis
- `python benchmarks/gating_functions.py` — gating comparison (none/relu/swish/gelu/softplus), beta sensitivity, timing overhead
- `python benchmarks/bmw_upper_bound.py` — BMW block-max vs global WAND tightness, pruning rate, block size sensitivity
- `python benchmarks/neural_calibration.py` — Platt vs isotonic calibration accuracy, hybrid fusion quality, timing
- `python benchmarks/multi_head_attention.py` — multi-head vs single-head quality, pruning safety/efficiency, head diversity

## Citation

If you use this work, please cite the following papers:

```bibtex
@preprint{Jeong2026BayesianBM25,
  author    = {Jeong, Jaepil},
  title     = {Bayesian {BM25}: {A} Probabilistic Framework for Hybrid Text
               and Vector Search},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18414940},
  url       = {https://doi.org/10.5281/zenodo.18414940}
}

@preprint{Jeong2026BayesianNeural,
  author    = {Jeong, Jaepil},
  title     = {From {Bayesian} Inference to Neural Computation: The Analytical
               Emergence of Neural Network Structure from Probabilistic
               Relevance Estimation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18512411},
  url       = {https://doi.org/10.5281/zenodo.18512411}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Copyright (c) 2023-2026 Cognica, Inc.
