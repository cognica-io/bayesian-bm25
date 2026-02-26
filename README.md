# Bayesian BM25

[[Blog](https://www.cognica.io/en/blog/posts/2026-02-01-bayesian-bm25-hybrid-search)] [[Papers](docs/papers)]

The reference implementation of the [Bayesian BM25](https://doi.org/10.5281/zenodo.18414940) and [From Bayesian Inference to Neural Computation](https://doi.org/10.5281/zenodo.18512411) papers, by the original author. Converts raw BM25 retrieval scores into calibrated relevance probabilities using Bayesian inference.

## Overview

Standard BM25 produces unbounded scores that lack consistent meaning across queries, making threshold-based filtering and multi-signal fusion unreliable. Bayesian BM25 addresses this by applying a sigmoid likelihood model with a composite prior (term frequency + document length normalization) and computing Bayesian posteriors that output well-calibrated probabilities in [0, 1]. A corpus-level base rate prior further improves calibration by 68–77% without requiring relevance labels.

Key capabilities:

- **Score-to-probability transform** — convert raw BM25 scores into calibrated relevance probabilities via sigmoid likelihood + composite prior + Bayesian posterior
- **Base rate calibration** — corpus-level base rate prior estimated from score distribution decomposes the posterior into three additive log-odds terms, reducing expected calibration error by 68--77% without relevance labels
- **Parameter learning** — batch gradient descent or online SGD with EMA-smoothed gradients and Polyak averaging, with three training modes: balanced (C1), prior-aware (C2), and prior-free (C3)
- **Probabilistic fusion** — combine multiple probability signals using AND, OR, NOT, and log-odds conjunction with multiplicative confidence scaling and optional per-signal reliability weights (Log-OP), which resolves the shrinkage problem of naive probabilistic AND
- **Learnable fusion weights** — `LearnableLogOddsWeights` learns per-signal reliability from labeled data via a Hebbian gradient that is backprop-free, starting from Naive Bayes uniform initialization (Remark 5.3.2)
- **Hybrid search** — `cosine_to_probability()` converts vector similarity scores to probabilities for fusion with BM25 signals via weighted log-odds conjunction
- **WAND pruning** — `wand_upper_bound()` computes safe Bayesian probability upper bounds for document pruning in top-k retrieval
- **Calibration metrics** — `expected_calibration_error()`, `brier_score()`, and `reliability_diagram()` for evaluating probability quality
- **Fusion debugger** — `FusionDebugger` records every intermediate value through the full pipeline (likelihood, prior, posterior, fusion) for transparent inspection, document comparison, and crossover detection; supports hierarchical fusion tracing with AND/OR/NOT composition
- **Search integration** — drop-in scorer wrapping [bm25s](https://github.com/xhluca/bm25s) that returns probabilities instead of raw scores

## Adoption

- [MTEB](https://github.com/embeddings-benchmark/mteb) — included as a baseline retrieval model (`bb25`) for the Massive Text Embedding Benchmark
- [txtai](https://github.com/neuml/txtai) — used for BM25 score normalization in hybrid search (`normalize="bayesian-bm25"`)

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

### WAND Pruning with Bayesian Upper Bounds

```python
from bayesian_bm25 import BayesianProbabilityTransform

transform = BayesianProbabilityTransform(alpha=1.5, beta=2.0, base_rate=0.01)

# Standard BM25 upper bound per query term
bm25_upper_bound = 5.0

# Bayesian upper bound for safe pruning -- any document's actual
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
from bayesian_bm25 import expected_calibration_error, brier_score, reliability_diagram

probabilities = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.2])
labels = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

ece = expected_calibration_error(probabilities, labels)   # lower is better
bs = brier_score(probabilities, labels)                   # lower is better
bins = reliability_diagram(probabilities, labels, n_bins=5)  # (avg_pred, avg_actual, count)
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

Evaluated on [BEIR](https://github.com/beir-cellar/beir) datasets (NFCorpus, SciFact) with k1=1.2, b=0.75, Lucene BM25. Queries are split 50/50 for training and evaluation. "Batch fit" uses gradient descent on training labels; all other Bayesian methods are unsupervised.

### Ranking Quality

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

### Probability Calibration

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

### Threshold Transfer

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
