# Bayesian BM25

[[Blog](https://www.cognica.io/en/blog/posts/2026-02-01-bayesian-bm25-hybrid-search)] [[Papers](docs/papers)]  

A probabilistic framework that converts raw BM25 retrieval scores into calibrated relevance probabilities using Bayesian inference.

## Overview

Standard BM25 produces unbounded scores that lack consistent meaning across queries, making threshold-based filtering and multi-signal fusion unreliable. Bayesian BM25 addresses this by applying a sigmoid likelihood model with a composite prior (term frequency + document length normalization) and computing Bayesian posteriors that output well-calibrated probabilities in [0, 1].

Key capabilities:

- **Score-to-probability transform** -- convert raw BM25 scores into calibrated relevance probabilities via sigmoid likelihood + composite prior + Bayesian posterior
- **Parameter learning** -- batch gradient descent or online SGD with EMA-smoothed gradients and Polyak averaging
- **Probabilistic fusion** -- combine multiple probability signals using log-odds conjunction, which resolves the shrinkage problem of naive probabilistic AND
- **Search integration** -- drop-in scorer wrapping [bm25s](https://github.com/xhluca/bm25s) that returns probabilities instead of raw scores

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

transform = BayesianProbabilityTransform(alpha=1.5, beta=1.0)

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

scorer = BayesianBM25Scorer(k1=1.2, b=0.75, method="lucene")
scorer.index(corpus_tokens, show_progress=False)

doc_ids, probabilities = scorer.retrieve([["machine", "learning"]], k=3)
```

### Combining Multiple Signals

```python
import numpy as np
from bayesian_bm25 import log_odds_conjunction, prob_and, prob_or

signals = np.array([0.85, 0.70, 0.60])

prob_and(signals)                # 0.357 (shrinkage problem)
log_odds_conjunction(signals)    # 0.773 (agreement-aware)
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
