# Bayesian BM25

[[Blog](https://www.cognica.io/en/blog/posts/2026-02-01-bayesian-bm25-hybrid-search)] [[Papers](docs/papers)]

The reference implementation of the [Bayesian BM25](https://doi.org/10.5281/zenodo.18414940), [From Bayesian Inference to Neural Computation](https://doi.org/10.5281/zenodo.18512411), and [Vector Scores as Likelihood Ratios](https://doi.org/10.5281/zenodo.19181568) papers, by the original author. Provides Bayesian score models, count-constrained relevance probabilities, and evidence fusion.

The latest paper, *From Scores to Evidence* (November 14, 2026), distinguishes a coherent score model, its identifiability, and empirical probability calibration. Given a slope, a reference population, and an expected relevant count, consistency determines the query intercept. The slope, count, and relevance ceiling require assumptions or judgments. True prior-free likelihood ratios add under conditional independence; normalized Logarithmic Opinion Pooling is a separate operation. The benchmark section reports the two-signal BM25 and all-MiniLM-L6-v2 evaluation, with additional paper experiments documented separately.

The [implementation and reproduction guide](docs/evidence-calibration.md) maps the revised theory to the APIs, population metadata, vector backgrounds, and the 23-table experiment. A complete [evidence calibration example](examples/evidence_calibration.py) shows the source and target priors explicitly.

The [reproduction record](docs/evidence-reproduction.md) reports the new text-identity runs for both encoders and links the generated results. Additional calibration comparisons and cost measurements live under `benchmarks/additional/` with separate result files.

## Overview

Standard BM25 produces scores whose units vary across queries. The recommended query mode sets a slope from each query's score scale and solves an offset so that the probabilities sum to a supplied expected relevant count over a specified population. An optional ceiling redistributes probability from the head while retaining that total. This core model preserves BM25 ordering; document-quality priors can re-rank it. Count consistency alone does not establish calibration against relevance judgments. The corpus mode retains the earlier pseudo-query slope, midpoint, and composite document prior for comparison.

Key capabilities:

- **Score-to-probability transform** -- model relevance probabilities from BM25 scores, an optional document prior, and supplied or learned calibration parameters
- **Per-query calibration** -- `BayesianBM25Scorer(calibration="query")` and `calibrate_query()` set the slope to `kappa / sigma_q`, the evidence per standard deviation of a query's matching scores, and solve the offset from the count constraint `sum_d P(R=1 | s_d) = expected_relevant` ([Jeong, 2026d](https://doi.org/10.5281/zenodo.20768747), Theorem 3.7, Procedure 3.14); the result is a monotone transform of the BM25 score per query, invariant to the scale of the scores, with at most `expected_relevant / p` documents above probability `p`
- **Relevance ceiling** -- `calibrate_count()` and `calibrate_query(..., ceiling=...)` implement `P = ceiling * sigmoid(ranking_score)` with the offset solved at `expected_relevant / ceiling`; posterior log-odds and prior-free evidence describe this capped posterior, while `ranking_score()` preserves ordering at its numerical ceiling
- **Dense tail units** -- `tail_scale(retrieved_similarities, k=...)` measures mean excess over the k-th best score; pass it as `scale` to calibrate the full population or a weighted tail sample, keeping the scale sample separate from the count population
- **Log-odds ranking and fusion** -- `score_to_log_odds()`, `get_log_odds()`, `log_odds_fusion()`, `log_odds_or()` / `log_odds_and()` / `log_odds_not()` and the `from_log_odds` / `return_log_odds` options of the fusion classes keep every step in log-odds; a probability rounds to 1 above about 36.7 in log-odds and is clamped at about 23.0, after which documents tie and their order is decided by tie-breaking
- **Base rate calibration** -- corpus-level base rate prior estimated from score distribution (95th percentile, mixture model, or elbow detection) decomposes the posterior into three additive log-odds terms; scores alone cannot identify the prevalence, so each of these rules adds a shape assumption (see the benchmarks below for how far that goes)
- **Parameter learning** -- batch gradient descent or online SGD with EMA-smoothed gradients and Polyak averaging, with three training modes: balanced (C1), prior-aware (C2), and prior-free (C3)
- **Probabilistic fusion** -- combine multiple probability signals using AND, OR, NOT, and log-odds conjunction with multiplicative confidence scaling, optional per-signal reliability weights (Log-OP), and sparse signal gating (ReLU/Swish/GELU/Softplus activations from [Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Theorems 6.5.3/6.7.4/6.8.1/Remark 6.5.4) with generalized beta control (Theorem 6.7.6)
- **Correlation-aware confidence scaling** -- `effective_signal_count()`, `confidence_exponent()` and `gaussian_evidence_weights()` give the exact scaling and weights for correlated Gaussian evidence ([Jeong, 2026d](https://doi.org/10.5281/zenodo.20768747), Theorem 5.8): `n` equicorrelated signals are worth `n / (1 + (n - 1) r)` independent ones, and the square-root law is the special case `r = 1 / (1 + sqrt(n))`; `solve_offset()` enforces a supplied total after pooling
- **Learnable fusion weights** -- `LearnableLogOddsWeights` learns per-signal reliability from labeled data via a Hebbian gradient that is backprop-free, starting from Naive Bayes uniform initialization (Remark 5.3.2); supports optional `base_rate` additive bias in log-odds space
- **Attention-based fusion** -- `AttentionLogOddsWeights` learns query-dependent signal weights via attention mechanism ([Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Section 8), with exact attention pruning via `compute_upper_bounds()` and `prune()` (Theorem 8.7.1); supports optional `base_rate`
- **Multi-head attention** -- `MultiHeadAttentionLogOddsWeights` creates multiple independent attention heads with different initializations and averages their log-odds for more robust fusion (Remark 8.6, Corollary 8.7.2)
- **Neural score calibration** -- `PlattCalibrator` (sigmoid, fitted by Newton's method on standardized scores, which converges whatever the scale and class balance) and `IsotonicCalibrator` (PAVA) convert raw neural model scores into calibrated probabilities for Bayesian fusion (Section 12.2 #5)
- **External prior features** -- `prior_fn` callable on `BayesianProbabilityTransform` allows custom document priors to replace the composite prior, enabling features like recency or popularity weighting (Section 12.2 #6)
- **Temporal adaptation** -- `TemporalBayesianTransform` uses exponential decay to weight recent observations more heavily in `fit()`, tracking concept drift in non-stationary relevance patterns (Section 12.2 #3)
- **Hybrid search** -- `calibrate_query()` applies a monotone score model to cosine similarities, including tail units and a supplied count; the earlier `cosine_to_probability()` only rescales cosine scores into the unit interval and does not identify relevance probabilities
- **Vector score models** -- `VectorProbabilityTransform` distinguishes a non-relevant background (add the log density ratio to the prior log-odds) from a marginal background (`P = base_rate * relevance_recall * f_R / f_G`). `fit_background()` defaults to a marginal KDE; Gaussian moments are an explicit shape model. `evidence()` removes the prior from the selected posterior. `calibrate_with_sample()` separates the retrieved density sample from evaluation points. External weights take precedence over gap detection, and `duplicate_threshold` excludes copies of the query from density estimation.
- **Index-aware density priors** -- `ivf_density_prior()` and `knn_density_prior()` provide density-based prior weights from IVF cell populations and k-NN distances for informing the vector calibration ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Strategy 4.6.2)
- **WAND pruning** -- `wand_upper_bound()` computes safe Bayesian probability upper bounds for document pruning in top-k retrieval, and `wand_upper_bound_log_odds()` / `QueryCalibration.upper_bound_log_odds()` the same bounds in log-odds, which keep discriminating once probability bounds saturate; `BlockMaxIndex` provides tighter block-level bounds for BMW-style pruning (Section 6.2, Corollary 7.4.2)
- **Calibration metrics** -- `calibration_report_from_log_odds()` and `CalibrationAccumulator` report log-loss skill, count ratio, and calibration slope with an explicit `CalibrationContext`; probability-input ECE, Brier, and reliability reports remain available
- **Fusion debugger** -- `FusionDebugger` records every intermediate value through the full pipeline (likelihood, prior, posterior, fusion) for transparent inspection, document comparison, and crossover detection; supports hierarchical fusion tracing with AND/OR/NOT composition and gating trace fields
- **Multi-field search** -- `MultiFieldScorer` maintains separate BM25 indexes per field and fuses field-level probabilities via log-odds conjunction with configurable per-field weights
- **Search integration** -- drop-in scorer wrapping [bm25s](https://github.com/xhluca/bm25s) that returns probabilities instead of raw scores

## Adoption

- [Apache Lucene](https://github.com/apache/lucene/pull/15827) -- merged as `BayesianScoreQuery` and `LogOddsFusionQuery` for probabilistic hybrid search
- [MTEB](https://github.com/embeddings-benchmark/mteb) -- included as a baseline retrieval model (`bb25`) for the Massive Text Embedding Benchmark
- [txtai](https://github.com/neuml/txtai) -- used for BM25 score normalization in hybrid search (`normalize="bayesian-bm25"`)
- [Vespa.ai](https://github.com/vespa-engine/sample-apps/tree/master/examples/bayesian_bm25) -- adopted as an official sample application
- [UQA](https://cognica-io.github.io/uqa/) -- used as the scoring operator for probabilistic text retrieval and multi-signal fusion in the unified query algebra

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

### Per-Query Calibration

```python
from bayesian_bm25 import BayesianBM25Scorer

corpus_tokens = [
    ["python", "machine", "learning"],
    ["deep", "learning", "neural", "networks"],
    ["data", "visualization", "tools"],
]

scorer = BayesianBM25Scorer(
    k1=1.2, b=0.75, method="lucene",
    calibration="query",    # slope and offset are derived for each query
    kappa=1.0,              # evidence per standard deviation of the query's matching scores
    expected_relevant=1.0,  # the posteriors of a query sum to this
)
scorer.index(corpus_tokens, show_progress=False)

doc_ids, probabilities = scorer.retrieve([["machine", "learning"]], k=3)
# probabilities: [0.672, 0.217, 0.110] -- they sum to expected_relevant

calibration = scorer.calibrate_query(["machine", "learning"])
calibration.slope, calibration.offset, calibration.prevalence
```

The same procedure is available for any scalar score, a cosine similarity included:

```python
import numpy as np
from bayesian_bm25 import calibrate_query

rng = np.random.default_rng(0)
matched_scores = rng.gamma(2.0, 2.0, size=2000)   # BM25 scores of the matching documents

calibration = calibrate_query(
    matched_scores,
    kappa=1.0,
    expected_relevant=3.0,
    population_size=100_000,   # documents that match no query term enter at score zero
)

log_odds = calibration.log_odds(matched_scores)        # rank, prune and fuse on this
evidence = calibration.evidence(matched_scores)        # prior-free log-likelihood ratio
probabilities = calibration.probability(matched_scores)  # report or threshold only

# Scale equivariance: a query whose scores are 20 times larger gets the same posteriors
longer = calibrate_query(20.0 * matched_scores, expected_relevant=3.0, population_size=100_000)
np.allclose(longer.probability(20.0 * matched_scores), probabilities)   # True

# Exact WAND / Block-Max WAND bound in log-odds for a score upper bound
calibration.upper_bound_log_odds(score_upper_bound=25.0)
```

`kappa`, `expected_relevant`, and `ceiling` are not identified by unlabeled scores. The default `ceiling=1` is an assumption. Counts refer to the calibration population, not the number of results requested. In the dilute regime every posterior is proportional to the count.

`CalibrationContext` records the population, parameter sources, and relevance target. A result limit, a SQL match count, and a semantic relevant-document count are different quantities. The nondegenerate model requires `0 < expected_relevant < ceiling * population_size`; an explicit zero count is rejected. A ceiling fitted to BEIR labels targets judged relevance. Interpreting it as a judgment-selection rate additionally requires score-independent selection among relevant documents.

### Count and Ceiling

```python
import numpy as np
from bayesian_bm25 import calibrate_count

# These logits must cover the reference population, or represent it with weights.
population_logits = np.array([12.0, 10.0, 5.0, 1.0, 0.0, -2.0])
calibration = calibrate_count(population_logits, expected_relevant=1.2, ceiling=0.4)
probabilities = calibration.probability(population_logits)
assert np.isclose(probabilities.sum(), 1.2)
assert np.all(probabilities <= 0.4)

ranking_scores = calibration.ranking_score(population_logits)
posterior_log_odds = calibration.log_odds(population_logits)
evidence = calibration.evidence(population_logits)
```

Lowering the ceiling requires re-solving the offset. Multiplying an existing posterior by a ceiling would lower its count. For `ceiling < 1`, even posterior log-odds eventually reach their numerical upper limit; use `ranking_score()` for ordering and `upper_bound_ranking_score()` for pruning. Use `evidence()` when fusing the capped model's prior-free evidence. A ceiling at or below 0.5 has no finite `QueryCalibration.even_odds_score`.

The scorer accepts scalar or per-query overrides without changing its defaults:

```python
doc_ids, probabilities = scorer.retrieve(
    [["machine", "learning"], ["visualization"]], k=3,
    expected_relevant=np.array([1.0, 0.8]), ceiling=np.array([0.6, 0.4]),
)
```

The full-corpus `get_probabilities()`, `get_log_odds()`, and `get_ranking_scores()` methods accept the same scalar query overrides. `retrieve(explain=True)` also exposes `ranking_scores`, `posterior_log_odds`, and the query calibrations separately from the requested output values.

### Similarity Calibration in Tail Units

```python
import numpy as np
from bayesian_bm25 import calibrate_query, tail_scale

population_similarities = np.linspace(-0.2, 0.9, 10000)
retrieved_similarities = np.sort(population_similarities)[-1000:]
scale = tail_scale(retrieved_similarities)
calibration = calibrate_query(
    population_similarities, scale=scale, kappa=1.0,
    expected_relevant=2.0, ceiling=0.8,
)
ranking_scores = calibration.ranking_score(retrieved_similarities)
probabilities = calibration.probability(retrieved_similarities)
```

`tail_scale()` returns zero when fewer than two scores or a tied head leave the slope undefined. The caller must supply a meaningful alternative scale in that case. A top-k list can supply tail units; it cannot supply the unseen population's probability mass. `population_size` with a common unscored score is appropriate only when that score is known, such as a lexical non-match.

### Ranking and Fusing on Log-Odds

```python
import numpy as np
from bayesian_bm25 import (
    BayesianProbabilityTransform, confidence_exponent, log_odds_fusion,
    log_odds_or, solve_offset,
)

transform = BayesianProbabilityTransform(alpha=1.5, beta=1.0, base_rate=0.01)
scores = np.array([30.0, 40.0, 60.0])

transform.score_to_probability(scores, 5.0, 0.5)   # [1 - 1e-10] * 3: tied
transform.score_to_log_odds(scores, 5.0, 0.5)      # [39.55, 54.55, 84.55]: ordered

# Pool prior-free evidence from two signals; add the prior once
evidence = np.array([[2.1, 3.0], [0.4, -1.2], [-3.0, 0.8]])   # (documents, signals)
rho = confidence_exponent(2, correlation=0.38)                # 0.535: close to the square-root law
fused = log_odds_fusion(evidence, alpha=rho, prior_logit=-9.2)

# Post-fusion consistency: one intercept so that the posteriors sum to the expected count
population_evidence = np.random.default_rng(0).normal(0.0, 2.0, size=50_000)
offset = solve_offset(population_evidence, expected_relevant=2.0)
posterior_log_odds = population_evidence - offset

# Boolean operations that stay strictly increasing for strong evidence
log_odds_or(np.array([[40.0, -0.85], [45.0, -0.85]]))   # [40.36, 45.36]; prob_or gives [1.0, 1.0]
```

For exact linear evidence addition use `fuse_evidence(evidence, prior_logit=...)`. Its `coefficients` may be signed and unnormalized, as returned by `gaussian_evidence_weights()`. `evidence_upper_bound(lower, upper, ...)` uses lower bounds for negative coefficients. Reliability weights in `log_odds_fusion()` remain nonnegative and sum to one. Passing posterior logits with `input_prior_log_odds=...` removes each source prior before pooling; an explicit target `prior_logit` is then required. Omitting that argument pools the supplied logits as posterior opinions.

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

# Gated fusion: ReLU/Swish/GELU/Softplus activation in logit space (Jeong, 2026b, Theorems 6.5-6.8)
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

### Vector Score Calibration

```python
import numpy as np
from bayesian_bm25 import VectorProbabilityTransform

# Estimate the marginal density of this query's corpus distances
corpus_distances = np.random.normal(0.8, 0.15, size=10000)
vpt = VectorProbabilityTransform.fit_background(corpus_distances, base_rate=0.01)

# Calibrate query-document distances via likelihood ratio
query_distances = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
probabilities = vpt.calibrate(query_distances)

# With BM25 probability weights for informed density estimation
bm25_probs = np.array([0.85, 0.60, 0.40, 0.20, 0.10])
probabilities = vpt.calibrate(query_distances, weights=bm25_probs)

# Explicit weights take precedence over gap detection and are preserved.
# Sharpen only when the application deliberately chooses that estimator.
sharpened = vpt.calibrate(
    query_distances, weights=bm25_probs, weight_temperature=0.05,
)

# Explicit KDE or GMM estimation
probabilities_kde = vpt.calibrate(query_distances, method="kde")
probabilities_gmm = vpt.calibrate(query_distances, method="gmm")

# Index-aware density priors for IVF / HNSW indexes
from bayesian_bm25 import ivf_density_prior, knn_density_prior

cell_prior = ivf_density_prior(cell_population=150, avg_population=100)
knn_prior = knn_density_prior(kth_distance=0.5, global_median_kth=0.8)

# Use density prior to inform calibration
probabilities = vpt.calibrate(query_distances, density_prior=np.full(5, cell_prior))

# Index-aware calibration: density estimated from local ANN sample,
# probabilities produced for a separate evaluation set
sample_distances = np.array([0.10, 0.15, 0.20, 0.50, 0.75, 0.80, 0.85])
sample_weights = np.array([0.85, 0.8, 0.7, 0.4, 0.2, 0.1, 0.05])
eval_distances = np.array([0.12, 0.30, 0.70])
probabilities = vpt.calibrate_with_sample(
    eval_distances, sample_distances, weights=sample_weights,
    relevance_recall=0.8,  # supplied recall of relevant documents into this range
)

# Evidence agrees with the selected posterior and its actual source prior.
evidence = vpt.evidence_with_sample(
    eval_distances, sample_distances, weights=sample_weights, relevance_recall=0.8,
)

# Posterior log-odds; exact density ratios may reach the estimator's numerical cap
log_odds = vpt.calibrate(query_distances, weights=bm25_probs, return_log_odds=True)

# Exact posterior P_base * f_R / f_G when the background is the marginal of the
# scored population, estimated from a sample of its distances
log_odds = vpt.calibrate(
    query_distances, weights=bm25_probs, method="kde",
    posterior="exact", background_distances=corpus_distances,
    return_log_odds=True,
)

# Rank-preserving variant: evidence projected onto non-increasing functions of distance
log_odds = vpt.calibrate(query_distances, weights=bm25_probs, monotone=True, return_log_odds=True)

# A copy of the query in the corpus sits at distance zero and captures gap detection;
# keep it out of the density estimate
probabilities = vpt.calibrate_with_sample(
    eval_distances, np.append(sample_distances, 0.0), duplicate_threshold=1e-6,
)
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
platt.fit(neural_scores, labels)             # Newton's method on standardized scores
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

# Bayesian upper bound for safe pruning -- any document's actual
# probability is guaranteed to be at most this value
bayesian_bound = transform.wand_upper_bound(bm25_upper_bound)

# The same bound in log-odds keeps discriminating between blocks whose
# probability bounds have saturated
log_odds_bound = transform.wand_upper_bound_log_odds(bm25_upper_bound)
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

# Report log-odds directly and name the evaluated population.
from bayesian_bm25 import CalibrationContext, calibration_report_from_log_odds

log_odds = np.log(probabilities) - np.log1p(-probabilities)
report = calibration_report_from_log_odds(
    log_odds, labels,
    context=CalibrationContext(reference_population="retrieved_list", count_source="judgments"),
)
print(report.log_loss_skill, report.count_ratio, report.calibration_slope)
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

Evaluated on the same 5 [BEIR](https://github.com/beir-cellar/beir) datasets using two signals: BM25 and all-MiniLM-L6-v2. Each signal retrieves top-1000 documents; fusion uses their union and pytrec_eval. BM25 uses k1=1.2, b=0.75, the Lucene variant, Snowball English stemming, and English stop-word removal. Exact-text identity merges duplicate texts and excludes query copies before scoring; all 3,569 canonical test queries are evaluated. Rankings retain unsaturated log-odds. Averages use unrounded scores and give each collection equal weight. The reproduction commands below regenerate these measurements; `-o <path>` writes raw results, per-query metrics, and training provenance to a local JSON file.

#### NDCG@10

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| BM25 | 51.20 | 25.32 | 31.89 | 15.65 | 67.91 | 38.40 |
| Dense | 52.69 | 36.87 | 31.65 | 21.64 | 64.51 | 41.47 |
| Convex | 56.19 | 37.15 | 35.61 | 19.65 | 73.38 | 44.40 |
| RRF | 56.03 | 36.85 | 35.25 | 20.09 | 71.43 | 43.93 |
| Bayesian-OR | 52.38 | 26.68 | 33.57 | 16.45 | 69.19 | 39.66 |
| Bayesian-LogOdds | 52.81 | 32.60 | 35.28 | 18.49 | 72.73 | 42.38 |
| Bayesian-LogOdds-Local | 56.00 | 37.18 | 34.18 | 19.50 | 73.75 | 44.12 |
| Bayesian-LogOdds-BR | 52.81 | 32.60 | 31.98 | 18.43 | 72.20 | 41.60 |
| **Bayesian-Balanced** | **57.66** | **40.59** | **35.88** | **21.42** | **72.84** | **45.68** |
| Bayesian-Balanced-Mix | 57.66 | 40.71 | 35.71 | 21.47 | 72.69 | 45.65 |
| Bayesian-Balanced-Elbow | 57.66 | 40.62 | 35.88 | 21.38 | 72.83 | 45.67 |
| Bayesian-Gated-ReLU | 52.37 | 27.46 | 32.96 | 17.05 | 69.08 | 39.78 |
| Bayesian-Gated-Swish | 52.35 | 27.25 | 29.39 | 16.84 | 68.70 | 38.91 |
| Bayesian-Gated-GELU | 52.34 | 27.54 | 31.26 | 17.08 | 69.42 | 39.53 |
| Bayesian-Gated-Swish-B2 | 52.35 | 27.56 | 31.66 | 17.13 | 69.45 | 39.63 |
| Bayesian-Gated-Softplus | 52.38 | 26.71 | 34.76 | 16.65 | 69.40 | 39.98 |
| Bayesian-Attention | 53.38 | 39.31 | 34.77 | 21.02 | 70.21 | 43.74 |
| **Bayesian-Attn-Norm** | **53.43** | **40.46** | **35.52** | **21.90** | **73.30** | **44.92** |
| Bayesian-Attn-Norm-CV | 53.43 | 40.46 | 35.52 | 21.90 | 73.30 | 44.92 |
| Bayesian-MultiHead | 53.40 | 39.73 | 34.63 | 21.14 | 69.92 | 43.76 |
| Bayesian-MultiHead-Norm | 57.44 | 38.64 | 35.72 | 21.84 | 70.40 | 44.81 |
| Bayesian-MultiField | 52.93 | -- | 31.30 | 16.42 | 62.17 | 40.70\* |
| Bayesian-MultiField-Bal | 58.64 | -- | 34.64 | 21.16 | 67.26 | 45.42\* |
| Bayesian-Vector-Balanced | 53.45 | 34.48 | 35.59 | 19.17 | 72.66 | 43.07 |
| Bayesian-Vector-Softplus | 55.81 | 34.79 | 32.21 | 18.82 | 69.32 | 42.19 |
| Bayesian-Vector-Attn | 51.99 | 27.02 | 33.93 | 16.37 | 69.13 | 39.69 |

#### MAP@10

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| BM25 | 42.66 | 19.11 | 11.82 | 9.16 | 63.22 | 29.19 |
| Dense | 43.96 | 29.14 | 11.11 | 12.94 | 59.59 | 31.35 |
| Convex | 47.36 | 29.28 | 13.50 | 11.78 | 69.13 | 34.21 |
| RRF | 47.17 | 28.85 | 13.40 | 11.96 | 66.58 | 33.59 |
| Bayesian-OR | 43.62 | 20.30 | 12.46 | 9.66 | 64.48 | 30.11 |
| Bayesian-LogOdds | 44.00 | 25.22 | 13.42 | 10.98 | 68.22 | 32.37 |
| Bayesian-LogOdds-Local | 47.19 | 29.31 | 12.46 | 11.70 | 69.23 | 33.98 |
| Bayesian-LogOdds-BR | 44.00 | 25.22 | 11.98 | 10.94 | 67.74 | 31.98 |
| **Bayesian-Balanced** | **48.99** | **32.77** | **13.87** | **12.85** | **68.52** | **35.40** |
| Bayesian-Balanced-Mix | 48.98 | 32.80 | 13.72 | 12.88 | 68.33 | 35.34 |
| Bayesian-Balanced-Elbow | 48.99 | 32.79 | 13.87 | 12.82 | 68.51 | 35.39 |
| Bayesian-Gated-ReLU | 43.59 | 20.91 | 12.09 | 10.00 | 64.21 | 30.16 |
| Bayesian-Gated-Swish | 43.57 | 20.77 | 10.50 | 9.87 | 63.93 | 29.73 |
| Bayesian-Gated-GELU | 43.58 | 21.01 | 11.15 | 10.02 | 64.66 | 30.08 |
| Bayesian-Gated-Swish-B2 | 43.59 | 21.04 | 11.29 | 10.06 | 64.69 | 30.13 |
| Bayesian-Gated-Softplus | 43.62 | 20.32 | 13.04 | 9.77 | 64.56 | 30.26 |
| Bayesian-Attention | 44.61 | 31.41 | 13.02 | 12.58 | 65.60 | 33.44 |
| **Bayesian-Attn-Norm** | **44.67** | **32.57** | **13.51** | **13.18** | **68.83** | **34.55** |
| Bayesian-Attn-Norm-CV | 44.67 | 32.57 | 13.51 | 13.18 | 68.83 | 34.55 |
| Bayesian-MultiHead | 44.62 | 31.75 | 13.05 | 12.67 | 65.22 | 33.46 |
| Bayesian-MultiHead-Norm | 48.65 | 30.83 | 13.81 | 13.13 | 66.10 | 34.50 |
| Bayesian-MultiField | 44.07 | -- | 11.55 | 9.63 | 58.08 | 30.83\* |
| Bayesian-MultiField-Bal | 49.58 | -- | 13.21 | 12.78 | 63.59 | 34.79\* |
| Bayesian-Vector-Balanced | 44.50 | 26.64 | 13.56 | 11.40 | 68.00 | 32.82 |
| Bayesian-Vector-Softplus | 47.19 | 28.09 | 11.36 | 11.41 | 64.75 | 32.56 |
| Bayesian-Vector-Attn | 43.30 | 20.47 | 12.71 | 9.59 | 64.32 | 30.08 |

#### Recall@10

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| BM25 | 78.54 | 31.98 | 14.58 | 16.36 | 80.78 | 44.45 |
| Dense | 80.43 | 44.13 | 15.50 | 23.09 | 78.33 | 48.30 |
| Convex | 84.21 | 45.04 | 16.91 | 20.60 | 84.89 | 50.33 |
| RRF | 84.05 | 45.03 | 17.07 | 21.11 | 84.76 | 50.40 |
| Bayesian-OR | 80.35 | 33.26 | 15.98 | 16.94 | 82.20 | 45.75 |
| Bayesian-LogOdds | 80.93 | 40.50 | 17.14 | 19.28 | 84.96 | 48.56 |
| Bayesian-LogOdds-Local | 84.01 | 45.24 | 16.36 | 20.40 | 86.22 | 50.45 |
| Bayesian-LogOdds-BR | 80.93 | 40.50 | 15.83 | 19.20 | 84.29 | 48.15 |
| **Bayesian-Balanced** | **85.13** | **47.51** | **17.32** | **22.59** | **84.83** | **51.48** |
| Bayesian-Balanced-Mix | 85.13 | 47.68 | 17.32 | 22.67 | 84.83 | 51.53 |
| Bayesian-Balanced-Elbow | 85.13 | 47.54 | 17.30 | 22.57 | 84.83 | 51.48 |
| Bayesian-Gated-ReLU | 80.43 | 34.34 | 16.30 | 17.76 | 82.58 | 46.28 |
| Bayesian-Gated-Swish | 80.39 | 34.01 | 14.55 | 17.48 | 81.91 | 45.67 |
| Bayesian-Gated-GELU | 80.32 | 34.35 | 15.19 | 17.72 | 82.64 | 46.04 |
| Bayesian-Gated-Swish-B2 | 80.32 | 34.35 | 15.40 | 17.80 | 82.64 | 46.10 |
| Bayesian-Gated-Softplus | 80.35 | 33.31 | 17.17 | 17.22 | 82.87 | 46.18 |
| Bayesian-Attention | 81.28 | 46.92 | 17.15 | 22.19 | 82.71 | 50.05 |
| **Bayesian-Attn-Norm** | **81.36** | **47.32** | **17.12** | **23.17** | **85.44** | **50.88** |
| Bayesian-Attn-Norm-CV | 81.36 | 47.32 | 17.12 | 23.17 | 85.44 | 50.88 |
| Bayesian-MultiHead | 81.36 | 47.53 | 16.98 | 22.31 | 82.71 | 50.18 |
| Bayesian-MultiHead-Norm | 85.25 | 46.03 | 17.26 | 23.26 | 82.33 | 50.83 |
| Bayesian-MultiField | 81.28 | -- | 14.94 | 17.14 | 73.20 | 46.64\* |
| Bayesian-MultiField-Bal | 87.25 | -- | 17.05 | 22.06 | 77.23 | 50.90\* |
| Bayesian-Vector-Balanced | 82.05 | 43.12 | 17.11 | 20.24 | 85.36 | 49.58 |
| Bayesian-Vector-Softplus | 83.17 | 39.19 | 15.38 | 18.56 | 81.96 | 47.65 |
| Bayesian-Vector-Attn | 79.78 | 33.96 | 16.41 | 16.93 | 82.45 | 45.91 |

\*Bayesian-MultiField/Bayesian-MultiField-Bal average over 4 datasets (FiQA corpus lacks title field).

Attention, normalized-attention, multi-head, and vector-attention methods use five-fold query cross-validation; they are not zero-shot. The historical Bayesian-Attn-Norm-CV row is retained as an alias of the same normalized-attention CV result. Other rows require no relevance labels. With `--tune`, the following methods fit only the disjoint official training split. ArguAna and SciDocs do not supply that split, so their tuned cells and five-collection averages are `--`; these collections remain in all default and cross-validated evaluations:

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | NDCG@10 Avg |
|---|---|---|---|---|---|---|
| **Bayesian-Balanced-Tuned** | **--** | **40.68** | **35.77** | **--** | **73.40** | **--** |
| Bayesian-Hybrid-AND-Tuned | -- | 27.88 | 34.10 | -- | 69.71 | -- |
| Bayesian-Tuned | -- | 24.66 | 31.11 | -- | 68.14 | -- |

#### Delta vs BM25 (NDCG@10)

| Method | Type | Delta |
|---|---|---|
| **Bayesian-Attn-Norm** | **5-fold CV** | **+6.53** |
| Bayesian-Attn-Norm-CV | 5-fold CV | +6.53 |
| Bayesian-Balanced-Mix | zero-shot | +7.25 |
| Bayesian-Balanced-Elbow | zero-shot | +7.28 |
| Bayesian-Balanced | zero-shot | +7.28 |
| Convex | zero-shot | +6.00 |
| Bayesian-Vector-Attn | 5-fold CV | +1.29 |
| Bayesian-LogOdds-Local | zero-shot | +5.73 |
| Bayesian-MultiHead-Norm | 5-fold CV | +6.41 |
| RRF | zero-shot | +5.53 |
| Bayesian-MultiHead | 5-fold CV | +5.37 |
| Bayesian-Attention | 5-fold CV | +5.34 |
| Dense | zero-shot | +3.08 |
| Bayesian-MultiField-Bal | zero-shot | +3.76\* |
| Bayesian-Gated-Softplus | zero-shot | +1.58 |
| Bayesian-Gated-ReLU | zero-shot | +1.39 |
| Bayesian-Gated-Swish-B2 | zero-shot | +1.23 |
| Bayesian-Gated-GELU | zero-shot | +1.13 |
| Bayesian-Gated-Swish | zero-shot | +0.51 |

\*Bayesian-MultiField-Bal delta computed over 4 datasets (FiQA corpus lacks title field).

#### Vector Calibration Experiments ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568))

The following experiments evaluate the vector calibration framework from [Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568) ("Vector Scores as Likelihood Ratios"). All methods use `VectorProbabilityTransform` with additive log-odds fusion (Theorem 7.1.1).

**Calibration baselines** (Section 8.2) -- monotone transforms preserve ranking but differ in probability calibration. ECE below is measured on the shared BM25/dense candidate pool, with unjudged documents labeled non-relevant. Platt scaling uses held-out predictions from five-fold CV. Brier and LogLoss are also retained in the result JSON; lower is better:

| Method | ArguAna ECE | FiQA ECE | NFCorpus ECE | SciDocs ECE | SciFact ECE |
|---|---|---|---|---|---|
| Dense-Kappa (global sigmoid) | 0.734 | 0.840 | 0.693 | 0.824 | 0.678 |
| Dense-Arctan | 0.536 | 0.576 | 0.397 | 0.507 | 0.447 |
| Dense-Platt (supervised) | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 |

**Conditional independence penalty** (Section 8.4, Stage 6) -- compares density-prior/gap weights with cross-modal BM25 weights. The historical CI-compliant/CI-violating row names are retained; the weighting source alone does not establish conditional independence:

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| VPT-DensityPrior (CI-compliant) | 49.86 | 19.01 | 24.93 | 14.92 | 44.63 | 30.67 |
| VPT-BM25Weights (CI-violating) | 53.75 | 32.35 | 34.63 | 19.02 | 72.56 | 42.46 |

The table reports the measured weighting-source comparison on all five collections. These results do not establish that either estimated density is a true class-conditional likelihood ratio, nor do they prove conditional independence. Ranking quality and probability calibration are evaluated separately.

**Bandwidth ablation** (Section 8.4, Stage 7) -- Silverman bandwidth scaling factor `c` in {0.2, 0.5, 1.0, 2.0} (Remark 4.4.2):

| Method | ArguAna | FiQA | NFCorpus | SciDocs | SciFact | Average |
|---|---|---|---|---|---|---|
| VPT-BW-0.2 | 52.08 | 25.87 | 33.16 | 15.87 | 68.71 | 39.14 |
| VPT-BW-0.5 | 52.66 | 26.81 | 34.33 | 16.66 | 69.65 | 40.02 |
| VPT-BW-1.0 | 53.71 | 29.17 | 34.82 | 17.98 | 71.03 | 41.34 |
| VPT-BW-2.0 | 54.42 | 33.64 | 35.29 | 19.75 | 71.96 | 43.01 |

The bandwidth rows use the same queries and candidate populations. The corrected runs retain log-odds rather than ranking saturated probabilities; the earlier near-zero ArguAna values were numerical ranking failures, not evidence that the query corpus contains no usable signal.

**Method descriptions:**

| Method | Description |
|---|---|
| BM25 | Sparse retrieval via bm25s (Lucene variant) |
| Dense | Cosine similarity via sentence-transformers |
| Convex | `w * dense_norm + (1-w) * bm25_norm`, w=0.5 |
| RRF | Reciprocal Rank Fusion, `sum(1/(k + rank))`, k=60 |
| Bayesian-OR | `log_odds_or`: probabilistic OR of BM25 and dense posteriors, ranked without probability saturation |
| Bayesian-LogOdds | Corpus-level lexical posterior log-odds plus dense log-odds calibrated via `logit = alpha * (sim - median)` |
| Bayesian-LogOdds-Local | Both raw BM25 and dense calibrated symmetrically via `logit = alpha * (score - median)`, combined |
| Bayesian-LogOdds-BR | Bayesian-LogOdds with base rate prior |
| **Bayesian-Balanced** | `balanced_log_odds_fusion`: min-max normalization of lexical posterior log-odds and dense log-odds, combined at equal weights |
| Bayesian-Balanced-Mix | Bayesian-Balanced with mixture-model base rate estimation |
| Bayesian-Balanced-Elbow | Bayesian-Balanced with elbow-detection base rate estimation |
| Bayesian-Gated-ReLU | `log_odds_conjunction` with ReLU gating in logit space ([Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Theorem 6.5.3) |
| Bayesian-Gated-Swish | `log_odds_conjunction` with Swish gating in logit space ([Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Theorem 6.7.4) |
| Bayesian-Gated-GELU | `log_odds_conjunction` with GELU gating ([Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Theorem 6.8.1): `logit * sigmoid(1.702 * logit)` |
| Bayesian-Gated-Swish-B2 | Generalized swish with `gating_beta=2.0` ([Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Theorem 6.7.6) |
| Bayesian-Gated-Softplus | `log_odds_conjunction` with softplus gating (Remark 6.5.4): `log(1 + exp(logit))`, evidence-preserving smooth ReLU |
| Bayesian-Attention | Query-dependent signal weighting via `AttentionLogOddsWeights` ([Jeong, 2026b](https://doi.org/10.5281/zenodo.18512411), Section 8) |
| **Bayesian-Attn-Norm** | Attention with per-signal logit normalization (`normalize=True`) and 7 features (sparse + dense + cross-signal) |
| Bayesian-Attn-Norm-CV | Historical name for the same held-out five-fold normalized-attention result as Bayesian-Attn-Norm |
| Bayesian-MultiHead | 4-head `MultiHeadAttentionLogOddsWeights`, averages log-odds across heads (Remark 8.6) |
| Bayesian-MultiHead-Norm | Multi-head + logit normalization + 7 features (Corollary 8.7.2) |
| Bayesian-MultiField | `MultiFieldScorer` (title + body) with `log_odds_conjunction`, sparse-only |
| Bayesian-MultiField-Bal | MultiField probs + dense via `balanced_log_odds_fusion` |
| Bayesian-Vector-Balanced | `VectorProbabilityTransform`-calibrated dense probabilities + BM25 via `balanced_log_odds_fusion` ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Theorem 3.1.1) |
| Bayesian-Vector-Softplus | VPT-calibrated dense + BM25 via softplus-gated `log_odds_conjunction` |
| Bayesian-Vector-Attn | VPT-calibrated dense + attention with logit normalization + 7 features |
| Dense-Kappa | Global sigmoid calibration: `P = sigmoid(kappa * (beta - d))` with corpus-level parameters ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Section 8.4 Stage 1) |
| Dense-Arctan | Arctangent normalization: `p = (2/pi) * arctan(alpha * s)` ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Section 8.2) |
| Dense-Platt | Supervised Platt scaling: `P = sigmoid(a * s + b)` with labeled data ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Section 8.2) |
| VPT-DensityPrior | VPT with gap detection / density prior only (CI-compliant, [Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Stage 6) |
| VPT-BM25Weights | VPT with BM25 cross-modal importance weights only (CI-violating, [Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Stage 6) |
| VPT-BW-{c} | VPT with bandwidth factor c applied to Silverman bandwidth ([Jeong, 2026c](https://doi.org/10.5281/zenodo.19181568), Stage 7) |
| Bayesian-Balanced-Tuned | Bayesian-Balanced + supervised `BayesianProbabilityTransform.fit()` + grid search over base_rate and fusion_weight |
| Bayesian-Hybrid-AND-Tuned | `log_odds_conjunction` of Bayesian BM25 and dense probs with tuned alpha |
| Bayesian-Tuned | Sparse-only Bayesian BM25 with tuned alpha, beta, and base_rate (no dense signal) |

**Why include underperforming methods?** The tables above deliberately include methods that underperform BM25. Each failure mode is informative:

- **Bayesian-OR** -- the corrected run ranks its unsaturated log-odds. Its earlier near-zero ArguAna value came from probability saturation and cannot support a claim that OR intrinsically fails on that collection.
- **Bayesian-Gated-\*** -- the table retains all gating variants, including those below the stronger fusion baselines. Performance of a gate depends on the evidence model and signal distributions.
- **Bayesian-MultiField** -- the sparse-only title/body variant and its dense-fused counterpart are retained. FiQA has no title field, so their explicitly marked averages cover the other four collections.

Reproduce:
```bash
# Label-free methods and five-fold cross-validated learned methods
python benchmarks/hybrid_beir.py -d <beir-data-dir>

# With tuning (auto-estimation + supervised learning + grid search)
python benchmarks/hybrid_beir.py -d <beir-data-dir> --tune

# Download BEIR datasets automatically
python benchmarks/hybrid_beir.py -d <beir-data-dir> --download
```

Requires `pip install bayesian-bm25[scorer] sentence-transformers pytrec-eval-terrier PyStemmer scipy`.

### Sparse Retrieval

Evaluated on [BEIR](https://github.com/beir-cellar/beir) NFCorpus and SciFact with k1=1.2, b=0.75, and Lucene BM25. Queries are split 50/50 with seed 42. Batch fitting (C1, C2, C3) and Platt scaling use only training labels. All ranking variants use top-1000 matching-document runs; NDCG@10 and MAP use the full relevance judgments, including relevant documents absent from the run. The original MAP columns report full-run AP, not AP renormalized over retrieved positives. Run the reproduction command below to regenerate the measurements; `-o <path>` writes the results to a local JSON file.

#### Ranking Quality

Base rate prior is a monotonic transform -- it does not change document ordering.

| Method | NFCorpus NDCG@10 | NFCorpus MAP | SciFact NDCG@10 | SciFact MAP |
|---|---|---|---|---|
| Raw BM25 | 0.2603 | 0.1067 | 0.5831 | 0.5415 |
| Bayesian (auto) | 0.2589 | 0.1055 | 0.5741 | 0.5322 |
| Bayesian (auto) + base rate | 0.2589 | 0.1055 | 0.5741 | 0.5322 |
| Bayesian (batch fit) | 0.2578 | 0.1057 | 0.5760 | 0.5323 |
| Bayesian (batch fit) + base rate | 0.2578 | 0.1057 | 0.5760 | 0.5323 |
| Platt scaling | 0.2603 | 0.1067 | 0.5831 | 0.5415 |
| Min-max normalization | 0.2603 | 0.1067 | 0.5831 | 0.5415 |
| Batch fit (prior-aware, C2) | 0.2593 | 0.1056 | 0.5816 | 0.5396 |
| Batch fit (prior-free, C3) | 0.2603 | 0.1067 | 0.5831 | 0.5415 |

#### Probability Calibration

Expected Calibration Error (ECE) and Brier score on the same BM25-matching document population for every method. Matching documents with probability zero are retained. Parenthesized ECE changes are relative to the no-base-rate row; lower is better.

| Method | NFCorpus ECE | NFCorpus Brier | SciFact ECE | SciFact Brier |
|---|---|---|---|---|
| Bayesian (no base rate) | 0.4597 | 0.2592 | 0.6113 | 0.4225 |
| Bayesian (base_rate=auto) | 0.0724 (-84.2%) | 0.0316 | 0.1437 (-76.5%) | 0.0686 |
| Bayesian (base_rate=0.001) | 0.0076 (-98.3%) | 0.0101 | 0.0205 (-96.6%) | 0.0099 |
| Batch fit (no base rate) | 0.0047 (-99.0%) | 0.0098 | 0.0066 (-98.9%) | 0.0036 |
| Batch fit + base_rate=auto | 0.0092 (-98.0%) | 0.0096 | 0.0015 (-99.8%) | 0.0011 |
| Platt scaling | 0.0006 (-99.9%) | 0.0093 | 0.0001 (-100.0%) | 0.0002 |
| Min-max normalization | 0.0189 (-95.9%) | 0.0105 | 0.0156 (-97.4%) | 0.0009 |
| Batch fit (prior-aware, C2) | 0.0455 (-90.1%) | 0.0261 | 0.0888 (-85.5%) | 0.0508 |
| Batch fit (prior-free, C3) | 0.0029 (-99.4%) | 0.0099 | 0.0058 (-99.1%) | 0.0030 |

#### Threshold Transfer

F1 scores using the best threshold found on training queries, applied unchanged to evaluation queries. Thresholds use unsaturated scores on the same matching-document population; no test label selects the threshold.

| Method | NFCorpus Train F1 | NFCorpus Test F1 | SciFact Train F1 | SciFact Test F1 |
|---|---|---|---|---|
| Bayesian (no base rate) | 0.1690 | 0.1630 | 0.3514 | 0.3343 |
| Batch fit (no base rate) | 0.1647 | 0.1594 | 0.3514 | 0.3343 |
| Batch fit + base_rate=auto | 0.1647 | 0.1594 | 0.3514 | 0.3343 |
| Platt scaling | 0.1810 | 0.1753 | 0.3537 | 0.3486 |
| Min-max normalization | 0.1810 | 0.1753 | 0.3537 | 0.3486 |
| Batch fit (prior-aware, C2) | 0.1708 | 0.1655 | 0.3560 | 0.3395 |
| Batch fit (prior-free, C3) | 0.1810 | 0.1753 | 0.3537 | 0.3486 |

Reproduce with `python benchmarks/base_rate.py` (requires `pip install bayesian-bm25[bench]`). The base rate benchmark also includes Platt scaling, min-max normalization, and prior-aware/prior-free training mode comparisons.

Additional benchmarks (no external datasets required):

- `python benchmarks/learnable_weights.py` -- learnable weight recovery, fusion quality, online convergence, and timing
- `python benchmarks/weighted_fusion.py` -- weighted vs uniform log-odds fusion across noise scenarios
- `python benchmarks/wand_upper_bound.py` -- WAND upper bound tightness and skip rate analysis
- `python benchmarks/gating_functions.py` -- gating comparison (none/relu/swish/gelu/softplus), beta sensitivity, timing overhead
- `python benchmarks/bmw_upper_bound.py` -- BMW block-max vs global WAND tightness, pruning rate, block size sensitivity
- `python benchmarks/neural_calibration.py` -- Platt vs isotonic calibration accuracy, hybrid fusion quality, timing
- `python benchmarks/multi_head_attention.py` -- multi-head vs single-head quality, pruning safety/efficiency, head diversity

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

@preprint{Jeong2026VectorLikelihoodRatios,
  author    = {Jeong, Jaepil},
  title     = {Vector Scores as Likelihood Ratios: {Index-Derived} {Bayesian}
               Calibration for Hybrid Search},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19181568},
  url       = {https://doi.org/10.5281/zenodo.19181568}
}

@preprint{Jeong2026UnifiedHybrid,
  author    = {Jeong, Jaepil},
  title     = {A Unified {Bayesian} Framework for Hybrid Search: Calibration
               and Log-Odds Fusion of Lexical and Vector Retrieval},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20768747},
  url       = {https://doi.org/10.5281/zenodo.20768747}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Copyright (c) 2023-2026 Cognica, Inc.
