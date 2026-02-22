# History

## 0.3.2 (2026-02-22)

- Support alpha + weights composability in `log_odds_conjunction()`
  - Per-signal weights (Theorem 8.3) and confidence scaling by signal count
    (Section 4.2) are orthogonal and compose multiplicatively:
    `sigma(n^alpha * sum(w_i * logit(P_i)))`
  - Change `alpha` default from `0.5` to `None` for backward compatibility:
    `None` resolves to `0.5` in unweighted mode and `0.0` in weighted mode
  - Explicit `alpha` applies in both unweighted and weighted modes
- Add comprehensive theorem verification tests (24 new tests)
  - Theorem 5.1.2 / 5.2.2: strict bounds for `prob_and` and `prob_or`
  - Theorem 4.1.2a: Log-OP / Product of Experts algebraic equivalence
  - Remark 5.2.3: heterogeneous signal combination (BM25 + cosine pipeline)
  - Proposition 4.3.2: single signal identity for all alpha values
  - Section 4.2 + Theorem 8.3: weighted alpha composition
  - Theorem 3.2.1 + Corollary 3.2.2: monotone shrinkage of product rule
  - Proposition 3.4.1: information loss in product rule vs log-odds conjunction
  - Theorem 4.4.1 + Proposition 4.4.2: sqrt(n) scaling law
  - Theorem 4.5.1 (iii): spread property (disagreement reduces confidence)
  - Remark 4.1.3: geometric mean residual vs log-odds mean
  - Theorem 6.2.1: sigmoid uniqueness (alternative activations fail)

## 0.3.1 (2026-02-21)

- Optimize posterior computation using two-step Bayes update (Remark 4.4.5)
  - Replaces `sigmoid(logit(L) + logit(br) + logit(p))` with two sequential
    Bayes updates using only multiplication and division
  - `score_to_probability()` delegates to `posterior()` instead of duplicating
    base_rate logic
- Vectorize scorer internals for faster retrieval
  - `_scores_to_probabilities()` processes all k documents per query in one
    vectorized numpy call instead of a scalar-by-scalar inner loop
  - Add `_compute_tf_batch()` for batch term frequency computation
  - Deduplicate pseudo-query sampling: `_sample_pseudo_query_scores()` is
    called once during indexing instead of separately by `_estimate_parameters()`
    and `_estimate_base_rate()`
- Add calibration metrics to the main package
  - `expected_calibration_error()`, `brier_score()`, `reliability_diagram()`
    are now importable from `bayesian_bm25` directly
  - `benchmarks/metrics.py` re-exports from the main package for backward
    compatibility
- Fix `norm_prior` docstring to correctly describe peak at 0.5 and floor at
  0.0/1.0

## 0.3.0 (2026-02-20)

- Add cosine similarity to probability conversion for hybrid text + vector search
  - `cosine_to_probability()`: maps cosine similarity [-1, 1] to probability (0, 1) via (1 + score) / 2 with epsilon clamping (Definition 7.1.2)
- Add weighted log-odds conjunction for per-signal reliability weighting
  - `log_odds_conjunction()` accepts optional `weights` parameter
  - Uses Log-OP formulation: sigma(sum(w_i * logit(P_i))) (Theorem 8.3, Remark 8.4)
  - Weights must be non-negative and sum to 1; unweighted behavior unchanged
- Add WAND upper bound for safe Bayesian document pruning
  - `BayesianProbabilityTransform.wand_upper_bound()`: computes tightest safe probability upper bound using p_max=0.9 (Theorem 6.1.2, Theorem 4.2.4)
  - Supports base_rate-aware bounds for tighter pruning
- Add prior-aware training modes (C1/C2/C3 conditions from Algorithm 8.3.1)
  - `fit()` and `update()` accept `mode` parameter: `"balanced"` (C1, default), `"prior_aware"` (C2), `"prior_free"` (C3)
  - Prior-aware (C2): trains on full Bayesian posterior with chain-rule gradients through dP/dL
  - Prior-free (C3): trains on likelihood, inference uses prior=0.5
- Add weighted fusion benchmark (`benchmarks/weighted_fusion.py`)
- Add WAND upper bound tightness benchmark (`benchmarks/wand_upper_bound.py`)
- Extend base rate benchmark with Platt scaling, min-max normalization, and C2/C3 training mode rows

## 0.2.0 (2026-02-18)

- Add corpus-level base rate prior for unsupervised probability calibration
  - `BayesianProbabilityTransform` accepts `base_rate` parameter
  - `BayesianBM25Scorer` accepts `base_rate` parameter (`None`, `"auto"`, or float)
  - Three-term log-odds posterior: `sigmoid(logit(L) + logit(b_r) + logit(p))`
  - Auto-estimation via 95th percentile pseudo-query heuristic
  - Reduces expected calibration error by 68–77% on BEIR datasets without relevance labels
- Add calibration verification benchmark (`benchmarks/calibration.py`)
- Add theorem verification tests for both papers

## 0.1.1 (2026-02-16)

- Fix `log_odds_conjunction` to use multiplicative log-odds mean from Paper 2

## 0.1.0 (2026-02-14)

- Initial release
- Sigmoid likelihood + composite prior (term frequency + document length) + Bayesian posterior
- Batch gradient descent and online SGD with EMA-smoothed gradients and Polyak averaging
- Probabilistic score combination: `prob_and`, `prob_or`, `log_odds_conjunction`
- `BayesianBM25Scorer` wrapping bm25s with auto-estimated sigmoid parameters
