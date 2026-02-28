# History

## 0.6.0 (2026-02-28)

- Add `balanced_log_odds_fusion()` for hybrid sparse-dense retrieval
  - Converts both Bayesian BM25 probabilities and dense cosine similarities
    to logit space, min-max normalizes each to equalize voting power, and
    combines with configurable weights
  - Prevents heavy-tailed sparse logits (from sigmoid unwrapping) from
    drowning the dense signal while preserving the Bayesian BM25 framework's
    document-length and term-frequency priors
  - Accepts `weight` parameter for asymmetric signal weighting (default 0.5)
  - Composes existing library functions (`logit`, `cosine_to_probability`,
    `_clamp_probability`) rather than reimplementing inline
- Add BEIR hybrid search benchmark (`benchmarks/hybrid_beir.py`)
  - Retrieve-then-evaluate protocol (top-1000 per signal, union candidates,
    pytrec_eval) on 5 BEIR datasets: ArguAna, FiQA, NFCorpus, SciDocs, SciFact
  - 9 fusion methods compared: BM25, Dense, Convex, RRF, Bayesian-OR,
    Bayesian-LogOdds, LO-Local, Bayesian-LO-BR, Bayesian-Balanced
  - Bayesian-Balanced achieves highest average NDCG@10 (41.36%), beating
    Convex (41.15%), RRF (40.48%), and BM25 (35.38%)
  - Also leads in MAP@10 (30.23%) and Recall@10 (49.92%)
  - Tokenization uses bm25s.tokenize with Snowball English stemmer + stop word
    removal, matching the BEIR official BM25 baseline (Lucene EnglishAnalyzer)
  - Embedding cache (.npz) to skip re-encoding across runs
- Add BEIR hybrid search results to README

## 0.5.0 (2026-02-26)

- Add `FusionDebugger` for transparent pipeline inspection (`bayesian_bm25.debug`)
  - Records every intermediate value through the full probability pipeline
    (likelihood, prior, posterior, fusion) so you can trace *why* a document
    received a particular fused score
  - `trace_bm25()`: trace a single BM25 score through sigmoid likelihood,
    composite prior, and Bayesian posterior, capturing logit-space intermediates
  - `trace_vector()`: trace cosine similarity through probability conversion
  - `trace_fusion()`: trace the combination of multiple probability signals
    with method-specific intermediates for `log_odds`, `prob_and`, `prob_or`,
    and `prob_not`
  - `trace_document()`: full pipeline trace composing BM25 + vector + fusion
    into a single `DocumentTrace` with all intermediate values
  - `trace_not()`: trace probabilistic negation (complement) of a single signal
  - `compare()`: compare two `DocumentTrace` objects to explain rank differences,
    identifying the dominant signal and crossover stages where signals disagree
  - `format_trace()`, `format_summary()`, `format_comparison()`: human-readable
    output for traces, one-line summaries, and side-by-side comparisons
- Support all four fusion methods as `method` parameter in `trace_document()`
  and `trace_fusion()`: `"log_odds"`, `"prob_and"`, `"prob_or"`, `"prob_not"`
  - `prob_and`: records `log_probs` and `log_prob_sum` intermediates
  - `prob_or`: records `complements`, `log_complements`, and
    `log_complement_sum` intermediates
  - `prob_not`: computes `prod(1 - p_i)` — the probability that NONE of the
    signals indicate relevance (complement of `prob_or`)
- Support hierarchical (nested) fusion
  - `trace_fusion()` returns a `FusionTrace` whose `fused_probability` can be
    fed directly into the next `trace_fusion()` call, enabling arbitrary
    composition trees such as `AND(OR(title, body), vector, NOT(spam))`
- Support weighted log-odds fusion in `trace_document()` via `weights` parameter
- Add `FusionDebugger` as lazy import in `bayesian_bm25.__init__`
- Add fusion debugger example (`examples/fusion_debugger.py`)
  - 12 examples covering single signal trace, full document trace, batch
    summaries, document comparison, crossover detection, prob_and/or/not
    tracing, hierarchical fusion, all-methods comparison, weighted fusion,
    and base_rate effect demonstration

## 0.4.1 (2026-02-25)

- Add `prob_not()` for probabilistic negation (complement rule)
  - Computes `P(NOT R) = 1 - P(R)` with epsilon clamping for numerical stability
  - In log-odds space, NOT corresponds to simple negation: `logit(1 - p) = -logit(p)`
  - Composes naturally with `prob_and()`, `prob_or()`, and `log_odds_conjunction()`
    for exclusion queries (e.g., "python AND NOT java")
  - Satisfies De Morgan's laws: `NOT(A AND B) = OR(NOT A, NOT B)` and
    `NOT(A OR B) = AND(NOT A, NOT B)`
- Add Boolean NOT example (`examples/boolean_not.py`)
  - Exclusion query scenario with ranking
  - Full Boolean operation comparison table
  - De Morgan's laws verification
  - Log-odds conjunction with exclusion

## 0.4.0 (2026-02-24)

- Add `LearnableLogOddsWeights` for per-signal reliability learning (Remark 5.3.2)
  - Learns weights that map from the Naive Bayes uniform initialization
    (w_i = 1/n) to per-signal reliability weights via softmax parameterization
    over unconstrained logits
  - Completes the correspondence to a fully parameterized single-layer network
    in log-odds space: `logit -> weighted sum -> sigmoid`
  - Hebbian gradient: `dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)`
    (pre-synaptic activity x post-synaptic error, backprop-free)
  - Batch `fit()` via gradient descent on BCE loss
  - Online `update()` via SGD with EMA-smoothed gradients, bias correction,
    L2 gradient clipping, learning rate decay, and Polyak averaging of weights
    in the simplex
  - Alpha (confidence scaling) is fixed, only weights are learned; the two are
    orthogonal (Paper 2, Section 4.2)
- Add theorem verification tests for Remark 5.3.2
  - Naive Bayes initialization: uniform 1/n weights match unweighted conjunction
  - Hebbian gradient structure: zero gradient when signals identical, correct
    direction for overestimating signals
  - Theorem 5.3.1: equal-quality signals maintain approximately uniform weights
- Add learnable weights benchmark (`benchmarks/learnable_weights.py`)
  - Weight recovery accuracy across 2–5 signals with varying noise
  - Fusion quality comparison: uniform vs oracle vs learned weights (BCE, MSE,
    Spearman)
  - Online convergence tracking: `update()` vs `fit()` target
  - Timing measurements for `fit()` and `update()` at various scales
- Add learnable fusion example (`examples/learnable_fusion.py`)
  - Batch fit, online update, Polyak-averaged inference, and alpha confidence
    scaling for a 3-signal hybrid search system

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
