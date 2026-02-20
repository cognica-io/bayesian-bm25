# History

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
