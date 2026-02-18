# History

## 0.2.0 (2026-02-18)

- Add corpus-level base rate prior for unsupervised probability calibration
  - `BayesianProbabilityTransform` accepts `base_rate` parameter
  - `BayesianBM25Scorer` accepts `base_rate` parameter (`None`, `"auto"`, or float)
  - Three-term log-odds posterior: `sigmoid(logit(L) + logit(b_r) + logit(p))`
  - Auto-estimation via 95th percentile pseudo-query heuristic
  - Reduces expected calibration error by 68--77% on BEIR datasets without relevance labels
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
