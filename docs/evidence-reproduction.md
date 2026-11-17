# From Scores to Evidence: implementation and reproduction record

The specification is the November 14, 2026 manuscript, *From Scores to Evidence: Bayesian Calibration and Identifiability in Hybrid Retrieval*. The implementation follows its equations and population definitions. Additional calibration experiments have separate code, commands, and results under `benchmarks/additional/` and `benchmarks/results/from_scores_to_evidence/additional/`.

## Theory mapped to code

| Paper requirement | Implementation | Verification |
| --- | --- | --- |
| Section 3: a supplied count determines an intercept on a specified population | `consistency.py`, `query_calibration.py` | Count conservation, weighted populations, affine equivariance, feasibility |
| Section 3.9: `p = omega * sigmoid(kappa * e - c)` | `count_calibration.py` | Solve at `R / omega`; separate ranking, posterior log-odds, evidence, and bounds |
| Remark 3.27: sure-document ceiling is the observed relevant fraction | `benchmarks/ceiling_slope.py` | Every stored estimate checked against its exact sampled binary labels |
| Sections 4.2-4.4: marginal and non-relevant backgrounds have different Bayes formulas | `vector_probability.py` | Declared background role, matching evidence, retrieved-range recall factor |
| Definition 4.8: externally supplied importance weights enter the weighted KDE | `vector_probability.py` | External guidance takes precedence; supplied weights remain unchanged unless sharpening is explicit |
| Section 4.7: query-to-corpus background moments and sparse-neighborhood prior | `benchmarks/simple_ivf.py`, `vector_probability.py` | Query background with exclusions; residual background is a named ablation; k-NN sign checked |
| Section 4.8: dense evidence measured in mean-excess tail units | `tail_scale.py`, `benchmarks/evidence_protocol.py` | Positive affine equivariance; explicit degenerate scale; tail depth 100 |
| Section 5: remove source priors and add the target prior once | `evidence_fusion.py`, `fusion.py` | Prior-once algebra, explicit opinion pooling, signed Gaussian coefficients and bounds |
| Sections 6.1 and 6.3: identity and ranking representation | `benchmarks/dataset_identity.py`, `ranking_eval.py` | Exact text identity, maximum alias qrels, self-exclusion, rank-encoded evaluation |
| Section 6.8: constants, judgments, transfer, quarters, and count sensitivity | `benchmarks/ceiling_slope.py` | Canonical-query CV, saved samples, fixed evidence on quarters, full-population evaluation |

`CalibrationContext` records parameter sources, the relevance target, and the population. The paper model is available through `BayesianBM25Scorer(calibration="query")`, `MultiFieldScorer(calibration="query")`, `calibrate_query`, and `calibrate_count`. Corpus calibration remains available for the paper's comparisons with the earlier estimator.

## Executed protocol

Both `all-MiniLM-L6-v2` and `all-mpnet-base-v2` were evaluated on the full canonical test-query sets. Document identity is exact `(title, body)`; query identity is exact query text. Query copies are removed before retrieval, density fitting, and count solving.

The runtime is Python 3.12.7, NumPy 2.2.6, SciPy 1.14.1, bm25s 0.2.14, sentence-transformers 5.3.0, PyStemmer 3.0.0, and pytrec_eval-terrier 0.5.10. Embedding caches record the model, exact ordered text inputs, and identity context.

| Collection | Canonical documents | Canonical test queries |
| --- | ---: | ---: |
| ArguAna | 8626 | 1298 |
| FiQA | 57599 | 648 |
| NFCorpus | 3593 | 323 |
| SciDocs | 25656 | 1000 |
| SciFact | 5183 | 300 |

Each signal retrieves 1000 documents. The dense tail unit uses the top 100 scores. Ceiling and slope grids have 12 and 13 values. Five-fold CV uses canonical queries; a fold's training and held-out predictions use that fold's training-mean relevant count. Thirty sure judgments and ninety stratified judgments are evaluated over ten draws. Every sampled query-document pair and label is saved. The quarter-corpus seed is 7; original scores and units remain fixed, while the intercept is solved again on the quarter.

Grid fitting uses full-population positive Gaussian quadrature with a count-error certificate and a literal-solver fallback. Every selected model is evaluated with the full-array count solver. Across the two ceiling runs, the largest recorded relative count error is below `1e-10`. All 1800 stored sure-ceiling parameter sets equal their observed sample fractions, including unavailable estimates with empty samples.

An empirical zero ceiling is infeasible for a positive count. An empty sure stratum provides no ceiling estimate. These cases are recorded as unavailable, with `--` in tables, without inserting a prior or a grid floor. The supplied supplementary patch's Jeffreys pseudocount is not part of the manuscript's fraction estimator and is excluded from the paper implementation.

Table 1 measures the original reference configurations as well as theoretical models. Its automatic vector configuration explicitly selects gap weights or sharpened external guidance and uses twice the Silverman bandwidth in `benchmarks/reference_vector.py`. The named weight-source and bandwidth ablations supply their own settings. Library calls to the weighted KDE retain the weights that they receive.

## Measured agreement

NDCG@10 is expressed in points. The following five-collection averages match the displayed manuscript figures:

| Configuration | Manuscript | Rerun |
| --- | ---: | ---: |
| RRF | 43.93 | 43.93 |
| Convex combination | 44.40 | 44.40 |
| Range-normalized fusion | 45.68 | 45.68 |
| Additive posterior, Gaussian-background density ratio | 45.33 | 45.33 |
| Additive log-linear posterior, bulk units | 43.77 | 43.77 |
| Additive log-linear posterior, dense tail units | 45.02 | 45.02 |
| Additive log-linear posterior, fitted slopes | 45.47 | 45.47 |
| Product of ranks | 44.46 | 44.46 |

The tail-unit posterior improves on RRF by `1.09 [0.67, 1.52]` points and on convex combination by `0.62 [0.37, 0.87]`, using query bootstrap intervals. Its ranking result and the usefulness of a calibrated head are separate measurements.

Solo dense-slope fits average `2.123` in bulk units and `1.128` in tail units for MiniLM, and `2.078` and `1.226` for MPNet. The lexical tail-unit fit averages `0.975`. These are the separate single-signal fits quoted in Section 4.8; jointly refitting both tail-unit slopes is an additional experiment.

For the fused posterior, the top-ten log-loss skill is:

| Model | ArguAna | FiQA | NFCorpus | SciDocs | SciFact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Theorem 3.7 | +0.03 | +0.12 | -1.26 | -0.55 | +0.42 |
| Ceiling and slope fitted | +0.12 | +0.16 | +0.12 | +0.07 | +0.43 |
| Ceiling from sure judgments | +0.12 | +0.15 | -0.04 | -0.08 | +0.42 |
| Ceiling and slope from stratified judgments | +0.08 | +0.14 | +0.06 | +0.05 | +0.42 |

The generated tables retain fresh numerical results. They are not bit-identical to the archived manuscript tables. Visible differences include SciFact's fitted head skill (`+0.43`, printed `+0.42`), SciDocs' sure-document head skill (`-0.08`, printed `-0.09`), and ArguAna's full-constant quarter head skill (`+0.20`, printed `+0.21`). The Table 1 reference vector rows average `43.07` and `42.19`, compared with the printed `43.03` and `42.08`. The saved JSON records the fitted parameters, exact judgment draws, CV counts, quarter membership, and identity fingerprints for inspection.

The corpus-wide count constraint holds throughout. A correct total does not establish the allocation of probability at the head. The fitted ceiling targets judged relevance under BEIR's treatment of unjudged documents as zero; interpreting it as judgment-selection efficiency requires the additional selection assumption in Remark 3.28.

## Artifacts and additional experiments

The reproduction commands generate local outputs under `benchmarks/results/from_scores_to_evidence/`. These generated files are not retained in the repository:

- `hybrid_beir*.json`: exact search and the two IVF backgrounds.
- `reevaluation.json`, `reevaluation_mpnet.json`: ranking and calibration.
- `ceiling_slope.json`, `ceiling_slope_mpnet.json`: constants and judgment experiments.
- `protocol_diagnosis.json`: identity and probability/log-odds comparisons.
- `tables/table01.md` through `tables/table23.md`: the numbered paper tables.
- `tables/supplement_*.md`: measurements supporting the paper's text.

After reproduction, inspect `tables/table17.md` for ranking, `tables/table19.md` for head calibration, `tables/table22.md` for corpus quarters, and `tables/table23.md` for count sensitivity. The generated `additional/tables/` directory contains the separate baseline and cost results.

The generated `additional/` directory contains equal-budget logistic and tuned convex comparisons, alternative tail fits and tie handling, and measured query cost.

The extra cost run uses fifty uniformly sampled queries per collection and encoder, fresh query encoding on `mps:0`, one BLAS thread, a supplied count of one, and a ceiling of 0.6. It reports stage times and p50/p95 with bootstrap intervals on a shared workstation. Corpus embedding and index construction are setup costs. These observations describe the implemented exhaustive NumPy pipeline, rather than a pruned or ANN serving system. They are separate from the paper's quality results.

## Verification

`python -m pytest -q -p no:cacheprovider` passes 1148 tests. The checks cover count conservation, score-unit equivariance, capped ranking, signed safe bounds, source-prior removal, vector backgrounds, identity, and population-aware evaluation. Ruff and ASCII/diff checks accompany the run. `python -m examples.evidence_calibration` exercises the complete API flow. `python -m benchmarks.gap_duplicates` reproduces Remark 4.12, including the 1000 genuine candidates tied at the floor after adding a query copy and restoration after excluding it.

Reproduction commands and additional-only commands are in [evidence-calibration.md](evidence-calibration.md).
