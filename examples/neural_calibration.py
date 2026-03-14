#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Integrating neural reranker scores with Bayesian BM25.

Neural rerankers produce raw scores that are not calibrated probabilities.
To combine them with BM25 probabilities via log-odds conjunction, the
neural scores must first be calibrated into probabilities.

This example shows:
  1. Platt scaling: learn sigmoid calibration from labeled data
  2. Isotonic regression: non-parametric monotone calibration via PAVA
  3. Hybrid fusion: calibrated neural scores + BM25 via log_odds_conjunction
"""

import numpy as np

from bayesian_bm25 import log_odds_conjunction
from bayesian_bm25.calibration import IsotonicCalibrator, PlattCalibrator

# =====================================================================
# Setup: generate synthetic neural reranker scores
# =====================================================================

rng = np.random.default_rng(42)

n_train = 500
n_test = 100

# Training data: raw neural scores in [-3, 3]
train_scores = rng.uniform(-3.0, 3.0, size=n_train)

# True relevance depends on score via a non-linear function:
# probability of relevance = sigmoid(1.5 * score^2 / 3 + 0.5 * score - 1)
# This creates a non-linear relationship that Platt scaling (linear in
# logit space) will approximate, while isotonic regression can capture
# exactly.


def true_relevance_prob(scores: np.ndarray) -> np.ndarray:
    """Non-linear function mapping raw scores to true relevance probability."""
    logits = 1.5 * (scores**2) / 3.0 + 0.5 * scores - 1.0
    return 1.0 / (1.0 + np.exp(-logits))


train_probs = true_relevance_prob(train_scores)
train_labels = (rng.random(n_train) < train_probs).astype(np.float64)

# Test data
test_scores = rng.uniform(-3.0, 3.0, size=n_test)
test_probs = true_relevance_prob(test_scores)
test_labels = (rng.random(n_test) < test_probs).astype(np.float64)

print("Synthetic neural reranker data:")
print(f"  Training samples: {n_train}")
print(f"  Test samples:     {n_test}")
print(f"  Score range:      [{train_scores.min():.2f}, {train_scores.max():.2f}]")
print(f"  Train relevance:  {int(train_labels.sum())} / {n_train}")
print(f"  Test relevance:   {int(test_labels.sum())} / {n_test}")

# =====================================================================
# 1. Platt calibration: sigmoid(a * score + b)
# =====================================================================

print("\n--- Platt Calibration (Sigmoid) ---")

platt = PlattCalibrator(a=1.0, b=0.0)
platt.fit(
    train_scores,
    train_labels,
    learning_rate=0.01,
    max_iterations=5000,
)

print(f"Learned parameters: a = {platt.a:.4f}, b = {platt.b:.4f}")

# Show calibrated values for a few test scores
sample_indices = [0, 10, 25, 50, 75, 99]
print(f"\n  {'Score':>8}  {'Platt P':>8}")
for idx in sample_indices:
    s = test_scores[idx]
    p = platt.calibrate(s)
    print(f"  {s:>8.4f}  {p:>8.4f}")

# Verify output is in (0,1)
platt_test = platt.calibrate(test_scores)
print(f"\nPlatt output range: [{platt_test.min():.6f}, {platt_test.max():.6f}]")
assert np.all(platt_test > 0.0) and np.all(platt_test < 1.0), (
    "Platt output must be in (0, 1)"
)

# Verify monotonicity
sorted_scores = np.sort(test_scores)
sorted_platt = platt.calibrate(sorted_scores)
diffs = np.diff(sorted_platt)
if platt.a > 0:
    assert np.all(diffs >= -1e-12), "Platt calibration must be monotonic"
    print("Monotonicity: verified (non-decreasing)")
else:
    assert np.all(diffs <= 1e-12), "Platt calibration must be monotonic"
    print("Monotonicity: verified (non-increasing)")

# =====================================================================
# 2. Isotonic calibration: non-parametric monotone mapping via PAVA
# =====================================================================

print("\n--- Isotonic Calibration (PAVA) ---")

isotonic = IsotonicCalibrator()
isotonic.fit(train_scores, train_labels)

print(f"  {'Score':>8}  {'Isotonic P':>10}")
for idx in sample_indices:
    s = test_scores[idx]
    p = isotonic.calibrate(s)
    print(f"  {s:>8.4f}  {p:>10.4f}")

# Verify monotonicity of isotonic calibration
sorted_isotonic = isotonic.calibrate(sorted_scores)
iso_diffs = np.diff(sorted_isotonic)
assert np.all(iso_diffs >= -1e-12), (
    "Isotonic calibration must be non-decreasing"
)
print("\nMonotonicity: verified (non-decreasing)")

n_breakpoints = len(isotonic._x) if isotonic._x is not None else 0
print(f"Number of PAVA breakpoints: {n_breakpoints}")

# =====================================================================
# 3. Comparison: raw vs Platt vs isotonic
# =====================================================================

print("\n--- Comparison Table ---")

# Normalize raw scores to (0,1) via sigmoid for a fair comparison
raw_sigmoid = 1.0 / (1.0 + np.exp(-test_scores))

print(
    f"  {'Score':>8}  {'Raw sigm':>9}  {'Platt':>8}  "
    f"{'Isotonic':>9}  {'True P':>7}  {'Label':>5}"
)
print("  " + "-" * 58)

# Show a spread of test points sorted by score
comparison_indices = np.argsort(test_scores)
step = len(comparison_indices) // 10
selected = comparison_indices[::step][:10]

for idx in selected:
    s = test_scores[idx]
    raw_p = raw_sigmoid[idx]
    platt_p = platt.calibrate(s)
    iso_p = isotonic.calibrate(s)
    true_p = test_probs[idx]
    label = int(test_labels[idx])
    print(
        f"  {s:>8.4f}  {raw_p:>9.4f}  {platt_p:>8.4f}  "
        f"{iso_p:>9.4f}  {true_p:>7.4f}  {label:>5}"
    )

# Calibration error (mean squared difference from true probability)
mse_raw = float(np.mean((raw_sigmoid - test_probs) ** 2))
mse_platt = float(np.mean((platt_test - test_probs) ** 2))
iso_test = isotonic.calibrate(test_scores)
mse_isotonic = float(np.mean((iso_test - test_probs) ** 2))

print(f"\nMean squared error vs true probability:")
print(f"  Raw sigmoid:  {mse_raw:.6f}")
print(f"  Platt:        {mse_platt:.6f}")
print(f"  Isotonic:     {mse_isotonic:.6f}")

# =====================================================================
# 4. Hybrid fusion: calibrated neural + BM25 via log_odds_conjunction
# =====================================================================

print("\n--- Hybrid Fusion: Calibrated Neural + BM25 ---")

# Simulate BM25 probabilities for the test documents
# BM25 provides a complementary signal with moderate noise
bm25_logits = np.where(test_labels == 1, 1.0, -1.0) + rng.normal(
    0, 0.8, n_test
)
bm25_probs = 1.0 / (1.0 + np.exp(-bm25_logits))

# Fuse raw neural sigmoid with BM25 (no calibration)
raw_fused = np.asarray(
    log_odds_conjunction(
        np.column_stack([raw_sigmoid, bm25_probs]),
        alpha=0.0,
    ),
    dtype=np.float64,
)

# Fuse Platt-calibrated neural with BM25
platt_fused = np.asarray(
    log_odds_conjunction(
        np.column_stack([platt_test, bm25_probs]),
        alpha=0.0,
    ),
    dtype=np.float64,
)

# Fuse isotonic-calibrated neural with BM25
iso_fused = np.asarray(
    log_odds_conjunction(
        np.column_stack([iso_test, bm25_probs]),
        alpha=0.0,
    ),
    dtype=np.float64,
)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute average precision for a ranked list."""
    ranked = np.argsort(-scores)
    sorted_labels = labels[ranked]
    cum_relevant = np.cumsum(sorted_labels)
    positions = np.arange(1, len(labels) + 1, dtype=np.float64)
    precisions = cum_relevant / positions
    return float(np.sum(precisions * sorted_labels) / max(1.0, labels.sum()))


ap_bm25 = average_precision(bm25_probs, test_labels)
ap_raw = average_precision(raw_sigmoid, test_labels)
ap_raw_fused = average_precision(raw_fused, test_labels)
ap_platt_fused = average_precision(platt_fused, test_labels)
ap_iso_fused = average_precision(iso_fused, test_labels)

print(f"Average precision (ranking quality):")
print(f"  BM25 only:                    {ap_bm25:.4f}")
print(f"  Raw neural only:              {ap_raw:.4f}")
print(f"  Raw neural + BM25 (no cal):   {ap_raw_fused:.4f}")
print(f"  Platt neural + BM25:          {ap_platt_fused:.4f}")
print(f"  Isotonic neural + BM25:       {ap_iso_fused:.4f}")

print("\nTop 5 documents by isotonic + BM25 fusion:")
ranked_indices = np.argsort(-iso_fused)
print(
    f"  {'Rank':>4}  {'Neural':>7}  {'Iso cal':>8}  "
    f"{'BM25':>6}  {'Fused':>6}  {'Label':>5}"
)
for rank, idx in enumerate(ranked_indices[:5], 1):
    print(
        f"  {rank:>4}  {test_scores[idx]:>7.3f}  {iso_test[idx]:>8.4f}  "
        f"{bm25_probs[idx]:>6.4f}  {iso_fused[idx]:>6.4f}  "
        f"{'Rel' if test_labels[idx] == 1 else '---':>5}"
    )

print("\nCalibration converts raw neural scores into probabilities,")
print("enabling principled fusion with BM25 via log-odds conjunction.")
