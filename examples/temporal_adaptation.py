#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Adapting to changing user behavior over time.

In real search systems, what users consider relevant changes over time.
TemporalBayesianTransform uses exponential decay to weight recent
observations more heavily, allowing the model to track these shifts.

This example shows:
  1. Concept drift: relevance model changes halfway through the data
  2. Temporal vs uniform: temporal weighting adapts faster to the shift
  3. Half-life tuning: effect of decay_half_life on adaptation speed
"""

import numpy as np

from bayesian_bm25 import BayesianProbabilityTransform, TemporalBayesianTransform
from bayesian_bm25.probability import sigmoid

# --- Setup: two-phase synthetic data ---
rng = np.random.default_rng(42)

# Phase 1 (t=0..199): user clicks follow alpha=1.5, beta=0.5
PHASE1_ALPHA = 1.5
PHASE1_BETA = 0.5
N_PHASE1 = 200

# Phase 2 (t=200..399): user behavior shifts to alpha=3.0, beta=2.0
PHASE2_ALPHA = 3.0
PHASE2_BETA = 2.0
N_PHASE2 = 200


def simulate_click(score: float, alpha: float, beta: float) -> float:
    """Return 1.0 (click) or 0.0 (skip) based on a sigmoid model."""
    prob = sigmoid(alpha * (score - beta))
    return 1.0 if rng.random() < prob else 0.0


# Generate Phase 1 observations
scores_p1 = rng.uniform(0.0, 3.5, size=N_PHASE1)
labels_p1 = np.array(
    [simulate_click(s, PHASE1_ALPHA, PHASE1_BETA) for s in scores_p1]
)
timestamps_p1 = np.arange(N_PHASE1, dtype=np.float64)

# Generate Phase 2 observations
scores_p2 = rng.uniform(0.0, 3.5, size=N_PHASE2)
labels_p2 = np.array(
    [simulate_click(s, PHASE2_ALPHA, PHASE2_BETA) for s in scores_p2]
)
timestamps_p2 = np.arange(N_PHASE1, N_PHASE1 + N_PHASE2, dtype=np.float64)

# Concatenate into full dataset
all_scores = np.concatenate([scores_p1, scores_p2])
all_labels = np.concatenate([labels_p1, labels_p2])
all_timestamps = np.concatenate([timestamps_p1, timestamps_p2])

print("=== Data Summary ===")
print(f"Phase 1 (t=0..{N_PHASE1 - 1}):   alpha={PHASE1_ALPHA}, beta={PHASE1_BETA}")
print(f"Phase 2 (t={N_PHASE1}..{N_PHASE1 + N_PHASE2 - 1}): alpha={PHASE2_ALPHA}, beta={PHASE2_BETA}")
print(f"Total observations: {len(all_scores)}")

# --- Concept drift detection: temporal vs uniform ---
print("\n=== Concept Drift Detection ===")

temporal = TemporalBayesianTransform(alpha=1.0, beta=0.0, decay_half_life=50)
temporal.fit(
    all_scores,
    all_labels,
    timestamps=all_timestamps,
    learning_rate=0.05,
    max_iterations=5000,
)

uniform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
uniform.fit(
    all_scores,
    all_labels,
    learning_rate=0.05,
    max_iterations=5000,
)

print(f"Phase 2 truth:    alpha={PHASE2_ALPHA:.3f}, beta={PHASE2_BETA:.3f}")
print(f"Temporal (hl=50): alpha={temporal.alpha:.3f}, beta={temporal.beta:.3f}")
print(f"Uniform:          alpha={uniform.alpha:.3f}, beta={uniform.beta:.3f}")

temporal_err = abs(temporal.alpha - PHASE2_ALPHA) + abs(temporal.beta - PHASE2_BETA)
uniform_err = abs(uniform.alpha - PHASE2_ALPHA) + abs(uniform.beta - PHASE2_BETA)
print(f"\nError vs Phase 2 truth:")
print(f"  Temporal: {temporal_err:.3f}")
print(f"  Uniform:  {uniform_err:.3f}")
if temporal_err < uniform_err:
    print("  -> Temporal model is closer to the current relevance pattern.")

# --- Half-life tuning ---
print("\n=== Half-Life Tuning ===")
print(f"Phase 2 truth: alpha={PHASE2_ALPHA:.3f}, beta={PHASE2_BETA:.3f}\n")
print(f"{'Half-life':>10}  {'Alpha':>7}  {'Beta':>7}  {'Alpha err':>10}  {'Beta err':>10}  {'Total err':>10}")
print("-" * 65)

half_lives = [20, 50, 100, 500, 10000]
for hl in half_lives:
    t = TemporalBayesianTransform(alpha=1.0, beta=0.0, decay_half_life=hl)
    t.fit(
        all_scores,
        all_labels,
        timestamps=all_timestamps,
        learning_rate=0.05,
        max_iterations=5000,
    )
    a_err = abs(t.alpha - PHASE2_ALPHA)
    b_err = abs(t.beta - PHASE2_BETA)
    print(f"{hl:>10}  {t.alpha:>7.3f}  {t.beta:>7.3f}  {a_err:>10.3f}  {b_err:>10.3f}  {a_err + b_err:>10.3f}")

print("\nShorter half-life = faster adaptation to recent data.")
print("Very short half-life may overfit to noise in the latest observations.")
print("Very large half-life approaches uniform weighting (no temporal decay).")

# --- Online adaptation: tracking the drift point ---
print("\n=== Online Adaptation ===")
print("Processing observations one-by-one and tracking parameter evolution.\n")

online = TemporalBayesianTransform(alpha=1.0, beta=0.0, decay_half_life=50)

print(f"{'Step':>5}  {'Phase':>6}  {'Score':>6}  {'Click':>5}  {'Alpha':>7}  {'Beta':>7}")

# Track parameter history for summary
alpha_at_drift = None
beta_at_drift = None

for i in range(len(all_scores)):
    score = float(all_scores[i])
    label = float(all_labels[i])
    online.update(score, label, learning_rate=0.05, momentum=0.9)

    phase = 1 if i < N_PHASE1 else 2

    # Record parameters right before the drift
    if i == N_PHASE1 - 1:
        alpha_at_drift = online.alpha
        beta_at_drift = online.beta

    # Print selected steps to show the trajectory
    if i < 5 or i in (N_PHASE1 - 1, N_PHASE1, N_PHASE1 + 1) or (i + 1) % 100 == 0:
        print(f"  {i:>3}  {phase:>5}   {score:5.2f}   {label:4.0f}   {online.alpha:7.3f}  {online.beta:7.3f}")

print(f"\nParameters at drift point (t={N_PHASE1 - 1}):")
print(f"  alpha={alpha_at_drift:.3f}, beta={beta_at_drift:.3f}")
print(f"  Phase 1 truth: alpha={PHASE1_ALPHA:.3f}, beta={PHASE1_BETA:.3f}")

print(f"\nFinal parameters (t={N_PHASE1 + N_PHASE2 - 1}):")
print(f"  alpha={online.alpha:.3f}, beta={online.beta:.3f}")
print(f"  Phase 2 truth: alpha={PHASE2_ALPHA:.3f}, beta={PHASE2_BETA:.3f}")
