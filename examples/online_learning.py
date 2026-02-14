#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Scenario: Adapting to user feedback in real time.

Start with auto-estimated parameters, then refine them incrementally
as users provide relevance feedback (click / skip).
"""

import numpy as np

from bayesian_bm25 import BayesianProbabilityTransform
from bayesian_bm25.probability import sigmoid

# --- Simulate a search system ---
rng = np.random.default_rng(42)

# True (unknown) user relevance model: alpha=2.5, beta=1.5
TRUE_ALPHA = 2.5
TRUE_BETA = 1.5


def simulate_user_click(score: float) -> float:
    """Simulate whether a user clicks (1) or skips (0) a result."""
    prob = sigmoid(TRUE_ALPHA * (score - TRUE_BETA))
    return 1.0 if rng.random() < prob else 0.0


# --- Start with rough initial estimates ---
transform = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
print(f"Initial parameters: alpha={transform.alpha:.3f}, beta={transform.beta:.3f}")

# --- Online learning loop ---
print("\nSimulating user sessions...")
print(f"{'Step':>5}  {'Score':>6}  {'Click':>5}  {'Alpha':>7}  {'Beta':>7}  {'Error':>7}")

errors = []
for step in range(1, 201):
    # Generate a random BM25 score for a shown result
    score = rng.uniform(0.0, 3.5)
    label = simulate_user_click(score)

    # Update parameters from this observation
    transform.update(score, label, learning_rate=0.05, momentum=0.9)

    # Track convergence
    alpha_err = abs(transform.alpha - TRUE_ALPHA)
    beta_err = abs(transform.beta - TRUE_BETA)
    total_err = alpha_err + beta_err
    errors.append(total_err)

    if step <= 10 or step % 50 == 0:
        print(f"  {step:>3}   {score:5.2f}   {label:4.0f}   {transform.alpha:7.3f}  {transform.beta:7.3f}  {total_err:7.3f}")

print(f"\nFinal parameters:  alpha={transform.alpha:.3f}, beta={transform.beta:.3f}")
print(f"Target parameters: alpha={TRUE_ALPHA:.3f}, beta={TRUE_BETA:.3f}")

# --- Mini-batch updates ---
print("\n--- Mini-batch variant ---")
transform2 = BayesianProbabilityTransform(alpha=1.0, beta=0.0)

for batch_num in range(1, 21):
    # Collect a batch of 10 observations
    batch_scores = rng.uniform(0.0, 3.5, size=10)
    batch_labels = np.array([simulate_user_click(s) for s in batch_scores])

    # Update with the whole batch at once
    transform2.update(batch_scores, batch_labels, learning_rate=0.05, momentum=0.9)

    if batch_num <= 3 or batch_num % 5 == 0:
        print(f"  Batch {batch_num:>2}: alpha={transform2.alpha:.3f}, beta={transform2.beta:.3f}")

print(f"\nFinal:  alpha={transform2.alpha:.3f}, beta={transform2.beta:.3f}")
print(f"Target: alpha={TRUE_ALPHA:.3f}, beta={TRUE_BETA:.3f}")

# --- Combining batch fit with online refinement ---
print("\n--- Batch warmup + online refinement ---")
# Step 1: Batch fit on historical data
historical_scores = rng.uniform(0.0, 3.5, size=100)
historical_labels = np.array([simulate_user_click(s) for s in historical_scores])

transform3 = BayesianProbabilityTransform(alpha=1.0, beta=0.0)
transform3.fit(
    historical_scores,
    historical_labels,
    learning_rate=0.05,
    max_iterations=2000,
)
print(f"  After batch fit (100 samples): alpha={transform3.alpha:.3f}, beta={transform3.beta:.3f}")

# Step 2: Online refinement from live feedback
for step in range(1, 101):
    score = rng.uniform(0.0, 3.5)
    label = simulate_user_click(score)
    transform3.update(score, label, learning_rate=0.01, momentum=0.95)

print(f"  After 100 online updates:      alpha={transform3.alpha:.3f}, beta={transform3.beta:.3f}")
print(f"  Target:                         alpha={TRUE_ALPHA:.3f}, beta={TRUE_BETA:.3f}")
