#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Benchmark: Learnable log-odds weights (Remark 5.3.2).

Demonstrates:
  1. Weight recovery: learned weights converge to oracle weights under
     heterogeneous signal quality
  2. Fusion quality: MSE and rank correlation of learned vs oracle vs
     uniform weights across noise scenarios
  3. Online convergence: how many streaming updates to match batch fit
  4. Scaling: weight learning across 2, 3, and 5 signals
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from bayesian_bm25.fusion import LearnableLogOddsWeights, log_odds_conjunction
from bayesian_bm25.probability import logit, sigmoid


def generate_signals(
    n_docs: int,
    n_signals: int,
    noise_levels: list[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic signals with controlled per-signal noise.

    Returns (labels, probs, oracle_weights) where oracle weights are
    inversely proportional to noise level (normalized to sum to 1).
    """
    labels = rng.integers(0, 2, size=n_docs).astype(np.float64)

    true_logits = np.where(labels == 1, 1.5, -1.5)
    signals = []
    for noise in noise_levels:
        noisy_logits = true_logits + rng.normal(0, noise, size=n_docs)
        signals.append(np.asarray(sigmoid(noisy_logits), dtype=np.float64))
    probs = np.column_stack(signals)

    # Oracle weights: inverse noise, normalized
    inv_noise = np.array([1.0 / n for n in noise_levels])
    oracle_weights = inv_noise / inv_noise.sum()

    return labels, probs, oracle_weights


def evaluate_fusion(
    fused: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate fusion quality: BCE, MSE, and Spearman rank correlation."""
    fused_c = np.clip(fused, 1e-10, 1.0 - 1e-10)
    bce = -float(np.mean(
        labels * np.log(fused_c) + (1 - labels) * np.log(1 - fused_c)
    ))
    mse = float(np.mean((fused - labels) ** 2))

    # Spearman rank correlation
    rank_f = np.argsort(np.argsort(-fused)).astype(np.float64)
    rank_l = np.argsort(np.argsort(-labels)).astype(np.float64)
    n = len(fused)
    d_sq = np.sum((rank_f - rank_l) ** 2)
    spearman = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))

    return {"BCE": bce, "MSE": mse, "Spearman": spearman}


def run_weight_recovery(rng: np.random.Generator) -> None:
    """Test whether learned weights converge to oracle weights."""
    print("=" * 72)
    print("Weight Recovery: Learned vs Oracle Weights")
    print("=" * 72)

    configs = [
        ([0.5, 2.0], "2 signals: reliable + noisy"),
        ([0.5, 1.0, 2.0], "3 signals: varying quality"),
        ([0.3, 0.5, 1.0, 1.5, 2.5], "5 signals: wide range"),
        ([1.0, 1.0], "2 signals: equal quality"),
        ([1.0, 1.0, 1.0], "3 signals: all equal"),
    ]

    col_s = 30
    print(f"\n  {'Scenario':<{col_s}}  {'Oracle':>20}  {'Learned':>20}  {'L1 Error':>8}")
    print(f"  {'-' * col_s}  {'-' * 20}  {'-' * 20}  {'-' * 8}")

    for noise_levels, desc in configs:
        n_signals = len(noise_levels)
        labels, probs, oracle_w = generate_signals(5000, n_signals, noise_levels, rng)

        learner = LearnableLogOddsWeights(n_signals=n_signals, alpha=0.0)
        learner.fit(probs, labels, learning_rate=0.1, max_iterations=3000)
        learned_w = learner.weights

        l1_error = float(np.sum(np.abs(learned_w - oracle_w)))

        def fmt_w(w):
            return "[" + ", ".join(f"{v:.3f}" for v in w) + "]"

        print(
            f"  {desc:<{col_s}}  {fmt_w(oracle_w):>20}  "
            f"{fmt_w(learned_w):>20}  {l1_error:>8.4f}"
        )


def run_fusion_quality(rng: np.random.Generator) -> None:
    """Compare fusion quality across methods and noise scenarios."""
    print("\n" + "=" * 72)
    print("Fusion Quality: Uniform vs Oracle vs Learned Weights")
    print("=" * 72)

    scenarios = [
        ([0.5, 2.0], "Signal 0 reliable, 1 noisy"),
        ([2.0, 0.5], "Signal 0 noisy, 1 reliable"),
        ([1.0, 1.0], "Equal reliability"),
        ([0.3, 1.0, 2.0], "3 signals: mixed quality"),
    ]

    n_train = 3000
    n_test = 2000

    col_s = 30
    print(f"\n  {'Scenario':<{col_s}}  {'Method':<16}  {'BCE':>7}  {'MSE':>7}  {'Spearman':>8}")
    print(f"  {'-' * col_s}  {'-' * 16}  {'-' * 7}  {'-' * 7}  {'-' * 8}")

    for noise_levels, desc in scenarios:
        n_sig = len(noise_levels)

        # Train / test split
        train_labels, train_probs, oracle_w = generate_signals(
            n_train, n_sig, noise_levels, rng
        )
        test_labels, test_probs, _ = generate_signals(
            n_test, n_sig, noise_levels, rng
        )

        # 1. Uniform weights
        uniform_w = np.full(n_sig, 1.0 / n_sig)
        fused_uniform = np.asarray(
            log_odds_conjunction(test_probs, alpha=0.0, weights=uniform_w),
            dtype=np.float64,
        )
        r_uniform = evaluate_fusion(fused_uniform, test_labels)

        # 2. Oracle weights (inverse noise)
        fused_oracle = np.asarray(
            log_odds_conjunction(test_probs, alpha=0.0, weights=oracle_w),
            dtype=np.float64,
        )
        r_oracle = evaluate_fusion(fused_oracle, test_labels)

        # 3. Learned weights (batch fit)
        learner = LearnableLogOddsWeights(n_signals=n_sig, alpha=0.0)
        learner.fit(train_probs, train_labels, learning_rate=0.1, max_iterations=2000)
        fused_learned = np.atleast_1d(np.asarray(learner(test_probs), dtype=np.float64))
        r_learned = evaluate_fusion(fused_learned, test_labels)

        methods = [
            ("Uniform", r_uniform),
            ("Oracle", r_oracle),
            ("Learned (fit)", r_learned),
        ]

        for i, (method, r) in enumerate(methods):
            scenario_label = desc if i == 0 else ""
            print(
                f"  {scenario_label:<{col_s}}  {method:<16}  "
                f"{r['BCE']:>7.4f}  {r['MSE']:>7.4f}  {r['Spearman']:>8.4f}"
            )
        print()


def run_online_convergence(rng: np.random.Generator) -> None:
    """Track online weight learning convergence against batch fit."""
    print("=" * 72)
    print("Online Convergence: update() vs fit() Target")
    print("=" * 72)

    noise_levels = [0.5, 2.0]
    n_train = 3000

    labels, probs, _ = generate_signals(n_train, 2, noise_levels, rng)

    # Batch fit baseline
    batch_learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)
    batch_learner.fit(probs, labels, learning_rate=0.1, max_iterations=3000)
    target_w = batch_learner.weights

    print(f"\n  Batch fit target weights: [{target_w[0]:.4f}, {target_w[1]:.4f}]")
    print(f"  Training samples: {n_train}")

    # Online learning: stream through data multiple epochs
    online_learner = LearnableLogOddsWeights(n_signals=2, alpha=0.0)

    header = (
        f"  {'Epoch':>5}  {'Updates':>7}  "
        f"{'w[0]':>7}  {'w[1]':>7}  "
        f"{'Avg w[0]':>8}  {'Avg w[1]':>8}  "
        f"{'L1 raw':>7}  {'L1 avg':>7}"
    )
    print(header)

    max_epochs = 20
    total_updates = 0
    raw_converged = None
    avg_converged = None

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n_train)
        batch_size = 32
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            online_learner.update(
                probs[idx], labels[idx],
                learning_rate=0.05, momentum=0.9, avg_decay=0.995,
            )
            total_updates += 1

        w_raw = online_learner.weights
        w_avg = online_learner.averaged_weights
        l1_raw = float(np.sum(np.abs(w_raw - target_w)))
        l1_avg = float(np.sum(np.abs(w_avg - target_w)))

        if l1_raw < 0.02 and raw_converged is None:
            raw_converged = epoch
        if l1_avg < 0.02 and avg_converged is None:
            avg_converged = epoch

        marker = ""
        if avg_converged == epoch:
            marker = " <-- avg matched"
        elif raw_converged == epoch:
            marker = " <-- raw matched"

        if epoch <= 5 or epoch % 5 == 0 or marker:
            print(
                f"  {epoch:>5}  {total_updates:>7}  "
                f"{w_raw[0]:>7.4f}  {w_raw[1]:>7.4f}  "
                f"{w_avg[0]:>8.4f}  {w_avg[1]:>8.4f}  "
                f"{l1_raw:>7.4f}  {l1_avg:>7.4f}{marker}"
            )

    print()
    if raw_converged:
        print(f"  Raw converged at epoch {raw_converged}")
    else:
        print(f"  Raw did not converge within {max_epochs} epochs")
    if avg_converged:
        print(f"  Avg converged at epoch {avg_converged}")
    else:
        print(f"  Avg did not converge within {max_epochs} epochs")


def run_timing(rng: np.random.Generator) -> None:
    """Measure wall-clock time for fit() and update() at various scales."""
    print("\n" + "=" * 72)
    print("Timing: fit() and update() Performance")
    print("=" * 72)

    configs = [
        (2, 1000),
        (2, 10000),
        (3, 1000),
        (3, 10000),
        (5, 1000),
        (5, 10000),
    ]

    col_c = 16
    print(f"\n  {'Config':<{col_c}}  {'fit() ms':>10}  {'update() ms':>12}  {'Speedup':>8}")
    print(f"  {'-' * col_c}  {'-' * 10}  {'-' * 12}  {'-' * 8}")

    for n_sig, n_docs in configs:
        noise = [0.5 + i * 0.5 for i in range(n_sig)]
        labels, probs, _ = generate_signals(n_docs, n_sig, noise, rng)

        # Time fit()
        learner_fit = LearnableLogOddsWeights(n_signals=n_sig, alpha=0.0)
        t0 = time.perf_counter()
        learner_fit.fit(probs, labels, learning_rate=0.1, max_iterations=1000)
        fit_ms = (time.perf_counter() - t0) * 1000

        # Time update() loop (single-sample streaming)
        learner_upd = LearnableLogOddsWeights(n_signals=n_sig, alpha=0.0)
        t0 = time.perf_counter()
        for i in range(min(n_docs, 1000)):
            learner_upd.update(probs[i], labels[i], learning_rate=0.05)
        upd_ms = (time.perf_counter() - t0) * 1000

        desc = f"n={n_sig}, m={n_docs}"
        speedup = upd_ms / fit_ms if fit_ms > 0 else float("inf")
        print(
            f"  {desc:<{col_c}}  {fit_ms:>10.1f}  {upd_ms:>12.1f}  {speedup:>7.1f}x"
        )


def main() -> None:
    rng = np.random.default_rng(42)

    run_weight_recovery(rng)
    print()
    run_fusion_quality(rng)
    run_online_convergence(rng)
    run_timing(rng)


if __name__ == "__main__":
    main()
