#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Live ranking demo: watch document rankings shift in real time
as user feedback updates the Bayesian probability model.

Scenario: a news search system where editorial staff provide relevance
feedback. The model starts with a nearly-flat sigmoid (low alpha), where
the composite prior (term frequency + document length) dominates the
ranking. As editors click on truly relevant results, the model learns a
steeper sigmoid and higher threshold, causing the BM25 score signal to
overtake the prior -- documents swap positions.

No external dependencies -- uses simulated BM25 scores.

Run:
    python examples/live_ranking.py
"""

import numpy as np

from bayesian_bm25 import BayesianProbabilityTransform
from bayesian_bm25.probability import sigmoid

# ---------------------------------------------------------------------------
# Corpus: 8 documents with pre-computed BM25 scores for query "AI regulation"
#
# Key design: some documents have high BM25 but weak prior (low tf, extreme
# doc length), others have moderate BM25 but strong prior (high tf, ideal
# doc length). When alpha is small, the prior dominates and these pairs
# swap -- as the model learns and alpha grows, BM25 score takes over.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    # High BM25, strong prior  (always top)
    {"id": "D01", "title": "EU AI Act: Full Text",
     "bm25": 9.2, "tf": 8, "dlr": 0.5},
    {"id": "D06", "title": "AI Regulation: US vs EU",
     "bm25": 8.5, "tf": 7, "dlr": 0.6},
    # --- Swap pair A ---
    # D02: higher BM25, but weak prior (low tf, very long doc)
    # D09: lower BM25, but strong prior (high tf, ideal length)
    {"id": "D02", "title": "AI Law Overview (Long Report)",
     "bm25": 7.3, "tf": 1, "dlr": 2.5},
    {"id": "D09", "title": "Regulatory Compliance for AI",
     "bm25": 6.5, "tf": 9, "dlr": 0.5},
    # --- Swap pair B ---
    # D03: higher BM25, but very short and sparse
    # D10: lower BM25, but rich content at ideal length
    {"id": "D03", "title": "AI Governance Brief",
     "bm25": 5.8, "tf": 1, "dlr": 0.1},
    {"id": "D10", "title": "Global AI Policy Landscape 2026",
     "bm25": 4.9, "tf": 8, "dlr": 0.5},
    # Low BM25, tangentially related  (always bottom)
    {"id": "D07", "title": "Tech Stocks Rally on AI Hype",
     "bm25": 3.0, "tf": 1, "dlr": 1.8},
    {"id": "D08", "title": "Deep Learning Tutorial",
     "bm25": 1.5, "tf": 1, "dlr": 2.0},
]

RELEVANT_IDS = {"D01", "D02", "D03", "D06", "D09", "D10"}

# True user relevance model (unknown to the system).
# High alpha = steep sigmoid; beta=5.0 = only BM25 > 5 is really relevant.
TRUE_ALPHA = 2.0
TRUE_BETA = 5.0

rng = np.random.default_rng(42)


def simulate_click(score: float) -> float:
    """Simulate editorial click (1) or skip (0) based on hidden user model."""
    p = sigmoid(TRUE_ALPHA * (score - TRUE_BETA))
    return 1.0 if rng.random() < float(p) else 0.0


def rank_documents(
    transform: BayesianProbabilityTransform,
    use_averaged: bool = False,
) -> list[dict]:
    """Score and rank all documents.  Optionally use Polyak-averaged params."""
    if use_averaged:
        t = BayesianProbabilityTransform(
            alpha=transform.averaged_alpha,
            beta=transform.averaged_beta,
            base_rate=transform.base_rate,
        )
    else:
        t = transform

    results = []
    for doc in DOCUMENTS:
        prob = float(t.score_to_probability(doc["bm25"], doc["tf"], doc["dlr"]))
        results.append({**doc, "prob": prob, "relevant": doc["id"] in RELEVANT_IDS})
    results.sort(key=lambda d: d["prob"], reverse=True)
    for i, doc in enumerate(results, 1):
        doc["rank"] = i
    return results


def print_ranking(
    results: list[dict],
    label: str,
    prev_ranks: dict[str, int] | None = None,
) -> dict[str, int]:
    """Print a formatted ranking table with optional rank-change arrows."""
    print(f"\n  {label}")
    print(f"  {'=' * 74}")
    print(f"  {'Rank':>4}  {'ID':<4}  {'BM25':>5}  {'P(R)':>7}  {'Rel':>3}  {'Move':>5}  Title")
    print(f"  {'-' * 74}")

    current_ranks = {}
    for doc in results:
        current_ranks[doc["id"]] = doc["rank"]
        rel = " * " if doc["relevant"] else "   "

        if prev_ranks and doc["id"] in prev_ranks:
            delta = prev_ranks[doc["id"]] - doc["rank"]
            if delta > 0:
                move = f" +{delta:d}  "
            elif delta < 0:
                move = f" {delta:d}  "
            else:
                move = "  .  "
        else:
            move = "     "

        print(
            f"  {doc['rank']:4d}  {doc['id']:<4}  {doc['bm25']:5.1f}  "
            f"{doc['prob']:7.4f}  {rel}  {move} {doc['title']}"
        )

    top5_relevant = sum(1 for d in results[:5] if d["relevant"])
    print(f"  {'-' * 74}")
    print(f"  Precision@5: {top5_relevant}/5")
    return current_ranks


def main() -> None:
    print("Live Ranking Demo: Online Learning with Bayesian BM25")
    print("=" * 78)
    print()
    print("Query: 'AI regulation'")
    print("Relevant documents marked with *")
    print()
    print("Key pairs to watch:")
    print("  D02 (BM25=7.3, tf=1, long)  vs  D09 (BM25=6.5, tf=9, ideal)")
    print("  D03 (BM25=5.8, tf=1, short) vs  D10 (BM25=4.9, tf=8, ideal)")
    print()
    print("When alpha is small, the prior (tf, doc length) dominates.")
    print("As the model learns, BM25 score takes over -> rank swaps.")

    # --- Phase 0: uninformed model (flat sigmoid) ---
    transform = BayesianProbabilityTransform(alpha=0.2, beta=0.0, base_rate=0.05)
    print(f"\nInitial: alpha={transform.alpha:.3f}, beta={transform.beta:.3f}")

    results = rank_documents(transform)
    prev = print_ranking(results, "INITIAL RANKING (alpha=0.2, prior-dominated)")

    # --- Phase 1: first editorial session (10 reviews) ---
    print("\n\n--- Phase 1: First editorial session (10 reviews) ---")
    print(f"  {'Score':>6}  {'Click':>5}   alpha    beta")
    for _ in range(10):
        score = rng.choice([d["bm25"] for d in DOCUMENTS])
        label = simulate_click(score)
        transform.update(score, label, learning_rate=0.15, momentum=0.8)
        click_str = "CLICK" if label == 1.0 else "skip "
        print(
            f"  {score:6.1f}  {click_str}  {transform.alpha:7.3f}  {transform.beta:7.3f}"
        )

    results = rank_documents(transform)
    prev = print_ranking(results, "AFTER 10 REVIEWS", prev)

    # --- Phase 2: second editorial session (20 reviews) ---
    print("\n\n--- Phase 2: Second editorial session (20 reviews) ---")
    for i in range(1, 21):
        score = rng.choice([d["bm25"] for d in DOCUMENTS])
        label = simulate_click(score)
        transform.update(score, label, learning_rate=0.08, momentum=0.9)
        if i % 10 == 0:
            print(
                f"  [{i:>2} reviews]  alpha={transform.alpha:.3f}  "
                f"beta={transform.beta:.3f}  "
                f"(Polyak: {transform.averaged_alpha:.3f}, "
                f"{transform.averaged_beta:.3f})"
            )

    results = rank_documents(transform)
    prev = print_ranking(results, "AFTER 30 REVIEWS", prev)

    # --- Phase 3: long-running adaptation (100 reviews) ---
    print("\n\n--- Phase 3: Continuous adaptation (100 reviews) ---")
    for i in range(1, 101):
        score = rng.uniform(1.0, 10.0)
        label = simulate_click(score)
        transform.update(score, label, learning_rate=0.03, momentum=0.95)
        if i % 25 == 0:
            print(
                f"  [{i:>3} reviews]  alpha={transform.alpha:.3f}  "
                f"beta={transform.beta:.3f}  "
                f"(Polyak: {transform.averaged_alpha:.3f}, "
                f"{transform.averaged_beta:.3f})"
            )

    results = rank_documents(transform)
    prev = print_ranking(results, "AFTER 130 REVIEWS (converged)", prev)

    # --- Final summary ---
    print("\n\n--- Summary ---")
    print(f"  Hidden model:   alpha={TRUE_ALPHA:.3f}  beta={TRUE_BETA:.3f}")
    print(f"  Learned:        alpha={transform.alpha:.3f}  beta={transform.beta:.3f}")
    print(
        f"  Polyak avg:     alpha={transform.averaged_alpha:.3f}  "
        f"beta={transform.averaged_beta:.3f}"
    )

    # Probability comparison table
    initial_t = BayesianProbabilityTransform(alpha=0.2, beta=0.0, base_rate=0.05)
    header = (
        f"  {'ID':<4}  {'BM25':>5}  {'TF':>2}  {'DLR':>4}"
        f"  {'P(init)':>8}  {'P(now)':>8}  {'Shift':>7}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 56}")
    for doc in sorted(DOCUMENTS, key=lambda d: d["bm25"], reverse=True):
        p0 = float(initial_t.score_to_probability(doc["bm25"], doc["tf"], doc["dlr"]))
        p1 = float(transform.score_to_probability(doc["bm25"], doc["tf"], doc["dlr"]))
        print(
            f"  {doc['id']:<4}  {doc['bm25']:5.1f}  {doc['tf']:2d}  {doc['dlr']:4.1f}"
            f"  {p0:8.4f}  {p1:8.4f}  {p1 - p0:+7.4f}"
        )


if __name__ == "__main__":
    main()
