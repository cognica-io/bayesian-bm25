#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Tests for BlockMaxIndex (BMW block-max upper bounds)."""

import numpy as np
import pytest

from bayesian_bm25.probability import BayesianProbabilityTransform
from bayesian_bm25.scorer import BlockMaxIndex


class TestBlockMaxIndex:
    def test_block_bound_ge_actual(self):
        """Block bound >= actual score for every doc in that block."""
        rng = np.random.default_rng(42)
        n_terms, n_docs = 5, 200
        block_size = 64
        score_matrix = rng.uniform(0.0, 10.0, size=(n_terms, n_docs))

        idx = BlockMaxIndex(block_size=block_size)
        idx.build(score_matrix)

        n_blocks = idx.n_blocks
        for t in range(n_terms):
            for b in range(n_blocks):
                bound = idx.block_upper_bound(t, b)
                start = b * block_size
                end = min(start + block_size, n_docs)
                actual_max = np.max(score_matrix[t, start:end])
                assert bound >= actual_max - 1e-12, (
                    f"Block bound {bound} < actual max {actual_max} "
                    f"for term={t}, block={b}"
                )
                # Also check every individual doc
                for d in range(start, end):
                    assert bound >= score_matrix[t, d] - 1e-12

    def test_block_bound_le_global_wand(self):
        """Block bound <= global WAND upper bound (tighter)."""
        rng = np.random.default_rng(99)
        n_terms, n_docs = 4, 300
        block_size = 64
        score_matrix = rng.uniform(0.0, 8.0, size=(n_terms, n_docs))

        idx = BlockMaxIndex(block_size=block_size)
        idx.build(score_matrix)

        for t in range(n_terms):
            global_max = np.max(score_matrix[t, :])
            for b in range(idx.n_blocks):
                block_bound = idx.block_upper_bound(t, b)
                assert block_bound <= global_max + 1e-12, (
                    f"Block bound {block_bound} > global max {global_max} "
                    f"for term={t}, block={b}"
                )

    def test_correct_block_count(self):
        """Correct block count for various corpus sizes (ceil(n_docs / block_size))."""
        cases = [
            # (n_docs, block_size, expected_n_blocks)
            (100, 64, 2),       # 100 / 64 = 1.5625 -> ceil = 2
            (128, 64, 2),       # exact multiple
            (129, 64, 3),       # one over
            (1, 64, 1),         # single doc
            (64, 64, 1),        # exact fit
            (500, 128, 4),      # 500 / 128 = 3.906 -> ceil = 4
            (1000, 256, 4),     # 1000 / 256 = 3.906 -> ceil = 4
        ]
        for n_docs, block_size, expected in cases:
            score_matrix = np.zeros((2, n_docs))
            bmi = BlockMaxIndex(block_size=block_size)
            bmi.build(score_matrix)
            assert bmi.n_blocks == expected, (
                f"n_docs={n_docs}, block_size={block_size}: "
                f"expected {expected} blocks, got {bmi.n_blocks}"
            )

    def test_pruning_safety(self):
        """No true positive pruned -- bayesian_block_upper_bound >= actual probability."""
        rng = np.random.default_rng(7)
        n_terms, n_docs = 3, 200
        block_size = 32
        score_matrix = rng.uniform(0.0, 6.0, size=(n_terms, n_docs))

        transform = BayesianProbabilityTransform(alpha=1.5, beta=2.0)

        bmi = BlockMaxIndex(block_size=block_size)
        bmi.build(score_matrix)

        for t in range(n_terms):
            for b in range(bmi.n_blocks):
                bayesian_bound = bmi.bayesian_block_upper_bound(
                    t, b, transform, p_max=0.9
                )
                start = b * block_size
                end = min(start + block_size, n_docs)
                for d in range(start, end):
                    raw_score = score_matrix[t, d]
                    # Compute actual probability with various tf/doc_len_ratio
                    for tf in [0, 1, 5, 10]:
                        for ratio in [0.1, 0.5, 1.0, 2.0]:
                            actual_prob = transform.score_to_probability(
                                raw_score, tf, ratio
                            )
                            assert bayesian_bound >= actual_prob - 1e-10, (
                                f"Pruning violation: bound={bayesian_bound}, "
                                f"actual={actual_prob}, term={t}, block={b}, "
                                f"doc={d}, tf={tf}, ratio={ratio}"
                            )

    def test_single_block(self):
        """Edge case: corpus fits in a single block (n_docs <= block_size)."""
        rng = np.random.default_rng(55)
        block_size = 128
        n_docs = 50  # fewer docs than block_size
        n_terms = 3
        score_matrix = rng.uniform(0.0, 5.0, size=(n_terms, n_docs))

        bmi = BlockMaxIndex(block_size=block_size)
        bmi.build(score_matrix)

        assert bmi.n_blocks == 1

        # The single block bound must equal the global max per term
        for t in range(n_terms):
            block_bound = bmi.block_upper_bound(t, 0)
            global_max = np.max(score_matrix[t, :])
            assert block_bound == pytest.approx(global_max)

    def test_invalid_block_size(self):
        """block_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            BlockMaxIndex(block_size=0)
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            BlockMaxIndex(block_size=-5)

    def test_build_validates_2d(self):
        """Non-2D input raises ValueError."""
        bmi = BlockMaxIndex(block_size=64)

        with pytest.raises(ValueError, match="must be 2D"):
            bmi.build(np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="must be 2D"):
            bmi.build(np.ones((2, 3, 4)))

    def test_n_blocks_before_build_raises(self):
        """Accessing n_blocks before build() raises RuntimeError."""
        bmi = BlockMaxIndex(block_size=64)
        with pytest.raises(RuntimeError, match="Call build"):
            _ = bmi.n_blocks

    def test_block_upper_bound_before_build_raises(self):
        """Calling block_upper_bound before build() raises RuntimeError."""
        bmi = BlockMaxIndex(block_size=64)
        with pytest.raises(RuntimeError, match="Call build"):
            bmi.block_upper_bound(0, 0)

    def test_properties(self):
        """block_size returns correct value."""
        for bs in [1, 32, 64, 128, 256, 1024]:
            bmi = BlockMaxIndex(block_size=bs)
            assert bmi.block_size == bs
