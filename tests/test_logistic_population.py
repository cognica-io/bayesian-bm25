#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Repeated full-population count solves must agree with literal scores."""

import numpy as np
import pytest
from scipy.special import expit

from bayesian_bm25.consistency import solve_offset
from benchmarks.logistic_population import LogisticPopulation


@pytest.mark.parametrize("kind", ["normal", "mixture", "zeros", "constant"])
def test_quadrature_count_roots_agree_with_full_population(kind):
    rng = np.random.default_rng(23)
    values = rng.normal(size=4000)
    if kind == "mixture":
        values[:50] += 20
    elif kind == "zeros":
        values[:3000] = 0
    elif kind == "constant":
        values[:] = 5
    population = LogisticPopulation(values)
    for slope in (2**-1.5, 1.0, 2**1.5):
        for target in (0.01, 1.0, 100.0, 3900.0):
            offset = population.solve(slope, target)
            exact = solve_offset(slope * values, target, tolerance=1e-12)
            assert offset == pytest.approx(exact, abs=2e-9)
            count = np.sum(expit(slope * values - offset))
            assert count == pytest.approx(target, rel=1e-10, abs=1e-12)
            bound = population.count_error_bound(slope, offset)
            assert abs(population.count(slope, offset) - count) <= bound + 1e-11 * target


def test_insufficient_quadrature_precision_uses_full_population():
    rng = np.random.default_rng(33)
    values = rng.normal(size=10000)
    population = LogisticPopulation(values, order=2, tolerance=1e-14)
    result = population.solve(2.8, 5000.0)
    assert population.fallback_count == 1
    assert result == pytest.approx(solve_offset(2.8 * values, 5000.0), abs=1e-12)


def test_grid_cannot_request_an_uncertified_slope():
    population = LogisticPopulation(np.arange(100.0))
    with pytest.raises(ValueError, match="max_slope"):
        population.solve(3.0, 1.0)
