#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Count-constrained posteriors with a relevance ceiling.

Section 3.9 of "From Scores to Evidence" separates the expected relevant
count from the ceiling on each posterior. For pre-offset logits ``g``,
the model is ``P = ceiling * sigmoid(g - offset)``. The offset is solved
at the latent count ``expected_relevant / ceiling`` on the same reference
population. This enforces a supplied total; calibration against actual
relevance still requires appropriate assumptions or judgments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bayesian_bm25.calibration_context import CalibrationContext
from bayesian_bm25.consistency import _as_population, dilute_offset, solve_offset
from bayesian_bm25.probability import sigmoid


def ceiling_log_odds(
    latent_log_odds: np.ndarray | float,
    ceiling: float,
) -> np.ndarray | float:
    """Stable ``logit(ceiling * sigmoid(x))`` without probability rounding.

    The identity is ``log(ceiling) - logaddexp(log(1-ceiling), -x)``.
    A ceiling of one returns ``x`` exactly. For a lower ceiling even
    posterior log-odds can saturate; retain the latent value for ranking.
    """
    if not math.isfinite(ceiling) or not (0.0 < ceiling <= 1.0):
        raise ValueError("ceiling must be finite and in (0, 1]")
    x = np.asarray(latent_log_odds, dtype=np.float64)
    result = (
        x if ceiling == 1.0
        else math.log(ceiling) - np.logaddexp(math.log1p(-ceiling), -x)
    )
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class CountCalibration:
    """A count-constrained model of pre-offset logits over one population.

    Inputs to the prediction methods must have the same slope and units
    as the logits used by :func:`calibrate_count`. ``ranking_score`` is
    the latent log-odds, whereas ``log_odds`` and ``evidence`` describe
    the capped relevance posterior. They coincide only at ceiling one.
    """

    offset: float
    expected_relevant: float
    population_size: float
    ceiling: float = 1.0
    context: CalibrationContext = field(default_factory=CalibrationContext)

    def __post_init__(self) -> None:
        if not isinstance(self.context, CalibrationContext):
            raise ValueError("context must be a CalibrationContext")
        if not math.isfinite(self.offset):
            raise ValueError("offset must be finite")
        if not math.isfinite(self.ceiling) or not (0.0 < self.ceiling <= 1.0):
            raise ValueError("ceiling must be finite and in (0, 1]")
        if not math.isfinite(self.population_size) or self.population_size <= 0.0:
            raise ValueError("population_size must be positive and finite")
        if not math.isfinite(self.expected_relevant) or not (
            0.0 < self.expected_relevant < self.ceiling * self.population_size
        ):
            raise ValueError(
                "expected_relevant must lie strictly between 0 and "
                "ceiling * population_size"
            )

    @property
    def prevalence(self) -> float:
        """Relevance prevalence of the specified population."""
        return self.expected_relevant / self.population_size

    @property
    def prior_log_odds(self) -> float:
        """Prior log-odds, without epsilon clamping."""
        return math.log(self.expected_relevant) - math.log(
            self.population_size - self.expected_relevant
        )

    def ranking_score(self, logits: np.ndarray | float) -> np.ndarray | float:
        """Latent log-odds ``g - offset``; retain this value for ranking."""
        result = np.asarray(logits, dtype=np.float64) - self.offset
        return float(result) if result.ndim == 0 else result

    def probability(self, logits: np.ndarray | float) -> np.ndarray | float:
        """Capped posterior ``ceiling * sigmoid(g - offset)``."""
        result = self.ceiling * np.asarray(sigmoid(self.ranking_score(logits)))
        return float(result) if result.ndim == 0 else result

    def log_odds(self, logits: np.ndarray | float) -> np.ndarray | float:
        """Actual posterior log-odds after applying the ceiling."""
        return ceiling_log_odds(self.ranking_score(logits), self.ceiling)

    def evidence(self, logits: np.ndarray | float) -> np.ndarray | float:
        """Modeled likelihood-ratio evidence with this prior removed."""
        result = np.asarray(self.log_odds(logits)) - self.prior_log_odds
        return float(result) if result.ndim == 0 else result

    def upper_bound_ranking_score(
        self, logit_upper_bound: np.ndarray | float,
    ) -> np.ndarray | float:
        """Safe bound on latent ranking scores under an input bound."""
        return self.ranking_score(logit_upper_bound)

    def upper_bound_log_odds(
        self, logit_upper_bound: np.ndarray | float,
    ) -> np.ndarray | float:
        """Safe bound on the actual capped posterior log-odds."""
        return self.log_odds(logit_upper_bound)


def calibrate_count(
    logits: np.ndarray,
    expected_relevant: float,
    *,
    ceiling: float | None = None,
    weights: np.ndarray | None = None,
    intercept: str = "exact",
    context: CalibrationContext | None = None,
) -> CountCalibration:
    """Solve a count-constrained ceiling model (Corollary 3.24).

    ``weights`` represent population multiplicities or sampling weights,
    not relevance weights. The count must refer to this same population.
    ``intercept="dilute"`` uses the softmax approximation and generally
    undershoots the count when latent probabilities are not small.
    Neither the count nor the ceiling is identified by these logits.
    """
    if intercept not in ("exact", "dilute"):
        raise ValueError("intercept must be 'exact' or 'dilute'")
    supplied_ceiling = ceiling is not None
    ceiling = 1.0 if ceiling is None else float(ceiling)
    if context is None:
        context = CalibrationContext(
            ceiling_source="supplied" if supplied_ceiling else "convention",
        )
    g, w, population_size = _as_population(logits, weights)
    # Validate before dividing the requested count by the ceiling.
    model = CountCalibration(
        0.0, float(expected_relevant), population_size, float(ceiling),
        context,
    )
    latent_count = model.expected_relevant / model.ceiling
    solver = solve_offset if intercept == "exact" else dilute_offset
    offset = solver(g, latent_count, weights=w)
    return CountCalibration(
        offset, model.expected_relevant, population_size, model.ceiling, model.context,
    )
