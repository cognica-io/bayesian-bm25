#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Population and assumption provenance for a calibration model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CalibrationContext:
    """Describe the population, parameter sources, and relevance target.

    Sources such as ``convention``, ``supplied``, ``training_mean``, and
    ``judgments`` describe how a parameter was chosen; they do not imply
    statistical identification or empirical validation. Population names
    refer to the target of the count, including when a weighted sample
    represents that population. ``judged_relevance`` distinguishes test
    collection labels from reader usefulness or answer correctness.
    """

    reference_population: str = "population"
    count_source: str = "supplied"
    slope_source: str = "supplied"
    ceiling_source: str = "convention"
    relevance_target: str = "judged_relevance"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        """Serializable provenance for results and experiment metadata."""
        return asdict(self)
