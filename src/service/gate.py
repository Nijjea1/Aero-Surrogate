"""Shared trust gate.

The policy is contracts.is_in_distribution (uncertainty AND distance from the
training set); this module adds the checks for malformed predictions and the
guidance the agent reads after a rejection. Thresholds are the contract's
defaults until a checkpoint supplies its own calibrated values.
"""

from __future__ import annotations

import math
from numbers import Real

from ..contracts import (
    DEFAULT_BODY_STYLE, OOD_SCORE_THRESHOLD, UNCERTAINTY_THRESHOLD, is_in_distribution,
)

_GUIDANCE = "return toward the last trusted geometry"


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def rejection_reason(
    evaluation: dict,
    threshold: float = UNCERTAINTY_THRESHOLD,
    body_style: str = DEFAULT_BODY_STYLE,
    ood_threshold: float = OOD_SCORE_THRESHOLD,
) -> str | None:
    """Return why a prediction cannot guide optimisation, or None when trusted."""
    for name, value in (("uncertainty threshold", threshold), ("ood threshold", ood_threshold)):
        if not _finite_number(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("cd", "cl", "uncertainty"):
        value = evaluation.get(name)
        if not _finite_number(value):
            return f"invalid prediction: {name} must be a finite number"
    if evaluation["uncertainty"] < 0:
        return "invalid prediction: uncertainty must be nonnegative"
    ood_score = evaluation.get("ood_score")
    if ood_score is not None and (not _finite_number(ood_score) or ood_score < 0):
        return "invalid prediction: ood_score must be a finite nonnegative number or null"
    if type(evaluation.get("in_distribution")) is not bool:
        return "invalid prediction: in_distribution must be a boolean"
    trusted, reason = is_in_distribution(
        evaluation["uncertainty"], ood_score, body_style,
        uncertainty_threshold=threshold, ood_threshold=ood_threshold,
    )
    if not trusted:
        return f"{reason}; {_GUIDANCE}"
    if not evaluation["in_distribution"]:
        return f"out of distribution according to the surrogate; {_GUIDANCE}"
    return None
