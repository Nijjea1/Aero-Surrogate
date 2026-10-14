"""Shared trust gate. The threshold is provisional until independently calibrated."""

from __future__ import annotations

import math
from numbers import Real

from ..contracts import UNCERTAINTY_THRESHOLD


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def rejection_reason(evaluation: dict, threshold: float = UNCERTAINTY_THRESHOLD) -> str | None:
    """Return why a prediction cannot guide optimisation, or None when trusted."""
    if not _finite_number(threshold) or threshold < 0:
        raise ValueError("uncertainty threshold must be finite and nonnegative")
    for name in ("cd", "cl", "uncertainty"):
        value = evaluation.get(name)
        if not _finite_number(value):
            return f"invalid prediction: {name} must be a finite number"
    if evaluation["uncertainty"] < 0:
        return "invalid prediction: uncertainty must be nonnegative"
    if type(evaluation.get("in_distribution")) is not bool:
        return "invalid prediction: in_distribution must be a boolean"
    if not evaluation["in_distribution"]:
        return "out of distribution: return toward the last trusted geometry"
    if evaluation["uncertainty"] > threshold:
        return f"uncertainty {evaluation['uncertainty']:.6g} exceeds threshold {threshold:.6g}; return toward the last trusted geometry"
    return None
