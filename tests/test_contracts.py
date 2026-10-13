"""Contract tests: the tripwire for the three seams.

    pytest -q

If someone changes contracts.py without telling the other two, these fail
before the integration call does. Fill these in on day 0.
"""

from __future__ import annotations

from src.contracts import PARAM_BOUNDS, clamp_params, default_params, validate_params


def test_defaults_are_valid():
    assert validate_params(default_params()) == []


def test_bounds_are_sane():
    for name, (lo, hi) in PARAM_BOUNDS.items():
        assert lo < hi, f"{name} has an empty range"


def test_clamp_pulls_into_range():
    p = default_params()
    p["ride_height"] = 99.0
    assert clamp_params(p)["ride_height"] == PARAM_BOUNDS["ride_height"][1]


# TODO day 0, once the stub exists:
#   - stub returns a full Evaluation with every pressure region
#   - stub is deterministic with noise=0
#   - uncertainty RISES toward the edges of the design box (the gate needs this)
#   - IterationRecord round-trips to dict
