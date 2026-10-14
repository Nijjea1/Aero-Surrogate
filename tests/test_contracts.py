"""Contract tests: the tripwire for the three seams.

    pytest -q

If someone changes contracts.py without telling the other two, these fail
before the integration call does.

Updated for schema 2.0.0, when the parameter space was replaced with the real
DrivAerNet one. These tests DID fail on that change, which is the point -- it
was raised in the group chat rather than discovered during integration.
"""

from __future__ import annotations

import pytest

from src.contracts import (
    BODY_STYLES,
    DEFAULT_BODY_STYLE,
    OOD_SCORE_THRESHOLD,
    PARAM_BOUNDS,
    PARAM_NAMES,
    PRESSURE_REGIONS,
    SCHEMA_VERSION,
    UNCERTAINTY_THRESHOLD,
    Design,
    Evaluation,
    IterationRecord,
    clamp_params,
    default_params,
    is_in_distribution,
    validate_body_style,
    validate_params,
)


# --------------------------------------------------------------------------
# Seam 2: the parameter space
# --------------------------------------------------------------------------


def test_defaults_are_valid():
    assert validate_params(default_params()) == []


def test_bounds_are_sane():
    for name, (lo, hi) in PARAM_BOUNDS.items():
        assert lo < hi, f"{name} has an empty range"


def test_clamp_pulls_into_range():
    p = default_params()
    name = PARAM_NAMES[0]
    p[name] = 1e6
    assert clamp_params(p)[name] == PARAM_BOUNDS[name][1]
    p[name] = -1e6
    assert clamp_params(p)[name] == PARAM_BOUNDS[name][0]


def test_clamp_refuses_to_drop_a_key():
    p = default_params()
    del p[PARAM_NAMES[0]]
    with pytest.raises(KeyError):
        clamp_params(p)


def test_validate_catches_unknown_and_missing():
    p = default_params()
    p["ride_height"] = 0.15          # a 1.0.0 name; must not be accepted
    problems = validate_params(p)
    assert any("unknown parameter 'ride_height'" in x for x in problems)

    q = default_params()
    del q[PARAM_NAMES[0]]
    assert any("missing parameter" in x for x in validate_params(q))


def test_baseline_is_zero_not_the_bound_midpoint():
    """Parameters are morph deltas, so zero is the real baseline body. The
    bounds are asymmetric, so the midpoint is a different, meaningless car."""
    assert set(default_params().values()) == {0.0}


def test_body_style_is_part_of_the_design():
    assert DEFAULT_BODY_STYLE in BODY_STYLES
    assert validate_body_style(DEFAULT_BODY_STYLE) == []
    assert validate_body_style("hatchback") != []


# --------------------------------------------------------------------------
# Seam 1: evaluation
# --------------------------------------------------------------------------


def test_design_round_trips_to_dict():
    d = Design(params=default_params(), body_style="notchback", design_id="X_1")
    out = d.to_dict()
    assert out["body_style"] == "notchback"
    assert out["params"] == default_params()


def test_evaluation_round_trips_to_dict():
    e = Evaluation(cd=0.291, cl=-0.04, uncertainty=0.006, in_distribution=True,
                   pressure_summary={k: 0.0 for k in PRESSURE_REGIONS})
    out = e.to_dict()
    assert out["cd"] == 0.291
    assert set(out["pressure_summary"]) == set(PRESSURE_REGIONS)
    assert out["ood_score"] is None


# --------------------------------------------------------------------------
# The gate. This is the project's headline claim, so it is tested hardest.
# --------------------------------------------------------------------------


def test_gate_accepts_a_confident_familiar_design():
    ok, reason = is_in_distribution(uncertainty=0.005, ood_score=1.0)
    assert ok and reason is None


def test_gate_rejects_high_uncertainty_with_a_reason():
    ok, reason = is_in_distribution(uncertainty=0.05, ood_score=1.0)
    assert not ok
    assert "uncertainty" in reason


def test_gate_rejects_a_confident_but_unfamiliar_design():
    """The measured failure that schema 2.0.0 exists to fix: with a body style
    held out of training, cd error roughly doubled while sigma widened only
    20%. A sigma-only gate waves that through. The ood_score must catch it
    even when the error bar looks fine."""
    ok, reason = is_in_distribution(uncertainty=0.005, ood_score=99.0)
    assert not ok, "a sigma-only gate would have accepted this"
    assert "out of distribution" in reason


def test_gate_rejects_an_unknown_body_style():
    ok, reason = is_in_distribution(uncertainty=0.005, ood_score=1.0,
                                    body_style="spaceship")
    assert not ok
    assert "body style" in reason


def test_gate_tolerates_a_model_with_no_ood_score():
    """Older checkpoints carry no detector; the gate must still work."""
    ok, _ = is_in_distribution(uncertainty=0.005, ood_score=None)
    assert ok


def test_thresholds_are_plausible():
    # Sigma below the CFD run-to-run spread in the labels (0.0064) would be
    # overconfident by construction, so the threshold must sit above it.
    assert 0.0064 < UNCERTAINTY_THRESHOLD < 0.1
    assert 0.0 < OOD_SCORE_THRESHOLD < 100.0


# --------------------------------------------------------------------------
# Seam 3: the run log
# --------------------------------------------------------------------------


def test_iteration_record_round_trips_to_dict():
    r = IterationRecord(run_id="r1", iteration=3, params=default_params(),
                        cd=0.29, cl=-0.04, uncertainty=0.006,
                        in_distribution=True, accepted=True)
    out = r.to_dict()
    assert out["iteration"] == 3
    assert out["body_style"] == DEFAULT_BODY_STYLE
    assert out["optimiser"] == "agent"
    assert set(out["params"]) == set(PARAM_NAMES)


def test_rejected_record_carries_the_reason():
    ok, reason = is_in_distribution(uncertainty=0.9, ood_score=1.0)
    r = IterationRecord(run_id="r1", iteration=1, params=default_params(),
                        cd=0.29, cl=-0.04, uncertainty=0.9,
                        in_distribution=ok, accepted=ok, reject_reason=reason)
    assert not r.accepted and r.reject_reason
