"""The synthetic predictor contract and its deliberate rejection fixtures."""

import math

import pytest

from src.contracts import (
    BODY_STYLES, Design, OOD_SCORE_THRESHOLD, PARAM_BOUNDS, PRESSURE_REGIONS,
    UNCERTAINTY_THRESHOLD, default_params, is_in_distribution,
)
from src.service.predictor import StubPredictor, load_predictor, validate_design_params


def test_deterministic_complete_synthetic_evaluation():
    predictor = StubPredictor()
    design = Design(default_params())
    result = predictor.evaluate(design)
    assert result.to_dict() == predictor.evaluate(design).to_dict()
    assert set(result.pressure_summary) == set(PRESSURE_REGIONS)
    assert all(math.isfinite(v) for v in result.pressure_summary.values())
    assert 0.20 < result.cd < 0.40
    assert result.in_distribution
    assert result.uncertainty < UNCERTAINTY_THRESHOLD
    assert result.model_version == "stub-v2-synthetic"
    assert result.latency_ms == 0.0
    assert design.params == default_params()


def test_uncertainty_increases_and_edge_is_rejected_fixture():
    predictor = StubPredictor()
    results = []
    for fraction in (0.8, 0.9, 0.95, 1.0):
        params = default_params()
        lo, hi = PARAM_BOUNDS["E_A_B_C_Pillar_Thickness"]
        params["E_A_B_C_Pillar_Thickness"] = lo + fraction * (hi - lo)
        results.append(predictor.evaluate(Design(params)))
    assert all(a.uncertainty < b.uncertainty for a, b in zip(results, results[1:]))
    assert results[-1].uncertainty > UNCERTAINTY_THRESHOLD
    assert not results[-1].in_distribution


@pytest.mark.parametrize("value", [True, False, "0.15", None, [], float("nan"), float("inf"), -float("inf"), 10 ** 400])
def test_rejects_invalid_numeric_inputs(value):
    params = default_params()
    params["B_Ramp_Angle"] = value
    assert validate_design_params(params)
    with pytest.raises(ValueError, match="finite number"):
        StubPredictor().evaluate(Design(params))


def test_missing_unknown_and_out_of_bounds_are_explicit():
    params = default_params()
    del params["B_Ramp_Angle"]
    params["unknown"] = 0.0
    params["A_Car_Width"] = 9999.0
    problems = validate_design_params(params)
    assert any("missing parameter 'B_Ramp_Angle'" in p for p in problems)
    assert any("unknown parameter 'unknown'" in p for p in problems)
    assert any("outside" in p for p in problems)
    with pytest.raises(ValueError):
        StubPredictor().evaluate(Design(params))


@pytest.mark.parametrize("value", [None, [], "params"])
def test_rejects_non_mapping(value):
    assert validate_design_params(value)


def test_all_bounds_are_accepted_but_untrusted():
    for side in (0, 1):
        params = {name: bounds[side] for name, bounds in PARAM_BOUNDS.items()}
        result = StubPredictor().evaluate(Design(params))
        assert not result.in_distribution


def test_checkpoint_selection_never_silently_falls_back():
    assert isinstance(load_predictor(), StubPredictor)
    with pytest.raises(NotImplementedError, match="Refusing"):
        load_predictor("missing-checkpoint.pt")
    with pytest.raises(ValueError, match="non-empty"):
        load_predictor("")


def test_every_parameter_and_body_style_moves_cd():
    # Guards the 1.0.0 bug where zip() truncated the objective to 8 parameters.
    predictor = StubPredictor()
    baseline = predictor.evaluate(Design(default_params())).cd
    for name, (lo, hi) in PARAM_BOUNDS.items():
        params = dict(default_params(), **{name: lo + 0.6 * (hi - lo)})
        assert predictor.evaluate(Design(params)).cd != pytest.approx(baseline, abs=1e-9), name
    by_style = {style: predictor.evaluate(Design(default_params(), style)).cd for style in BODY_STYLES}
    assert len(set(by_style.values())) == len(BODY_STYLES)


def test_baseline_body_is_trusted_for_every_style():
    for style in BODY_STYLES:
        result = StubPredictor().evaluate(Design(default_params(), style))
        assert result.in_distribution
        assert result.ood_score is not None
        assert is_in_distribution(result.uncertainty, result.ood_score, style) == (True, None)


def test_jointly_extreme_corner_is_rejected_by_ood_score_alone():
    params = {name: lo + 0.8 * (hi - lo) for name, (lo, hi) in PARAM_BOUNDS.items()}
    result = StubPredictor().evaluate(Design(params))
    assert result.uncertainty < UNCERTAINTY_THRESHOLD
    assert result.ood_score > OOD_SCORE_THRESHOLD
    assert not result.in_distribution


def test_unknown_body_style_is_rejected():
    with pytest.raises(ValueError, match="unknown body style"):
        StubPredictor().evaluate(Design(default_params(), "hatchback"))
