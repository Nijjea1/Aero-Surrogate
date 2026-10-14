"""The synthetic predictor contract and its deliberate rejection fixtures."""

import math

import pytest

from src.contracts import Design, PARAM_BOUNDS, PRESSURE_REGIONS, UNCERTAINTY_THRESHOLD, default_params
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
    assert result.model_version == "stub-v1-synthetic"
    assert result.latency_ms == 0.0
    assert design.params == default_params()


def test_uncertainty_increases_and_edge_is_rejected_fixture():
    predictor = StubPredictor()
    results = []
    for fraction in (0.5, 0.7, 0.9, 1.0):
        params = default_params()
        lo, hi = PARAM_BOUNDS["ride_height"]
        params["ride_height"] = lo + fraction * (hi - lo)
        results.append(predictor.evaluate(Design(params)))
    assert all(a.uncertainty < b.uncertainty for a, b in zip(results, results[1:]))
    assert results[-1].uncertainty > UNCERTAINTY_THRESHOLD
    assert not results[-1].in_distribution


@pytest.mark.parametrize("value", [True, False, "0.15", None, [], float("nan"), float("inf"), -float("inf"), 10 ** 400])
def test_rejects_invalid_numeric_inputs(value):
    params = default_params()
    params["ride_height"] = value
    assert validate_design_params(params)
    with pytest.raises(ValueError, match="finite number"):
        StubPredictor().evaluate(Design(params))


def test_missing_unknown_and_out_of_bounds_are_explicit():
    params = default_params()
    del params["ride_height"]
    params["unknown"] = 0.0
    params["track_width"] = 99.0
    problems = validate_design_params(params)
    assert any("missing parameter 'ride_height'" in p for p in problems)
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
