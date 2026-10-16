"""Fair comparison is defined by actual calls, never by optimistic log counts."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from src.agent.comparison import (
    BudgetedEvaluator, ComparisonProtocol, ComparisonRefused, ComparisonStopped, summarize_pair,
)
from src.contracts import default_params


def protocol(**kwargs):
    return ComparisonProtocol("a" * 64, "fixture-v1", "revision", .01, 10., **kwargs)


def design(value=0):
    return dict(default_params(), B_Ramp_Angle=value)


def evaluate(params, style):
    return {"cd": .3 - params["B_Ramp_Angle"] * .001, "cl": 0., "uncertainty": .002,
            "ood_score": 1., "in_distribution": True, "pressure_summary": {}, "latency_ms": 0.}


def finish(spec=None, *, run_id="agent", evaluator=evaluate, stop_reason="evaluation budget used"):
    spec = spec or protocol(evaluation_budget=2)
    wrapped = BudgetedEvaluator(evaluator, spec)
    wrapped(design(), spec.body_style)
    wrapped(design(1), spec.body_style)
    return wrapped.summary(run_id, run_id, stop_reason)


def test_protocol_frozen_canonical_and_json_roundtrip():
    spec = protocol(constraints=[" B_Ramp_Angle>=-1", "A_Car_Length <= 1.123456789"])
    assert spec == ComparisonProtocol.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert spec.protocol_id == protocol(constraints=["A_Car_Length<=1.123456789", "B_Ramp_Angle >= -1.0"]).protocol_id
    assert "1.123456789" in spec.constraints[0]
    with pytest.raises(FrozenInstanceError):
        spec.evaluation_budget = 3
    spec.baseline_params["B_Ramp_Angle"] = 4
    assert spec.baseline_params == default_params()


@pytest.mark.parametrize("kwargs", [
    {"checkpoint_sha256": "abc"}, {"model_version": ""}, {"git_revision": ""},
    {"uncertainty_threshold": True}, {"ood_threshold": float("nan")},
    {"body_style": "wrong"}, {"evaluation_budget": True}, {"max_attempts": 1},
    {"constraints": ["B_Ramp_Angle > 0"]},
])
def test_invalid_protocol_refused(kwargs):
    values = dict(checkpoint_sha256="a" * 64, model_version="v", git_revision="rev",
                  uncertainty_threshold=.01, ood_threshold=10.)
    values.update(kwargs)
    with pytest.raises(ValueError):
        ComparisonProtocol(**values)


def test_calls_baseline_rejections_cap_and_trusted_curve():
    seen = []

    def gated(params, style):
        seen.append(params)
        result = evaluate(params, style)
        if params["B_Ramp_Angle"] == 1:
            result.update(cd=.01, uncertainty=.02)
        return result

    wrapped = BudgetedEvaluator(gated, protocol(evaluation_budget=3))
    for value in (0, 1, 2):
        wrapped(design(value), "fastback")
    with pytest.raises(ComparisonStopped, match="budget"):
        wrapped(design(3), "fastback")
    result = wrapped.summary("agent", "a", "evaluation budget used")
    assert result["complete"] and len(seen) == result["calls"] == 3
    assert result["counts"] == {"accepted": 2, "rejected": 1, "error": 0}
    assert result["best_curve"] == [.3, .3, .298]
    assert result["delta_cd"] == pytest.approx(-.002)


@pytest.mark.parametrize("params,style", [
    ({}, "fastback"), (dict(default_params(), B_Ramp_Angle=True), "fastback"),
    (dict(default_params(), B_Ramp_Angle="1"), "fastback"),
    (design(100), "fastback"), (design(1), "fastback"), (design(), "notchback"),
    (dict(default_params(), extra=1), "fastback"),
])
def test_invalid_start_does_not_spend(params, style):
    wrapped = BudgetedEvaluator(evaluate, protocol())
    with pytest.raises(ComparisonRefused):
        wrapped(params, style)
    assert wrapped.calls == 0 and wrapped.attempts == 1


def test_constraint_and_duplicate_refused_before_inference_and_attempt_limit():
    wrapped = BudgetedEvaluator(evaluate, protocol(evaluation_budget=2, max_attempts=3,
                                                 constraints=("B_Ramp_Angle <= 0",)))
    wrapped(design(), "fastback")
    for params in (design(1), design()):
        with pytest.raises(ComparisonRefused):
            wrapped(params, "fastback")
    with pytest.raises(ComparisonStopped, match="attempt budget"):
        wrapped(design(-1), "fastback")
    result = wrapped.summary("random", "r", "evaluation budget used")
    assert result["attempts"] == 3 and result["calls"] == 1 and not result["complete"]


def test_predictor_failure_spends_call_and_stops_future_calls():
    def broken(params, style):
        raise RuntimeError("boom")

    wrapped = BudgetedEvaluator(broken, protocol())
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(design(), "fastback")
    with pytest.raises(ComparisonStopped):
        wrapped(design(1), "fastback")
    result = wrapped.summary("agent", "a", "evaluation budget used")
    assert result["calls"] == 1 and result["counts"]["error"] == 1 and not result["complete"]


@pytest.mark.parametrize("change", [{"cd": float("nan")}, {"in_distribution": 1}, {"uncertainty": -1}])
def test_malformed_prediction_is_accounted_fatal_error(change):
    wrapped = BudgetedEvaluator(lambda p, s: dict(evaluate(p, s), **change), protocol())
    with pytest.raises(ValueError, match="invalid prediction"):
        wrapped(design(), "fastback")
    assert wrapped.summary("agent", "a", "error")["counts"]["error"] == 1


def test_cancel_and_partial_stops_never_claim_complete():
    cancel = [False]
    wrapped = BudgetedEvaluator(evaluate, protocol(), should_stop=lambda: cancel[0])
    wrapped(design(), "fastback")
    cancel[0] = True
    with pytest.raises(ComparisonStopped, match="cancelled"):
        wrapped(design(1), "fastback")
    assert wrapped.calls == 1
    assert wrapped.summary("agent", "a", "evaluation budget used")["stop_reason"] == "cancelled"
    assert not finish(stop_reason="budget: provider spend cap")["complete"]


@pytest.mark.parametrize("budget", [1, 2])
def test_untrusted_baseline_stops_every_method_without_improvement_claim(budget):
    def gate_baseline(params, style):
        return dict(evaluate(params, style), uncertainty=.02 if params == default_params() else .002)

    wrapped = BudgetedEvaluator(gate_baseline, protocol(evaluation_budget=budget))
    wrapped(design(), "fastback")
    with pytest.raises(ComparisonStopped, match="baseline rejected"):
        wrapped(design(1), "fastback")
    result = wrapped.summary("random", "rejected-baseline", "evaluation budget used")
    assert not result["complete"] and result["calls"] == 1
    assert result["stop_reason"].startswith("baseline rejected:")
    assert result["baseline_cd"] is result["best_cd"] is None
    assert result["delta_cd"] is result["improvement_fraction"] is None


def test_evidence_copies_and_matched_pair():
    result = finish()
    other = finish(run_id="random")
    pair = summarize_pair(result, other)
    assert pair["complete"]
    result["outcomes"][0]["params"]["B_Ramp_Angle"] = 2
    assert pair["runs"][0]["outcomes"][0]["params"] == default_params()


@pytest.mark.parametrize("change", [
    {"checkpoint_sha256": "b" * 64}, {"body_style": "notchback"},
    {"uncertainty_threshold": .02}, {"constraints": ("B_Ramp_Angle >= 0",)},
])
def test_pair_refuses_changed_conditions(change):
    right = finish(replace(protocol(evaluation_budget=2), **change), run_id="random")
    with pytest.raises(ValueError, match="protocols differ"):
        summarize_pair(finish(), right)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(calls=25), lambda s: s.update(best_cd=.001),
    lambda s: s.update(attempts=3), lambda s: s["outcomes"][0].update(status="rejected"),
    lambda s: s["protocol"].update(baseline_params=design(1)),
    lambda s: s["outcomes"][1].update(params=default_params()),
])
def test_pair_recomputes_and_refuses_changed_accounting(mutate):
    left = finish()
    mutate(left)
    with pytest.raises(ValueError):
        summarize_pair(left, finish(run_id="random"))


def test_incomplete_pair_is_retained_not_padded():
    wrapped = BudgetedEvaluator(evaluate, protocol(evaluation_budget=2))
    wrapped(design(), "fastback")
    left = wrapped.summary("agent", "a", "cancelled", metadata={"model": "nano"})
    pair = summarize_pair(left, finish(run_id="random"))
    assert not pair["complete"] and pair["runs"][0]["calls"] == 1
    json.dumps(pair, allow_nan=False)


def test_existing_designer_runs_through_shared_evaluator(tmp_path):
    from src.agent import designer
    from src.agent.nemotron import NemotronProposer
    from src.service.runlog import RunLog
    from tests.test_nemotron import FakeClient, evaluate_call

    spec = protocol(evaluation_budget=2)
    wrapped = BudgetedEvaluator(evaluate, spec)
    log = RunLog(tmp_path)
    designer.run(2, "comparison-agent", False, evaluator=wrapped, thresholds=spec.thresholds,
                 provenance="fixture checkpoint", proposer=NemotronProposer(FakeClient([
                     [evaluate_call({"B_Ramp_Angle": 1.})],
                 ])), log=log, metadata={"comparison_protocol": spec.to_dict()})
    finished = log.read_events("comparison-agent")[-1]
    result = wrapped.summary("agent", "comparison-agent", finished["stop_reason"])
    assert result["complete"] and result["calls"] == len(log.read("comparison-agent")) == 2
    assert result["best_cd"] == finished["best_cd"]
