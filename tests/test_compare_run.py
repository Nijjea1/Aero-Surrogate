"""Exercise the comparison runner/CLI without checkpoints, providers or spend."""

import copy
import json
from dataclasses import replace

import pytest

from src.agent import compare_run
from src.agent.comparison import BudgetedEvaluator
from src.agent.constraints import parse_constraint
from src.agent.nemotron import NemotronProposer
from src.agent.provider import ProviderError
from src.contracts import Design, Evaluation, default_params
from src.service.runlog import RunLog
from tests.test_nemotron import FakeClient, evaluate_call


class FakePredictor:
    name = "torch"
    model_version = "offline-fixture"

    def __init__(self, *, reject_baseline=False, fail_proposal=False):
        self.seen = []
        self.reject_baseline = reject_baseline
        self.fail_proposal = fail_proposal

    def evaluate(self, design):
        self.seen.append(design)
        if self.fail_proposal and design.params != default_params():
            raise RuntimeError("offline predictor failure")
        sigma = .03 if self.reject_baseline and design.params == default_params() else .002
        return Evaluation(cd=.3 - design.params["B_Ramp_Angle"] * .001, cl=0., uncertainty=sigma,
                          in_distribution=True, ood_score=1., model_version=self.model_version)


@pytest.fixture(autouse=True)
def forbid_live_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("comparison tests must use FakeClient; live provider creation is forbidden")

    monkeypatch.setattr(compare_run.designer.TokenFactoryClient, "from_env", forbidden)


@pytest.fixture
def identity(monkeypatch):
    value = {
        "synthetic": False, "predictor": "torch", "checkpoint_sha256": "a" * 64,
        "model_version": "offline-fixture", "git_revision": "b" * 40,
        "uncertainty_threshold": .01, "ood_score_threshold": 10.,
        "source_dirty": False,
    }
    monkeypatch.setattr(compare_run, "describe_predictor", lambda predictor: copy.deepcopy(value))
    return value


def proposer_for(protocol, replies=None):
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 1.})]] if replies is None else replies)
    proposer = NemotronProposer(client, constraints=[parse_constraint(c) for c in protocol.constraints],
                                thinking=False)
    return proposer, client


def test_successful_agent_is_complete_and_records_identity_before_spending(identity, tmp_path):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2,
                                        constraints=("B_Ramp_Angle >= 0",))
    proposer, client = proposer_for(protocol)
    log = RunLog(tmp_path)
    report = compare_run.run_agent(predictor, protocol, "matched-agent", proposer=proposer, log=log)
    assert report["complete"] and report["calls"] == 2 and len(client.sent) == 1
    assert report["best_cd"] == pytest.approx(.299)
    assert report["delta_cd"] == pytest.approx(-.001)
    assert report["metadata"]["usage"]["calls"] == 1
    assert report["metadata"]["model_provenance"] == identity
    started = log.read_events("matched-agent")[0]
    assert started["comparison_protocol"] == protocol.to_dict()
    assert started["protocol_id"] == protocol.protocol_id
    assert started["model_provenance"]["checkpoint_sha256"] == "a" * 64
    assert all(design.body_style == protocol.body_style for design in predictor.seen)


def test_rejected_baseline_is_recorded_and_stops_before_provider(identity, tmp_path):
    predictor = FakePredictor(reject_baseline=True)
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol)
    report = compare_run.run_agent(predictor, protocol, "rejected-baseline", proposer=proposer,
                                   log=RunLog(tmp_path))
    assert not report["complete"] and report["calls"] == len(predictor.seen) == 1
    assert not client.sent and client.usage.calls == 0
    assert "baseline rejected" in report["stop_reason"]
    assert report["baseline_cd"] is report["delta_cd"] is report["improvement_fraction"] is None
    assert report["counts"] == {"accepted": 0, "rejected": 1, "error": 0}


def test_custom_admission_budget_does_not_change_predictor_identity(identity, tmp_path):
    predictor = FakePredictor()
    protocol = replace(compare_run.protocol_for(predictor, evaluation_budget=2), max_attempts=4)
    proposer, client = proposer_for(protocol)
    result = compare_run.run_agent(predictor, protocol, "custom-attempts", proposer=proposer,
                                   log=RunLog(tmp_path))
    assert result["complete"] and len(client.sent) == 1


def test_event_only_run_identity_cannot_be_reused(identity, tmp_path):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol)
    log = RunLog(tmp_path)
    log.append_event("existing-events", {"type": "run_started"})
    with pytest.raises(ValueError, match="exists"):
        compare_run.run_agent(predictor, protocol, "existing-events", proposer=proposer, log=log)
    assert not client.sent and not predictor.seen


def test_proposer_feasibility_rules_must_match_protocol(identity, tmp_path):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2, constraints=("B_Ramp_Angle >= 0",))
    proposer, client = proposer_for(protocol)
    proposer.constraints = []
    with pytest.raises(ValueError, match="constraints"):
        compare_run.run_agent(predictor, protocol, "wrong-rules", proposer=proposer, log=RunLog(tmp_path))
    assert not predictor.seen and not client.sent


def test_changed_source_fingerprint_prevents_matched_execution(identity, tmp_path):
    identity["source_sha256"] = "c" * 64
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    identity["source_sha256"] = "d" * 64
    proposer, client = proposer_for(protocol)
    with pytest.raises(ValueError, match="differ"):
        compare_run.run_agent(predictor, protocol, "changed-source", proposer=proposer, log=RunLog(tmp_path))
    assert not predictor.seen and not client.sent


def test_public_optimiser_binding_checks_real_identity(identity):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    score = compare_run.evaluator_for(predictor, protocol)
    assert score(protocol.baseline_params, protocol.body_style)["cd"] == .3
    identity["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differ"):
        compare_run.evaluator_for(predictor, protocol)


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "c" * 64), ("model_version", "another-model"),
    ("git_revision", "c" * 40), ("uncertainty_threshold", .02), ("ood_score_threshold", 5.),
])
def test_changed_serving_identity_refused_before_provider_or_scoring(identity, tmp_path, field, value):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol)
    identity[field] = value
    with pytest.raises(ValueError, match="differ"):
        compare_run.run_agent(predictor, protocol, "changed", proposer=proposer, log=RunLog(tmp_path))
    assert not client.sent and not predictor.seen
    assert not list(tmp_path.glob("*.jsonl"))


@pytest.mark.parametrize("field,value", [
    ("synthetic", True), ("checkpoint_sha256", None), ("ood_score_threshold", None),
])
def test_protocol_requires_real_identified_detector(identity, field, value):
    identity[field] = value
    with pytest.raises(ValueError, match="requires"):
        compare_run.protocol_for(FakePredictor())


def test_provider_failure_returns_partial_summary_without_extra_scoring(identity, tmp_path):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol, [ProviderError("offline provider failure")])
    report = compare_run.run_agent(predictor, protocol, "provider-failure", proposer=proposer,
                                   log=RunLog(tmp_path))
    assert not report["complete"] and report["calls"] == 1 and len(client.sent) == 1
    assert report["stop_reason"].startswith("provider error")
    assert report["best_curve"] == [.3] and len(report["outcomes"]) == 1
    assert report["metadata"]["usage"]["calls"] == 0


def test_predictor_failure_is_accounted_even_without_iteration_record(identity, tmp_path):
    predictor = FakePredictor(fail_proposal=True)
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol)
    log = RunLog(tmp_path)
    report = compare_run.run_agent(predictor, protocol, "predictor-failure", proposer=proposer, log=log)
    assert not report["complete"] and report["calls"] == len(predictor.seen) == 2
    assert report["counts"]["error"] == 1 and len(log.read("predictor-failure")) == 1
    assert report["metadata"]["error"] == "RuntimeError: offline predictor failure"
    assert len(client.sent) == 1


def test_cancel_before_baseline_has_no_scoring_or_provider_spend(identity, tmp_path):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol)
    report = compare_run.run_agent(predictor, protocol, "cancelled", proposer=proposer,
                                   should_stop=lambda: True, log=RunLog(tmp_path))
    assert report["stop_reason"] == "cancelled" and report["calls"] == 0
    assert not report["complete"] and not predictor.seen and not client.sent


def test_existing_run_id_refused_without_spending(identity, tmp_path):
    predictor = FakePredictor()
    protocol = compare_run.protocol_for(predictor, evaluation_budget=2)
    proposer, client = proposer_for(protocol)
    log = RunLog(tmp_path)
    compare_run.run_agent(predictor, protocol, "same-id", proposer=proposer, log=log)
    before = list(log.read("same-id"))
    next_proposer, next_client = proposer_for(protocol)
    with pytest.raises(ValueError, match="run ID exists"):
        compare_run.run_agent(predictor, protocol, "same-id", proposer=next_proposer, log=log)
    assert not next_client.sent and log.read("same-id") == before


def pair_files(tmp_path, protocol, *, partial=False):
    paths = []
    for method in ("agent", "random"):
        predictor = FakePredictor()
        wrapped = BudgetedEvaluator(lambda p, s: predictor.evaluate(Design(p, s)).to_dict(), protocol)
        wrapped(default_params(), protocol.body_style)
        reason = "cancelled" if partial and method == "agent" else "evaluation budget used"
        if reason == "evaluation budget used":
            wrapped(dict(default_params(), B_Ramp_Angle=1.), protocol.body_style)
        summary = wrapped.summary(method, method, reason, metadata={"seed": 0} if method == "random" else {})
        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("partial,exit_code", [(False, 0), (True, 1)])
def test_cli_pair_writes_honest_complete_or_partial_artifact(identity, tmp_path, partial, exit_code):
    protocol = compare_run.protocol_for(FakePredictor(), evaluation_budget=2)
    left, right = pair_files(tmp_path, protocol, partial=partial)
    output = tmp_path / "pair.json"
    assert compare_run.main(["pair", left, right, "--out", str(output)]) == exit_code
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] is (not partial)
    assert report["runs"][0]["calls"] == (1 if partial else 2)


def test_cli_refuses_output_overwrite_before_reading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep this exact content", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        compare_run.main(["pair", "missing-left.json", "missing-right.json", "--out", str(output)])
    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep this exact content"


def test_cli_mismatched_pair_does_not_create_output(identity, tmp_path):
    protocol = compare_run.protocol_for(FakePredictor(), evaluation_budget=2)
    left, right = pair_files(tmp_path, protocol)
    summary = json.loads((tmp_path / "random.json").read_text(encoding="utf-8"))
    summary["protocol_id"] = "wrong"
    (tmp_path / "random.json").write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "pair.json"
    assert compare_run.main(["pair", left, right, "--out", str(output)]) == 1
    assert not output.exists()
