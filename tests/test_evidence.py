"""Evidence claims must be reconstructable from real-identity logs, offline."""

import json

import pytest

from src.agent.evidence import summarize_evidence


class Log:
    def __init__(self, rows, *, synthetic=False, digest="a" * 64, finish=True):
        self.rows = rows
        self.events = [{"type": "run_started", "model": "fake/nemotron", "evaluation_budget": 3,
                        "body_style": "fastback", "constraints": [],
                        "model_provenance": {"synthetic": synthetic, "checkpoint_sha256": digest}}]
        if finish:
            self.events.append({"type": "run_finished", "stop_reason": "evaluation budget used",
                                "scored": len(rows), "usage": {"cost_usd": 0.002}})

    def read(self, run_id):
        return self.rows

    def read_events(self, run_id):
        return self.events


def row(i, cd, accepted=True):
    return {"iteration": i, "cd": cd, "accepted": accepted,
            "run_id": "example", "body_style": "fastback",
            "reject_reason": None if accepted else "OOD: unlike training designs",
            "params": {"example": i}}


def test_evidence_uses_only_trusted_scores_and_preserves_rejection_followup():
    rows = [row(0, 0.3), row(1, 0.1, False), row(2, 0.29)]
    result = summarize_evidence(Log(rows), "example")
    assert result["best_trusted"] == rows[2]
    assert result["delta_cd"] == pytest.approx(-0.01)
    assert result["relative_change"] == pytest.approx(-0.01 / 0.3)
    assert result["first_trusted_improvement"] == rows[2]
    assert result["rejections"][0]["following_scored_proposal"] == rows[2]
    assert result["complete"]


def test_evidence_no_improvement_or_adaptation_is_not_invented():
    result = summarize_evidence(Log([row(0, 0.3), row(1, 0.31), row(2, 0.1, False)]), "example")
    assert result["delta_cd"] == 0
    assert result["first_trusted_improvement"] is None
    assert result["rejections"][0]["following_scored_proposal"] is None


def test_rejected_baseline_cannot_support_improvement_claim():
    result = summarize_evidence(Log([row(0, 0.3, False), row(1, 0.29)], finish=False), "example")
    assert result["delta_cd"] is None and result["relative_change"] is None
    assert result["first_trusted_improvement"] is None
    assert not result["complete"] and result["stop_reason"] == "unfinished"


@pytest.mark.parametrize("kwargs", [{"synthetic": True}, {"synthetic": None}, {"digest": "a" * 12},
                                    {"digest": "z" * 64}])
def test_evidence_refuses_unattributed_or_stub_results(kwargs):
    with pytest.raises(ValueError):
        summarize_evidence(Log([row(0, 0.3)], **kwargs), "example")


def test_evidence_cli_qualification_failure_never_constructs_provider(tmp_path, monkeypatch):
    from src.agent.evidence import main
    from src.agent.provider import TokenFactoryClient
    from src.service import qualify

    calls = []
    monkeypatch.setattr(TokenFactoryClient, "from_env", lambda **kw: calls.append(kw))
    monkeypatch.setattr(qualify, "qualify_service",
                        lambda checkpoint, manifest: {"passed": False, "errors": ["wrong hash"]})
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps({}), encoding="utf-8")
    output = tmp_path / "evidence.json"
    assert main(["--checkpoint", str(tmp_path / "missing.pt"), "--references", str(refs),
                 "--model", "nano", "--run-id", "real-nano", "--out", str(output)]) == 1
    assert calls == [] and not output.exists()


def test_evidence_cli_never_overwrites_existing_output(tmp_path):
    from src.agent.evidence import main

    output = tmp_path / "evidence.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--checkpoint", "missing", "--references", "missing", "--model", "nano",
              "--run-id", "real-nano", "--out", str(output)])
    assert output.read_text(encoding="utf-8") == "existing"


def test_evidence_cli_refuses_large_budget_before_any_execution(tmp_path):
    from src.agent.evidence import main

    with pytest.raises(SystemExit):
        main(["--checkpoint", "missing", "--references", "missing", "--model", "super",
              "--run-id", "real-super", "--out", str(tmp_path / "evidence.json"), "--run-budget", "1"])


def test_summary_does_not_call_a_torn_or_inconsistent_tail_complete():
    log = Log([row(0, 0.3), row(1, 0.29)])
    log.events[-1]["scored"] = 3
    assert not summarize_evidence(log, "example")["complete"]
    log.rows[1]["iteration"] = 5
    with pytest.raises(ValueError, match="ordering"):
        summarize_evidence(log, "example")


def test_reused_event_identity_is_refused_even_if_scored_log_was_lost(tmp_path):
    from src.service.runlog import RunLog

    log = RunLog(tmp_path)
    log.append_event("event-only", {"type": "run_started"})
    assert log.exists("event-only")
    with pytest.raises(FileExistsError, match="event log"):
        log.reserve("event-only")


@pytest.fixture
def evidence_cli(monkeypatch, tmp_path):
    from src.agent.provider import TokenFactoryClient
    from src.service import predictor, provenance, qualify
    from src.service.predictor import StubPredictor
    from tests.test_nemotron import FakeClient, evaluate_call

    identity = {"synthetic": False, "checkpoint_sha256": "a" * 64}
    monkeypatch.setattr(predictor, "TorchPredictor", lambda path: StubPredictor())
    monkeypatch.setattr(provenance, "describe_predictor", lambda predictor: identity)
    monkeypatch.setattr(qualify, "qualify_service", lambda path, refs: {
        "passed": True, "errors": [], "provenance": identity, "results": []})
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": i * 0.1})] for i in range(1, 10)])
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return client

    monkeypatch.setattr(TokenFactoryClient, "from_env", factory)
    refs = tmp_path / "references.json"
    refs.write_text("{}", encoding="utf-8")
    output = tmp_path / "evidence.json"
    args = ["--checkpoint", "offline-fixture", "--references", str(refs), "--model", "nano",
            "--run-id", "evidence-cli", "--log-dir", str(tmp_path / "logs"), "--out", str(output)]
    return args, output, client, constructed


def test_evidence_cli_complete_trace_is_success(evidence_cli):
    from src.agent.evidence import main

    args, output, client, constructed = evidence_cli
    assert main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] and report["scored"] == 10
    assert len(client.sent) == 9 and len(constructed) == 1


def test_evidence_cli_provider_stop_is_saved_but_not_success(evidence_cli):
    from src.agent.evidence import main
    from src.agent.provider import ProviderError

    args, output, client, constructed = evidence_cli
    client.replies = [ProviderError("offline failure")]
    assert main(args) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["complete"] and report["stop_reason"].startswith("provider error")
    assert report["scored"] == 1 and len(client.sent) == 1


def test_evidence_cli_event_only_collision_stops_before_provider(evidence_cli, tmp_path):
    from src.agent.evidence import main
    from src.service.runlog import RunLog

    args, output, client, constructed = evidence_cli
    log = RunLog(tmp_path / "logs")
    log.append_event("evidence-cli", {"type": "run_started"})
    assert main(args) == 1
    assert not output.exists() and not constructed and not client.sent
