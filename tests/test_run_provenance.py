"""Reproducible identity is additive event metadata, not a contract revision."""

from fastapi.testclient import TestClient

from src.agent import designer
from src.service.app import create_app
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog
from src.service.settings import Settings


def test_metadata_cannot_replace_authoritative_run_fields(tmp_path):
    log = RunLog(tmp_path)
    identity = {"checkpoint_sha256": "a" * 64, "synthetic": False}
    designer.run(1, "identity", True, evaluator=designer.predictor_evaluator(StubPredictor()),
                 model_provenance=identity, log=log, metadata={"type": "forged", "model": "forged",
                                                            "evaluation_budget": 999, "note": "retained"})
    start = log.read_events("identity")[0]
    assert start["type"] == "run_started" and start["model"] is None
    assert start["evaluation_budget"] == 1 and start["note"] == "retained"
    assert start["model_provenance"] == identity


def test_cli_records_service_identity_once(monkeypatch, tmp_path):
    log = RunLog(tmp_path)
    identity = {"checkpoint_sha256": "a" * 64, "synthetic": False}
    calls = []

    def health():
        calls.append(1)
        return {"predictor": "torch", "synthetic": False, "provenance": identity}

    monkeypatch.setattr(designer, "service_health", health)
    monkeypatch.setattr(designer, "service_thresholds", lambda: {})
    monkeypatch.setattr(designer, "evaluate_design", designer.predictor_evaluator(StubPredictor()))
    designer.run(1, "http-identity", True, log=log)
    assert calls == [1]
    assert log.read_events("http-identity")[0]["model_provenance"] == identity


def test_health_preserves_short_hash_and_adds_full_identity(tmp_path):
    predictor = StubPredictor()
    predictor.checkpoint_sha256 = "a" * 12
    predictor.checkpoint_sha256_full = "a" * 64
    with TestClient(create_app(predictor, settings=Settings(runlog_dir=str(tmp_path)))) as client:
        health = client.get("/health").json()
        assert health["checkpoint_sha256"] == "a" * 12
        assert health["provenance"]["checkpoint_sha256"] == "a" * 64
        assert health["provenance"]["synthetic"] is True
