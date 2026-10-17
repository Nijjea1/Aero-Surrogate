"""The live reliability rehearsal, against the real app with a fake model: no network, no spend."""

import json

import pytest
from fastapi.testclient import TestClient

from src.service.rehearse import main, rehearse
from tests.test_runs import TOKEN, Factory, logdir, make_app  # noqa: F401  (fixture)

FAST = {"poll_s": 0.01, "run_timeout_s": 5.0, "iterations": 3, "allow_synthetic": True}


def test_runs_finish_rotate_styles_and_never_record_the_token(logdir):
    factory = Factory()
    with TestClient(make_app(logdir, factory)) as client:
        report = rehearse(client, TOKEN, runs=4, **FAST)
    assert report["passed"], report["errors"]
    assert [run["body_style"] for run in report["runs"]] == ["fastback", "notchback", "estateback", "fastback"]
    assert all(run["history_rows"] == 3 and run["stop_reason"] == "evaluation budget used" for run in report["runs"])
    assert report["summary"]["complete"] == 4 and report["summary"]["success_rate"] == 1.0
    assert report["cancel_check"]["started_as"] == "public" and report["cancel_check"]["passed"]
    assert report["summary"]["spent_usd"] > 0
    assert TOKEN not in json.dumps(report)


def test_synthetic_service_is_refused_before_any_run(logdir):
    factory = Factory()
    with TestClient(make_app(logdir, factory)) as client:
        report = rehearse(client, TOKEN, runs=2, **dict(FAST, allow_synthetic=False))
    assert not report["passed"] and "synthetic" in report["errors"][0]
    assert factory.requests == []


def test_spend_cap_stops_new_runs(logdir):
    # Each fake call costs $0.001; a 3-evaluation run makes two calls.
    with TestClient(make_app(logdir, Factory())) as client:
        report = rehearse(client, TOKEN, runs=5, max_spend_usd=0.003, check_cancel=False, **FAST)
    assert len(report["runs"]) == 2
    assert not report["passed"] and "reached the cap" in report["errors"][0]


def test_failed_runs_fail_the_rehearsal(logdir):
    with TestClient(make_app(logdir, Factory(fail=RuntimeError("model exploded")))) as client:
        report = rehearse(client, TOKEN, runs=2, check_cancel=False, **FAST)
    assert not report["passed"]
    assert [run["state"] for run in report["runs"]] == ["failed", "failed"]
    assert "2 run(s) did not complete" in report["errors"]


def test_cancel_check_uses_the_admin_when_public_runs_are_closed(logdir):
    with TestClient(make_app(logdir, Factory(), public_runs=False)) as client:
        report = rehearse(client, TOKEN, runs=1, **FAST)
    assert report["cancel_check"]["started_as"] == "admin" and report["cancel_check"]["passed"]
    assert report["passed"], report["errors"]


def test_cli_needs_the_token_in_the_environment_and_a_fresh_output(tmp_path, monkeypatch):
    monkeypatch.delenv("AERO_ADMIN_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main(["--service-url", "http://127.0.0.1:9", "--out", str(tmp_path / "r.json")])
    monkeypatch.setenv("AERO_ADMIN_TOKEN", TOKEN)
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit):
        main(["--service-url", "http://127.0.0.1:9", "--out", str(existing)])
    with pytest.raises(SystemExit):
        main(["--service-url", "http://127.0.0.1:9", "--runs", "0", "--out", str(tmp_path / "r.json")])
