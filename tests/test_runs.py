"""The run-control API: start, poll, cancel, limits, restarts. Fake models only; no network, no spend."""

import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.agent import designer
from src.agent.nemotron import NemotronProposer
from src.agent.provider import ProviderError, Usage
from src.service.app import create_app
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog
from src.service.runs import INTERRUPTED, TERMINAL, RunManager
from src.service.settings import Settings
from tests.test_nemotron import evaluate_call, stub_evaluator

TOKEN = "test-admin-token-0123456789"
ADMIN = {"Authorization": f"Bearer {TOKEN}"}


class ScriptClient:
    """Answers every call with a new valid one-parameter design."""

    model = "fake/nemotron"

    def __init__(self, gate: threading.Event | None = None, fail: Exception | None = None):
        self.usage = Usage()
        self.calls = 0
        self.gate = gate
        self.entered = threading.Event()
        self.fail = fail

    def chat(self, messages, **kwargs):
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5), "test never released the gate"
        if self.fail is not None:
            raise self.fail
        self.calls += 1
        self.usage.add(100, 50, 0.001)
        reply = {"role": "assistant", "content": "",
                 "tool_calls": [evaluate_call({"B_Ramp_Angle": 0.5 * self.calls}, call_id=f"c{self.calls}")]}
        return {"message": reply, "finish_reason": "tool_calls", "prompt_tokens": 100,
                "completion_tokens": 50, "cost_usd": 0.001, "latency_s": 0.0}


class Factory:
    """Records what it was asked for and hands out ScriptClient proposers."""

    def __init__(self, gate=None, fail=None, error=None):
        self.gate, self.fail, self.error = gate, fail, error
        self.clients, self.requests = [], []

    def __call__(self, model, constraints, thinking):
        self.requests.append((model, list(constraints), thinking))
        if self.error is not None:
            raise self.error
        client = ScriptClient(self.gate, self.fail)
        self.clients.append(client)
        return NemotronProposer(client, constraints=list(constraints), thinking=thinking == "on")


class Clock:
    """UTC time that advances one minute per reading, so start order is unambiguous."""

    def __init__(self, start=None):
        self.now = start or datetime.now(timezone.utc).replace(microsecond=0)

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


@pytest.fixture
def logdir(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    return tmp_path


def make_app(logdir, factory=None, clock=None, **overrides):
    settings = Settings(admin_token=TOKEN, runlog_dir=str(logdir), **overrides)
    predictor, runlog = StubPredictor(), RunLog(str(logdir))
    app = create_app(predictor, runlog, settings)
    app.state.run_manager = RunManager(predictor, runlog, settings, proposer_factory=factory or Factory(),
                                       clock=clock or Clock())
    return app


@pytest.fixture
def gate():
    event = threading.Event()
    yield event
    event.set()  # never leave a run thread blocked


def wait_for(client, run_id, states=TERMINAL, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        status = client.get(f"/runs/{run_id}").json()
        if status["state"] in states:
            return status
        assert time.monotonic() < deadline, f"run stuck in {status['state']}"
        time.sleep(0.01)


def start(client, headers=None, **body):
    return client.post("/runs", json={"model": "nano", "iterations": 3, **body}, headers=headers or {})


def test_run_starts_in_background_and_finishes(logdir):
    factory = Factory()
    with TestClient(make_app(logdir, factory)) as client:
        response = start(client, iterations=4, body_style="notchback",
                         constraints=["A_Car_Roof_Height >= 0"])
        assert response.status_code == 202
        started = response.json()
        assert started["state"] in ("queued", "running") and started["run_id"].startswith("agent-")
        status = wait_for(client, started["run_id"])
        assert status["state"] == "finished" and status["stop_reason"] == "evaluation budget used"
        assert status["scored"] == 4 and status["best_cd"] is not None
        assert status["cost_usd"] == pytest.approx(0.003)  # three model calls after the baseline
        assert status["timing"]["steps"] == 4 and status["timing"]["model_calls"] == 3
        assert status["latest_step"]["iteration"] == 3 and status["latest_step"]["reasoning"]
        assert status["started_by"] == "public" and status["requested_via"] == "api"
        assert status["body_style"] == "notchback" and status["iterations"] == 4
        assert status["constraints"] == ["A_Car_Roof_Height >= 0"]
        assert status["run_budget_usd"] == Settings().public_run_budget_usd
        history = client.get(f"/history/{started['run_id']}").json()
        assert [row["iteration"] for row in history] == [0, 1, 2, 3]
        assert min(r["cd"] for r in history if r["accepted"]) == status["best_cd"]
        assert factory.requests[0][0] == "nano" and factory.requests[0][2] == "auto"
        assert [str(c) for c in factory.requests[0][1]] == ["A_Car_Roof_Height >= 0"]


def test_cancel_mid_run(logdir, gate):
    factory = Factory(gate=gate)
    with TestClient(make_app(logdir, factory)) as client:
        run_id = start(client, iterations=5).json()["run_id"]
        assert wait_until(lambda: factory.clients and factory.clients[0].entered.is_set())
        live = client.get(f"/runs/{run_id}").json()  # baseline scored; model call in flight
        assert live["state"] == "running" and live["scored"] == 1 and live["cost_usd"] == 0.0
        assert live["latest_step"]["iteration"] == 0 and live["timing"]["steps"] == 1
        cancelled = client.post(f"/runs/{run_id}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["cancel_requested"]
        gate.set()
        status = wait_for(client, run_id)
        assert status["state"] == "cancelled" and status["stop_reason"] == "cancelled"
        assert status["scored"] < 5
        # Cancelling a finished run is a no-op that returns its status.
        again = client.post(f"/runs/{run_id}/cancel")
        assert again.status_code == 200 and again.json()["state"] == "cancelled"


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_concurrency_limit(logdir, gate):
    factory = Factory(gate=gate)
    with TestClient(make_app(logdir, factory, max_concurrent_runs=1)) as client:
        first = start(client, headers=ADMIN).json()["run_id"]
        refused = start(client, headers=ADMIN)
        assert refused.status_code == 429 and "limit is 1" in refused.json()["detail"]
        gate.set()
        wait_for(client, first)
        assert start(client, headers=ADMIN).status_code == 202


@pytest.mark.parametrize("body,fragment", [
    ({"iterations": 26}, "above the public limit of 25"),
    ({"run_budget_usd": 0.2}, "above the public limit"),
    ({"run_budget_usd": 0.0}, "positive"),
    ({"iterations": 1}, ">= 2"),
    ({"iterations": "3"}, None),
    ({"iterations": True}, None),
    ({"model": "ultra"}, "model must be one of"),
    ({"body_style": "pickup"}, "unknown body style"),
    ({"thinking": "maybe"}, "thinking"),
    ({"constraints": ["roof >= 0"]}, "unknown parameter"),
    ({"constraints": ["A_Car_Roof_Height >= 0"] * 9}, "at most 8"),
    ({"surprise": 1}, None),
])
def test_invalid_or_over_ceiling_requests_are_refused(logdir, body, fragment):
    factory = Factory()
    with TestClient(make_app(logdir, factory)) as client:
        response = start(client, **body)
        assert response.status_code == 422
        if fragment:
            assert fragment in response.json()["detail"]
    assert factory.requests == []  # refused before a provider was built


def test_admin_token_lifts_public_limits(logdir):
    with TestClient(make_app(logdir)) as client:
        assert start(client, iterations=30).status_code == 422
        ok = start(client, headers=ADMIN, iterations=30, run_budget_usd=0.4)
        assert ok.status_code == 202 and ok.json()["started_by"] == "admin"
        assert start(client, headers=ADMIN, iterations=51).status_code == 422
        assert start(client, headers=ADMIN, run_budget_usd=0.6).status_code == 422
        wait_for(client, ok.json()["run_id"])


def test_public_runs_disabled_needs_the_token(logdir):
    with TestClient(make_app(logdir, public_runs=False)) as client:
        refused = start(client)
        assert refused.status_code == 401 and refused.headers["WWW-Authenticate"] == "Bearer"
        ok = start(client, headers=ADMIN)
        assert ok.status_code == 202
        assert client.post(f"/runs/{ok.json()['run_id']}/cancel").status_code == 401
        wait_for(client, ok.json()["run_id"])  # reading stays open


def test_wrong_token_is_refused_not_treated_as_public(logdir):
    with TestClient(make_app(logdir)) as client:
        for headers in ({"Authorization": "Bearer wrong-token-0123456789"}, {"Authorization": TOKEN}):
            response = start(client, headers=headers)
            assert response.status_code == 401 and response.headers["WWW-Authenticate"] == "Bearer"
        assert client.post("/runs/agent-000000000000/cancel", headers={"Authorization": "Bearer x"}).status_code == 401
        assert client.get("/runs").json() == []


def test_public_daily_budget_survives_a_restart(logdir, gate):
    day = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)
    limits = dict(public_daily_budget_usd=0.15, public_run_budget_usd=0.10, max_concurrent_runs=2)
    first = Factory(gate=gate)
    with TestClient(make_app(logdir, first, Clock(day), **limits)) as old:
        in_flight = start(old).json()["run_id"]
        assert wait_until(lambda: first.clients and first.clients[0].entered.is_set())
        # "Restart": a second service over the same logs, with no memory of the first.
        with TestClient(make_app(logdir, Factory(), Clock(day), **limits)) as new:
            refused = start(new)
            assert refused.status_code == 429 and "daily budget" in refused.json()["detail"]
            # The in-flight run is reserved at its full budget; admin runs do not count.
            assert new.app.state.run_manager.public_spend_today() == pytest.approx(0.10)
            admin = start(new, headers=ADMIN)
            assert admin.status_code == 202
            wait_for(new, admin.json()["run_id"])
            gate.set()
            wait_for(old, in_flight)
            # Finished: only what it actually cost (two calls) still counts.
            assert new.app.state.run_manager.public_spend_today() == pytest.approx(0.002)
            later = start(new)
            assert later.status_code == 202
            wait_for(new, later.json()["run_id"])
    tomorrow = RunManager(StubPredictor(), RunLog(str(logdir)), Settings(**limits),
                          proposer_factory=Factory(), clock=Clock(day + timedelta(days=1)))
    assert tomorrow.public_spend_today() == 0.0


def test_orphans_are_closed_on_startup_only_when_stale(logdir):
    log = RunLog(str(logdir))
    for run_id in ("old-run", "new-run"):
        log.append_event(run_id, {"type": "run_started", "model": "fake", "body_style": "fastback",
                                  "evaluation_budget": 5, "run_budget_usd": 0.1, "constraints": [],
                                  "started_by": "public", "started_at": "2026-10-14T08:00:00Z"})
        log.append_event(run_id, {"type": "step", "iteration": 0, "propose_s": 0.0, "model_calls": 0,
                                  "model_latency_s": 0.0, "score_s": 0.1, "surrogate_latency_ms": 1.0,
                                  "step_s": 0.1})
    stale = time.time() - 3600
    os.utime(logdir / "events" / "old-run.jsonl", (stale, stale))
    with TestClient(make_app(logdir)) as client:
        assert client.app.state.run_manager.closed_orphans == ["old-run"]
        old = client.get("/runs/old-run").json()
        assert old["state"] == "interrupted" and old["stop_reason"] == INTERRUPTED
        assert old["cost_usd"] is None and old["timing"]["steps"] == 1
        new = client.get("/runs/new-run").json()
        assert new["state"] == "unknown" and new["stop_reason"] is None
        assert client.post("/runs/new-run/cancel").status_code == 409  # not ours to stop
    assert [e["type"] for e in log.read_events("new-run")] == ["run_started", "step"]
    assert log.read_events("old-run")[-1]["type"] == "run_finished"
    # Closing is idempotent: a second startup finds nothing to close.
    again = RunManager(StubPredictor(), log, Settings(), proposer_factory=Factory())
    assert again.closed_orphans == []


def test_provider_not_configured_is_503_and_leaves_nothing_behind(logdir):
    factory = Factory(error=ProviderError("NEBIUS_API_KEY is not set"))
    with TestClient(make_app(logdir, factory)) as client:
        response = start(client)
        assert response.status_code == 503 and "NEBIUS_API_KEY" in response.json()["detail"]
        assert client.get("/runs").json() == []
    assert list(logdir.rglob("*.jsonl")) == []


def test_runs_are_written_to_the_services_log_whatever_the_environment_says(
        logdir, monkeypatch, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(elsewhere))
    with TestClient(make_app(logdir)) as client:
        response = start(client)
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        wait_for(client, run_id)
    assert (logdir / f"{run_id}.jsonl").exists()
    assert not list(elsewhere.rglob("*.jsonl"))


def test_exception_in_the_run_thread_marks_it_failed(logdir):
    with TestClient(make_app(logdir, Factory(fail=RuntimeError("model exploded")))) as client:
        run_id = start(client).json()["run_id"]
        status = wait_for(client, run_id)
        assert status["state"] == "failed"
        assert "model exploded" in status["error"] and status["stop_reason"].startswith("error:")
        assert status["scored"] == 1  # the baseline, scored before the first model call
        assert start(client).status_code == 202  # a failed run frees its slot


def test_unknown_and_invalid_ids(logdir):
    with TestClient(make_app(logdir)) as client:
        assert client.get("/runs/agent-000000000000").status_code == 404
        assert client.post("/runs/agent-000000000000/cancel").status_code == 404
        assert client.get("/runs/bad.id").status_code == 422
        assert client.get("/runs?limit=0").status_code == 422


def test_list_is_newest_first_and_includes_cli_runs(logdir):
    designer.run(3, "cli-run", True, evaluator=stub_evaluator())
    with TestClient(make_app(logdir)) as client:
        ids = []
        for _ in range(3):
            ids.append(start(client).json()["run_id"])
            wait_for(client, ids[-1])
        listed = [s["run_id"] for s in client.get("/runs").json()]
        assert listed == ids[::-1] + ["cli-run"]
        assert [s["run_id"] for s in client.get("/runs?limit=2").json()] == ids[::-1][:2]


def test_cli_started_run_is_visible(logdir):
    designer.run(3, "cli-run", True, evaluator=stub_evaluator())
    with TestClient(make_app(logdir)) as client:
        status = client.get("/runs/cli-run").json()
        assert status["state"] == "finished" and status["scored"] == 3
        assert status["started_by"] is None and status["cost_usd"] is None
        assert status["latest_step"]["iteration"] == 2 and status["timing"]["steps"] == 3
        assert client.post("/runs/cli-run/cancel").status_code == 403  # public: not a public run
        admin = client.post("/runs/cli-run/cancel", headers=ADMIN)
        assert admin.status_code == 200 and admin.json()["state"] == "finished"


def test_public_may_not_cancel_an_admin_run(logdir, gate):
    factory = Factory(gate=gate)
    with TestClient(make_app(logdir, factory)) as client:
        run_id = start(client, headers=ADMIN).json()["run_id"]
        assert client.post(f"/runs/{run_id}/cancel").status_code == 403
        assert client.post(f"/runs/{run_id}/cancel", headers=ADMIN).json()["cancel_requested"]
        gate.set()
        assert wait_for(client, run_id)["state"] == "cancelled"


def test_shutdown_cancels_active_runs(logdir, gate):
    factory = Factory(gate=gate)
    app = make_app(logdir, factory)
    with TestClient(app) as client:
        run_id = start(client, iterations=10).json()["run_id"]
        assert wait_until(lambda: factory.clients and factory.clients[0].entered.is_set())
        threading.Timer(0.05, gate.set).start()  # the in-flight call returns after the stop request
    status = app.state.run_manager.status(run_id)
    assert status["state"] == "cancelled" and status["scored"] < 10
