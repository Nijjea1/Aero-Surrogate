"""Regression tests for the Oct 15 independent review of the run API work."""

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.agent import designer
from src.agent.nemotron import NemotronProposer
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog
from src.service.runs import STOPPING, RunManager, state_from_reason
from src.service.settings import Settings
from tests.test_nemotron import FakeClient, evaluate_call, row, stub_evaluator, BASE, propose
from tests.test_runs import ADMIN, TOKEN, Clock, Factory, make_app, start, wait_for, wait_until

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def fixed(when):
    return lambda: when


def write_run(root, run_id, *, started_at, started_by="public", budget=0.10, finished=None, torn=False):
    events = root / "events"
    events.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "run_started", "run_id": run_id, "started_by": started_by,
                         "started_at": started_at, "run_budget_usd": budget})]
    if finished is not None:
        lines.append(json.dumps({"type": "run_finished", "run_id": run_id, **finished}))
    text = "\n".join(lines) + "\n"
    if torn:
        text += '{"type": "step", "iteration": 1, "timing": {"x": 1}'  # crash mid-append
    (events / f"{run_id}.jsonl").write_text(text, encoding="utf-8")
    (root / f"{run_id}.jsonl").write_text("", encoding="utf-8")


def manager(root, **settings):
    return RunManager(StubPredictor(), RunLog(str(root)),
                      Settings(admin_token=TOKEN, runlog_dir=str(root), **settings),
                      proposer_factory=Factory(), clock=fixed(NOW))


# ---- finding 1: a torn line can no longer hide a run from the daily budget


def test_crash_mid_append_then_orphan_closure_keeps_the_run_counted(tmp_path):
    write_run(tmp_path, "agent-torn", started_at="2026-10-15T09:00:00Z", torn=True)
    old = time.time() - 7200
    os.utime(tmp_path / "events" / "agent-torn.jsonl", (old, old))
    runs = manager(tmp_path, public_daily_budget_usd=0.10)
    assert runs.closed_orphans == ["agent-torn"]
    events = RunLog(str(tmp_path)).read_events("agent-torn")
    assert [e["type"] for e in events] == ["run_started", "run_finished"]  # readable, closed
    with runs._lock:
        assert runs.public_spend_today() == pytest.approx(0.10)  # unknown usage: full budget
    from src.service.runs import RunLimitError
    with pytest.raises(RunLimitError, match="daily budget"):
        runs.start({"model": "nano", "iterations": 3}, "public")


def test_unreadable_recent_log_counts_at_the_public_run_budget(tmp_path, monkeypatch):
    write_run(tmp_path, "agent-ok", started_at="2026-10-15T09:00:00Z",
              finished={"stop_reason": "evaluation budget used", "usage": {"cost_usd": 0.01}})
    runs = manager(tmp_path, public_run_budget_usd=0.05)
    real = runs.runlog.read_events

    def flaky(run_id):
        if run_id == "agent-ok":
            raise OSError("disk hiccup")
        return real(run_id)

    monkeypatch.setattr(runs.runlog, "read_events", flaky)
    runs._spend_cache.clear()
    with runs._lock:
        assert runs.public_spend_today() == pytest.approx(0.05)


# ---- finding 5/6: scan only recent logs; count runs still open across midnight


def test_daily_budget_counts_yesterdays_unfinished_public_runs_only(tmp_path):
    write_run(tmp_path, "agent-y-open", started_at="2026-10-14T23:59:00Z", budget=0.10)
    write_run(tmp_path, "agent-y-done", started_at="2026-10-14T22:00:00Z", budget=0.10,
              finished={"stop_reason": "evaluation budget used", "usage": {"cost_usd": 0.08}})
    write_run(tmp_path, "agent-today", started_at="2026-10-15T08:00:00Z",
              finished={"stop_reason": "cancelled", "usage": {"cost_usd": 0.02}})
    write_run(tmp_path, "agent-admin", started_at="2026-10-15T08:00:00Z", started_by="admin", budget=0.5)
    write_run(tmp_path, "agent-ancient", started_at="2026-10-15T08:00:00Z", budget=0.10,
              finished={"stop_reason": "evaluation budget used", "usage": {"cost_usd": 0.07}})
    ancient = (NOW - timedelta(days=3)).timestamp()
    os.utime(tmp_path / "events" / "agent-ancient.jsonl", (ancient, ancient))
    runs = manager(tmp_path)
    with runs._lock:
        # 0.10 (yesterday, still open) + 0.02 (today, billed); admin and the
        # untouched-for-days file are not part of today's public spend.
        assert runs.public_spend_today() == pytest.approx(0.12)


def test_spend_scan_reuses_unchanged_files(tmp_path, monkeypatch):
    write_run(tmp_path, "agent-a", started_at="2026-10-15T08:00:00Z",
              finished={"stop_reason": "evaluation budget used", "usage": {"cost_usd": 0.03}})
    runs = manager(tmp_path)
    reads = []
    real = runs.runlog.read_events
    monkeypatch.setattr(runs.runlog, "read_events", lambda run_id: reads.append(run_id) or real(run_id))
    with runs._lock:
        runs.public_spend_today()
        runs.public_spend_today()
    assert reads.count("agent-a") == 1


def test_run_listing_and_status_are_rate_limited(tmp_path):
    with TestClient(make_app(tmp_path, rate_limit_per_minute=2)) as client:
        assert client.get("/runs").status_code == 200
        assert client.get("/runs").status_code == 200
        limited = client.get("/runs")
        assert limited.status_code == 429 and "Retry-After" in limited.headers
        assert client.get("/health").status_code == 200


# ---- finding 2: timing keeps failed attempts; baseline does not skew medians


def test_failed_attempts_are_in_the_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    client = FakeClient(["no tool"] * 4 + [[evaluate_call({"B_Ramp_Angle": 2.0})]])
    designer.run(2, "slow", False, evaluator=stub_evaluator(), proposer=NemotronProposer(client),
                 provenance="synthetic stub")
    timing = RunLog(tmp_path).read_events("slow")[-1]["timing"]
    assert client.usage.calls == 5 and timing["model_calls"] == 5
    assert timing["failed_proposals"] == 1
    steps = [e for e in RunLog(tmp_path).read_events("slow") if e["type"] == "step"]
    assert steps[1]["model_calls"] == 5 and steps[1]["failed_proposals"] == 1


def test_model_time_of_a_run_that_never_scores_again_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    client = FakeClient(["no tool"] * 8)
    designer.run(5, "stalled", False, evaluator=stub_evaluator(), proposer=NemotronProposer(client),
                 provenance="synthetic stub")
    timing = RunLog(tmp_path).read_events("stalled")[-1]["timing"]
    assert timing["steps"] == 1 and timing["model_calls"] == 8 and timing["failed_proposals"] == 2


def test_medians_describe_live_steps_not_the_instant_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    ticks = iter(range(1000))
    monkeypatch.setattr(designer, "_now", lambda: float(next(ticks)) * 10)
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 2.0})]])
    designer.run(2, "medians", False, evaluator=stub_evaluator(), proposer=NemotronProposer(client),
                 provenance="synthetic stub")
    timing = RunLog(tmp_path).read_events("medians")[-1]["timing"]
    steps = [e for e in RunLog(tmp_path).read_events("medians") if e["type"] == "step"]
    assert [s["live"] for s in steps] == [False, True]
    assert timing["propose_s_median"] == steps[1]["propose_s"]


# ---- finding 3: stuck runs are closed on shutdown; late orphans close lazily


def test_shutdown_closes_runs_that_do_not_stop_in_time(tmp_path, gate):
    factory = Factory(gate=gate)
    app = make_app(tmp_path, factory)
    with TestClient(app) as client:
        run_id = start(client, iterations=5).json()["run_id"]
        assert wait_until(lambda: factory.clients and factory.clients[0].entered.is_set())
        runs = app.state.run_manager
        runs.shutdown(timeout_s=0.1)  # the model call is still blocked
        events = RunLog(str(tmp_path)).read_events(run_id)
        assert events[-1]["type"] == "run_finished" and events[-1]["stop_reason"] == STOPPING
        assert events[-1]["usage"]["calls"] == 0 and events[-1]["scored"] == 1
        assert client.get(f"/runs/{run_id}").json()["state"] == "interrupted"
        gate.set()


def test_an_orphan_that_goes_stale_later_is_closed_on_read(tmp_path):
    write_run(tmp_path, "agent-late", started_at="2026-10-15T09:00:00Z")
    runs = manager(tmp_path)
    assert runs.status("agent-late")["state"] == "unknown"  # fresh: may be live elsewhere
    runs.orphan_after_s = 0.0
    old = time.time() - 5
    os.utime(tmp_path / "events" / "agent-late.jsonl", (old, old))
    assert runs.status("agent-late")["state"] == "interrupted"


# ---- finding 4: public callers cannot lock the admin out


def test_admin_has_its_own_slots(tmp_path, gate):
    factory = Factory(gate=gate)
    with TestClient(make_app(tmp_path, factory)) as client:
        public = start(client, iterations=5)
        assert public.status_code == 202
        assert start(client, iterations=5).status_code == 429  # public slot taken
        admin = start(client, headers=ADMIN, iterations=5)
        assert admin.status_code == 202  # the admin is never locked out by the public
        assert start(client, headers=ADMIN, iterations=5).status_code == 429
        assert "cancel_token" not in admin.json()
        gate.set()
        wait_for(client, public.json()["run_id"])
        wait_for(client, admin.json()["run_id"])


# ---- finding 7: hypotheses judged only against a trusted base; no self-contradiction


def test_contradictory_expectation_is_refused():
    call = {"id": "c1", "type": "function", "function": {"name": "evaluate", "arguments": json.dumps({
        "changes": {"B_Ramp_Angle": 1.0}, "hypothesis": "h", "expected_effect": "lower_cd",
        "expected_delta_cd": 0.002})}}
    client = FakeClient([[call], [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert "contradicts" in proposal.feedback[0]


def test_untrusted_base_is_not_judged(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    from src.agent.constraints import parse_constraint
    rule = parse_constraint("A_Car_Roof_Height >= 10")  # the baseline breaks it: reference only
    client = FakeClient([[evaluate_call({"A_Car_Roof_Height": 12.0})]])
    designer.run(2, "untrusted-base", False, evaluator=stub_evaluator(),
                 proposer=NemotronProposer(client, constraints=[rule]), constraints=[rule],
                 provenance="synthetic stub")
    hits = RunLog(tmp_path).read_events("untrusted-base")[-1]["hypothesis_hits"]
    assert hits["judged"] == 0


# ---- finding 9: states say what happened


@pytest.mark.parametrize("reason, state", [
    ("evaluation budget used", "finished"),
    ("budget: run budget $0.10 reached", "finished"),
    ("cancelled", "cancelled"),
    ("provider error: HTTP 503", "failed"),
    ("2 consecutive steps without a valid proposal", "failed"),
    ("error: RuntimeError: boom", "failed"),
    ("interrupted: service restarted", "interrupted"),
    (None, "finished"),
])
def test_state_from_reason(reason, state):
    assert state_from_reason(reason) == state


@pytest.fixture
def gate():
    event = threading.Event()
    yield event
    event.set()
