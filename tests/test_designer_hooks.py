"""Loop hooks for the run API (stop, progress), step timing, in-process scoring."""

import itertools

import pytest

from src.agent import designer
from src.agent.nemotron import NemotronProposer
from src.contracts import OOD_SCORE_THRESHOLD, UNCERTAINTY_THRESHOLD, default_params
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog
from tests.test_nemotron import FakeClient, evaluate_call, stub_evaluator


@pytest.fixture
def logdir(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    return RunLog(tmp_path)


@pytest.fixture
def fake_clock(monkeypatch):
    """Every reading advances one second, so each interval is a known count."""
    ticks = itertools.count()
    monkeypatch.setattr(designer, "_now", lambda: float(next(ticks)))


def steps(log, run_id):
    return [e for e in log.read_events(run_id) if e["type"] == "step"]


def test_every_scored_step_reports_where_its_time_went(logdir, fake_clock):
    designer.run(4, "timed", True, evaluator=stub_evaluator())
    timing = steps(logdir, "timed")
    assert [t["iteration"] for t in timing] == [0, 1, 2, 3]
    # _now is read at step start, after proposing, after scoring, and at the end.
    assert all(t["propose_s"] == 1.0 and t["score_s"] == 1.0 and t["step_s"] == 3.0 for t in timing)
    assert all(t["model_calls"] == 0 and t["surrogate_latency_ms"] == 0.0 for t in timing)
    summary = logdir.read_events("timed")[-1]["timing"]
    assert summary["steps"] == 4 and summary["step_s_total"] == 12.0
    assert summary["propose_s_median"] == 1.0 and summary["model_share"] == pytest.approx(1 / 3)


def test_live_steps_count_model_calls_including_corrections(logdir):
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 99.0})],
                         [evaluate_call({"B_Ramp_Angle": 2.0}, call_id="c2")]])
    designer.run(2, "calls", False, evaluator=stub_evaluator(), proposer=NemotronProposer(client),
                 provenance="synthetic stub")
    timing = steps(logdir, "calls")
    assert [t["model_calls"] for t in timing] == [0, 2]
    assert logdir.read_events("calls")[-1]["timing"]["model_calls"] == 2


def test_failed_run_still_reports_timing_for_completed_steps(logdir):
    good = stub_evaluator()

    def flaky(params, style):
        if params != default_params():
            raise RuntimeError("scoring died")
        return good(params, style)

    with pytest.raises(RuntimeError):
        designer.run(3, "flaky", True, evaluator=flaky)
    finished = logdir.read_events("flaky")[-1]
    assert finished["type"] == "run_finished" and finished["timing"]["steps"] == 1


def test_should_stop_before_the_first_step_scores_nothing(logdir):
    designer.run(5, "stopped", True, evaluator=stub_evaluator(), should_stop=lambda: True)
    assert logdir.read("stopped") == []
    finished = logdir.read_events("stopped")[-1]
    assert finished["stop_reason"] == "cancelled" and finished["timing"] is None


def test_should_stop_between_steps(logdir):
    seen = []
    designer.run(10, "midway", True, evaluator=stub_evaluator(),
                 should_stop=lambda: len(seen) >= 3, on_step=lambda record, timing: seen.append(record))
    assert len(logdir.read("midway")) == 3
    assert [r["iteration"] for r in seen] == [0, 1, 2]
    assert logdir.read_events("midway")[-1]["stop_reason"] == "cancelled"


def test_cancel_is_seen_between_correction_rounds(logdir):
    # The first live round is invalid; cancellation arrives before the retry,
    # so the second model call is never made.
    stop = {"now": False}
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 99.0})],
                         [evaluate_call({"B_Ramp_Angle": 2.0}, call_id="c2")]])
    original = client.chat

    def chat_then_cancel(messages, **kwargs):
        reply = original(messages, **kwargs)
        stop["now"] = True
        return reply

    client.chat = chat_then_cancel
    designer.run(5, "mid-step", False, evaluator=stub_evaluator(), proposer=NemotronProposer(client),
                 provenance="synthetic stub", should_stop=lambda: stop["now"])
    assert len(client.sent) == 1
    assert len(logdir.read("mid-step")) == 1  # only the baseline
    assert logdir.read_events("mid-step")[-1]["stop_reason"] == "cancelled"


def test_on_step_receives_the_logged_record_and_its_timing(logdir):
    got = []
    designer.run(2, "progress", True, evaluator=stub_evaluator(),
                 on_step=lambda record, timing: got.append((record, timing)))
    assert [r for r, _ in got] == logdir.read("progress")
    assert all(t["iteration"] == r["iteration"] for r, t in got)


def test_in_process_scoring_matches_the_service_contract(logdir):
    predictor = StubPredictor()
    evaluate = designer.predictor_evaluator(predictor)
    result = evaluate(default_params(), "estateback")
    assert result["model_version"] == "stub-v2-synthetic" and "ood_score" in result
    assert designer.predictor_thresholds(predictor) == {"threshold": UNCERTAINTY_THRESHOLD,
                                                        "ood_threshold": OOD_SCORE_THRESHOLD}
    assert designer.predictor_provenance(predictor).startswith("synthetic stub")

    class Calibrated:
        name, model_version = "torch", "v7"
        uncertainty_threshold, ood_threshold = 0.013, 4.62

    assert designer.predictor_thresholds(Calibrated()) == {"threshold": 0.013, "ood_threshold": 4.62}
    assert designer.predictor_provenance(Calibrated()) == "torch surrogate v7"
