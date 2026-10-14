import math

import pytest

from src.agent import designer
from src.contracts import PARAM_BOUNDS, default_params
from src.service.gate import rejection_reason
from src.service.runlog import RunLog


def prediction(**updates):
    result = {"cd": 0.3, "cl": -0.04, "uncertainty": 0.003, "in_distribution": True}
    result.update(updates)
    return result


@pytest.mark.parametrize("updates,fragment", [
    ({"cd": math.nan}, "finite"), ({"cl": math.inf}, "finite"),
    ({"uncertainty": -0.1}, "nonnegative"), ({"uncertainty": True}, "finite"),
    ({"in_distribution": "true"}, "boolean"), ({"in_distribution": False}, "out of distribution"),
    ({"uncertainty": 0.0151}, "exceeds"),
    ({"cd": 10 ** 1000}, "finite"),
])
def test_gate_rejects_untrusted_or_invalid_prediction(updates, fragment):
    assert fragment in rejection_reason(prediction(**updates))


def test_gate_accepts_threshold_equality():
    assert rejection_reason(prediction(uncertainty=0.015)) is None


def test_rejected_low_drag_never_updates_best_and_feedback_changes_next_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    scored = []

    def evaluate(params):
        scored.append(dict(params))
        if len(scored) == 2:
            return prediction(cd=0.001, uncertainty=0.023, in_distribution=False)
        return prediction(cd=0.3 - 0.001 * len(scored))

    assert designer.run(5, "feedback", True, evaluator=evaluate) == "feedback"
    records = RunLog(tmp_path).read("feedback")
    assert len(records) == len(scored) == 5
    assert [row["iteration"] for row in records] == list(range(5))
    assert scored[0] == default_params()
    assert scored[1] == {name: bounds[1] for name, bounds in PARAM_BOUNDS.items()}
    assert not records[1]["accepted"]
    assert records[1]["best_cd_so_far"] == records[0]["cd"]
    assert "Previous gate feedback" in records[2]["reasoning"]
    assert all(scored[2][name] < bounds[1] for name, bounds in PARAM_BOUNDS.items())
    best = [row["best_cd_so_far"] for row in records]
    assert all(left >= right for left, right in zip(best, best[1:]))
    assert min(best) > records[1]["cd"]


def test_all_rejected_run_has_no_best(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    designer.run(3, "rejected", True, evaluator=lambda _: prediction(in_distribution=False))
    records = RunLog(tmp_path).read("rejected")
    assert all(row["best_cd_so_far"] is None for row in records)
    assert records[2]["params"] == default_params()


def test_scripted_loop_integrates_with_actual_stub(tmp_path, monkeypatch):
    from src.contracts import Design
    from src.service.predictor import StubPredictor

    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    predictor = StubPredictor()
    designer.run(25, "actual-stub", True, evaluator=lambda params: predictor.evaluate(Design(params)).to_dict())
    records = RunLog(tmp_path).read("actual-stub")
    assert len(records) == 25
    assert records[0]["accepted"]
    assert not records[1]["accepted"]
    assert all(record["accepted"] for record in records[2:])
    assert records[-1]["best_cd_so_far"] < records[0]["cd"]
    assert all(record["best_cd_so_far"] >= records[-1]["best_cd_so_far"] for record in records)


@pytest.mark.parametrize("existing", ["", '{"old":"data"}\n'])
def test_run_refuses_reuse_even_empty(tmp_path, monkeypatch, existing):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    target = tmp_path / "existing.jsonl"
    target.write_text(existing)
    with pytest.raises(FileExistsError):
        designer.run(2, "existing", True, evaluator=lambda _: prediction())
    assert target.read_text() == existing


def test_live_mode_is_explicitly_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    with pytest.raises(NotImplementedError, match="Live Nemotron"):
        designer.run(1, "live", False)
    assert not (tmp_path / "live.jsonl").exists()


@pytest.mark.parametrize("iterations", [0, -1, True, 1.5])
def test_iteration_budget_must_be_positive_integer(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        designer.run(iterations, "invalid", True)


@pytest.mark.parametrize("bad", [{}, {**default_params(), "ride_height": True},
                                   {**default_params(), "unknown": 1.0},
                                   {**default_params(), "ride_height": math.nan}])
def test_invalid_proposal_never_scored(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    monkeypatch.setattr(designer, "scripted_proposer", lambda _: {"params": bad, "reasoning": "bad"})
    scored = []
    with pytest.raises(ValueError):
        designer.run(1, "invalid-proposal", True, evaluator=lambda params: scored.append(params))
    assert scored == []


def test_evaluate_uses_configured_url_and_finite_timeout(monkeypatch):
    monkeypatch.setenv("AERO_SERVICE_URL", "http://localhost:9999/")
    captured = {}

    class Response:
        def raise_for_status(self):
            captured["status_checked"] = True

        def json(self):
            return prediction()

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(designer.requests, "post", post)
    assert designer.evaluate_design(default_params()) == prediction()
    assert captured == {"url": "http://localhost:9999/evaluate", "json": {"params": default_params()},
                        "timeout": (3.05, 30), "status_checked": True}
