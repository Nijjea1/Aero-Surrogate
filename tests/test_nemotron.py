"""Live proposer and loop, driven by a scripted fake model. No network, no spend."""

import json

import pytest

from src.agent import designer
from src.agent.constraints import parse_constraint, violations
from src.agent.nemotron import NemotronProposer, NoProposal, TOOLS, state_message
from src.agent.provider import BudgetExceeded, ProviderError, Usage
from src.contracts import Design, PARAM_BOUNDS, default_params
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog


def tool_call(name, args, call_id="c1"):
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def evaluate_call(changes, hypothesis="Test hypothesis: this should reduce wake drag.", call_id="c1",
                  expected_effect="lower_cd", **extra):
    args = {"changes": changes, "hypothesis": hypothesis, "expected_effect": expected_effect, **extra}
    return tool_call("evaluate", args, call_id)


class FakeClient:
    """Replays scripted assistant messages and records what it was sent."""

    model = "fake/nemotron"

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.usage = Usage()

    def chat(self, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            reply = {"role": "assistant", "content": "", "tool_calls": reply}
        elif isinstance(reply, str):
            reply = {"role": "assistant", "content": reply}
        self.usage.add(100, 50, 0.001)
        return {"message": reply, "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop",
                "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.001, "latency_s": 0.0}


def row(iteration, params, cd, accepted=True, reason=None):
    return {"iteration": iteration, "params": params, "cd": cd, "uncertainty": 0.005,
            "ood_score": 2.0, "accepted": accepted, "reject_reason": reason}


def propose(proposer, history):
    return proposer.propose(history, body_style="fastback", evaluations_left=5, total_evaluations=10)


BASE = default_params()


def test_valid_evaluate_applies_changes_to_best_trusted_design():
    best = dict(BASE, B_Ramp_Angle=-3.0)
    history = [row(0, BASE, 0.30), row(1, best, 0.29), row(2, dict(BASE, A_Car_Width=100.0), 0.20,
                                                         accepted=False, reason="uncertainty too high")]
    proposer = NemotronProposer(FakeClient([[evaluate_call({"B_Diffusor_Angle": 4.0})]]))
    proposal = propose(proposer, history)
    assert proposal.params == dict(best, B_Diffusor_Angle=4.0)  # rejected low Cd is never the base
    assert proposal.changes == {"B_Diffusor_Angle": 4.0}
    assert proposal.rounds == 1 and proposal.feedback == []


def test_out_of_bounds_is_fed_back_not_clamped_then_corrected():
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 99.0})], [evaluate_call({"B_Ramp_Angle": 5.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert proposal.params["B_Ramp_Angle"] == 5.0
    assert proposal.rounds == 2
    assert "outside" in proposal.feedback[0] and "not clamped" in proposal.feedback[0]
    tool_reply = client.sent[1][-1]
    assert tool_reply["role"] == "tool" and tool_reply["tool_call_id"] == "c1"
    assert "outside" in tool_reply["content"]


@pytest.mark.parametrize("changes,fragment", [
    ({"ride_height": 0.1}, "unknown parameter"),
    ({"B_Ramp_Angle": True}, "finite number"),
    ({}, "non-empty"),
])
def test_bad_changes_are_explained(changes, fragment):
    client = FakeClient([[evaluate_call(changes)], [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert fragment in proposal.feedback[0]


def test_duplicate_design_is_refused():
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 0.0})], [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert "already scored" in proposal.feedback[0]


def test_constraint_is_enforced_before_scoring():
    rule = parse_constraint("A_Car_Roof_Height >= 0")
    client = FakeClient([[evaluate_call({"A_Car_Roof_Height": -20.0})],
                         [evaluate_call({"A_Car_Roof_Height": 10.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client, constraints=[rule]), [row(0, BASE, 0.3)])
    assert "violates constraint A_Car_Roof_Height >= 0" in proposal.feedback[0]
    assert "A_Car_Roof_Height >= 0" in client.sent[0][1]["content"]


def test_text_only_replies_exhaust_rounds_and_raise():
    client = FakeClient(["I think we should lower the roof."] * 3)
    with pytest.raises(NoProposal, match="no tool call"):
        propose(NemotronProposer(client, max_rounds=3), [row(0, BASE, 0.3)])
    assert len(client.sent) == 3
    assert "Call evaluate now" in client.sent[1][-1]["content"]


def test_auxiliary_tools_answer_then_evaluate():
    history = [row(0, BASE, 0.30), row(1, dict(BASE, B_Ramp_Angle=-3.0), 0.29)]
    client = FakeClient([
        [tool_call("compare", {"iteration_a": 0, "iteration_b": 1}), tool_call("history", {"last_n": 2}, "c2")],
        [evaluate_call({"B_Diffusor_Angle": 3.0}, call_id="c3")],
    ])
    proposal = propose(NemotronProposer(client), history)
    tool_replies = [m for m in client.sent[1] if m["role"] == "tool"]
    assert "B_Ramp_Angle" in tool_replies[0]["content"] and "Cd -0.01000" in tool_replies[0]["content"]
    assert "it 1:" in tool_replies[1]["content"]
    assert proposal.rounds == 2


def test_only_one_evaluate_per_step_and_malformed_arguments():
    client = FakeClient([[tool_call("evaluate", "{not json"), evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2"),
                          evaluate_call({"B_Ramp_Angle": 2.0}, call_id="c3")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert proposal.params["B_Ramp_Angle"] == 1.0
    assert "JSON object" in proposal.feedback[0]


def test_state_message_is_bounded_and_carries_rejection_reasons():
    history = [row(i, dict(BASE, B_Ramp_Angle=float(i % 10)), 0.3 - i * 1e-4) for i in range(40)]
    history.append(row(40, dict(BASE, A_Car_Width=150.0), 0.2, accepted=False,
                       reason="uncertainty 0.0230 exceeds 0.0130"))
    text = state_message(history, body_style="fastback", evaluations_left=3, total_evaluations=44,
                         constraints=[], best_pressure=None, provenance="synthetic stub", window=8)
    assert "REJECTED: uncertainty 0.0230 exceeds 0.0130" in text
    assert "it 32:" not in text and "it 33:" in text
    assert "synthetic stub" in text
    assert len(text) < 2500


def test_tools_publish_every_contract_parameter_with_bounds():
    properties = TOOLS[0]["function"]["parameters"]["properties"]["changes"]["properties"]
    assert set(properties) == set(PARAM_BOUNDS)
    assert "[-8, 15]" in properties["B_Ramp_Angle"]["description"]


def test_constraint_parsing():
    assert str(parse_constraint("A_Car_Roof_Height>=0")) == "A_Car_Roof_Height >= 0"
    assert violations(dict(BASE, A_Car_Roof_Height=-1.0), [parse_constraint("A_Car_Roof_Height >= 0")])
    for bad in ("A_Car_Roof_Height = 0", "nonsense >= 1", "A_Car_Roof_Height >= x"):
        with pytest.raises(ValueError):
            parse_constraint(bad)


# ----------------------------------------------------------------- live loop


def stub_evaluator():
    predictor = StubPredictor()
    return lambda params, style: predictor.evaluate(Design(params, style)).to_dict()


def live_run(tmp_path, monkeypatch, replies, iterations=4, **kwargs):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    client = FakeClient(replies)
    constraints = kwargs.pop("constraints", [])
    proposer = NemotronProposer(client, constraints=constraints)
    designer.run(iterations, "live", False, evaluator=stub_evaluator(), proposer=proposer,
                 constraints=constraints, provenance="synthetic stub", **kwargs)
    log = RunLog(tmp_path)
    return client, log.read("live"), log.read_events("live")


def test_live_run_feeds_rejection_into_the_next_step(tmp_path, monkeypatch):
    all_high = {name: hi for name, (_, hi) in PARAM_BOUNDS.items()}
    replies = [
        [evaluate_call({"B_Ramp_Angle": -3.0}, "Lower the ramp to reduce base wake.")],
        [evaluate_call({"A_Car_Width": all_high["A_Car_Width"]}, "Widen to the limit.")],
        [evaluate_call({"B_Diffusor_Angle": 4.0}, "Back off; try diffuser instead.")],
    ]
    client, rows, events = live_run(tmp_path, monkeypatch, replies)
    assert [r["accepted"] for r in rows] == [True, True, False, True]
    assert rows[1]["reasoning"].startswith("Hypothesis: Lower the ramp to reduce base wake. | Change: "
                                           "B_Ramp_Angle 0 → -3 | Expected: lower Cd | Observed: Cd ")
    assert rows[2]["reasoning"].endswith("— no trusted result")
    assert "uncertainty" in rows[2]["reject_reason"]
    # The rejection and its reason are in the very next model context.
    next_state = client.sent[2][1]["content"]
    assert "REJECTED: uncertainty" in next_state
    assert rows[3]["params"]["A_Car_Width"] == 0.0  # built on the trusted best, not the rejected design
    assert rows[2]["best_cd_so_far"] == rows[1]["best_cd_so_far"]
    assert events[0]["type"] == "run_started" and events[0]["model"] == "fake/nemotron"
    assert events[-1]["type"] == "run_finished" and events[-1]["stop_reason"] == "evaluation budget used"
    assert events[-1]["usage"]["calls"] == 3


def test_live_run_stops_after_consecutive_invalid_steps(tmp_path, monkeypatch):
    client, rows, events = live_run(tmp_path, monkeypatch, ["no tools"] * 8, iterations=10,
                                    max_consecutive_failures=2)
    assert len(rows) == 1  # only the baseline was scored
    assert [e["type"] for e in events].count("no_valid_proposal") == 2
    assert "consecutive" in events[-1]["stop_reason"]
    assert len(client.sent) == 8  # 2 steps x 4 rounds, then stop


def test_provider_error_and_spend_cap_stop_visibly(tmp_path, monkeypatch):
    _, rows, events = live_run(tmp_path / "a", monkeypatch, [ProviderError("HTTP 503")])
    assert len(rows) == 1 and events[-1]["stop_reason"].startswith("provider error")
    assert any(e["type"] == "provider_error" for e in events)
    _, rows, events = live_run(tmp_path / "b", monkeypatch, [BudgetExceeded("cap reached")])
    assert events[-1]["stop_reason"] == "budget: cap reached"


def test_run_cost_budget_stops_the_run(tmp_path, monkeypatch):
    replies = [[evaluate_call({"B_Ramp_Angle": float(i)}, call_id=f"c{i}")] for i in range(1, 9)]
    client, rows, events = live_run(tmp_path, monkeypatch, replies, iterations=9, run_budget_usd=0.0025)
    assert len(client.sent) == 3 and len(rows) == 4
    assert "run budget" in events[-1]["stop_reason"]


def test_corrections_are_logged_as_events(tmp_path, monkeypatch):
    replies = [[evaluate_call({"B_Ramp_Angle": 99.0})], [evaluate_call({"B_Ramp_Angle": 2.0}, call_id="c2")]]
    _, rows, events = live_run(tmp_path, monkeypatch, replies, iterations=2)
    assert len(rows) == 2
    assert any(e["type"] == "proposal_corrected" and "outside" in e["detail"] for e in events)


def test_baseline_that_breaks_a_constraint_is_reference_only(tmp_path, monkeypatch):
    rule = parse_constraint("A_Car_Roof_Height >= 10")
    replies = [[evaluate_call({"A_Car_Roof_Height": 12.0})]]
    _, rows, _ = live_run(tmp_path, monkeypatch, replies, iterations=2, constraints=[rule])
    assert not rows[0]["accepted"] and rows[0]["reject_reason"].startswith("reference only")
    assert rows[0]["best_cd_so_far"] is None
    assert rows[1]["accepted"] and rows[1]["best_cd_so_far"] == rows[1]["cd"]


def test_thinking_switch_is_sent_only_when_disabled():
    seen = []

    class Recording(FakeClient):
        def chat(self, messages, **kwargs):
            seen.append(kwargs.get("extra"))
            return super().chat(messages, **kwargs)

    for thinking in (True, False):
        proposer = NemotronProposer(Recording([[evaluate_call({"B_Ramp_Angle": 1.0})]]), thinking=thinking)
        propose(proposer, [row(0, BASE, 0.3)])
    assert seen == [None, {"chat_template_kwargs": {"enable_thinking": False}}]


def test_quoted_numbers_are_parsed_but_junk_is_refused():
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": "abc"})], [evaluate_call({"B_Ramp_Angle": "2.5"}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert "got 'abc'" in proposal.feedback[0]
    assert proposal.params["B_Ramp_Angle"] == 2.5 and proposal.changes == {"B_Ramp_Angle": 2.5}


def test_loop_gates_with_the_serving_models_thresholds(tmp_path, monkeypatch):
    # A calibrated checkpoint threshold (5.0) above the contract fallback (3.0):
    # an ood_score of 4.0 must be trusted, exactly as /evaluate would say.
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    evaluation = {"cd": 0.3, "cl": 0.0, "uncertainty": 0.005, "in_distribution": True, "ood_score": 4.0}
    designer.run(2, "calibrated", True, evaluator=lambda *_: dict(evaluation),
                 thresholds={"threshold": 0.013, "ood_threshold": 5.0})
    assert RunLog(tmp_path).read("calibrated")[0]["accepted"]
    designer.run(2, "fallback", True, evaluator=lambda *_: dict(evaluation))
    assert not RunLog(tmp_path).read("fallback")[0]["accepted"]


# ---------------------------------------------------- audit regressions, Oct 14


def test_run_budget_bounds_correction_rounds(tmp_path, monkeypatch):
    # Audit 2: the budget was only checked between steps, so a step's four
    # correction rounds and further failing steps kept spending.
    client, rows, events = live_run(tmp_path, monkeypatch, ["no tool"] * 20, iterations=10,
                                    run_budget_usd=0.0025, max_consecutive_failures=5)
    assert len(client.sent) == 3  # 0.001 per fake call: stops once 0.0025 is passed
    assert "run budget" in events[-1]["stop_reason"]


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf"), True])
def test_run_budget_must_be_finite_and_nonnegative(tmp_path, monkeypatch, budget):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="run budget"):
        designer.run(2, "bad-budget", False, evaluator=stub_evaluator(),
                     proposer=NemotronProposer(FakeClient([])), run_budget_usd=budget)
    assert not (tmp_path / "bad-budget.jsonl").exists()


@pytest.mark.parametrize("call", [
    tool_call("compare", {"iteration_a": [], "iteration_b": 0}),
    tool_call("compare", {"iteration_a": True, "iteration_b": 0}),
    tool_call("history", {"last_n": "5"}),
    {"id": "c1", "type": "function", "function": "evaluate"},
    {"id": "c1", "type": "function", "function": {"name": "evaluate", "arguments": 42}},
    {"id": 7, "function": {"name": "evaluate", "arguments": "[1, 2]"}},
])
def test_malformed_tool_calls_get_feedback_not_a_crash(call):
    # Audit 3: compare with iteration_a=[] raised TypeError and killed the run.
    client = FakeClient([[call], [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert proposal.params["B_Ramp_Angle"] == 1.0
    tool_reply = [m for m in client.sent[1] if m["role"] == "tool"][0]
    assert tool_reply["content"].startswith("error")


@pytest.mark.parametrize("reply", [
    {"role": "assistant", "content": None, "tool_calls": "evaluate"},
    {"role": "assistant", "content": 5, "tool_calls": [None, 3]},
])
def test_malformed_messages_count_as_no_tool_call(reply):
    client = FakeClient([reply, [evaluate_call({"B_Ramp_Angle": 1.0})]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert proposal.feedback == ["no tool call in reply"]


def test_signed_zero_cannot_rescore_a_design():
    # Audit 5: -0.0 serialised differently from 0.0 and slipped past.
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": -0.0})],
                         [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert "already scored" in proposal.feedback[0]


def test_integer_and_float_spellings_are_the_same_design():
    history = [row(0, BASE, 0.3), row(1, dict(BASE, B_Ramp_Angle=2.0), 0.29)]
    client = FakeClient([[evaluate_call({"B_Ramp_Angle": 2})],
                         [evaluate_call({"B_Ramp_Angle": 3.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), history)
    assert "already scored" in proposal.feedback[0]


def test_changing_more_than_two_parameters_is_refused():
    changes = {"B_Ramp_Angle": 1.0, "B_Diffusor_Angle": 1.0, "A_Car_Width": 5.0}
    client = FakeClient([[evaluate_call(changes)], [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    proposal = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert "change at most 2 parameters" in proposal.feedback[0]
    # Restating an unchanged value does not count as a change.
    restated = {"B_Ramp_Angle": 1.0, "B_Diffusor_Angle": 1.0, "A_Car_Width": 0.0}
    client = FakeClient([[evaluate_call(restated)]])
    assert propose(NemotronProposer(client), [row(0, BASE, 0.3)]).rounds == 1


def test_scoring_failure_still_finishes_the_run(tmp_path, monkeypatch):
    # Audit 4: a failing evaluator left only run_started behind.
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))

    def broken(*_):
        raise RuntimeError("model/service failed")

    with pytest.raises(RuntimeError):
        designer.run(2, "broken", True, evaluator=broken)
    events = RunLog(tmp_path).read_events("broken")
    assert [e["type"] for e in events] == ["run_started", "run_finished"]
    assert events[-1]["stop_reason"] == "error: RuntimeError: model/service failed"


def test_service_down_leaves_no_orphan_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))

    def down():
        raise designer.requests.ConnectionError("refused")

    monkeypatch.setattr(designer, "service_health", down)
    with pytest.raises(designer.requests.ConnectionError):
        designer.run(2, "orphan", True)
    assert not (tmp_path / "orphan.jsonl").exists()
    assert RunLog(tmp_path).read_events("orphan") == []
