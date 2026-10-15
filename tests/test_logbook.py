"""The agent's reasoning as an engineer's logbook: the model states a hypothesis
and an expected effect; the code writes what was observed. No network, no spend."""

import sys

import pytest

from src.agent import designer
from src.agent.nemotron import (
    CD_NOISE_TOLERANCE, NemotronProposer, Proposal, TOOLS, judge, logbook_entry, read_logbook, state_message,
)
from src.service.runlog import RunLog
from tests.test_nemotron import BASE, FakeClient, evaluate_call, propose, row, stub_evaluator, tool_call


def proposal(expected_effect="lower_cd", base_cd=0.28710, base_iteration=3, delta=None,
             hypothesis="A shallower ramp should delay separation and shrink the base wake."):
    base = dict(BASE, B_Ramp_Angle=-3.0)
    return Proposal(params=dict(base, B_Ramp_Angle=-1.0, B_Diffusor_Angle=4.0),
                    changes={"B_Ramp_Angle": -1.0, "B_Diffusor_Angle": 4.0}, hypothesis=hypothesis,
                    expected_effect=expected_effect, rounds=1, expected_delta_cd=delta, base_params=base,
                    base_cd=base_cd, base_iteration=base_iteration, base_trusted=True)


# ------------------------------------------------------------------ the line


def test_accepted_as_expected_line():
    line = logbook_entry(proposal(delta=-0.001), 0.28650, None)
    assert line == ("Hypothesis: A shallower ramp should delay separation and shrink the base wake. | "
                    "Change: B_Ramp_Angle -3 → -1, B_Diffusor_Angle 0 → 4 | Expected: lower Cd (~-0.00100) | "
                    "Observed: Cd 0.28710 → 0.28650 (-0.00060) vs iteration 3, trusted — as expected")
    assert read_logbook(line) == {"expected": "lower Cd", "rejected": False, "delta": -0.0006,
                                  "verdict": "as expected"}


def test_opposite_and_within_noise_and_no_change():
    assert logbook_entry(proposal(), 0.28751, None).endswith(
        "Expected: lower Cd | Observed: Cd 0.28710 → 0.28751 (+0.00041) vs iteration 3, trusted — opposite to expected")
    assert logbook_entry(proposal("higher_cd"), 0.28710 + CD_NOISE_TOLERANCE / 2, None).endswith(
        "(+0.00000) vs iteration 3, trusted — within noise")
    assert logbook_entry(proposal("no_change"), 0.287102, None).endswith(
        "Expected: no change in Cd | Observed: Cd 0.28710 → 0.28710 (+0.00000) vs iteration 3, trusted — as expected")
    assert judge("no_change", 0.001) == "not as expected"
    assert judge("higher_cd", 0.001) == "as expected" and judge("higher_cd", -0.001) == "opposite to expected"


def test_rejected_line_has_no_trusted_result():
    line = logbook_entry(proposal(), 0.2, "uncertainty 0.0230 exceeds 0.0130")
    assert line.endswith("Expected: lower Cd | Observed: REJECTED by the gate (uncertainty 0.0230 exceeds 0.0130)"
                         " — no trusted result")
    assert "0.20000" not in line  # an untrusted Cd is never reported as an observation
    assert read_logbook(line)["rejected"]


def test_line_is_one_bounded_line_whatever_the_model_writes():
    line = logbook_entry(proposal(hypothesis="Long | multi\nline hypothesis. " * 40), 0.2865, None)
    assert "\n" not in line and line.count(" | ") == 3
    hypothesis = line.split(" | ")[0]
    assert len(hypothesis) <= len("Hypothesis: ") + 200 and hypothesis.endswith("…")
    assert read_logbook(line)["verdict"] == "as expected"


def test_plain_reasoning_is_not_a_logbook_entry():
    assert read_logbook("Synthetic scripted step: perturb B_Ramp_Angle from the best trusted design.") is None
    assert read_logbook(None) is None


# ------------------------------------------------------------- tool arguments


def test_tool_schema_asks_for_hypothesis_and_expected_effect():
    parameters = TOOLS[0]["function"]["parameters"]
    assert parameters["required"] == ["changes", "hypothesis", "expected_effect"]
    assert parameters["properties"]["expected_effect"]["enum"] == ["lower_cd", "higher_cd", "no_change"]
    assert parameters["properties"]["expected_delta_cd"]["type"] == "number"
    assert "reasoning" not in parameters["properties"]
    assert [tool["function"]["name"] for tool in TOOLS] == ["evaluate", "compare", "history"]


def test_proposal_carries_expectation_and_base_design():
    history = [row(0, BASE, 0.30), row(1, dict(BASE, B_Ramp_Angle=-3.0), 0.29)]
    client = FakeClient([[evaluate_call({"B_Diffusor_Angle": 4.0, "B_Ramp_Angle": -3.0}, "Diffuser recovers pressure.",
                                        expected_effect="lower_cd", expected_delta_cd=-0.001)]])
    got = propose(NemotronProposer(client), history)
    assert (got.hypothesis, got.expected_effect, got.expected_delta_cd) == ("Diffuser recovers pressure.", "lower_cd",
                                                                            -0.001)
    assert (got.base_cd, got.base_iteration, got.base_trusted) == (0.29, 1, True)
    # Restating an unchanged base value is not a change in the logbook either.
    assert "Change: B_Diffusor_Angle 0 → 4 | " in logbook_entry(got, 0.289, None)


def test_without_a_trusted_design_the_base_is_the_baseline():
    history = [row(0, BASE, 0.30, accepted=False, reason="reference only: x")]
    got = propose(NemotronProposer(FakeClient([[evaluate_call({"B_Ramp_Angle": 1.0})]])), history)
    assert (got.base_cd, got.base_iteration, got.base_trusted) == (0.30, 0, False)
    assert "vs iteration 0 (untrusted), trusted" in logbook_entry(got, 0.29, None)


def test_legacy_reasoning_argument_is_taken_as_the_hypothesis():
    call = tool_call("evaluate", {"changes": {"B_Ramp_Angle": 1.0}, "reasoning": "Old style.",
                                  "expected_effect": "LOWER_CD"})
    got = propose(NemotronProposer(FakeClient([[call]])), [row(0, BASE, 0.3)])
    assert got.hypothesis == "Old style." and got.expected_effect == "lower_cd" and got.rounds == 1


@pytest.mark.parametrize("args,fragment", [
    ({"expected_effect": "lower_cd"}, "'hypothesis' is required"),
    ({"hypothesis": "  ", "expected_effect": "lower_cd"}, "'hypothesis' is required"),
    ({"hypothesis": "Why.", "reasoning": "ignored"}, "'expected_effect' must be one of lower_cd, higher_cd, no_change"),
    ({"hypothesis": "Why.", "expected_effect": "better"}, "(got 'better')"),
    ({"hypothesis": "Why.", "expected_effect": "lower_cd", "expected_delta_cd": "a lot"}, "'expected_delta_cd'"),
])
def test_missing_or_invalid_expectation_is_refused_before_scoring(args, fragment):
    bad = tool_call("evaluate", {"changes": {"B_Ramp_Angle": 1.0}, **args})
    client = FakeClient([[bad], [evaluate_call({"B_Ramp_Angle": 1.0}, call_id="c2")]])
    got = propose(NemotronProposer(client), [row(0, BASE, 0.3)])
    assert got.rounds == 2  # a correction round, like any other refusal
    assert fragment in got.feedback[0] and got.feedback[0].startswith("rejected before scoring")


# ------------------------------------------------------------ learning signal


def logged(iteration, params, cd, base_cd, effect="lower_cd", accepted=True, reason=None):
    entry = proposal(effect, base_cd=base_cd, base_iteration=0)
    entry.params, entry.base_params = params, dict(BASE)
    return dict(row(iteration, params, cd, accepted, reason),
                reasoning=logbook_entry(entry, cd, None if accepted else reason))


def test_state_message_shows_last_expectation_against_observation():
    history = [row(0, BASE, 0.3), logged(1, dict(BASE, B_Ramp_Angle=1.0), 0.30041, 0.3)]
    text = state_message(history, body_style="fastback", evaluations_left=3, total_evaluations=5,
                         constraints=[], best_pressure=None, provenance="synthetic stub", window=8)
    assert "Last step: expected lower Cd, observed +0.00041 (opposite to expected)." in text
    assert "accepted; expected lower Cd: opposite to expected" in text
    history.append(logged(2, dict(BASE, B_Ramp_Angle=9.0), 0.2, 0.3, accepted=False, reason="ood 9 exceeds 3"))
    text = state_message(history, body_style="fastback", evaluations_left=2, total_evaluations=5,
                         constraints=[], best_pressure=None, provenance="synthetic stub", window=8)
    assert "REJECTED: ood 9 exceeds 3; expected lower Cd\n" in text
    assert "Last step: expected lower Cd, but the gate rejected it (no trusted result)." in text


def test_state_message_with_logbook_history_stays_bounded():
    history = [row(0, BASE, 0.3)] + [
        logged(i, dict(BASE, B_Ramp_Angle=float(i % 10), B_Diffusor_Angle=float(i % 7)), 0.3 - i * 1e-4, 0.3)
        for i in range(1, 41)]
    text = state_message(history, body_style="fastback", evaluations_left=3, total_evaluations=44,
                         constraints=[], best_pressure=None, provenance="synthetic stub", window=8)
    assert "Last step: expected lower Cd, observed -0.00400 (as expected)." in text
    assert len(text) < 2500


# ------------------------------------------------------------- run and stats


def scripted_evaluator(outcomes):
    """Cd by B_Ramp_Angle; a None Cd means an untrusted (out of distribution) result."""
    def evaluate(params, _style):
        cd = outcomes[params["B_Ramp_Angle"]]
        return {"cd": 0.25 if cd is None else cd, "cl": 0.0, "uncertainty": 0.003, "in_distribution": cd is not None}
    return evaluate


STEPS = [  # (ramp, Cd or None if rejected, expected effect)
    (1.0, 0.290, "lower_cd"),      # vs 0.300: as expected
    (2.0, 0.295, "lower_cd"),      # vs best 0.290: opposite
    (3.0, None, "higher_cd"),      # rejected: not judged
    (4.0, 0.290, "no_change"),     # no_change: not judged
    (5.0, 0.291, "higher_cd"),     # as expected
    (6.0, 0.290005, "lower_cd"),   # within noise: judged, a miss
]


def live(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    outcomes = {0.0: 0.300, **{ramp: cd for ramp, cd, _ in STEPS}}
    replies = [[evaluate_call({"B_Ramp_Angle": ramp}, f"Step {i}.", f"c{i}", effect)]
               for i, (ramp, _, effect) in enumerate(STEPS, 1)]
    return FakeClient(replies), scripted_evaluator(outcomes)


def test_run_records_logbook_lines_and_hypothesis_hits(tmp_path, monkeypatch):
    client, evaluate = live(tmp_path, monkeypatch)
    designer.run(len(STEPS) + 1, "logbook", False, evaluator=evaluate, proposer=NemotronProposer(client),
                 provenance="synthetic stub")
    log = RunLog(tmp_path)
    rows = log.read("logbook")
    assert rows[0]["reasoning"].startswith("Synthetic scripted baseline")  # baseline unchanged
    assert [read_logbook(r["reasoning"])["verdict"] for r in rows[1:]] == [
        "as expected", "opposite to expected", None, "as expected", "as expected", "within noise"]
    assert rows[2]["reasoning"].endswith("Cd 0.29000 → 0.29500 (+0.00500) vs iteration 1, trusted — opposite to expected")
    assert rows[3]["reasoning"].endswith("— no trusted result") and "REJECTED by the gate (" in rows[3]["reasoning"]
    finished = log.read_events("logbook")[-1]
    assert finished["hypothesis_hits"] == {"judged": 4, "as_expected": 2, "rate": 0.5}
    assert {"stop_reason", "scored", "best_cd", "usage", "timing"} <= set(finished)
    # The next model context saw the previous verdict.
    assert "Last step: expected lower Cd, observed +0.00500 (opposite to expected)." in client.sent[2][1]["content"]


def test_dry_run_records_keep_their_plain_reasoning(tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    designer.run(4, "dry", True, evaluator=stub_evaluator())
    log = RunLog(tmp_path)
    reasons = [r["reasoning"] for r in log.read("dry")]
    assert reasons[0] == "Synthetic scripted baseline: start from the unmodified baseline body."
    assert all(r.startswith("Synthetic") and " | Expected: " not in r for r in reasons)
    assert log.read_events("dry")[-1]["hypothesis_hits"] == {"judged": 0, "as_expected": 0, "rate": None}


def test_hypothesis_summary_line():
    assert designer.hypothesis_summary({"judged": 11, "as_expected": 7, "rate": 7 / 11}) == (
        "Hypotheses: 7 of 11 trusted predictions had the expected direction (64%).")
    assert designer.hypothesis_summary({"judged": 0, "as_expected": 0, "rate": None}) is None
    assert designer.hypothesis_summary(None) is None


def test_cli_prints_the_hypothesis_line(tmp_path, monkeypatch, capsys):
    client, evaluate = live(tmp_path, monkeypatch)
    real_run = designer.run
    monkeypatch.setattr(designer, "run", lambda *a, **k: real_run(
        *a, **{**k, "evaluator": evaluate, "proposer": NemotronProposer(client), "provenance": "synthetic stub"}))
    monkeypatch.setattr(sys, "argv", ["designer", "--iterations", str(len(STEPS) + 1), "--run-id", "cli"])
    designer.main()
    out = capsys.readouterr().out
    assert "Hypotheses: 2 of 4 trusted predictions had the expected direction (50%)." in out
