"""Baseline optimisers: a fair fight with the agent, logged the same way."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agent import designer
from src.agent.constraints import parse_constraint
from src.contracts import PARAM_BOUNDS, PARAM_NAMES, default_params
from src.optim import baseline
from src.service.gate import rejection_reason
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog

STUB = StubPredictor()
EVALUATE = designer.predictor_evaluator(STUB)
THRESHOLDS = designer.predictor_thresholds(STUB)


def run(tmp_path, method="cmaes", iterations=30, run_id="r", seed=0, **kwargs):
    log = RunLog(tmp_path)
    kwargs.setdefault("evaluator", EVALUATE)
    kwargs.setdefault("thresholds", THRESHOLDS)
    baseline.run(method, iterations, run_id, seed, log=log, **kwargs)
    return log.read(run_id), log.read_events(run_id)


def test_unit_cube_round_trip():
    lo = {name: bounds[0] for name, bounds in PARAM_BOUNDS.items()}
    hi = {name: bounds[1] for name, bounds in PARAM_BOUNDS.items()}
    assert np.allclose(baseline.to_vector(lo), 0) and np.allclose(baseline.to_vector(hi), 1)
    x = baseline.to_vector(default_params())
    assert np.all((x > 0) & (x < 1))
    assert baseline.to_params(x) == default_params()
    assert baseline.to_params(np.full(len(PARAM_NAMES), 2.0)) == hi


def test_constraints_narrow_the_search_box():
    lo, hi = baseline.search_box([parse_constraint("A_Car_Roof_Height>=0"),
                                  parse_constraint("A_Car_Width<0")])
    i, j = PARAM_NAMES.index("A_Car_Roof_Height"), PARAM_NAMES.index("A_Car_Width")
    assert lo[i] == pytest.approx(60 / 150) and hi[i] == 1
    assert hi[j] < 60 / 210 and lo[j] == 0
    with pytest.raises(ValueError, match="no valid value"):
        baseline.search_box([parse_constraint("A_Car_Width>200")])


@pytest.mark.parametrize("method", baseline.METHODS)
def test_run_is_logged_like_the_agent(tmp_path, method):
    rows, events = run(tmp_path, method)
    assert [row["iteration"] for row in rows] == list(range(30))
    assert rows[0]["params"] == default_params()
    for row in rows:
        assert row["optimiser"] == method and row["reasoning"] == ""
        assert all(PARAM_BOUNDS[k][0] <= v <= PARAM_BOUNDS[k][1] for k, v in row["params"].items())
        assert row["accepted"] == (row["reject_reason"] is None)
    best = None
    for row in rows:
        if row["accepted"]:
            best = row["cd"] if best is None else min(best, row["cd"])
        assert row["best_cd_so_far"] == best
    started, *steps, finished = events
    assert started["type"] == "run_started" and started["optimiser"] == method
    assert started["evaluation_budget"] == 30 and started["gate_thresholds"] == THRESHOLDS
    assert [step["type"] for step in steps] == ["step"] * 30
    assert finished["type"] == "run_finished" and finished["scored"] == 30
    assert finished["best_cd"] == best and finished["stop_reason"] == "evaluation budget used"
    assert finished["timing"]["steps"] == 30


def test_same_seed_same_run_and_different_seed_different_run(tmp_path):
    a, _ = run(tmp_path, run_id="a", seed=1)
    b, _ = run(tmp_path, run_id="b", seed=1)
    c, _ = run(tmp_path, run_id="c", seed=2)
    assert [r["params"] for r in a] == [r["params"] for r in b]
    assert [r["params"] for r in a] != [r["params"] for r in c]


def test_cmaes_beats_random_search_and_the_baseline():
    """Medians over seeds on the stub, scored and gated as in a run (without
    the log, whose appends dominate the time)."""
    def best(method, seed):
        optimiser = baseline.make_optimiser(method, np.random.default_rng(seed))
        found = start
        for _ in range(59):
            x = optimiser.ask()
            evaluation = EVALUATE(baseline.to_params(x), "fastback")
            trusted = rejection_reason(evaluation, **THRESHOLDS) is None
            optimiser.tell(x, evaluation["cd"] if trusted else math.inf)
            found = min(found, evaluation["cd"]) if trusted else found
        return found

    start = EVALUATE(default_params(), "fastback")["cd"]
    cma = np.median([best("cmaes", seed) for seed in range(5)])
    rnd = np.median([best("random", seed) for seed in range(5)])
    assert cma < rnd <= start


@pytest.mark.parametrize("method", baseline.METHODS)
def test_proposals_obey_constraints_and_the_baseline_is_reference_only(tmp_path, method):
    rule = "A_Car_Roof_Height>10"
    rows, events = run(tmp_path, method, constraints=[parse_constraint(rule)])
    assert events[0]["constraints"] == [str(parse_constraint(rule))]
    assert rows[0]["reject_reason"].startswith("reference only")
    assert all(row["params"]["A_Car_Roof_Height"] > 10 for row in rows[1:])


def test_gated_designs_never_count_and_never_steer(tmp_path):
    def evaluate(params, body_style):
        result = EVALUATE(params, body_style)
        if params != default_params():
            result.update(cd=0.01, uncertainty=0.5, in_distribution=False)
        return result

    rows, _ = run(tmp_path, iterations=26, evaluator=evaluate)
    assert sum(row["accepted"] for row in rows) == 1
    assert {row["best_cd_so_far"] for row in rows} == {rows[0]["cd"]}


@pytest.mark.parametrize("method", baseline.METHODS)
def test_a_gated_baseline_stops_the_run_incomplete(tmp_path, method):
    """Same rule as the agent's comparison seam: no trusted start, no search."""
    def evaluate(params, body_style):
        return dict(EVALUATE(params, body_style), uncertainty=0.5)

    rows, events = run(tmp_path, method, iterations=25, evaluator=evaluate)
    assert len(rows) == 1 and not rows[0]["accepted"]
    finished = events[-1]
    assert finished["scored"] == 1 and finished["best_cd"] is None
    assert finished["stop_reason"].startswith("baseline rejected: ")


def test_cmaes_halves_its_step_when_a_generation_is_all_gated():
    rng = np.random.default_rng(0)
    cma = baseline.CMAES(np.full(4, 0.5), 0.2, np.zeros(4), np.ones(4), rng, popsize=6)
    for _ in range(6):
        cma.tell(cma.ask(), math.inf)
    assert cma.sigma == pytest.approx(0.1)
    assert np.allclose(cma.mean, 0.5)


def test_cmaes_samples_stay_in_the_box():
    rng = np.random.default_rng(0)
    lo, hi = np.full(16, 0.4), np.full(16, 0.6)
    cma = baseline.CMAES(np.full(16, 0.59), 0.5, lo, hi, rng)
    for i in range(100):
        x = cma.ask()
        assert np.all(x >= lo) and np.all(x <= hi)
        cma.tell(x, float(np.sum(x)))


def test_invalid_prediction_stops_the_run_visibly(tmp_path):
    def evaluate(params, body_style):
        return dict(EVALUATE(params, body_style), cd=math.nan)

    log = RunLog(tmp_path)
    with pytest.raises(ValueError, match="invalid prediction"):
        baseline.run("random", 5, "bad", evaluator=evaluate, thresholds=THRESHOLDS, log=log)
    finished = log.read_events("bad")[-1]
    assert finished["type"] == "run_finished" and finished["stop_reason"].startswith("error")


def test_bad_method_is_refused_before_the_run_exists(tmp_path):
    log = RunLog(tmp_path)
    with pytest.raises(ValueError, match="unknown method"):
        baseline.run("hillclimb", 5, "x", evaluator=EVALUATE, log=log)
    assert log.list_runs() == []
