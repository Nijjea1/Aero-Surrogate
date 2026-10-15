"""Baseline optimisers under Track B's frozen comparison protocol.

Uses src.agent.comparison and src.agent.compare_run (Pranoy's seam); skipped
where they are absent. Fixture predictors only: no checkpoint, no provider.
"""

from __future__ import annotations

import copy
import json

import pytest

compare_run = pytest.importorskip("src.agent.compare_run")
comparison = pytest.importorskip("src.agent.comparison")

from src.contracts import Evaluation, default_params  # noqa: E402
from src.optim import baseline  # noqa: E402
from src.service.runlog import RunLog  # noqa: E402


class FakePredictor:
    name = "torch"
    model_version = "offline-fixture"

    def __init__(self, reject_baseline=False):
        self.reject_baseline = reject_baseline
        self.calls = 0

    def evaluate(self, design):
        self.calls += 1
        p = design.params
        sigma = 0.03 if self.reject_baseline and p == default_params() else 0.002
        cd = 0.3 - 0.001 * p["B_Ramp_Angle"] + 0.00002 * (p["A_Car_Roof_Height"] ** 2) / 100
        return Evaluation(cd=cd, cl=0.0, uncertainty=sigma, in_distribution=True, ood_score=1.0,
                          model_version=self.model_version)


@pytest.fixture
def identity(monkeypatch):
    value = {"synthetic": False, "predictor": "torch", "checkpoint_sha256": "a" * 64,
             "model_version": "offline-fixture", "git_revision": "b" * 40,
             "uncertainty_threshold": 0.01, "ood_score_threshold": 10.0, "source_dirty": False}
    monkeypatch.setattr(compare_run, "describe_predictor", lambda predictor: copy.deepcopy(value))
    return value


def protocol(budget=12, constraints=()):
    return compare_run.protocol_for(FakePredictor(), evaluation_budget=budget, constraints=constraints)


@pytest.mark.parametrize("method", baseline.METHODS)
def test_a_protocol_run_is_complete_logged_and_pairs(identity, tmp_path, method):
    log = RunLog(tmp_path)
    frozen = protocol(constraints=("B_Ramp_Angle >= 0",))
    predictor = FakePredictor()
    report = baseline.run_protocol(method, predictor, frozen, f"cmp-{method}", seed=1, log=log)
    assert report["complete"] and report["calls"] == predictor.calls == 12
    assert report["method"] == ("random_search" if method == "random" else "cmaes")
    assert report["stop_reason"] == "evaluation budget used" and report["metadata"]["seed"] == 1
    assert report["best_cd"] <= report["baseline_cd"]
    rows = log.read(f"cmp-{method}")
    assert [row["cd"] for row in rows] == [row["cd"] for row in report["outcomes"]]
    assert all(row["params"]["B_Ramp_Angle"] >= 0 for row in rows)
    json.dumps(report, allow_nan=False)

    other = baseline.run_protocol("random" if method == "cmaes" else "cmaes", FakePredictor(), frozen,
                                  "cmp-other", seed=2, log=log)
    comparison.summarize_pair(report, other)   # Track B's validator accepts the pair


def test_a_rejected_baseline_is_summarised_as_incomplete(identity, tmp_path):
    predictor = FakePredictor(reject_baseline=True)
    report = baseline.run_protocol("cmaes", predictor, protocol(), "gated", log=RunLog(tmp_path))
    assert not report["complete"] and report["calls"] == predictor.calls == 1
    assert report["stop_reason"].startswith("baseline rejected")


def test_another_checkpoint_is_refused_before_anything_is_logged(identity, tmp_path):
    frozen = protocol()
    identity["checkpoint_sha256"] = "c" * 64
    log = RunLog(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        baseline.run_protocol("random", FakePredictor(), frozen, "wrong", log=log)
    assert not log.exists("wrong")
