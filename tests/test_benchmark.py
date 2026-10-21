"""The agent-against-baselines summary behind the results page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent import designer
from src.agent.constraints import parse_constraint
from src.eval import benchmark
from src.eval.harness import portable_paths
from src.optim import baseline
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog

STUB = StubPredictor()
SCORING = {"evaluator": designer.predictor_evaluator(STUB), "thresholds": designer.predictor_thresholds(STUB),
           "provenance": designer.predictor_provenance(STUB)}


@pytest.fixture
def log(tmp_path):
    log = RunLog(tmp_path)
    for method in ("cmaes", "random"):
        for seed in range(3):
            baseline.run(method, 12, f"{method}-{seed}", seed, log=log, **SCORING)
    designer.run(12, "scripted-0", dry_run=True, log=log, evaluator=SCORING["evaluator"],
                 thresholds=SCORING["thresholds"], provenance=SCORING["provenance"])
    return log


def test_runs_are_grouped_into_one_fair_comparison(log):
    result = benchmark.build(log)
    assert result["format"] == benchmark.FORMAT and result["excluded"] == []
    (comparison,) = result["comparisons"]
    assert comparison["budget"] == 12 and comparison["body_style"] == "fastback"
    methods = {m["method"]: m for m in comparison["methods"]}
    assert [m["method"] for m in comparison["methods"]] == ["scripted", "cmaes", "random"]
    assert methods["cmaes"]["n_runs"] == 3 and methods["scripted"]["n_runs"] == 1
    curve = methods["cmaes"]["curve_pct"]
    assert len(curve["median"]) == 12 and curve["median"][0] == 0.0
    assert all(lo <= mid <= hi for lo, mid, hi in zip(curve["q25"], curve["median"], curve["q75"]))
    # Best-so-far never gets worse along a run.
    assert all(b <= a for a, b in zip(curve["best"], curve["best"][1:]))
    json.dumps(result, allow_nan=False)


def test_unfair_or_unfinished_runs_never_mix_in(log, tmp_path):
    baseline.run("cmaes", 20, "longer", 0, log=log, **SCORING)
    baseline.run("cmaes", 12, "constrained", 0, log=log, constraints=[parse_constraint("A_Car_Width>=0")], **SCORING)
    baseline.run("cmaes", 12, "same-seed", 1, log=log, **SCORING)
    log.reserve("cut-off")
    result = benchmark.build(log)
    assert len(result["comparisons"]) == 3
    main = next(c for c in result["comparisons"] if c["budget"] == 12 and not c["constraints"])
    assert sum(m["n_runs"] for m in main["methods"] if m["method"] == "cmaes") == 3
    reasons = {e["run_id"]: e["reason"] for e in result["excluded"]}
    assert reasons["same-seed"].startswith("same seed as") and "cut-off" in reasons


def test_baseline_seeds_run_once_and_rerun_skips_them(tmp_path, monkeypatch):
    log = RunLog(tmp_path)
    monkeypatch.setattr(designer, "evaluate_design", SCORING["evaluator"])
    monkeypatch.setattr(designer, "service_thresholds", lambda: SCORING["thresholds"])
    monkeypatch.setattr(designer, "service_health", lambda: {"synthetic": True})
    first = benchmark.run_baseline_seeds(2, 8, "fastback", [], log)
    assert sorted(first) == sorted(f"bench-{m}-fastback-8-s{s}" for m in ("cmaes", "random") for s in range(2))
    assert benchmark.run_baseline_seeds(2, 8, "fastback", [], log) == []


def test_published_files_name_no_local_disk(tmp_path):
    root = Path.cwd()
    out = portable_paths({
        "inside": str(root / "runs" / "splits.json"),
        "outside": "C:/Users/someone/AppData/Local/Temp/best.pt",
        "posix": "/home/someone/data/table.csv",
        "command": r"harness.py --checkpoint C:\Users\someone\ckpt\best.pt --out runs/m.json",
        "ratio": "cd 0.2/0.3 over runs/x",
    }, root)
    assert out["inside"] == "runs/splits.json"
    assert out["outside"] == "best.pt" and out["posix"] == "table.csv"
    assert out["command"] == "harness.py --checkpoint best.pt --out runs/m.json"
    assert out["ratio"] == "cd 0.2/0.3 over runs/x"


def test_committed_results_files_are_publishable():
    web = Path(__file__).resolve().parents[1] / "web" / "results"
    for name in ("metrics.json", "benchmark.json"):
        text = (web / name).read_text(encoding="utf-8")
        json.loads(text)
        assert "Users" not in text and "/home/" not in text, f"{name} names a local disk"
    assert json.loads((web / "metrics.json").read_text(encoding="utf-8")).get("label")


def _matched_summary(method, run_id, *, delta=1, budget=2):
    from src.agent.comparison import BudgetedEvaluator, ComparisonProtocol
    from src.contracts import default_params

    protocol = ComparisonProtocol("a" * 64, "fixture-v1", "revision", .01, 10., evaluation_budget=budget)

    def score(params, style):
        return {"cd": .3 - params["B_Ramp_Angle"] * .001, "cl": 0., "uncertainty": .002,
                "ood_score": 1., "in_distribution": True, "pressure_summary": {}}

    evaluator = BudgetedEvaluator(score, protocol)
    evaluator(default_params(), "fastback")
    evaluator(dict(default_params(), B_Ramp_Angle=delta), "fastback")
    return evaluator.summary(method, run_id, "evaluation budget used", metadata={
        "provider_model": "nvidia/nemotron-3-super-120b-a12b", "seed": 0,
        "usage": {"cost_usd": .02} if method == "nemotron" else {},
        "timing": {"step_s_median": 1.2, "step_s_total": 2.4}})


def test_matched_export_checks_protocol_and_aggregates_actual_calls(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for method in ("nemotron", "cmaes", "random"):
        (source / f"{method}.json").write_text(json.dumps(_matched_summary(method, method)))
    result = benchmark.build_matched(source, publish_to=tmp_path / "public")
    assert result["comparisons"][0]["n_runs"] == 3
    methods = result["comparisons"][0]["methods"]
    assert methods[0]["label"] == "Nemotron Super"
    assert methods[0]["final_pct"]["median"] == pytest.approx(-1 / 3)
    assert methods[0]["cost_usd_median"] == .02
    assert methods[0]["sum_step_s_median"] == 2.4
    assert len(result["published_runs"]) == 3
    assert len(list((tmp_path / "public").glob("*.json"))) == 3
    # Numeric tampering cannot become a headline by merely editing a summary.
    bad = json.loads((source / "random.json").read_text())
    bad["best_cd"] = .01
    (source / "random.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not match"):
        benchmark.build_matched(source)


def test_matched_export_retains_incomplete_evidence_but_excludes_aggregate(tmp_path):
    for method, budget in (("nemotron", 2), ("random", 3)):
        data = _matched_summary(method, method, budget=budget)
        # Different declared budgets cannot be compared at all.
        (tmp_path / f"{method}.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocols differ"):
        benchmark.build_matched(tmp_path)
