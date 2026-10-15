"""Agent against the baseline optimisers, over many runs.  OWNER: Paramveer.

    python -m src.eval.benchmark                          # summarise runs/logs
    python -m src.eval.benchmark --baseline-seeds 5 --iterations 25 --body-style fastback

One run is an anecdote: on 25 evaluations, a lucky random draw can beat
CMA-ES. This gathers every finished run in the log, groups them into fair
comparisons and writes web/results/benchmark.json for the results page.

A comparison is only runs that are fair against each other: the same body
style, the same evaluation budget, the same surrogate and the same
constraints. Runs that did not use their whole budget (cancelled, failed,
provider errors) are left out and listed, never silently dropped.

--baseline-seeds N first runs CMA-ES and random search for seeds 0..N-1
(skipping any already finished in the log), so the baselines have a spread to
compare against. They score through the service, like the agent, unless
--checkpoint names the checkpoint the service serves: then they score
in-process with the same predictor, gate thresholds and provenance, which
avoids the service's per-minute rate limit. Agent runs cost money; start
those yourself with src.agent.designer.

Progress is measured as the change in best trusted Cd against the run's own
iteration 0, the unmodified car, so runs on slightly different surrogates
or styles never get averaged by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..agent.constraints import parse_constraint
from ..service.runlog import RunLog
from .harness import portable_paths

FORMAT = "aero-surrogate-benchmark/1"
METHOD_ORDER = ("agent", "scripted", "cmaes", "random")


def method_of(records: list[dict], started: dict) -> tuple[str, str]:
    """(key, label) for a run: which proposer made it."""
    optimiser = records[0].get("optimiser", "agent") if records else "agent"
    if optimiser == "cmaes":
        return "cmaes", "CMA-ES"
    if optimiser == "random":
        return "random", "Random search"
    if started.get("mode") == "scripted" or not started.get("model"):
        return "scripted", "Scripted agent (dry run)"
    return f"agent:{started['model']}", f"Agent ({started['model']})"


def summarise_run(log: RunLog, run_id: str) -> tuple[dict | None, str | None]:
    """One run's numbers, or (None, why it is left out)."""
    try:
        records = log.read(run_id)
        events = log.read_events(run_id)
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"
    started = next((e for e in events if e.get("type") == "run_started"), None)
    finished = next((e for e in reversed(events) if e.get("type") == "run_finished"), None)
    if started is None:
        return None, "no run_started event"
    if finished is None:
        return None, "not finished"
    budget = started.get("evaluation_budget")
    if not records or budget is None or len(records) < budget:
        return None, f"used {len(records)} of {budget} evaluations ({finished.get('stop_reason')})"
    start_cd = records[0]["cd"]
    best = [r.get("best_cd_so_far") for r in records[:budget]]
    # Before anything is trusted (a reference-only baseline), progress is zero.
    progress = [0.0 if b is None else 100.0 * (b - start_cd) / start_cd for b in best]
    trusted = [r for r in records[:budget] if r["accepted"]]
    best_cd = min((r["cd"] for r in trusted), default=None)
    found_at = next((r["iteration"] for r in trusted if r["cd"] == best_cd), None)
    key, label = method_of(records, started)
    timing = finished.get("timing") or {}
    usage = finished.get("usage") or {}
    return {
        "run_id": run_id,
        "method": key,
        "label": label,
        "body_style": started.get("body_style") or records[0].get("body_style"),
        "budget": budget,
        "surrogate": started.get("surrogate"),
        "constraints": sorted(started.get("constraints") or []),
        "seed": started.get("seed"),
        "start_cd": start_cd,
        "best_cd": best_cd,
        "found_at": found_at,
        "progress_pct": progress,
        "final_pct": progress[-1],
        "gated": sum(1 for r in records[:budget] if not r["accepted"]
                     and not str(r.get("reject_reason") or "").startswith("reference only")),
        "step_s_median": timing.get("step_s_median"),
        "cost_usd": usage.get("cost_usd") if isinstance(usage, dict) else None,
    }, None


def _quantiles(rows: list[list[float]]) -> dict:
    a = np.asarray(rows, dtype=float)
    q25, median, q75 = np.percentile(a, [25, 50, 75], axis=0)
    return {"median": median.round(4).tolist(), "q25": q25.round(4).tolist(), "q75": q75.round(4).tolist(),
            "best": a.min(axis=0).round(4).tolist(), "worst": a.max(axis=0).round(4).tolist()}


def summarise_method(runs: list[dict]) -> dict:
    finals = [r["final_pct"] for r in runs]
    steps = [r["step_s_median"] for r in runs if r["step_s_median"] is not None]
    costs = [r["cost_usd"] for r in runs if r["cost_usd"] is not None]
    return {
        "method": runs[0]["method"],
        "label": runs[0]["label"],
        "n_runs": len(runs),
        "run_ids": [r["run_id"] for r in runs],
        "curve_pct": _quantiles([r["progress_pct"] for r in runs]),
        "final_pct": {"median": float(np.median(finals)), "best": min(finals), "worst": max(finals),
                      "per_run": [round(f, 4) for f in finals]},
        "found_at_median": float(np.median([r["found_at"] for r in runs if r["found_at"] is not None]))
        if any(r["found_at"] is not None for r in runs) else None,
        "gated_median": float(np.median([r["gated"] for r in runs])),
        "step_s_median": float(np.median(steps)) if steps else None,
        "cost_usd_total": round(sum(costs), 6) if costs else None,
        "start_cd_median": float(np.median([r["start_cd"] for r in runs])),
    }


def build(log: RunLog) -> dict:
    runs, excluded = [], []
    for entry in log.list_runs():
        summary, reason = summarise_run(log, entry["run_id"])
        if summary is None:
            excluded.append({"run_id": entry["run_id"], "reason": reason})
        else:
            runs.append(summary)
    groups: dict[tuple, list[dict]] = {}
    seen: dict[tuple, str] = {}
    for run in sorted(runs, key=lambda r: r["run_id"]):
        key = (run["body_style"], run["budget"], run["surrogate"], tuple(run["constraints"]))
        if run["seed"] is not None:
            # A baseline run is a pure function of its seed: a second run with
            # the same seed is the same run, and would count it twice.
            twin = key + (run["method"], run["seed"])
            if twin in seen:
                excluded.append({"run_id": run["run_id"], "reason": f"same seed as {seen[twin]}"})
                continue
            seen[twin] = run["run_id"]
        groups.setdefault(key, []).append(run)
    comparisons = []
    for (style, budget, surrogate, constraints), members in groups.items():
        by_method: dict[str, list[dict]] = {}
        for run in members:
            by_method.setdefault(run["method"], []).append(run)
        methods = [summarise_method(sorted(rs, key=lambda r: r["run_id"])) for rs in by_method.values()]
        methods.sort(key=lambda m: (METHOD_ORDER.index(m["method"].split(":")[0]), m["method"]))
        comparisons.append({"body_style": style, "budget": budget, "surrogate": surrogate,
                            "constraints": list(constraints), "n_runs": len(members), "methods": methods})
    # Fullest comparisons first: the one with an agent and the most runs.
    comparisons.sort(key=lambda c: (-any(m["method"].startswith("agent") for m in c["methods"]),
                                    -len(c["methods"]), -c["n_runs"]))
    return {"format": FORMAT, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "log_root": str(log.root), "metric": "change in best trusted Cd against the run's own "
            "iteration 0 (the unmodified car), percent; lower is better",
            "comparisons": comparisons, "excluded": excluded}


def run_baseline_seeds(seeds: int, iterations: int, body_style: str, constraints: list[str],
                       log: RunLog, checkpoint: str | None = None) -> list[str]:
    """Run CMA-ES and random search for seeds 0..seeds-1 not finished in the log yet."""
    from ..agent import designer
    from ..optim import baseline
    rules = [parse_constraint(rule) for rule in constraints]
    scoring = {}
    if checkpoint:
        from ..service.predictor import load_predictor
        predictor = load_predictor(checkpoint)
        scoring = {"evaluator": designer.predictor_evaluator(predictor),
                   "thresholds": designer.predictor_thresholds(predictor),
                   "provenance": designer.predictor_provenance(predictor)}
    # A stable tag per constraint set, so a rerun finds the runs it already made.
    tag = "-c" + hashlib.sha1("|".join(sorted(constraints)).encode()).hexdigest()[:6] if constraints else ""
    existing = {entry["run_id"] for entry in log.list_runs()}
    started = []
    for method in ("cmaes", "random"):
        for seed in range(seeds):
            base = f"bench-{method}-{body_style}-{iterations}{tag}-s{seed}"
            run_id, attempt = base, 1
            while run_id in existing:
                if summarise_run(log, run_id)[0] is not None:
                    break  # this seed already finished
                attempt += 1
                run_id = f"{base}-retry{attempt}"
            else:
                baseline.run(method, iterations, run_id, seed, body_style=body_style, constraints=rules,
                             log=log, **scoring)
                started.append(run_id)
    return started


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise agent and baseline runs into benchmark.json.")
    ap.add_argument("--runs", default=os.environ.get("AERO_RUNLOG_DIR", "runs/logs"), help="run log directory")
    ap.add_argument("--out", default="web/results/benchmark.json")
    ap.add_argument("--baseline-seeds", type=int, default=0, help="first run CMA-ES and random for this many seeds")
    ap.add_argument("--iterations", type=int, default=25, help="budget for --baseline-seeds runs")
    ap.add_argument("--body-style", default="fastback", help="body style for --baseline-seeds runs")
    ap.add_argument("--constraint", action="append", default=[], metavar="RULE",
                    help="constraint for --baseline-seeds runs, matching the agent run's. Repeatable.")
    ap.add_argument("--checkpoint", help="score --baseline-seeds runs in-process with this checkpoint "
                    "(the one the service serves) instead of over HTTP")
    args = ap.parse_args(argv)
    log = RunLog(args.runs)
    if args.baseline_seeds:
        started = run_baseline_seeds(args.baseline_seeds, args.iterations, args.body_style, args.constraint, log,
                                     args.checkpoint)
        print(f"ran {len(started)} baseline runs")
    result = build(log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(portable_paths(result), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    for comparison in result["comparisons"]:
        print(f"{comparison['body_style']}, {comparison['budget']} evaluations, {comparison['surrogate']}")
        for m in comparison["methods"]:
            print(f"  {m['label']:<28} {m['n_runs']:>3} runs   median {m['final_pct']['median']:+.2f}%   "
                  f"best {m['final_pct']['best']:+.2f}%")
    if result["excluded"]:
        print(f"left out {len(result['excluded'])} unfinished or short runs")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
