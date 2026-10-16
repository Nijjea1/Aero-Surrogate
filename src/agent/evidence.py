"""Qualify a checkpoint before spending money, then retain an honest agent trace.

This is an explicit local command, not a startup hook or a unit-test task. A
missing or mismatched checkpoint/reference package stops before the provider is
constructed. Evidence is based on recorded scores; improvement means a lower
trusted surrogate prediction, never an independently validated CFD improvement.
The Nano/Super sequence is intentionally two commands so Nano can be inspected
before the more expensive run. Nothing retries merely to obtain a better story.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from src.agent import designer
from src.agent.constraints import parse_constraint
from src.service.runlog import RunLog


def summarize_evidence(log: RunLog, run_id: str) -> dict:
    """Describe observed improvements and rejection follow-ups without inventing them."""
    rows, events = log.read(run_id), log.read_events(run_id)
    starts = [r for r in events if r.get("type") == "run_started"]
    finishes = [r for r in events if r.get("type") == "run_finished"]
    if len(starts) != 1 or len(finishes) > 1:
        raise ValueError("evidence requires exactly one start and at most one finish event")
    started, finished = starts[0], finishes[0] if finishes else None
    identity = started.get("model_provenance")
    if not isinstance(identity, dict) or identity.get("synthetic") is not False:
        raise ValueError("real-model evidence requires explicit non-synthetic provenance")
    # A short display hash cannot establish which artifact was qualified.
    digest = identity.get("checkpoint_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("real-model evidence requires full checkpoint SHA-256")
    for i, row in enumerate(rows):
        if (row.get("run_id") != run_id or type(row.get("iteration")) is not int
                or row["iteration"] != i or row.get("body_style") != started.get("body_style")
                or type(row.get("accepted")) is not bool
                or isinstance(row.get("cd"), bool) or not isinstance(row.get("cd"), (int, float))
                or not math.isfinite(row["cd"])):
            raise ValueError("evidence scored records have inconsistent identities, ordering or scores")
    baseline = rows[0] if rows else None
    trusted = [r for r in rows if r.get("accepted") is True]
    best = min(trusted, key=lambda r: r["cd"]) if trusted else None
    comparable = bool(baseline and baseline.get("accepted") is True)
    delta = best["cd"] - baseline["cd"] if comparable and best else None
    improvements = ([r for r in trusted if r["iteration"] > baseline["iteration"]
                     and r["cd"] < baseline["cd"]] if comparable else [])
    rejections = []
    for i, row in enumerate(rows):
        if row.get("accepted") is False:
            following = rows[i + 1] if i + 1 < len(rows) else None
            rejections.append({"rejected": row, "following_scored_proposal": following,
                               "adaptation_claim": "inspect the recorded change and reason"})
    return {"evidence_version": 1, "run_id": run_id, "model_provenance": identity,
            "provider_model": started.get("model"), "body_style": started.get("body_style"),
            "constraints": started.get("constraints"), "gate_thresholds": started.get("gate_thresholds"),
            "evaluation_budget": started.get("evaluation_budget"), "scored": len(rows),
            "complete": bool(finished and finished.get("stop_reason") == "evaluation budget used"
                             and finished.get("scored") == len(rows)
                             and len(rows) == started.get("evaluation_budget")),
            "stop_reason": finished.get("stop_reason") if finished else "unfinished",
            "baseline": baseline, "best_trusted": best, "delta_cd": delta,
            "relative_change": delta / baseline["cd"] if delta is not None and baseline["cd"] != 0 else None,
            "first_trusted_improvement": improvements[0] if improvements else None,
            "rejections": rejections, "usage": finished.get("usage") if finished else None,
            "timing": finished.get("timing") if finished else None,
            "claim": "trusted surrogate predictions; not CFD-validated improvements"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--model", choices=("nano", "super"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-dir", default="runs/logs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--body-style", default="fastback")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--run-budget", type=float, default=0.04)
    args = parser.parse_args(argv)
    # This command is the small evidence pair, not an escape hatch around the
    # user's rule to discuss a live run costing over roughly five cents.
    if not math.isfinite(args.run_budget) or not 0 < args.run_budget <= 0.05:
        parser.error("evidence run budget must be positive and at most $0.05")
    from src.service.predictor import TorchPredictor
    from src.service.provenance import describe_predictor
    from src.service.qualify import qualify_service

    output = Path(args.out)
    if output.exists():
        parser.error("evidence output exists; choose a fresh path")
    try:
        references = json.loads(Path(args.references).read_text(encoding="utf-8"))
        qualification = qualify_service(args.checkpoint, references)
        if not qualification["passed"]:
            print(json.dumps(qualification, indent=2, allow_nan=False))
            return 1
        predictor = TorchPredictor(args.checkpoint)
        identity = describe_predictor(predictor)
        if identity["checkpoint_sha256"] != qualification["provenance"]["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after qualification")
        constraints = [parse_constraint(rule) for rule in args.constraint]
        log = RunLog(args.log_dir)
        log.validate_run_id(args.run_id)
        if log.exists(args.run_id):
            raise ValueError("run ID already exists; evidence never reuses a run")
        # Prove writable artifact storage before making any provider call.
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump({"status": "qualified; run not yet complete", "qualification": qualification},
                      stream, indent=2, allow_nan=False)

        def baseline_check(record, timing):
            if record["iteration"] == 0 and not record["accepted"]:
                raise ValueError(f"evidence baseline is not trusted/feasible: {record['reject_reason']}")

        try:
            designer.run(10 if args.model == "nano" else 25, args.run_id, False,
                         evaluator=designer.predictor_evaluator(predictor),
                         thresholds=designer.predictor_thresholds(predictor),
                         provenance=designer.predictor_provenance(predictor), model_provenance=identity,
                         model=args.model, thinking="off" if args.model == "nano" else "auto",
                         body_style=args.body_style, constraints=constraints,
                         run_budget_usd=args.run_budget, log=log,
                         on_step=baseline_check,
                         metadata={"qualification": qualification})
        finally:
            try:
                report = summarize_evidence(log, args.run_id)
            except FileNotFoundError:
                report = None
            if report is not None:
                report["qualification"] = qualification
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Evidence saved to {output}")
        return 0 if report is not None and report["complete"] else 1
    except (ValueError, OSError, RuntimeError) as exc:
        # Provider errors already scrub keys. Do not dump configuration/env.
        print(f"Evidence run stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
