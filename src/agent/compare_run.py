"""Freeze a comparison protocol, run its agent side, or validate a matched pair.

The optimiser owner uses ComparisonProtocol/BudgetedEvaluator directly and saves
the same summary format. This module does not implement another person's search
algorithm or a second run API. An agent execution always qualifies the artifact
first and checks every protocol condition before making any provider call.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from src.agent import designer
from src.agent.comparison import BudgetedEvaluator, ComparisonProtocol, summarize_pair
from src.agent.constraints import parse_constraint
from src.service.provenance import describe_predictor
from src.service.runlog import RunLog


def protocol_for(predictor, *, body_style="fastback", constraints=(), evaluation_budget=25):
    """Freeze actual serving conditions; a stub or missing detector cannot qualify."""
    identity = describe_predictor(predictor)
    if identity["synthetic"] or not identity["checkpoint_sha256"]:
        raise ValueError("comparison requires a real checkpoint with full identity")
    if identity["ood_score_threshold"] is None:
        raise ValueError("comparison requires a parameter-space OOD detector")
    return ComparisonProtocol(
        checkpoint_sha256=identity["checkpoint_sha256"], model_version=identity["model_version"],
        git_revision=identity["git_revision"], uncertainty_threshold=identity["uncertainty_threshold"],
        ood_threshold=identity["ood_score_threshold"], body_style=body_style,
        constraints=tuple(constraints), evaluation_budget=evaluation_budget,
        source_sha256=identity.get("source_sha256"))


def assert_predictor_matches(predictor, protocol):
    actual = protocol_for(predictor, body_style=protocol.body_style,
                          constraints=protocol.constraints, evaluation_budget=protocol.evaluation_budget)
    actual = replace(actual, max_attempts=protocol.max_attempts)
    if actual.to_dict() != protocol.to_dict():
        raise ValueError("checkpoint or serving conditions differ from the frozen comparison protocol")


def evaluator_for(predictor, protocol, *, should_stop=None):
    """Bind an optimiser or agent to the checked artifact, gate and budget.

    Use this public entry point for real comparisons. BudgetedEvaluator also
    accepts arbitrary callables for offline fixtures, which alone cannot prove
    that a callable belongs to the artifact named by a protocol.
    """
    assert_predictor_matches(predictor, protocol)
    return BudgetedEvaluator(designer.predictor_evaluator(predictor), protocol,
                             should_stop=should_stop)


def run_agent(predictor, protocol, run_id, *, proposer=None, model="nano", run_budget=0.04,
              log=None, should_stop=None) -> dict:
    """Reusable fake-backed runner; record partial/error accounting in all outcomes.

    A rejected baseline is recorded then stops before any provider request. The
    caller persists the returned summary, including errors, instead of discarding
    unsuccessful runs. Qualification against independent references is enforced
    by the command-line entry point before this lower-level runner is called.
    """
    evaluator = evaluator_for(predictor, protocol, should_stop=should_stop)
    log = log or RunLog()
    log.validate_run_id(run_id)
    if log.exists(run_id):
        raise ValueError("comparison run ID exists; choose a fresh ID")
    if proposer is not None:
        expected_constraints = [parse_constraint(c) for c in protocol.constraints]
        if proposer.constraints != expected_constraints:
            raise ValueError("injected proposer constraints differ from comparison protocol")

    def baseline_check(record, timing):
        if record["iteration"] == 0 and not record["accepted"]:
            raise ValueError(f"comparison baseline rejected: {record['reject_reason']}")

    error = None
    try:
        designer.run(protocol.evaluation_budget, run_id, False, evaluator=evaluator,
                     thresholds=protocol.thresholds, provenance=designer.predictor_provenance(predictor),
                     model_provenance=describe_predictor(predictor), body_style=protocol.body_style,
                     constraints=[parse_constraint(c) for c in protocol.constraints], proposer=proposer,
                     model=model, thinking="off" if model == "nano" else "auto",
                     run_budget_usd=run_budget, should_stop=should_stop, on_step=baseline_check, log=log,
                     metadata={"comparison_protocol": protocol.to_dict(), "protocol_id": protocol.protocol_id})
    except (ValueError, OSError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        events = log.read_events(run_id)
    except FileNotFoundError:
        events = []
    started = next((e for e in events if e.get("type") == "run_started"), {})
    finished = next((e for e in reversed(events) if e.get("type") == "run_finished"), {})
    reason = finished.get("stop_reason") or f"error: {error or 'run has no finish event'}"
    return evaluator.summary("nemotron", run_id, reason, metadata={
        "provider_model": started.get("model"), "thinking": started.get("thinking"),
        "proposal_policy": "at most two changes from best trusted design; no repeats",
        "provider_corrections": [e for e in events if e.get("type") in
                                 ("proposal_corrected", "no_valid_proposal")],
        "usage": finished.get("usage"), "timing": finished.get("timing"),
        "model_provenance": started.get("model_provenance"), "error": error})


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("protocol", help="freeze conditions; no provider calls")
    make.add_argument("--checkpoint", required=True)
    make.add_argument("--body-style", default="fastback")
    make.add_argument("--constraint", action="append", default=[])
    make.add_argument("--evaluations", type=int, default=25)
    make.add_argument("--out", required=True)
    agent = commands.add_parser("agent", help="qualified live agent half of a comparison")
    agent.add_argument("--checkpoint", required=True)
    agent.add_argument("--references", required=True)
    agent.add_argument("--protocol", required=True)
    agent.add_argument("--model", choices=("nano", "super"), default="nano")
    agent.add_argument("--run-budget", type=float, default=0.04)
    agent.add_argument("--run-id", required=True)
    agent.add_argument("--log-dir", default="runs/logs")
    agent.add_argument("--out", required=True)
    pair = commands.add_parser("pair", help="validate both recorded summaries; no provider calls")
    pair.add_argument("left")
    pair.add_argument("right")
    pair.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists():
        parser.error("output exists; choose a fresh path")
    try:
        if args.command == "pair":
            report = summarize_pair(_read(args.left), _read(args.right))
        else:
            from src.service.predictor import TorchPredictor

            predictor = TorchPredictor(args.checkpoint)
            if args.command == "protocol":
                report = protocol_for(predictor, body_style=args.body_style, constraints=args.constraint,
                                      evaluation_budget=args.evaluations).to_dict()
            else:
                from src.service.qualify import qualify_service

                if not math.isfinite(args.run_budget) or not 0 < args.run_budget <= 0.05:
                    raise ValueError("comparison live budget must be positive and at most $0.05")
                qualification = qualify_service(args.checkpoint, _read(args.references))
                if not qualification["passed"]:
                    raise ValueError("checkpoint qualification failed: " + "; ".join(qualification["errors"]))
                if describe_predictor(predictor)["checkpoint_sha256"] != qualification["provenance"]["checkpoint_sha256"]:
                    raise ValueError("checkpoint changed after qualification")
                protocol = ComparisonProtocol.from_dict(_read(args.protocol))
                assert_predictor_matches(predictor, protocol)
                log = RunLog(args.log_dir)
                if log.exists(args.run_id):
                    raise ValueError("comparison run ID exists; choose a fresh ID")
                # Reserve writable output before the first provider call. Keep a
                # visible unfinished marker if the process itself is interrupted.
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8") as stream:
                    json.dump({"status": "qualified; run not yet complete", "qualification": qualification}, stream)
                report = run_agent(predictor, protocol, args.run_id, model=args.model,
                                   run_budget=args.run_budget, log=log)
                report["metadata"]["qualification"] = qualification
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                return 0 if report["complete"] else 1
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return 0 if args.command != "pair" or report["complete"] else 1
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Comparison stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
