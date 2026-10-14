"""The Nemotron design agent.  OWNER: Pranoy.  GATE: running Oct 17.

    python -m src.agent.designer --iterations 25 --model super
    python -m src.agent.designer --iterations 10 --dry-run      # scripted, no API

The uncertainty gate is the point of this file. When the surrogate is not
confident, the proposal is rejected and the agent is TOLD WHY, so it adapts
instead of hammering the same region.

Both modes score every design over HTTP through the service. --dry-run swaps
Nemotron for a scripted proposer so the loop can be tested for free. Live mode
calls Token Factory through src.agent.provider, which enforces the spend cap.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from numbers import Real

import requests

from ..contracts import (
    DEFAULT_BODY_STYLE, PARAM_BOUNDS, IterationRecord, default_params,
    validate_body_style, validate_params,
)
from ..service.gate import rejection_reason
from ..service.runlog import RunLog
from .constraints import Constraint, parse_constraint, violations
from .nemotron import NemotronProposer, NoProposal
from .provider import BudgetExceeded, ProviderError, TokenFactoryClient


def evaluate_design(params: dict[str, float], body_style: str = DEFAULT_BODY_STYLE) -> dict:
    """POST to the service's /evaluate."""
    _validate_proposal(params)
    base_url = os.environ.get("AERO_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
    response = requests.post(f"{base_url}/evaluate", json={"params": params, "body_style": body_style},
                             timeout=(3.05, 30))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("/evaluate response must be a JSON object")
    return result


def service_health() -> dict:
    base_url = os.environ.get("AERO_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
    response = requests.get(f"{base_url}/health", timeout=(3.05, 10))
    response.raise_for_status()
    return response.json()


def service_thresholds() -> dict:
    """The gate thresholds the serving model uses, so the agent and /evaluate agree."""
    base_url = os.environ.get("AERO_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
    response = requests.get(f"{base_url}/schema", timeout=(3.05, 10))
    response.raise_for_status()
    schema = response.json()
    return {"threshold": float(schema["uncertainty_threshold"]),
            "ood_threshold": float(schema["ood_score_threshold"])}


def describe_provenance(health: dict) -> str:
    if health.get("synthetic"):
        return "synthetic stub (numbers are a test fixture, not aerodynamic results)"
    version = health.get("model_version") or "unknown version"
    return f"{health.get('predictor', 'model')} surrogate {version}"


def scripted_proposer(history: list[dict]) -> dict:
    """Offline stand-in for Nemotron. Build this FIRST."""
    if not history:
        return {"params": default_params(), "reasoning": "Synthetic scripted baseline: start from the unmodified baseline body."}
    if len(history) == 1:
        return {"params": {name: hi for name, (_, hi) in PARAM_BOUNDS.items()},
                "reasoning": "Synthetic boundary probe to exercise the trust gate; this is not an aerodynamic hypothesis."}
    trusted = [step for step in history if step["accepted"]]
    if not trusted:
        return {"params": default_params(), "reasoning": "No trusted prediction exists; return to the baseline body after gate feedback."}
    best = min(trusted, key=lambda step: step["cd"])
    params = dict(best["params"])
    name = list(PARAM_BOUNDS)[(len(history) - 2) % len(PARAM_BOUNDS)]
    lo, hi = PARAM_BOUNDS[name]
    # Stay near the trusted starting region; alternate directions across passes.
    direction = -1 if ((len(history) - 2) // len(PARAM_BOUNDS)) % 2 == 0 else 1
    params[name] = min(lo + 0.7 * (hi - lo), max(lo + 0.3 * (hi - lo), params[name] + direction * 0.04 * (hi - lo)))
    feedback = history[-1].get("reject_reason")
    reasoning = f"Synthetic scripted step: perturb {name} from the best trusted design."
    if feedback:
        reasoning += f" Previous gate feedback: {feedback}. Return to the trusted region."
    return {"params": params, "reasoning": reasoning}


def _validate_proposal(params: dict[str, float]) -> None:
    if not isinstance(params, dict):
        raise ValueError("params must be a complete parameter dictionary")
    for name, value in params.items():
        try:
            finite = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"parameter '{name}' must be a finite number")
    problems = validate_params(params)
    if problems:
        raise ValueError("; ".join(problems))


def run(iterations: int, run_id: str, dry_run: bool, *,
        evaluator: Callable[[dict[str, float], str], dict] | None = None,
        body_style: str = DEFAULT_BODY_STYLE,
        proposer: NemotronProposer | None = None,
        constraints: list[Constraint] | None = None,
        provenance: str | None = None,
        model: str | None = None,
        run_budget_usd: float = 0.50,
        max_consecutive_failures: int = 2,
        thinking: str = "auto",
        thresholds: dict | None = None) -> str:
    """Log up to `iterations` scored attempts, including baseline iteration zero.

    The body style is fixed for the run: the same offsets describe a different
    car on each style, so mixing styles would make attempts incomparable.

    Live runs stop early, visibly, with a run_finished event naming the reason:
    repeated invalid proposals, a provider failure, the global spend cap, or
    this run's own cost budget. Dry-run always scores exactly `iterations`.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    problems = validate_body_style(body_style)
    if problems:
        raise ValueError("; ".join(problems))
    constraints = list(constraints or [])
    if not dry_run and proposer is None:
        client = TokenFactoryClient.from_env(model=model)
        think = thinking == "on" or (thinking == "auto" and "nano" not in client.model.lower())
        proposer = NemotronProposer(client, constraints=constraints, thinking=think)
    log = RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
    log.reserve(run_id)
    evaluate = evaluator or evaluate_design
    if provenance is None:
        provenance = describe_provenance(service_health()) if evaluator is None else "injected evaluator"
    if thresholds is None:
        thresholds = service_thresholds() if evaluator is None else {}
    log.append_event(run_id, {
        "type": "run_started", "mode": "scripted" if dry_run else "nemotron",
        "model": None if dry_run else proposer.client.model, "body_style": body_style,
        "thinking": None if dry_run else proposer.thinking,
        "evaluation_budget": iterations, "constraints": [str(c) for c in constraints],
        "surrogate": provenance, "gate_thresholds": thresholds,
    })
    history: list[dict] = []
    best_cd = None
    best_pressure = None
    failures = 0
    stop_reason = "evaluation budget used"
    started = time.monotonic()
    while len(history) < iterations:
        if dry_run or not history:
            proposal = scripted_proposer(history)
            params, reasoning = proposal["params"], proposal["reasoning"]
        else:
            if proposer.client.usage.cost_usd >= run_budget_usd:
                stop_reason = f"run cost budget ${run_budget_usd:.2f} reached"
                break
            try:
                proposal = proposer.propose(
                    history, body_style=body_style, evaluations_left=iterations - len(history),
                    total_evaluations=iterations, best_pressure=best_pressure, provenance=provenance)
            except NoProposal as exc:
                failures += 1
                log.append_event(run_id, {"type": "no_valid_proposal", "detail": str(exc)})
                if failures >= max_consecutive_failures:
                    stop_reason = f"{failures} consecutive steps without a valid proposal"
                    break
                continue
            except BudgetExceeded as exc:
                stop_reason = f"spend cap: {exc}"
                break
            except ProviderError as exc:
                log.append_event(run_id, {"type": "provider_error", "detail": str(exc)})
                stop_reason = f"provider error: {exc}"
                break
            failures = 0
            params, reasoning = proposal.params, proposal.reasoning
            for problem in proposal.feedback:
                log.append_event(run_id, {"type": "proposal_corrected", "detail": problem})
        _validate_proposal(params)
        evaluation = evaluate(params, body_style)
        reason = rejection_reason(evaluation, body_style=body_style, **thresholds)
        # Invalid nonfinite predictions cannot be written as valid JSONL.
        if reason and reason.startswith("invalid prediction:"):
            raise ValueError(reason)
        broken = violations(params, constraints)
        if reason is None and broken:
            # Only the baseline can reach here; live proposals are checked
            # before scoring. It stays a reference, never the best design.
            reason = "reference only: " + "; ".join(broken)
        if reason is None and (best_cd is None or evaluation["cd"] < best_cd):
            best_cd = evaluation["cd"]
            best_pressure = evaluation.get("pressure_summary") or None
        record = IterationRecord(
            run_id=run_id, iteration=len(history), params=dict(params), body_style=body_style,
            cd=evaluation["cd"], cl=evaluation["cl"], uncertainty=evaluation["uncertainty"],
            in_distribution=evaluation["in_distribution"], accepted=reason is None,
            reject_reason=reason, reasoning=reasoning, best_cd_so_far=best_cd,
            wall_clock_s=time.monotonic() - started, ood_score=evaluation.get("ood_score"),
        )
        log.append(record)
        history.append(record.to_dict())
    usage = None if dry_run else vars(proposer.client.usage).copy()
    log.append_event(run_id, {"type": "run_finished", "stop_reason": stop_reason,
                              "scored": len(history), "best_cd": best_cd, "usage": usage})
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the design loop against the evaluation service.")
    parser.add_argument("--iterations", type=int, default=25, help="Total scored attempts, including the baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--body-style", default=DEFAULT_BODY_STYLE, help="Body style for the whole run")
    parser.add_argument("--dry-run", action="store_true", help="Scripted proposals, no API calls")
    parser.add_argument("--model", default=None, help="nano | super | full model ID (default NEMOTRON_MODEL)")
    parser.add_argument("--constraint", action="append", default=[], metavar="RULE",
                        help="Numeric rule enforced in code, e.g. 'A_Car_Roof_Height>=0'. Repeatable.")
    parser.add_argument("--run-budget", type=float, default=0.50, help="Stop this run after this many USD")
    parser.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                        help="Model thinking; auto = off for Nano, on for Super")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        constraints = [parse_constraint(rule) for rule in args.constraint]
    except ValueError as exc:
        parser.error(str(exc))
    prefix = "scripted" if args.dry_run else "agent"
    run_id = args.run_id or f"{prefix}-{uuid.uuid4().hex[:12]}"
    if args.dry_run:
        print("Scripted development loop: no model calls; scripted proposals are not an aerodynamic result.")
    try:
        result = run(args.iterations, run_id, args.dry_run, body_style=args.body_style,
                     constraints=constraints, model=args.model, run_budget_usd=args.run_budget,
                     thinking=args.thinking)
    except (ValueError, OSError, requests.RequestException, ProviderError) as exc:
        parser.exit(1, f"Run failed: {exc}\n")
    log = RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
    rows = log.read(result)
    finished = log.read_events(result)[-1]
    for row in rows:
        verdict = "ok " if row["accepted"] else "REJ"
        print(f"{row['iteration']:3d} {verdict} Cd {row['cd']:.5f}  {row['reasoning'][:110]}")
    print(f"Run ID: {result}. {finished['scored']} scored; stopped because {finished['stop_reason']}. "
          f"Best trusted Cd: {finished['best_cd']}")
    if finished["usage"]:
        usage = finished["usage"]
        print(f"Model calls {usage['calls']}, tokens {usage['prompt_tokens']}+{usage['completion_tokens']}, "
              f"cost ${usage['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
