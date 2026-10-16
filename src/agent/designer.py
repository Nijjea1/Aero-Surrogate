"""The Nemotron design agent.  OWNER: Pranoy.  GATE: running Oct 17.

    python -m src.agent.designer --iterations 25 --model super
    python -m src.agent.designer --iterations 10 --dry-run      # scripted, no API

The uncertainty gate is the point of this file. When the surrogate is not
confident, the proposal is rejected and the agent is TOLD WHY, so it adapts
instead of hammering the same region.

From the command line both modes score every design over HTTP through the
service. The service's own run API (src/service/runs.py) runs the same loop
in-process with `predictor_evaluator`. --dry-run swaps Nemotron for a scripted
proposer so the loop can be tested for free. Live mode calls Token Factory
through src.agent.provider, which enforces the spend cap.

Every scored step writes a `step` event with where its time went: waiting for
the model versus scoring. run_finished carries the totals and medians, so the
speed claim can be made honestly (a millisecond surrogate does not make a
millisecond agent step).
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import uuid
from collections.abc import Callable
from numbers import Real

import requests

from ..contracts import (
    DEFAULT_BODY_STYLE, OOD_SCORE_THRESHOLD, PARAM_BOUNDS, UNCERTAINTY_THRESHOLD, Design,
    IterationRecord, default_params, validate_body_style, validate_params,
)
from ..service.gate import rejection_reason
from ..service.runlog import RunLog
from .constraints import Constraint, parse_constraint, violations
from .nemotron import Cancelled, NemotronProposer, NoProposal, Proposal, judge, logbook_entry
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


def predictor_evaluator(predictor) -> Callable[[dict[str, float], str], dict]:
    """Score in-process with a service predictor instead of over HTTP."""
    return lambda params, body_style: predictor.evaluate(Design(params, body_style)).to_dict()


def predictor_thresholds(predictor) -> dict:
    """The thresholds a predictor gates with, matching what /schema reports."""
    uncertainty = getattr(predictor, "uncertainty_threshold", None)
    ood = getattr(predictor, "ood_threshold", None)
    return {"threshold": UNCERTAINTY_THRESHOLD if uncertainty is None else float(uncertainty),
            "ood_threshold": OOD_SCORE_THRESHOLD if ood is None else float(ood)}


def predictor_provenance(predictor) -> str:
    return describe_provenance({"synthetic": predictor.name == "stub", "predictor": predictor.name,
                                "model_version": getattr(predictor, "model_version", None)})


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
        thresholds: dict | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_step: Callable[[dict, dict], None] | None = None,
        metadata: dict | None = None,
        model_provenance: dict | None = None,
        log: RunLog | None = None) -> str:
    """Log up to `iterations` scored attempts, including baseline iteration zero.

    The body style is fixed for the run: the same offsets describe a different
    car on each style, so mixing styles would make attempts incomparable.

    Live runs stop early, visibly, with a run_finished event naming the reason:
    repeated invalid proposals, a provider failure, the global spend cap, or
    this run's own cost budget (checked before every model call). An
    exception also writes run_finished before propagating. Dry-run always
    scores exactly `iterations` unless cancelled.

    should_stop is polled before every step and every model call; when it
    returns True the run ends with stop reason "cancelled". on_step(record,
    timing) is called after each scored step, for live progress. metadata
    (e.g. who started the run, and when) is recorded in run_started. log
    defaults to AERO_RUNLOG_DIR; the service passes its own RunLog.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if (isinstance(run_budget_usd, bool) or not isinstance(run_budget_usd, (int, float))
            or not math.isfinite(run_budget_usd) or run_budget_usd < 0):
        raise ValueError("run budget must be a finite nonnegative number of USD")
    problems = validate_body_style(body_style)
    if problems:
        raise ValueError("; ".join(problems))
    constraints = list(constraints or [])
    if not dry_run:
        if proposer is None:
            client = TokenFactoryClient.from_env(model=model)
            think = thinking == "on" or (thinking == "auto" and "nano" not in client.model.lower())
            proposer = NemotronProposer(client, constraints=constraints, thinking=think)
        # Enforced before every provider call, including correction rounds.
        proposer.budget_usd = run_budget_usd
        proposer.should_stop = should_stop
    # Readiness checks come before the run is reserved, so a service that is
    # down cannot leave an orphaned, never-started run behind.
    if provenance is None:
        if evaluator is None:
            health = service_health()
            provenance = describe_provenance(health)
            if model_provenance is None:
                model_provenance = health.get("provenance") or {
                    key: health.get(key) for key in ("predictor", "synthetic", "schema_version",
                                                     "model_version", "checkpoint_sha256",
                                                     "output_provenance")}
        else:
            provenance = "injected evaluator"
    if thresholds is None:
        thresholds = service_thresholds() if evaluator is None else {}
    log = log or RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
    log.reserve(run_id)
    log.append_event(run_id, {
        **(metadata or {}),
        "type": "run_started", "mode": "scripted" if dry_run else "nemotron",
        "model": None if dry_run else proposer.client.model, "body_style": body_style,
        "thinking": None if dry_run else proposer.thinking,
        "evaluation_budget": iterations, "run_budget_usd": None if dry_run else run_budget_usd,
        "constraints": [str(c) for c in constraints],
        "surrogate": provenance, "gate_thresholds": thresholds,
        "model_provenance": model_provenance,
    })
    state = {"history": [], "best_cd": None, "timings": [], "verdicts": [], "carry": _no_carry()}
    try:
        stop_reason = _drive(state, iterations, run_id, dry_run, log, evaluator or evaluate_design,
                             body_style, proposer, constraints, provenance, thresholds,
                             max_consecutive_failures, should_stop, on_step)
    except BaseException as exc:
        # Every started run gets a terminal event, even when scoring or the
        # service fails, so a reader never mistakes a dead run for a live one.
        stop_reason = f"error: {type(exc).__name__}: {exc}"
        _finish(log, run_id, stop_reason, state, None if dry_run else proposer)
        raise
    _finish(log, run_id, stop_reason, state, None if dry_run else proposer)
    return run_id


def _finish(log: RunLog, run_id: str, stop_reason: str, state: dict,
            proposer: NemotronProposer | None) -> None:
    usage = vars(proposer.client.usage).copy() if proposer is not None else None
    log.append_event(run_id, {"type": "run_finished", "stop_reason": stop_reason,
                              "scored": len(state["history"]), "best_cd": state["best_cd"],
                              "usage": usage, "timing": timing_summary(state["timings"], state["carry"]),
                              "hypothesis_hits": hypothesis_hits(state["verdicts"])})


def hypothesis_hits(verdicts: list[str]) -> dict:
    """How often a trusted live step moved Cd the way the model said it would.
    Only lower/higher expectations are judged; rejected and no_change steps
    are left out, and a change within noise counts as a miss."""
    judged, hits = len(verdicts), verdicts.count("as expected")
    return {"judged": judged, "as_expected": hits, "rate": hits / judged if judged else None}


def hypothesis_summary(hits: dict | None) -> str | None:
    """The CLI's one-line summary of hypothesis_hits; None when nothing was judged."""
    if not hits or not hits.get("judged"):
        return None
    return (f"Hypotheses: {hits['as_expected']} of {hits['judged']} trusted predictions had the "
            f"expected direction ({hits['rate']:.0%}).")


def _no_carry() -> dict:
    return {"propose_s": 0.0, "model_calls": 0, "model_latency_s": 0.0, "failed_proposals": 0}


def timing_summary(timings: list[dict], unscored: dict | None = None) -> dict | None:
    """Totals and medians of where step time went. None before any step.

    Medians describe live (model-proposed) steps when there are any, so the
    instant scripted baseline does not halve them on a short run. Totals
    include `unscored`: model time spent on attempts that never produced a
    scored design (failed proposals at the end of a run).
    """
    unscored = unscored or _no_carry()
    if not timings and not unscored["model_calls"]:
        return None
    live = [t for t in timings if t.get("live")] or timings
    propose = [t["propose_s"] for t in timings]
    score = [t["score_s"] for t in timings]
    step = [t["step_s"] for t in timings]
    surrogate = [t["surrogate_latency_ms"] for t in timings if t["surrogate_latency_ms"] is not None]
    total = sum(step) + unscored["propose_s"]
    waiting = sum(propose) + unscored["propose_s"]
    return {
        "steps": len(timings),
        "propose_s_total": waiting, "score_s_total": sum(score), "step_s_total": total,
        "propose_s_median": statistics.median(t["propose_s"] for t in live) if live else None,
        "score_s_median": statistics.median(score) if score else None,
        "step_s_median": statistics.median(t["step_s"] for t in live) if live else None,
        "surrogate_latency_ms_median": statistics.median(surrogate) if surrogate else None,
        "model_share": waiting / total if total > 0 else None,
        "model_calls": sum(t["model_calls"] for t in timings) + unscored["model_calls"],
        "failed_proposals": sum(t.get("failed_proposals", 0) for t in timings) + unscored["failed_proposals"],
    }


def _bank(state: dict, step_started: float, calls_before: int, latency_before: float,
          proposer: NemotronProposer, failed: bool) -> None:
    """Carry the time and calls of an attempt that scored nothing into the
    next scored step (or the run's totals), so no model time goes missing."""
    carry = state["carry"]
    carry["propose_s"] += _now() - step_started
    carry["model_calls"] += proposer.client.usage.calls - calls_before
    carry["model_latency_s"] += proposer.client.usage.latency_s - latency_before
    carry["failed_proposals"] += int(failed)


def _drive(state: dict, iterations: int, run_id: str, dry_run: bool, log: RunLog,
           evaluate: Callable[[dict[str, float], str], dict], body_style: str,
           proposer: NemotronProposer | None, constraints: list[Constraint], provenance: str,
           thresholds: dict, max_consecutive_failures: int,
           should_stop: Callable[[], bool] | None = None,
           on_step: Callable[[dict, dict], None] | None = None) -> str:
    """Propose, score, gate and log until done. Returns the stop reason."""
    history = state["history"]
    best_pressure = None
    failures = 0
    started = time.monotonic()
    while len(history) < iterations:
        if should_stop is not None and should_stop():
            return "cancelled"
        step_started = _now()
        calls_before = proposer.client.usage.calls if proposer is not None else 0
        latency_before = proposer.client.usage.latency_s if proposer is not None else 0.0
        if dry_run or not history:
            proposal = scripted_proposer(history)
            params, reasoning = proposal["params"], proposal["reasoning"]
        else:
            try:
                proposal = proposer.propose(
                    history, body_style=body_style, evaluations_left=iterations - len(history),
                    total_evaluations=iterations, best_pressure=best_pressure, provenance=provenance)
            except Cancelled:
                _bank(state, step_started, calls_before, latency_before, proposer, failed=False)
                return "cancelled"
            except NoProposal as exc:
                _bank(state, step_started, calls_before, latency_before, proposer, failed=True)
                failures += 1
                log.append_event(run_id, {"type": "no_valid_proposal", "detail": str(exc)})
                if failures >= max_consecutive_failures:
                    return f"{failures} consecutive steps without a valid proposal"
                continue
            except BudgetExceeded as exc:
                _bank(state, step_started, calls_before, latency_before, proposer, failed=False)
                return f"budget: {exc}"
            except ProviderError as exc:
                _bank(state, step_started, calls_before, latency_before, proposer, failed=False)
                log.append_event(run_id, {"type": "provider_error", "detail": str(exc)})
                return f"provider error: {exc}"
            failures = 0
            params, reasoning = proposal.params, None  # written after scoring, from the result
            for problem in proposal.feedback:
                log.append_event(run_id, {"type": "proposal_corrected", "detail": problem})
        _validate_proposal(params)
        proposed = _now()
        evaluation = evaluate(params, body_style)
        scored = _now()
        reason = rejection_reason(evaluation, body_style=body_style, **thresholds)
        # Invalid nonfinite predictions cannot be written as valid JSONL.
        if reason and reason.startswith("invalid prediction:"):
            raise ValueError(reason)
        broken = violations(params, constraints)
        if reason is None and broken:
            # Only the baseline can reach here; live proposals are checked
            # before scoring. It stays a reference, never the best design.
            reason = "reference only: " + "; ".join(broken)
        if reason is None and (state["best_cd"] is None or evaluation["cd"] < state["best_cd"]):
            state["best_cd"] = evaluation["cd"]
            best_pressure = evaluation.get("pressure_summary") or None
        if isinstance(proposal, Proposal):
            # The observed half of the logbook line comes from the score and
            # the gate, never from the model.
            reasoning = logbook_entry(proposal, evaluation["cd"], reason)
            # Judged only against a trusted base: a gate-rejected or
            # reference-only baseline is not a fair yardstick.
            if (reason is None and proposal.base_cd is not None and proposal.base_trusted
                    and proposal.expected_effect != "no_change"):
                state["verdicts"].append(judge(proposal.expected_effect, evaluation["cd"] - proposal.base_cd))
        record = IterationRecord(
            run_id=run_id, iteration=len(history), params=dict(params), body_style=body_style,
            cd=evaluation["cd"], cl=evaluation["cl"], uncertainty=evaluation["uncertainty"],
            in_distribution=evaluation["in_distribution"], accepted=reason is None,
            reject_reason=reason, reasoning=reasoning, best_cd_so_far=state["best_cd"],
            wall_clock_s=time.monotonic() - started, ood_score=evaluation.get("ood_score"),
        )
        log.append(record)
        history.append(record.to_dict())
        latency_ms = evaluation.get("latency_ms")
        carry, state["carry"] = state["carry"], _no_carry()
        timing = {
            "iteration": record.iteration,
            "live": isinstance(proposal, Proposal),
            "propose_s": proposed - step_started + carry["propose_s"],
            "model_calls": ((proposer.client.usage.calls - calls_before) if proposer is not None else 0)
                           + carry["model_calls"],
            "model_latency_s": ((proposer.client.usage.latency_s - latency_before) if proposer is not None
                                else 0.0) + carry["model_latency_s"],
            "failed_proposals": carry["failed_proposals"],
            "score_s": scored - proposed,
            "surrogate_latency_ms": float(latency_ms) if isinstance(latency_ms, (int, float)) else None,
            "step_s": _now() - step_started + carry["propose_s"],
        }
        state["timings"].append(timing)
        log.append_event(run_id, {"type": "step", **timing})
        if on_step is not None:
            on_step(history[-1], timing)
    return "evaluation budget used"


# Indirection so tests can drive step timing with a fake clock.
_now = time.perf_counter


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
    timing = finished.get("timing")
    if timing:
        share = f", {timing['model_share']:.0%} of it waiting for the model" if timing["model_share"] is not None else ""
        surrogate = timing["surrogate_latency_ms_median"]
        print(f"Time per step (median): {timing['step_s_median']:.2f}s total, "
              f"{timing['propose_s_median']:.2f}s proposing, {timing['score_s_median'] * 1000:.1f} ms scoring"
              + (f" (surrogate itself {surrogate:.2f} ms)" if surrogate else "") + share + ".")
    hypotheses = hypothesis_summary(finished.get("hypothesis_hits"))
    if hypotheses:
        print(hypotheses)


if __name__ == "__main__":
    main()
