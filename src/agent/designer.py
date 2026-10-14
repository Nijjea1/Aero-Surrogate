"""The Nemotron design agent.  OWNER: Pranoy.  GATE: running Oct 17.

    python -m src.agent.designer --iterations 25 --dry-run

The uncertainty gate is the point of this file. When the surrogate is not
confident, reject the proposal and TELL THE AGENT WHY, so it adapts instead of
hammering the same region.

Build a --dry-run scripted proposer first so the loop can be tested before
Token Factory is wired up. The gate for Oct 17 is "the loop runs end to end",
not "the predictions are good".
"""

from __future__ import annotations

import argparse
import math
import os
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

SYSTEM_PROMPT = """You are an aerodynamicist minimising trusted predicted Cd.
- parameters are offsets from the baseline body of the run's body style;
  all zeros is the unmodified baseline
- stay inside the bounds given
- if a proposal is REJECTED for low confidence, move back toward regions the
  model understands; do not repeat the rejected move
- change one or two parameters at a time so effects are attributable
- respond ONLY with JSON: {"params": {...}, "reasoning": "..."}"""


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


def ask_nemotron(history: list[dict], api_key: str) -> dict:
    """One planning step via Token Factory. -> {"params": ..., "reasoning": ...}"""
    raise NotImplementedError("Live Nemotron integration is not implemented; use --dry-run for the scripted proposer")


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
        body_style: str = DEFAULT_BODY_STYLE) -> str:
    """Log exactly iterations scored attempts, including baseline iteration zero.

    The body style is fixed for the run: the same offsets describe a different
    car on each style, so mixing styles would make attempts incomparable.

    Dry-run means scripted proposals, not bypassed HTTP or fabricated service
    results. Predictor provenance comes from /health and each Evaluation.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    problems = validate_body_style(body_style)
    if problems:
        raise ValueError("; ".join(problems))
    if not dry_run:
        raise NotImplementedError("Live Nemotron integration is not implemented; use --dry-run")
    log = RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
    log.reserve(run_id)
    evaluate = evaluator or evaluate_design
    history: list[dict] = []
    best_cd = None
    started = time.monotonic()
    for iteration in range(iterations):
        proposal = scripted_proposer(history)
        params = proposal["params"]
        _validate_proposal(params)
        evaluation = evaluate(params, body_style)
        reason = rejection_reason(evaluation, body_style=body_style)
        # Invalid nonfinite predictions cannot be written as valid JSONL.
        if reason and reason.startswith("invalid prediction:"):
            raise ValueError(reason)
        if reason is None:
            best_cd = evaluation["cd"] if best_cd is None else min(best_cd, evaluation["cd"])
        record = IterationRecord(
            run_id=run_id, iteration=iteration, params=dict(params), body_style=body_style,
            cd=evaluation["cd"], cl=evaluation["cl"], uncertainty=evaluation["uncertainty"],
            in_distribution=evaluation["in_distribution"], accepted=reason is None,
            reject_reason=reason, reasoning=proposal["reasoning"], best_cd_so_far=best_cd,
            wall_clock_s=time.monotonic() - started, ood_score=evaluation.get("ood_score"),
        )
        log.append(record)
        history.append(record.to_dict())
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline scripted design loop against /evaluate.")
    parser.add_argument("--iterations", type=int, default=25, help="Total scored attempts, including the initial design")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--body-style", default=DEFAULT_BODY_STYLE, help="Body style for the whole run")
    parser.add_argument("--dry-run", action="store_true", help="Use scripted proposals; still calls the evaluation service")
    args = parser.parse_args()
    run_id = args.run_id or f"scripted-{uuid.uuid4().hex[:12]}"
    print("Scripted development loop: synthetic stub outputs do not demonstrate aerodynamic improvement.")
    try:
        result = run(args.iterations, run_id, args.dry_run, body_style=args.body_style)
    except (ValueError, OSError, requests.RequestException, NotImplementedError) as exc:
        parser.exit(1, f"Run failed: {exc}\n")
    print(f"Completed {args.iterations} scored attempts. Run ID: {result}")


if __name__ == "__main__":
    main()
