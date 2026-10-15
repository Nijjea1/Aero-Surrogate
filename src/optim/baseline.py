"""Baseline optimisers the agent is measured against.  OWNER: Paramveer.

    python -m src.optim.baseline --method cmaes --iterations 25
    python -m src.optim.baseline --method random --iterations 25 --seed 3

This has to be a FAIR fight, so everything except the proposer is shared with
the agent (src.agent.designer):

    parameter space   the 16 contract dials, searched in the unit cube
    budget            `iterations` scored attempts, iteration 0 included
    start             iteration 0 is default_params(), the unmodified body;
                      CMA-ES centres its first generation there
    scoring           the same HTTP /evaluate call (or an injected evaluator)
    gate              the same rejection_reason with the service's thresholds;
                      a gated design scores as infinitely bad, and a gated
                      baseline stops the run there, incomplete, as the agent's
                      comparison seam (src.agent.comparison) does
    constraints       the same rules; the search box is narrowed so no
                      proposal breaks one, as the agent's are refused unscored
    run log           the same IterationRecords and run events, with
                      optimiser "random" or "cmaes" and empty reasoning

If the agent does not beat these, we report it anyway. "The LLM matched CMA-ES
but produced an explainable trace and respected stated constraints" is an
honest finding. Hiding the comparison is what would hurt us.

CMA-ES is implemented here (Hansen's (mu/mu_w, lambda) update, about 60 lines)
rather than imported, so it adds no dependency and its handling of the gate is
visible: only trusted designs are selected, and a generation with none halves
the step size around the last trusted mean.

On the synthetic stub most uniform random designs fail the gate, because the
stub's edge uncertainty is a test fixture. On a real checkpoint the gate is
calibrated so that about 99% of real designs, which were sampled uniformly from
this box, pass.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid
from collections.abc import Callable

import numpy as np
import requests

from ..agent import designer
from ..agent.constraints import Constraint, parse_constraint, violations
from ..contracts import (
    DEFAULT_BODY_STYLE, PARAM_BOUNDS, PARAM_NAMES, IterationRecord, default_params,
    validate_body_style,
)
from ..service.gate import rejection_reason
from ..service.runlog import RunLog

METHODS = ("random", "cmaes")

_LO = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
_SPAN = np.array([PARAM_BOUNDS[name][1] - PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
# Logged values are rounded to this many decimals; strict constraints keep a
# margin wider than the rounding so a rounded proposal cannot land on the limit.
_DECIMALS = 4
_STRICT_MARGIN = 1e-3


def to_vector(params: dict[str, float]) -> np.ndarray:
    """Map the param dict into the unit cube so every axis is comparable."""
    return (np.array([params[name] for name in PARAM_NAMES], dtype=float) - _LO) / _SPAN


def to_params(vec: np.ndarray) -> dict[str, float]:
    values = _LO + np.clip(np.asarray(vec, dtype=float), 0.0, 1.0) * _SPAN
    return {name: round(float(v), _DECIMALS) for name, v in zip(PARAM_NAMES, values)}


def search_box(constraints: list[Constraint] = ()) -> tuple[np.ndarray, np.ndarray]:
    """The unit-cube box left once the constraints are applied, as (lo, hi)."""
    lo, hi = np.zeros(len(PARAM_NAMES)), np.ones(len(PARAM_NAMES))
    for rule in constraints:
        i = PARAM_NAMES.index(rule.name)
        margin = _STRICT_MARGIN if rule.op in ("<", ">") else 0.0
        if rule.op in (">=", ">"):
            lo[i] = max(lo[i], (rule.value + margin - _LO[i]) / _SPAN[i])
        else:
            hi[i] = min(hi[i], (rule.value - margin - _LO[i]) / _SPAN[i])
        if lo[i] > hi[i]:
            raise ValueError(f"constraints leave no valid value for {rule.name}")
    return lo, hi


class RandomSearch:
    """Uniform samples from the search box, independent of every score."""

    name = "random"

    def __init__(self, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator):
        self.lo, self.hi, self.rng = lo, hi, rng

    def ask(self) -> np.ndarray:
        return self.rng.uniform(self.lo, self.hi)

    def tell(self, x: np.ndarray, score: float) -> None:
        pass

    def describe(self) -> dict:
        return {}


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES in the unit cube, asked and told one design at a time.

    Scores are minimised; math.inf marks a gated design. Selection takes the
    best mu trusted designs of a generation (fewer if fewer were trusted), so a
    rejected design never pulls the search toward itself. Samples outside the
    box are redrawn, which costs no evaluations.
    """

    name = "cmaes"

    def __init__(self, mean: np.ndarray, sigma: float, lo: np.ndarray, hi: np.ndarray,
                 rng: np.random.Generator, popsize: int | None = None):
        n = len(mean)
        self.n, self.lo, self.hi, self.rng = n, lo, hi, rng
        self.mean = np.clip(np.asarray(mean, dtype=float), lo, hi)
        self.sigma = float(sigma)
        self.popsize = popsize or 4 + int(3 * math.log(n))
        self.mu = self.popsize // 2
        weights = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D, self.C = np.eye(n), np.ones(n), np.eye(n)
        self.generation = 0
        self._queue: list[np.ndarray] = []
        self._told: list[tuple[np.ndarray, float]] = []

    def ask(self) -> np.ndarray:
        if not self._queue:
            self._queue = [self._sample() for _ in range(self.popsize - len(self._told))]
        return self._queue.pop(0)

    def tell(self, x: np.ndarray, score: float) -> None:
        self._told.append((np.asarray(x, dtype=float), float(score)))
        if len(self._told) == self.popsize:
            self._update(self._told)
            self._told, self._queue = [], []

    def describe(self) -> dict:
        return {"sigma0": self.sigma, "popsize": self.popsize, "mu": self.mu}

    def _sample(self, tries: int = 100) -> np.ndarray:
        for _ in range(tries):
            x = self.mean + self.sigma * (self.B @ (self.D * self.rng.standard_normal(self.n)))
            if np.all(x >= self.lo) and np.all(x <= self.hi):
                return x
        return np.clip(x, self.lo, self.hi)

    def _update(self, told: list[tuple[np.ndarray, float]]) -> None:
        self.generation += 1
        trusted = sorted((pair for pair in told if math.isfinite(pair[1])), key=lambda pair: pair[1])
        if not trusted:
            self.sigma *= 0.5
            return
        k = min(self.mu, len(trusted))
        w = self.weights[:k] / self.weights[:k].sum()
        mueff = 1.0 / np.sum(w ** 2)
        steps = (np.array([x for x, _ in trusted[:k]]) - self.mean) / self.sigma
        y = w @ steps
        self.mean = self.mean + self.sigma * y

        inv_sqrt_c = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * mueff) * (inv_sqrt_c @ y)
        ps_norm = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
        hsig = float(ps_norm / self.chi_n < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * mueff) * y
        rank_mu = (steps * w[:, None]).T @ steps
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))

        self.C = (self.C + self.C.T) / 2
        eigenvalues, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigenvalues, 1e-20))


def make_optimiser(method: str, rng: np.random.Generator, constraints: list[Constraint] = (),
                   sigma: float = 0.2, popsize: int | None = None):
    lo, hi = search_box(constraints)
    if method == "random":
        return RandomSearch(lo, hi, rng)
    if method == "cmaes":
        return CMAES(to_vector(default_params()), sigma, lo, hi, rng, popsize)
    raise ValueError(f"unknown method '{method}' (expected one of {METHODS})")


def run(method: str, iterations: int, run_id: str, seed: int = 0, *,
        evaluator: Callable[[dict[str, float], str], dict] | None = None,
        body_style: str = DEFAULT_BODY_STYLE,
        constraints: list[Constraint] | None = None,
        thresholds: dict | None = None,
        provenance: str | None = None,
        sigma: float = 0.2,
        popsize: int | None = None,
        log: RunLog | None = None) -> str:
    """Log `iterations` scored attempts, iteration 0 being the baseline body.

    Gated designs score as infinitely bad, same as the agent experiences.
    Without an injected evaluator, scoring, provenance and thresholds come from
    the service, exactly as for the agent's CLI.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    problems = validate_body_style(body_style)
    if problems:
        raise ValueError("; ".join(problems))
    constraints = list(constraints or [])
    optimiser = make_optimiser(method, np.random.default_rng(seed), constraints, sigma, popsize)
    if provenance is None:
        provenance = (designer.describe_provenance(designer.service_health())
                      if evaluator is None else "injected evaluator")
    if thresholds is None:
        thresholds = designer.service_thresholds() if evaluator is None else {}
    evaluate = evaluator or designer.evaluate_design
    log = log or RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
    log.reserve(run_id)
    log.append_event(run_id, {
        "type": "run_started", "mode": method, "optimiser": method, "seed": seed,
        "body_style": body_style, "evaluation_budget": iterations,
        "constraints": [str(c) for c in constraints],
        "surrogate": provenance, "gate_thresholds": thresholds, **optimiser.describe(),
    })
    state = {"scored": 0, "best_cd": None, "timings": []}
    try:
        stop_reason = _drive(state, optimiser, iterations, run_id, log, evaluate, body_style, constraints,
                             thresholds)
    except BaseException as exc:
        _finish(log, run_id, f"error: {type(exc).__name__}: {exc}", state)
        raise
    _finish(log, run_id, stop_reason, state)
    return run_id


def _drive(state: dict, optimiser, iterations: int, run_id: str, log: RunLog,
           evaluate: Callable[[dict[str, float], str], dict], body_style: str,
           constraints: list[Constraint], thresholds: dict) -> str:
    """Score the run; return why it stopped."""
    started = time.monotonic()
    for iteration in range(iterations):
        step_started = time.perf_counter()
        x = None if iteration == 0 else optimiser.ask()
        params = default_params() if x is None else to_params(x)
        proposed = time.perf_counter()
        evaluation = evaluate(params, body_style)
        scored = time.perf_counter()
        reason = rejection_reason(evaluation, body_style=body_style, **thresholds)
        if reason and reason.startswith("invalid prediction:"):
            raise ValueError(reason)
        broken = violations(params, constraints)
        if reason is None and broken:
            # Only the baseline can reach here: the search box obeys the rules.
            reason = "reference only: " + "; ".join(broken)
        if x is not None:
            optimiser.tell(x, evaluation["cd"] if reason is None else math.inf)
        if reason is None and (state["best_cd"] is None or evaluation["cd"] < state["best_cd"]):
            state["best_cd"] = evaluation["cd"]
        log.append(IterationRecord(
            run_id=run_id, iteration=iteration, params=params, body_style=body_style,
            cd=evaluation["cd"], cl=evaluation["cl"], uncertainty=evaluation["uncertainty"],
            in_distribution=evaluation["in_distribution"], accepted=reason is None,
            reject_reason=reason, reasoning="", best_cd_so_far=state["best_cd"],
            wall_clock_s=time.monotonic() - started, optimiser=optimiser.name,
            ood_score=evaluation.get("ood_score"),
        ))
        state["scored"] += 1
        latency_ms = evaluation.get("latency_ms")
        timing = {
            "iteration": iteration, "live": False, "propose_s": proposed - step_started,
            "model_calls": 0, "model_latency_s": 0.0, "failed_proposals": 0,
            "score_s": scored - proposed,
            "surrogate_latency_ms": float(latency_ms) if isinstance(latency_ms, (int, float)) else None,
            "step_s": time.perf_counter() - step_started,
        }
        state["timings"].append(timing)
        log.append_event(run_id, {"type": "step", **timing})
        if x is None and reason and not reason.startswith("reference only"):
            # No trusted start means nothing fair to improve on.
            return f"baseline rejected: {reason}"
    return "evaluation budget used"


def _finish(log: RunLog, run_id: str, stop_reason: str, state: dict) -> None:
    log.append_event(run_id, {"type": "run_finished", "stop_reason": stop_reason,
                              "scored": state["scored"], "best_cd": state["best_cd"],
                              "usage": None, "timing": designer.timing_summary(state["timings"])})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline optimiser against the evaluation service.")
    parser.add_argument("--method", choices=METHODS, default="cmaes")
    parser.add_argument("--iterations", type=int, default=25, help="Total scored attempts, including the baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--body-style", default=DEFAULT_BODY_STYLE, help="Body style for the whole run")
    parser.add_argument("--constraint", action="append", default=[], metavar="RULE",
                        help="Numeric rule, e.g. 'A_Car_Roof_Height>=0'. Repeatable.")
    parser.add_argument("--sigma", type=float, default=0.2, help="CMA-ES initial step, as a fraction of each range")
    parser.add_argument("--popsize", type=int, default=None, help="CMA-ES designs per generation (default 4+3ln n)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        constraints = [parse_constraint(rule) for rule in args.constraint]
    except ValueError as exc:
        parser.error(str(exc))
    run_id = args.run_id or f"{args.method}-{uuid.uuid4().hex[:12]}"
    try:
        result = run(args.method, args.iterations, run_id, args.seed, body_style=args.body_style,
                     constraints=constraints, sigma=args.sigma, popsize=args.popsize)
    except (ValueError, OSError, requests.RequestException) as exc:
        parser.exit(1, f"Run failed: {exc}\n")
    log = RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
    rows = log.read(result)
    for row in rows:
        verdict = "ok " if row["accepted"] else "REJ"
        best = row["best_cd_so_far"]
        print(f"{row['iteration']:3d} {verdict} Cd {row['cd']:.5f}  best {best if best is None else f'{best:.5f}'}"
              + (f"  ({row['reject_reason'].split(';')[0]})" if row["reject_reason"] else ""))
    finished = log.read_events(result)[-1]
    trusted = sum(row["accepted"] for row in rows)
    print(f"Run ID: {result}. {finished['scored']} scored, {trusted} trusted. "
          f"Best trusted Cd: {finished['best_cd']}")


if __name__ == "__main__":
    main()
