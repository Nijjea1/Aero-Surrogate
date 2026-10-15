"""The live Nemotron proposer.  OWNER: Pranoy.

One design step = one short tool-calling conversation that must end in a valid
`evaluate` call. The server, not the model, owns scoring, the trust gate and
logging: this module only turns the model's tool call into a validated
proposal, and turns invalid ones into precise feedback the model can fix.

Context is rebuilt from the run state every step (system prompt + one compact
state message), so prompt size and cost stay flat however long the run is.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from ..contracts import PARAM_BOUNDS, validate_params
from .constraints import Constraint, violations
from .provider import BudgetExceeded, ProviderError, TokenFactoryClient

SYSTEM_PROMPT = """You are an aerodynamicist minimising the drag coefficient (Cd) of a car,
working with a fast neural surrogate instead of CFD.

The design is sixteen morph parameters, each an OFFSET from the baseline body
of a fixed body style (all zeros = unmodified baseline). Angles are degrees,
lengths millimetres, curvatures and inclinations dimensionless.

Every step, call `evaluate` exactly once with the parameters you want to change
from the current best trusted design, plus your reasoning. You may first call
`compare` or `history` if it helps, but keep it brief.

Rules:
- Change at most two parameters per step so their effect is attributable
  (enforced: a step that changes more is refused).
- Stay inside each parameter's bounds and the listed constraints.
- The surrogate has a trust gate. A REJECTED result means the model does not
  understand that region (too uncertain, or too unlike its training data).
  Do not repeat or extend a rejected move; go back toward trusted designs.
- An accepted result is trusted, not necessarily better. Keep what lowers Cd.
- Do not propose a design that was already scored.
- Reasoning, in one or two sentences: hypothesis, the change, expected effect.
  Write it like an engineer's logbook, not a parameter dump."""


def _param_description(name: str) -> str:
    lo, hi = PARAM_BOUNDS[name]
    unit = "deg" if name.endswith("_Angle") else "mm" if any(
        name.endswith(s) for s in ("_Length", "_Width", "_Height", "_Thickness", "_Offset")
    ) else "dimensionless"
    return f"{unit}, offset from baseline, bounds [{lo:g}, {hi:g}]"


TOOLS = [
    {"type": "function", "function": {
        "name": "evaluate",
        "description": "Score a new design: the current best trusted design with these parameters changed. "
                       "Uses one evaluation from the budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "object",
                    "description": "Parameters to set (absolute offset values, not increments).",
                    "properties": {name: {"type": "number", "description": _param_description(name)}
                                   for name in PARAM_BOUNDS},
                    "additionalProperties": False,
                },
                "reasoning": {"type": "string", "description": "Hypothesis, change, expected effect."},
            },
            "required": ["changes", "reasoning"],
        },
    }},
    {"type": "function", "function": {
        "name": "compare",
        "description": "Parameter and Cd differences between two scored iterations of this run. Free.",
        "parameters": {
            "type": "object",
            "properties": {"iteration_a": {"type": "integer"}, "iteration_b": {"type": "integer"}},
            "required": ["iteration_a", "iteration_b"],
        },
    }},
    {"type": "function", "function": {
        "name": "history",
        "description": "Older scored attempts than those shown in the state message. Free.",
        "parameters": {
            "type": "object",
            "properties": {"last_n": {"type": "integer", "minimum": 1, "maximum": 30}},
            "required": ["last_n"],
        },
    }},
]


def _call_name(call: dict) -> str | None:
    function = call.get("function")
    return function.get("name") if isinstance(function, dict) else None


class NoProposal(RuntimeError):
    """The model did not produce a valid design within the step's round limit."""


class Cancelled(RuntimeError):
    """The run was asked to stop; raised before the next model call."""


@dataclass
class Proposal:
    params: dict[str, float]
    changes: dict[str, float]
    reasoning: str
    rounds: int
    feedback: list[str] = field(default_factory=list)


def design_key(params: dict[str, float]) -> tuple[float, ...]:
    """Numeric identity of a design in contract order. Adding 0.0 folds -0.0
    into 0.0, so the same car cannot be rescored under a different spelling."""
    return tuple(float(params[name]) + 0.0 for name in PARAM_BOUNDS)


def _int_arg(args: dict, name: str) -> int | None:
    value = args.get(name)
    return None if isinstance(value, bool) or not isinstance(value, int) else value


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _offsets(params: dict[str, float]) -> str:
    changed = {name: _fmt(value) for name, value in params.items() if value != 0.0}
    return json.dumps(changed) if changed else "baseline (all zeros)"


def _attempt_line(row: dict) -> str:
    verdict = "accepted" if row["accepted"] else f"REJECTED: {row['reject_reason']}"
    ood = f", ood {row['ood_score']:.2f}" if row.get("ood_score") is not None else ""
    return (f"it {row['iteration']}: {_offsets(row['params'])} -> Cd {row['cd']:.5f} "
            f"(sd {row['uncertainty']:.4f}{ood}) {verdict}")


def _best(history: list[dict]) -> dict | None:
    trusted = [row for row in history if row["accepted"]]
    return min(trusted, key=lambda row: row["cd"]) if trusted else None


def state_message(history: list[dict], *, body_style: str, evaluations_left: int,
                  total_evaluations: int, constraints: list[Constraint],
                  best_pressure: dict | None, provenance: str, window: int) -> str:
    best = _best(history)
    lines = [f"Body style: {body_style}. Evaluations left: {evaluations_left} of {total_evaluations}.",
             f"Surrogate: {provenance}."]
    if best:
        lines.append(f"Best trusted design: iteration {best['iteration']}, Cd {best['cd']:.5f}, "
                     f"offsets {_offsets(best['params'])}.")
        if best_pressure:
            regions = ", ".join(f"{k} {v:+.3f}" for k, v in best_pressure.items())
            lines.append(f"Mean pressure coefficient by region on the best design: {regions}.")
    else:
        lines.append("No trusted design yet: changes apply to the baseline body.")
    lines.append("Constraints enforced by code: " + (", ".join(map(str, constraints)) or "none") + ".")
    lines.append(f"Most recent attempts (newest last, {min(window, len(history))} of {len(history)}):")
    lines.extend(_attempt_line(row) for row in history[-window:])
    lines.append("Call evaluate with your next design.")
    return "\n".join(lines)


@dataclass
class NemotronProposer:
    client: TokenFactoryClient
    constraints: list[Constraint] = field(default_factory=list)
    window: int = 8
    max_rounds: int = 4
    max_tokens: int = 2048
    temperature: float = 0.6
    # Nano thinks for ~1.6k tokens per step by default and can be cut off
    # before its tool call; with thinking off it answers in ~110 tokens.
    # Super thinks briefly either way. Measured on Token Factory 2026-10-14.
    thinking: bool = True
    max_changes: int = 2
    # The run's spend limit, checked before every call including correction
    # rounds. None means only the global spend cap applies.
    budget_usd: float | None = None
    # Polled before every model call, so a cancelled run stops between
    # correction rounds too. A call already in flight is bounded by its timeout.
    should_stop: Callable[[], bool] | None = None

    def propose(self, history: list[dict], *, body_style: str, evaluations_left: int,
                total_evaluations: int, best_pressure: dict | None = None,
                provenance: str = "unknown") -> Proposal:
        best = _best(history)
        base = dict(best["params"]) if best else {name: 0.0 for name in PARAM_BOUNDS}
        scored = {design_key(row["params"]) for row in history}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_message(
                history, body_style=body_style, evaluations_left=evaluations_left,
                total_evaluations=total_evaluations, constraints=self.constraints,
                best_pressure=best_pressure, provenance=provenance, window=self.window)},
        ]
        feedback: list[str] = []
        for round_number in range(1, self.max_rounds + 1):
            if self.should_stop is not None and self.should_stop():
                raise Cancelled("run cancelled")
            if self.budget_usd is not None and self.client.usage.cost_usd >= self.budget_usd:
                raise BudgetExceeded(f"run budget ${self.budget_usd:.4f} reached: "
                                     f"${self.client.usage.cost_usd:.4f} spent")
            reply = self.client.chat(
                messages, tools=TOOLS, tool_choice="auto", max_tokens=self.max_tokens,
                temperature=self.temperature, budget_usd=self.budget_usd,
                extra=None if self.thinking else {"chat_template_kwargs": {"enable_thinking": False}})
            message = reply.get("message") if isinstance(reply, dict) else None
            message = message if isinstance(message, dict) else {}
            calls = message.get("tool_calls")
            calls = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
            content = message.get("content")
            messages.append({"role": "assistant", "content": content if isinstance(content, str) else "",
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                problem = ("reply was cut off before a tool call" if reply.get("finish_reason") == "length"
                           else "no tool call in reply")
                feedback.append(problem)
                messages.append({"role": "user", "content": f"{problem}. Call evaluate now."})
                continue
            proposal = None
            for call in calls:
                result, candidate = self._handle(call, history, base, scored, proposal is not None)
                if candidate:
                    proposal = candidate
                elif _call_name(call) == "evaluate":
                    feedback.append(result)
                call_id = call.get("id")
                messages.append({"role": "tool", "tool_call_id": call_id if isinstance(call_id, str) else "",
                                 "content": result})
            if proposal:
                proposal.rounds, proposal.feedback = round_number, feedback
                return proposal
        raise NoProposal(f"no valid design after {self.max_rounds} rounds: " + "; ".join(feedback[-3:]))

    def _handle(self, call: dict, history: list[dict], base: dict[str, float], scored: set[tuple],
                already_proposed: bool) -> tuple[str, Proposal | None]:
        function = call.get("function")
        if not isinstance(function, dict):
            return "error: malformed tool call", None
        name = function.get("name")
        raw = function.get("arguments")
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except (TypeError, ValueError):
            return "error: arguments must be a JSON object", None
        if name == "compare":
            return self._compare(history, args), None
        if name == "history":
            last_n = args.get("last_n")
            if isinstance(last_n, bool) or not isinstance(last_n, int) or not 1 <= last_n <= 30:
                return "error: last_n must be an integer from 1 to 30", None
            return "\n".join(_attempt_line(row) for row in history[-last_n:]) or "no attempts yet", None
        if name != "evaluate":
            return f"error: unknown tool '{name}'", None
        if already_proposed:
            return "ignored: only one evaluate per step", None
        return self._evaluate(args, base, scored)

    def _evaluate(self, args: dict, base: dict[str, float], scored: set[tuple]) -> tuple[str, Proposal | None]:
        changes, reasoning = args.get("changes"), args.get("reasoning")
        if not isinstance(changes, dict) or not changes:
            return "rejected before scoring: 'changes' must be a non-empty object", None
        if not isinstance(reasoning, str) or not reasoning.strip():
            return "rejected before scoring: 'reasoning' is required", None
        problems, values = [], {}
        for key, value in changes.items():
            if key not in PARAM_BOUNDS:
                problems.append(f"unknown parameter '{key}'")
                continue
            # Small models sometimes quote numbers ("0.2"); that is the value
            # they meant, so parse it. Anything else is refused, never guessed.
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"'{key}' must be a finite number (got {value!r})")
            else:
                values[key] = float(value)
        if problems:
            return "rejected before scoring: " + "; ".join(problems), None
        changed = sorted(key for key, value in values.items() if value + 0.0 != float(base[key]) + 0.0)
        if len(changed) > self.max_changes:
            return (f"rejected before scoring: change at most {self.max_changes} parameters per step "
                    f"(this changes {len(changed)}: {', '.join(changed)})"), None
        params = dict(base, **values)
        problems = validate_params(params) + violations(params, self.constraints)
        if problems:
            return "rejected before scoring (not clamped): " + "; ".join(problems), None
        if design_key(params) in scored:
            return "rejected before scoring: this exact design was already scored; try something new", None
        return "accepted for scoring", Proposal(params=params, changes=values,
                                                reasoning=reasoning.strip(), rounds=0)

    @staticmethod
    def _compare(history: list[dict], args: dict) -> str:
        rows = {row["iteration"]: row for row in history}
        first, second = _int_arg(args, "iteration_a"), _int_arg(args, "iteration_b")
        if first is None or second is None:
            return "error: iteration_a and iteration_b must be integers"
        a, b = rows.get(first), rows.get(second)
        if a is None or b is None:
            return "error: both iterations must already be scored in this run"
        diff = {name: _fmt(b["params"][name] - a["params"][name])
                for name in PARAM_BOUNDS if b["params"][name] != a["params"][name]}
        return (f"iteration {b['iteration']} minus {a['iteration']}: parameter changes {json.dumps(diff)}; "
                f"Cd {b['cd'] - a['cd']:+.5f}; accepted {a['accepted']} -> {b['accepted']}")


__all__ = ["Cancelled", "NemotronProposer", "NoProposal", "Proposal", "ProviderError", "SYSTEM_PROMPT",
           "TOOLS", "design_key", "state_message"]
