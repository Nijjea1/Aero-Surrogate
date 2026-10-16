"""Track B's reproducible scoring seam for an agent/optimiser comparison.

The optimiser stays in its owner's namespace. It can call BudgetedEvaluator
with a complete parameter dictionary and the fixed body style, just as the
agent does. A predictor invocation spends one evaluation even if it raises;
validation failures and duplicates never reach the predictor. This is separate
from the provider's dollar budget and from the frozen IterationRecord schema.

Call ``summary`` in a runner's finally block and persist the JSON beside its
normal run log. Method-specific settings (seed, provider model, proposal
policy) belong in metadata; they are disclosed rather than falsely equated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from ..contracts import DEFAULT_BODY_STYLE, SCHEMA_VERSION, default_params, validate_body_style
from ..service.gate import rejection_reason
from ..service.predictor import validate_design_params
from .constraints import parse_constraint, violations


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ComparisonProtocol:
    """Immutable shared conditions, with an explicit all-zero starting design.

    The SHA identifies the complete checkpoint file, not its display version.
    Constraints are canonicalised so whitespace/order do not create false
    mismatches. A baseline outside the feasible set is refused up front.
    """

    checkpoint_sha256: str
    model_version: str
    git_revision: str
    uncertainty_threshold: float
    ood_threshold: float
    body_style: str = DEFAULT_BODY_STYLE
    constraints: tuple[str, ...] = ()
    evaluation_budget: int = 25
    max_attempts: int = 2500
    source_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a full SHA-256 digest")
        object.__setattr__(self, "checkpoint_sha256", self.checkpoint_sha256.lower())
        if (self.source_sha256 is not None and
                (not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256))):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest or null")
        for name in ("model_version", "git_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.body_style, str) or validate_body_style(self.body_style):
            raise ValueError("invalid body_style")
        for name in ("uncertainty_threshold", "ood_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        for name in ("evaluation_budget", "max_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_attempts < self.evaluation_budget:
            raise ValueError("max_attempts must cover the evaluation budget")
        if not isinstance(self.constraints, (tuple, list)) or not all(isinstance(c, str) for c in self.constraints):
            raise ValueError("constraints must be rule strings")
        rules = [parse_constraint(rule) for rule in self.constraints]
        if any(not math.isfinite(rule.value) for rule in rules):
            raise ValueError("constraint values must be finite")
        if violations(default_params(), rules):
            raise ValueError("the all-zero baseline must satisfy every constraint")
        canonical = (f"{rule.name} {rule.op} {format(Decimal(str(rule.value)), 'f')}" for rule in rules)
        object.__setattr__(self, "constraints", tuple(sorted(set(canonical))))

    @property
    def baseline_params(self) -> dict:
        return default_params()

    @property
    def thresholds(self) -> dict:
        return {"threshold": self.uncertainty_threshold, "ood_threshold": self.ood_threshold}

    def to_dict(self) -> dict:
        return {
            "protocol_version": 1, "schema_version": SCHEMA_VERSION,
            "checkpoint_sha256": self.checkpoint_sha256, "model_version": self.model_version,
            "git_revision": self.git_revision, "body_style": self.body_style,
            "baseline_params": self.baseline_params, "constraints": list(self.constraints),
            "uncertainty_threshold": self.uncertainty_threshold, "ood_threshold": self.ood_threshold,
            "evaluation_budget": self.evaluation_budget, "max_attempts": self.max_attempts,
            "source_sha256": self.source_sha256,
            "budget_policy": "predictor calls including baseline, gate rejections and predictor errors",
            "duplicate_policy": "refuse before scoring", "invalid_policy": "refuse before scoring",
        }

    @property
    def protocol_id(self) -> str:
        return hashlib.sha256(_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping) -> ComparisonProtocol:
        if not isinstance(value, Mapping):
            raise ValueError("protocol must be an object")
        names = ("checkpoint_sha256", "model_version", "git_revision", "uncertainty_threshold",
                 "ood_threshold", "body_style", "constraints", "evaluation_budget", "max_attempts", "source_sha256")
        try:
            protocol = cls(**{name: value[name] for name in names})
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete comparison protocol") from exc
        if dict(value) != protocol.to_dict():
            raise ValueError("unsupported, changed or noncanonical comparison protocol")
        return protocol


class ComparisonRefused(ValueError):
    """Invalid/duplicate proposal; no predictor invocation was made."""


class ComparisonStopped(RuntimeError):
    """Cancellation, exhausted budget, or a previous fatal scoring failure."""


def _result(protocol, outcomes, stop_reason):
    best = None
    curve = []
    for outcome in outcomes:
        if outcome["status"] == "accepted":
            best = outcome["cd"] if best is None else min(best, outcome["cd"])
        curve.append(best)
    baseline = outcomes[0]["cd"] if outcomes and outcomes[0]["status"] == "accepted" else None
    counts = {status: sum(row["status"] == status for row in outcomes)
              for status in ("accepted", "rejected", "error")}
    complete = (baseline is not None and len(outcomes) == protocol.evaluation_budget and not counts["error"]
                and stop_reason == "evaluation budget used")
    return {"calls": len(outcomes), "counts": counts, "complete": complete,
            "baseline_cd": baseline, "best_cd": best, "best_curve": curve,
            "delta_cd": best - baseline if baseline is not None and best is not None else None,
            "improvement_fraction": ((baseline - best) / baseline
                                     if baseline is not None and baseline > 0 and best is not None else None)}


class BudgetedEvaluator:
    """A serialised callable using exactly the protocol's scoring conditions.

    The first actual evaluation must be the common baseline. Parameters and
    results are copied so a caller or predictor cannot mutate evidence later.
    Malformed predictions are fatal; ordinary trust rejections remain valid
    scored outcomes. This adapter does not call any provider or write files.
    """

    def __init__(self, evaluator: Callable, protocol: ComparisonProtocol, *, should_stop=None):
        self.protocol = protocol
        self._evaluate = evaluator
        self._should_stop = should_stop
        self._rules = [parse_constraint(rule) for rule in protocol.constraints]
        self._lock = threading.RLock()
        self._seen = set()
        self._outcomes = []
        self._refusals = []
        self._attempts = 0
        self._halted = None
        self._started = time.monotonic()

    @property
    def calls(self):
        with self._lock:
            return len(self._outcomes)

    @property
    def attempts(self):
        with self._lock:
            return self._attempts

    def __call__(self, params, body_style):
        with self._lock:
            if self._halted:
                raise ComparisonStopped(self._halted)
            if self._should_stop is not None and self._should_stop():
                self._halted = "cancelled"
                raise ComparisonStopped(self._halted)
            if self.calls >= self.protocol.evaluation_budget:
                raise ComparisonStopped("evaluation budget used")
            if self._attempts >= self.protocol.max_attempts:
                self._halted = "proposal attempt budget used"
                raise ComparisonStopped(self._halted)
            self._attempts += 1
            problems = validate_design_params(params)
            if body_style != self.protocol.body_style:
                problems.append("body style differs from comparison protocol")
            if not problems:
                problems.extend(violations(params, self._rules))
            if not problems and not self._outcomes and params != self.protocol.baseline_params:
                problems.append("the first evaluation must be the all-zero baseline")
            key = tuple(float(params[name]) for name in default_params()) if not problems else None
            if not problems and key in self._seen:
                problems.append("this exact design was already scored")
            if problems:
                detail = "; ".join(problems)
                self._refusals.append({"attempt": self._attempts, "reason": detail})
                raise ComparisonRefused(detail)
            self._seen.add(key)
            row = {"call": self.calls + 1, "params": dict(params), "status": "error", "cd": None,
                   "reject_reason": None, "error": None}
            # Reserve before calling so a failure still consumes its slot.
            self._outcomes.append(row)
            try:
                result = self._evaluate(dict(params), body_style)
                if not isinstance(result, dict):
                    raise ValueError("invalid prediction: evaluation must be an object")
                reason = rejection_reason(result, body_style=body_style, **self.protocol.thresholds)
                if reason and reason.startswith("invalid prediction:"):
                    raise ValueError(reason)
                _json(result)  # Evidence must remain strict JSON, never NaN.
                row.update(status="rejected" if reason else "accepted", cd=float(result["cd"]),
                           reject_reason=reason, evaluation=copy.deepcopy(result))
                if row["call"] == 1 and reason:
                    self._halted = f"baseline rejected: {reason}"
                return copy.deepcopy(result)
            except BaseException as exc:
                self._halted = f"error: {type(exc).__name__}"
                row["error"] = self._halted
                raise

    def summary(self, method: str, run_id: str, stop_reason: str, *, metadata=None) -> dict:
        with self._lock:
            if not all(isinstance(v, str) and v.strip() for v in (method, run_id, stop_reason)):
                raise ValueError("method, run_id and stop_reason are required")
            reason = self._halted or stop_reason
            result = {"summary_version": 1, "protocol": self.protocol.to_dict(),
                      "protocol_id": self.protocol.protocol_id, "method": method, "run_id": run_id,
                      "stop_reason": reason, "attempts": self.attempts,
                      "refusals": copy.deepcopy(self._refusals), "outcomes": copy.deepcopy(self._outcomes),
                      "wall_clock_s": time.monotonic() - self._started,
                      "metadata": copy.deepcopy(metadata or {})}
            result.update(_result(self.protocol, self._outcomes, reason))
            _json(result)
            return result


def summarize_pair(left: dict, right: dict) -> dict:
    """Reject incomparable/tampered artifacts; retain incomplete outcomes.

    No winner is declared: one matched pair is evidence of those runs only.
    All derived results are checked against recorded calls and the shared gate.
    """
    protocols = []
    for summary in (left, right):
        protocol = ComparisonProtocol.from_dict(summary["protocol"])
        protocols.append(protocol)
        if summary.get("summary_version") != 1 or summary.get("protocol_id") != protocol.protocol_id:
            raise ValueError("summary protocol identity differs")
        rows = summary["outcomes"]
        if not isinstance(rows, list) or len(rows) > protocol.evaluation_budget:
            raise ValueError("invalid predictor call accounting")
        seen = set()
        for index, row in enumerate(rows):
            params = row["params"]
            if (validate_design_params(params) or violations(params, [parse_constraint(c) for c in protocol.constraints])
                    or row["call"] != index + 1 or (index == 0 and params != protocol.baseline_params)):
                raise ValueError("recorded design differs from comparison protocol")
            key = tuple(float(params[name]) for name in default_params())
            if key in seen:
                raise ValueError("duplicate recorded predictor call")
            seen.add(key)
            if row["status"] == "error":
                if not row.get("error") or index != len(rows) - 1 or row["cd"] is not None:
                    raise ValueError("invalid failed predictor call accounting")
            else:
                evaluation = row["evaluation"]
                reason = rejection_reason(evaluation, body_style=protocol.body_style, **protocol.thresholds)
                status = "rejected" if reason else "accepted"
                if ((reason and reason.startswith("invalid prediction:")) or row["status"] != status
                        or row["cd"] != evaluation["cd"] or row["reject_reason"] != reason):
                    raise ValueError("recorded gate outcome differs")
        if (type(summary["attempts"]) is not int or summary["attempts"] > protocol.max_attempts
                or summary["attempts"] != len(rows) + len(summary["refusals"])):
            raise ValueError("invalid proposal attempt accounting")
        derived = _result(protocol, rows, summary["stop_reason"])
        if any(summary.get(key) != value for key, value in derived.items()):
            raise ValueError("summary does not match recorded predictor calls")
        _json(summary)
    if protocols[0] != protocols[1]:
        raise ValueError("comparison protocols differ")
    if left["run_id"] == right["run_id"]:
        raise ValueError("comparison needs two distinct runs")
    return {"protocol_id": protocols[0].protocol_id, "complete": left["complete"] and right["complete"],
            "runs": [copy.deepcopy(left), copy.deepcopy(right)]}
