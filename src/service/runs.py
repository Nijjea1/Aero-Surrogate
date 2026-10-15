"""Agent runs started over the API.  OWNER: Pranoy.

RunManager starts design-agent runs (src.agent.designer.run) in background
threads, enforces who may spend how much, and reports progress. The run log's
event files are the source of truth for everything that must survive a
restart (who started a run, when, what it cost, how it ended); the in-memory
table only holds what is live in this process: the cancel flag, the thread,
and step-by-step progress.

Limits, all checked before a run starts (refused, never silently lowered):
- role ceilings on iterations and per-run budget (public or admin);
- at most settings.max_concurrent_runs queued or running runs per role in
  this process. Public and admin runs have separate slots, so public callers
  can never lock the admin out (worst case 2 x max_concurrent_runs at once);
- the public daily budget: today's (UTC) public spend from the event logs,
  counting a finished run at its billed cost and an unfinished one at its full
  run budget (reserved while in flight, or when its outcome is unknown).
  Public runs started yesterday that are still unfinished are counted too,
  so spend cannot slip across midnight unseen. A recent event log that cannot
  be read counts at the full public run budget: the check fails closed.

Cancelling: the admin can cancel any run this process owns. A public caller
can cancel only a public run, and only with the cancel_token returned by
POST /runs, so strangers cannot stop each other's runs.

Orphans: a run with run_started but no run_finished that this process does
not own was cut off by a crash or restart, or is live in another process (a
CLI run). Orphans whose events have not changed for `orphan_after_s`
(default 30 minutes, longer than a worst-case stalled model call) are closed
with a run_finished event, stop_reason "interrupted: service restarted", at
startup and again whenever their status is read later. Newer ones are left
untouched and reported as "unknown", because they may still be running
elsewhere. On shutdown, runs whose threads do not stop in time are closed
with "interrupted: service stopping" and their known usage. Closure without
usage records None, and the daily budget then counts the run at its full
run budget.

Several service processes must use separate log roots (see RunLog.list_runs).
"""

from __future__ import annotations

import hmac
import logging
import math
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..agent import designer
from ..agent.constraints import Constraint, parse_constraint
from ..agent.nemotron import NemotronProposer
from ..agent.provider import TokenFactoryClient
from ..contracts import DEFAULT_BODY_STYLE, validate_body_style
from .runlog import RunLog
from .security import ADMIN, PUBLIC
from .settings import AGENT_MODELS, Settings

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
TERMINAL = ("finished", "cancelled", "failed", "interrupted")
THINKING = ("auto", "on", "off")
MAX_CONSTRAINTS = 8
INTERRUPTED = "interrupted: service restarted"
STOPPING = "interrupted: service stopping"
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class RunError(Exception):
    """Base for refusals; the endpoint maps each subclass to a status code."""


class RunRequestError(RunError):
    """The request is invalid or above the caller's ceilings (422)."""


class RunLimitError(RunError):
    """Too many runs, or the public daily budget is used up (429)."""


class RunUnavailable(RunError):
    """The agent cannot be started here, e.g. no provider key (503)."""


class RunNotFound(RunError):
    """No such run (404)."""


class RunForbidden(RunError):
    """The caller may not act on this run (403)."""


class RunConflict(RunError):
    """The run is not managed by this process, so it cannot be cancelled (409)."""


def default_proposer_factory(model: str, constraints: list[Constraint], thinking: str) -> NemotronProposer:
    """A live Token Factory proposer. Raises ProviderError without NEBIUS_API_KEY.

    thinking "auto" means off for Nano (it otherwise thinks for ~1.6k tokens
    and can be cut off before its tool call) and on for everything else.
    """
    client = TokenFactoryClient.from_env(model=model)
    think = thinking == "on" or (thinking == "auto" and model != "nano")
    return NemotronProposer(client, constraints=list(constraints), thinking=think)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(text) -> float | None:
    try:
        return datetime.strptime(text, _TIME_FORMAT).replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


_FAILED_PREFIXES = ("error:", "provider error", )


def state_from_reason(stop_reason: str | None) -> str:
    """Terminal state named by a run_finished stop reason.

    "finished" means the run ended on its own terms (its evaluation budget or
    its spend budget). A provider failure, a model that kept producing no
    valid design, or an exception is "failed", so a UI reading only the state
    never shows those as successes.
    """
    reason = stop_reason if isinstance(stop_reason, str) else ""
    if reason == "cancelled":
        return "cancelled"
    if reason.startswith(_FAILED_PREFIXES) or "without a valid proposal" in reason:
        return "failed"
    if reason.startswith("interrupted"):
        return "interrupted"
    return "finished"


def _latest(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {key: record.get(key) for key in ("iteration", "cd", "accepted", "reject_reason", "reasoning")}


def _best_cd(records: list[dict]) -> float | None:
    trusted = [r["cd"] for r in records if r.get("accepted")]
    return min(trusted) if trusted else None


@dataclass
class _Run:
    """A run owned by this process. Guarded by RunManager._lock."""
    run_id: str
    role: str
    model: str
    body_style: str
    iterations: int
    constraints: list[str]
    run_budget_usd: float
    thinking: bool
    started_at: str
    proposer: NemotronProposer
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    state: str = "queued"
    scored: int = 0
    best_cd: float | None = None
    latest: dict | None = None
    timings: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None
    cancel_token: str | None = None


class RunManager:
    def __init__(self, predictor, runlog: RunLog, settings: Settings, *,
                 proposer_factory: Callable[[str, list[Constraint], str], NemotronProposer] | None = None,
                 clock: Callable[[], datetime] = _utc_now,
                 orphan_after_s: float = 1800.0):
        self.predictor = predictor
        self.runlog = runlog
        self.settings = settings
        self.proposer_factory = proposer_factory or default_proposer_factory
        self.clock = clock
        self.orphan_after_s = orphan_after_s
        self._lock = threading.Lock()
        self._runs: dict[str, _Run] = {}
        # (mtime_ns, size) -> (run_started, run_finished) per events file, so
        # the daily-budget check re-reads only files that changed.
        self._spend_cache: dict[str, tuple[tuple[int, int], dict | None, dict | None]] = {}
        self.closed_orphans = self._close_orphans()

    # ----- log reading -------------------------------------------------

    @property
    def _events_dir(self) -> Path:
        return self.runlog.root / "events"

    def _check_id(self, run_id: str) -> None:
        self.runlog.validate_run_id(run_id)

    def _events(self, run_id: str) -> list[dict]:
        return self.runlog.read_events(run_id)

    def _records(self, run_id: str) -> list[dict]:
        try:
            return self.runlog.read(run_id)
        except FileNotFoundError:
            return []

    def _logged_ids(self) -> list[str]:
        ids = set()
        for folder in (self._events_dir, self.runlog.root):
            if folder.is_dir():
                ids.update(path.stem for path in folder.glob("*.jsonl"))
        return sorted(ids)

    def _mtime(self, run_id: str) -> float | None:
        for path in (self._events_dir / f"{run_id}.jsonl", self.runlog.root / f"{run_id}.jsonl"):
            try:
                return path.stat().st_mtime
            except OSError:
                continue
        return None

    @staticmethod
    def _split(events: list[dict]) -> tuple[dict | None, dict | None, list[dict]]:
        started = next((e for e in events if e.get("type") == "run_started"), None)
        finished = next((e for e in reversed(events) if e.get("type") == "run_finished"), None)
        steps = [e for e in events if e.get("type") == "step"]
        return started, finished, steps

    # ----- startup -----------------------------------------------------

    def _close_orphans(self) -> list[str]:
        """Close stale runs that started but never finished; see the module doc."""
        if not self._events_dir.is_dir():
            return []
        return [path.stem for path in sorted(self._events_dir.glob("*.jsonl"))
                if self._close_if_orphaned(path.stem)]

    def _close_if_orphaned(self, run_id: str) -> bool:
        """Close one run if it started, never finished, is not ours and has
        been silent longer than orphan_after_s. True when it was closed."""
        with self._lock:
            if run_id in self._runs:
                return False
        try:
            self._check_id(run_id)
            started, finished, steps = self._split(self.runlog.read_events(run_id))
            if started is None or finished is not None:
                return False
            mtime = self._mtime(run_id)
            if mtime is None or time.time() - mtime <= self.orphan_after_s:
                return False  # may be live in another process: report "unknown"
            records = self._records(run_id)
            self.runlog.append_event(run_id, {
                "type": "run_finished", "stop_reason": INTERRUPTED, "scored": len(records),
                "best_cd": _best_cd(records), "usage": None,
                "timing": designer.timing_summary(steps),
            })
            return True
        except (OSError, ValueError) as exc:
            log.warning("could not inspect run %s for orphan closure: %s", run_id, exc)
            return False

    # ----- starting ----------------------------------------------------

    def _validate(self, request: Mapping, role: str) -> dict:
        if not isinstance(request, Mapping):
            raise RunRequestError("request must be an object")
        model = request.get("model")
        if model not in AGENT_MODELS:
            raise RunRequestError(f"model must be one of {list(AGENT_MODELS)}")
        iterations = request.get("iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 2:
            raise RunRequestError("iterations must be an integer >= 2 (the baseline plus at least one proposal)")
        body_style = request.get("body_style") or DEFAULT_BODY_STYLE
        problems = validate_body_style(body_style)
        if problems:
            raise RunRequestError("; ".join(problems))
        rules = request.get("constraints") or []
        if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
            raise RunRequestError("constraints must be a list of rule strings")
        if len(rules) > MAX_CONSTRAINTS:
            raise RunRequestError(f"at most {MAX_CONSTRAINTS} constraints")
        try:
            constraints = [parse_constraint(rule) for rule in rules]
        except ValueError as exc:
            raise RunRequestError(str(exc)) from None
        thinking = request.get("thinking") or "auto"
        if thinking not in THINKING:
            raise RunRequestError(f"thinking must be one of {list(THINKING)}")
        if role == ADMIN:
            max_iterations, max_budget = self.settings.admin_max_iterations, self.settings.admin_run_budget_usd
        elif role == PUBLIC:
            max_iterations, max_budget = self.settings.public_max_iterations, self.settings.public_run_budget_usd
        else:
            raise RunRequestError(f"unknown role {role!r}")
        budget = request.get("run_budget_usd")
        if budget is None:
            budget = max_budget
        if (isinstance(budget, bool) or not isinstance(budget, (int, float))
                or not math.isfinite(budget) or budget <= 0):
            raise RunRequestError("run_budget_usd must be a positive finite number")
        if iterations > max_iterations:
            raise RunRequestError(f"iterations {iterations} is above the {role} limit of {max_iterations}")
        if budget > max_budget:
            raise RunRequestError(f"run_budget_usd {budget:g} is above the {role} limit of ${max_budget:g}")
        return {"model": model, "iterations": iterations, "body_style": body_style,
                "constraints": constraints, "run_budget_usd": float(budget), "thinking": thinking}

    def _recent_event_files(self, since: float) -> list[Path]:
        """Event files modified at or after `since` (epoch seconds). Older
        files cannot hold today's spend, so the scan stays proportional to
        recent activity rather than to every run ever made."""
        if not self._events_dir.is_dir():
            return []
        files = []
        for path in self._events_dir.glob("*.jsonl"):
            try:
                if path.stat().st_mtime >= since:
                    files.append(path)
            except OSError:
                continue
        return files

    def _start_and_finish(self, path: Path) -> tuple[dict | None, dict | None]:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._spend_cache.get(path.stem)
        if cached and cached[0] == key:
            return cached[1], cached[2]
        started, finished, _ = self._split(self.runlog.read_events(path.stem))
        self._spend_cache[path.stem] = (key, started, finished)
        return started, finished

    def public_spend_today(self) -> float:
        """Today's (UTC) public spend, reserved budgets included. Call under _lock.

        Counted: public runs started today (billed cost if finished, else their
        whole run budget), public runs started yesterday that are still
        unfinished (their budget is still at risk today), and any recent event
        log that cannot be read (at the full public run budget: fail closed).
        """
        now = self.clock().astimezone(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        total, seen = 0.0, set()
        for path in self._recent_event_files((midnight - timedelta(days=1)).timestamp()):
            try:
                started, finished = self._start_and_finish(path)
            except (OSError, ValueError) as exc:
                log.warning("counting unreadable run %s at the public run budget: %s", path.stem, exc)
                total += self.settings.public_run_budget_usd
                continue
            if started is None:
                continue
            seen.add(path.stem)
            day = str(started.get("started_at", ""))[:10]
            if started.get("started_by") != PUBLIC or day not in (today, yesterday):
                continue
            usage = finished.get("usage") if finished else None
            cost = usage.get("cost_usd") if isinstance(usage, dict) else None
            if day == yesterday:
                if finished is None:  # still running across midnight
                    total += float(started.get("run_budget_usd") or 0.0)
            elif isinstance(cost, (int, float)) and math.isfinite(cost):
                total += cost
            else:  # in flight, or outcome unknown: its whole budget is reserved
                total += float(started.get("run_budget_usd") or 0.0)
        for run in self._runs.values():  # registered, run_started not yet written
            if run.run_id not in seen and run.state in ACTIVE and run.role == PUBLIC:
                total += run.run_budget_usd
        return total

    def start(self, request: Mapping, role: str) -> dict:
        """Validate, check limits and start a run in the background."""
        spec = self._validate(request, role)
        try:
            proposer = self.proposer_factory(spec["model"], spec["constraints"], spec["thinking"])
        except Exception as exc:  # e.g. ProviderError: NEBIUS_API_KEY is not set
            raise RunUnavailable(f"the agent provider is not available: {exc}") from None
        with self._lock:
            active = sum(run.state in ACTIVE and run.role == role for run in self._runs.values())
            if active >= self.settings.max_concurrent_runs:
                raise RunLimitError(f"{active} {role} run(s) already in progress; the limit is "
                                    f"{self.settings.max_concurrent_runs}. Try again later.")
            if role == PUBLIC:
                spent = self.public_spend_today()
                limit = self.settings.public_daily_budget_usd
                if spent + spec["run_budget_usd"] > limit + 1e-12:
                    raise RunLimitError(
                        f"the public daily budget of ${limit:g} is used up "
                        f"(${spent:.4f} spent or reserved today, this run needs ${spec['run_budget_usd']:g}); "
                        "it resets at 00:00 UTC")
            run_id = f"agent-{uuid.uuid4().hex[:12]}"
            run = _Run(run_id=run_id, role=role, model=proposer.client.model,
                       body_style=spec["body_style"], iterations=spec["iterations"],
                       constraints=[str(c) for c in spec["constraints"]],
                       run_budget_usd=spec["run_budget_usd"], thinking=bool(proposer.thinking),
                       started_at=self.clock().astimezone(timezone.utc).strftime(_TIME_FORMAT),
                       proposer=proposer,
                       cancel_token=secrets.token_urlsafe(16) if role == PUBLIC else None)
            run.thread = threading.Thread(target=self._work, args=(run, spec["constraints"]),
                                          name=f"run-{run_id}", daemon=True)
            self._runs[run_id] = run
            run.thread.start()
            status = self._live_status(run)
            if run.cancel_token:  # shown once, to the caller who started it
                status["cancel_token"] = run.cancel_token
            return status

    def _work(self, run: _Run, constraints: list[Constraint]) -> None:
        with self._lock:
            if run.state == "queued":
                run.state = "running"

        def on_step(record: dict, timing: dict) -> None:
            with self._lock:
                run.scored = record["iteration"] + 1
                run.best_cd = record.get("best_cd_so_far")
                run.latest = _latest(record)
                run.timings.append(dict(timing))

        try:
            designer.run(
                run.iterations, run.run_id, False,
                evaluator=designer.predictor_evaluator(self.predictor),
                thresholds=designer.predictor_thresholds(self.predictor),
                provenance=designer.predictor_provenance(self.predictor),
                proposer=run.proposer, constraints=constraints, body_style=run.body_style,
                run_budget_usd=run.run_budget_usd, should_stop=run.stop.is_set, on_step=on_step,
                metadata={"started_by": run.role, "started_at": run.started_at, "requested_via": "api"},
                log=self.runlog,
            )
        except BaseException as exc:  # never let a run thread die silently
            log.exception("agent run %s failed", run.run_id)
            with self._lock:
                run.state, run.error = "failed", f"{type(exc).__name__}: {exc}"
                run.stop_reason = f"error: {run.error}"
            return
        try:
            _, finished, _ = self._split(self._events(run.run_id))
        except (OSError, ValueError) as exc:
            finished = None
            log.warning("run %s finished but its events could not be read: %s", run.run_id, exc)
        with self._lock:
            run.stop_reason = finished.get("stop_reason") if finished else None
            run.state = state_from_reason(run.stop_reason)

    # ----- reporting ---------------------------------------------------

    def _live_status(self, run: _Run) -> dict:
        """Status of an owned run from memory. Call under _lock."""
        return {
            "run_id": run.run_id, "state": run.state, "model": run.model, "body_style": run.body_style,
            "iterations": run.iterations, "scored": run.scored, "best_cd": run.best_cd,
            "latest_step": dict(run.latest) if run.latest else None, "stop_reason": run.stop_reason,
            "cost_usd": run.proposer.client.usage.cost_usd,
            "timing": designer.timing_summary(list(run.timings)),
            "started_by": run.role, "started_at": run.started_at, "requested_via": "api",
            "constraints": list(run.constraints), "run_budget_usd": run.run_budget_usd,
            "thinking": run.thinking, "cancel_requested": run.stop.is_set(), "error": run.error,
        }

    def _logged_status(self, run_id: str, close_stale: bool = False) -> dict | None:
        if close_stale:
            self._close_if_orphaned(run_id)
        events = self._events(run_id)
        records = self._records(run_id)
        if not events and not records:
            return None
        started, finished, steps = self._split(events)
        started = started or {}
        usage = finished.get("usage") if finished else None
        if finished:
            state = state_from_reason(finished.get("stop_reason"))
            timing = finished.get("timing")
        else:
            state = "unknown"  # live elsewhere, or cut off and not yet closed
            timing = designer.timing_summary(steps)
        latest = records[-1] if records else None
        return {
            "run_id": run_id, "state": state, "model": started.get("model"),
            "body_style": started.get("body_style") or (latest or {}).get("body_style"),
            "iterations": started.get("evaluation_budget"),
            "scored": finished.get("scored", len(records)) if finished else len(records),
            "best_cd": finished.get("best_cd") if finished else _best_cd(records),
            "latest_step": _latest(latest),
            "stop_reason": finished.get("stop_reason") if finished else None,
            "cost_usd": usage.get("cost_usd") if isinstance(usage, dict) else None,
            "timing": timing,
            "started_by": started.get("started_by"), "started_at": started.get("started_at"),
            "requested_via": started.get("requested_via"),
            "constraints": list(started.get("constraints") or []),
            "run_budget_usd": started.get("run_budget_usd"), "thinking": started.get("thinking"),
            "cancel_requested": False, "error": None,
        }

    def status(self, run_id: str) -> dict:
        self._check_id(run_id)
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None and run.state in ACTIVE:
                return self._live_status(run)
        # Terminal or not ours: the logs are complete (designer.run writes
        # run_finished before the owning thread marks the run terminal).
        logged = self._logged_status(run_id, close_stale=run is None)
        if run is None:
            if logged is None:
                raise RunNotFound(f"run {run_id} not found")
            return logged
        with self._lock:
            live = self._live_status(run)
        if logged is None:  # failed before run_started was written
            return live
        logged.update(state=live["state"], error=live["error"], cancel_requested=live["cancel_requested"],
                      started_by=live["started_by"], started_at=live["started_at"])
        if live["state"] == "failed" and logged["stop_reason"] is None:
            logged["stop_reason"] = live["stop_reason"]
        return logged

    def list(self, limit: int = 20) -> list[dict]:
        """Newest first, by started_at, falling back to file mtime."""
        with self._lock:
            owned = {run_id: _parse_time(run.started_at) for run_id, run in self._runs.items()}
        keys = {}
        for run_id in set(self._logged_ids()) | set(owned):
            try:
                self._check_id(run_id)
                started, _, _ = self._split(self._events(run_id))
            except ValueError:
                continue
            when = owned.get(run_id) or _parse_time((started or {}).get("started_at"))
            keys[run_id] = (when if when is not None else (self._mtime(run_id) or 0.0),
                            self._mtime(run_id) or 0.0, run_id)
        newest = sorted(keys, key=keys.get, reverse=True)
        summaries = []
        for run_id in newest:
            if len(summaries) >= limit:
                break
            try:
                summaries.append(self.status(run_id))
            except (RunNotFound, ValueError) as exc:
                log.warning("skipping run %s in listing: %s", run_id, exc)
        return summaries

    # ----- control -----------------------------------------------------

    def cancel(self, run_id: str, role: str, cancel_token: str | None = None) -> dict:
        """Admin: any run this process owns. Public: a public run, with its token."""
        self._check_id(run_id)
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            current = self.status(run_id)  # raises RunNotFound
            if role != ADMIN and (current["started_by"] != PUBLIC or current["state"] not in TERMINAL):
                # Not ours, so its cancel token cannot be checked here.
                raise RunForbidden("only the admin may cancel this run")
            if current["state"] in TERMINAL:
                return current
            raise RunConflict(f"run {run_id} is not managed by this service process and cannot be cancelled here")
        if role != ADMIN:
            if run.role != PUBLIC:
                raise RunForbidden("only the admin may cancel this run")
            if not (cancel_token and run.cancel_token
                    and hmac.compare_digest(cancel_token.encode(), run.cancel_token.encode())):
                raise RunForbidden("cancelling a public run needs the cancel_token returned when it started")
        with self._lock:
            if run.state in ACTIVE:
                run.stop.set()
        return self.status(run_id)

    def shutdown(self, timeout_s: float = 5.0) -> None:
        """Ask active runs to stop and wait briefly for them."""
        with self._lock:
            threads = [run.thread for run in self._runs.values() if run.state in ACTIVE and run.thread]
            for run in self._runs.values():
                if run.state in ACTIVE:
                    run.stop.set()
        deadline = time.monotonic() + timeout_s
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        # A run still blocked (e.g. inside a model call) dies with the process.
        # Close its log now, with what it is known to have spent, rather than
        # leave it "unknown" until a much later restart.
        with self._lock:
            stuck = [run for run in self._runs.values()
                     if run.state in ACTIVE and run.thread is not None and run.thread.is_alive()]
        for run in stuck:
            try:
                self.runlog.append_event(run.run_id, {
                    "type": "run_finished", "stop_reason": STOPPING, "scored": run.scored,
                    "best_cd": run.best_cd, "usage": vars(run.proposer.client.usage).copy(),
                    "timing": designer.timing_summary(list(run.timings)),
                })
            except (OSError, ValueError) as exc:
                log.warning("could not close run %s on shutdown: %s", run.run_id, exc)
            with self._lock:
                run.state, run.stop_reason = "interrupted", STOPPING


__all__ = ["ACTIVE", "INTERRUPTED", "STOPPING", "RunConflict", "RunError", "RunForbidden", "RunLimitError",
           "RunManager", "RunNotFound", "RunRequestError", "RunUnavailable", "TERMINAL",
           "default_proposer_factory", "state_from_reason"]
