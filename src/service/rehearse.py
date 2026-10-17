"""Live reliability rehearsal: real agent runs, start to finish, over HTTP.

Unlike `qualify`, this spends Token Factory credit. It checks the service is
ready and serving a real model, then starts admin runs one at a time (rotating
body styles), polls each to its end state, checks its history, and records
outcome, cost and step time. It also starts one run and cancels it with its
cancel token, the path the Runs page uses.

The admin token is read from AERO_ADMIN_TOKEN, never from the command line, and
never written to the report. New runs stop starting once --max-spend is reached.

    AERO_ADMIN_TOKEN=... python -m src.service.rehearse --service-url https://api.example.com \\
        --runs 20 --model super --out runs/reliability.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

from ..contracts import BODY_STYLES
from .qualify import _Remote

ACTIVE = ("queued", "running")


class _Http(_Remote):
    """The qualification client, plus request headers for the admin token."""

    def get(self, path, **kwargs):
        return self.session.get(self.url + path, timeout=(3.05, 30), allow_redirects=False, **kwargs)


def _wait(client, run_id, *, poll_s, timeout_s, sleep, clock):
    deadline = clock() + timeout_s
    while True:
        status = client.get(f"/runs/{run_id}").json()
        if status["state"] not in ACTIVE or clock() >= deadline:
            return status
        sleep(poll_s)


def _start(client, body, headers):
    response = client.post("/runs", json=body, headers=headers)
    if response.status_code != 202:
        raise RuntimeError(f"POST /runs returned {response.status_code}: {response.text[:200]}")
    return response.json()


def _cancel_check(client, admin, model, *, poll_s, timeout_s, sleep, clock):
    """Start a run, then cancel it with the token it returned (public) or as admin."""
    body = {"model": model, "iterations": 10, "body_style": BODY_STYLES[0]}
    response = client.post("/runs", json=body, headers={})
    public = response.status_code == 202
    started = response.json() if public else _start(client, body, admin)
    headers = {"X-Cancel-Token": started["cancel_token"]} if public else admin
    cancelled = client.post(f"/runs/{started['run_id']}/cancel", headers=headers)
    status = _wait(client, started["run_id"], poll_s=poll_s, timeout_s=timeout_s, sleep=sleep, clock=clock)
    return {"run_id": started["run_id"], "started_as": "public" if public else "admin",
            "cancel_status": cancelled.status_code, "state": status["state"],
            "cost_usd": status.get("cost_usd") or 0.0,
            # A run can legitimately finish before the cancel lands.
            "passed": cancelled.status_code in (200, 202) and status["state"] in ("cancelled", "finished")}


def rehearse(client, admin_token: str, *, runs: int, model: str = "super", iterations: int = 25,
             styles=BODY_STYLES, max_spend_usd: float = 0.75, check_cancel: bool = True,
             allow_synthetic: bool = False, poll_s: float = 2.0, run_timeout_s: float = 600.0,
             sleep=time.sleep, clock=time.monotonic) -> dict:
    admin = {"Authorization": f"Bearer {admin_token}"}
    report = {"rehearsal_version": 1, "model": model, "iterations": iterations, "requested_runs": runs,
              "max_spend_usd": max_spend_usd, "passed": False, "errors": [], "runs": [], "cancel_check": None}
    ready = client.get("/ready")
    health = client.get("/health").json()
    report["service"] = {key: health.get(key) for key in
                         ("predictor", "synthetic", "schema_version", "model_version", "checkpoint_sha256")}
    if ready.status_code != 200:
        report["errors"].append(f"/ready returned {ready.status_code}")
        return report
    if health.get("synthetic") and not allow_synthetic:
        report["errors"].append("service is serving the synthetic stub; refusing to spend on it")
        return report
    spent = 0.0
    wait = {"poll_s": poll_s, "timeout_s": run_timeout_s, "sleep": sleep, "clock": clock}
    if check_cancel:
        try:
            report["cancel_check"] = _cancel_check(client, admin, model, **wait)
            spent += report["cancel_check"]["cost_usd"]
        except (RuntimeError, KeyError, ValueError) as exc:
            report["cancel_check"] = {"passed": False, "error": str(exc)}
    for index in range(runs):
        if spent >= max_spend_usd:
            report["errors"].append(f"stopped before run {index + 1}: spend ${spent:.4f} reached the cap")
            break
        style = styles[index % len(styles)]
        body = {"model": model, "iterations": iterations, "body_style": style}
        started_at = clock()
        try:
            started = _start(client, body, admin)
        except (RuntimeError, ValueError) as exc:
            report["runs"].append({"index": index + 1, "body_style": style, "passed": False, "error": str(exc)})
            continue
        status = _wait(client, started["run_id"], **wait)
        if status["state"] in ACTIVE:  # timed out: stop it rather than leave it spending
            client.post(f"/runs/{started['run_id']}/cancel", headers=admin)
        history = client.get(f"/history/{started['run_id']}")
        rows = history.json() if history.status_code == 200 else []
        baseline = rows[0]["cd"] if rows and rows[0].get("accepted") else None
        best = status.get("best_cd")
        timing = status.get("timing") or {}
        cost = status.get("cost_usd") or 0.0
        spent += cost
        complete = (status["state"] == "finished" and status.get("stop_reason") == "evaluation budget used"
                    and len(rows) == iterations)
        report["runs"].append({
            "index": index + 1, "run_id": started["run_id"], "body_style": style, "state": status["state"],
            "stop_reason": status.get("stop_reason"), "scored": status.get("scored"), "history_rows": len(rows),
            "baseline_cd": baseline, "best_cd": best,
            "change_fraction": (best - baseline) / baseline if baseline and best is not None else None,
            "rejections": sum(1 for row in rows if not row.get("accepted")),
            "cost_usd": cost, "step_s_median": timing.get("step_s_median"),
            "wall_s": clock() - started_at, "error": status.get("error"), "passed": complete})
    finished = [run for run in report["runs"] if run["passed"]]
    changes = [run["change_fraction"] for run in finished if run["change_fraction"] is not None]
    steps = [run["step_s_median"] for run in finished if run["step_s_median"] is not None]
    report["summary"] = {
        "runs": len(report["runs"]), "complete": len(finished),
        "success_rate": len(finished) / len(report["runs"]) if report["runs"] else None,
        "median_change_fraction": statistics.median(changes) if changes else None,
        "worst_change_fraction": max(changes) if changes else None,
        "median_step_s": statistics.median(steps) if steps else None,
        "spent_usd": spent}
    if len(report["runs"]) < runs and not report["errors"]:
        report["errors"].append("fewer runs than requested")
    if len(finished) != len(report["runs"]):
        report["errors"].append(f"{len(report['runs']) - len(finished)} run(s) did not complete")
    if check_cancel and not report["cancel_check"]["passed"]:
        report["errors"].append("cancel check failed")
    report["passed"] = not report["errors"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--model", choices=("nano", "super"), default="super")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-spend", type=float, default=0.75, help="Stop starting runs after this many USD")
    parser.add_argument("--no-cancel-check", action="store_true")
    parser.add_argument("--out", required=True, help="New report file; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    token = (os.environ.get("AERO_ADMIN_TOKEN") or "").strip()
    if not token:
        parser.error("set AERO_ADMIN_TOKEN in the environment (never on the command line)")
    if not 1 <= args.runs <= 100 or not math.isfinite(args.max_spend) or not 0 < args.max_spend <= 5:
        parser.error("--runs must be 1-100 and --max-spend in (0, 5]")
    out = Path(args.out)
    if out.exists():
        parser.error("output exists; choose a fresh path")
    client = _Http(args.service_url)
    try:
        report = rehearse(client, token, runs=args.runs, model=args.model, iterations=args.iterations,
                          max_spend_usd=args.max_spend, check_cancel=not args.no_cancel_check)
    finally:
        client.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = report.get("summary", {})
    print(f"Rehearsal {'passed' if report['passed'] else 'FAILED'}: {summary.get('complete')} of "
          f"{summary.get('runs')} runs complete, spent ${summary.get('spent_usd', 0):.4f}. Report: {out}")
    for error in report["errors"]:
        print(f"  - {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
