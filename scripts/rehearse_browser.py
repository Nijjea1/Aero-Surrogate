"""Exercise real local services in fresh Playwright contexts; paid calls are opt-in.

Examples (run with the qualified project environment):
    python scripts/rehearse_browser.py --project-root . --attempts 2 --out runs/browser-smoke
    python scripts/rehearse_browser.py --project-root . --attempts 20 --live-runs 4 --max-spend 1 --out runs/browser-rehearsal

Never deploys, changes production logs, exports browser storage, or records secrets.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def request_json(url: str, body=None, headers=None, *, timeout=15):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def isolated_environment(root: Path, output: Path, api_port: int, web_port: int,
                         live_runs: int, max_spend: float) -> dict:
    """Keep the provider credential only; never inherit a production budget or log path."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("AERO_", "NEBIUS_", "NEMOTRON_"))}
    key = os.environ.get("NEBIUS_API_KEY", "")
    if live_runs and not key:
        env_file = root / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "NEBIUS_API_KEY":
                    key = value.strip().strip("\"'")
                    break
    if live_runs and not key:
        raise ValueError("live runs requested, but NEBIUS_API_KEY is unavailable")
    empty_env = output / "empty.env"
    empty_env.write_text("# Isolated rehearsal: no production configuration is loaded.\n", encoding="utf-8")
    environment.update({
        "AERO_ENV_FILE": str(empty_env), "AERO_CHECKPOINT": str(root / "runs/drivaernet-random/best.pt"),
        "AERO_FIELD_CHECKPOINT": str(root / "runs/fields/best.pt") if (root / "runs/fields/best.pt").exists() else "",
        "AERO_RUNLOG_DIR": str(output / "logs"), "AERO_SPEND_LEDGER": str(output / "spend.jsonl"),
        "AERO_SPEND_CAP_USD": str(max_spend), "AERO_PUBLIC_DAILY_BUDGET_USD": str(max_spend),
        "AERO_PUBLIC_RUN_BUDGET_USD": str(min(.10, max_spend)),
        "AERO_PUBLIC_RUNS": "1" if live_runs else "0", "AERO_ADMIN_TOKEN": secrets.token_urlsafe(32),
        "AERO_MAX_CONCURRENT_RUNS": "1", "AERO_PUBLIC_MAX_ITERATIONS": "25",
        "AERO_RATE_LIMIT_PER_MINUTE": "0", "AERO_CORS_ORIGINS": f"http://127.0.0.1:{web_port}",
        "AERO_TRUST_PROXY": "0", "AERO_WEB_API": f"http://127.0.0.1:{api_port}",
        "AERO_WEB_MESH_PER_MINUTE": "0", "PYTHONUNBUFFERED": "1",
    })
    if live_runs:
        environment["NEBIUS_API_KEY"] = key
    environment.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(root / ".audit-track-b/browsers"))
    return environment


def seed_historical_logs(root: Path, output: Path) -> int:
    """Copy read-only historical evidence into isolated service storage."""
    source = root / "runs/matched-20261016/logs"
    destination = output / "logs"
    (destination / "events").mkdir(parents=True)
    count = 0
    for path in source.glob("*.jsonl"):
        shutil.copy2(path, destination / path.name)
        event = source / "events" / path.name
        if event.exists():
            shutil.copy2(event, destination / "events" / path.name)
        count += 1
    if count < 15:
        raise ValueError("the 15 historical matched run logs are required for this rehearsal")
    return count


class LocalServices:
    def __init__(self, root: Path, output: Path, environment: dict, api_port: int, web_port: int):
        self.root, self.output, self.environment = root, output, environment
        self.api, self.web = f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{web_port}"
        self.processes, self.handles = [], []
        self.commands = [
            [sys.executable, "-m", "uvicorn", "src.service.app:app", "--host", "127.0.0.1",
             "--port", str(api_port), "--workers", "1", "--no-proxy-headers"],
            [sys.executable, "web/serve.py", "--host", "127.0.0.1", "--port", str(web_port), "--api", self.api],
        ]

    def __enter__(self):
        try:
            for index, command in enumerate(self.commands):
                handle = (self.output / f"service-{index}.log").open("w", encoding="utf-8")
                self.handles.append(handle)
                self.processes.append(subprocess.Popen(command, cwd=self.root, env=self.environment,
                                                      stdout=handle, stderr=subprocess.STDOUT,
                                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
            for endpoint in [self.api + "/ready", self.web + "/healthz"]:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if any(process.poll() is not None for process in self.processes):
                        raise RuntimeError("local service exited; inspect isolated service logs")
                    try:
                        if request_json(endpoint, timeout=2)[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.2)
                else:
                    raise RuntimeError("local service readiness timed out")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for handle in self.handles:
            handle.close()


def live_indices(attempts: int, live_runs: int) -> set[int]:
    return {index * attempts // live_runs for index in range(live_runs)} if live_runs else set()


def cancellation_check(api: str) -> dict:
    """One public run, immediately cancelled. Tokens remain only in local variables."""
    status, started = request_json(api + "/runs", {"model": "super", "iterations": 25, "body_style": "fastback"})
    if status != 202:
        raise RuntimeError(f"public cancellation setup refused with HTTP {status}")
    run_id, token = started["run_id"], started["cancel_token"]
    wrong, _ = request_json(f"{api}/runs/{run_id}/cancel", {}, {"X-Cancel-Token": "invalid-rehearsal-token"})
    if wrong not in (401, 403):
        raise AssertionError(f"wrong cancellation token returned HTTP {wrong}")
    correct, _ = request_json(f"{api}/runs/{run_id}/cancel", {}, {"X-Cancel-Token": token})
    if correct != 200:
        raise AssertionError(f"correct cancellation token returned HTTP {correct}")
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        _, result = request_json(f"{api}/runs/{run_id}")
        if result["state"] in ("cancelled", "finished", "failed", "interrupted"):
            break
        time.sleep(.25)
    if result["state"] != "cancelled":
        raise AssertionError("public cancellation did not reach cancelled state")
    return {"passed": True, "run_id": run_id, "wrong_token_status": wrong,
            "correct_token_status": correct, "state": result["state"], "cost_usd": result.get("cost_usd")}


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def exercise_attempt(browser, services: LocalServices, output: Path, number: int,
                     engine: str, live: bool, record: bool) -> dict:
    from playwright.sync_api import expect

    started_at, started = datetime.now(timezone.utc).isoformat(), time.monotonic()
    row = {"attempt": number, "engine": engine, "live_requested": live, "passed": False,
           "started_at": started_at, "checks": [], "page_errors": [], "failed_requests": [],
           "http_errors": [], "screenshots": []}
    context_options = {"viewport": {"width": 1600, "height": 1000}, "reduced_motion": "reduce"}
    if record:
        context_options.update(record_video_dir=str(output / "recordings"), record_video_size={"width": 1600, "height": 1000})
    context = browser.new_context(**context_options)
    page = context.new_page()
    page.set_default_timeout(30_000)
    page.on("pageerror", lambda error: row["page_errors"].append(str(error)))
    origins = {services.api, services.web}

    def failed(request):
        if _origin(request.url) in origins:
            row["failed_requests"].append({"path": urlsplit(request.url).path,
                                           "failure": request.failure,
                                           "navigation_abort": "ABORT" in str(request.failure).upper()})

    page.on("requestfailed", failed)
    page.on("response", lambda response: row["http_errors"].append(
        {"path": urlsplit(response.url).path, "status": response.status})
        if _origin(response.url) in origins and response.status >= 400
        and urlsplit(response.url).path != "/favicon.ico" else None)

    def check(name, detail=None):
        row["checks"].append({"check": name, "passed": True, **({"detail": detail} if detail is not None else {})})

    def screenshot(label):
        path = output / f"attempt-{number:02d}-{engine}-{label}.png"
        page.screenshot(path=str(path), full_page=True)
        row["screenshots"].append(path.name)

    try:
        page.goto(services.web + "/index.html", wait_until="networkidle")
        expect(page.locator("#viewer canvas")).to_be_visible(timeout=60_000)
        expect(page.locator("#viewer-message")).to_be_hidden(timeout=60_000)
        expect(page.locator("#verdict")).to_have_attribute("data-state", "trusted")
        expect(page.locator("#cd-value")).to_have_text("0.2246")
        check("real baseline and WebGL viewer", {"cd_display": page.locator("#cd-value").inner_text()})
        slider = page.locator("#p-B_Ramp_Angle")
        slider.evaluate("element => { element.value = '2'; element.dispatchEvent(new Event('input', {bubbles:true})); }")
        expect(page.locator("#cd-value")).not_to_have_text("0.2246")
        expect(page.locator("#verdict")).to_have_attribute("data-state", "trusted")
        check("parameter change updates real prediction")
        page.locator('[data-preset="edge"]').click()
        expect(page.locator("#verdict")).to_have_attribute("data-state", "rejected")
        check("legal joint-extreme corner rejected", page.locator("#verdict-reason").inner_text())
        screenshot("rejected-corner")
        page.locator("#reset-button").click()
        expect(page.locator("#verdict")).to_have_attribute("data-state", "trusted")
        expect(page.locator("#cd-value")).to_have_text("0.2246")
        screenshot("designer")
        page.locator("#runs-link").click()
        expect(page.locator("#run-list .run-item").first).to_be_visible()
        expect(page.locator("#run-list")).to_contain_text("CMA-ES")
        expect(page.locator("#run-list")).to_contain_text("Random")
        expect(page.locator("#logbook tr").first).to_be_visible()
        check("historical matched runs and logbook visible")
        if live:
            page.locator('#start-model button[data-model="super"]').click()
            page.locator("#start-iterations").fill("25")
            with page.expect_response(lambda response: response.url == services.api + "/runs"
                                      and response.request.method == "POST") as response:
                page.locator("#start-button").click()
            result = response.value
            if result.status != 202:
                raise RuntimeError(f"live public start refused with HTTP {result.status}; provider/budget unavailable")
            # Do not retain the returned cancel token in the report.
            run_id = result.json()["run_id"]
            row["run_id"] = run_id
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                _, status = request_json(f"{services.api}/runs/{run_id}")
                if status["state"] in ("finished", "failed", "cancelled", "interrupted"):
                    break
                page.wait_for_timeout(500)
            if status["state"] != "finished" or status.get("scored") != 25:
                raise AssertionError(f"live run incomplete: {status['state']}, scored={status.get('scored')}, reason={status.get('stop_reason')}")
            _, history = request_json(f"{services.api}/history/{run_id}")
            if len(history) != 25:
                raise AssertionError("completed live run history is incomplete")
            page.goto(services.web + f"/runs.html?run={run_id}", wait_until="networkidle")
            expect(page.locator("#logbook tr")).to_have_count(25)
            expect(page.locator("#run-state")).to_contain_text("Finished")
            page.reload(wait_until="networkidle")
            expect(page.locator("#logbook tr")).to_have_count(25)
            row["live"] = {"state": status["state"], "scored": status["scored"], "cost_usd": status.get("cost_usd"),
                           "best_cd": status.get("best_cd"), "timing": status.get("timing"),
                           "provider_model": status.get("model"), "history_rows": len(history)}
            check("real public Super completed and history survives reload")
        screenshot("runs")
        page.locator("#results-link").click()
        expect(page.locator("#headline-line")).to_contain_text("0.875")
        expect(page.locator("#bench-table tbody tr")).to_have_count(3)
        for method in ["Nemotron Super", "CMA-ES", "Random search"]:
            expect(page.locator("#bench-table")).to_contain_text(method)
        expect(page.locator("#headline")).to_contain_text("86.6")
        check("release scalar and all three benchmark methods visible")
        screenshot("results")
        if row["page_errors"]:
            raise AssertionError("browser page errors were recorded")
        failures = [item for item in row["failed_requests"] if not item["navigation_abort"]]
        # Busy optional pressure work is a declared bounded capability; retain its status.
        unexpected = [item for item in row["http_errors"] if not (item["path"] == "/fields" and item["status"] == 503)]
        if failures or unexpected:
            raise AssertionError("unexpected same-origin request failures were recorded")
        row["passed"] = True
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        try:
            screenshot("failure")
        except Exception:
            pass
    finally:
        context.close()
        if record and page.video:
            row["recording"] = str(Path(page.video.path()).relative_to(output))
        row["wall_s"] = round(time.monotonic() - started, 3)
    return row


def write_reports(output: Path, report: dict):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "judge-checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["attempt", "engine", "started_at", "live_requested", "passed", "run_id", "wall_s", "error"])
        writer.writeheader()
        for row in report["attempts"]:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True, help="New isolated output directory")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--live-runs", type=int, default=0, help="Explicitly opt in to this many real 25-evaluation public Super runs")
    parser.add_argument("--max-spend", type=float, default=1.0, help="Isolated application ledger cap, USD")
    parser.add_argument("--engines", nargs="+", choices=("chromium", "firefox"), default=["chromium", "firefox"])
    parser.add_argument("--record", action="store_true", help="Record raw browser WebM; no composition or publishing")
    args = parser.parse_args(argv)
    if not 1 <= args.attempts <= 100 or not 0 <= args.live_runs <= args.attempts:
        parser.error("attempts must be 1-100; live-runs must be between zero and attempts")
    if not math.isfinite(args.max_spend) or not 0 < args.max_spend <= 5:
        parser.error("max-spend must be finite and in (0, 5]")
    root, output = args.project_root.resolve(), args.out.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a new path to preserve previous evidence")
    if not (root / "src/service/app.py").is_file():
        parser.error("project-root must be the Aero Surrogate repository")
    report = {"browser_rehearsal_version": 1, "scope": "real local services, fresh logged-out contexts; not public hosted testing",
              "requested_attempts": args.attempts, "requested_live_runs": args.live_runs,
              "max_spend_usd": args.max_spend, "passed": False, "attempts": [], "errors": []}
    output.mkdir(parents=True)
    try:
        from playwright.sync_api import sync_playwright

        api_port, web_port = free_port(), free_port()
        while web_port == api_port:
            web_port = free_port()
        environment = isolated_environment(root, output, api_port, web_port, args.live_runs, args.max_spend)
        report["seeded_historical_runs"] = seed_historical_logs(root, output)
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = environment["PLAYWRIGHT_BROWSERS_PATH"]
        with LocalServices(root, output, environment, api_port, web_port) as services:
            _, health = request_json(services.api + "/health")
            expected_hash = hashlib.sha256((root / "runs/drivaernet-random/best.pt").read_bytes()).hexdigest()
            if health.get("synthetic") is not False or health.get("provenance", {}).get("checkpoint_sha256") != expected_hash:
                raise AssertionError("service is not serving the selected real scalar checkpoint")
            report["service"] = {key: health.get(key) for key in ["predictor", "synthetic", "schema_version", "model_version", "checkpoint_sha256"]}
            report["service"]["checkpoint_sha256_full"] = expected_hash
            selected = live_indices(args.attempts, args.live_runs)
            with sync_playwright() as playwright:
                browsers = {}
                try:
                    for engine in args.engines:
                        options = {"headless": True}
                        if engine == "chromium":
                            options["args"] = ["--enable-webgl", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
                        else:
                            options["firefox_user_prefs"] = {"webgl.force-enabled": True, "webgl.disabled": False}
                        browsers[engine] = getattr(playwright, engine).launch(**options)
                    for index in range(args.attempts):
                        engine = args.engines[index % len(args.engines)]
                        row = exercise_attempt(browsers[engine], services, output, index + 1, engine,
                                               index in selected, args.record)
                        report["attempts"].append(row)
                        write_reports(output, report)
                        print(f"Attempt {index + 1}/{args.attempts} {engine}: {'passed' if row['passed'] else 'FAILED'}", flush=True)
                        if not row["passed"]:
                            break  # preserve failure evidence; never spend past an unresolved failure
                    if args.live_runs and len(report["attempts"]) == args.attempts and all(row["passed"] for row in report["attempts"]):
                        report["cancel_check"] = cancellation_check(services.api)
                finally:
                    for browser in browsers.values():
                        browser.close()
        report["passed"] = len(report["attempts"]) == args.attempts and all(row["passed"] for row in report["attempts"])
        ledger = output / "spend.jsonl"
        report["spent_usd"] = sum(json.loads(line).get("cost_usd", 0) for line in ledger.read_text().splitlines()) if ledger.exists() else 0
    except Exception as error:
        # Exception text is limited to our validation and browser errors; never include environment/config dumps.
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["passed"] = False
    write_reports(output, report)
    print(f"Browser rehearsal {'passed' if report['passed'] else 'FAILED'}; report: {output / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
