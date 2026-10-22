"""Check local API/web/model wiring with isolated logs and no provider calls.

Starts loopback-only temporary processes, exercises real HTTP, and shuts them
down even on failure. --checkpoint selects a real scalar artifact; without it
the report explicitly identifies the synthetic fixture. The scripted agent
checks orchestration only and never establishes aerodynamic search performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.contracts import PARAM_BOUNDS, default_params  # noqa: E402


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _request(url: str, payload=None, *, timeout=30):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={} if payload is None else {"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.headers, response.read()


def _json(url: str, payload=None):
    return json.loads(_request(url, payload)[2])


def _ready(process, url: str, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("local process exited before readiness")
        try:
            if _request(url, timeout=1)[0] == 200:
                return
        except (OSError, URLError):
            pass
        time.sleep(0.1)
    raise RuntimeError("local process readiness timed out")


def check(*, checkpoint=None, field_checkpoint=None) -> dict:
    report = {"format": "aero-surrogate-local-check/1", "scope": "local HTTP wiring; no browser or paid provider calls",
              "passed": False, "checks": [], "errors": []}
    processes = []
    with tempfile.TemporaryDirectory(prefix="aero-local-check-") as temporary:
        work = Path(temporary)
        api_port, web_port = _port(), _port()
        while web_port == api_port:
            web_port = _port()
        api, web = f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{web_port}"
        env = dict(os.environ, AERO_CHECKPOINT=str(Path(checkpoint).resolve()) if checkpoint else "",
                   AERO_FIELD_CHECKPOINT=str(Path(field_checkpoint).resolve()) if field_checkpoint else "",
                   AERO_RUNLOG_DIR=str(work / "logs"), AERO_SPEND_LEDGER=str(work / "spend.jsonl"),
                   AERO_SERVICE_URL=api, AERO_CORS_ORIGINS=web, AERO_PUBLIC_RUNS="0",
                   AERO_RATE_LIMIT_PER_MINUTE="0", AERO_ADMIN_TOKEN="", AERO_ENV_FILE=str(work / "no.env"),
                   NEBIUS_API_KEY="", AERO_WEB_API=api, AERO_WEB_TRUST_PROXY="0", AERO_TRUST_PROXY="0")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (work / "processes.log").open("w", encoding="utf8") as output:
            try:
                for command, ready in [
                    ([sys.executable, "-m", "uvicorn", "src.service.app:app", "--host", "127.0.0.1",
                      "--port", str(api_port), "--no-proxy-headers"], api + "/ready"),
                    ([sys.executable, "web/serve.py", "--host", "127.0.0.1", "--port", str(web_port),
                      "--api", api], web + "/healthz"),
                ]:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                               stderr=subprocess.STDOUT, creationflags=flags)
                    processes.append(process)
                    _ready(process, ready)
                health = _json(api + "/health")
                if bool(health.get("synthetic")) != (checkpoint is None):
                    raise RuntimeError("unexpected synthetic/real model identity")
                if checkpoint:
                    expected = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                    if health.get("provenance", {}).get("checkpoint_sha256") != expected:
                        raise RuntimeError("serving checkpoint hash differs from the requested artifact")
                report["service"] = {k: health.get(k) for k in ["predictor", "synthetic", "schema_version",
                                                               "model_version", "checkpoint_sha256"]}
                report["service"]["checkpoint_sha256_full"] = health.get("provenance", {}).get("checkpoint_sha256")
                report["checks"].append("API ready and expected model identity")
                design = {"params": default_params(), "body_style": "fastback"}
                baseline = _json(api + "/evaluate", design)
                if not baseline["in_distribution"]:
                    raise RuntimeError("baseline is not trusted")
                report["baseline"] = baseline
                report["checks"].append("trusted baseline evaluation")
                corner = {"params": {name: hi for name, (_, hi) in PARAM_BOUNDS.items()}, "body_style": "fastback"}
                rejected = _json(api + "/evaluate", corner)
                if rejected["in_distribution"]:
                    raise RuntimeError("joint-extreme corner was unexpectedly trusted")
                report["checks"].append("joint-extreme corner rejected")
                config = _request(web + "/config.js")[2].decode()
                if api not in config:
                    raise RuntimeError("web config does not point at the checked API")
                if _request(web + "/index.html")[0] != 200:
                    raise RuntimeError("designer page is unavailable")
                mesh = _json(web + "/api/mesh", design)
                if not mesh.get("vertices") or not mesh.get("faces"):
                    raise RuntimeError("dynamic viewer mesh is empty")
                report["checks"].append("web designer, API config and dynamic mesh")
                if field_checkpoint:
                    fields = _json(api + "/fields", design)
                    if not fields.get("cp") or fields["provenance"]["checkpoint_sha256"] != hashlib.sha256(Path(field_checkpoint).read_bytes()).hexdigest():
                        raise RuntimeError("field response or checkpoint identity is invalid")
                    report["checks"].append("real field inference (functional only; calibration not established)")
                agent = subprocess.run([sys.executable, "-m", "src.agent.designer", "--dry-run", "--iterations", "5",
                                        "--run-id", "local-check-scripted"], cwd=ROOT, env=env,
                                       capture_output=True, timeout=60, creationflags=flags)
                if agent.returncode:
                    raise RuntimeError("scripted agent HTTP check failed")
                history = _json(api + "/history/local-check-scripted")
                if len(history) != 5 or not any(not r["accepted"] for r in history):
                    raise RuntimeError("scripted agent history or rejection wiring is invalid")
                report["checks"].append("five-step scripted agent over HTTP with persisted history and gate rejection")
                report["passed"] = True
            except (RuntimeError, ValueError, KeyError, OSError, HTTPError, subprocess.SubprocessError) as exc:
                report["errors"].append(f"{type(exc).__name__}: {exc}")
            finally:
                for process in reversed(processes):
                    if os.name == "nt" and process.poll() is None:
                        # The Windows venv launcher can own a child Python process;
                        # terminate the whole tree we created so it cannot retain
                        # log handles or loopback ports after this check exits.
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=flags, timeout=15, check=False)
                    elif process.poll() is None:
                        process.terminate()
                for process in reversed(processes):
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="local trusted scalar checkpoint; otherwise tests the labelled synthetic fixture")
    parser.add_argument("--field-checkpoint", help="optional trusted experimental field checkpoint")
    parser.add_argument("--out", help="new JSON report path; never overwrites existing evidence")
    args = parser.parse_args(argv)
    if args.out and Path(args.out).exists():
        parser.error("output exists; choose a new report path")
    report = check(checkpoint=args.checkpoint, field_checkpoint=args.field_checkpoint)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
