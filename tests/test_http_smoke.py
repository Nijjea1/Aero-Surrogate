"""Exercise uvicorn and the agent CLI as separate processes over real HTTP."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests


def test_service_and_cli_over_http(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, AERO_CHECKPOINT="", AERO_SERVICE_URL=url,
               AERO_RUNLOG_DIR=str(tmp_path / "logs"))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (tmp_path / "server.log").open("w", encoding="utf-8") as output:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.service.app:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                assert server.poll() is None, "uvicorn exited before readiness"
                try:
                    health = requests.get(f"{url}/health", timeout=1)
                    if health.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, "uvicorn readiness timed out"
                time.sleep(0.1)
            assert health.json()["synthetic"] is True
            process = subprocess.run(
                [sys.executable, "-m", "src.agent.designer", "--dry-run",
                 "--iterations", "10", "--run-id", "http-smoke"],
                cwd=root, env=env, capture_output=True, text=True,
                timeout=20, creationflags=flags,
            )
            assert process.returncode == 0, process.stdout + process.stderr
            response = requests.get(f"{url}/history/http-smoke", timeout=3)
            assert response.status_code == 200
            rows = response.json()
            assert len(rows) == 10
            assert rows[0]["accepted"]
            assert not rows[1]["accepted"]
            assert rows[1]["best_cd_so_far"] == rows[0]["best_cd_so_far"]
            assert "gate feedback" in rows[2]["reasoning"]
            assert all(b["best_cd_so_far"] <= a["best_cd_so_far"]
                       for a, b in zip(rows, rows[1:]))
            assert (tmp_path / "logs" / "http-smoke.jsonl").exists()
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
