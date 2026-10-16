"""web/serve.py: the page's dev server and its /api/mesh endpoint."""

from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("serve", Path(__file__).resolve().parents[1] / "web" / "serve.py")
serve = importlib.util.module_from_spec(_spec)
sys.modules["serve"] = serve   # dataclasses resolve annotations through the module
_spec.loader.exec_module(serve)


@pytest.fixture(scope="module")
def server():
    servers = serve.make_servers("localhost", 0)
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield servers
    for s in servers:
        s.shutdown()
        s.server_close()


def _post(server, body: bytes, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    conn.request("POST", "/api/mesh", body=body, headers=headers or {"Content-Type": "application/json"})
    response = conn.getresponse()
    return response.status, json.loads(response.read() or b"{}")


def test_listens_on_this_machine_only(server):
    hosts = {s.server_address[0] for s in server}
    assert hosts <= {"127.0.0.1", "::1"}
    assert "127.0.0.1" in hosts


def test_mesh_endpoint(server):
    status, payload = _post(server[0], json.dumps({"params": {}, "body_style": "fastback"}).encode())
    assert status == 200
    assert payload["n_faces"] > 0 and payload["body_style"] == "fastback"


@pytest.mark.parametrize("body, headers", [
    (b"[1, 2]", None),                                              # not an object
    (json.dumps({"params": {"A_Car_Width": "wide"}}).encode(), None),  # not a number
    (json.dumps({"body_style": "pickup"}).encode(), None),          # unknown style
    (b"", {"Content-Length": "-5"}),                                # negative length
    (b"", {"Content-Length": str(10**9)}),                           # too large
    (b"", {"Content-Length": "lots"}),                               # not a number
])
def test_bad_requests_are_refused(server, body, headers):
    status, payload = _post(server[0], body, headers)
    assert status == 422
    assert payload["detail"]


def test_unknown_path_is_404(server):
    conn = http.client.HTTPConnection("127.0.0.1", server[0].server_address[1], timeout=10)
    conn.request("POST", "/api/other", body=b"{}")
    assert conn.getresponse().status == 404


def _get(server, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request("GET", path, headers=headers or {})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


@pytest.fixture()
def deployed():
    settings = serve.Settings(api="https://api.aero.example.com", mesh_per_minute=2, mesh_workers=1, trust_proxy=True)
    servers = serve.make_servers("localhost", 0, settings)
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield servers[0]
    for s in servers:
        s.shutdown()
        s.server_close()


def test_health_and_no_folder_listings(server):
    status, _, body = _get(server[0], "/healthz")
    assert status == 200 and json.loads(body) == {"ok": True}
    assert _get(server[0], "/results/")[0] == 404


def test_pages_get_safety_headers_and_revalidate(server):
    status, headers, _ = _get(server[0], "/index.html")
    assert status == 200
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-cache"


def test_config_names_the_deployed_api(server, deployed):
    # Without --api the static fallback leaves the pages' own default alone.
    status, _, body = _get(server[0], "/config.js")
    assert status == 200 and b"window.AERO_CONFIG = window.AERO_CONFIG || {}" in body
    status, headers, body = _get(deployed, "/config.js")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert body == b'window.AERO_CONFIG = {"api": "https://api.aero.example.com"};\n'


def test_mesh_rate_limit_is_per_forwarded_client(deployed):
    body = json.dumps({"params": {}, "body_style": "fastback"}).encode()
    alice = {"Content-Type": "application/json", "X-Forwarded-For": "203.0.113.9, 198.51.100.1"}
    bob = {"Content-Type": "application/json", "X-Forwarded-For": "198.51.100.2"}
    assert [_post(deployed, body, alice)[0] for _ in range(2)] == [200, 200]
    conn = http.client.HTTPConnection("127.0.0.1", deployed.server_address[1], timeout=30)
    conn.request("POST", "/api/mesh", body=body, headers=alice)
    refused = conn.getresponse()
    assert refused.status == 429 and int(refused.getheader("Retry-After")) >= 1
    assert _post(deployed, body, bob)[0] == 200   # the last X-Forwarded-For entry is the client


def test_rate_limiter_window():
    limiter = serve.RateLimiter(2)
    assert limiter.retry_after("a", now=0) == 0 and limiter.retry_after("a", now=1) == 0
    assert limiter.retry_after("a", now=2) == pytest.approx(58)
    assert limiter.retry_after("a", now=61) == 0
    assert serve.RateLimiter(0).retry_after("a") == 0


def test_settings_from_the_environment_and_checks():
    s = serve.Settings.from_env({"AERO_WEB_API": " https://api.example.org ", "AERO_WEB_MESH_PER_MINUTE": "30",
                                 "AERO_WEB_MESH_WORKERS": "3", "AERO_WEB_TRUST_PROXY": "1"})
    assert s == serve.Settings("https://api.example.org", 30, 3, True)
    assert serve.Settings.from_env({}) == serve.Settings()
    for bad in ("ftp://x", "api.example.org", "https://x/?a=1"):
        with pytest.raises(ValueError):
            serve.Settings(api=bad).check()
