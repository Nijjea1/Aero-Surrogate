"""web/serve.py: the page's dev server and its /api/mesh endpoint."""

from __future__ import annotations

import http.client
import importlib.util
import json
import threading
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("serve", Path(__file__).resolve().parents[1] / "web" / "serve.py")
serve = importlib.util.module_from_spec(_spec)
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
