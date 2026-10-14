"""Local dev server for the web UI.  OWNER: Paramveer.

    python web/serve.py            # http://localhost:5173

Serves web/ and one endpoint the page needs that the service does not have:

    POST /api/mesh  {"params": {...}, "body_style": "fastback"}
        -> src.geometry.viewer.mesh_payload(...)

Predictions still come from the service on :8000. Standard library only, plus
the geometry module.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import warnings
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent))

from src.geometry.viewer import mesh_payload  # noqa: E402

MAX_BODY = 64 * 1024


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.split("?")[0] != "/api/mesh":
            self._json(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                raise ValueError("request too large")
            request = json.loads(self.rfile.read(length) or b"{}")
            params = request.get("params", {})
            if not isinstance(params, dict) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in params.values()
            ):
                raise ValueError("params must be an object of numbers")
            style = request.get("body_style") or "fastback"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # unknown names come back in ignored_params
                payload = mesh_payload(params, style)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(422, {"detail": str(exc)})
            return
        self._json(200, payload)

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # The page changes often during development; never serve a stale copy.
        if self.command == "GET":
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/"):
            super().log_message(fmt, *args)


class DualStackServer(ThreadingHTTPServer):
    """Answer on both ::1 and 127.0.0.1, so http://localhost works either way."""

    address_family = socket.AF_INET6

    def server_bind(self):
        with contextlib.suppress(Exception):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        return super().server_bind()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()
    server = DualStackServer(("::", args.port), partial(Handler, directory=str(WEB)))
    print(f"Aero Surrogate UI on http://localhost:{args.port}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
