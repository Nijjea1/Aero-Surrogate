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
import threading
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
        # Read the body before any reply: closing a socket with unread request
        # bytes makes Windows reset the connection, and the client never sees
        # the response.
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True
            self._json(422, {"detail": "bad request length"})
            return
        raw = self.rfile.read(length)
        if self.path.split("?")[0] != "/api/mesh":
            self._json(404, {"detail": "not found"})
            return
        try:
            request = json.loads(raw or b"{}")
            if not isinstance(request, dict):
                raise ValueError("body must be a JSON object")
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


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_servers(host: str, port: int) -> list[ThreadingHTTPServer]:
    """This machine only by default: 127.0.0.1 and ::1, so http://localhost
    works whichever one the browser resolves. Pass --host 0.0.0.0 to share
    the page on the network on purpose."""
    handler = partial(Handler, directory=str(WEB))
    if host != "localhost":
        cls = IPv6Server if ":" in host else ThreadingHTTPServer
        return [cls((host, port), handler)]
    servers = [ThreadingHTTPServer(("127.0.0.1", port), handler)]
    with contextlib.suppress(OSError):  # no IPv6 loopback on this machine
        servers.append(IPv6Server(("::1", port), handler))
    return servers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--host", default="localhost", help="default: this machine only")
    args = parser.parse_args()
    servers = make_servers(args.host, args.port)
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    print(f"Aero Surrogate UI on http://localhost:{args.port}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        servers[0].serve_forever()


if __name__ == "__main__":
    main()
