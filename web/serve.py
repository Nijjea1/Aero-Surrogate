"""The web UI's server.  OWNER: Paramveer.

    python web/serve.py                                   # http://localhost:5173, API on :8000
    python web/serve.py --host 0.0.0.0 --api https://api.aero.example.com \\
        --mesh-per-minute 60 --trust-proxy                # deployed behind Caddy (web/DEPLOY.md)

Serves web/ and the one endpoint the page needs that the service does not have:

    POST /api/mesh  {"params": {...}, "body_style": "fastback"}
        -> src.geometry.viewer.mesh_payload(...)
    GET  /config.js   which API the pages call (--api), so a shared link needs no ?api=
    GET  /healthz     {"ok": true}

Predictions still come from the service. Standard library only, plus the
geometry module (numpy). Every setting can also come from the environment
(AERO_WEB_API, AERO_WEB_MESH_PER_MINUTE, AERO_WEB_MESH_WORKERS,
AERO_WEB_TRUST_PROXY), which is how the container is configured.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import socket
import sys
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent))

from src.geometry.viewer import mesh_payload  # noqa: E402

MAX_BODY = 64 * 1024
MESH_WAIT_S = 10            # how long a request waits for a free mesh worker before a 503


@dataclass(frozen=True)
class Settings:
    api: str | None = None          # the service the pages call; None keeps the pages' own default
    mesh_per_minute: int = 0        # per client; 0 = no limit (local use)
    mesh_workers: int = 2           # meshes built at once; the rest wait, then get a 503
    trust_proxy: bool = False       # client address from the last X-Forwarded-For entry (Caddy sets it)

    @classmethod
    def from_env(cls, env=os.environ) -> "Settings":
        def number(name, default):
            raw = (env.get(name) or "").strip()
            return int(raw) if raw else default
        return cls(
            api=(env.get("AERO_WEB_API") or "").strip() or None,
            mesh_per_minute=number("AERO_WEB_MESH_PER_MINUTE", 0),
            mesh_workers=max(1, number("AERO_WEB_MESH_WORKERS", 2)),
            trust_proxy=(env.get("AERO_WEB_TRUST_PROXY") or "").strip() in {"1", "true", "yes"},
        )

    def check(self) -> "Settings":
        if self.api is not None:
            parts = urlsplit(self.api)
            if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
                raise ValueError(f"--api must be an http(s) origin with an optional path, got {self.api!r}")
        if self.mesh_per_minute < 0 or self.mesh_workers < 1:
            raise ValueError("--mesh-per-minute must be >= 0 and --mesh-workers >= 1")
        return self


@lru_cache(maxsize=64)
def _mesh_json(params: tuple, style: str) -> bytes:
    """The encoded mesh for one design. A slider dragged back and forth, or a
    second car of the same style in the race, is answered from memory."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # unknown names come back in ignored_params
        return json.dumps(mesh_payload(dict(params), style)).encode()


class RateLimiter:
    """At most `per_minute` requests per client in any 60 s window."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def retry_after(self, client: str, now: float | None = None) -> float:
        """0 if the request may go ahead (and it is counted), else seconds to wait."""
        if self.per_minute <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return 60 - (now - q[0])
            q.append(now)
            if len(self.hits) > 10_000:          # forget idle clients
                for key in [k for k, v in self.hits.items() if not v]:
                    del self.hits[key]
            return 0.0


class Handler(SimpleHTTPRequestHandler):
    settings = Settings()
    limiter = RateLimiter(0)
    workers = threading.BoundedSemaphore(2)

    def client(self) -> str:
        if self.settings.trust_proxy:
            forwarded = [p.strip() for p in self.headers.get("X-Forwarded-For", "").split(",") if p.strip()]
            if forwarded:
                return forwarded[-1]
        return self.client_address[0]

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/healthz":
            self._json(200, {"ok": True})
            return
        if path == "/config.js" and self.settings.api:
            body = f"window.AERO_CONFIG = {json.dumps({'api': self.settings.api})};\n".encode()
            self._send(200, body, "text/javascript", cache="no-store")
            return
        super().do_GET()

    def list_directory(self, path):
        # Nothing on the site is browsed as a folder.
        self.send_error(404, "not found")
        return None

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
        wait = self.limiter.retry_after(self.client())
        if wait > 0:
            self._json(429, {"detail": "too many 3D models requested; try again shortly"},
                       headers={"Retry-After": str(max(1, round(wait)))})
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
            # Rounded so values that differ only in float noise share a cache entry.
            key = tuple(sorted((k, round(float(v), 6)) for k, v in params.items()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(422, {"detail": str(exc)})
            return
        if not self.workers.acquire(timeout=MESH_WAIT_S):
            self._json(503, {"detail": "the 3D model builder is busy; try again shortly"}, headers={"Retry-After": "2"})
            return
        try:
            data = _mesh_json(key, str(style))
        except (ValueError, TypeError) as exc:
            self._json(422, {"detail": str(exc)})
            return
        except Exception as exc:  # a broken install must answer, not drop the connection
            self.log_error("mesh build failed: %r", exc)
            self._json(500, {"detail": "the 3D model could not be built"})
            return
        finally:
            self.workers.release()
        self._send(200, data)

    def _json(self, status: int, body: dict, headers: dict | None = None) -> None:
        self._send(status, json.dumps(body).encode(), headers=headers)

    def _send(self, status: int, data: bytes, content_type: str = "application/json",
              cache: str = "no-store", headers: dict | None = None) -> None:
        # The mesh is about 0.6 MB of base64; gzip cuts it to roughly a third.
        if len(data) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
            data = gzip.compress(data, compresslevel=5)
            encoding = "gzip"
        else:
            encoding = None
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # Pages revalidate on every load (Last-Modified answers a 304), so a
        # deploy is seen at once; nothing on the site needs to be framed.
        if self.command == "GET" and not any(h.lower().startswith(b"cache-control") for h in self._headers_buffer):
            self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/") and self.path != "/healthz":
            super().log_message(fmt, *args)


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_servers(host: str, port: int, settings: Settings | None = None) -> list[ThreadingHTTPServer]:
    """This machine only by default: 127.0.0.1 and ::1, so http://localhost
    works whichever one the browser resolves. Pass --host 0.0.0.0 to serve
    the page beyond this machine on purpose."""
    settings = (settings or Settings()).check()

    class Bound(Handler):
        pass

    Bound.settings = settings
    Bound.limiter = RateLimiter(settings.mesh_per_minute)
    Bound.workers = threading.BoundedSemaphore(settings.mesh_workers)

    def handler(*args, **kwargs):
        return Bound(*args, directory=str(WEB), **kwargs)

    if host != "localhost":
        cls = IPv6Server if ":" in host else ThreadingHTTPServer
        return [cls((host, port), handler)]
    servers = [ThreadingHTTPServer(("127.0.0.1", port), handler)]
    with contextlib.suppress(OSError):  # no IPv6 loopback on this machine
        servers.append(IPv6Server(("::1", port), handler))
    return servers


def main(argv=None) -> None:
    env = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 5173))
    parser.add_argument("--host", default="localhost", help="default: this machine only")
    parser.add_argument("--api", default=env.api, help="the service the pages call, e.g. https://api.aero.example.com")
    parser.add_argument("--mesh-per-minute", type=int, default=env.mesh_per_minute,
                        help="3D models per client per minute (0 = no limit)")
    parser.add_argument("--mesh-workers", type=int, default=env.mesh_workers, help="3D models built at once")
    parser.add_argument("--trust-proxy", action="store_true", default=env.trust_proxy,
                        help="read the client from X-Forwarded-For (only behind your own proxy)")
    args = parser.parse_args(argv)
    settings = Settings(args.api, args.mesh_per_minute, args.mesh_workers, args.trust_proxy)
    try:
        servers = make_servers(args.host, args.port, settings)
    except ValueError as exc:
        parser.error(str(exc))
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    where = "localhost" if args.host == "localhost" else args.host
    print(f"Aero Surrogate UI on http://{where}:{args.port}  api={settings.api or 'page default'}  "
          f"mesh_limit={settings.mesh_per_minute or 'off'}/min  workers={settings.mesh_workers}  "
          f"trust_proxy={'on' if settings.trust_proxy else 'off'}  (Ctrl+C to stop)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        servers[0].serve_forever()


if __name__ == "__main__":
    main()
