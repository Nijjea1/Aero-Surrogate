"""Hosting guards: rate limiting, body-size limits, startup checks.  OWNER: Pranoy.

Two pure ASGI middlewares, installed by create_app:

    RateLimitMiddleware   POST /evaluate, /compare and /runs* per client IP,
                          AERO_RATE_LIMIT_PER_MINUTE (0 = off). Fixed one-minute
                          windows, in memory, so one process only (the service
                          runs with exactly one worker anyway).
    BodySizeMiddleware    rejects bodies over AERO_MAX_BODY_BYTES with 413,
                          from Content-Length when sent, else by counting the
                          streamed bytes before the app sees any of them.

The client IP is the TCP peer. X-Forwarded-For is honoured only when
AERO_TRUST_PROXY=1, and then only its last entry: the one written by the
reverse proxy directly in front of us. Earlier entries are client-supplied
and spoofable. Set it only when the app port is reachable solely through
that proxy. (Read here because Settings does not have it yet.)
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Callable

from starlette.responses import JSONResponse

RATE_LIMITED_PATHS = ("/evaluate", "/compare")
RATE_LIMITED_PREFIXES = ("/runs",)


def is_rate_limited(method: str, path: str) -> bool:
    if method != "POST":
        return False
    path = path.rstrip("/") or "/"
    return path in RATE_LIMITED_PATHS or any(
        path == prefix or path.startswith(prefix + "/") for prefix in RATE_LIMITED_PREFIXES)


class RateLimiter:
    """Fixed-window counter per key, thread-safe, holding at most max_keys keys.

    When full, expired windows are dropped first, then the least recently
    seen keys, so a flood of distinct IPs cannot grow memory without bound.
    """

    def __init__(self, per_window: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000):
        self.per_window = int(per_window)
        self.window_s = float(window_s)
        self.clock = clock
        self.max_keys = int(max_keys)
        self._windows: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.per_window > 0

    def __len__(self) -> int:
        return len(self._windows)

    def hit(self, key: str) -> float | None:
        """Count one request; None if allowed, else seconds until the window resets."""
        if not self.enabled:
            return None
        now = self.clock()
        with self._lock:
            start, count = self._windows.get(key, (now, 0))
            if now - start >= self.window_s:
                start, count = now, 0
            if count >= self.per_window:
                self._windows.move_to_end(key)
                return max(start + self.window_s - now, 0.0)
            self._windows[key] = (start, count + 1)
            self._windows.move_to_end(key)
            if len(self._windows) > self.max_keys:
                self._evict(now)
            return None

    def _evict(self, now: float) -> None:
        expired = [k for k, (start, _) in self._windows.items() if now - start >= self.window_s]
        for key in expired:
            del self._windows[key]
        while len(self._windows) > self.max_keys:
            self._windows.popitem(last=False)


def client_ip(scope, trust_proxy: bool) -> str:
    if trust_proxy:
        for name, value in scope.get("headers") or ():
            if name == b"x-forwarded-for":
                last = value.decode("latin-1").split(",")[-1].strip()
                if last:
                    return last
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimitMiddleware:
    def __init__(self, app, limiter: RateLimiter, trust_proxy: bool = False):
        self.app = app
        self.limiter = limiter
        self.trust_proxy = trust_proxy

    async def __call__(self, scope, receive, send):
        if (scope["type"] == "http" and self.limiter.enabled
                and is_rate_limited(scope["method"], scope["path"])):
            retry = self.limiter.hit(client_ip(scope, self.trust_proxy))
            if retry is not None:
                seconds = max(1, math.ceil(retry))
                response = JSONResponse(
                    status_code=429, headers={"Retry-After": str(seconds)},
                    content={"detail": f"rate limit of {self.limiter.per_window} scoring "
                                       f"requests per minute exceeded; retry in {seconds} s"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class BodySizeMiddleware:
    """Reject request bodies larger than max_bytes with 413.

    The body is read (up to the limit) before the app runs and then replayed,
    so the limit holds with or without Content-Length and no partial body is
    ever parsed. Bodies are small JSON here, so buffering is cheap.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    def _too_large(self):
        return JSONResponse(status_code=413, headers={"Connection": "close"},
                            content={"detail": f"request body exceeds {self.max_bytes} bytes"})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    await JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})(
                        scope, receive, send)
                    return
                if declared > self.max_bytes:
                    await self._too_large()(scope, receive, send)
                    return
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # Client disconnected before finishing the body.
                await self.app(scope, _replay([message], receive), send)
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await self._too_large()(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        replay = [{"type": "http.request", "body": b"".join(chunks), "more_body": False}]
        await self.app(scope, _replay(replay, receive), send)


def _replay(messages: list, receive):
    pending = list(messages)

    async def wrapped():
        if pending:
            return pending.pop(0)
        return await receive()
    return wrapped


def check_writable_dir(path: str | os.PathLike) -> Path:
    """Create path if needed and prove a file can be written there; raise RuntimeError if not."""
    root = Path(path).resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix=".write-check-", delete=True) as probe:
            probe.write(b"ok")
            probe.flush()
    except OSError as exc:
        raise RuntimeError(
            f"run log directory {str(root)!r} (AERO_RUNLOG_DIR) is not writable: "
            f"{exc.strerror or exc}") from None
    return root


def is_writable_dir(path: str | os.PathLike) -> bool:
    try:
        check_writable_dir(path)
        return True
    except RuntimeError:
        return False


def startup_summary(settings, predictor, schema_version: str, log_dir, trust_proxy: bool) -> str:
    """One line describing the running configuration. Never includes secret values."""
    rate = (f"{settings.rate_limit_per_minute}/min" if settings.rate_limit_per_minute > 0
            else "off")
    return ("aero-surrogate ready: "
            f"predictor={getattr(predictor, 'name', type(predictor).__name__)} "
            f"schema={schema_version} "
            f"cors={','.join(settings.cors_origins) or '-'} "
            f"admin_token={'set' if settings.admin_token else 'unset'} "
            f"public_runs={'on' if settings.public_runs else 'off'} "
            f"rate_limit={rate} trust_proxy={'on' if trust_proxy else 'off'} "
            f"max_body={settings.max_body_bytes}B "
            f"log_dir={log_dir}")
