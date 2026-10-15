"""Hosting guards: rate limit, body size, startup checks, /ready. No network."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from src.contracts import default_params
from src.service.app import create_app
from src.service.limits import BodySizeMiddleware, RateLimiter, is_rate_limited, trust_proxy_from_env
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog
from src.service.settings import Settings

TOKEN = "s3cret-admin-token-value"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_client(tmp_path, monkeypatch=None, **overrides):
    settings = Settings(runlog_dir=str(tmp_path / "logs"), **overrides)
    app = create_app(StubPredictor(), settings=settings)
    clock = Clock()
    app.state.rate_limiter.clock = clock
    return TestClient(app), clock


def body():
    return {"params": default_params()}


# ---- rate limiting ---------------------------------------------------------

def test_rate_limit_allows_n_then_429_with_retry_after(tmp_path):
    client, clock = make_client(tmp_path, rate_limit_per_minute=3)
    with client:
        assert [client.post("/evaluate", json=body()).status_code for _ in range(3)] == [200] * 3
        clock.now += 15
        blocked = client.post("/compare", json={"design_a": body(), "design_b": body()})
        assert blocked.status_code == 429
        assert blocked.json()["detail"].startswith("rate limit of 3")
        assert blocked.headers["retry-after"] == "45"


def test_rate_limit_window_resets(tmp_path):
    client, clock = make_client(tmp_path, rate_limit_per_minute=2)
    with client:
        for _ in range(2):
            assert client.post("/evaluate", json=body()).status_code == 200
        assert client.post("/evaluate", json=body()).status_code == 429
        clock.now += 60
        assert client.post("/evaluate", json=body()).status_code == 200


def test_rate_limit_separate_ips_independent():
    clock = Clock()
    limiter = RateLimiter(1, clock=clock)
    assert limiter.hit("1.1.1.1") is None
    assert limiter.hit("1.1.1.1") == pytest.approx(60.0)
    assert limiter.hit("2.2.2.2") is None


def test_rate_limit_memory_is_bounded():
    clock = Clock()
    limiter = RateLimiter(5, clock=clock, max_keys=3)
    for n in range(10):
        limiter.hit(f"10.0.0.{n}")
    assert len(limiter) == 3
    clock.now += 61
    limiter.hit("fresh")
    limiter.hit("fresh2")
    assert len(limiter) <= 3


def test_gets_are_not_limited(tmp_path):
    client, _ = make_client(tmp_path, rate_limit_per_minute=1)
    with client:
        assert client.post("/evaluate", json=body()).status_code == 200
        assert client.post("/evaluate", json=body()).status_code == 429
        for path in ("/health", "/schema", "/ready"):
            assert client.get(path).status_code == 200


def test_zero_disables_rate_limit(tmp_path):
    client, _ = make_client(tmp_path, rate_limit_per_minute=0)
    with client:
        assert all(client.post("/evaluate", json=body()).status_code == 200 for _ in range(30))


def test_runs_paths_are_limited_once_they_exist():
    assert is_rate_limited("POST", "/runs")
    assert is_rate_limited("POST", "/runs/abc/cancel")
    assert is_rate_limited("POST", "/evaluate/")
    assert not is_rate_limited("GET", "/runs")
    assert not is_rate_limited("POST", "/runsx")
    assert not is_rate_limited("GET", "/history/x")


@pytest.mark.parametrize("trust, expect_second", [("0", 429), ("1", 200)])
def test_forwarded_for_only_with_trusted_proxy(tmp_path, monkeypatch, trust, expect_second):
    monkeypatch.setenv("AERO_TRUST_PROXY", trust)
    client, _ = make_client(tmp_path, rate_limit_per_minute=1)
    with client:
        first = client.post("/evaluate", json=body(), headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"})
        second = client.post("/evaluate", json=body(), headers={"X-Forwarded-For": "9.9.9.9, 2.2.2.2"})
        assert first.status_code == 200
        assert second.status_code == expect_second
        # Spoofing the first (client-supplied) entry never earns a fresh window.
        spoof = client.post("/evaluate", json=body(), headers={"X-Forwarded-For": "7.7.7.7, 1.1.1.1"})
        assert spoof.status_code == 429


def test_trust_proxy_flag_parsing():
    assert trust_proxy_from_env({}) is False
    assert trust_proxy_from_env({"AERO_TRUST_PROXY": "true"}) is True
    with pytest.raises(ValueError, match="AERO_TRUST_PROXY"):
        trust_proxy_from_env({"AERO_TRUST_PROXY": "maybe"})


def test_429_keeps_cors_headers(tmp_path):
    client, _ = make_client(tmp_path, rate_limit_per_minute=1)
    origin = {"Origin": "http://localhost:5173"}
    with client:
        client.post("/evaluate", json=body(), headers=origin)
        blocked = client.post("/evaluate", json=body(), headers=origin)
        assert blocked.status_code == 429
        assert blocked.headers["access-control-allow-origin"] == "http://localhost:5173"


# ---- body size -------------------------------------------------------------

def test_oversized_content_length_is_413(tmp_path):
    client, _ = make_client(tmp_path, max_body_bytes=1024)
    with client:
        payload = json.dumps({"params": default_params(), "design_id": "x" * 2000})
        response = client.post("/evaluate", content=payload,
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert "1024" in response.json()["detail"]
        assert client.post("/evaluate", json=body()).status_code == 200


def test_oversized_chunked_body_is_413(tmp_path):
    client, _ = make_client(tmp_path, max_body_bytes=1024)

    def chunks():
        for _ in range(5):
            yield b" " * 400

    with client:
        response = client.post("/evaluate", content=chunks(),
                               headers={"Content-Type": "application/json"})
        assert "content-length" not in {k.lower() for k in response.request.headers}
        assert response.status_code == 413


def test_small_chunked_body_is_replayed_intact(tmp_path):
    client, _ = make_client(tmp_path, max_body_bytes=4096)
    raw = json.dumps(body()).encode()

    def chunks():
        for start in range(0, len(raw), 50):
            yield raw[start:start + 50]

    with client:
        response = client.post("/evaluate", content=chunks(),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 200


def test_body_middleware_counts_streamed_bytes_without_calling_app():
    called = []

    async def app(scope, receive, send):
        called.append(True)

    sent = []
    messages = [{"type": "http.request", "body": b"a" * 600, "more_body": True},
                {"type": "http.request", "body": b"a" * 600, "more_body": False}]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/evaluate", "headers": []}
    asyncio.run(BodySizeMiddleware(app, 1024)(scope, receive, send))
    assert not called
    assert sent[0]["status"] == 413


def test_invalid_content_length_is_400():
    sent = []

    async def app(scope, receive, send):
        raise AssertionError("app must not run")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/evaluate",
             "headers": [(b"content-length", b"abc")]}
    asyncio.run(BodySizeMiddleware(app, 1024)(scope, receive, send))
    assert sent[0]["status"] == 400


# ---- startup and readiness -------------------------------------------------

def test_unwritable_log_dir_fails_startup(tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    app = create_app(StubPredictor(), settings=Settings(runlog_dir=str(blocker / "logs")))
    with pytest.raises(RuntimeError, match="AERO_RUNLOG_DIR.*not writable"):
        with TestClient(app):
            pass


def test_writable_log_dir_is_created(tmp_path):
    target = tmp_path / "deep" / "logs"
    with TestClient(create_app(StubPredictor(), settings=Settings(runlog_dir=str(target)))):
        assert target.is_dir()
        assert list(target.iterdir()) == []  # the probe file is removed


def test_startup_summary_never_shows_the_token(tmp_path, capsys):
    settings = Settings(runlog_dir=str(tmp_path / "logs"), admin_token=TOKEN, public_runs=False,
                        rate_limit_per_minute=7)
    with TestClient(create_app(StubPredictor(), settings=settings)):
        pass
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("aero-surrogate ready:"))
    assert TOKEN not in out
    for part in ("predictor=stub", "schema=2.0.0", "admin_token=set", "public_runs=off",
                 "rate_limit=7/min", "http://localhost:5173", str((tmp_path / "logs").resolve())):
        assert part in line


def test_injected_runlog_root_is_checked(tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")
    app = create_app(StubPredictor(), runlog=RunLog(str(blocker / "logs")),
                     settings=Settings(runlog_dir=str(tmp_path / "ok")))
    with pytest.raises(RuntimeError, match="not writable"):
        with TestClient(app):
            pass


def test_ready_200_then_503_when_log_dir_unwritable(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/ready").json() == {"ready": True}
        logs = tmp_path / "logs"
        logs.rmdir()
        logs.write_text("now a file")
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["problems"] == ["run log directory not writable"]
        assert client.get("/health").status_code == 200


def test_ready_503_without_predictor(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        client.app.state.predictor = None
        response = client.get("/ready")
        assert response.status_code == 503
        assert "predictor not loaded" in response.json()["problems"]


def test_cors_unchanged(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        allowed = client.options("/evaluate", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
        denied = client.options("/evaluate", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in denied.headers
