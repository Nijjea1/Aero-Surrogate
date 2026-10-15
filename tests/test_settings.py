"""Service settings and caller identification."""

import pytest
from fastapi.testclient import TestClient

from src.service.app import create_app
from src.service.predictor import StubPredictor
from src.service.security import ADMIN, PUBLIC, AuthError, caller_role
from src.service.settings import Settings

TOKEN = "t" * 32


def test_defaults_need_no_configuration():
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.cors_origins == ("http://localhost:3000", "http://localhost:5173")
    assert settings.admin_token is None and settings.public_runs


def test_environment_overrides_and_trailing_slashes():
    settings = Settings.from_env({
        "AERO_CORS_ORIGINS": "https://demo.example.org/, http://localhost:5173",
        "AERO_ADMIN_TOKEN": TOKEN, "AERO_PUBLIC_RUNS": "0", "AERO_MAX_CONCURRENT_RUNS": "2",
        "AERO_PUBLIC_DAILY_BUDGET_USD": "0.5", "AERO_RUNLOG_DIR": "/data/logs",
    })
    assert settings.cors_origins == ("https://demo.example.org", "http://localhost:5173")
    assert settings.admin_token == TOKEN and not settings.public_runs
    assert settings.max_concurrent_runs == 2 and settings.public_daily_budget_usd == 0.5
    assert settings.runlog_dir == "/data/logs"


@pytest.mark.parametrize("env", [
    {"AERO_CORS_ORIGINS": "*"},
    {"AERO_CORS_ORIGINS": "demo.example.org"},
    {"AERO_ADMIN_TOKEN": "short"},
    {"AERO_PUBLIC_RUNS": "maybe"},
    {"AERO_MAX_CONCURRENT_RUNS": "0"},
    {"AERO_MAX_CONCURRENT_RUNS": "1.5"},
    {"AERO_PUBLIC_RUN_BUDGET_USD": "-1"},
    {"AERO_PUBLIC_DAILY_BUDGET_USD": "nan"},
    {"AERO_MAX_BODY_BYTES": "10"},
])
def test_invalid_settings_fail_loudly(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_caller_role():
    open_ = Settings(admin_token=TOKEN)
    closed = Settings(admin_token=TOKEN, public_runs=False)
    assert caller_role(None, open_) == PUBLIC
    assert caller_role(f"Bearer {TOKEN}", open_) == ADMIN
    assert caller_role(f"bearer   {TOKEN} ", open_) == ADMIN
    for bad in (f"Bearer {TOKEN}x", "Bearer ", f"Basic {TOKEN}", TOKEN):
        with pytest.raises(AuthError):
            caller_role(bad, open_)
    with pytest.raises(AuthError, match="needs an admin token"):
        caller_role(None, closed)
    with pytest.raises(AuthError):  # no admin configured: any token is wrong
        caller_role(f"Bearer {TOKEN}", Settings())


def test_cors_follows_settings():
    settings = Settings(cors_origins=("https://demo.example.org",))
    with TestClient(create_app(StubPredictor(), settings=settings)) as client:
        allowed = client.options("/evaluate", headers={"Origin": "https://demo.example.org",
                                                       "Access-Control-Request-Method": "POST",
                                                       "Access-Control-Request-Headers": "Authorization"})
        assert allowed.headers["access-control-allow-origin"] == "https://demo.example.org"
        refused = client.options("/evaluate", headers={"Origin": "http://localhost:5173",
                                                       "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in refused.headers
