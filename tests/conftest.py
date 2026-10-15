"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def isolated_run_logs(tmp_path_factory, monkeypatch):
    """Point every test's default run-log directory at a private temp folder.

    create_app() starts a RunManager on the configured log directory, and on
    startup it closes stale unfinished runs there. Without this, a test that
    builds an app without its own RunLog would do that to the developer's
    real runs/logs. Tests that need a specific directory still set their own.
    """
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path_factory.mktemp("runlogs")))
