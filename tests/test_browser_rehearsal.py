"""Browser rehearsal must never borrow production budgets or spend by default."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location("rehearse_browser", Path(__file__).resolve().parents[1] / "scripts/rehearse_browser.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_default_rehearsal_strips_provider_key_and_production_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "private-test-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://unrelated-provider.invalid")
    monkeypatch.setenv("AERO_RUNLOG_DIR", "production-logs")
    monkeypatch.setenv("AERO_SPEND_LEDGER", "production-ledger")
    monkeypatch.setenv("AERO_SPEND_CAP_USD", "1000")
    output = tmp_path / "output"
    output.mkdir()
    env = module().isolated_environment(tmp_path, output, 8000, 5173, 0, 1.0)
    assert "NEBIUS_API_KEY" not in env
    assert "NEBIUS_BASE_URL" not in env
    assert env["AERO_PUBLIC_RUNS"] == "0"
    assert env["AERO_RUNLOG_DIR"] == str(output / "logs")
    assert env["AERO_SPEND_LEDGER"] == str(output / "spend.jsonl")
    assert env["AERO_SPEND_CAP_USD"] == "1.0"
    assert "private-test-key" not in (output / "empty.env").read_text()


def test_live_opt_in_loads_only_key_from_local_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    (tmp_path / ".env").write_text("NEBIUS_API_KEY=private-test-key\nAERO_SPEND_CAP_USD=1000\nAERO_RUNLOG_DIR=production\n")
    output = tmp_path / "output"
    output.mkdir()
    env = module().isolated_environment(tmp_path, output, 8000, 5173, 4, .75)
    assert env["NEBIUS_API_KEY"] == "private-test-key"
    assert env["AERO_PUBLIC_RUNS"] == "1"
    assert env["AERO_SPEND_CAP_USD"] == ".75" or env["AERO_SPEND_CAP_USD"] == "0.75"
    assert env["AERO_RUNLOG_DIR"] == str(output / "logs")


def test_live_runs_without_credentials_fail_before_spending(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        module().isolated_environment(tmp_path, output, 8000, 5173, 1, 1.0)


def test_live_runs_are_distributed_and_no_default_live_calls():
    assert module().live_indices(20, 4) == {0, 5, 10, 15}
    assert module().live_indices(20, 0) == set()


def test_browser_engines_are_grouped_to_avoid_concurrent_software_webgl():
    rehearsal = module()
    schedule = [rehearsal.engine_for_attempt(index, 20, ["chromium", "firefox"])
                for index in range(20)]
    assert schedule == ["chromium"] * 10 + ["firefox"] * 10
    assert rehearsal.engine_for_attempt(0, 1, ["firefox"]) == "firefox"


def test_context_page_crash_is_an_attempt_failure_and_context_is_closed(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from types import SimpleNamespace

    class Context:
        closed = False

        def new_page(self):
            raise RuntimeError("Target crashed before navigation")

        def close(self):
            self.closed = True

    context = Context()
    browser = SimpleNamespace(new_context=lambda **options: context)
    services = SimpleNamespace(api="http://127.0.0.1:8000", web="http://127.0.0.1:5173")
    result = module().exercise_attempt(browser, services, tmp_path, 11, "firefox", False, False)
    assert result["attempt"] == 11
    assert result["engine"] == "firefox"
    assert result["passed"] is False
    assert "Target crashed before navigation" in result["error"]
    assert context.closed
    assert result["checks"] == []


def test_context_creation_crash_is_retained_without_a_cleanup_crash(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from types import SimpleNamespace

    def fail(**options):
        raise RuntimeError("context unavailable")

    services = SimpleNamespace(api="http://127.0.0.1:8000", web="http://127.0.0.1:5173")
    result = module().exercise_attempt(SimpleNamespace(new_context=fail), services,
                                       tmp_path, 1, "chromium", False, False)
    assert result["passed"] is False
    assert "context unavailable" in result["error"]
    assert "cleanup_error" not in result


def test_historical_seed_copies_without_changing_originals(tmp_path):
    source = tmp_path / "runs/matched-20261016/logs"
    (source / "events").mkdir(parents=True)
    original = json.dumps({"iteration": 0, "cd": .2246}) + "\n"
    for index in range(15):
        (source / f"matched-{index}.jsonl").write_text(original)
        (source / "events" / f"matched-{index}.jsonl").write_text('{"type":"run_finished"}\n')
    output = tmp_path / "isolated"
    output.mkdir()
    assert module().seed_historical_logs(tmp_path, output) == 15
    assert (source / "matched-0.jsonl").read_text() == original
    assert (output / "logs/matched-0.jsonl").read_text() == original
