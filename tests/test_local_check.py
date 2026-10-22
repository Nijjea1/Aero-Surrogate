"""The local wiring rehearsal must stay offline and terminate its processes."""

from scripts.check_local import check


def test_synthetic_http_wiring_check_is_explicitly_scripted(monkeypatch):
    # A user's configured artifact/key must not accidentally turn this fixture
    # into paid inference or mutate their historical run directory.
    monkeypatch.setenv("AERO_CHECKPOINT", "not-a-real-checkpoint")
    monkeypatch.setenv("NEBIUS_API_KEY", "never-send-this")
    report = check()
    assert report["passed"], report["errors"]
    assert report["service"]["synthetic"] is True
    assert "no browser or paid provider calls" in report["scope"]
    assert any("scripted" in line and "history" in line for line in report["checks"])
