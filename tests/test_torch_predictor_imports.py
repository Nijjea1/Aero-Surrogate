"""The lightweight service must not need torch. Runs without torch installed."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from src.service.app import create_app
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def test_service_import_does_not_load_torch():
    code = ("import sys; import src.service.app, src.service.predictor; "
            "assert 'torch' not in sys.modules, 'torch imported at module load'; "
            "assert not any(m.startswith('src.train') for m in sys.modules)")
    env = dict(os.environ, AERO_CHECKPOINT="")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_stub_health_reports_synthetic_provenance(tmp_path):
    with TestClient(create_app(StubPredictor(), RunLog(str(tmp_path / "logs")))) as client:
        health = client.get("/health").json()
        assert health["predictor"] == "stub"
        assert health["model_version"] == "stub-v2-synthetic"
        assert health["checkpoint_sha256"] is None
        assert set(health["output_provenance"]) == {
            "cd", "uncertainty", "ood_score", "cl", "pressure_summary"}
        assert set(health["output_provenance"].values()) == {"synthetic"}
        schema = client.get("/schema").json()
        assert schema["ood_score_threshold"] == 3.0
        assert schema["uncertainty_threshold"] == 0.013
