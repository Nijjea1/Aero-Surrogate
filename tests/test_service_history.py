import pytest
from fastapi.testclient import TestClient

from src.agent.designer import run
from src.contracts import Design, PARAM_BOUNDS, PRESSURE_REGIONS, default_params
from src.service.app import create_app
from src.service.predictor import StubPredictor
from src.service.runlog import RunLog


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(StubPredictor(), RunLog(str(tmp_path)))) as client:
        yield client


def test_compare_direction_and_trust(client):
    a = default_params()
    b = dict(a, ride_height=0.14)
    response = client.post("/compare", json={"design_a": {"params": a}, "design_b": {"params": b}})
    assert response.status_code == 200
    result = response.json()
    assert result["trusted"]
    assert result["delta"]["cd"] == pytest.approx(result["design_b"]["cd"] - result["design_a"]["cd"])
    assert set(result["delta"]["pressure_summary"]) == set(PRESSURE_REGIONS)
    assert result["unavailable_pressure_regions"] == []
    b = {name: hi for name, (_, hi) in PARAM_BOUNDS.items()}
    result = client.post("/compare", json={"design_a": {"params": a}, "design_b": {"params": b}}).json()
    assert not result["trusted"]
    assert result["reject_reasons"]["design_b"]


def test_compare_validates_both_designs(client):
    assert client.post("/compare", json={"design_a": {"params": default_params()}, "design_b": {"params": {}}}).status_code == 422


def test_history_reads_scripted_run(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_RUNLOG_DIR", str(tmp_path))
    predictor = StubPredictor()
    run(10, "integration", True, evaluator=lambda params: predictor.evaluate(Design(params)).to_dict())
    rows = client.get("/history/integration").json()
    assert len(rows) == 10
    assert rows[1]["accepted"] is False
    assert rows[1]["best_cd_so_far"] == rows[0]["best_cd_so_far"]
    assert "gate feedback" in rows[2]["reasoning"]
    assert client.get("/history/missing").status_code == 404
    assert client.get("/history/bad.id").status_code == 422


def test_corrupt_history_is_server_error(client, tmp_path):
    (tmp_path / "corrupt.jsonl").write_text('invalid\n', encoding="utf-8")
    assert client.get("/history/corrupt").status_code == 500


def test_invalid_utf8_history_is_server_error(client, tmp_path):
    (tmp_path / "corrupt-bytes.jsonl").write_bytes(b"\xff\n")
    response = client.get("/history/corrupt-bytes")
    assert response.status_code == 500
    assert "invalid UTF-8" in response.json()["detail"]
