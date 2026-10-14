import math

import pytest
from fastapi.testclient import TestClient

from src.contracts import Evaluation, PRESSURE_REGIONS, default_params
from src.service.app import create_app
from src.service.predictor import StubPredictor


@pytest.fixture
def client():
    with TestClient(create_app(StubPredictor())) as client:
        yield client


def test_discovery_and_evaluation_contract(client):
    assert client.get("/health").json()["synthetic"] is True
    schema = client.get("/schema").json()
    assert schema["defaults"] == default_params()
    response = client.post("/evaluate", json={"params": schema["defaults"]})
    assert response.status_code == 200
    result = Evaluation(**response.json())
    assert result.in_distribution
    assert set(result.pressure_summary) == set(PRESSURE_REGIONS)
    assert math.isfinite(result.cd)
    assert response.json() == client.post("/evaluate", json={"params": default_params()}).json()


@pytest.mark.parametrize("value", [None, {}, [], "bad", True])
def test_invalid_parameter_container(client, value):
    assert client.post("/evaluate", json={"params": value}).status_code == 422


@pytest.mark.parametrize("value", [None, "0.15", True, [], {}, 99])
def test_invalid_values(client, value):
    params = default_params()
    params["ride_height"] = value
    response = client.post("/evaluate", json={"params": params})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_nonfinite_and_unknown_keys(client):
    body = '{"params":{"ride_height":NaN}}'
    assert client.post("/evaluate", content=body, headers={"Content-Type": "application/json"}).status_code == 422
    params = default_params()
    params["typo"] = 1
    assert client.post("/evaluate", json={"params": params}).status_code == 422
    assert client.post("/evaluate", json={"params": default_params(), "extra": 1}).status_code == 422


def test_cors(client):
    response = client.options("/evaluate", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"
    })
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_configured_checkpoint_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("AERO_CHECKPOINT", "missing.pt")
    with pytest.raises((NotImplementedError, FileNotFoundError, RuntimeError)):
        with TestClient(create_app()):
            pass
