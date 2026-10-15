"""The optional surface API must not change scalar behavior or hide failures."""

import pytest
from fastapi.testclient import TestClient

from src.contracts import default_params
from src.service.app import create_app
from src.service.predictor import StubPredictor
from src.service.settings import Settings


def test_fields_disabled_and_health_are_explicit(tmp_path):
    with TestClient(create_app(StubPredictor(), settings=Settings(runlog_dir=str(tmp_path)))) as client:
        assert client.get("/health").json()["fields"]["available"] is False
        assert client.get("/ready").status_code == 200
        assert client.post("/fields", json={"params": default_params()}).status_code == 503
        assert client.post("/evaluate", json={"params": default_params()}).status_code == 200


def test_bad_configured_field_checkpoint_fails_startup(tmp_path):
    app = create_app(StubPredictor(), settings=Settings(runlog_dir=str(tmp_path),
                                                      field_checkpoint=str(tmp_path / "missing.pt")))
    with pytest.raises(ValueError, match="checkpoint|file|found"):
        with TestClient(app):
            pass


def test_field_config_parsing_and_bounds():
    config = Settings.from_env({"AERO_FIELD_CHECKPOINT": "  checkpoints/fields.pt ",
                                "AERO_FIELD_CHUNK_SIZE": "64", "AERO_FIELD_MAX_FACES": "1000"})
    assert config.field_checkpoint == "checkpoints/fields.pt"
    assert config.field_chunk_size == 64 and config.field_max_faces == 1000
    for name in ("AERO_FIELD_MAX_FACES", "AERO_FIELD_MAX_VERTICES", "AERO_FIELD_MAX_CLOUD",
                 "AERO_FIELD_CHUNK_SIZE", "AERO_FIELD_MAX_CLASSIFIER_POINTS", "AERO_FIELD_MAX_RESPONSE_BYTES"):
        with pytest.raises(ValueError, match=name):
            Settings.from_env({name: "0"})
    for name in ("AERO_FIELD_MAX_CLOUD", "AERO_FIELD_CHUNK_SIZE"):
        with pytest.raises(ValueError, match=name):
            Settings.from_env({name: "4097"})


class Fields:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def provenance(self):
        return {"available": True, "checkpoint_sha256": "b" * 64}

    def predict(self, design):
        self.calls.append(design)
        if self.error:
            raise self.error
        return {"association": "face", "cp": [0.2], "sigma": [0.1], "face_indices": [0]}


def test_fields_api_validates_design_and_keeps_scalar_provenance(tmp_path):
    service = Fields()
    with TestClient(create_app(StubPredictor(), settings=Settings(runlog_dir=str(tmp_path)),
                               field_service=service)) as client:
        health = client.get("/health").json()
        assert health["fields"]["available"] and health["output_provenance"]["cd"] == "synthetic"
        body = {"params": default_params(), "body_style": "estateback", "design_id": "example"}
        assert client.post("/fields", json=body).json()["association"] == "face"
        assert service.calls[0].body_style == "estateback"
        assert service.calls[0].design_id == "example"
        body["params"]["B_Ramp_Angle"] = 999
        assert client.post("/fields", json=body).status_code == 422
        assert len(service.calls) == 1


def test_fields_busy_and_failed_output_are_visible_with_cors(tmp_path):
    from src.service.fields import FieldInputError, FieldOutputError, FieldUnavailable

    for error, status in ((FieldUnavailable("busy"), 503), (FieldInputError("bad mesh"), 422),
                          (FieldOutputError("invalid field"), 500)):
        with TestClient(create_app(StubPredictor(), settings=Settings(runlog_dir=str(tmp_path)),
                                   field_service=Fields(error))) as client:
            response = client.post("/fields", json={"params": default_params()},
                                   headers={"Origin": "http://localhost:5173"})
            assert response.status_code == status
            assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
            if status == 503:
                assert response.headers["retry-after"] == "1"


def test_fields_share_scoring_rate_limit(tmp_path):
    with TestClient(create_app(StubPredictor(), settings=Settings(runlog_dir=str(tmp_path),
                               rate_limit_per_minute=1), field_service=Fields())) as client:
        assert client.post("/fields", json={"params": default_params()}).status_code == 200
        assert client.post("/evaluate", json={"params": default_params()}).status_code == 429
