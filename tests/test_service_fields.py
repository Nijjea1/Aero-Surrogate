"""Track B field-serving tests use Avneet's actual trainer, never a live API.

Synthetic fixture training verifies the artifact/adapter seam without claiming
physics accuracy. Malformed-artifact cases are derived from that real artifact.
"""

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from src.contracts import Design, PRESSURE_REGIONS, default_params
from src.service.fields import (
    FieldCheckpointError, FieldInputError, FieldLimits, FieldOutputError,
    FieldService, FieldUnavailable,
)


@pytest.fixture(scope="module")
def field_artifact(tmp_path_factory):
    from src.train.train_fields import train_fields

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("service-fields")
    (root / "fields").mkdir()
    rng = np.random.default_rng(21)
    for run in range(1, 7):
        xyz = rng.normal(size=(64, 3)).astype(np.float32)
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        area = np.full(64, 0.01, np.float32)
        cp = (-0.2 + xyz[:, 0] * 0.5 + run * 0.01).astype(np.float32)
        meta = {"run": run, "version": "fields-v1", "regions": list(PRESSURE_REGIONS),
                "a_ref_m2": 2., "total_area_m2": 1., "n_cells_total": 64}
        np.savez_compressed(root / "fields" / f"run_{run}.npz", query_xyz=xyz,
                            query_normal=xyz, query_area=area, cp=cp,
                            pprime2=np.abs(cp), region=np.zeros(64, np.int16),
                            cloud_xyz=xyz, meta=json.dumps(meta))
    cfg = {"run_name": "test-service-fields", "seed": 0,
           "data": {"root": str(root), "val_runs": 1, "test_runs": 1,
                    "n_query": 32, "n_cloud": 16},
           "model": {"latent_dim": 8, "local_k": 4, "local_dim": 8, "n_freq": 0,
                     "hidden": 8, "n_layers": 1, "heteroscedastic": True},
           "train": {"epochs": 1, "variance_epochs": 1, "batch_size": 2,
                     "lr": 0.001, "weight_decay": 0., "out_dir": str(root / "runs")}}
    result = train_fields(cfg, device_str="cpu")
    yield Path(result["checkpoint"])
    torch.set_num_threads(previous_threads)


@pytest.fixture
def service(field_artifact):
    return FieldService.load(field_artifact)


def test_trainer_checkpoint_through_actual_http_serialization(field_artifact, tmp_path):
    from fastapi.testclient import TestClient
    from src.service.app import create_app
    from src.service.predictor import StubPredictor
    from src.service.settings import Settings

    settings = Settings(runlog_dir=str(tmp_path / "logs"), field_checkpoint=str(field_artifact))
    with TestClient(create_app(StubPredictor(), settings=settings)) as client:
        health = client.get("/health").json()
        assert health["fields"]["checkpoint_sha256"] == hashlib.sha256(field_artifact.read_bytes()).hexdigest()
        assert client.get("/ready").status_code == 200
        reply = client.post("/fields", json={"params": default_params(), "body_style": "fastback"})
        assert reply.status_code == 200
        result = reply.json()
        assert result["mesh"]["n_faces"] == len(result["cp"]) == len(result["sigma"])
        assert result["mesh"]["face_indices"] == list(range(len(result["cp"])))
        assert result["provenance"]["field_trust_status"] == "unknown_for_generated_geometry"
        scalar = client.post("/evaluate", json={"params": default_params()}).json()
        assert scalar["model_version"] == "stub-v2-synthetic"


def _tiny_payload():
    mesh = trimesh.creation.box(extents=(2., 1., 0.8))
    return {"vertices": base64.b64encode(np.asarray(mesh.vertices, "<f4").tobytes()).decode(),
            "faces": base64.b64encode(np.asarray(mesh.faces, "<u4").tobytes()).decode(),
            "n_vertices": len(mesh.vertices), "n_faces": len(mesh.faces)}


def test_real_trainer_checkpoint_canonical_face_mapping(service, field_artifact):
    from src.geometry.viewer import mesh_payload

    design = Design(params=default_params(), design_id="baseline")
    result = service.predict(design)
    payload = mesh_payload(design.params, design.body_style)
    vertices, faces, mesh_hash = service._decode_mesh(payload)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    direct = service.predictor.predict(mesh, max_faces=None, chunk=256)
    assert result["mesh"]["face_indices"] == list(range(len(faces)))
    assert result["mesh"]["n_vertices"] == len(vertices)
    assert result["mesh"]["sha256"] == mesh_hash
    expected_hash = hashlib.sha256(b"viewer-f32le-u32le-v1\0" +
                                   base64.b64decode(payload["vertices"]) +
                                   base64.b64decode(payload["faces"])).hexdigest()
    assert mesh_hash == expected_hash
    np.testing.assert_allclose(result["cp"], direct.cp)
    np.testing.assert_allclose(result["sigma"], direct.sigma)
    assert result["region"] == direct.region.tolist()
    assert result["design_id"] == "baseline"
    assert result["provenance"]["checkpoint_sha256"] == hashlib.sha256(field_artifact.read_bytes()).hexdigest()
    assert result["provenance"]["uncertainty"]["used_by_cd_gate"] is False
    assert result["provenance"]["field_trust_status"] == "unknown_for_generated_geometry"
    assert "cd" not in result and "cl" not in result
    assert result["unavailable_pressure_regions"] == list(PRESSURE_REGIONS[1:])
    assert all(result["pressure_summary"][name] is None for name in PRESSURE_REGIONS[1:])
    assert result["pressure_summary"][PRESSURE_REGIONS[0]] == pytest.approx(
        np.average(direct.cp, weights=mesh.area_faces))
    assert next(service.predictor.model.parameters()).device.type == "cpu"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c.update(schema_version="1.0.0"), "schema_version"),
    (lambda c: c["preprocess"].update(version="params-v1"), "preprocess.version"),
    (lambda c: c["preprocess"].update(normalize="other"), "preprocessing"),
    (lambda c: c["preprocess"].update(n_cloud=5000), "n_cloud"),
    (lambda c: c["norm_stats"].update(cp_std=0.), "cp_std"),
    (lambda c: c["norm_stats"].update(cp_mean=float("nan")), "cp_mean"),
    (lambda c: c.update(region_classifier=None), "region_classifier"),
    (lambda c: c["region_classifier"].update(labels=[-1]), "labels"),
    (lambda c: c["region_classifier"].update(points=[[float("nan")] * 6]), "points"),
    (lambda c: c["region_classifier"].update(k=0), "classifier.k"),
    (lambda c: c["region_classifier"].update(normal_weight=-1.), "normal_weight"),
    (lambda c: c["config"]["model"].update(heteroscedastic=False), "variance head"),
    (lambda c: c["config"]["model"].update(hidden=999999), "hidden"),
    (lambda c: c["state_dict"]["mean_head.weight"].fill_(float("inf")), "nonfinite"),
    (lambda c: c["state_dict"].update(mean_head_weight=torch.ones(2)), "keys"),
    (lambda c: c["state_dict"].update({"mean_head.weight": c["state_dict"]["mean_head.weight"].double()}), "dtype"),
    (lambda c: c["state_dict"].pop("mean_head.weight"), "cannot be served"),
])
def test_malformed_artifacts_refuse_load(field_artifact, tmp_path, mutation, message):
    ckpt = torch.load(field_artifact, weights_only=True)
    mutation(ckpt)
    path = tmp_path / "bad.pt"
    torch.save(ckpt, path)
    with pytest.raises(FieldCheckpointError, match=message):
        FieldService.load(path)


def test_bad_file_limits_and_checkpoint_errors(field_artifact, tmp_path):
    with pytest.raises(FieldCheckpointError, match="missing"):
        FieldService.load(tmp_path / "missing.pt")
    with pytest.raises(FieldCheckpointError, match="size"):
        FieldService.load(field_artifact, FieldLimits(max_checkpoint_bytes=1))
    with pytest.raises(ValueError):
        FieldLimits(chunk_size=0)
    with pytest.raises(ValueError):
        FieldLimits(max_cloud=8192)


def test_invalid_design_rejected_before_geometry(service, monkeypatch):
    monkeypatch.setattr("src.geometry.viewer.mesh_payload", lambda *a: pytest.fail("geometry called"))
    with pytest.raises(FieldInputError):
        service.predict(Design(params={**default_params(), "A_Car_Length": 131.}))
    with pytest.raises(FieldInputError):
        service.predict(Design(params=default_params(), body_style="unknown"))


@pytest.mark.parametrize("body_style", ["notchback", "estateback"])
def test_other_body_styles_have_complete_canonical_fields(service, body_style):
    result = service.predict(Design(params=default_params(), body_style=body_style))
    n = result["mesh"]["n_faces"]
    assert len(result["cp"]) == len(result["sigma"]) == len(result["region"]) == n
    assert result["body_style"] == body_style


@pytest.mark.parametrize("kind", ["faces", "vertices", "buffer", "nan", "indices", "degenerate"])
def test_mesh_invalid_or_oversized_rejected(service, monkeypatch, kind):
    payload = _tiny_payload()
    if kind == "faces":
        payload["n_faces"] = service.limits.max_faces + 1
    elif kind == "vertices":
        payload["n_vertices"] = service.limits.max_vertices + 1
    elif kind == "buffer":
        payload["vertices"] = "!!!"
    elif kind == "nan":
        vertices = np.frombuffer(base64.b64decode(payload["vertices"]), "<f4").copy()
        vertices[0] = np.nan
        payload["vertices"] = base64.b64encode(vertices.tobytes()).decode()
    else:
        faces = np.frombuffer(base64.b64decode(payload["faces"]), "<u4").copy()
        faces[0] = 999999 if kind == "indices" else faces[1]
        if kind == "degenerate":
            faces[:3] = 0
        payload["faces"] = base64.b64encode(faces.tobytes()).decode()
    monkeypatch.setattr("src.geometry.viewer.mesh_payload", lambda *a: payload)
    with pytest.raises(FieldInputError):
        service.predict(Design(params=default_params()))


def test_busy_slot_is_nonblocking_and_released_after_error(service, monkeypatch):
    entered, release = Event(), Event()

    def blocked(design):
        entered.set()
        assert release.wait(5)
        raise FieldOutputError("test failure")

    monkeypatch.setattr(service, "_predict", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(service.predict, Design(params=default_params()))
        try:
            assert entered.wait(5)
            with pytest.raises(FieldUnavailable, match="busy"):
                service.predict(Design(params=default_params()))
        finally:
            release.set()
        with pytest.raises(FieldOutputError):
            first.result()
    assert service._slot.acquire(blocking=False)
    service._slot.release()


@pytest.mark.parametrize("bad", ["shape", "cp", "sigma", "region", "raise", "size"])
def test_bad_inference_outputs_are_not_exposed(service, monkeypatch, bad):
    payload = _tiny_payload()
    monkeypatch.setattr("src.geometry.viewer.mesh_payload", lambda *a: payload)
    n = payload["n_faces"]
    result = SimpleNamespace(cp=np.ones(n), sigma=np.ones(n), region=np.zeros(n, np.int16))
    if bad == "shape":
        result.cp = np.ones(n - 1)
    elif bad == "cp":
        result.cp[0] = np.inf
    elif bad == "sigma":
        result.sigma[0] = -1
    elif bad == "region":
        result.region[0] = 99
    elif bad == "size":
        service.limits = replace(service.limits, max_response_bytes=1)

    def predict(*args, **kwargs):
        if bad == "raise":
            raise RuntimeError("failed")
        return result

    monkeypatch.setattr(service.predictor, "predict", predict)
    with pytest.raises(FieldOutputError):
        service.predict(Design(params=default_params()))
