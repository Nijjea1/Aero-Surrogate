"""Bounded, CPU-only stage-2 serving, separate from the frozen Evaluation seam.

The training adapter owns inference. This wrapper refuses incomplete artifacts,
uses the viewer's exact triangle order, and never promotes Cp sigma into Cd
confidence. Limits are conservative starting values, not a real-model benchmark.
Only operator-configured local checkpoints are loaded; requests cannot name files.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore

from src.contracts import Design, PARAM_NAMES, PRESSURE_REGIONS, SCHEMA_VERSION, validate_body_style
from src.service.predictor import validate_design_params


class FieldCheckpointError(ValueError):
    """An explicitly configured field artifact cannot safely be served."""


class FieldInputError(ValueError):
    """The design or canonical mesh cannot be served without altering it."""


class FieldUnavailable(RuntimeError):
    """The single field inference slot is occupied; retry later."""


class FieldOutputError(RuntimeError):
    """Inference did not return a complete finite field."""


@dataclass(frozen=True)
class FieldLimits:
    max_faces: int = 50_000
    max_vertices: int = 50_000
    max_cloud: int = 4096
    chunk_size: int = 256
    max_classifier_points: int = 100_000
    max_response_bytes: int = 12_000_000
    max_checkpoint_bytes: int = 256_000_000

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"field limit {name} must be a positive integer")
        if self.chunk_size > 4096 or self.max_cloud > 4096:
            raise ValueError("field chunk and cloud limits must be <= 4096")


def _positive_int(value, name: str, maximum: int):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise FieldCheckpointError(f"{name} must be an integer in [1, {maximum}]")


def _validate_checkpoint(ckpt: dict, limits: FieldLimits):
    import numpy as np
    import torch

    if not isinstance(ckpt, dict):
        raise FieldCheckpointError("field checkpoint must be a dictionary")
    if ckpt.get("schema_version") != SCHEMA_VERSION:
        raise FieldCheckpointError(f"field schema_version must be {SCHEMA_VERSION}")
    prep = ckpt.get("preprocess")
    if not isinstance(prep, dict) or prep.get("version") != "fields-v1":
        raise FieldCheckpointError("field preprocess.version must be fields-v1")
    expected = {
        "input": "surface point cloud + query points",
        "normalize": "bbox centre / max vertex radius, as stage 1",
        "target": "CpMeanTrim, standardised by norm_stats",
    }
    for key, value in expected.items():
        if prep.get(key) != value:
            raise FieldCheckpointError(f"unsupported field preprocessing: {key}")
    _positive_int(prep.get("n_cloud"), "preprocess.n_cloud", limits.max_cloud)
    version = ckpt.get("version")
    if not isinstance(version, str) or not version.strip() or len(version) > 256:
        raise FieldCheckpointError("field version must be a nonempty bounded string")
    norm = ckpt.get("norm_stats")
    if not isinstance(norm, dict):
        raise FieldCheckpointError("missing field norm_stats")
    for key in ("cp_mean", "cp_std"):
        value = norm.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise FieldCheckpointError(f"field norm_stats.{key} must be finite")
    if norm["cp_std"] <= 0:
        raise FieldCheckpointError("field cp_std must be positive")

    cfg = ckpt.get("config")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict):
        raise FieldCheckpointError("missing field model config")
    model = cfg["model"]
    if model.get("heteroscedastic", True) is not True:
        raise FieldCheckpointError("field checkpoint must include a variance head")
    for key, default, maximum in (("latent_dim", 256, 512), ("local_dim", 96, 256),
                                  ("hidden", 256, 512), ("n_layers", 4, 8)):
        _positive_int(model.get(key, default), f"model.{key}", maximum)
    freq = model.get("n_freq", 8)
    if isinstance(freq, bool) or not isinstance(freq, int) or not 0 <= freq <= 16:
        raise FieldCheckpointError("model.n_freq must be an integer in [0, 16]")
    ks = model.get("local_k", [8, 32, 128])
    ks = [ks] if isinstance(ks, int) else ks
    if not isinstance(ks, (list, tuple)) or not 1 <= len(ks) <= 4:
        raise FieldCheckpointError("model.local_k must contain 1 to 4 neighbourhood sizes")
    for k in ks:
        _positive_int(k, "model.local_k", 128)
    dropout = model.get("dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise FieldCheckpointError("model.dropout must be in [0, 1)")

    state = ckpt.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise FieldCheckpointError("missing field state_dict")
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise FieldCheckpointError("field state_dict must contain named tensors")
        if tensor.layout != torch.strided or tensor.numel() > 20_000_000:
            raise FieldCheckpointError("unsupported field state tensor layout or size")
        if not torch.isfinite(tensor).all():
            raise FieldCheckpointError(f"nonfinite field state tensor: {key}")

    classifier = ckpt.get("region_classifier")
    if not isinstance(classifier, dict):
        raise FieldCheckpointError("field region_classifier is required; fallback labels are forbidden")
    try:
        points = np.asarray(classifier.get("points"), dtype=np.float64)
        labels = np.asarray(classifier.get("labels"))
    except (TypeError, ValueError) as exc:
        raise FieldCheckpointError("malformed field region_classifier arrays") from exc
    if (points.ndim != 2 or points.shape[1] != 6 or
            not 1 <= len(points) <= limits.max_classifier_points or not np.isfinite(points).all()):
        raise FieldCheckpointError("field classifier needs bounded finite (N, 6) points")
    if (labels.shape != (len(points),) or labels.dtype.kind not in "iu" or
            (labels < 0).any() or (labels >= len(PRESSURE_REGIONS)).any()):
        raise FieldCheckpointError("field classifier labels must be valid region indices")
    _positive_int(classifier.get("k", 5), "region_classifier.k", min(len(points), 32))
    weight = classifier.get("normal_weight", 0.35)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
        raise FieldCheckpointError("field classifier normal_weight must be finite and positive")


class FieldService:
    """A loaded real field adapter with one nonblocking inference slot.

    Canonical geometry comes from mesh_payload, not params_to_mesh: their cap
    topology differs. Hashing the exact little-endian viewer buffers makes a
    stale or differently tessellated viewer mesh detectable without guessing.
    """

    def __init__(self, predictor, checkpoint_sha256: str, limits: FieldLimits):
        self.predictor = predictor
        self.checkpoint_sha256 = checkpoint_sha256
        self.limits = limits
        self._slot = BoundedSemaphore(1)

    @classmethod
    def load(cls, checkpoint: str | Path, limits: FieldLimits | None = None):
        import importlib
        import numpy as np
        import torch
        from src.train.field_predict import FieldPredictor
        from src.train.train_fields import build_model_from_cfg

        limits = limits or FieldLimits()
        try:
            # These are lazy in the upstream adapter. Check them here rather
            # than discovering an incomplete serving image on its first call.
            importlib.import_module("trimesh")
            importlib.import_module("scipy.spatial")
            path = Path(checkpoint)
            if not path.is_file() or path.stat().st_size > limits.max_checkpoint_bytes:
                raise FieldCheckpointError("field checkpoint missing or exceeds size limit")
            # The artifact contains tensors and plain data, not pickled model objects.
            # weights_only prevents the configured file from executing arbitrary code.
            payload = path.read_bytes()
            import io
            ckpt = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            _validate_checkpoint(ckpt, limits)
            expected = build_model_from_cfg(ckpt["config"]).state_dict()
            state = ckpt["state_dict"]
            if state.keys() != expected.keys():
                raise FieldCheckpointError("field checkpoint cannot be served: state_dict keys do not match model")
            for key, tensor in state.items():
                if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
                    raise FieldCheckpointError(f"field state tensor shape/dtype mismatch: {key}")
            predictor = FieldPredictor(ckpt, device=torch.device("cpu"))
            # Exercise the actual network/sigma decode at startup, cheaply; do not
            # require a production geometry or pretend this calibrates the field.
            cloud = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
            with torch.inference_mode():
                mean, sigma = predictor.model.split(predictor.model(cloud, cloud[:, :2], cloud[:, :2]))
            if mean.shape != (1, 2, 1) or sigma.shape != mean.shape:
                raise FieldCheckpointError("field startup probe returned incompatible shapes")
            if not torch.isfinite(mean).all() or not torch.isfinite(sigma).all() or not (sigma > 0).all():
                raise FieldCheckpointError("field startup probe returned invalid Cp or sigma")
            if not np.isfinite(mean.numpy() * predictor.cp_std + predictor.cp_mean).all():
                raise FieldCheckpointError("field startup probe Cp decoding overflowed")
            if not np.isfinite(sigma.numpy() * predictor.cp_std).all():
                raise FieldCheckpointError("field startup probe sigma decoding overflowed")
            return cls(predictor, hashlib.sha256(payload).hexdigest(), limits)
        except FieldCheckpointError:
            raise
        except Exception as exc:
            raise FieldCheckpointError(f"field checkpoint cannot be served: {type(exc).__name__}: {exc}") from exc

    def provenance(self) -> dict:
        return {
            "available": True, "synthetic": False,
            "model_version": self.predictor.version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocess_version": "fields-v1",
            "geometry_source": "src.geometry.viewer.mesh_payload",
            "geometry_fidelity": "approximate parametric viewer mesh; not CFD geometry",
            "field_trust_status": "unknown_for_generated_geometry",
            "uncertainty": {"quantity": "Cp standard deviation", "units": "dimensionless",
                            "calibration": "not established for generated viewer geometry",
                            "used_by_cd_gate": False},
            "coordinate_system": {"x": "nose to tail", "y": "lateral", "z": "up", "units": "metres"},
            "orientation_validation": "training-to-viewer orientation not independently validated",
            "limits": {"max_faces": self.limits.max_faces, "chunk_size": self.limits.chunk_size,
                       "max_concurrent_inferences": 1, "production_benchmark": "pending"},
        }

    def predict(self, design: Design) -> dict:
        problems = validate_design_params(design.params) + validate_body_style(design.body_style)
        if problems:
            raise FieldInputError("; ".join(problems))
        if not self._slot.acquire(blocking=False):
            raise FieldUnavailable("field inference is busy; retry later")
        try:
            return self._predict(design)
        finally:
            self._slot.release()

    def _predict(self, design: Design) -> dict:
        import numpy as np
        import trimesh
        from src.geometry.viewer import mesh_payload

        started = time.perf_counter()
        payload = mesh_payload(design.params, design.body_style)
        vertices, faces, mesh_hash = self._decode_mesh(payload)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        areas = np.asarray(mesh.area_faces)
        normals = np.asarray(mesh.face_normals)
        radius = float(np.linalg.norm(vertices - mesh.bounds.mean(axis=0), axis=1).max())
        if not np.isfinite(radius) or radius <= 0 or not np.isfinite(areas).all() or (areas <= 0).any():
            raise FieldInputError("canonical mesh contains degenerate geometry")
        if not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-5):
            raise FieldInputError("canonical mesh normals are invalid")
        geometry_finished = time.perf_counter()
        try:
            pred = self.predictor.predict(mesh, max_faces=None, chunk=self.limits.chunk_size)
        except Exception as exc:
            raise FieldOutputError(f"field inference failed: {type(exc).__name__}") from exc
        inference_finished = time.perf_counter()
        cp, sigma, region = np.asarray(pred.cp), np.asarray(pred.sigma), np.asarray(pred.region)
        n = len(faces)
        if any(array.shape != (n,) for array in (cp, sigma, region)):
            raise FieldOutputError("field result does not align with canonical faces")
        if not np.isfinite(cp).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise FieldOutputError("field Cp/sigma must be finite and sigma positive")
        if region.dtype.kind not in "iu" or (region < 0).any() or (region >= len(PRESSURE_REGIONS)).any():
            raise FieldOutputError("field classifier returned invalid region indices")
        summaries = {}
        for index, name in enumerate(PRESSURE_REGIONS):
            selected = region == index
            summaries[name] = float(np.average(cp[selected], weights=areas[selected])) if selected.any() else None
        response = {
            "field_schema_version": "1.0.0", "design_id": design.design_id,
            "design_sha256": hashlib.sha256(json.dumps(
                {"params": {key: float(design.params[key]) for key in PARAM_NAMES},
                 "body_style": design.body_style}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "body_style": design.body_style,
            "mesh": {"sha256": mesh_hash, "hash_format": "viewer-f32le-u32le-v1",
                     "n_vertices": len(vertices), "n_faces": n, "association": "face",
                     "query_location": "face_centroid", "face_indices": list(range(n))},
            "cp": cp.tolist(), "sigma": sigma.tolist(), "region": region.tolist(),
            "regions": list(PRESSURE_REGIONS), "pressure_summary": summaries,
            "unavailable_pressure_regions": [key for key, value in summaries.items() if value is None],
            "provenance": self.provenance(),
            "timing": {"geometry_ms": (geometry_finished - started) * 1000,
                       "inference_ms": (inference_finished - geometry_finished) * 1000,
                       "through_summary_ms": (time.perf_counter() - started) * 1000},
        }
        try:
            size = len(json.dumps(response, allow_nan=False, separators=(",", ":")).encode())
        except (ValueError, OverflowError) as exc:
            raise FieldOutputError("field result cannot be serialized as finite JSON") from exc
        if size > self.limits.max_response_bytes:
            raise FieldOutputError("field response exceeds configured byte limit")
        return response

    def _decode_mesh(self, payload: dict):
        import numpy as np

        nv, nf = payload.get("n_vertices"), payload.get("n_faces")
        for count, maximum, name in ((nv, self.limits.max_vertices, "vertices"),
                                     (nf, self.limits.max_faces, "faces")):
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= maximum:
                raise FieldInputError(f"canonical mesh {name} exceed supported count")
        buffers = []
        try:
            for name, count in (("vertices", nv), ("faces", nf)):
                encoded = payload[name]
                if not isinstance(encoded, str) or len(encoded) > 4 * ((count * 12 + 2) // 3):
                    raise ValueError("invalid encoded size")
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) != count * 12:
                    raise ValueError("buffer length does not match mesh count")
                buffers.append(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise FieldInputError("canonical mesh buffers are malformed") from exc
        vertices = np.frombuffer(buffers[0], dtype="<f4").reshape(nv, 3)
        faces = np.frombuffer(buffers[1], dtype="<u4").reshape(nf, 3)
        if not np.isfinite(vertices).all() or (faces >= nv).any():
            raise FieldInputError("canonical mesh has nonfinite vertices or out-of-range indices")
        digest = hashlib.sha256(b"viewer-f32le-u32le-v1\0" + buffers[0] + buffers[1]).hexdigest()
        return vertices, faces, digest
