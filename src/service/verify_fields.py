"""Offline fixed-query parity for Avneet's independent field reference pack.

The pack omits the actual cloud and rounds its query coordinates. The ORIGINAL
extracted run NPZ is therefore mandatory: replay selection and infer on its
unrounded float32 tensors. Never invent a cloud from the listed query points or
widen tolerance to hide rounding error. This qualifies checkpoint point outputs,
not design-to-mesh /fields behavior, region prediction, CFD accuracy or a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

from src.contracts import PRESSURE_REGIONS, SCHEMA_VERSION
from src.service.fields import FieldLimits, FieldService

MAX_REFERENCE_BYTES = 2_000_000
MAX_SOURCE_BYTES = 64_000_000
MAX_SAMPLE_POINTS = 256


def _read_bounded(path: str | Path, maximum: int, label: str) -> bytes:
    with Path(path).open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"{label} exceeds {maximum} byte limit")
    return payload


def _integer(value, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _array(value, shape, label: str, *, integer=False):
    import numpy as np

    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{label} shape must be {shape}")
    # Reject ragged/oversized rows before numpy constructs an array.
    flat = value
    if len(shape) == 2:
        if any(not isinstance(row, list) or len(row) != shape[1] for row in value):
            raise ValueError(f"{label} shape must be {shape}")
        flat = [element for row in value for element in row]
    if any(type(item) not in ((int,) if integer else (int, float)) for item in flat):
        raise ValueError(f"{label} must contain {'integer' if integer else 'numeric'} values")
    result = np.asarray(value)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _validate_manifest(manifest):
    import numpy as np

    if not isinstance(manifest, dict) or manifest.get("kind") != "field_reference":
        raise ValueError("reference kind must be field_reference")
    tolerance = manifest.get("tolerance")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 < tolerance <= 1e-4:
        raise ValueError("reference tolerance must be positive and at most 1e-4")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("reference checkpoint metadata is required")
    if not isinstance(checkpoint.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", checkpoint["sha256"]):
        raise ValueError("reference checkpoint sha256 must be a full lowercase digest")
    for key in ("schema_version", "contract_schema_version"):
        if checkpoint.get(key) != SCHEMA_VERSION:
            raise ValueError(f"reference checkpoint {key} must be {SCHEMA_VERSION}")
    prep = checkpoint.get("preprocess")
    if not isinstance(prep, dict) or prep.get("version") != "fields-v1":
        raise ValueError("reference checkpoint preprocessing must be fields-v1")
    _integer(prep.get("n_cloud"), "reference n_cloud", 4096)
    if not isinstance(checkpoint.get("version"), str) or not checkpoint["version"].strip():
        raise ValueError("reference checkpoint version is required")
    conventions = manifest.get("conventions")
    if not isinstance(conventions, dict) or conventions.get("length_units") != "metres":
        raise ValueError("reference conventions must declare metres")
    regions = conventions.get("region_order")
    if (not isinstance(regions, list) or not 1 <= len(regions) <= 64 or
            any(not isinstance(name, str) or not name for name in regions) or len(set(regions)) != len(regions)):
        raise ValueError("reference region_order must contain unique names")
    if conventions.get("contract_region_order") != list(PRESSURE_REGIONS):
        raise ValueError("reference contract_region_order differs from the service")
    sample = manifest.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("reference sample is required")
    count = _integer(sample.get("n_points"), "sample.n_points", MAX_SAMPLE_POINTS)
    _integer(sample.get("run"), "sample.run", 1_000_000)
    arrays = {}
    for key, shape in (("query_xyz", (count, 3)), ("query_normal", (count, 3)),
                       ("query_area_m2", (count,)), ("region_index", (count,)),
                       ("expected_cp", (count,)), ("expected_sigma", (count,))):
        arrays[key] = _array(sample.get(key), shape, f"sample.{key}", integer=key == "region_index")
    # The generator rounds areas to 9 and sigma to 8 decimals. A tiny positive
    # original may serialize as zero; enforce strict positivity on the original
    # source areas and inferred sigma instead of rejecting honest rounding.
    if (arrays["query_area_m2"] < 0).any() or (arrays["expected_sigma"] < 0).any():
        raise ValueError("reference sample areas and sigma must be nonnegative (positive before rounding)")
    if (arrays["region_index"] < 0).any() or (arrays["region_index"] >= len(regions)).any():
        raise ValueError("reference sample region indices are out of range")
    if not np.allclose(np.linalg.norm(arrays["query_normal"], axis=1), 1., atol=2e-6, rtol=0):
        raise ValueError("reference sample normals must have unit length")
    splits = manifest.get("split_ids")
    if not isinstance(splits, dict) or splits.get("by") != "run number, never by point":
        raise ValueError("reference must declare run-level split_ids")
    runs = splits.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"train", "val", "test"}:
        raise ValueError("reference needs train/val/test split runs")
    seen = set()
    for name, ids in runs.items():
        if not isinstance(ids, list) or not ids or len(ids) > 1_000_000:
            raise ValueError(f"reference split {name} must contain run ids")
        for run in ids:
            _integer(run, f"reference split {name} run", 1_000_000)
            if run in seen:
                raise ValueError("reference run splits overlap or contain duplicates")
            seen.add(run)
    if sample["run"] not in runs["test"]:
        raise ValueError("reference sample run must be held out in test")
    return arrays


def _load_source(payload: bytes):
    """Bound decompression AND NPY header allocations before loading arrays."""
    import numpy as np

    needed = {"query_xyz", "query_normal", "query_area", "cp", "cloud_xyz", "region", "meta"}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(members) > 16 or len(set(names)) != len(names):
            raise ValueError("source NPZ has too many or duplicate members")
        if sum(member.file_size for member in members) > MAX_SOURCE_BYTES:
            raise ValueError("source NPZ decompressed bytes exceed limit")
        if not {name + ".npy" for name in needed}.issubset(names):
            raise ValueError("source NPZ is missing original field arrays")
        for member in members:
            if member.filename not in {name + ".npy" for name in needed}:
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported source NPY version")
                if (dtype.hasobject or len(shape) > 2 or
                        math.prod(shape) * dtype.itemsize > member.file_size - stream.tell()):
                    raise ValueError("source NPY header has unsupported dtype/shape/size")
                if member.filename == "meta.npy" and member.file_size > 1_000_000:
                    raise ValueError("source metadata exceeds byte limit")
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            arrays = {name: source[name] for name in needed if name != "meta"}
            meta = json.loads(str(source["meta"]))
    count = len(arrays["cp"]) if arrays["cp"].ndim else 0
    if not 1 <= count <= 200_000:
        raise ValueError("source query count is outside supported bounds")
    for key, shape in (("query_xyz", (count, 3)), ("query_normal", (count, 3)),
                       ("query_area", (count,)), ("cp", (count,))):
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"source {key} must be finite original float32 with shape {shape}")
    cloud = arrays["cloud_xyz"]
    if (cloud.ndim != 2 or cloud.shape[1] != 3 or not 1 <= len(cloud) <= 100_000 or
            cloud.dtype != np.float32 or not np.isfinite(cloud).all()):
        raise ValueError("source cloud_xyz must be bounded finite original float32 (N, 3)")
    region = arrays["region"]
    if region.shape != (count,) or region.dtype.kind not in "iu":
        raise ValueError("source region must contain integer labels aligned with query points")
    if (arrays["query_area"] <= 0).any():
        raise ValueError("source query areas must be positive")
    if not np.allclose(np.linalg.norm(arrays["query_normal"], axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError("source normals must have unit length")
    if not isinstance(meta, dict) or meta.get("version") != "fields-v1":
        raise ValueError("source metadata version must be fields-v1")
    return arrays, meta


def verify_fields(checkpoint: str, manifest: object, source_npz: str | None = None) -> dict:
    """Return a finite, auditable report; never certify unrelated model claims."""
    report = {"passed": False, "errors": [], "qualification_scope": "fixed_query_checkpoint_parity",
              "endpoint_qualified": False, "geometry_preprocessing_qualified": False,
              "region_classifier_qualified": False, "uncertainty_calibration_qualified": False,
              "source_authenticated_by_reference_hash": False,
              "limitations": ["The reference pack omits its cloud and rounds sample coordinates.",
                              "The supplied original source NPZ is required; its hash is recorded, not authenticated by the pack.",
                              "Parity is not CFD accuracy, viewer orientation validation, field trust or calibration."],
              "provenance": None, "results": {}}
    try:
        import numpy as np
        import torch

        expected = _validate_manifest(manifest)
        report["reference_content_sha256"] = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        report["declared_reference_source"] = {
            key: manifest.get(key) for key in ("generated_by", "independent_of", "generated_utc")
        }
        if source_npz is None:
            raise ValueError("original --source-npz is required: the reference does not contain the actual cloud or unrounded queries")
        payload = _read_bounded(source_npz, MAX_SOURCE_BYTES, "source NPZ")
        report["source_npz_sha256"] = hashlib.sha256(payload).hexdigest()
        arrays, meta = _load_source(payload)
        sample = manifest["sample"]
        if meta.get("run") != sample["run"] or type(meta.get("run")) is not int:
            raise ValueError("source NPZ run differs from reference sample")
        if meta.get("regions") != manifest["conventions"]["region_order"]:
            raise ValueError("source region-name ordering differs from reference")
        count = sample["n_points"]
        if count > len(arrays["cp"]):
            raise ValueError("reference sample count exceeds source queries")
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(len(arrays["cp"]), size=count, replace=False))
        n_cloud = manifest["checkpoint"]["preprocess"]["n_cloud"]
        cloud_indices = np.sort(rng.choice(len(arrays["cloud_xyz"]),
                                          size=min(n_cloud, len(arrays["cloud_xyz"])), replace=False))
        for key, source, decimals in (("query_xyz", "query_xyz", 6), ("query_normal", "query_normal", 6),
                                       ("query_area_m2", "query_area", 9)):
            if not np.array_equal(arrays[source][take].round(decimals), expected[key]):
                raise ValueError(f"source selected {key} differs from rounded reference sample")
        if not np.array_equal(arrays["region"][take], expected["region_index"]):
            raise ValueError("source selected region labels differ from reference sample")
        service = FieldService.load(checkpoint)
        report["provenance"] = service.provenance()
        identity = manifest["checkpoint"]
        if service.checkpoint_sha256 != identity["sha256"]:
            raise ValueError("checkpoint SHA-256 differs from reference")
        if service.predictor.version != identity["version"] or service.predictor.preprocess != identity["preprocess"]:
            raise ValueError("checkpoint version/preprocess differs from reference")
        # FieldService deliberately does not retain training split payloads. Read
        # only the same fingerprinted bounded bytes to verify this reference is
        # actually held out by the loaded artifact rather than only by its JSON.
        checkpoint_bytes = _read_bounded(checkpoint, FieldLimits().max_checkpoint_bytes, "checkpoint")
        if hashlib.sha256(checkpoint_bytes).hexdigest() != service.checkpoint_sha256:
            raise ValueError("checkpoint changed during qualification")
        ckpt = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        if ckpt.get("splits") != manifest["split_ids"]["runs"]:
            raise ValueError("checkpoint run splits differ from reference")
        if ckpt.get("regions") != meta["regions"]:
            raise ValueError("checkpoint region ordering differs from source")
        cloud = torch.from_numpy(arrays["cloud_xyz"][cloud_indices]).unsqueeze(0)
        query = torch.from_numpy(arrays["query_xyz"][take]).unsqueeze(0)
        normals = torch.from_numpy(arrays["query_normal"][take]).unsqueeze(0)
        with torch.inference_mode():
            mean, sigma = service.predictor.model.split(service.predictor.model(cloud, query, normals))
        observed = {"cp": mean[0, :, 0].numpy() * service.predictor.cp_std + service.predictor.cp_mean,
                    "sigma": sigma[0, :, 0].numpy() * service.predictor.cp_std}
        if any(value.shape != (count,) or not np.isfinite(value).all() for value in observed.values()):
            raise ValueError("field model returned malformed/nonfinite decoded outputs")
        if (observed["sigma"] <= 0).any():
            raise ValueError("field model returned nonpositive decoded sigma")
        report["selection"] = {"run": sample["run"], "seed": 0, "query_indices": take.tolist(),
                               "cloud_indices": cloud_indices.tolist(), "inputs": "original unrounded float32 source tensors"}
        report["software"] = {"numpy": np.__version__, "torch": str(torch.__version__)}
        report["absolute_tolerance"] = manifest["tolerance"]
        for name, actual in observed.items():
            target = expected[f"expected_{name}"]
            errors = np.abs(actual.astype(np.float64) - target)
            agrees = bool((errors <= manifest["tolerance"]).all())
            report["results"][name] = {"passed": agrees, "max_absolute_error": float(errors.max()),
                                        "expected": target.tolist(), "actual": actual.tolist()}
            if not agrees:
                report["errors"].append(f"{name} differs from independent reference beyond absolute tolerance")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not report["errors"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--out", help="Create a new report; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    try:
        payload = _read_bounded(args.references, MAX_REFERENCE_BYTES, "reference JSON")
        manifest = json.loads(payload.decode("utf-8"))
        report = verify_fields(args.checkpoint, manifest, args.source_npz)
        report["reference_file_sha256"] = hashlib.sha256(payload).hexdigest()
    except (OSError, ValueError) as exc:
        report = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"],
                  "qualification_scope": "fixed_query_checkpoint_parity", "endpoint_qualified": False}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        try:
            with Path(args.out).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded + "\n")
        except OSError as exc:
            report["passed"] = False
            report["errors"].append(f"cannot create new report: {exc}")
            encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
