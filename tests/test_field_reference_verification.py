"""Independent field references require the original cloud, never an invented one."""

import copy
import hashlib
import json
import zipfile

import numpy as np
import pytest
import torch

from src.contracts import PRESSURE_REGIONS
from src.service.verify_fields import main, verify_fields


@pytest.fixture(scope="module")
def reference_pack(tmp_path_factory):
    from src.train.reference import build_fields
    from src.train.train_fields import train_fields

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("independent-field-reference")
    data = root / "data" / "drivaerml"
    (data / "fields").mkdir(parents=True)
    rng = np.random.default_rng(26)
    for run in range(1, 7):
        xyz = rng.normal(size=(64, 3)).astype(np.float32)
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        cp = (-0.2 + xyz[:, 0] * 0.5 + run * 0.01).astype(np.float32)
        area = np.full(64, 0.01, np.float32)
        meta = {"run": run, "version": "fields-v1", "regions": list(PRESSURE_REGIONS),
                "a_ref_m2": 2., "total_area_m2": 1., "n_cells_total": 64}
        np.savez_compressed(data / "fields" / f"run_{run}.npz", query_xyz=xyz,
                            query_normal=xyz, query_area=area, cp=cp,
                            pprime2=np.abs(cp), region=np.zeros(64, np.int16),
                            cloud_xyz=xyz, meta=json.dumps(meta))
    cfg = {"run_name": "fields", "seed": 0,
           "data": {"root": str(data), "val_runs": 1, "test_runs": 1,
                    "n_query": 32, "n_cloud": 16},
           "model": {"latent_dim": 8, "local_k": 4, "local_dim": 8, "n_freq": 8,
                     "hidden": 8, "n_layers": 1, "heteroscedastic": True},
           "train": {"epochs": 1, "variance_epochs": 1, "batch_size": 2,
                     "lr": 0.001, "weight_decay": 0., "out_dir": str(root / "runs")}}
    artifact = train_fields(cfg, device_str="cpu")["checkpoint"]
    manifest = build_fields(root, root / "pack", n_points=32)
    source = data / "fields" / f"run_{manifest['sample']['run']}.npz"
    yield root, artifact, source, manifest
    torch.set_num_threads(previous_threads)


def test_exact_source_replays_independent_pack_without_endpoint_claim(reference_pack):
    root, checkpoint, source, manifest = reference_pack
    report = verify_fields(checkpoint, manifest, str(source))
    assert report["passed"], report["errors"]
    assert report["qualification_scope"] == "fixed_query_checkpoint_parity"
    assert report["endpoint_qualified"] is False
    assert report["geometry_preprocessing_qualified"] is False
    assert report["region_classifier_qualified"] is False
    assert report["uncertainty_calibration_qualified"] is False
    assert report["source_authenticated_by_reference_hash"] is False
    assert report["source_npz_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["provenance"]["checkpoint_sha256"] == manifest["checkpoint"]["sha256"]
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(64, size=32, replace=False))
    cloud = np.sort(rng.choice(64, size=16, replace=False))
    assert report["selection"]["query_indices"] == take.tolist()
    assert report["selection"]["cloud_indices"] == cloud.tolist()
    assert report["results"]["cp"]["max_absolute_error"] <= 1e-4
    assert report["results"]["sigma"]["max_absolute_error"] <= 1e-4
    # Generator's descriptive performance claims are neither recomputed nor
    # silently promoted into a Track B certification.
    assert "coefficient_claims_supported" not in report
    json.dumps(report, allow_nan=False)


def test_missing_original_source_is_visible(reference_pack, tmp_path):
    _, checkpoint, _, manifest = reference_pack
    for source in (None, str(tmp_path / "missing.npz")):
        report = verify_fields(checkpoint, manifest, source)
        assert not report["passed"] and report["errors"]
        assert report["provenance"] is None


@pytest.mark.parametrize("mutation,message", [
    (lambda p: p.update(kind="scalar_reference"), "kind"),
    (lambda p: p.update(tolerance=0.01), "tolerance"),
    (lambda p: p.update(tolerance=float("nan")), "tolerance"),
    (lambda p: p["sample"].update(n_points=257), "n_points"),
    (lambda p: p["sample"]["query_xyz"].pop(), "shape"),
    (lambda p: p["sample"]["query_normal"][0].__setitem__(0, float("nan")), "finite"),
    (lambda p: p["sample"]["expected_sigma"].__setitem__(0, -1.), "positive"),
    (lambda p: p["sample"]["region_index"].__setitem__(0, 999), "region"),
    (lambda p: p["checkpoint"].update(sha256="f" * 64), "SHA-256"),
    (lambda p: p["checkpoint"].update(version="wrong"), "version"),
    (lambda p: p["checkpoint"]["preprocess"].update(n_cloud=8), "preprocess"),
    (lambda p: p["checkpoint"].update(schema_version="1.0.0"), "schema_version"),
    (lambda p: p["conventions"].update(length_units="mm"), "metres"),
    (lambda p: p["conventions"]["region_order"].reverse(), "ordering"),
    (lambda p: p["sample"]["query_xyz"][0].__setitem__(0, 0.), "query_xyz"),
    (lambda p: p["sample"]["expected_cp"].__setitem__(0, 100.), "cp differs"),
    (lambda p: p["sample"]["expected_sigma"].__setitem__(0, 100.), "sigma differs"),
    (lambda p: p["split_ids"]["runs"]["train"].append(p["sample"]["run"]), "overlap"),
])
def test_invalid_or_mismatched_reference_fails(reference_pack, mutation, message):
    _, checkpoint, source, manifest = reference_pack
    modified = copy.deepcopy(manifest)
    mutation(modified)
    report = verify_fields(checkpoint, modified, str(source))
    assert report["passed"] is False
    assert message in " ".join(report["errors"]), report


@pytest.mark.parametrize("kind", ["run", "cloud", "query", "dtype", "metadata", "object", "absent"])
def test_incorrect_source_is_never_substituted(reference_pack, tmp_path, kind):
    _, checkpoint, source, manifest = reference_pack
    with np.load(source, allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    if kind == "run":
        meta = json.loads(str(arrays["meta"]))
        meta["run"] += 1
        arrays["meta"] = json.dumps(meta)
    elif kind == "cloud":
        arrays["cloud_xyz"] = arrays["cloud_xyz"] * 0.1
    elif kind == "query":
        arrays["query_xyz"] = arrays["query_xyz"] * 0.5
    elif kind == "dtype":
        arrays["query_xyz"] = arrays["query_xyz"].astype(np.float64)
    elif kind == "metadata":
        arrays["meta"] = "not json"
    elif kind == "object":
        arrays["cloud_xyz"] = np.array([object()], dtype=object)
    else:
        arrays.pop("cloud_xyz")
    wrong = tmp_path / "wrong.npz"
    np.savez_compressed(wrong, **arrays)
    report = verify_fields(checkpoint, manifest, str(wrong))
    assert not report["passed"], report
    assert report["errors"]


def test_source_decompression_bound_checked_before_loading(reference_pack, tmp_path, monkeypatch):
    import src.service.verify_fields as module

    _, checkpoint, source, manifest = reference_pack
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 1500)
    path = tmp_path / "huge.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cloud_xyz.npy", b"0" * 3000)
    assert path.stat().st_size < 1500
    report = verify_fields(checkpoint, manifest, str(path))
    assert not report["passed"] and "decompressed" in " ".join(report["errors"])


def test_cli_creates_evidence_exclusively_and_exits_nonzero_on_failure(reference_pack, tmp_path, capsys):
    root, checkpoint, source, manifest = reference_pack
    reference = root / "pack" / "field_reference.json"
    out = tmp_path / "report.json"
    args = ["--checkpoint", checkpoint, "--references", str(reference),
            "--source-npz", str(source), "--out", str(out)]
    assert main(args) == 0
    first = out.read_bytes()
    assert json.loads(first)["reference_file_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert main(args) == 1
    assert out.read_bytes() == first
    assert "cannot create new report" in capsys.readouterr().out
    args[args.index(str(source))] = str(tmp_path / "missing.npz")
    args[-1] = str(tmp_path / "failed.json")
    assert main(args) == 1
    assert not json.loads((tmp_path / "failed.json").read_text())["passed"]


def test_cli_invalid_reference_and_input_size_are_visible(tmp_path, monkeypatch, capsys):
    import src.service.verify_fields as module

    path = tmp_path / "bad.json"
    path.write_text("not json")
    args = ["--checkpoint", "missing.pt", "--references", str(path), "--source-npz", "missing.npz"]
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    monkeypatch.setattr(module, "MAX_REFERENCE_BYTES", 2)
    assert main(args) == 1
    assert "byte limit" in capsys.readouterr().out
