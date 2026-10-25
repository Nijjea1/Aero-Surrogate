"""Stage-2 grading (src/eval/fields.py) on synthetic DrivAerML-shaped arrays."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval import fields

REGIONS = ["front", "roof", "rear"]


def _write_run(root, run, n=600, seed=None):
    rng = np.random.default_rng(run if seed is None else seed)
    xyz = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    normal = rng.normal(size=(n, 3)).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    cp = (0.6 * normal[:, 0] - 0.3 * xyz[:, 2]).astype(np.float32)   # a smooth, learnable field
    arrays = {
        "query_xyz": xyz, "query_normal": normal,
        "query_area": rng.uniform(0.5, 1.5, n).astype(np.float32),
        "cp": cp, "pprime2": np.zeros(n, np.float32), "shear": np.zeros((n, 3), np.float32),
        "region": (np.arange(n) % len(REGIONS)).astype(np.int16),
        "cloud_xyz": xyz[:200],
    }
    meta = {"run": run, "regions": REGIONS, "a_ref_m2": 2.0, "total_area_m2": float(arrays["query_area"].sum()) * 4}
    (root / "fields").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "fields" / f"run_{run}.npz", **arrays, meta=json.dumps(meta))


@pytest.fixture()
def root(tmp_path):
    for run in (1, 2, 3, 4):
        _write_run(tmp_path, run)
    return tmp_path


def test_a_perfect_field_scores_perfectly_and_beats_the_references(root):
    test, train = fields.load_runs(root, [4]), fields.load_runs(root, [1, 2, 3])
    perfect = lambda a: (a["cp"].astype(np.float64), np.full(len(a["cp"]), 0.05))  # noqa: E731
    out = fields.grade(test, train, perfect)
    assert out["available"] and out["test_runs"] == [4] and out["n_train_runs"] == 3
    assert out["overall"]["r2"] == pytest.approx(1.0) and out["overall"]["mae"] == 0
    assert out["overall"]["coverage_1sigma"] == 1.0
    assert out["references"]["mean"]["r2"] < 0.05
    assert out["references"]["nearest_training_car"]["r2"] < out["overall"]["r2"]
    assert set(out["per_region"]) == set(REGIONS)
    for r in out["per_region"].values():
        assert r["cp_pred"] == pytest.approx(r["cp_true"])
    c = out["integrated_cd"]
    assert c["mae"] == pytest.approx(0) and c["pred_mean"] == pytest.approx(c["true_mean"])


def test_pressure_drag_integrates_and_scales_to_the_whole_surface(root):
    (_, arrays, meta), = fields.load_runs(root, [1])
    cp = arrays["cp"].astype(np.float64)
    expected = float((cp * -arrays["query_normal"][:, 0] * arrays["query_area"]).sum()) / 2.0 * 4
    assert fields._pressure_drag(cp, arrays, meta) == pytest.approx(expected, rel=1e-5)


def test_a_constant_field_scores_like_the_mean_reference(root):
    test, train = fields.load_runs(root, [4]), fields.load_runs(root, [1, 2, 3])
    mean = float(np.concatenate([a["cp"] for _, a, _ in train]).mean())
    out = fields.grade(test, train, lambda a: (np.full(len(a["cp"]), mean), np.full(len(a["cp"]), 1.0)))
    assert out["overall"]["r2"] == pytest.approx(out["references"]["mean"]["r2"], abs=1e-6)


def test_missing_checkpoint_or_data_is_reported_not_raised(root, tmp_path):
    out = fields.grade_checkpoint(tmp_path / "nope.pt", root)
    assert out == {"available": False, "reason": out["reason"]} and "could not be loaded" in out["reason"]


def test_a_real_field_checkpoint_is_graded_on_its_own_test_cars(root, tmp_path):
    torch = pytest.importorskip("torch")
    from src.train.train_fields import build_model_from_cfg

    cfg = {"model": {"latent_dim": 16, "local_k": [4], "local_dim": 8, "n_freq": 2, "hidden": 16, "n_layers": 2,
                     "heteroscedastic": True}}
    model = build_model_from_cfg(cfg)
    model.freeze_mean_path()
    path = tmp_path / "fields.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg, "norm_stats": {"cp_mean": 0.0, "cp_std": 1.0},
                "preprocess": {"n_cloud": 64}, "splits": {"train": [1, 2], "val": [3], "test": [4]},
                "version": "fixture", "test_metrics": {"cp_r2": 0.1, "per_region": {}}}, path)
    out = fields.grade_checkpoint(path, root)
    assert out["available"], out.get("reason")
    assert out["test_runs"] == [4] and out["n_points"] == 600
    assert np.isfinite(out["overall"]["mae"]) and 0 <= out["overall"]["coverage_1sigma"] <= 1
    assert out["checkpoint"]["version"] == "fixture" and "per_region" not in out["checkpoint"]["self_reported_test_metrics"]

    assert out["checkpoint"]["sha256"] == fields._sha256(path)
    assert out["evaluation"]["splits"] == {"train": [1, 2], "val": [3], "test": [4]}
    assert {row["run"] for row in out["evaluation"]["arrays"]} == {1, 2, 4}
    assert out["evaluation"]["reference"]["name"] == "pooled nearest-training-point"
    _, arrays, _ = fields.load_runs(root, [4])[0]
    small, _ = fields.checkpoint_predictor(path, chunk=37)
    large, _ = fields.checkpoint_predictor(path, chunk=600)
    for first, second in zip(small(arrays), large(arrays)):
        np.testing.assert_allclose(first, second, rtol=2e-5, atol=2e-6)

    # A test car with no extracted arrays is reported, not raised.
    (root / "fields" / "run_4.npz").unlink()
    missing = fields.grade_checkpoint(path, root)
    assert not missing["available"] and "field data missing" in missing["reason"]


def test_load_runs_accepts_dataset_root_or_fields_directory(root):
    for dataset, direct in zip(fields.load_runs(root, [1, 4]), fields.load_runs(root / "fields", [1, 4])):
        assert dataset[0] == direct[0] and dataset[2] == direct[2]
        np.testing.assert_array_equal(dataset[1]["cp"], direct[1]["cp"])


def _available_block():
    return {"available": True, "overall": {"r2": 0.5, "mae": 0.1, "coverage_1sigma": 0.7},
            "references": {}, "n_points": 10, "test_runs": [4],
            "integrated_cd": {"true_mean": 0.2, "pred_mean": 0.19, "mae": 0.01}}


@pytest.mark.parametrize("existing", [False, True])
def test_cli_creates_output_or_preserves_other_sections(tmp_path, monkeypatch, existing):
    path = tmp_path / "new folder" / "metrics.json"
    if existing:
        path.parent.mkdir()
        path.write_text(json.dumps({"headline": {"r2": 0.8}, "field": {"old": True}}))
    calls = []
    def fake_grade(*args, **kwargs):
        calls.append((args, kwargs))
        return _available_block()
    monkeypatch.setattr(fields, "grade_checkpoint", fake_grade)
    assert fields.main(["--checkpoint", "fixture.pt", "--metrics", str(path),
                        "--chunk-size", "64", "--reference-workers", "2"]) == 0
    out = json.loads(path.read_text())
    assert out["field"] == _available_block()
    assert out.get("headline") == ({"r2": 0.8} if existing else None)
    assert calls[0][1] == {"chunk_size": 64, "reference_workers": 2}


@pytest.mark.parametrize("content", ['{broken', '[]', 'null'])
def test_cli_refuses_invalid_existing_output_without_overwriting(tmp_path, monkeypatch, content, capsys):
    path = tmp_path / "metrics.json"
    path.write_text(content)
    def unexpected(*args, **kwargs):
        pytest.fail("invalid output must be rejected before grading")
    monkeypatch.setattr(fields, "grade_checkpoint", unexpected)
    assert fields.main(["--checkpoint", "fixture.pt", "--metrics", str(path)]) == 2
    assert path.read_text() == content
    assert "cannot merge field metrics" in capsys.readouterr().out


def test_cli_missing_checkpoint_writes_explicit_unavailable_block(tmp_path):
    path = tmp_path / "new" / "metrics.json"
    assert fields.main(["--checkpoint", str(tmp_path / "missing.pt"), "--metrics", str(path)]) == 1
    out = json.loads(path.read_text())
    assert out["field"]["available"] is False
    assert "could not be loaded" in out["field"]["reason"]
    assert "overall" not in out["field"] and "references" not in out["field"]


@pytest.mark.parametrize("option", ["--chunk-size", "--reference-workers"])
def test_cli_rejects_unbounded_or_empty_worker_arguments(option):
    with pytest.raises(SystemExit) as exc:
        fields.main(["--checkpoint", "unused.pt", option, "0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("corruption", ["not_npz", "missing_array", "nonfinite"])
def test_corrupt_field_arrays_are_explicitly_unavailable(root, tmp_path, monkeypatch, corruption):
    if corruption == "not_npz":
        (root / "fields" / "run_4.npz").write_bytes(b"not a numpy archive")
    else:
        _, arrays, meta = fields.load_runs(root, [4])[0]
        if corruption == "missing_array":
            del arrays["query_normal"]
        else:
            arrays["cp"][0] = np.nan
        np.savez_compressed(root / "fields" / "run_4.npz", **arrays, meta=json.dumps(meta))
    monkeypatch.setattr(fields, "checkpoint_predictor", lambda *args, **kwargs:
                        (lambda arrays: pytest.fail("corrupt input must not be graded"),
                         {"splits": {"train": [1], "test": [4]}}))
    out = fields.grade_checkpoint(tmp_path / "fixture.pt", root)
    assert out["available"] is False and "field data invalid" in out["reason"]
    assert "overall" not in out


@pytest.mark.parametrize("splits", [{"train": [1], "test": [1]}, {"train": [1], "test": [4, 4]},
                                    {"train": [1], "test": "4"}])
def test_invalid_recorded_splits_are_not_graded(tmp_path, monkeypatch, splits):
    monkeypatch.setattr(fields, "checkpoint_predictor", lambda *args, **kwargs: (None, {"splits": splits}))
    out = fields.grade_checkpoint(tmp_path / "fixture.pt", tmp_path)
    assert out["available"] is False and "split" in out["reason"]
