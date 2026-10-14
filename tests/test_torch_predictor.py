"""TorchPredictor against a REAL params-v1 checkpoint.

No real checkpoint is committed, so a session fixture makes one the way Avneet
does: a synthetic DrivAerNet-shaped parameter table (16 contract columns, body
style, three cat_ indicators, Average Cd) run through src.train.train.train with
configs/drivaernet.yaml, shrunk to a tiny MLP and a few epochs on CPU. The
checkpoint is therefore byte-for-byte the format train.py saves, including the
OOD detector fitted on train and calibrated on val.

The reference computation below is written straight from docs/HANDOFF.md
section 3, independently of src/service/predictor.py, so a column-order or
denormalisation bug in the service shows up as a mismatch.
"""

from __future__ import annotations

import copy
import hashlib
import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("yaml")

from fastapi.testclient import TestClient  # noqa: E402

from src.contracts import (  # noqa: E402
    BODY_STYLES, Design, Evaluation, OOD_SCORE_THRESHOLD, PARAM_BOUNDS, PRESSURE_REGIONS,
    SCHEMA_VERSION, UNCERTAINTY_THRESHOLD, default_params, is_in_distribution,
)
from src.service.app import create_app  # noqa: E402
from src.service.predictor import (  # noqa: E402
    CheckpointError, StubPredictor, TorchPredictor, load_predictor,
)
from src.service.runlog import RunLog  # noqa: E402

STYLE_PREFIX = {"fastback": "F", "notchback": "N", "estateback": "E"}


def _synthetic_table(path, n_per_style=120, seed=0):
    """DrivAerNet-shaped CSV: morph deltas around the zero baseline, Cd linear-ish."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0, len(PARAM_BOUNDS))
    offset = {"fastback": 0.0, "notchback": 0.010, "estateback": 0.025}
    rows = []
    for style in BODY_STYLES:
        for i in range(n_per_style):
            params = {name: float(np.clip(rng.normal(0.0, (hi - lo) / 10.0), lo, hi))
                      for name, (lo, hi) in PARAM_BOUNDS.items()}
            scaled = np.array([params[name] / (hi - lo) for name, (lo, hi) in PARAM_BOUNDS.items()])
            cd = 0.28 + offset[style] + 0.03 * float(weights @ scaled) + rng.normal(0.0, 0.004)
            rows.append({"Experiment": f"{STYLE_PREFIX[style]}_S_WWC_WM_{i:03d}",
                         "Category": style, **params,
                         **{f"cat_{s}": float(s == style) for s in BODY_STYLES},
                         "Average Cd": cd})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """(checkpoint path, loaded checkpoint dict) produced by src.train.train."""
    from src.data.dataset import load_config
    from src.train.train import train

    root = tmp_path_factory.mktemp("torch-ckpt")
    _synthetic_table(root / "parametric.csv")
    cfg = load_config("configs/drivaernet.yaml")
    cfg["data"]["labels_csv"] = str(root / "parametric.csv")
    cfg["model"].update(hidden_dim=32, n_layers=2, dropout=0.0)
    cfg["train"].update(epochs=40, variance_epochs=40, patience=1000, variance_patience=1000,
                        lr=3e-3, variance_lr=3e-3, batch_size=64, out_dir=str(root / "runs"))
    # KNOWN GAP (Avneet): src/train/train.py hardcodes SCHEMA_VERSION = "1.0.0"
    # while training the 2.0.0 parameter space, so as of 47ab1c7 every real
    # checkpoint is stamped 1.0.0 and TorchPredictor rightly refuses it. Patch
    # the stamp here to what train.py should import from src.contracts.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("src.train.train.SCHEMA_VERSION", SCHEMA_VERSION)
        result = train(cfg, device_str="cpu")
    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["preprocess"]["version"] == "params-v1"
    assert ckpt["ood"] is not None
    return result["checkpoint"], ckpt


@pytest.fixture(scope="session")
def predictor(trained):
    return TorchPredictor(trained[0])


def _save(tmp_path, ckpt, name="variant.pt"):
    path = tmp_path / name
    torch.save(ckpt, path)
    return str(path)


def _reference(ckpt, design):
    """docs/HANDOFF.md section 3, transcribed. Returns (cd, sigma, score)."""
    from src.train.model import build_model, split_prediction
    from src.train.ood import OODDetector

    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm, spec = ckpt["norm_stats"], ckpt["preprocess"]
    ood = OODDetector.from_dict(ckpt["ood"]) if ckpt.get("ood") else None

    cols = spec["param_columns"]
    x = np.array([[design.params[c] for c in cols if not c.startswith("cat_")]
                  + [1.0 if c == f"cat_{design.body_style}" else 0.0
                     for c in cols if c.startswith("cat_")]], dtype=np.float32)
    x = (x - np.array(norm["param_mean"])) / np.array(norm["param_std"])
    with torch.no_grad():
        # float32 for the model: HANDOFF's float64 row would not match the weights.
        mu, sigma = split_prediction(model(torch.from_numpy(x.astype(np.float32))))
    cd = float(mu[0]) * norm["target_std"] + norm["target_mean"]
    sigma = float(sigma[0]) * norm["target_std"]
    score = ood.score_one(x) if ood else None
    return cd, sigma, score


def _random_designs(n, seed=1):
    rng = np.random.default_rng(seed)
    designs = [Design(default_params(), style) for style in BODY_STYLES]
    for i in range(n):
        # Drawn like the training table, so these are ordinary designs.
        params = {name: float(np.clip(rng.normal(0.0, (hi - lo) / 10.0), lo, hi))
                  for name, (lo, hi) in PARAM_BOUNDS.items()}
        designs.append(Design(params, BODY_STYLES[i % len(BODY_STYLES)]))
    return designs


# --------------------------------------------------------------------------
# Reproducing the handoff recipe
# --------------------------------------------------------------------------


def test_fixture_checkpoint_has_the_columns_the_contract_needs(trained):
    cols = trained[1]["preprocess"]["param_columns"]
    assert cols[:16] == list(PARAM_BOUNDS)
    assert cols[16:] == [f"cat_{style}" for style in BODY_STYLES]


@pytest.mark.parametrize("index", range(9))
def test_matches_independent_handoff_computation(trained, predictor, index):
    design = _random_designs(6)[index]
    cd, sigma, score = _reference(trained[1], design)
    result = predictor.evaluate(design)
    assert result.cd == pytest.approx(cd, rel=1e-5, abs=1e-7)
    assert result.uncertainty == pytest.approx(sigma, rel=1e-4, abs=1e-8)
    assert result.ood_score == pytest.approx(score, rel=1e-5)
    expected_ok, _ = is_in_distribution(sigma, score, design.body_style,
                                        ood_threshold=trained[1]["ood"]["threshold"])
    assert result.in_distribution is expected_ok
    assert 0.1 < result.cd < 0.5 and result.uncertainty > 0


def test_evaluation_metadata(trained, predictor):
    path, ckpt = trained
    result = predictor.evaluate(Design(default_params()))
    assert result.model_version == ckpt["version"]
    assert result.latency_ms > 0
    assert result.cl == 0.0 and result.pressure_summary == {}
    with open(path, "rb") as handle:
        assert predictor.checkpoint_sha256 == hashlib.sha256(handle.read()).hexdigest()[:12]
    assert predictor.output_provenance == {
        "cd": "model", "uncertainty": "model", "ood_score": "model",
        "cl": "placeholder", "pressure_summary": "unavailable",
    }


def test_model_version_falls_back_to_file_hash(trained, tmp_path):
    ckpt = copy.deepcopy(trained[1])
    del ckpt["version"]
    loaded = TorchPredictor(_save(tmp_path, ckpt))
    assert loaded.model_version == f"sha256-{loaded.checkpoint_sha256}"


# --------------------------------------------------------------------------
# Column order and body style
# --------------------------------------------------------------------------


def _permuted(ckpt, perm, move_weights=True):
    """The same model with its input columns stored in a different order."""
    out = copy.deepcopy(ckpt)
    spec, norm = out["preprocess"], out["norm_stats"]
    spec["param_columns"] = [spec["param_columns"][i] for i in perm]
    norm["param_mean"] = [norm["param_mean"][i] for i in perm]
    norm["param_std"] = [norm["param_std"][i] for i in perm]
    if move_weights:
        out["state_dict"]["encoder.0.weight"] = out["state_dict"]["encoder.0.weight"][:, perm].clone()
    out["ood"]["mean"] = [out["ood"]["mean"][i] for i in perm]
    cov_inv = np.asarray(out["ood"]["cov_inv"])
    out["ood"]["cov_inv"] = cov_inv[np.ix_(perm, perm)].tolist()
    return out


def test_column_order_comes_from_the_checkpoint(trained, predictor, tmp_path):
    assert "encoder.0.weight" in trained[1]["state_dict"]
    perm = list(range(19))[::-1]
    reordered = TorchPredictor(_save(tmp_path, _permuted(trained[1], perm), "reordered.pt"))
    # Same model, columns listed in another order, weights moved to match: the
    # answer must not change. A loader that ignored param_columns would differ.
    wrong = TorchPredictor(_save(tmp_path, _permuted(trained[1], perm, move_weights=False),
                                 "wrong.pt"))
    for design in _random_designs(4):
        a, b = predictor.evaluate(design), reordered.evaluate(design)
        assert b.cd == pytest.approx(a.cd, rel=1e-5)
        assert b.uncertainty == pytest.approx(a.uncertainty, rel=1e-4)
        assert b.ood_score == pytest.approx(a.ood_score, rel=1e-6)
    assert any(abs(wrong.evaluate(d).cd - predictor.evaluate(d).cd) > 1e-4
               for d in _random_designs(4))


def test_swapping_parameter_values_changes_the_prediction(predictor):
    base = default_params()
    a = dict(base, A_Car_Length=80.0, A_Car_Width=-40.0)
    b = dict(base, A_Car_Length=-40.0, A_Car_Width=80.0)
    assert predictor.evaluate(Design(a)).cd != pytest.approx(predictor.evaluate(Design(b)).cd,
                                                             abs=1e-6)


def test_body_style_one_hot_is_applied(predictor):
    cols = predictor.param_columns
    by_style = {}
    for style in BODY_STYLES:
        design = Design(default_params(), style)
        x = predictor.model_input(design)[0] * predictor._param_std + predictor._param_mean
        for other in BODY_STYLES:
            assert x[cols.index(f"cat_{other}")] == pytest.approx(float(other == style))
        by_style[style] = predictor.evaluate(design).cd
    assert len({round(cd, 6) for cd in by_style.values()}) == 3


# --------------------------------------------------------------------------
# The gate: calibrated OOD threshold, extreme corners
# --------------------------------------------------------------------------


def test_uses_the_checkpoint_calibrated_ood_threshold(trained, predictor, monkeypatch):
    calibrated = trained[1]["ood"]["threshold"]
    assert predictor.ood_threshold == calibrated
    assert predictor.uncertainty_threshold == UNCERTAINTY_THRESHOLD
    assert calibrated > OOD_SCORE_THRESHOLD + 0.5, "fixture must separate the two thresholds"

    design = Design(default_params())
    assert predictor.evaluate(design).uncertainty < UNCERTAINTY_THRESHOLD
    between = (OOD_SCORE_THRESHOLD + calibrated) / 2.0
    monkeypatch.setattr(predictor._ood, "score_one", lambda x: between)
    result = predictor.evaluate(design)
    assert result.ood_score == between
    assert result.in_distribution is True  # the contract default 3.0 would have rejected it
    monkeypatch.setattr(predictor._ood, "score_one", lambda x: calibrated + 0.1)
    assert predictor.evaluate(design).in_distribution is False


def test_real_designs_between_the_thresholds_pass(trained, predictor):
    calibrated = trained[1]["ood"]["threshold"]
    hits = [r for r in map(predictor.evaluate, _random_designs(30))
            if OOD_SCORE_THRESHOLD < r.ood_score < calibrated
            and r.uncertainty < UNCERTAINTY_THRESHOLD]
    assert hits, "expected ordinary designs scoring above 3.0 in 19 standardised dims"
    assert all(r.in_distribution for r in hits)


@pytest.mark.parametrize("corner", ["all_high", "all_low", "alternating"])
def test_jointly_extreme_in_bounds_corner_is_out_of_distribution(trained, predictor, corner):
    names = list(PARAM_BOUNDS)
    pick = {"all_high": lambda i: 1, "all_low": lambda i: 0, "alternating": lambda i: i % 2}[corner]
    params = {name: PARAM_BOUNDS[name][pick(i)] for i, name in enumerate(names)}
    result = predictor.evaluate(Design(params))  # every value legal: no ValueError
    assert result.ood_score > trained[1]["ood"]["threshold"]
    assert result.in_distribution is False


# --------------------------------------------------------------------------
# Invalid designs and unservable checkpoints
# --------------------------------------------------------------------------


def test_invalid_designs_raise_value_error(predictor):
    params = default_params()
    params["B_Ramp_Angle"] = 99.0
    with pytest.raises(ValueError, match="outside"):
        predictor.evaluate(Design(params))
    params = default_params()
    del params["A_Car_Width"]
    with pytest.raises(ValueError, match="missing"):
        predictor.evaluate(Design(params))
    with pytest.raises(ValueError, match="body style"):
        predictor.evaluate(Design(default_params(), "suv"))


def test_missing_file_fails_at_load(tmp_path):
    missing = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        TorchPredictor(missing)
    with pytest.raises(FileNotFoundError):
        load_predictor(missing)


def test_garbage_and_non_dict_files_fail_at_load(tmp_path):
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="not a readable"):
        TorchPredictor(str(garbage))
    with pytest.raises(CheckpointError, match="not a checkpoint dict"):
        TorchPredictor(_save(tmp_path, [1, 2, 3]))


def _variant(ckpt, **changes):
    out = copy.deepcopy(ckpt)
    for key, value in changes.items():
        if key == "preprocess_version":
            out["preprocess"]["version"] = value
        elif key == "columns":
            out["preprocess"]["param_columns"] = value
        else:
            out[key] = value
    return out


@pytest.mark.parametrize("changes, message", [
    ({"preprocess_version": "pointcloud-v1"}, "point-cloud"),
    ({"preprocess_version": "params-v9"}, "unknown preprocess version"),
    ({"schema_version": "1.0.0"}, "schema"),
    ({"columns": ["ride_height"] + list(PARAM_BOUNDS)[1:] + [f"cat_{s}" for s in BODY_STYLES]},
     "neither contract parameters"),
    ({"columns": list(PARAM_BOUNDS) + ["cat_fastback", "cat_notchback", "cat_suv"]},
     "neither contract parameters"),
    ({"columns": list(PARAM_BOUNDS) + ["cat_fastback", "cat_notchback", "cat_notchback"]},
     "duplicate"),
])
def test_unservable_checkpoints_fail_at_load(trained, tmp_path, changes, message):
    with pytest.raises(CheckpointError, match=message):
        TorchPredictor(_save(tmp_path, _variant(trained[1], **changes)))


def test_feature_space_detector_is_refused_for_a_parameter_model(trained, tmp_path):
    ckpt = copy.deepcopy(trained[1])
    assert ckpt["ood"].get("space", "parameters") == "parameters"
    ckpt["ood"]["space"] = "features"
    with pytest.raises(CheckpointError, match="measures 'features'"):
        TorchPredictor(_save(tmp_path, ckpt))


def test_mismatched_weights_fail_at_load(trained, tmp_path):
    ckpt = copy.deepcopy(trained[1])
    del ckpt["state_dict"]["mean_head.weight"]
    with pytest.raises(CheckpointError, match="cannot rebuild the model"):
        TorchPredictor(_save(tmp_path, ckpt))


def test_missing_required_key_fails_at_load(trained, tmp_path):
    ckpt = copy.deepcopy(trained[1])
    del ckpt["preprocess"]
    with pytest.raises(CheckpointError, match="preprocess"):
        TorchPredictor(_save(tmp_path, ckpt))


def test_bad_checkpoint_fails_startup_without_stub_fallback(trained, tmp_path, monkeypatch):
    bad = _save(tmp_path, _variant(trained[1], preprocess_version="pointcloud-v1"))
    monkeypatch.setenv("AERO_CHECKPOINT", bad)
    with pytest.raises(CheckpointError):
        with TestClient(create_app(runlog=RunLog(str(tmp_path / "logs")))):
            pass


def test_checkpoint_without_detector_reports_ood_unavailable(trained, tmp_path):
    loaded = TorchPredictor(_save(tmp_path, _variant(trained[1], ood=None)))
    result = loaded.evaluate(Design(default_params()))
    assert result.ood_score is None
    assert loaded.ood_threshold is None
    assert loaded.output_provenance["ood_score"] == "unavailable"
    with TestClient(create_app(loaded, RunLog(str(tmp_path / "logs")))) as client:
        assert client.get("/schema").json()["ood_score_threshold"] == OOD_SCORE_THRESHOLD


# --------------------------------------------------------------------------
# Through the service
# --------------------------------------------------------------------------


@pytest.fixture
def torch_client(predictor, tmp_path):
    with TestClient(create_app(predictor, RunLog(str(tmp_path / "logs")))) as client:
        yield client


def test_health_reports_the_real_model(trained, predictor, torch_client):
    health = torch_client.get("/health").json()
    assert health["predictor"] == "torch"
    assert health["synthetic"] is False
    assert health["schema_version"] == SCHEMA_VERSION
    assert health["model_version"] == trained[1]["version"]
    assert health["checkpoint_sha256"] == predictor.checkpoint_sha256
    assert len(health["checkpoint_sha256"]) == 12
    assert health["output_provenance"]["cl"] == "placeholder"
    assert health["output_provenance"]["pressure_summary"] == "unavailable"
    assert health["output_provenance"]["cd"] == "model"


def test_schema_reports_the_calibrated_thresholds(trained, torch_client):
    schema = torch_client.get("/schema").json()
    assert schema["ood_score_threshold"] == pytest.approx(trained[1]["ood"]["threshold"])
    assert schema["uncertainty_threshold"] == UNCERTAINTY_THRESHOLD
    assert schema["defaults"] == default_params()


def test_evaluate_returns_a_valid_evaluation(trained, torch_client):
    response = torch_client.post("/evaluate", json={"params": default_params(),
                                                    "body_style": "notchback"})
    assert response.status_code == 200
    result = Evaluation(**response.json())
    cd, sigma, score = _reference(trained[1], Design(default_params(), "notchback"))
    assert result.cd == pytest.approx(cd, rel=1e-5)
    assert result.cl == 0.0
    assert result.pressure_summary == {}
    assert result.model_version == trained[1]["version"]
    assert math.isfinite(result.ood_score)
    bad = default_params()
    bad["B_Ramp_Angle"] = 99.0
    assert torch_client.post("/evaluate", json={"params": bad}).status_code == 422


def test_compare_lists_every_region_unavailable(torch_client):
    a = default_params()
    b = dict(a, A_Car_Roof_Height=-30.0)
    response = torch_client.post("/compare", json={"design_a": {"params": a},
                                                   "design_b": {"params": b}})
    assert response.status_code == 200
    body = response.json()
    assert body["unavailable_pressure_regions"] == sorted(PRESSURE_REGIONS)
    assert body["delta"]["pressure_summary"] == {}
    assert body["delta"]["cl"] == 0.0
    assert body["delta"]["cd"] == pytest.approx(body["design_b"]["cd"] - body["design_a"]["cd"])


def test_compare_gates_with_the_calibrated_threshold(trained, predictor, torch_client, monkeypatch):
    # A score between the contract fallback (3.0) and the calibrated threshold
    # is trusted by /evaluate, so /compare must agree rather than reject it.
    between = (OOD_SCORE_THRESHOLD + trained[1]["ood"]["threshold"]) / 2.0
    monkeypatch.setattr(predictor._ood, "score_one", lambda x: between)
    body = torch_client.post("/compare", json={"design_a": {"params": default_params()},
                                               "design_b": {"params": default_params()}}).json()
    assert body["design_a"]["in_distribution"] is True
    assert body["trusted"] is True and body["reject_reasons"]["design_a"] is None


def test_configured_checkpoint_is_served(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("AERO_CHECKPOINT", trained[0])
    with TestClient(create_app(runlog=RunLog(str(tmp_path / "logs")))) as client:
        assert client.get("/health").json()["predictor"] == "torch"


def test_stub_provenance_is_synthetic():
    assert set(StubPredictor.output_provenance.values()) == {"synthetic"}
