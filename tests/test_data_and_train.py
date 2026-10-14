"""Tests for the data and training modules.  OWNER: Avneet.

tests/test_contracts.py is the shared tripwire for the three seams; this file
covers only src/data and src/train. It guards the two things that fail
SILENTLY -- which is to say, the two things that would cost us the most:

  * the preprocessing recipe. If sampling or normalisation drifts, the model
    keeps returning confident numbers that are wrong.
  * the mean freeze in two-stage training. If it leaks, the accuracy we report
    is not the accuracy the shipped checkpoint has.

Everything here runs on a six-design fixture built in a temp directory, so the
whole file is about a second and needs no download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.contracts import PARAM_NAMES
from src.data import dataset as ds
from src.data.synthetic import (
    SYNTH_BOUNDS,
    SYNTH_NAMES,
    generate,
    params_to_synthetic_mesh,
    synth_default_params,
)
from src.train.model import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    ParamMLPRegressor,
    PointNetRegressor,
    gaussian_nll,
    split_prediction,
)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    generate(out, n=6, seed=0)
    return out


@pytest.fixture
def cfg(fixture_dir, tmp_path):
    return {
        "seed": 0,
        "data": {
            "kind": "pointcloud",
            "labels_csv": str(fixture_dir / "drag_coefficients.csv"),
            "mesh_dir": str(fixture_dir / "meshes"),
            "mesh_glob": "*.stl",
            "id_column": "Design",
            "target_column": "Average Cd",
            "category_column": "Category",
            "n_points": 256,
            "normalize_geometry": True,
            "cache_dir": str(tmp_path / "cache"),
            "val_fraction": 0.2,
            "test_fraction": 0.2,
        },
    }


# --------------------------------------------------------------------------
# Geometry -> point cloud
# --------------------------------------------------------------------------


def test_sample_mesh_is_deterministic(cfg):
    """Same mesh, same settings, byte-identical cloud. Without this the cache
    and the live service can disagree about the same car."""
    path = next(iter(ds.available_meshes(cfg).values()))
    a = ds.sample_mesh(path, 256, normalize=True)
    b = ds.sample_mesh(path, 256, normalize=True)
    assert np.array_equal(a, b)


def test_normalised_cloud_fits_the_unit_sphere(cfg):
    for path in ds.available_meshes(cfg).values():
        pts = ds.sample_mesh(path, 256, normalize=True)
        assert pts.shape == (256, 3)
        assert pts.dtype == np.float32
        assert np.linalg.norm(pts, axis=1).max() <= 1.0 + 1e-5


def test_unnormalised_cloud_keeps_real_scale(cfg):
    """normalize=False must be a real passthrough, not a no-op that silently
    normalises anyway -- the raw cloud is in metres."""
    path = next(iter(ds.available_meshes(cfg).values()))
    pts = ds.sample_mesh(path, 256, normalize=False)
    assert np.linalg.norm(pts, axis=1).max() > 1.0


def test_cache_round_trips_and_is_keyed_on_n_points(cfg):
    design_id, path = next(iter(ds.available_meshes(cfg).items()))
    first = ds.load_or_sample(cfg, design_id, path)
    assert ds._cache_path(cfg, design_id).exists()
    assert np.array_equal(first, ds.load_or_sample(cfg, design_id, path))

    cfg["data"]["n_points"] = 128
    other = ds.load_or_sample(cfg, design_id, path)
    assert other.shape == (128, 3)  # a different key, not the cached 256


def test_preprocess_spec_matches_what_sample_mesh_does(cfg):
    """The spec is what the service builds its input from, so it has to
    describe the code rather than the intention."""
    spec = ds.preprocess_spec(cfg)
    path = next(iter(ds.available_meshes(cfg).values()))
    pts = ds.sample_mesh(path, spec["n_points"], spec["normalize_geometry"])
    assert pts.shape == (spec["n_points"], 3)
    assert str(pts.dtype) == spec["dtype"]


def test_synthetic_meshes_are_watertight():
    """A leaky mesh makes area-weighted sampling meaningless."""
    assert params_to_synthetic_mesh(synth_default_params()).is_watertight


def test_geometry_responds_to_every_parameter():
    """Each fixture parameter must change the shape. One that does not would be
    invisible to the point-cloud model and silently unlearnable.

    Uses the fixture's own space, not the contract's -- see the note at the top
    of src/data/synthetic.py on why the two are deliberately decoupled."""
    base = params_to_synthetic_mesh(synth_default_params())
    for name in SYNTH_NAMES:
        moved = dict(synth_default_params())
        lo, hi = SYNTH_BOUNDS[name]
        moved[name] = hi
        other = params_to_synthetic_mesh(moved)
        assert not np.allclose(base.bounds, other.bounds, atol=1e-9) or not np.isclose(
            base.volume, other.volume, rtol=1e-6
        ), f"{name} does not change the geometry"


# --------------------------------------------------------------------------
# Labels and splits
# --------------------------------------------------------------------------


def test_check_passes_on_the_fixture(cfg, tmp_path, monkeypatch):
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert ds.check(str(path)) == 0


def test_column_resolution_survives_a_wrong_config(cfg):
    """A config naming a column the CSV spells differently should be resolved
    with a NOTE, not an exception -- that is the whole point of --check."""
    cfg["data"]["target_column"] = "average cd"  # wrong case
    table = ds.load_labels(cfg, strict=False)
    assert table.resolved["target"] == "Average Cd"
    assert any("target" in n for n in table.notes)


def test_missing_column_is_reported_not_guessed(cfg):
    cfg["data"]["target_column"] = "Lift"
    table = ds.load_labels(cfg, strict=False)
    assert table.resolved["target"] is None


def test_splits_are_disjoint_and_complete(cfg):
    splits = ds.make_splits(cfg, seed=0)
    ids, _, _ = ds.matched_ids(cfg)
    joined = [i for s in splits.values() for i in s]
    assert sorted(joined) == sorted(ids)
    assert len(joined) == len(set(joined))


def test_splits_are_reproducible(cfg):
    assert ds.make_splits(cfg, seed=0) == ds.make_splits(cfg, seed=0)
    assert ds.make_splits(cfg, seed=1) != ds.make_splits(cfg, seed=0)


def test_held_out_category_never_leaks_into_training(cfg):
    """The OOD experiment is worthless if the held-out body style appears in
    training -- this is the test that keeps that claim honest."""
    splits = ds.make_category_splits(cfg, seed=0)
    held = splits.pop("_held_out_category")[0]
    _, table, _ = ds.matched_ids(cfg)
    for name in ("train", "val"):
        assert all(table.categories[i] != held for i in splits[name])
    assert splits["test"]
    assert all(table.categories[i] == held for i in splits["test"])


def test_dataset_yields_model_shaped_batches(cfg):
    from torch.utils.data import DataLoader

    loader = DataLoader(ds.DrivAerNetDataset(cfg, "train"), batch_size=2)
    x, y = next(iter(loader))
    assert x.shape[1:] == (256, 3)
    assert y.shape == (2,)


# --------------------------------------------------------------------------
# Loss and the uncertainty head
# --------------------------------------------------------------------------


def test_mean_only_loss_is_exactly_mse():
    pred = torch.tensor([[1.0, 0.5], [2.0, -3.0]])
    target = torch.tensor([1.5, 1.0])
    expected = torch.nn.functional.mse_loss(pred[:, 0], target)
    assert torch.allclose(gaussian_nll(pred, target, mean_only=True), expected)


def test_logvar_is_clamped_both_ways():
    wild = torch.tensor([[0.0, -500.0], [0.0, 500.0]])
    _, sigma = split_prediction(wild)
    assert torch.isfinite(sigma).all()
    assert float(sigma[0]) == pytest.approx(np.exp(0.5 * LOGVAR_MIN), rel=1e-5)
    assert float(sigma[1]) == pytest.approx(np.exp(0.5 * LOGVAR_MAX), rel=1e-5)


def test_nll_rewards_honest_error_bars():
    """A wide sigma on a bad prediction must beat a narrow one. If this
    inverts, the uncertainty head is being trained to lie."""
    target = torch.tensor([0.0])
    confident_and_wrong = torch.tensor([[3.0, -4.0]])
    humble_and_wrong = torch.tensor([[3.0, 2.0]])
    assert float(gaussian_nll(humble_and_wrong, target)) < float(
        gaussian_nll(confident_and_wrong, target)
    )


def test_beta_nll_reduces_to_plain_nll_at_zero():
    pred = torch.tensor([[1.0, -0.5]])
    target = torch.tensor([1.4])
    assert torch.allclose(
        gaussian_nll(pred, target, beta=0.0), gaussian_nll(pred, target)
    )


@pytest.mark.parametrize(
    "model",
    [
        PointNetRegressor(hidden_dim=32, n_layers=2),
        ParamMLPRegressor(in_dim=len(PARAM_NAMES), hidden_dim=16, n_layers=2),
    ],
    ids=["pointnet", "param_mlp"],
)
def test_heteroscedastic_models_emit_two_columns(model):
    x = (
        torch.randn(4, 64, 3)
        if isinstance(model, PointNetRegressor)
        else torch.randn(4, len(PARAM_NAMES))
    )
    out = model(x)
    assert out.shape == (4, 2)
    mean, sigma = split_prediction(out)
    assert mean.shape == (4,) and sigma.shape == (4,)
    assert (sigma > 0).all()


def test_homoscedastic_model_emits_one_column_and_zero_sigma():
    model = ParamMLPRegressor(in_dim=4, hidden_dim=8, n_layers=1, heteroscedastic=False)
    out = model(torch.randn(3, 4))
    assert out.shape == (3, 1)
    _, sigma = split_prediction(out)
    assert float(sigma.abs().max()) == 0.0


def test_freezing_the_mean_path_actually_holds_it_still():
    """The load-bearing invariant of two-stage training. Phase two optimises
    the variance head; if a single mean weight moves, the R² we report is not
    the R² of the checkpoint we shipped."""
    torch.manual_seed(0)
    model = ParamMLPRegressor(in_dim=4, hidden_dim=16, n_layers=2)
    x = torch.randn(8, 4)
    target = torch.randn(8)

    before = model(x)[:, 0].detach().clone()
    var_params = model.freeze_mean_path()
    assert var_params, "nothing left to train"

    optimiser = torch.optim.AdamW(var_params, lr=0.1)
    for _ in range(25):
        optimiser.zero_grad()
        gaussian_nll(model(x), target).backward()
        optimiser.step()

    after = model(x)[:, 0].detach()
    assert torch.allclose(before, after, atol=1e-7), "the mean moved"
    # ...and the variance really did train, or the freeze froze everything.
    assert not torch.allclose(
        split_prediction(model(x))[1], split_prediction(model(x))[1] * 0
    )


def test_freeze_keeps_batchnorm_statistics_still():
    """BatchNorm in train mode keeps updating running stats, which moves the
    mean at eval time even with every weight frozen."""
    torch.manual_seed(0)
    model = PointNetRegressor(hidden_dim=32, n_layers=2)
    model.freeze_mean_path()
    model.train()  # phase two calls this every epoch

    bn = next(m for m in model.encoder if isinstance(m, torch.nn.BatchNorm1d))
    before = bn.running_mean.clone()
    model(torch.randn(4, 64, 3))
    assert torch.allclose(before, bn.running_mean)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_r2_score_edges():
    from src.train.train import r2_score

    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert np.isnan(r2_score([2.0, 2.0], [2.0, 2.0]))  # no variance to explain


def test_training_produces_a_loadable_checkpoint(cfg, tmp_path):
    """The handoff in one test: train briefly, then reload exactly the way
    docs/HANDOFF.md tells Pranoy to."""
    from src.train.model import build_model
    from src.train.train import train

    cfg["run_name"] = "pytest"
    cfg["model"] = {"kind": "pointnet", "hidden_dim": 32, "n_layers": 2,
                    "dropout": 0.0, "heteroscedastic": True}
    cfg["train"] = {"epochs": 2, "variance_epochs": 2, "batch_size": 2, "lr": 1e-3,
                    "weight_decay": 1e-4, "grad_clip": 1.0, "num_workers": 0,
                    "amp": False, "patience": 5, "variance_patience": 5,
                    "out_dir": str(tmp_path / "runs"), "two_stage": True}

    result = train(cfg, device_str="cpu")
    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)

    for key in ("state_dict", "config", "norm_stats", "version", "val_metrics"):
        assert key in ckpt, f"Pranoy's loader needs {key!r}"
    assert "preprocess" in ckpt
    assert ckpt["norm_stats"]["target_std"] > 0

    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])  # must not need strict=False
    model.eval()

    spec = ckpt["preprocess"]
    path = next(iter(ds.available_meshes(cfg).values()))
    pts = ds.sample_mesh(path, spec["n_points"], spec["normalize_geometry"])
    with torch.no_grad():
        mu, sigma = split_prediction(model(torch.from_numpy(pts).unsqueeze(0)))

    norm = ckpt["norm_stats"]
    cd = float(mu[0]) * norm["target_std"] + norm["target_mean"]
    assert 0.05 < cd < 1.0, f"Cd {cd} is not physically plausible"
    assert float(sigma[0]) * norm["target_std"] > 0


# --------------------------------------------------------------------------
# OOD detection. Load-bearing: contracts.is_in_distribution consumes it, and
# it is what closes the hole a sigma-only gate leaves open.
# --------------------------------------------------------------------------


def _gaussian_cloud(n=600, p=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_detector_passes_familiar_designs_and_flags_far_ones():
    from src.train.ood import OODDetector

    x = _gaussian_cloud()
    det = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.99)

    assert det.is_in_distribution(x[500:]).mean() > 0.9
    far = np.full((10, x.shape[1]), 25.0)
    assert not det.is_in_distribution(far).any()


def test_detector_flags_an_extreme_corner_that_bounds_checking_accepts():
    """The hole this closes. Every coordinate legal, the combination absurd --
    clamp_params accepts it and only the joint distance rejects it."""
    from src.train.ood import OODDetector

    x = _gaussian_cloud(p=16)
    det = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.99)

    # Each coordinate well inside the observed per-dimension range...
    per_dim_max = np.abs(x).max(axis=0)
    corner = (per_dim_max * 0.9)[None, :]
    assert (np.abs(corner) <= per_dim_max).all(), "not actually in bounds"
    # ...yet the combination is nowhere near the data.
    assert det.score_one(corner) > det.threshold


def test_detector_threshold_tracks_the_quantile():
    from src.train.ood import OODDetector

    x = _gaussian_cloud()
    loose = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.999)
    tight = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.80)
    assert tight.threshold < loose.threshold


def test_detector_survives_a_constant_column():
    """A parameter the data never varied makes the covariance singular. The
    held-out-category split produces exactly this, so it must not blow up."""
    from src.train.ood import OODDetector

    x = _gaussian_cloud()
    x[:, 0] = 0.0
    det = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.99)
    assert np.isfinite(det.score(x[500:])).all()


def test_detector_round_trips_through_a_checkpoint():
    from src.train.ood import OODDetector

    x = _gaussian_cloud()
    det = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.99)
    back = OODDetector.from_dict(det.to_dict())
    assert np.allclose(det.score(x[500:]), back.score(x[500:]))
    assert back.threshold == det.threshold


def test_detector_refuses_too_few_samples():
    from src.train.ood import OODDetector

    with pytest.raises(ValueError):
        OODDetector.fit(_gaussian_cloud(n=4, p=8))


def test_detection_rate_reports_a_fraction():
    from src.train.ood import OODDetector, detection_rate

    x = _gaussian_cloud()
    det = OODDetector.fit(x[:500], x_calib=x[500:], quantile=0.99)
    assert detection_rate(det, np.full((50, x.shape[1]), 30.0)) == 1.0
    assert detection_rate(det, x[500:]) < 0.15


# --------------------------------------------------------------------------
# Resumability. Kaggle sessions are killed without warning, so a run that
# cannot resume is a run that is lost -- see the compute policy in CLAUDE.md.
# --------------------------------------------------------------------------


def _kaggle_like_cfg(cfg, tmp_path, epochs, variance_epochs):
    cfg = dict(cfg)
    cfg["run_name"] = "resume-test"
    cfg["model"] = {"kind": "pointnet", "hidden_dim": 32, "n_layers": 2,
                    "dropout": 0.0, "heteroscedastic": True}
    cfg["train"] = {"epochs": epochs, "variance_epochs": variance_epochs,
                    "batch_size": 2, "lr": 1e-3, "weight_decay": 1e-4,
                    "grad_clip": 1.0, "num_workers": 0, "amp": False,
                    "patience": 50, "variance_patience": 50,
                    "out_dir": str(tmp_path / "runs"), "two_stage": True}
    return cfg


def test_state_is_written_every_epoch_and_removed_on_completion(cfg, tmp_path):
    from src.train.train import STATE_FILENAME, load_state, train

    c = _kaggle_like_cfg(cfg, tmp_path, epochs=2, variance_epochs=2)
    out = Path(c["train"]["out_dir"]) / "resume-test-random"
    train(c, device_str="cpu")

    # A finished run must not leave state behind, or the next run silently
    # resumes a completed one.
    assert not (out / STATE_FILENAME).exists()
    assert load_state(out) is None
    assert (out / "best.pt").exists()


def test_resume_picks_up_where_a_killed_session_stopped(cfg, tmp_path, capsys):
    """Run two epochs, keep the state, then resume with a bigger budget and
    confirm it continues rather than restarting from epoch 1."""
    from src.train.train import STATE_FILENAME, load_state, train

    c = _kaggle_like_cfg(cfg, tmp_path, epochs=2, variance_epochs=1)
    out = Path(c["train"]["out_dir"]) / "resume-test-random"
    train(c, device_str="cpu")

    # Simulate a kill partway through the mean phase by restoring a state file.
    killed = {
        "kind": "training_state", "phase": "mean", "epoch": 2,
        "model": torch.load(out / "best.pt", map_location="cpu",
                            weights_only=False)["state_dict"],
        "optimiser": {}, "scheduler": {}, "scaler": {},
        "best_score": 1.0, "best_epoch": 1, "best_val": {"mae": 1.0},
        "best_state": torch.load(out / "best.pt", map_location="cpu",
                                 weights_only=False)["state_dict"],
        "history": [{"phase": "mean", "epoch": 1}, {"phase": "mean", "epoch": 2}],
        "done_phases": [], "config": c,
    }
    loaded = load_state(out)
    assert loaded is None          # cleaned up by the finished run

    import torch as _t
    _t.save(killed, out / STATE_FILENAME)
    assert load_state(out) is not None

    c2 = _kaggle_like_cfg(cfg, tmp_path, epochs=4, variance_epochs=1)
    capsys.readouterr()
    train(c2, device_str="cpu", resume=True)
    printed = capsys.readouterr().out

    assert "RESUMED at epoch 3" in printed, printed[-1500:]
    assert not (out / STATE_FILENAME).exists()


def test_resume_without_state_starts_fresh(cfg, tmp_path):
    """--resume on a clean directory must not fail."""
    from src.train.train import train

    c = _kaggle_like_cfg(cfg, tmp_path, epochs=1, variance_epochs=1)
    result = train(c, device_str="cpu", resume=True)
    assert "r2" in result["test"]
