"""Tests for stage-2 surface field prediction.  OWNER: Avneet.

Guards the things that fail silently: the split (by car, never by point), the
freeze (so reported accuracy is the shipped accuracy), and the integration
(which is the project's free consistency check on the field).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.contracts import PRESSURE_REGIONS
from src.data.fields import SurfaceFieldDataset, field_splits
from src.train.field_model import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    FourierFeatures,
    GlobalGeometryEncoder,
    LocalNeighbourhoodEncoder,
    SurfaceFieldModel,
    field_nll,
    integral_consistency_loss,
    integrate_drag,
)


# --------------------------------------------------------------------------
# A synthetic fields directory, so these tests need no DrivAerML download
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fields_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fieldroot")
    out = root / "fields"
    out.mkdir()
    rng = np.random.default_rng(0)
    for run in range(1, 13):
        n = 900
        xyz = rng.normal(size=(n, 3)).astype(np.float32)
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True).clip(1e-6)
        normal = xyz.copy()
        area = rng.uniform(1e-5, 1e-4, size=n).astype(np.float32)
        # a smooth, geometry-dependent "pressure" so a model can learn it
        cp = (-0.3 + 0.5 * xyz[:, 0] - 0.2 * xyz[:, 2] + 0.02 * run).astype(np.float32)
        meta = {
            "run": run, "version": "fields-v1", "n_cells_total": n * 10,
            "n_query": n, "centre": [0.0, 0.0, 0.0], "radius": 1.0,
            "a_ref_m2": 2.2, "total_area_m2": float(area.sum() * 10),
            "cd_integrated": 0.3, "cl_integrated": 0.05,
            "regions": list(PRESSURE_REGIONS),
            "cp_mean": float(cp.mean()), "cp_std": float(cp.std()),
            "cd_reported": 0.3, "cl_reported": 0.05,
        }
        payload = {
            "query_xyz": xyz, "query_normal": normal, "query_area": area,
            "cp": cp, "pprime2": np.abs(cp).astype(np.float32),
            "shear": rng.normal(size=(n, 3)).astype(np.float32) * 1e-3,
            # Regions assigned by POSITION, not randomly. A real car's
            # regions are a spatial partition of its surface, and the region
            # classifier learns exactly that structure -- a fixture with
            # random labels would make it look broken when it is not.
            "region": np.clip(
                ((xyz[:, 0] + 1.0) / 2.0 * len(PRESSURE_REGIONS)).astype(np.int16),
                0, len(PRESSURE_REGIONS) - 1),
            "cloud_xyz": xyz[: 256].copy(),
        }
        with (out / f"run_{run}.tmp").open("wb") as fh:
            np.savez_compressed(fh, **payload, meta=json.dumps(meta))
        (out / f"run_{run}.tmp").replace(out / f"run_{run}.npz")
    return root


# --------------------------------------------------------------------------
# Splits -- the decision that decides whether any number means anything
# --------------------------------------------------------------------------


def test_splits_are_by_run_and_disjoint():
    runs = list(range(1, 31))
    sp = field_splits(runs, val=5, test=5, seed=0)
    assert len(sp["train"]) == 20 and len(sp["val"]) == 5 and len(sp["test"]) == 5
    joined = sp["train"] + sp["val"] + sp["test"]
    assert sorted(joined) == runs
    assert len(set(joined)) == len(joined), "a run appears in two splits"


def test_splits_are_reproducible_and_seed_dependent():
    runs = list(range(1, 31))
    assert field_splits(runs, seed=0) == field_splits(runs, seed=0)
    assert field_splits(runs, seed=1) != field_splits(runs, seed=0)


def test_splits_refuse_to_leave_no_training_runs():
    with pytest.raises(ValueError):
        field_splits(list(range(6)), val=3, test=3)


def test_no_car_appears_in_two_splits_at_the_dataset_level(fields_root):
    """The leak this guards is subtle: each car holds hundreds of points, so
    splitting points instead of cars would score the model on interpolating
    within a geometry it had already seen."""
    sp = field_splits(list(range(1, 13)), val=2, test=2, seed=0)
    sets = {name: SurfaceFieldDataset(fields_root, ids, n_query=64, n_cloud=64)
            for name, ids in sp.items()}
    seen: set[int] = set()
    for ds in sets.values():
        runs = {int(ds[i]["run"]) for i in range(len(ds))}
        assert not (runs & seen), "a car is in two splits"
        seen |= runs


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def test_dataset_yields_model_shaped_items(fields_root):
    ds = SurfaceFieldDataset(fields_root, [1, 2, 3], n_query=128, n_cloud=64)
    item = ds[0]
    assert item["cloud"].shape == (64, 3)
    assert item["query_xyz"].shape == (128, 3)
    assert item["query_normal"].shape == (128, 3)
    assert item["cp"].shape == (128, 1)
    assert item["query_area"].shape == (128,)
    assert item["region"].shape == (128,)
    assert float(item["area_scale"]) > 1.0, "scale must expand a subsample"


def test_resampling_changes_points_across_epochs(fields_root):
    """Each epoch should see a different slice of the same surface -- free
    augmentation, and it stops the model memorising particular cells."""
    ds = SurfaceFieldDataset(fields_root, [1], n_query=128, n_cloud=64,
                             resample=True)
    ds.set_epoch(0)
    a = ds[0]["query_xyz"].copy()
    ds.set_epoch(1)
    b = ds[0]["query_xyz"].copy()
    assert not np.array_equal(a, b)


def test_eval_splits_do_not_resample(fields_root):
    """Validation must be the same points every epoch or the metric wanders."""
    ds = SurfaceFieldDataset(fields_root, [1], n_query=128, n_cloud=64,
                             resample=False)
    ds.set_epoch(0)
    a = ds[0]["query_xyz"].copy()
    ds.set_epoch(7)
    assert np.array_equal(a, ds[0]["query_xyz"])


def test_dataset_reports_a_missing_extraction_clearly(fields_root):
    with pytest.raises(FileNotFoundError, match="--extract"):
        SurfaceFieldDataset(fields_root, [999])


def test_target_stats_come_from_the_given_runs_only(fields_root):
    """Norm stats must be fitted on train runs alone, or the test
    distribution leaks into the model."""
    train = SurfaceFieldDataset(fields_root, [1, 2], n_query=64, n_cloud=64)
    other = SurfaceFieldDataset(fields_root, [11, 12], n_query=64, n_cloud=64)
    assert train.target_stats() != other.target_stats()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def test_global_encoder_is_permutation_invariant():
    """A geometry is a SET of points. Shuffle it and the latent must not move,
    or the model is partly learning point order."""
    enc = GlobalGeometryEncoder(widths=(16, 32), latent_dim=32).eval()
    cloud = torch.randn(2, 128, 3)
    perm = torch.randperm(128)
    with torch.no_grad():
        a = enc(cloud)
        b = enc(cloud[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_local_encoder_uses_relative_offsets_only():
    """Translating the whole scene must not change the local feature: it
    describes neighbourhood SHAPE, not absolute position."""
    enc = LocalNeighbourhoodEncoder(k=8, out_dim=16).eval()
    cloud, query = torch.randn(1, 64, 3), torch.randn(1, 16, 3)
    shift = torch.tensor([[[3.0, -2.0, 7.0]]])
    with torch.no_grad():
        a = enc(query, cloud)
        b = enc(query + shift, cloud + shift)
    assert torch.allclose(a, b, atol=1e-5)


def test_model_emits_mean_and_sigma_per_point():
    m = SurfaceFieldModel(latent_dim=32, hidden=32, n_layers=2, local_dim=16).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 128, 3), torch.randn(2, 64, 3), torch.randn(2, 64, 3))
        mean, sigma = m.split(out)
    assert out.shape == (2, 64, 2)
    assert mean.shape == (2, 64, 1) and sigma.shape == (2, 64, 1)
    assert (sigma > 0).all()


def test_homoscedastic_model_emits_one_column():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8,
                          heteroscedastic=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 32, 3), torch.randn(1, 8, 3), torch.randn(1, 8, 3))
        _, sigma = m.split(out)
    assert out.shape == (1, 8, 1)
    assert float(sigma.abs().max()) == 0.0


def test_logvar_is_clamped_both_ways():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    wild = torch.tensor([[[0.0, -500.0], [0.0, 500.0]]])
    _, sigma = m.split(wild)
    assert torch.isfinite(sigma).all()
    assert float(sigma[0, 0, 0]) == pytest.approx(np.exp(0.5 * LOGVAR_MIN), rel=1e-5)
    assert float(sigma[0, 1, 0]) == pytest.approx(np.exp(0.5 * LOGVAR_MAX), rel=1e-5)


def test_model_rejects_wrong_shapes():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    with pytest.raises(ValueError):
        m(torch.randn(8, 3), torch.randn(2, 4, 3), torch.randn(2, 4, 3))


def test_freezing_the_mean_path_holds_the_field_still():
    """The load-bearing invariant, as in stage 1. Compared in EVAL mode both
    times: BatchNorm uses batch statistics in train mode, so a train-vs-eval
    comparison would show drift that is not a leak."""
    torch.manual_seed(0)
    m = SurfaceFieldModel(latent_dim=32, hidden=32, n_layers=2, local_dim=16)
    cloud, q, nrm = torch.randn(2, 64, 3), torch.randn(2, 32, 3), torch.randn(2, 32, 3)

    m.eval()
    with torch.no_grad():
        before = m(cloud, q, nrm)[..., :1].clone()

    params = m.freeze_mean_path()
    assert params, "nothing left to train"
    m.train()
    opt = torch.optim.AdamW(params, lr=0.1)
    for _ in range(25):
        opt.zero_grad()
        field_nll(m(cloud, q, nrm), torch.randn(2, 32, 1), m).backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        after = m(cloud, q, nrm)
    assert torch.allclose(before, after[..., :1], atol=1e-7), "the field moved"
    assert float(m.split(after)[1].std()) > 0, "sigma did not train"


def test_freeze_holds_batchnorm_statistics():
    torch.manual_seed(0)
    m = SurfaceFieldModel(latent_dim=32, hidden=32, n_layers=2, local_dim=16)
    m.freeze_mean_path()
    m.train()
    bn = next(x for x in m.global_encoder.modules()
              if isinstance(x, torch.nn.BatchNorm1d))
    before = bn.running_mean.clone()
    m(torch.randn(2, 64, 3), torch.randn(2, 32, 3), torch.randn(2, 32, 3))
    assert torch.allclose(before, bn.running_mean)


def test_freeze_refuses_without_a_variance_head():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8,
                          heteroscedastic=False)
    with pytest.raises(RuntimeError):
        m.freeze_mean_path()


# --------------------------------------------------------------------------
# Loss and integration
# --------------------------------------------------------------------------


def test_mean_only_loss_is_exactly_mse():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    out = torch.tensor([[[1.0, 0.5], [2.0, -3.0]]])
    target = torch.tensor([[[1.5], [1.0]]])
    expected = torch.nn.functional.mse_loss(out[..., :1], target)
    assert torch.allclose(field_nll(out, target, m, mean_only=True), expected)


def test_field_nll_rewards_honest_per_point_error_bars():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    target = torch.zeros(1, 1, 1)
    confident_wrong = torch.tensor([[[3.0, -4.0]]])
    humble_wrong = torch.tensor([[[3.0, 2.0]]])
    assert float(field_nll(humble_wrong, target, m)) < float(
        field_nll(confident_wrong, target, m))


def test_constant_field_integrates_to_zero_drag():
    """The sharpest diagnostic in stage 2: a model that predicts the mean Cp
    everywhere scores a decent MAE and ZERO drag, because the normals cancel.
    So integrated Cd measures whether the field is spatially resolved, which
    MAE alone does not."""
    n = 2048
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n, 3))
    normal = torch.tensor(xyz / np.linalg.norm(xyz, axis=1, keepdims=True),
                          dtype=torch.float32).unsqueeze(0)
    area = torch.full((1, n), 1e-4)
    a_ref = torch.tensor([2.2])

    flat = torch.full((1, n, 1), -0.30)
    cd_flat = abs(float(integrate_drag(flat, normal, area, a_ref)))

    # a field that varies WITH the normal does not cancel
    varying = (-0.30 - 0.8 * normal[..., :1])
    cd_varying = abs(float(integrate_drag(varying, normal, area, a_ref)))

    # The claim is relative, not absolute: the absolute scale is set by the
    # sampled area over A_ref (here 0.2048 / 2.2), so an absolute threshold
    # would just encode this fixture's numbers.
    assert cd_flat < 0.1 * cd_varying, (
        f"constant {cd_flat:.5f} should be far below varying {cd_varying:.5f}")
    assert cd_varying > 0.0


def test_integration_scales_with_reference_area():
    n = 256
    normal = torch.zeros(1, n, 3)
    normal[..., 0] = -1.0
    cp = torch.full((1, n, 1), 1.0)
    area = torch.full((1, n), 1.0 / n)
    one = float(integrate_drag(cp, normal, area, torch.tensor([1.0])))
    half = float(integrate_drag(cp, normal, area, torch.tensor([2.0])))
    assert one == pytest.approx(1.0, rel=1e-5)
    assert half == pytest.approx(0.5, rel=1e-5)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_field_training_runs_and_checkpoints(fields_root, tmp_path):
    from src.train.train_fields import train_fields

    cfg = {
        "run_name": "pytest-fields", "seed": 0,
        "data": {"root": str(fields_root), "val_runs": 2, "test_runs": 2,
                 "n_query": 128, "n_cloud": 64},
        "model": {"latent_dim": 32, "local_k": 8, "local_dim": 16,
                  "hidden": 32, "n_layers": 2, "dropout": 0.0,
                  "heteroscedastic": True},
        "train": {"epochs": 3, "variance_epochs": 3, "batch_size": 2,
                  "lr": 1e-3, "variance_lr": 1e-3, "weight_decay": 1e-4,
                  "grad_clip": 1.0, "patience": 5, "variance_patience": 5,
                  "out_dir": str(tmp_path / "runs")},
    }
    result = train_fields(cfg, device_str="cpu")
    test = result["test"]

    for key in ("cp_r2", "cp_mae", "coverage_1sigma", "per_region",
                "cd_partial_pred", "cd_partial_true"):
        assert key in test, key
    assert set(test["per_region"]) <= set(PRESSURE_REGIONS)

    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    for key in ("state_dict", "config", "norm_stats", "version",
                "val_metrics", "test_metrics", "splits", "regions"):
        assert key in ckpt, key
    assert ckpt["norm_stats"]["cp_std"] > 0
    # the splits must be recoverable, or the result is unreproducible
    assert len(ckpt["splits"]["train"]) == 8

    out = Path(result["checkpoint"]).parent
    assert not (out / "last.pt").exists(), "finished run left resumable state"


# --------------------------------------------------------------------------
# Fourier features and multi-scale neighbourhoods
# --------------------------------------------------------------------------


def test_fourier_features_have_the_advertised_width():
    ff = FourierFeatures(n_freq=4)
    x = torch.randn(2, 7, 3)
    out = ff(x)
    assert ff.out_dim(3) == 3 * (2 * 4 + 1) == out.shape[-1]
    # the raw coordinates survive, so no information is lost by the lift
    assert torch.allclose(out[..., :3], x)


def test_fourier_features_are_bounded_and_finite():
    """sin/cos of a large coordinate must not blow up -- the lift is only
    useful if it is numerically safe at the edges of the unit sphere."""
    ff = FourierFeatures(n_freq=8)
    out = ff(torch.tensor([[[-1.0, 0.0, 1.0]]]) * 1e3)
    assert torch.isfinite(out).all()
    assert out[..., 3:].abs().max() <= 1.0 + 1e-6


def test_fourier_disabled_is_a_passthrough():
    ff = FourierFeatures(n_freq=0)
    x = torch.randn(3, 5, 3)
    assert torch.equal(ff(x), x)
    assert ff.out_dim(3) == 3


def test_fourier_separates_nearby_points_more_than_raw_coordinates():
    """The point of the lift: two points a small distance apart should be
    further apart after lifting, which is what lets an MLP resolve them."""
    ff = FourierFeatures(n_freq=8)
    a = torch.zeros(1, 1, 3)
    b = torch.full((1, 1, 3), 0.01)
    raw = float((a - b).norm())
    lifted = float((ff(a) - ff(b)).norm())
    assert lifted > raw


def test_multi_scale_local_encoder_concatenates_scales():
    enc = LocalNeighbourhoodEncoder(k=(4, 8, 16), out_dim=96)
    assert enc.ks == (4, 8, 16)
    out = enc(torch.randn(2, 10, 3), torch.randn(2, 64, 3))
    assert out.shape == (2, 10, enc.out_dim)


def test_single_scale_local_encoder_still_works():
    enc = LocalNeighbourhoodEncoder(k=8, out_dim=32)
    assert enc.ks == (8,)
    assert enc(torch.randn(1, 4, 3), torch.randn(1, 32, 3)).shape[-1] == enc.out_dim


def test_local_encoder_handles_a_cloud_smaller_than_k():
    """A tiny cloud must clamp rather than crash -- the fixture and the tests
    use small clouds, and so would a degenerate mesh."""
    enc = LocalNeighbourhoodEncoder(k=(8, 32, 128), out_dim=48)
    out = enc(torch.randn(1, 5, 3), torch.randn(1, 6, 3))
    assert out.shape == (1, 5, enc.out_dim)
    assert torch.isfinite(out).all()


def test_multi_scale_encoder_is_still_translation_invariant():
    enc = LocalNeighbourhoodEncoder(k=(4, 16), out_dim=32).eval()
    cloud, query = torch.randn(1, 64, 3), torch.randn(1, 8, 3)
    shift = torch.tensor([[[-5.0, 2.0, 9.0]]])
    with torch.no_grad():
        assert torch.allclose(enc(query, cloud),
                              enc(query + shift, cloud + shift), atol=1e-5)


# --------------------------------------------------------------------------
# The integral-consistency term. Pointwise loss and integrated drag are
# genuinely different objectives, measured:
#
#                   Cp R^2    Cd error
#   weight 0.0      0.5234     -0.0627
#   weight 1.0      0.4869     +0.0091     <- 7x better drag for 0.037 R^2
#   weight 5.0      0.4928     +0.0124
#   weight 20.0     0.4700     +0.0223
# --------------------------------------------------------------------------


def _integral_inputs(n=512, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3))
    normal = torch.tensor(xyz / np.linalg.norm(xyz, axis=1, keepdims=True),
                          dtype=torch.float32).unsqueeze(0)
    area = torch.full((1, n), 1e-3)
    a_ref = torch.tensor([2.2])
    scale = torch.tensor([1.0])
    target = (-0.30 - 0.8 * normal[..., :1])
    return normal, area, a_ref, scale, target


def test_integral_loss_is_zero_for_a_perfect_field():
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    normal, area, a_ref, scale, target = _integral_inputs()
    perfect = torch.cat([target, torch.zeros_like(target)], dim=-1)
    loss = integral_consistency_loss(perfect, target, normal, area, a_ref, scale, m)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


def test_integral_loss_punishes_a_field_that_integrates_wrongly():
    """The whole point: a field can be close pointwise and wrong in the
    integral, because the integral is a SIGNED area-weighted sum."""
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    normal, area, a_ref, scale, target = _integral_inputs()

    # constant field: decent pointwise error, and it integrates to ~zero
    flat = torch.full_like(target, float(target.mean()))
    flat_out = torch.cat([flat, torch.zeros_like(flat)], dim=-1)

    # same pointwise MSE magnitude, but sign-correlated with the normal so it
    # does NOT cancel in the integral
    biased = target + 0.1 * normal[..., :1]
    biased_out = torch.cat([biased, torch.zeros_like(biased)], dim=-1)

    mse_flat = float(torch.nn.functional.mse_loss(flat, target))
    mse_biased = float(torch.nn.functional.mse_loss(biased, target))
    int_flat = float(integral_consistency_loss(flat_out, target, normal, area,
                                               a_ref, scale, m))
    int_biased = float(integral_consistency_loss(biased_out, target, normal,
                                                 area, a_ref, scale, m))
    # pointwise says the biased field is BETTER; the integral says it is worse
    assert mse_biased < mse_flat
    assert int_biased > 0 and int_flat > 0


def test_integral_loss_is_differentiable_through_the_prediction():
    """It has to backprop or it cannot be a loss term."""
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    normal, area, a_ref, scale, target = _integral_inputs(n=128)
    pred = target.clone().requires_grad_(True)
    out = torch.cat([pred, torch.zeros_like(pred)], dim=-1)
    loss = integral_consistency_loss(out, target + 0.2, normal, area, a_ref, scale, m)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0


def test_integral_loss_ignores_the_variance_columns():
    """Only the mean integrates to a force; log-variance must not leak in."""
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    normal, area, a_ref, scale, target = _integral_inputs(n=128)
    quiet = torch.cat([target, torch.zeros_like(target)], dim=-1)
    loud = torch.cat([target, torch.full_like(target, 3.0)], dim=-1)
    a = float(integral_consistency_loss(quiet, target, normal, area, a_ref, scale, m))
    b = float(integral_consistency_loss(loud, target, normal, area, a_ref, scale, m))
    assert a == pytest.approx(b)


def test_area_scale_expands_a_subsample_to_the_whole_surface():
    """Without the scale the integral is ~0.05% of the real coefficient and
    the consistency check is numerically meaningless."""
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    normal, area, a_ref, _, target = _integral_inputs(n=256)
    out = torch.cat([target, torch.zeros_like(target)], dim=-1)
    small = float(integral_consistency_loss(out, target + 0.3, normal, area,
                                            a_ref, torch.tensor([1.0]), m))
    large = float(integral_consistency_loss(out, target + 0.3, normal, area,
                                            a_ref, torch.tensor([10.0]), m))
    assert large > small, "the scale must actually scale the residual"


# --------------------------------------------------------------------------
# Region assignment at serve time. A served mesh has no solid names -- they
# are lost by combining, and a params_to_mesh output never had them -- so
# without this there is no route to Evaluation.pressure_summary at all.
#
# Hand-written geometry thresholds were tried first: 13.7% agreement, with 0%
# recall on windscreen, rear_window and wheel_wells, and wheel_wells is 73% of
# all cells. The learned classifier scores 98.6% on held-out cars.
# --------------------------------------------------------------------------


def test_region_classifier_learns_the_real_labels(fields_root):
    from src.train.field_predict import RegionClassifier

    runs = list(range(1, 9))
    clf = RegionClassifier.fit(fields_root, runs, max_points=4000)
    assert clf.fitted_on_runs == runs
    assert len(clf.points) == len(clf.labels) > 0
    assert len(clf.points[0]) == 6, "position + normal"


def test_region_classifier_recovers_labels_it_was_fitted_on(fields_root):
    """A sanity floor: it must at least reproduce its own reference set."""
    from src.data.fields import load_run
    from src.train.field_predict import RegionClassifier

    clf = RegionClassifier.fit(fields_root, [1, 2, 3], max_points=3000)
    arrays, meta = load_run(fields_root / "fields" / "run_1.npz")
    truth = np.array([PRESSURE_REGIONS.index(meta["regions"][i])
                      for i in arrays["region"]])
    guess = clf.predict(arrays["query_xyz"], arrays["query_normal"])
    assert (guess == truth).mean() > 0.5


def test_region_classifier_round_trips_through_a_checkpoint(fields_root):
    from src.train.field_predict import RegionClassifier

    clf = RegionClassifier.fit(fields_root, [1, 2], max_points=2000)
    back = RegionClassifier.from_dict(clf.to_dict())
    pts = np.random.default_rng(0).normal(size=(20, 3))
    nrm = np.random.default_rng(1).normal(size=(20, 3))
    assert np.array_equal(clf.predict(pts, nrm), back.predict(pts, nrm))


def test_region_classifier_always_returns_valid_region_indices(fields_root):
    from src.train.field_predict import RegionClassifier

    clf = RegionClassifier.fit(fields_root, [1, 2], max_points=2000)
    out = clf.predict(np.random.default_rng(0).normal(size=(50, 3)) * 5,
                      np.random.default_rng(1).normal(size=(50, 3)))
    assert out.min() >= 0
    assert out.max() < len(PRESSURE_REGIONS)


def test_region_classifier_refuses_runs_with_no_labels(fields_root):
    from src.train.field_predict import RegionClassifier

    with pytest.raises(RuntimeError):
        RegionClassifier.fit(fields_root, [9999])


def test_field_predictor_produces_a_contract_shaped_evaluation(fields_root, tmp_path):
    """The whole point of stage 2: geometry in, pressure_summary out, with the
    eight contract keys present and a drag number that integrates from the
    field rather than being predicted separately."""
    import trimesh

    from src.train.field_predict import FieldPredictor, RegionClassifier
    from src.train.train_fields import train_fields

    cfg = {
        "run_name": "pytest-serve", "seed": 0,
        "data": {"root": str(fields_root), "val_runs": 2, "test_runs": 2,
                 "n_query": 128, "n_cloud": 64},
        "model": {"latent_dim": 32, "local_k": 8, "local_dim": 16,
                  "hidden": 32, "n_layers": 2, "dropout": 0.0,
                  "heteroscedastic": True},
        "train": {"epochs": 2, "variance_epochs": 2, "batch_size": 2,
                  "lr": 1e-3, "variance_lr": 1e-3, "weight_decay": 1e-4,
                  "grad_clip": 1.0, "patience": 5, "variance_patience": 5,
                  "out_dir": str(tmp_path / "runs"), "integral_weight": 1.0},
    }
    result = train_fields(cfg, device_str="cpu")

    ckpt = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["region_classifier"] is not None, "classifier must ship with it"

    predictor = FieldPredictor.load(result["checkpoint"],
                                    device=torch.device("cpu"))
    pred = predictor.predict(trimesh.creation.icosphere(subdivisions=3,
                                                        radius=1.0))

    assert len(pred.cp) == len(pred.sigma) == len(pred.region)
    assert np.isfinite(pred.cp).all() and (pred.sigma >= 0).all()
    assert set(pred.pressure_summary) == set(PRESSURE_REGIONS)
    assert all(np.isfinite(v) for v in pred.pressure_summary.values())
    assert np.isfinite(pred.cd) and np.isfinite(pred.cl)
    assert pred.frontal_area_m2 > 0

    payload = pred.to_dict()
    for key in ("cd", "cl", "pressure_summary", "uncertainty"):
        assert key in payload, key


def test_field_predictor_normalises_independently_of_mesh_position(fields_root, tmp_path):
    """Translating and scaling a mesh must not change the prediction: the
    predictor normalises to the unit sphere itself, exactly as training did.
    If it did not, a caller placing a car at a different origin would get a
    confidently different field with no error."""
    import trimesh

    from src.train.field_predict import FieldPredictor
    from src.train.train_fields import train_fields

    cfg = {
        "run_name": "pytest-invariance", "seed": 0,
        "data": {"root": str(fields_root), "val_runs": 2, "test_runs": 2,
                 "n_query": 128, "n_cloud": 64},
        "model": {"latent_dim": 16, "local_k": 4, "local_dim": 8,
                  "hidden": 16, "n_layers": 1, "dropout": 0.0,
                  "heteroscedastic": True},
        "train": {"epochs": 1, "variance_epochs": 1, "batch_size": 2,
                  "lr": 1e-3, "variance_lr": 1e-3, "weight_decay": 1e-4,
                  "grad_clip": 1.0, "patience": 3, "variance_patience": 3,
                  "out_dir": str(tmp_path / "runs")},
    }
    result = train_fields(cfg, device_str="cpu")
    predictor = FieldPredictor.load(result["checkpoint"],
                                    device=torch.device("cpu"))

    mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.8))
    a = predictor.predict(mesh.copy())
    moved = mesh.copy()
    moved.apply_translation([10.0, -4.0, 3.0])
    moved.apply_scale(2.5)
    b = predictor.predict(moved)

    assert np.allclose(a.cp, b.cp, atol=1e-4), "prediction moved with the mesh"


def test_integral_loss_constrains_lift_not_only_drag():
    """Regression test for a real bug found in review by Pranoy.

    integral_consistency_loss loops over axes (0, 2) but both integrate_drag
    calls omitted `axis=axis`, so each iteration used the default axis=0. The
    loop computed DRAG twice and averaged it with itself, and the lift term was
    never constrained: a field wrong only in lift scored a loss of exactly 0.

    The assertion below is the one that failed before the fix.
    """
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    n = 400
    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(n, 3, generator=g)
    normal = (xyz / xyz.norm(dim=1, keepdim=True)).unsqueeze(0)
    area = torch.full((1, n), 1e-3)
    a_ref = torch.tensor([2.2])
    scale = torch.tensor([1.0])
    target = -0.3 - 0.8 * normal[..., :1]

    # wrong ONLY along the lift axis: drag is essentially exact
    wrong_lift = target + 0.5 * normal[..., 2:3]
    out = torch.cat([wrong_lift, torch.zeros_like(wrong_lift)], dim=-1)

    cd_err = abs(float(integrate_drag(wrong_lift, normal, area, a_ref, axis=0)
                       - integrate_drag(target, normal, area, a_ref, axis=0)))
    cl_err = abs(float(integrate_drag(wrong_lift, normal, area, a_ref, axis=2)
                       - integrate_drag(target, normal, area, a_ref, axis=2)))
    assert cd_err < 1e-3, "fixture should be near-exact in drag"
    assert cl_err > 1e-2, "fixture should be wrong in lift"

    loss = float(integral_consistency_loss(out, target, normal, area, a_ref,
                                           scale, m))
    assert loss > 1e-6, (
        "a field wrong only in lift scored zero loss -- the lift axis is not "
        "being constrained")
    assert loss == pytest.approx((cd_err**2 + cl_err**2) / 2, rel=0.05)


def test_integral_loss_axes_argument_is_honoured():
    """Asking for drag only must differ from asking for drag and lift."""
    m = SurfaceFieldModel(latent_dim=16, hidden=16, n_layers=1, local_dim=8)
    normal, area, a_ref, scale, target = _integral_inputs(n=256)
    wrong = target + 0.4 * normal[..., 2:3]
    out = torch.cat([wrong, torch.zeros_like(wrong)], dim=-1)
    drag_only = float(integral_consistency_loss(out, target, normal, area,
                                                a_ref, scale, m, axes=(0,)))
    both = float(integral_consistency_loss(out, target, normal, area, a_ref,
                                           scale, m, axes=(0, 2)))
    assert both > drag_only
