"""Seam 2: parameters -> mesh.

The service and the viewer both build on these meshes, so every design in the
box must give a closed, car-sized body, and every dial must move the shape
the way its name says.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import contracts
from src.geometry.params import (
    BODY_STYLES,
    MORPH_BOUNDS,
    frontal_area,
    normalize_points,
    params_to_mesh,
    random_params,
    sample_surface,
)


def _extent(mesh):
    lo, hi = mesh.bounds
    return hi - lo   # length, width, height


def _assert_sane(mesh):
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0, "normals point inward"
    assert np.isfinite(mesh.vertices).all()
    length, width, height = _extent(mesh)
    assert 4.2 < length < 5.0
    assert 1.7 < width < 2.1
    assert 1.2 < height < 1.6
    assert mesh.bounds[0][2] >= -1e-9, "car sits below the ground"


@pytest.mark.parametrize("style", BODY_STYLES)
def test_baseline_is_a_closed_car(style):
    _assert_sane(params_to_mesh({}, style))


@pytest.mark.parametrize("style", BODY_STYLES)
@pytest.mark.parametrize("name", list(MORPH_BOUNDS))
@pytest.mark.parametrize("end", [0, 1])
def test_each_dial_at_its_limit_stays_sane(style, name, end):
    """Perturbing one parameter to either bound gives a sane shape that differs
    from the baseline -- no dial is dead, none breaks the body."""
    base = params_to_mesh({}, style)
    mesh = params_to_mesh({name: MORPH_BOUNDS[name][end]}, style)
    _assert_sane(mesh)
    assert mesh.volume == pytest.approx(base.volume, rel=0.2)
    if mesh.vertices.shape == base.vertices.shape:
        assert not np.allclose(mesh.vertices, base.vertices), f"{name} has no effect"


@pytest.mark.parametrize("style", BODY_STYLES)
def test_dials_move_the_right_way(style):
    def ext(**p):
        return _extent(params_to_mesh(p, style))

    base = ext()
    assert ext(A_Car_Length=130)[0] > base[0] + 0.1
    assert ext(H_Front_Bumper_Length=60)[0] > base[0] + 0.05
    assert ext(A_Car_Width=150)[1] > base[1] + 0.1
    assert ext(A_Car_Roof_Height=90)[2] > base[2] + 0.05
    assert ext(A_Car_Roof_Height=-60)[2] < base[2] - 0.04


@pytest.mark.parametrize("style", BODY_STYLES)
def test_frontal_area(style):
    base = frontal_area({}, style)
    assert 1.9 < base < 2.5          # DrivAer is about 2.16 m^2
    assert frontal_area({"A_Car_Width": 150}, style) > base
    assert frontal_area({"A_Car_Roof_Height": 90}, style) > base
    assert frontal_area({"A_Car_Roof_Height": -60}, style) < base


def _silhouette_area(mesh, h=0.005):
    """Area of the mesh seen from the front, by scanline rasterisation: each
    row z cuts every triangle it crosses into a y-interval, and a cell counts
    if its centre lies inside any of them."""
    tri = mesh.triangles[:, :, 1:]                   # (n, 3, [y, z])
    lo, hi = mesh.bounds[0][1:], mesh.bounds[1][1:]
    ys = np.arange(lo[0] + h / 2, hi[0], h)
    covered = 0
    for z in np.arange(lo[1] + h / 2, hi[1], h):
        zt = tri[:, :, 1]
        t = tri[(zt.min(axis=1) <= z) & (zt.max(axis=1) >= z)]
        if not len(t):
            continue
        a, b = t, np.roll(t, -1, axis=1)             # the three edges
        dz = b[:, :, 1] - a[:, :, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - a[:, :, 1]) / dz
            y = a[:, :, 0] + s * (b[:, :, 0] - a[:, :, 0])
        valid = (dz != 0) & (s >= 0) & (s <= 1)
        y_lo = np.where(valid, y, np.inf).min(axis=1)
        y_hi = np.where(valid, y, -np.inf).max(axis=1)
        ok = np.isfinite(y_lo) & np.isfinite(y_hi)
        start = np.searchsorted(ys, y_lo[ok], side="left")
        stop = np.searchsorted(ys, y_hi[ok], side="right")
        mark = np.zeros(len(ys) + 1, dtype=np.int64)
        np.add.at(mark, start, 1)
        np.add.at(mark, stop, -1)
        covered += int((np.cumsum(mark)[:-1] > 0).sum())
    return covered * h * h


@pytest.mark.parametrize("style", BODY_STYLES)
def test_frontal_area_matches_the_mesh(style):
    """The analytic area agrees with the mesh's own front silhouette."""
    for p in ({}, {"A_Car_Width": 150, "A_Car_Roof_Height": 90}, {"A_Car_Width": -60, "A_Car_Roof_Height": -60}):
        assert frontal_area(p, style) == pytest.approx(_silhouette_area(params_to_mesh(p, style)), rel=0.01)


@pytest.mark.parametrize("style", BODY_STYLES)
def test_roof_height_does_not_change_length(style):
    """A taller roof steepens the glass rather than stretching the car."""
    lo, hi = MORPH_BOUNDS["A_Car_Roof_Height"]
    base = _extent(params_to_mesh({}, style))[0]
    for value in (lo, hi):
        assert _extent(params_to_mesh({"A_Car_Roof_Height": value}, style))[0] == pytest.approx(base, abs=0.02)


def test_jointly_extreme_designs_stay_car_sized():
    """Every dial at one of its limits at once: still a closed, car-sized
    body. DrivAerNet's longest designs are about 4.85 m."""
    rng = np.random.default_rng(1)
    for k in range(30):
        p = {n: MORPH_BOUNDS[n][rng.integers(0, 2)] for n in MORPH_BOUNDS}
        mesh = params_to_mesh(p, BODY_STYLES[k % len(BODY_STYLES)])
        assert mesh.is_watertight and mesh.volume > 0
        assert 4.2 < _extent(mesh)[0] < 5.1


def test_sample_surface_matches_training_normalisation():
    pts = sample_surface({}, 4096)
    assert pts.shape == (4096, 3)
    assert pts.dtype == np.float32
    radius = np.linalg.norm(pts, axis=1)
    assert radius.max() <= 1.0 + 1e-6
    assert radius.max() > 0.8
    np.testing.assert_array_equal(pts, sample_surface({}, 4096))
    assert not np.array_equal(pts, sample_surface({}, 4096, seed=1))


def test_normalize_points_uses_bounding_box_centre():
    pts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    out = normalize_points(pts)
    # Bounding-box centre is x = 2; the centroid (x = 1.075) would not give +/-1.
    np.testing.assert_allclose(out[:2, 0], [-1.0, 1.0], atol=1e-6)


def test_missing_parameters_mean_baseline():
    full = {name: 0.0 for name in MORPH_BOUNDS}
    np.testing.assert_allclose(params_to_mesh({}).vertices, params_to_mesh(full).vertices)


def test_out_of_bounds_values_are_clamped():
    lo, hi = MORPH_BOUNDS["A_Car_Width"]
    clamped = params_to_mesh({"A_Car_Width": hi})
    np.testing.assert_allclose(params_to_mesh({"A_Car_Width": hi * 10}).vertices, clamped.vertices)


def test_unknown_parameters_warn():
    with pytest.warns(UserWarning, match="ride_height"):
        params_to_mesh({"ride_height": 0.14})


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="body style"):
        params_to_mesh({}, "pickup")
    with pytest.raises(ValueError, match="finite"):
        params_to_mesh({"A_Car_Width": float("nan")})


def test_random_designs_are_sane():
    rng = np.random.default_rng(0)
    for k in range(10):
        p = random_params(rng)
        assert set(p) == set(MORPH_BOUNDS)
        assert all(MORPH_BOUNDS[n][0] <= v <= MORPH_BOUNDS[n][1] for n, v in p.items())
        _assert_sane(params_to_mesh(p, BODY_STYLES[k % len(BODY_STYLES)]))


@pytest.mark.skipif(
    not set(contracts.PARAM_BOUNDS) & set(MORPH_BOUNDS),
    reason="contracts 2.0.0 not on this branch yet",
)
def test_geometry_bounds_match_the_contract():
    assert MORPH_BOUNDS == contracts.PARAM_BOUNDS
    assert BODY_STYLES == contracts.BODY_STYLES
