"""The 3D viewer's mesh payload: shape, region labels and the illustrative field."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from src.geometry.params import BODY_STYLES, MORPH_BOUNDS, frontal_area
from src.geometry.viewer import REGIONS, mesh_payload


def _decode(payload, key, dtype):
    return np.frombuffer(base64.b64decode(payload[key]), dtype=dtype)


@pytest.fixture(scope="module", params=BODY_STYLES)
def payload(request):
    return mesh_payload({}, request.param)


def test_payload_is_json_and_consistent(payload):
    json.dumps(payload)
    vertices = _decode(payload, "vertices", np.float32).reshape(-1, 3)
    faces = _decode(payload, "faces", np.uint32).reshape(-1, 3)
    assert len(vertices) == payload["n_vertices"]
    assert len(faces) == payload["n_faces"]
    assert faces.max() < len(vertices)
    assert len(_decode(payload, "region", np.int8)) == len(vertices)
    assert len(_decode(payload, "cp", np.float32)) == len(vertices)
    assert payload["frontal_area_m2"] == pytest.approx(frontal_area({}, payload["body_style"]))
    assert payload["ignored_params"] == []


def test_regions_match_the_contract():
    from src import contracts
    assert REGIONS == contracts.PRESSURE_REGIONS


def test_every_region_is_on_the_car_where_it_belongs(payload):
    vertices = _decode(payload, "vertices", np.float32).reshape(-1, 3)
    region = _decode(payload, "region", np.int8)
    for i, name in enumerate(REGIONS):
        assert (region == i).sum() > 10, f"{name} is missing"

    x_mean = {name: vertices[region == i, 0].mean() for i, name in enumerate(REGIONS)}
    z_mean = {name: vertices[region == i, 2].mean() for i, name in enumerate(REGIONS)}
    assert x_mean["front_fascia"] < x_mean["hood"] < x_mean["windscreen"] < x_mean["roof"]
    assert x_mean["roof"] < x_mean["rear_window"] < x_mean["rear_fascia"]
    assert z_mean["roof"] > z_mean["hood"] > z_mean["underbody"]


def test_illustrative_pressure_has_the_right_pattern(payload):
    region = _decode(payload, "region", np.int8)
    cp = _decode(payload, "cp", np.float32)
    assert np.isfinite(cp).all()
    assert cp.min() >= -1.0 and cp.max() <= 1.0
    mean = {name: cp[region == i].mean() for i, name in enumerate(REGIONS)}
    assert mean["front_fascia"] > 0.3          # stagnation at the nose
    assert mean["roof"] < mean["hood"] < 0     # suction grows over the top
    assert mean["front_fascia"] > mean["windscreen"] > mean["roof"]


def test_old_parameter_names_are_reported():
    payload = mesh_payload({"ride_height": 0.14}, "fastback")
    assert payload["ignored_params"] == ["ride_height"]


def test_flow_outline_wraps_the_mesh(payload):
    """The airflow steers around this outline, so it must hug the real body."""
    flow = payload["flow"]
    vertices = _decode(payload, "vertices", np.float32).reshape(-1, 3)
    region = _decode(payload, "region", np.int8)
    body = vertices[region != REGIONS.index("wheel_wells")]
    x = np.asarray(flow["x"])
    assert len(x) == len(flow["z_top"]) == len(flow["z_bot"]) == len(flow["half_width"])
    assert np.all(np.diff(x) > 0)
    assert x[0] == pytest.approx(body[:, 0].min(), abs=0.02)
    assert x[-1] == pytest.approx(body[:, 0].max(), abs=0.02)
    assert max(flow["z_top"]) == pytest.approx(body[:, 2].max(), abs=0.03)
    assert max(flow["half_width"]) == pytest.approx(np.abs(body[:, 1]).max(), abs=0.08)
    assert np.all(np.asarray(flow["z_bot"]) < np.asarray(flow["z_top"]))
    marks = flow["landmarks"]
    assert marks["hood_front"] < marks["cowl"] < marks["roof_front"] < marks["roof_end"] < marks["deck_start"]
    assert flow["axles"][0] < flow["axles"][1]
    json.dumps(flow)


def test_baseline_area_is_the_style_default():
    wide = mesh_payload({"A_Car_Width": MORPH_BOUNDS["A_Car_Width"][1]}, "estateback")
    assert wide["baseline_frontal_area_m2"] == pytest.approx(frontal_area({}, "estateback"))
    assert wide["frontal_area_m2"] > wide["baseline_frontal_area_m2"]


def test_shape_follows_the_parameters():
    base = mesh_payload({}, "notchback")
    wide = mesh_payload({"A_Car_Width": MORPH_BOUNDS["A_Car_Width"][1]}, "notchback")
    assert wide["width_m"] > base["width_m"] + 0.1
    assert wide["frontal_area_m2"] > base["frontal_area_m2"]
