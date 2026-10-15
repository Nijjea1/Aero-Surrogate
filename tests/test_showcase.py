"""The real test-set cars the page puts next to CFD ground truth."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import BODY_STYLES, PARAM_BOUNDS, PARAM_NAMES
from src.eval import data as D
from src.eval import showcase


@pytest.fixture
def table(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "params.csv"
    header = ["Experiment", *PARAM_NAMES, D.TARGET, D.LABEL_STD, showcase.LABEL_CL]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for prefix in ("F", "N", "E"):
            for i in range(40):
                params = [rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS.values()]
                writer.writerow([f"{prefix}_S_WW_WM_{i:03d}", *params, 0.2 + 0.002 * i, 0.006, -0.1])
    return D.load_table(path)


def test_picks_test_designs_spread_by_cfd_drag(table):
    split = D.make_splits(table.ids)
    cars = showcase.pick(table, split["test"], per_style=3)
    assert [c["body_style"] for c in cars] == [s for s in BODY_STYLES for _ in range(3)]
    assert all(c["design_id"] in split["test"] for c in cars)
    assert len({c["design_id"] for c in cars}) == len(cars)
    for style in BODY_STYLES:
        cd = [c["cd_cfd"] for c in cars if c["body_style"] == style]
        assert cd == sorted(cd) and cd[0] < cd[-1]
    assert cars[0]["cd_cfd_std"] == pytest.approx(0.006) and cars[0]["cl_cfd"] == pytest.approx(-0.1)
    assert set(cars[0]["params"]) == set(PARAM_NAMES)


def test_out_of_bounds_designs_are_never_shown(table):
    ids = [i for i in table.ids if i.startswith("F_")][:5]
    table.columns["A_Car_Width"][table.rows(ids)] = 999.0
    cars = showcase.pick(table, ids, per_style=2)
    assert cars == []


def test_percentiles_avoid_the_extremes():
    assert showcase.percentiles(1) == [50]
    assert showcase.percentiles(4) == [12.5, 37.5, 62.5, 87.5]


def test_committed_file_is_servable():
    """web/designs.json is what the page loads: in bounds, labelled, attributed."""
    path = Path(__file__).resolve().parents[1] / "web" / "designs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "CC BY-NC" in payload["source"]
    assert payload["designs"]
    for car in payload["designs"]:
        assert car["body_style"] in BODY_STYLES
        assert car["design_id"][0] == car["body_style"][0].upper()
        assert set(car["params"]) == set(PARAM_NAMES)
        assert all(PARAM_BOUNDS[k][0] <= v <= PARAM_BOUNDS[k][1] for k, v in car["params"].items())
        assert 0.1 < car["cd_cfd"] < 0.5
