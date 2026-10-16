"""The drag race's physics (web/physics.js), run in Node. Skipped without Node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PHYSICS = (Path(__file__).resolve().parents[1] / "web" / "physics.js").as_uri()
NODE = shutil.which("node")

SCRIPT = """
import { standingStart, positionAt, aeroLoads } from "%s";
const run = (cdA, extra = {}) => standingStart({ cdA, ...extra });
const slick = run(0.208 * 2.2), base = run(0.2217 * 2.2), twin = run(0.2217 * 2.2);
const aeroTop = Math.cbrt(2 * 150e3 / (1.225 * 0.2217 * 2.2));
const mid = positionAt(base.samples, base.time / 2), past = positionAt(base.samples, base.time + 2);
console.log(JSON.stringify({
  slick: slick.time, base: base.time, twin: twin.time, finished: base.finished,
  trap: base.trapSpeed, aeroTop, lastX: base.samples.at(-1).x,
  midX: mid.x, past: past.x, pastV: past.v,
  launchA: base.samples[1].v / base.samples[1].t,
  lighter: run(0.2217 * 2.2, { power: 300e3 }).time,
  drag130: aeroLoads({ cd: 0.2217, area: 2.2, speedKmh: 130 }).drag,
}));
""" % PHYSICS


@pytest.fixture(scope="module")
def race():
    if NODE is None:
        pytest.skip("node is not installed")
    out = subprocess.run([NODE, "--input-type=module", "-e", SCRIPT], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_less_drag_finishes_first_and_equal_cars_tie(race):
    assert race["slick"] < race["base"]
    assert race["twin"] == race["base"]
    assert race["lighter"] < race["base"]          # more power, quicker


def test_the_run_is_physical(race):
    assert race["finished"] and race["lastX"] == pytest.approx(1000)
    assert 0 < race["trap"] < race["aeroTop"]       # drag and rolling resistance cap the speed
    assert race["launchA"] == pytest.approx(9.81, rel=0.05)   # grip-limited launch, mu = 1
    assert 0 < race["midX"] < 1000
    # Past the line a car coasts on at its trap speed, so the gap keeps reading true.
    assert race["past"] == pytest.approx(1000 + 2 * race["trap"], rel=1e-6)
    assert race["pastV"] == pytest.approx(race["trap"])


def test_road_loads_match_the_formula(race):
    v = 130 / 3.6
    assert race["drag130"] == pytest.approx(0.5 * 1.225 * v * v * 0.2217 * 2.2)


MODES = """
import { simulateRace, topSpeed, coastDistance, positionAt, timeAt } from "%s";
const A = 2.2, slick = 0.20 * A, draggy = 0.24 * A;
const out = {};
for (const mode of ["distance", "top", "coast"]) {
  const opts = { distance: 50000, v0: 130 / 3.6 };
  const a = simulateRace(mode, { cdA: slick, ...opts }), b = simulateRace(mode, { cdA: draggy, ...opts });
  out[mode] = { a: a.end, b: b.end, vmax: a.vmax, finished: a.finished };
}
const coast = simulateRace("coast", { cdA: slick, v0: 130 / 3.6 });
const t = coast.end.t / 2, x = positionAt(coast.samples, t).x;
out.coastClosed = coastDistance({ cdA: slick, v0: 130 / 3.6 });
out.roundTrip = [t, timeAt(coast.samples, x)];
out.topAnalytic = topSpeed({ cdA: slick });
out.longFinish = simulateRace("distance", { cdA: slick, distance: 100000 }).finished;
console.log(JSON.stringify(out));
""" % PHYSICS


@pytest.fixture(scope="module")
def modes():
    if NODE is None:
        pytest.skip("node is not installed")
    out = subprocess.run([NODE, "--input-type=module", "-e", MODES], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_less_drag_wins_every_mode(modes):
    assert modes["distance"]["a"]["t"] < modes["distance"]["b"]["t"]
    assert modes["distance"]["a"]["x"] == pytest.approx(50000)
    assert modes["top"]["a"]["v"] > modes["top"]["b"]["v"]
    assert modes["coast"]["a"]["x"] > modes["coast"]["b"]["x"]
    assert modes["coast"]["a"]["v"] == 0


def test_long_races_finish_and_top_speed_matches_the_balance(modes):
    """A 100 km race is no longer cut off at 300 s, and top speed is where power meets drag plus rolling."""
    assert modes["longFinish"] and modes["distance"]["finished"]
    p = 150e3
    v = modes["topAnalytic"]
    assert (0.5 * 1.225 * 0.20 * 2.2 * v * v + 0.010 * 1600 * 9.81) * v == pytest.approx(p, rel=1e-6)
    assert modes["top"]["a"]["v"] == pytest.approx(v - 0.3 / 3.6, abs=0.01)


def test_coast_down_matches_the_closed_form_and_time_inverts_position(modes):
    assert modes["coast"]["a"]["x"] == pytest.approx(modes["coastClosed"], rel=2e-3)
    t, back = modes["roundTrip"]
    assert back == pytest.approx(t, abs=0.06)
