"""Synthetic stand-in for DrivAerNet.  OWNER: Avneet.

A development fixture, not a dataset. It generates car-like STL meshes from the
frozen parameter space in src.contracts plus a labels CSV with an analytic Cd,
so the whole pipeline -- check, cache, train, checkpoint -- can be exercised
before a single byte of DrivAerNet is downloaded.

    python -m src.data.synthetic --n 300 --out data/synthetic

The CSV it writes carries the same shape as the published DrivAerNet files: an
id column, a Cd column, a Category column, and one column per design
parameter. So configs/synthetic.yaml and configs/tabular.yaml both read it, and
swapping to real data is a path change in the config.

The labels come from analytic_cd() below -- a smooth function with known optima
and gaussian noise. That means the achievable R^2 is known in advance (~0.97),
which is the whole point: when training underperforms on this fixture the bug
is in our code, not in the data.

KNOWN LIMIT, so nobody over-reads a number from here: analytic_cd is a sum of
squared terms plus one interaction, so a quadratic polynomial regression is the
exactly-correct model class for it and scores ~0.96 where the neural models
score ~0.87. That gap is a property of this fixture, not a deficiency in the
models and not a target. Real Cd is not a quadratic in eight parameters. Use
this fixture to prove that code runs, caches, calibrates and checkpoints --
never to compare model families, and never as a reportable result.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# The fixture has its OWN parameter space, deliberately not src.contracts.
#
# It used to import PARAM_BOUNDS, and schema 2.0.0 broke it -- correctly, but
# for no good reason. This generator's job is to produce known-answer data that
# exercises the loader, the cache, the trainer and the checkpoint. It does not
# need to model the real design space, and coupling it to a contract that will
# keep moving just means the dev loop breaks every time the contract does.
#
# These eight are intuitive absolute dimensions that make recognisable car
# shapes. The real space is 16 morph deltas plus a body style; see
# src/contracts.py. Do not confuse the two, and never report a number produced
# from this fixture.
# --------------------------------------------------------------------------

SYNTH_BOUNDS: dict[str, tuple[float, float]] = {
    "ride_height": (0.10, 0.20),         # m, floor to underbody
    "rear_deck_angle": (-5.0, 20.0),     # deg, positive = upswept
    "windscreen_rake": (25.0, 65.0),     # deg from vertical
    "boat_tail_angle": (0.0, 18.0),      # deg, rear taper per side
    "front_overhang": (0.70, 1.05),      # m
    "roof_curvature": (0.00, 0.35),      # dimensionless
    "rear_diffuser_angle": (0.0, 15.0),  # deg
    "track_width": (1.50, 1.70),         # m
}

SYNTH_NAMES: list[str] = list(SYNTH_BOUNDS)


def synth_default_params() -> dict[str, float]:
    """Midpoint of every fixture bound."""
    return {k: (lo + hi) / 2.0 for k, (lo, hi) in SYNTH_BOUNDS.items()}


# Overall length is fixed; the parameters move everything else. Real DrivAerNet
# variants hold wheelbase constant too.
BODY_LENGTH: float = 4.60
HOOD_HEIGHT: float = 0.72
ROOF_HEIGHT: float = 1.42
N_STATIONS: int = 28
N_RING: int = 24

# rear_deck_angle ranges that define a body style. Holding them apart is what
# makes the held-out-category split in make_splits a real generalisation test
# rather than a reshuffle.
CATEGORY_DECK_RANGES: dict[str, tuple[float, float]] = {
    "fastback": (-5.0, 4.0),
    "notchback": (4.0, 12.0),
    "estate": (12.0, 20.0),
}

NOISE_SD: float = 0.004


def analytic_cd(params: dict[str, float]) -> float:
    """Ground-truth Cd for a synthetic design. Smooth, with real optima.

    Hand-built so each parameter matters in a physically plausible direction:
    an optimum ride height, an optimum rear deck angle, monotone gains from
    windscreen rake and boat-tail taper up to a point, a frontal-area penalty
    on track width, and one interaction (a low car gains more from a diffuser).
    """
    p = params
    cd = 0.255
    cd += 1.10 * (p["ride_height"] - 0.135) ** 2
    cd += 2.6e-4 * (p["rear_deck_angle"] - 12.0) ** 2
    cd += -7.5e-4 * (p["windscreen_rake"] - 25.0)
    cd += 9.0e-6 * (p["windscreen_rake"] - 25.0) ** 2
    cd += -1.6e-3 * min(p["boat_tail_angle"], 12.0)
    cd += 2.0e-4 * max(p["boat_tail_angle"] - 12.0, 0.0) ** 2
    cd += 0.075 * (p["roof_curvature"] - 0.16) ** 2
    cd += -1.2e-3 * p["rear_diffuser_angle"] + 1.1e-4 * p["rear_diffuser_angle"] ** 2
    cd += 0.16 * (p["track_width"] - 1.50)
    cd += 0.035 * (p["front_overhang"] - 0.85) ** 2
    cd += -0.012 * (0.20 - p["ride_height"]) * p["rear_diffuser_angle"]
    return cd


def sample_params(rng: np.random.Generator, category: str) -> dict[str, float]:
    """Uniform inside SYNTH_BOUNDS, except the deck angle, which the category
    pins to its own band."""
    out: dict[str, float] = {}
    for name, (lo, hi) in SYNTH_BOUNDS.items():
        out[name] = float(rng.uniform(lo, hi))
    lo, hi = CATEGORY_DECK_RANGES[category]
    out["rear_deck_angle"] = float(rng.uniform(lo, hi))
    return out


def _top_profile(x: float, p: dict[str, float]) -> float:
    """Height of the body's upper surface at station x."""
    floor = p["ride_height"]
    nose_h = floor + 0.34
    overhang = p["front_overhang"]

    # A more raked screen takes a longer run to climb the same height.
    rise = ROOF_HEIGHT - HOOD_HEIGHT
    screen_len = max(0.22, rise * math.tan(math.radians(p["windscreen_rake"])))
    x_screen_end = overhang + screen_len

    # The roof runs until the rear deck starts; the deck's descent angle is the
    # base fastback descent softened by an upswept deck.
    descent = math.radians(20.0 - p["rear_deck_angle"])
    deck_len = 1.35
    x_deck = BODY_LENGTH - deck_len

    if x <= overhang:
        t = x / max(overhang, 1e-6)
        return nose_h + (HOOD_HEIGHT - nose_h) * math.sqrt(t)
    if x <= x_screen_end:
        t = (x - overhang) / max(screen_len, 1e-6)
        return HOOD_HEIGHT + rise * t
    if x <= x_deck:
        span = max(x_deck - x_screen_end, 1e-6)
        t = (x - x_screen_end) / span
        bulge = p["roof_curvature"] * 0.22 * math.sin(math.pi * t)
        return ROOF_HEIGHT + bulge
    t = x - x_deck
    return ROOF_HEIGHT - math.tan(descent) * t


def _bottom_profile(x: float, p: dict[str, float]) -> float:
    """Height of the underbody at station x. Flat, then the diffuser sweeps up."""
    floor = p["ride_height"]
    x_diff = BODY_LENGTH - 1.00
    if x <= x_diff:
        return floor
    return floor + math.tan(math.radians(p["rear_diffuser_angle"])) * (x - x_diff)


def _half_width(x: float, p: dict[str, float]) -> float:
    """Half-width at station x: a tapered nose, a full midsection, then the
    boat tail pulling the rear in."""
    w0 = p["track_width"] / 2.0
    x_bt = BODY_LENGTH - 1.25

    if x < 0.60:
        t = x / 0.60
        w = w0 * (0.58 + 0.42 * math.sqrt(t))
    else:
        w = w0
    if x > x_bt:
        w -= math.tan(math.radians(p["boat_tail_angle"])) * (x - x_bt)
    return max(w, 0.12 * w0)


def _ring(x: float, p: dict[str, float], exponent: float = 3.4) -> np.ndarray:
    """One cross-section: a superellipse between the floor and the roofline."""
    z_bot = _bottom_profile(x, p)
    z_top = _top_profile(x, p)
    cz = 0.5 * (z_top + z_bot)
    hz = max(0.5 * (z_top - z_bot), 1e-3)
    w = _half_width(x, p)

    t = np.linspace(0.0, 2.0 * np.pi, N_RING, endpoint=False)
    k = 2.0 / exponent
    y = w * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** k
    z = cz + hz * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** k
    return np.stack([np.full_like(t, x), y, z], axis=1)


def params_to_synthetic_mesh(params: dict[str, float]):
    """Stitch the station rings into a closed body. Returns a trimesh.Trimesh.

    Deliberately NOT the real params_to_mesh -- that is Paramveer's, in
    src/geometry, and it works on the contract's parameter space, not this
    fixture's. This exists only so there are meshes to train on today.
    """
    import trimesh

    xs = np.linspace(0.0, BODY_LENGTH, N_STATIONS)
    rings = [_ring(float(x), params) for x in xs]
    verts = np.concatenate(rings, axis=0)

    faces: list[tuple[int, int, int]] = []
    m = N_RING
    for i in range(len(rings) - 1):
        a, b = i * m, (i + 1) * m
        for j in range(m):
            k = (j + 1) % m
            faces.append((a + j, a + k, b + k))
            faces.append((a + j, b + k, b + j))

    # Cap both ends with a fan to the ring centroid.
    front_c = len(verts)
    rear_c = front_c + 1
    verts = np.vstack([verts, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    last = (len(rings) - 1) * m
    for j in range(m):
        k = (j + 1) % m
        faces.append((front_c, j, k))
        faces.append((rear_c, last + k, last + j))

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def generate(out_dir: Path, n: int, seed: int = 0) -> dict:
    """Write meshes/<id>.stl and drag_coefficients.csv. Returns a summary."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    categories = list(CATEGORY_DECK_RANGES)
    rows: list[dict] = []
    n_leaky = 0

    for i in range(n):
        category = categories[i % len(categories)]
        params = sample_params(rng, category)
        design_id = f"SYN_{category[:3].upper()}_{i:04d}"

        mesh = params_to_synthetic_mesh(params)
        if not mesh.is_watertight:
            n_leaky += 1
        mesh.export(mesh_dir / f"{design_id}.stl")

        cd = analytic_cd(params) + float(rng.normal(0.0, NOISE_SD))
        row = {"Design": design_id, "Average Cd": round(cd, 6), "Category": category}
        row.update({k: round(params[k], 6) for k in SYNTH_NAMES})
        row["Cd_clean"] = round(analytic_cd(params), 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "drag_coefficients.csv"
    df.to_csv(csv_path, index=False)

    cd = df["Average Cd"].to_numpy()
    noise_var = NOISE_SD**2
    return {
        "n": n,
        "csv": str(csv_path),
        "mesh_dir": str(mesh_dir),
        "cd_mean": float(cd.mean()),
        "cd_std": float(cd.std()),
        "cd_min": float(cd.min()),
        "cd_max": float(cd.max()),
        "leaky_meshes": n_leaky,
        "r2_ceiling": float(1.0 - noise_var / cd.var()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="number of designs")
    ap.add_argument("--out", default="data/synthetic", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary = generate(Path(args.out), args.n, args.seed)
    print(f"wrote {summary['n']} designs")
    print(f"  csv        {summary['csv']}")
    print(f"  meshes     {summary['mesh_dir']}")
    print(
        f"  Cd         mean {summary['cd_mean']:.4f}  sd {summary['cd_std']:.4f}  "
        f"range [{summary['cd_min']:.4f}, {summary['cd_max']:.4f}]"
    )
    print(f"  leaky      {summary['leaky_meshes']} meshes not watertight")
    print(f"  R2 ceiling {summary['r2_ceiling']:.4f}  (noise sd {NOISE_SD})")


if __name__ == "__main__":
    main()
