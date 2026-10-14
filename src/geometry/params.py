"""Seam 2: parameters -> mesh.  OWNER: Paramveer.

Built against contracts 2.0.0: sixteen DrivAerNet morph deltas plus a body
style. Every value is an offset from the baseline DrivAer body of that style,
so all zeros is the unmodified car.

The surrogate never sees this mesh -- the parametric model predicts straight
from the parameters (docs/HANDOFF.md section 5). The mesh has to look right in
the viewer and move the right way for every dial, not reproduce ANSA's morphs.

How the car is built: a side profile (roof line, underbody line, half-width)
is laid out from the parameters and sampled at stations along the car. Each
station gets a rounded-box cross-section that leans inward above the belt
line, the stations are lofted into one closed body, and four wheels are added.
frontal_area() integrates the same cross-sections, so the number and the mesh
cannot disagree.

Axes: x runs front to back from the nose (x = 0), y is lateral, z is up from
the ground (z = 0). Units are metres.
"""

from __future__ import annotations

import argparse
import math
import warnings

import numpy as np

from .. import contracts as _contracts

# --------------------------------------------------------------------------
# The parameter space.
#
# Contracts 2.0.0 lives on Avneet's branch until the team signs it off and it
# merges. Until then main still carries 1.0.0, so this module keeps its own
# copy of the 2.0.0 bounds and uses the contract's as soon as they match.
# test_geometry checks the two agree once 2.0.0 is on main.
# --------------------------------------------------------------------------

_BOUNDS_2_0_0: dict[str, tuple[float, float]] = {
    "B_Ramp_Angle": (-8.0, 15.0),               # deg, underbody rear ramp
    "B_Diffusor_Angle": (-8.0, 15.0),           # deg, rear diffuser
    "B_Trunklid_Angle": (-8.0, 20.0),           # deg, trunk lid
    "D_Rear_Window_Inclination": (-1.8, 2.5),   # morph amount
    "D_Winscreen_Inclination": (-1.8, 2.5),     # morph amount
    "D_Winscreen_Length": (-50.0, 50.0),        # mm
    "D_Rear_Window_Length": (-95.0, 150.0),     # mm
    "E_A_B_C_Pillar_Thickness": (-15.0, 5.0),   # mm
    "E_Fenders_Arch_Offset": (-25.0, 55.0),     # mm
    "G_Trunklid_Curvature": (-0.4, 0.8),        # dimensionless
    "G_Trunklid_Length": (-40.0, 40.0),         # mm
    "H_Front_Bumper_Curvature": (-0.4, 1.0),    # dimensionless
    "H_Front_Bumper_Length": (-25.0, 60.0),     # mm
    "A_Car_Length": (-60.0, 130.0),             # mm, overall
    "A_Car_Width": (-60.0, 150.0),              # mm, overall
    "A_Car_Roof_Height": (-60.0, 90.0),         # mm
}

MORPH_BOUNDS: dict[str, tuple[float, float]] = (
    dict(_contracts.PARAM_BOUNDS)
    if set(_contracts.PARAM_BOUNDS) == set(_BOUNDS_2_0_0)
    else dict(_BOUNDS_2_0_0)
)
BODY_STYLES: list[str] = list(
    getattr(_contracts, "BODY_STYLES", ["fastback", "notchback", "estateback"])
)
DEFAULT_BODY_STYLE: str = getattr(_contracts, "DEFAULT_BODY_STYLE", "fastback")

# --------------------------------------------------------------------------
# The baseline body. Approximate DrivAer dimensions: 4.6 m long, 1.82 m wide
# without mirrors, 1.42 m tall, 2.786 m wheelbase.
# --------------------------------------------------------------------------

_N_STATIONS = 160          # cross-sections along the car
_N_RING = 64               # vertices per cross-section
_EXPONENT = 3.0            # superellipse: 2 is an ellipse, higher is boxier
_SMOOTH_SIGMA = 0.05       # m, rounds the corners of the side profile

_HALF_WIDTH = 0.909
_ROOF_Z = 1.42
_BELT_Z = 0.98             # the greenhouse leans inward above this
_COWL_Z = 0.98             # windscreen base
_HOOD_FRONT = (0.30, 0.76)
_NOSE_TOP_Z = 0.58
_NOSE_BOTTOM_Z = 0.32
_CLEARANCE = 0.14          # flat underbody height
_MIN_CLEARANCE = 0.08
_ROOF_FRONT_X = 2.38
_WINDSCREEN_ANGLE = 28.0   # deg from horizontal
_FRONT_AXLE_X = 0.90
_WHEELBASE = 2.786
_RAMP_ANGLE = 3.0          # deg, baseline underbody ramp
_DIFFUSER_ANGLE = 7.0      # deg, baseline diffuser
_DIFFUSER_LENGTH = 0.40
_TUMBLEHOME = 0.30         # fraction the greenhouse narrows by at the roof

_WHEEL_RADIUS = 0.33
_TYRE_WIDTH = 0.22
_WHEEL_INSET = 0.09        # wheel centre plane, inboard of the body side

# Morph amounts are unitless in the dataset. Assumed: one unit tilts the glass
# three degrees, positive is more raked.
_DEG_PER_INCLINATION = 3.0

# Rear of each body style. roof_end is where the roof meets the rear glass.
_STYLES: dict[str, dict[str, float]] = {
    "fastback": {"roof_end": 2.85, "rear_window_angle": 16.0,
                 "deck_z": 1.05, "deck_length": 0.45, "trunklid_angle": 2.0,
                 "roof_drop": 0.0},
    "notchback": {"roof_end": 3.05, "rear_window_angle": 30.0,
                  "deck_z": 1.03, "deck_length": 0.85, "trunklid_angle": 2.0,
                  "roof_drop": 0.0},
    "estateback": {"roof_end": 4.30, "rear_window_angle": 70.0,
                   "deck_z": 1.00, "deck_length": 0.12, "trunklid_angle": 0.0,
                   "roof_drop": 0.03},
}


# --------------------------------------------------------------------------
# Public API -- signatures shared with the service and the viewer.
# --------------------------------------------------------------------------


def params_to_mesh(params: dict[str, float], body_style: str = DEFAULT_BODY_STYLE):
    """Build a mesh for one design. Returns trimesh.Trimesh.

    Missing parameters are the baseline (0.0). Values outside the bounds are
    clamped so the viewer can still draw the car; the service is what rejects
    them. Names this module does not know are ignored with a warning.
    """
    import trimesh

    shape = _build_profile(_resolve(params), body_style)
    body = _loft(shape)
    wheels = [_wheel(x, side * shape["wheel_y"]) for x in shape["axles"] for side in (-1, 1)]
    return trimesh.util.concatenate([body, *wheels])


def frontal_area(params: dict[str, float], body_style: str = DEFAULT_BODY_STYLE) -> float:
    """Projected frontal area in m^2. Needed to turn drag force into Cd."""
    shape = _build_profile(_resolve(params), body_style)
    dz = 0.002
    z = np.arange(0.0, float(shape["z_top"].max()) + dz, dz) + dz / 2

    # Half-width of the body at each height: the widest station there.
    body_half = _half_width_at(shape, z).max(axis=0)

    # Each wheel projects to a rectangle; union it with the body, per side.
    lo = shape["wheel_y"] - _TYRE_WIDTH / 2
    hi = shape["wheel_y"] + _TYRE_WIDTH / 2
    in_wheel = z <= 2 * _WHEEL_RADIUS
    overlap = np.clip(np.minimum(body_half, hi) - lo, 0.0, hi - lo)
    half = body_half + np.where(in_wheel, (hi - lo) - overlap, 0.0)
    return float(2.0 * half.sum() * dz)


def sample_surface(
    params: dict[str, float],
    n_points: int = 4096,
    body_style: str = DEFAULT_BODY_STYLE,
    seed: int = 0,
) -> np.ndarray:
    """Point cloud (n_points, 3). This is the surrogate's input.

    Matches src.data.dataset.sample_mesh on Avneet's branch: area-weighted
    surface sampling, normalised by the MESH's bounding-box centre and max
    vertex radius. Seeded, so the same design gives the same cloud.
    """
    import trimesh

    mesh = params_to_mesh(params, body_style)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    except TypeError:  # older trimesh has no seed kwarg
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    centre = mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(mesh.vertices - centre, axis=1).max())
    return normalize_points(np.asarray(points), centre=centre, radius=radius)


def normalize_points(
    points: np.ndarray,
    centre: np.ndarray | None = None,
    radius: float | None = None,
) -> np.ndarray:
    """Center on the bounding-box centre, scale to the unit sphere.

    Pass the mesh's centre and radius to match training exactly; without
    them both come from the points themselves.
    """
    points = np.asarray(points, dtype=np.float64)
    if centre is None:
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
    if radius is None:
        radius = float(np.linalg.norm(points - centre, axis=1).max())
    out = (points - np.asarray(centre)) / max(float(radius), 1e-9)
    return np.ascontiguousarray(out, dtype=np.float32)


def random_params(rng=None) -> dict[str, float]:
    """Uniform sample from the design box. Seeds optimisation runs."""
    rng = rng or np.random.default_rng()
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in MORPH_BOUNDS.items()}


# --------------------------------------------------------------------------
# Building the shape.
# --------------------------------------------------------------------------


def _resolve(params: dict[str, float]) -> dict[str, float]:
    """Every morph parameter, clamped into its bounds. Missing means 0.0."""
    unknown = sorted(set(params) - set(MORPH_BOUNDS))
    if unknown:
        warnings.warn(
            f"geometry ignores parameters it does not know: {unknown} "
            "(expected the contracts 2.0.0 names)",
            stacklevel=3,
        )
    out = {}
    for name, (lo, hi) in MORPH_BOUNDS.items():
        value = float(params.get(name, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"'{name}' is not a finite number: {value}")
        out[name] = min(max(value, lo), hi)
    return out


def _build_profile(p: dict[str, float], body_style: str) -> dict:
    """Side profile and plan-view width, sampled at stations along x."""
    if body_style not in _STYLES:
        raise ValueError(f"unknown body style '{body_style}' (expected one of {BODY_STYLES})")
    style = _STYLES[body_style]

    front = p["H_Front_Bumper_Length"] / 1000      # nose moves forward
    stretch = p["A_Car_Length"] / 1000             # cabin gets longer
    roof_z = _ROOF_Z + p["A_Car_Roof_Height"] / 1000
    half_width = _HALF_WIDTH + p["A_Car_Width"] / 2000

    # Windscreen: pivots on the roof front edge.
    ws_angle = math.radians(_WINDSCREEN_ANGLE - _DEG_PER_INCLINATION * p["D_Winscreen_Inclination"])
    ws_len = (roof_z - _COWL_Z) / math.sin(math.radians(_WINDSCREEN_ANGLE)) + p["D_Winscreen_Length"] / 1000
    roof_front_x = _ROOF_FRONT_X + front
    cowl = (roof_front_x - ws_len * math.cos(ws_angle), roof_z - ws_len * math.sin(ws_angle))

    # Rear glass: pivots on the roof rear edge.
    roof_end_x = style["roof_end"] + front + stretch
    roof_end_z = roof_z - style["roof_drop"]
    rw_base = style["rear_window_angle"]
    rw_angle = math.radians(rw_base - _DEG_PER_INCLINATION * p["D_Rear_Window_Inclination"])
    rw_len = (roof_end_z - style["deck_z"]) / math.sin(math.radians(rw_base)) + p["D_Rear_Window_Length"] / 1000
    deck_start = (roof_end_x + rw_len * math.cos(rw_angle), roof_end_z - rw_len * math.sin(rw_angle))

    # Trunk lid: tilts and bows between the rear glass and the tail.
    deck_len = max(style["deck_length"] + p["G_Trunklid_Length"] / 1000, 0.05)
    length = deck_start[0] + deck_len
    lid = math.tan(math.radians(style["trunklid_angle"] + p["B_Trunklid_Angle"]))
    s = np.linspace(0.0, 1.0, 9)[1:]
    deck_x = deck_start[0] + s * deck_len
    deck_z = deck_start[1] + s * deck_len * lid + 0.05 * p["G_Trunklid_Curvature"] * np.sin(np.pi * s)

    roof_s = np.linspace(0.0, 1.0, 7)
    roof_x = roof_front_x + roof_s * (roof_end_x - roof_front_x)
    roof_line = roof_z - roof_s * style["roof_drop"] + 0.015 * np.sin(np.pi * roof_s)

    top_x = np.concatenate([[0.0, _HOOD_FRONT[0] + front, cowl[0]], roof_x, [deck_start[0]], deck_x])
    top_z = np.concatenate([[_NOSE_TOP_Z, _HOOD_FRONT[1], cowl[1]], roof_line, [deck_start[1]], deck_z])

    # Underbody: flat, then the rear ramp, then the diffuser.
    front_axle = _FRONT_AXLE_X + front
    rear_axle = front_axle + _WHEELBASE + stretch
    diffuser_x = length - _DIFFUSER_LENGTH
    ramp_x = min(rear_axle + 0.15, diffuser_x)
    ramp = math.tan(math.radians(_RAMP_ANGLE + p["B_Ramp_Angle"]))
    diffuser = math.tan(math.radians(_DIFFUSER_ANGLE + p["B_Diffusor_Angle"]))
    ramp_end_z = _CLEARANCE + (diffuser_x - ramp_x) * ramp
    bot_x = np.array([0.0, 0.45 + front, ramp_x, diffuser_x, length])
    bot_z = np.array([_NOSE_BOTTOM_Z, _CLEARANCE, _CLEARANCE, ramp_end_z,
                      ramp_end_z + _DIFFUSER_LENGTH * diffuser])

    x = np.linspace(0.0, length, _N_STATIONS)
    z_top = _smooth_profile(top_x, top_z, x)
    z_bot = _smooth_profile(bot_x, bot_z, x)
    z_bot = np.clip(z_bot, _MIN_CLEARANCE, z_top - 0.20)

    # Plan view: blunt rounded nose, slight taper into the tail.
    nose_len = 0.30 + 0.15 * p["H_Front_Bumper_Curvature"]
    nose = 0.55 + 0.45 * np.sqrt(np.clip(x / nose_len, 0.0, 1.0))
    tail = 1.0 - 0.07 * np.clip((x - (length - 0.7)) / 0.7, 0.0, 1.0) ** 2
    width = half_width * nose * tail

    # Fender flares, faded out above the belt line in _half_width_scale.
    flare = p["E_Fenders_Arch_Offset"] / 1000
    bulge = flare * (np.exp(-((x - front_axle) / 0.3) ** 2) + np.exp(-((x - rear_axle) / 0.3) ** 2))

    return {
        "x": x, "z_top": z_top, "z_bot": z_bot, "width": width, "bulge": bulge,
        "tumblehome": _TUMBLEHOME - 0.003 * p["E_A_B_C_Pillar_Thickness"],
        "axles": (front_axle, rear_axle),
        "wheel_y": half_width - _WHEEL_INSET,
        "length": length,
    }


def _smooth_profile(kx: np.ndarray, kz: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Piecewise-linear through the keypoints, Gaussian-rounded corners."""
    kx = np.maximum.accumulate(np.asarray(kx, dtype=float))
    fine = np.linspace(x[0], x[-1], 1200)
    z = np.interp(fine, kx, kz)
    step = fine[1] - fine[0]
    half = int(math.ceil(3 * _SMOOTH_SIGMA / step))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step / _SMOOTH_SIGMA) ** 2)
    kernel /= kernel.sum()
    z = np.convolve(np.pad(z, half, mode="edge"), kernel, mode="valid")
    return np.interp(x, fine, z)


def _half_width_scale(shape: dict, z: np.ndarray) -> np.ndarray:
    """Tumblehome above the belt, fender flare below it. z is (stations, k)."""
    z_top = shape["z_top"][:, None]
    above = np.clip((z - _BELT_Z) / np.maximum(z_top - _BELT_Z, 1e-6), 0.0, 1.0)
    above = np.where(z_top > _BELT_Z, above, 0.0)
    lean = 1.0 - shape["tumblehome"] * above ** 1.5
    below = np.clip((_BELT_Z - z) / 0.15, 0.0, 1.0)
    flare = 1.0 + (shape["bulge"] / shape["width"])[:, None] * below
    return lean * flare


def _half_width_at(shape: dict, z: np.ndarray) -> np.ndarray:
    """Body half-width at each station and height, (stations, len(z)). 0 outside."""
    zc = ((shape["z_top"] + shape["z_bot"]) / 2)[:, None]
    hz = ((shape["z_top"] - shape["z_bot"]) / 2)[:, None]
    u = np.abs((z[None, :] - zc) / hz)
    base = shape["width"][:, None] * np.clip(1.0 - u ** _EXPONENT, 0.0, None) ** (1.0 / _EXPONENT)
    zz = np.broadcast_to(z[None, :], base.shape)
    return np.where(u <= 1.0, base * _half_width_scale(shape, zz), 0.0)


def _loft(shape: dict):
    """Close the stations into one watertight body, normals outward."""
    import trimesh

    n, m = _N_STATIONS, _N_RING
    t = np.linspace(0.0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    ring_y = np.sign(c) * np.abs(c) ** (2 / _EXPONENT)
    ring_z = np.sign(s) * np.abs(s) ** (2 / _EXPONENT)

    zc = ((shape["z_top"] + shape["z_bot"]) / 2)[:, None]
    hz = ((shape["z_top"] - shape["z_bot"]) / 2)[:, None]
    z = zc + hz * ring_z[None, :]
    y = shape["width"][:, None] * ring_y[None, :] * _half_width_scale(shape, z)
    x = np.broadcast_to(shape["x"][:, None], z.shape)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    front_centre = [shape["x"][0], 0.0, zc[0, 0]]
    back_centre = [shape["x"][-1], 0.0, zc[-1, 0]]
    vertices = np.vstack([vertices, front_centre, back_centre])
    front_i, back_i = n * m, n * m + 1

    i, j = np.meshgrid(np.arange(n - 1), np.arange(m), indexing="ij")
    a = i * m + j
    b = i * m + (j + 1) % m
    c_ = a + m
    d = b + m
    sides = np.concatenate([np.stack([a, b, d], -1), np.stack([a, d, c_], -1)]).reshape(-1, 3)

    j = np.arange(m)
    front_cap = np.stack([np.full(m, front_i), (j + 1) % m, j], -1)
    last = (n - 1) * m
    back_cap = np.stack([np.full(m, back_i), last + j, last + (j + 1) % m], -1)

    faces = np.vstack([sides, front_cap, back_cap])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _wheel(x: float, y: float):
    import trimesh

    wheel = trimesh.creation.cylinder(radius=_WHEEL_RADIUS, height=_TYRE_WIDTH, sections=48)
    wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    wheel.apply_translation([x, y, _WHEEL_RADIUS])
    return wheel


# --------------------------------------------------------------------------
# python -m src.geometry.params --body-style notchback --out car.stl
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export one design as a mesh.")
    parser.add_argument("--body-style", default=DEFAULT_BODY_STYLE, choices=BODY_STYLES)
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                        help="morph deltas, e.g. A_Car_Roof_Height=40")
    parser.add_argument("--out", default=None, help="write .stl / .obj / .glb")
    args = parser.parse_args(argv)

    params = {}
    for item in args.set:
        name, _, value = item.partition("=")
        params[name] = float(value)

    mesh = params_to_mesh(params, args.body_style)
    lo, hi = mesh.bounds
    print(f"{args.body_style}: faces={len(mesh.faces)} watertight={mesh.is_watertight} "
          f"L={hi[0] - lo[0]:.3f} W={hi[1] - lo[1]:.3f} H={hi[2] - lo[2]:.3f} m "
          f"frontal_area={frontal_area(params, args.body_style):.3f} m^2")
    if args.out:
        mesh.export(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
