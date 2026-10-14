"""Mesh payload for the 3D viewer.  OWNER: Paramveer.

    mesh_payload(params, body_style) -> dict, JSON-ready

The browser gets the car's triangles plus two per-vertex fields:

    region  index into PRESSURE_REGIONS (-1 for the sides, which no region
            covers), so the viewer can paint the service's pressure_summary
            onto the surface it describes.
    cp      an ILLUSTRATIVE pressure coefficient from the surface angle to the
            oncoming flow. It is not a prediction and the viewer labels it so.
            Stage 2's predicted surface pressure replaces it.

web/serve.py serves this locally. If the hosted service wants it, a route is
one line: return mesh_payload(request.params, request.body_style).
"""

from __future__ import annotations

import base64

import numpy as np

from .params import (
    DEFAULT_BODY_STYLE,
    MORPH_BOUNDS,
    _BELT_Z,
    _TYRE_WIDTH,
    _WHEEL_RADIUS,
    _build_profile,
    _loft,
    _resolve,
    _wheel,
    frontal_area,
)

# Same keys and order as contracts.PRESSURE_REGIONS (1.0.0 and 2.0.0 alike).
REGIONS: list[str] = [
    "front_fascia", "hood", "windscreen", "roof",
    "rear_window", "rear_fascia", "underbody", "wheel_wells",
]
_R = {name: i for i, name in enumerate(REGIONS)}
_NONE = -1


def mesh_payload(params: dict[str, float], body_style: str = DEFAULT_BODY_STYLE) -> dict:
    import trimesh

    shape = _build_profile(_resolve(params), body_style)
    body = _loft(shape, split_caps=True)
    wheels = trimesh.util.concatenate(
        [_wheel(x, side * shape["wheel_y"]) for x in shape["axles"] for side in (-1, 1)]
    )

    body_normals = vertex_normals(body.vertices, body.faces)
    body_region = classify_regions(body.vertices, body_normals, shape["landmarks"])
    wheel_region = np.full(len(wheels.vertices), _R["wheel_wells"], dtype=np.int8)

    vertices = np.vstack([body.vertices, wheels.vertices])
    normals = np.vstack([body_normals, vertex_normals(wheels.vertices, wheels.faces)])
    faces = np.vstack([body.faces, wheels.faces + len(body.vertices)])
    region = np.concatenate([body_region, wheel_region])
    cp = np.concatenate([
        illustrative_cp(body_normals, convexity(body.vertices, body.faces, body_normals),
                        body.vertices[:, 2]),
        illustrative_cp(normals[len(body.vertices):]),
    ])

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return {
        "body_style": body_style,
        "vertices": _b64(vertices.astype(np.float32)),
        "faces": _b64(faces.astype(np.uint32)),
        "region": _b64(region.astype(np.int8)),
        "cp": _b64(cp.astype(np.float32)),
        "regions": REGIONS,
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
        "bounds": [lo.tolist(), hi.tolist()],
        "length_m": float(hi[0] - lo[0]),
        "width_m": float(hi[1] - lo[1]),
        "height_m": float(hi[2] - lo[2]),
        "frontal_area_m2": frontal_area(params, body_style),
        # The baseline body's area, so drag forces compare like with like.
        "baseline_frontal_area_m2": frontal_area({}, body_style),
        "ignored_params": sorted(set(params) - set(MORPH_BOUNDS)),
        "flow": flow_outline(shape),
    }


def flow_outline(shape: dict, n: int = 96) -> dict:
    """The outline the airflow animation steers around, in the mesh's frame
    (x nose to tail, z up): roof and underbody lines and the plan half-width
    at `n` stations, the surface landmarks, and where the wheels are."""
    x = np.linspace(float(shape["x"][0]), float(shape["x"][-1]), n)
    pick = lambda key: np.round(np.interp(x, shape["x"], shape[key]), 4).tolist()  # noqa: E731
    return {
        "x": np.round(x, 4).tolist(),
        "z_top": pick("z_top"),
        "z_bot": pick("z_bot"),
        "half_width": pick("width"),
        "landmarks": {k: round(float(v), 4) for k, v in shape["landmarks"].items()},
        "axles": [round(float(a), 4) for a in shape["axles"]],
        "wheel_y": round(float(shape["wheel_y"]), 4),
        "wheel_radius": _WHEEL_RADIUS,
        "tyre_width": _TYRE_WIDTH,
    }


def classify_regions(vertices: np.ndarray, normals: np.ndarray, marks: dict) -> np.ndarray:
    """Label each body vertex with a pressure region, or -1 for the sides."""
    x, z = vertices[:, 0], vertices[:, 2]
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    out = np.full(len(vertices), _NONE, dtype=np.int8)

    top = nz >= 0.2
    bands = [
        (x < marks["cowl"], "hood"),
        ((x >= marks["cowl"]) & (x < marks["roof_front"]), "windscreen"),
        ((x >= marks["roof_front"]) & (x < marks["roof_end"]), "roof"),
        (x >= marks["roof_end"], "rear_window"),   # rear glass and trunk lid
    ]
    for mask, name in bands:
        out[top & mask] = _R[name]

    # Steep rear glass (an estate's) faces backward more than up.
    rear_glass = (x >= marks["roof_end"]) & (x < marks["deck_start"]) & (z > _BELT_Z) & (np.abs(ny) < 0.75)
    out[rear_glass] = _R["rear_window"]

    out[(nx > 0.5) & (x >= marks["deck_start"] - 0.02) & ~top] = _R["rear_fascia"]
    out[(x < marks["hood_front"] + 0.1) & (nz < 0.6)] = _R["front_fascia"]
    out[(nx < -0.5) & (x < marks["cowl"]) & ~top] = _R["front_fascia"]
    out[nz < -0.5] = _R["underbody"]
    out[(np.abs(ny) > 0.75) & (out != _R["underbody"])] = _NONE
    return out


def illustrative_cp(normals: np.ndarray, kappa: np.ndarray | None = None,
                    z: np.ndarray | None = None) -> np.ndarray:
    """Pressure coefficient from the surface shape and the flow (+x). Not a model.

    Forward-facing surfaces take the potential-flow pattern: Cp = 1 at a
    stagnation point, easing to mild suction where the flow runs parallel.
    Where attached flow wraps over a convex edge (roof front, A-pillars) it
    speeds up and the pressure drops; in a concave corner (the cowl) it rises.
    Flow over the top of the car is faster the higher it goes, so upward
    faces get more suction with height: a little on the hood, most on the roof.
    Rear-facing surfaces sit in separated flow at a mild base pressure.
    The underbody, in the ground's shadow, is damped toward slight suction.
    """
    nx, nz = normals[:, 0], normals[:, 2]
    facing = -nx
    front = 1.0 - 1.1 * (1.0 - facing ** 2)
    rear = -0.3 + 0.1 * nx ** 2
    cp = np.where(facing > 0, front, rear)
    if kappa is not None:
        attached = np.clip((facing + 0.4) / 0.4, 0.0, 1.0)
        cp = cp - 0.12 * np.clip(kappa, -6.0, 14.0) * attached
    if z is not None:
        elevation = np.clip((z - 0.6) / 0.8, 0.0, 1.0)
        along = 1.0 - np.clip(facing / 0.35, 0.0, 1.0)   # glass facing the flow is exempt
        cp = cp - 0.35 * np.clip(nz, 0.0, 1.0) * elevation * along
    cp = np.where(nz < -0.5, 0.4 * cp - 0.15, cp)
    return np.clip(cp, -1.0, 1.0)


def convexity(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
              smoothing: int = 3) -> np.ndarray:
    """Signed curvature per vertex, 1/m: positive on convex edges, negative in corners.

    From how far the neighbours' centroid sits below the tangent plane,
    then averaged over the neighbourhood so single-row creases do not flicker.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    pairs = [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]
    nbr_sum = np.zeros_like(v)
    nbr_d2 = np.zeros(len(v))
    count = np.zeros(len(v))
    for a, b in pairs:
        np.add.at(nbr_sum, f[:, a], v[f[:, b]])
        np.add.at(nbr_d2, f[:, a], np.sum((v[f[:, b]] - v[f[:, a]]) ** 2, axis=1))
        np.add.at(count, f[:, a], 1)
    count = np.maximum(count, 1)
    delta = nbr_sum / count[:, None] - v
    height = np.einsum("ij,ij->i", delta, normals)
    kappa = -2.0 * height / np.maximum(nbr_d2 / count, 1e-12)
    for _ in range(smoothing):
        total = np.zeros(len(v))
        for a, b in pairs:
            np.add.at(total, f[:, a], kappa[f[:, b]])
        kappa = 0.5 * kappa + 0.5 * total / count
    return kappa


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, without trimesh's scipy dependency."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    face_n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], face_n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")
