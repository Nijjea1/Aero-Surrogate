"""Seam 2: parameters -> mesh.  OWNER: Paramveer.  DUE: Oct 14 EOD.

Pranoy's service and the 3D viewer both import from here, so this blocks two
people. Keep these signatures exactly as they are -- the bodies are yours.
"""

from __future__ import annotations

import numpy as np

from ..contracts import PARAM_BOUNDS, clamp_params


def params_to_mesh(params: dict[str, float]):
    """Build a mesh for one design. Returns trimesh.Trimesh."""
    raise NotImplementedError("TODO Paramveer: Oct 14")


def frontal_area(params: dict[str, float]) -> float:
    """Projected frontal area in m^2. Needed to turn drag force into Cd."""
    raise NotImplementedError("TODO Paramveer")


def sample_surface(params: dict[str, float], n_points: int = 4096) -> np.ndarray:
    """Point cloud (n_points, 3). This is the surrogate's input."""
    raise NotImplementedError("TODO Paramveer")


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center on centroid, scale to unit sphere."""
    raise NotImplementedError("TODO Paramveer")


def random_params(rng=None) -> dict[str, float]:
    """Uniform sample from the design box. Seeds optimisation runs."""
    rng = rng or np.random.default_rng()
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PARAM_BOUNDS.items()}
