"""Baseline optimisers the agent is measured against.  OWNER: Paramveer.

    python -m src.optim.baseline --method cmaes --iterations 25

This has to be a FAIR fight: same parameter space, same evaluation budget, same
uncertainty gate, same starting design. If we rig it a judge will ask.

If the agent does not beat these, we report it anyway. "The LLM matched CMA-ES
but produced an explainable trace and respected stated constraints" is an
honest finding. Hiding the comparison is what would hurt us.
"""

from __future__ import annotations

import numpy as np

from ..contracts import PARAM_BOUNDS, PARAM_NAMES


def to_vector(params: dict[str, float]) -> np.ndarray:
    """Map the param dict into the unit cube so every axis is comparable."""
    raise NotImplementedError("TODO Paramveer")


def to_params(vec: np.ndarray) -> dict[str, float]:
    raise NotImplementedError("TODO Paramveer")


def random_search(iterations: int, rng):
    raise NotImplementedError("TODO Paramveer: Oct 21")


def cmaes(iterations: int, rng, scores: dict):
    """`pip install cma` if you want the real thing; keep this interface."""
    raise NotImplementedError("TODO Paramveer: Oct 22")


def run(method: str, iterations: int, run_id: str, seed: int = 0) -> None:
    """Gated designs score as infinitely bad, same as the agent experiences."""
    raise NotImplementedError("TODO Paramveer")
