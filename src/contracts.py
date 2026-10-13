"""Frozen interfaces between the three workstreams.

FROZEN 2026-10-13. Changing anything in this file requires all three of us
to agree in the group chat, because every other module imports from here.

    Avneet    - data and training        (src/data, src/train)
    Pranoy    - agent and backend        (src/service, src/agent)
    Paramveer - geometry, eval, product  (src/geometry, src/eval, src/optim, web)

There are exactly three seams:

    1. Evaluation  -> Design / Evaluation below, served at POST /evaluate
    2. Geometry    -> params_to_mesh + PARAM_BOUNDS in src/geometry
    3. Run log     -> IterationRecord below, written by the agent, read by charts

Build against these even while the implementations behind them are fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Seam 2 lives here as data: the design parameter space.
# Paramveer owns the implementation in src/geometry/params.py; these names and
# bounds are the contract everyone codes against.
#
# Values are normalised to the ranges below. The agent proposes inside these
# bounds; the surrogate was trained inside them; anything outside is by
# definition out of distribution.
# --------------------------------------------------------------------------

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "ride_height": (0.10, 0.20),        # m, floor to underbody
    "rear_deck_angle": (-5.0, 20.0),    # deg, positive = upswept
    "windscreen_rake": (25.0, 65.0),    # deg from vertical
    "boat_tail_angle": (0.0, 18.0),     # deg, rear taper per side
    "front_overhang": (0.70, 1.05),     # m
    "roof_curvature": (0.00, 0.35),     # dimensionless
    "rear_diffuser_angle": (0.0, 15.0),  # deg
    "track_width": (1.50, 1.70),        # m
}

PARAM_NAMES: list[str] = list(PARAM_BOUNDS)


def default_params() -> dict[str, float]:
    """Midpoint of every bound. The starting design for optimisation runs."""
    return {k: (lo + hi) / 2.0 for k, (lo, hi) in PARAM_BOUNDS.items()}


def clamp_params(params: dict[str, float]) -> dict[str, float]:
    """Clamp a proposal into the valid box. Never silently drop a key."""
    out: dict[str, float] = {}
    for name, (lo, hi) in PARAM_BOUNDS.items():
        if name not in params:
            raise KeyError(f"missing design parameter: {name}")
        out[name] = float(min(max(params[name], lo), hi))
    return out


def validate_params(params: dict[str, float]) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid."""
    problems: list[str] = []
    for name in PARAM_BOUNDS:
        if name not in params:
            problems.append(f"missing parameter '{name}'")
    for name, value in params.items():
        if name not in PARAM_BOUNDS:
            problems.append(f"unknown parameter '{name}'")
            continue
        lo, hi = PARAM_BOUNDS[name]
        if not (lo <= float(value) <= hi):
            problems.append(f"'{name}' = {value} outside [{lo}, {hi}]")
    return problems


# --------------------------------------------------------------------------
# Seam 1: what the surrogate is asked and what it answers.
# --------------------------------------------------------------------------


@dataclass
class Design:
    """One candidate car, expressed in the parameter space above."""

    params: dict[str, float]
    design_id: str | None = None   # set when the design came from the dataset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evaluation:
    """What the surrogate returns for one design.

    uncertainty is a standard deviation on cd, in the same units as cd.
    in_distribution is the gate the agent must respect: False means the model
    is extrapolating and the number should not be trusted or optimised against.
    """

    cd: float
    cl: float
    uncertainty: float
    in_distribution: bool
    pressure_summary: dict[str, float] = field(default_factory=dict)
    model_version: str = "stub"
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Region keys used in Evaluation.pressure_summary. Fixed so the viewer and the
# agent prompt can rely on them. Values are mean pressure coefficient.
PRESSURE_REGIONS: list[str] = [
    "front_fascia",
    "hood",
    "windscreen",
    "roof",
    "rear_window",
    "rear_fascia",
    "underbody",
    "wheel_wells",
]


# --------------------------------------------------------------------------
# Seam 3: the run log. Pranoy appends one of these per agent step;
# Paramveer's charts and the eval harness read them.
# --------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """One step of an optimisation run, agent or baseline optimiser alike."""

    run_id: str
    iteration: int
    params: dict[str, float]
    cd: float
    cl: float
    uncertainty: float
    in_distribution: bool
    accepted: bool               # False when the uncertainty gate rejected it
    reject_reason: str | None = None
    reasoning: str = ""          # empty for the baseline optimiser
    best_cd_so_far: float | None = None
    wall_clock_s: float = 0.0
    optimiser: str = "agent"     # "agent" | "random" | "cmaes"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Thresholds everyone shares. Tune once, here.
# --------------------------------------------------------------------------

# Reject a proposal whose cd std dev exceeds this. Calibrate against the
# reliability diagram once the real model exists; this is a placeholder.
UNCERTAINTY_THRESHOLD: float = 0.015

SCHEMA_VERSION: str = "1.0.0"
