"""Frozen interfaces between the three workstreams.

FROZEN 2026-10-13. REVISED 2026-10-14 (schema 2.0.0) -- the design parameter
space was replaced with the real one. Avneet raised it; Pranoy and Paramveer
must confirm before building against it. Nothing had been built against 1.0.0
yet, which is why the change was cheap to make.

    Avneet    - data and training        (src/data, src/train)
    Pranoy    - agent and backend        (src/service, src/agent)
    Paramveer - geometry, eval, product  (src/geometry, src/eval, src/optim, web)

There are exactly three seams:

    1. Evaluation  -> Design / Evaluation below, served at POST /evaluate
    2. Geometry    -> params_to_mesh + PARAM_BOUNDS in src/geometry
    3. Run log     -> IterationRecord below, written by the agent, read by charts

Build against these even while the implementations behind them are fake.

--------------------------------------------------------------------------
WHY 2.0.0 CHANGED THE PARAMETER SPACE

1.0.0 invented eight plausible-sounding dimensions (ride_height,
boat_tail_angle, roof_curvature and so on). The real DrivAerNet data does not
have them. Three of the eight have no counterpart in the dataset at all, and
the rest are morph DELTAS from a baseline body in mixed units, not absolute
dimensions. A model trained on the real data could not have been served behind
a 1.0.0 /evaluate without a translation that does not exist.

So the space below is the real one, trimmed by measurement rather than taste.
Of DrivAerNet's 23 morph parameters, permutation importance on a trained model
showed the six side-mirror and door-handle dials and the greenhouse angle
contribute under 0.003 R^2 between them. Dropping them is not a compromise --
it measurably improves the model, most of all where it matters:

    all 23 + body style    random R^2 0.873    held-out-category R^2 0.277
    16 aero + body style   random R^2 0.875    held-out-category R^2 0.479
    8 aero + body style    random R^2 0.845    held-out-category R^2 0.069
    8 aero, no body style  random R^2 0.488    held-out-category R^2 0.281

Body style is load-bearing, not a label. The parameters are deltas from a
DIFFERENT baseline body per style, so the same delta describes a different
shape on a fastback than on an estateback, and a model that cannot tell which
baseline it is perturbing loses 0.39 R^2.

Units: angles in degrees, lengths in millimetres, curvatures dimensionless.
All are offsets from the baseline DrivAer body of the given style; 0.0
everywhere means the unmodified baseline.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Seam 2 lives here as data: the design parameter space.
# Paramveer owns the implementation in src/geometry/params.py; these names and
# bounds are the contract everyone codes against.
#
# The bounds are the DrivAerNet training range rounded OUTWARD to the clean
# values the dataset authors evidently sampled between (the data stops at
# -7.986 and 14.999, so the sampler's box was -8 to 15).
#
# Outward, not inward: rounding in leaves a sliver of real training designs
# the agent can never propose, which wastes data and makes the eval splits
# contain designs the agent could not have reached. The sub-0.1 overhang at
# each edge is negligible, and the OOD score below is what actually guards the
# boundary.
#
# The agent proposes inside these bounds; the surrogate was trained inside
# them; outside is out of distribution by definition -- and being inside them
# is necessary but NOT sufficient, see is_in_distribution below.
# --------------------------------------------------------------------------

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
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

PARAM_NAMES: list[str] = list(PARAM_BOUNDS)

# The three DrivAerNet body styles. Part of the design, not metadata: each has
# its own baseline body, so the parameters above mean different things on each.
# The agent chooses one, and a style outside this list is out of distribution.
BODY_STYLES: list[str] = ["fastback", "notchback", "estateback"]
DEFAULT_BODY_STYLE: str = "fastback"


def default_params() -> dict[str, float]:
    """The unmodified baseline body: every morph delta at zero.

    Note this is NOT the midpoint of the bounds -- the bounds are asymmetric
    because the dataset explored asymmetrically, and zero is the real baseline
    DrivAer geometry, which is the only design in here with a physical meaning
    of its own. Optimisation runs start here.
    """
    return {name: 0.0 for name in PARAM_BOUNDS}


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


def validate_body_style(style: str) -> list[str]:
    if style not in BODY_STYLES:
        return [f"unknown body style '{style}' (expected one of {BODY_STYLES})"]
    return []


# --------------------------------------------------------------------------
# Seam 1: what the surrogate is asked and what it answers.
# --------------------------------------------------------------------------


@dataclass
class Design:
    """One candidate car: a body style plus morph deltas from its baseline."""

    params: dict[str, float]
    body_style: str = DEFAULT_BODY_STYLE
    design_id: str | None = None   # set when the design came from the dataset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evaluation:
    """What the surrogate returns for one design.

    uncertainty is a standard deviation on cd, in the same units as cd.

    in_distribution is the gate the agent must respect: False means the model
    is extrapolating and the number should not be trusted or optimised against.

    IT IS NOT A THRESHOLD ON uncertainty. That was measured and it does not
    work: with one body style held out of training, cd error roughly doubles
    while uncertainty widens about 20%, so a sigma cutoff waves the design
    through. uncertainty answers "how noisy are the labels around here", which
    is not the same question as "have I seen anything like this". Use
    is_in_distribution below, which asks both.

    ood_score is the second signal: distance from the training set, in
    standardised parameter space. Lower is more familiar. None when the serving
    model does not provide one.
    """

    cd: float
    cl: float
    uncertainty: float
    in_distribution: bool
    pressure_summary: dict[str, float] = field(default_factory=dict)
    model_version: str = "stub"
    latency_ms: float = 0.0
    ood_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Region keys used in Evaluation.pressure_summary. Fixed so the viewer and the
# agent prompt can rely on them. Values are mean pressure coefficient.
#
# The real scalar predictor leaves this summary empty. Experimental surface
# fields supply separate region means through /fields; the scalar endpoint and
# current agent loop do not silently substitute synthetic regional values.
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
    body_style: str = DEFAULT_BODY_STYLE
    reject_reason: str | None = None
    reasoning: str = ""          # empty for the baseline optimiser
    best_cd_so_far: float | None = None
    wall_clock_s: float = 0.0
    optimiser: str = "agent"     # "agent" | "random" | "cmaes"
    ood_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Thresholds everyone shares. Tune once, here.
# --------------------------------------------------------------------------

# Operational cutoff for predicted Cd standard deviation. This is a screening
# policy, not a guarantee of interval coverage or an irreducible noise floor.
# Validate empirical coverage on held-out labels separately from this cutoff.
UNCERTAINTY_THRESHOLD: float = 0.013

# Reject a proposal whose distance from the training set exceeds this.
# Mahalanobis-style distance in standardised parameter space; calibrated so
# that a set fraction of held-out training designs pass. Set by
# src/train/ood.py at training time and stored in the checkpoint -- this is the
# fallback for a model that carries no calibrated value.
OOD_SCORE_THRESHOLD: float = 3.0


def is_in_distribution(
    uncertainty: float,
    ood_score: float | None = None,
    body_style: str = DEFAULT_BODY_STYLE,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    ood_threshold: float = OOD_SCORE_THRESHOLD,
) -> tuple[bool, str | None]:
    """The gate. Returns (ok, reason_if_not).

    Three questions, all of which must pass. The reason string goes into
    IterationRecord.reject_reason and into the agent's prompt, because an
    agent told WHY it was rejected adapts, and one told only "no" hammers.
    """
    if body_style not in BODY_STYLES:
        return False, f"unknown body style '{body_style}'"
    if uncertainty > uncertainty_threshold:
        return False, (
            f"uncertainty {uncertainty:.4f} exceeds {uncertainty_threshold:.4f} "
            f"-- the surrogate is unsure about this design"
        )
    if ood_score is not None and ood_score > ood_threshold:
        return False, (
            f"out of distribution (score {ood_score:.2f} > {ood_threshold:.2f}) "
            f"-- this design is unlike anything in the training set, so the "
            f"prediction is extrapolation regardless of its error bar"
        )
    return True, None


SCHEMA_VERSION: str = "2.0.0"
