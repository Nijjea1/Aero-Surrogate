"""The thing behind POST /evaluate.  OWNER: Pranoy.

Two implementations, same interface:
    StubPredictor  - plausible numbers, no model, no GPU. BUILD THIS FIRST
                     (Oct 13) so Paramveer is not blocked on Avneet.
    TorchPredictor - loads one of Avneet's checkpoints. Swap it in Oct 15.

Make the stub's uncertainty RISE toward the edges of the design box, or the
uncertainty gate has nothing to fire on and cannot be tested.

Schema 2.0.0: sixteen DrivAerNet morph deltas plus a body style. All-zero
params is the unmodified baseline body of that style.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..contracts import (
    BODY_STYLES, Design, Evaluation, PARAM_BOUNDS, PRESSURE_REGIONS,
    is_in_distribution, validate_body_style,
)


def validate_design_params(params: object) -> list[str]:
    """Validate without coercing strings/bools or silently clipping proposals."""
    if not isinstance(params, Mapping):
        return ["params must be an object containing every design parameter"]
    problems = [f"missing parameter '{name}'" for name in PARAM_BOUNDS if name not in params]
    for name, value in params.items():
        if name not in PARAM_BOUNDS:
            problems.append(f"unknown parameter '{name}'")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"'{name}' must be a finite number")
            continue
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            problems.append(f"'{name}' must be a finite number")
            continue
        lo, hi = PARAM_BOUNDS[name]
        if not lo <= value <= hi:
            problems.append(f"'{name}' = {value} outside [{lo}, {hi}]")
    return problems


class BasePredictor:
    name = "base"

    def evaluate(self, design: Design) -> Evaluation:
        raise NotImplementedError


class StubPredictor(BasePredictor):
    """Synthetic integration fixture, never an aerodynamic accuracy claim.

    All outputs are deterministic functions of the design. Every parameter and
    the body style affect cd. Edge uncertainty and the distance-style ood_score
    are engineered test fixtures, not calibrated OOD detection: a single dial at
    its limit trips the uncertainty check, while many dials pushed towards their
    limits together trip the ood_score check, mirroring the two failure modes
    the real gate guards against. Zero latency is a fixture value.
    """

    name = "stub"

    def evaluate(self, design: Design) -> Evaluation:
        problems = validate_design_params(design.params) + validate_body_style(design.body_style)
        if problems:
            raise ValueError("; ".join(problems))
        position = {name: _position(name, design.params[name]) for name in PARAM_BOUNDS}
        # An intentionally smooth objective so the offline loop has a useful
        # target. These coefficients are invented, not fitted CFD results.
        cd = 0.27 + _SYNTHETIC_STYLE_OFFSET[design.body_style] + sum(
            0.025 * (position[name] - _position(name, target)) ** 2
            for name, target in _SYNTHETIC_CD_OPTIMUM.items()
        )
        cl = -0.04 + 0.06 * (position["A_Car_Roof_Height"] - 0.5) - 0.04 * (
            position["B_Diffusor_Angle"] - 0.5
        )
        edge_distance = max(abs(2.0 * value - 1.0) for value in position.values())
        uncertainty = 0.003 + 0.020 * edge_distance ** 2
        # Euclidean distance from the centre of a uniformly sampled box, in
        # standard deviations (a uniform variable has sd 1/sqrt(12) of its range).
        ood_score = math.sqrt(sum(((value - 0.5) * math.sqrt(12.0)) ** 2
                                  for value in position.values()))
        in_distribution, _ = is_in_distribution(uncertainty, ood_score, design.body_style)
        pressure_summary = {
            region: baseline + 0.12 * (position[name] - 0.5)
            for region, (name, baseline) in _SYNTHETIC_REGION_DRIVER.items()
        }
        return Evaluation(
            cd=cd,
            cl=cl,
            uncertainty=uncertainty,
            in_distribution=in_distribution,
            pressure_summary=pressure_summary,
            model_version="stub-v2-synthetic",
            latency_ms=0.0,
            ood_score=ood_score,
        )


def _position(name: str, value: float) -> float:
    """Where value sits in its bound: 0.0 at the lower limit, 1.0 at the upper."""
    lo, hi = PARAM_BOUNDS[name]
    return (float(value) - lo) / (hi - lo)


# Invented synthetic optimum per parameter, in the parameter's own units. Keyed
# by name so a contract change fails loudly instead of zip silently truncating.
_SYNTHETIC_CD_OPTIMUM: dict[str, float] = {
    "B_Ramp_Angle": -3.0,
    "B_Diffusor_Angle": 4.0,
    "B_Trunklid_Angle": 6.0,
    "D_Rear_Window_Inclination": 0.6,
    "D_Winscreen_Inclination": 0.8,
    "D_Winscreen_Length": -10.0,
    "D_Rear_Window_Length": 30.0,
    "E_A_B_C_Pillar_Thickness": -8.0,
    "E_Fenders_Arch_Offset": -12.0,
    "G_Trunklid_Curvature": 0.3,
    "G_Trunklid_Length": 15.0,
    "H_Front_Bumper_Curvature": 0.5,
    "H_Front_Bumper_Length": -10.0,
    "A_Car_Length": 60.0,
    "A_Car_Width": -30.0,
    "A_Car_Roof_Height": -25.0,
}
_SYNTHETIC_STYLE_OFFSET: dict[str, float] = {
    "fastback": 0.0, "notchback": 0.004, "estateback": 0.012,
}
_SYNTHETIC_REGION_DRIVER: dict[str, tuple[str, float]] = {
    "front_fascia": ("H_Front_Bumper_Curvature", 0.70),
    "hood": ("H_Front_Bumper_Length", -0.10),
    "windscreen": ("D_Winscreen_Inclination", 0.25),
    "roof": ("A_Car_Roof_Height", -0.45),
    "rear_window": ("D_Rear_Window_Inclination", -0.25),
    "rear_fascia": ("B_Trunklid_Angle", -0.30),
    "underbody": ("B_Ramp_Angle", -0.15),
    "wheel_wells": ("E_Fenders_Arch_Offset", 0.10),
}
if (set(_SYNTHETIC_CD_OPTIMUM) != set(PARAM_BOUNDS)
        or set(_SYNTHETIC_STYLE_OFFSET) != set(BODY_STYLES)
        or set(_SYNTHETIC_REGION_DRIVER) != set(PRESSURE_REGIONS)):
    raise RuntimeError("synthetic stub tables are out of step with src/contracts.py")


class TorchPredictor(BasePredictor):
    name = "torch"

    def __init__(self, checkpoint: str, device: str = "cpu"):
        raise NotImplementedError(
            "Real checkpoint inference is unavailable until the training adapter "
            "is integrated. Refusing to substitute synthetic predictions."
        )

    def evaluate(self, design: Design) -> Evaluation:
        raise NotImplementedError


def load_predictor(checkpoint: str | None = None) -> BasePredictor:
    """Torch predictor when a checkpoint is given, else the stub."""
    if checkpoint is None:
        return StubPredictor()
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint must be a non-empty path or None")
    return TorchPredictor(checkpoint)
