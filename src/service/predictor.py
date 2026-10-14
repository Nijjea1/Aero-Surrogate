"""The thing behind POST /evaluate.  OWNER: Pranoy.

Two implementations, same interface:
    StubPredictor  - plausible numbers, no model, no GPU. BUILD THIS FIRST
                     (Oct 13) so Paramveer is not blocked on Avneet.
    TorchPredictor - loads one of Avneet's checkpoints. Swap it in Oct 15.

Make the stub's uncertainty RISE toward the edges of the design box, or the
uncertainty gate has nothing to fire on and cannot be tested.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..contracts import Design, Evaluation, PARAM_BOUNDS, PRESSURE_REGIONS


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

    All outputs are deterministic functions of normalised parameters. Edge
    uncertainty is an engineered test fixture, not calibrated OOD detection.
    Zero latency is a fixture value, not a measured inference speed.
    """

    name = "stub"

    def evaluate(self, design: Design) -> Evaluation:
        problems = validate_design_params(design.params)
        if problems:
            raise ValueError("; ".join(problems))
        normalised = {
            name: (float(design.params[name]) - lo) / (hi - lo)
            for name, (lo, hi) in PARAM_BOUNDS.items()
        }
        # An intentionally smooth objective so the offline loop has a useful
        # target. These coefficients are invented, not fitted CFD results.
        targets = (0.40, 0.35, 0.65, 0.65, 0.45, 0.55, 0.45, 0.40)
        cd = 0.27 + sum(
            0.025 * (value - target) ** 2
            for value, target in zip(normalised.values(), targets)
        )
        cl = -0.04 + 0.06 * (normalised["ride_height"] - 0.5) - 0.04 * (
            normalised["rear_diffuser_angle"] - 0.5
        )
        edge_distance = max(abs(2.0 * value - 1.0) for value in normalised.values())
        uncertainty = 0.003 + 0.020 * edge_distance ** 2
        region_baselines = (0.70, -0.10, 0.25, -0.45, -0.25, -0.30, -0.15, 0.10)
        pressure_summary = {
            region: baseline + 0.12 * (value - 0.5)
            for region, baseline, value in zip(
                PRESSURE_REGIONS, region_baselines, normalised.values()
            )
        }
        return Evaluation(
            cd=cd,
            cl=cl,
            uncertainty=uncertainty,
            in_distribution=edge_distance < 0.90,
            pressure_summary=pressure_summary,
            model_version="stub-v1-synthetic",
            latency_ms=0.0,
        )


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
