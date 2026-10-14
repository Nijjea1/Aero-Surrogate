"""The thing behind POST /evaluate.  OWNER: Pranoy.

Two implementations, same interface:
    StubPredictor  - plausible numbers, no model, no GPU. BUILD THIS FIRST
                     (Oct 13) so Paramveer is not blocked on Avneet.
    TorchPredictor - loads one of Avneet's params-v1 checkpoints, following
                     docs/HANDOFF.md section 3. Needs requirements-model.txt.

Make the stub's uncertainty RISE toward the edges of the design box, or the
uncertainty gate has nothing to fire on and cannot be tested.

Schema 2.0.0: sixteen DrivAerNet morph deltas plus a body style. All-zero
params is the unmodified baseline body of that style.

Torch is imported lazily inside TorchPredictor so the lightweight service
(requirements-service.txt, no torch) can import this module and serve the stub.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from pathlib import Path

from ..contracts import (
    BODY_STYLES, Design, Evaluation, OOD_SCORE_THRESHOLD, PARAM_BOUNDS, PRESSURE_REGIONS,
    SCHEMA_VERSION, UNCERTAINTY_THRESHOLD, is_in_distribution, validate_body_style,
)

# Where each Evaluation field comes from, reported by /health so nobody reads a
# placeholder as a prediction.
#   model       - computed by the loaded checkpoint
#   synthetic   - the deterministic stub fixture, not aerodynamics
#   placeholder - a fixed filler value the contract requires; carries no information
#   unavailable - the model does not produce it; the field is empty or null
OUTPUT_FIELDS = ("cd", "uncertainty", "ood_score", "cl", "pressure_summary")


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
    model_version: str | None = None
    checkpoint_sha256: str | None = None
    # Thresholds this predictor's in_distribution flag was computed with.
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD
    ood_threshold: float | None = OOD_SCORE_THRESHOLD
    output_provenance: dict[str, str] = {}

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
    model_version = "stub-v2-synthetic"
    output_provenance = {field: "synthetic" for field in OUTPUT_FIELDS}

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
            model_version=self.model_version,
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


class CheckpointError(RuntimeError):
    """The checkpoint cannot be served. Raised at load time, never papered over."""


PARAMS_PREPROCESS_VERSION = "params-v1"
POINTCLOUD_PREPROCESS_VERSION = "pointcloud-v1"
_REQUIRED_KEYS = ("state_dict", "config", "norm_stats", "preprocess", "schema_version")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_param_columns(columns: object) -> list[str]:
    """The checkpoint's input columns, checked against the 2.0.0 contract.

    Every name must be a contract parameter or a cat_<body style> indicator,
    each exactly once, and every contract parameter and body style must be
    present: a model that ignores a dial the agent turns, or cannot tell which
    baseline body it is perturbing, would answer without saying so.
    """
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(name, str) for name in columns)):
        raise CheckpointError("preprocess.param_columns must be a non-empty list of names")
    problems = []
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate columns {duplicates}")
    unknown = [name for name in columns
               if name not in PARAM_BOUNDS
               and not (name.startswith("cat_") and name[4:] in BODY_STYLES)]
    if unknown:
        problems.append(f"columns that are neither contract parameters nor "
                        f"cat_<body style> indicators: {unknown}")
    missing = [name for name in PARAM_BOUNDS if name not in columns]
    missing += [f"cat_{style}" for style in BODY_STYLES if f"cat_{style}" not in columns]
    if missing:
        problems.append(f"contract inputs the model does not take: {missing}")
    if problems:
        raise CheckpointError(
            f"checkpoint input columns do not match schema {SCHEMA_VERSION}: " + "; ".join(problems)
        )
    return list(columns)


class TorchPredictor(BasePredictor):
    """Serves a params-v1 checkpoint exactly as docs/HANDOFF.md section 3 says.

    Everything that can be wrong with a checkpoint is checked here, at
    construction, so a bad AERO_CHECKPOINT fails startup instead of the first
    request. There is no fallback to the stub.

    Stage 1 predicts Cd and its standard deviation only. cl is a fixed 0.0
    placeholder and pressure_summary is empty; output_provenance says so.
    """

    name = "torch"

    def __init__(self, checkpoint: str, device: str = "cpu"):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        try:
            import numpy as np
            import torch

            from ..train.model import build_model, split_prediction
            from ..train.ood import OODDetector
        except ImportError as exc:
            raise CheckpointError(
                "serving a checkpoint needs torch and numpy: "
                "pip install -r requirements-model.txt"
            ) from exc
        self._np, self._torch, self._split_prediction = np, torch, split_prediction

        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = _file_sha256(path)[:12]
        self.device = torch.device(device)

        # weights_only=False unpickles arbitrary Python objects, which means
        # loading EXECUTES code embedded in the file. Only load checkpoints a
        # trusted teammate produced with src/train/train.py.
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:  # noqa: BLE001 - any failure means "not a checkpoint"
            raise CheckpointError(f"{path} is not a readable torch checkpoint: {exc}") from exc
        if not isinstance(ckpt, dict):
            raise CheckpointError(f"{path} holds a {type(ckpt).__name__}, not a checkpoint dict")
        missing = [key for key in _REQUIRED_KEYS if key not in ckpt]
        if missing:
            raise CheckpointError(f"{path} is missing checkpoint keys {missing}")
        if ckpt["schema_version"] != SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint was trained against schema {ckpt['schema_version']!r}; "
                f"this service speaks {SCHEMA_VERSION!r}"
            )

        spec = ckpt["preprocess"]
        version = spec.get("version") if isinstance(spec, dict) else None
        if version == POINTCLOUD_PREPROCESS_VERSION:
            raise CheckpointError(
                "point-cloud (pointcloud-v1) checkpoints are not served: /evaluate "
                "receives parameters, and turning them into a mesh would put the "
                "approximate viewer morph in the prediction path. Use a params-v1 "
                "checkpoint."
            )
        if version != PARAMS_PREPROCESS_VERSION:
            raise CheckpointError(
                f"unknown preprocess version {version!r}; this service serves "
                f"{PARAMS_PREPROCESS_VERSION!r} checkpoints only"
            )
        self.param_columns = _check_param_columns(spec.get("param_columns"))
        n_inputs = len(self.param_columns)

        norm = ckpt["norm_stats"]
        try:
            self._param_mean = np.asarray(norm["param_mean"], dtype=np.float64)
            self._param_std = np.asarray(norm["param_std"], dtype=np.float64)
            self._target_mean = float(norm["target_mean"])
            self._target_std = float(norm["target_std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"norm_stats is incomplete for a params-v1 checkpoint: {exc}") from exc
        if self._param_mean.shape != (n_inputs,) or self._param_std.shape != (n_inputs,):
            raise CheckpointError(
                f"norm_stats has {self._param_mean.shape}/{self._param_std.shape} entries "
                f"for {n_inputs} input columns"
            )
        if (not np.all(np.isfinite(self._param_mean)) or not np.all(np.isfinite(self._param_std))
                or np.any(self._param_std <= 0)):
            raise CheckpointError("norm_stats param_mean/param_std must be finite with std > 0")
        if not (math.isfinite(self._target_mean) and math.isfinite(self._target_std)
                and self._target_std > 0):
            raise CheckpointError("norm_stats target_mean/target_std must be finite with std > 0")

        try:
            model = build_model(ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])  # strict: a partial load is a wrong model
            model.to(self.device).eval()
            with torch.inference_mode():
                probe = model(torch.zeros((1, n_inputs), dtype=torch.float32, device=self.device))
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(f"cannot rebuild the model from the checkpoint: {exc}") from exc
        if tuple(probe.shape) != (1, 2):
            raise CheckpointError(
                f"model emits {tuple(probe.shape)[1:]} outputs per design; serving needs "
                f"(mean, log-variance). A model without an uncertainty head would report "
                f"zero uncertainty, which the gate would read as certainty."
            )
        self._model = model

        self._ood = None
        if ckpt.get("ood"):
            try:
                detector = OODDetector.from_dict(ckpt["ood"])
                ood_mean = np.asarray(detector.mean, dtype=np.float64)
                cov_inv = np.asarray(detector.cov_inv, dtype=np.float64)
                threshold = float(detector.threshold)
            except Exception as exc:  # noqa: BLE001
                raise CheckpointError(f"cannot rebuild the OOD detector: {exc}") from exc
            if ood_mean.shape != (n_inputs,) or cov_inv.shape != (n_inputs, n_inputs):
                raise CheckpointError("OOD detector dimensions do not match the input columns")
            if not math.isfinite(threshold) or threshold <= 0:
                raise CheckpointError(f"OOD detector threshold {threshold} is not calibrated")
            self._ood = detector

        # The checkpoint carries no calibrated sigma cutoff, so the contract's
        # stays. The OOD threshold is the detector's own, calibrated on the
        # validation split; the contract's 3.0 is a placeholder that would
        # reject most real designs in 19 standardised dimensions.
        self.uncertainty_threshold = UNCERTAINTY_THRESHOLD
        self.ood_threshold = float(self._ood.threshold) if self._ood is not None else None

        ckpt_version = ckpt.get("version")
        self.model_version = (ckpt_version if isinstance(ckpt_version, str) and ckpt_version.strip()
                              else f"sha256-{self.checkpoint_sha256}")
        self.held_out_category = ckpt.get("held_out_category")
        self.output_provenance = {
            "cd": "model",
            "uncertainty": "model",
            "ood_score": "model" if self._ood is not None else "unavailable",
            "cl": "placeholder",
            "pressure_summary": "unavailable",
        }

    def model_input(self, design: Design):
        """The standardised (1, P) float64 row, built by walking param_columns."""
        row = []
        for column in self.param_columns:
            if column.startswith("cat_"):
                row.append(1.0 if column == f"cat_{design.body_style}" else 0.0)
            else:
                row.append(float(design.params[column]))
        raw = self._np.asarray([row], dtype=self._np.float64)
        return (raw - self._param_mean) / self._param_std

    def evaluate(self, design: Design) -> Evaluation:
        problems = validate_design_params(design.params) + validate_body_style(design.body_style)
        if problems:
            raise ValueError("; ".join(problems))
        x = self.model_input(design)
        torch = self._torch

        start = time.perf_counter()
        with torch.inference_mode():
            pred = self._model(torch.from_numpy(x.astype(self._np.float32)).to(self.device))
            mu, sigma = self._split_prediction(pred)
        cd = float(mu[0]) * self._target_std + self._target_mean
        uncertainty = float(sigma[0]) * self._target_std  # a scale: std only, no mean
        ood_score = self._ood.score_one(x) if self._ood is not None else None
        latency_ms = (time.perf_counter() - start) * 1000.0

        if not (math.isfinite(cd) and math.isfinite(uncertainty)
                and (ood_score is None or math.isfinite(ood_score))):
            raise RuntimeError("model produced a non-finite prediction")
        in_distribution, _ = is_in_distribution(
            uncertainty, ood_score, design.body_style,
            uncertainty_threshold=self.uncertainty_threshold,
            ood_threshold=self.ood_threshold if self.ood_threshold is not None else OOD_SCORE_THRESHOLD,
        )
        return Evaluation(
            cd=cd,
            cl=0.0,                 # placeholder: stage 1 does not predict lift
            uncertainty=uncertainty,
            in_distribution=in_distribution,
            pressure_summary={},    # unavailable until stage-2 surface fields
            model_version=self.model_version,
            latency_ms=latency_ms,
            ood_score=ood_score,
        )


def load_predictor(checkpoint: str | None = None) -> BasePredictor:
    """Torch predictor when a checkpoint is given, else the stub.

    A configured checkpoint that cannot be served raises; it never falls back
    to the synthetic stub.
    """
    if checkpoint is None:
        return StubPredictor()
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint must be a non-empty path or None")
    return TorchPredictor(checkpoint)
