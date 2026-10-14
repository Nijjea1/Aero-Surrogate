"""FastAPI surrogate service.  OWNER: Pranoy.  DUE: stub live Oct 13.

    uvicorn src.service.app:app --reload --port 8000

Implemented endpoints:
    GET  /health              -> predictor, model version, checkpoint hash, output provenance
    GET  /schema              -> param bounds, body styles, defaults, thresholds
    POST /evaluate            -> Evaluation (the contract)
    POST /compare             -> field delta between two designs
    GET  /history/{run_id}    -> the run log

Serve the synthetic stub by default. A nonempty AERO_CHECKPOINT loads that
params-v1 checkpoint with TorchPredictor (needs requirements-model.txt); a
checkpoint that cannot be served fails startup and never falls back to the stub.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, FiniteFloat, field_validator

from src.contracts import (
    BODY_STYLES, DEFAULT_BODY_STYLE, Design, Evaluation, OOD_SCORE_THRESHOLD, PARAM_BOUNDS,
    PRESSURE_REGIONS, SCHEMA_VERSION, UNCERTAINTY_THRESHOLD, default_params,
    validate_body_style,
)
from src.service.predictor import BasePredictor, load_predictor, validate_design_params
from src.service.gate import rejection_reason
from src.service.runlog import RunLog


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    params: dict[str, FiniteFloat]
    body_style: str = DEFAULT_BODY_STYLE
    design_id: str | None = None

    @field_validator("params", mode="before")
    @classmethod
    def validate_parameters(cls, value):
        problems = validate_design_params(value)
        if problems:
            raise ValueError("; ".join(problems))
        return value

    @field_validator("body_style")
    @classmethod
    def validate_style(cls, value):
        problems = validate_body_style(value)
        if problems:
            raise ValueError("; ".join(problems))
        return value


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    design_a: DesignRequest
    design_b: DesignRequest


def create_app(predictor: BasePredictor | None = None, runlog: RunLog | None = None) -> FastAPI:
    """Inject a predictor for tests; otherwise load configuration at startup."""
    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = predictor or load_predictor(
            os.environ.get("AERO_CHECKPOINT") or None
        )
        application.state.runlog = runlog or RunLog(os.environ.get("AERO_RUNLOG_DIR", "runs/logs"))
        yield

    application = FastAPI(title="Aero Surrogate", version=SCHEMA_VERSION, lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Raw invalid input can contain NaN/Infinity, which cannot be echoed
        # in a standards-compliant JSON error response.
        details = [{"loc": list(error["loc"]), "msg": error["msg"],
                    "type": error["type"]} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    def _threshold(name: str, default: float) -> float:
        value = getattr(application.state.predictor, name, None)
        return default if value is None else value

    @application.get("/health")
    def health():
        predictor = application.state.predictor
        return {"ok": True, "predictor": predictor.name,
                "synthetic": predictor.name == "stub",
                "schema_version": SCHEMA_VERSION,
                "model_version": getattr(predictor, "model_version", None),
                "checkpoint_sha256": getattr(predictor, "checkpoint_sha256", None),
                "output_provenance": dict(getattr(predictor, "output_provenance", {}) or {})}

    @application.get("/schema")
    def schema():
        return {"schema_version": SCHEMA_VERSION, "param_bounds": PARAM_BOUNDS,
                "defaults": default_params(), "pressure_regions": PRESSURE_REGIONS,
                "body_styles": BODY_STYLES, "default_body_style": DEFAULT_BODY_STYLE,
                "param_units": "offsets from the baseline body of the chosen style: "
                               "angles in degrees, lengths in mm, curvatures and "
                               "inclinations dimensionless",
                "uncertainty_threshold": _threshold("uncertainty_threshold", UNCERTAINTY_THRESHOLD),
                "uncertainty_units": "Cd standard deviation",
                "ood_score_threshold": _threshold("ood_threshold", OOD_SCORE_THRESHOLD)}

    @application.post("/evaluate", response_model=Evaluation)
    def evaluate(request: DesignRequest):
        return application.state.predictor.evaluate(
            Design(params=request.params, body_style=request.body_style,
                   design_id=request.design_id)
        )

    @application.post("/compare")
    def compare(request: CompareRequest):
        a = evaluate(request.design_a).to_dict()
        b = evaluate(request.design_b).to_dict()
        reasons = {"design_a": rejection_reason(a, body_style=request.design_a.body_style),
                   "design_b": rejection_reason(b, body_style=request.design_b.body_style)}
        common = a["pressure_summary"].keys() & b["pressure_summary"].keys()
        return {
            "design_a": a, "design_b": b,
            "delta": {"cd": b["cd"] - a["cd"], "cl": b["cl"] - a["cl"],
                      "pressure_summary": {key: b["pressure_summary"][key] - a["pressure_summary"][key]
                                           for key in sorted(common)}},
            "delta_direction": "design_b minus design_a",
            "same_body_style": request.design_a.body_style == request.design_b.body_style,
            "trusted": all(reason is None for reason in reasons.values()),
            "reject_reasons": reasons,
            "comparison_kind": "scalar and regional summaries; not a surface-field comparison",
            "unavailable_pressure_regions": sorted(set(PRESSURE_REGIONS) - common),
        }

    @application.get("/history/{run_id}")
    def history(run_id: str):
        try:
            return application.state.runlog.read(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            # Invalid IDs are user errors; corrupt stored records are server errors.
            status = 500 if str(exc).startswith("Malformed run log") else 422
            raise HTTPException(status_code=status, detail=str(exc)) from None

    return application


app = create_app()
