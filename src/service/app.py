"""FastAPI surrogate service.  OWNER: Pranoy.  DUE: stub live Oct 13.

    uvicorn src.service.app:app --reload --port 8000

Endpoints to build:
    GET  /health              -> {"ok": true, "predictor": ...}
    GET  /schema              -> param bounds, defaults, pressure regions
    POST /evaluate            -> Evaluation (the contract)
    POST /compare             -> field delta between two designs
    GET  /history/{run_id}    -> the run log

Serve the stub by default. AERO_CHECKPOINT env var swaps in the real model --
that swap is the Oct 15 integration checkpoint.
"""

from __future__ import annotations

# TODO Pranoy: FastAPI app, CORS for localhost:3000 and :5173,
# validate with contracts.validate_params, return 422 on problems.
