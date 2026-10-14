# Track B local development

By default this service uses deterministic **synthetic** predictions (the
stub). Setting `AERO_CHECKPOINT` serves one of Avneet's real params-v1
checkpoints instead; see "Serving a real checkpoint" below. The scripted agent
is an offline development fixture, not Nemotron. The stub claims no
aerodynamic accuracy, calibrated OOD detection, or real inference speed.

## Setup on Windows PowerShell

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-service.txt
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --host 127.0.0.1 --port 8000
```

The lightweight requirements omit Torch and geometry dependencies. The shared
full requirements file is unchanged. Use the service in one process for this
development milestone. Logs use process-local locks and exclusively reserved
run IDs; multiworker/distributed logging is a later integration task.

In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m src.agent.designer --dry-run --iterations 10 --run-id demo-001
Invoke-RestMethod http://127.0.0.1:8000/history/demo-001
```

Use a fresh run ID each time. Existing runs are never overwritten. Ten
iterations means ten evaluated designs, including the unmodified baseline body
(all offsets zero) at iteration 0. `--body-style` picks fastback (default),
notchback or estateback for the whole run.
Iteration 1 deliberately probes the boundary and gets rejected; subsequent
proposals return to the trusted region. `accepted` means trusted, not improved.
`best_cd_so_far` only changes for trusted improvements.

On macOS/Linux use `.venv/bin/python` instead of the Windows executable path.
The same Python module commands work.

## API

Open `http://127.0.0.1:8000/docs` for interactive requests.

| Endpoint | Payload/result |
|---|---|
| GET /health | Ready predictor, synthetic flag, schema version, `model_version`, `checkpoint_sha256` (torch only), `output_provenance` |
| GET /schema | Bounds, body styles, default parameters, pressure regions, the thresholds the serving predictor uses |
| POST /evaluate | `{"params": <complete parameter object>, "body_style": "fastback"}` returns Evaluation (`body_style` optional) |
| POST /compare | `{"design_a": {"params": ...}, "design_b": {"params": ...}}` |
| GET /history/{run_id} | Array of IterationRecord objects; 404 for an absent run |

Example evaluation:

```powershell
$schema = Invoke-RestMethod http://127.0.0.1:8000/schema
$body = @{params = $schema.defaults} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8000/evaluate -Method Post -ContentType application/json -Body $body
```

Parameters must be complete, known, finite numeric values inside their bounds.
Booleans and numeric strings are rejected. Invalid requests return 422 with
an error list. Inputs are never silently clamped.

Comparison deltas mean **design B minus design A**. Comparison includes both
Evaluations, gate reasons, and whether both are trusted. It compares scalars
and regional summaries, not full surface fields. Missing region values appear
in `unavailable_pressure_regions`; they are not fabricated.

Schema 2.0.0: sixteen DrivAerNet morph parameters, each an offset from the
baseline body of the chosen style. All zeros is the baseline car.

The shared gate rejects nonfinite predictions, negative standard deviation or
ood_score, and false distribution flags, then applies
`contracts.is_in_distribution`: unknown body style, uncertainty above the
threshold, or ood_score (distance from the training set) above its threshold.
In the stub, one parameter at its limit trips the uncertainty check and many
parameters pushed out together trip the ood_score check; both are development
fixtures, not calibrated detection.

## Configuration

Set environment variables before launching each process; this milestone does
not automatically load `.env`:

```powershell
$env:AERO_SERVICE_URL = 'http://127.0.0.1:8000'
$env:AERO_RUNLOG_DIR = 'runs/logs'
```

`AERO_RUNLOG_DIR` must identify the same directory for the service and agent.
`AERO_CHECKPOINT` must be empty for the stub. A nonempty value loads that
checkpoint with `TorchPredictor`; anything wrong with it fails startup. There
is never a fallback to the stub.

## Serving a real checkpoint

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
$env:AERO_CHECKPOINT = 'runs/drivaernet-random/best.pt'
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --host 127.0.0.1 --port 8000
```

CPU torch is enough. Loading uses `torch.load(..., weights_only=False)`, which
executes code pickled in the file: only load checkpoints a trusted teammate
produced with `src/train/train.py`.

Only `params-v1` checkpoints (design parameters in, Cd out) are served. Startup
fails with a clear error for a missing or unreadable file, a point-cloud
(`pointcloud-v1`) or unknown preprocess version, a `schema_version` other than
the contract's, input columns that are not exactly the 16 contract parameters
plus `cat_<body style>` indicators, mismatched weights, or a model without an
uncertainty head. Model input is built by walking the checkpoint's
`preprocess.param_columns` in order and standardised with its `norm_stats`,
following `docs/HANDOFF.md` section 3.

What the real model does and does not produce (`/health` -> `output_provenance`):

| Field | Source |
|---|---|
| `cd`, `uncertainty` | model (`uncertainty` is the Cd standard deviation) |
| `ood_score` | model's Mahalanobis OOD detector, or `null` if the checkpoint has none |
| `cl` | placeholder `0.0`: stage 1 does not predict lift |
| `pressure_summary` | unavailable: always `{}`; `/compare` lists every region as unavailable |

`in_distribution` uses the OOD threshold calibrated into the checkpoint (99%
of held-out validation designs pass), not the contract's 3.0 placeholder, and
the contract's uncertainty threshold (checkpoints carry no calibrated one).
`/schema` reports both values actually in use.

Run IDs contain 1-128 letters, digits, hyphens or underscores; Windows device
names are rejected. Logs live at `runs/logs/<run_id>.jsonl` and are Git-ignored.
Malformed completed records produce a visible error. Incomplete trailing
records are excluded from history, and appending to such a log is refused.

Next milestones: pass the predictor's calibrated thresholds to the shared gate
(`/compare` and the agent still call `rejection_reason` with the contract
defaults, so with a real checkpoint they can reject a design `/evaluate` marks
in distribution), live Token Factory tools, paid-call
budgets, run start/status/cancel API, hosted authentication and deployment.
