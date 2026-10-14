# Track B local development

This service currently uses deterministic **synthetic** predictions. The
scripted agent is an offline development fixture, not Nemotron. No aerodynamic
accuracy, calibrated OOD detection, or real inference speed is claimed.

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
iterations means ten evaluated designs, including the midpoint at iteration 0.
Iteration 1 deliberately probes the boundary and gets rejected; subsequent
proposals return to the trusted region. `accepted` means trusted, not improved.
`best_cd_so_far` only changes for trusted improvements.

On macOS/Linux use `.venv/bin/python` instead of the Windows executable path.
The same Python module commands work.

## API

Open `http://127.0.0.1:8000/docs` for interactive requests.

| Endpoint | Payload/result |
|---|---|
| GET /health | Ready predictor, synthetic flag, schema version |
| GET /schema | Bounds, default parameters, pressure regions, threshold |
| POST /evaluate | `{"params": <complete parameter object>}` returns Evaluation |
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

The shared gate rejects nonfinite predictions, negative standard deviation,
false distribution flags, and uncertainty above the contract threshold. The
threshold and stub distribution flags are development fixtures.

## Configuration

Set environment variables before launching each process; this milestone does
not automatically load `.env`:

```powershell
$env:AERO_SERVICE_URL = 'http://127.0.0.1:8000'
$env:AERO_RUNLOG_DIR = 'runs/logs'
```

`AERO_RUNLOG_DIR` must identify the same directory for the service and agent.
`AERO_CHECKPOINT` must be empty for the stub. A nonempty checkpoint currently
fails startup explicitly because the real adapter is not implemented.

Run IDs contain 1-128 letters, digits, hyphens or underscores; Windows device
names are rejected. Logs live at `runs/logs/<run_id>.jsonl` and are Git-ignored.
Malformed completed records produce a visible error. Incomplete trailing
records are excluded from history, and appending to such a log is refused.

Next milestones: real checkpoint adapter, live Token Factory tools, paid-call
budgets, run start/status/cancel API, hosted authentication and deployment.
