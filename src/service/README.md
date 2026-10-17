# Track B local development

By default this service uses deterministic **synthetic** predictions (the
stub). Setting `AERO_CHECKPOINT` serves one of Avneet's real params-v1
checkpoints instead; see "Serving a real checkpoint" below. The design agent
runs live on Nemotron through Token Factory, or as a free scripted fixture with
`--dry-run`. The stub claims no aerodynamic accuracy, calibrated OOD
detection, or real inference speed.

Stage-2 fields are a separate optional capability: set `AERO_FIELD_CHECKPOINT`
and call `POST /fields`. This does not change `/evaluate` or its trust gate.
Checkpoint qualification, the field/viewer contract, bounded real-agent evidence,
and fair-comparison integration are specified in **[INTEGRATION.md](INTEGRATION.md)**.

## Setup on Windows PowerShell

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-service.txt
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
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

## Live Nemotron agent

Needs `NEBIUS_API_KEY` in `.env` (loaded automatically; see `.env.example`).
Check access first: two tiny calls, under $0.001.

```powershell
.\.venv\Scripts\python.exe -m src.agent.provider --smoke --model nano
.\.venv\Scripts\python.exe -m src.agent.designer --model nano --iterations 10
.\.venv\Scripts\python.exe -m src.agent.designer --model super --iterations 25 --constraint "A_Car_Roof_Height>=0"
```

Each step rebuilds a compact prompt (system rules, best trusted design, the
last eight attempts with their gate verdicts and reasons, constraints, budget
left) and asks the model to call one of three tools: `evaluate` (uses one
evaluation), `compare` and `history` (free, answered from the run). The
server still owns scoring, the gate and logging; the model never writes its
own result.

`evaluate` takes `changes` (from the best trusted design), `hypothesis` (the
physical reason, one or two sentences), `expected_effect` (`lower_cd`,
`higher_cd` or `no_change`) and optionally `expected_delta_cd` (e.g.
`-0.001`). A legacy free-form `reasoning` string is accepted as the
hypothesis when `hypothesis` is absent, since small models sometimes still
send it. After scoring, the code writes each record's `reasoning` as one
logbook line; the Observed part comes from the score and the gate, never the
model:

```text
Hypothesis: A shallower ramp should delay separation. | Change: B_Ramp_Angle -3 → -1 | Expected: lower Cd (~-0.00100) | Observed: Cd 0.28710 → 0.28650 (-0.00060) vs iteration 3, trusted — as expected
... | Expected: lower Cd | Observed: REJECTED by the gate (uncertainty 0.0230 exceeds 0.0130) — no trusted result
```

The comparison is against the design the step changed from (the best trusted
design, or the baseline, marked untrusted, if nothing is trusted yet).
Verdicts are `as expected`, `opposite to expected`, or `within noise` when
|ΔCd| < 1e-5; a `no_change` expectation is `as expected` within that
tolerance and `not as expected` outside it. The next step's prompt carries the
verdicts and a `Last step: expected lower Cd, observed +0.00041 (opposite to
expected).` line. `run_finished` records `hypothesis_hits` (`judged`,
`as_expected`, `rate`) over trusted live steps that expected lower or higher
Cd; rejected and `no_change` steps are not judged, and a change within noise
counts as a miss. The CLI prints it as `Hypotheses: 7 of 11 trusted
predictions had the expected direction (64%).` The baseline and `--dry-run`
steps keep their plain reasoning.

Before anything is scored, a proposal is refused with a precise reason (never
clamped) if it lacks a hypothesis or a valid expected effect, names an
unknown parameter, changes more than two parameters,
leaves the bounds, breaks a `--constraint`, or repeats a design already scored
(compared numerically, so `2`, `2.0` and `-0.0`/`0.0` spellings cannot slip
through). Malformed tool calls get an error reply, never a crash. The model
gets up to four rounds per step to correct itself. The run stops visibly, with
the reason in its `run_finished` event, after two steps in a row without a
valid proposal, a provider failure, the run's `--run-budget` (default $0.50,
checked before every model call including correction rounds), or the global
spend cap. A run that fails with an exception still gets a `run_finished`
event; service readiness is checked before a run ID is reserved. The agent
gates with the thresholds `/schema` reports, so a real checkpoint's calibrated
OOD threshold applies to the agent and `/compare` as well.

`--constraint` takes numeric rules (`NAME >= NUMBER`, also `<=`, `<`, `>`);
turning a sentence like "keep the roof street-legal" into such a rule is up to
whoever starts the run, not automatic.

Costs: before each call the client reserves its worst-case cost in
`runs/spend.jsonl` under a lock shared by threads and processes, and refuses
the call if the total plus outstanding reservations would pass
`AERO_SPEND_CAP_USD` (default $5). After the call the reservation is settled
to the billed amount, released if nothing was billed, or kept in full if the
response carried no usable usage. Measured on the stub:
25 Super evaluations cost about $0.025, 10 Nano evaluations about $0.002.
`--thinking auto` (default) turns model thinking off for Nano, which otherwise
spends ~1.6k tokens per step and can be cut off before its tool call, and
leaves it on for Super, which thinks briefly.

Logs: scored attempts in `runs/logs/<run_id>.jsonl` (what `/history` serves
and the charts read); everything else - run start and finish, corrected or
refused proposals, provider errors, token usage - in
`runs/logs/events/<run_id>.jsonl`.

## API

Open `http://127.0.0.1:8000/docs` for interactive requests.

| Endpoint | Payload/result |
|---|---|
| GET /health | Ready predictor, synthetic flag, schema version, `model_version`, `checkpoint_sha256` (torch only), `output_provenance` |
| GET /ready | `{"ready": true}`, or 503 with `problems` when the predictor is missing or the log directory is not writable (container health check) |
| GET /schema | Bounds, body styles, default parameters, pressure regions, the thresholds the serving predictor uses |
| POST /evaluate | `{"params": <complete parameter object>, "body_style": "fastback"}` returns Evaluation (`body_style` optional) |
| POST /fields | Same validated design request; separate per-face Cp/sigma response tied to an exact viewer mesh hash; 503 when disabled/busy. Replies over 1 KB are gzipped when the client sends `Accept-Encoding: gzip` (browsers do; the real-model reply drops from about 990 KB to about 410 KB) |
| POST /compare | `{"design_a": {"params": ...}, "design_b": {"params": ...}}` |
| GET /history/{run_id} | Array of IterationRecord objects; 404 for an absent run |
| POST /runs | Start an agent run in the background (see Run control below); 202 with its status |
| GET /runs?limit= | Recent runs, newest first (API and CLI runs alike) |
| GET /runs/{run_id} | One run's state, progress, cost and timing |
| POST /runs/{run_id}/cancel | Ask a run to stop at its next step or model call |

### Run control

`POST /runs` body: `{"model": "nano" | "super", "iterations": 25, "body_style":
"fastback", "constraints": ["A_Car_Roof_Height >= 0"], "run_budget_usd": 0.10,
"thinking": "auto"}`; only `model` and `iterations` are required. The run
scores in-process with the serving model and the same gate thresholds as
`/evaluate`, and is logged like a CLI run, so `/history/{run_id}` streams it.

| Caller | How | Limits |
|---|---|---|
| public | no `Authorization` header (only when `AERO_PUBLIC_RUNS=1`) | `AERO_PUBLIC_MAX_ITERATIONS`, `AERO_PUBLIC_RUN_BUDGET_USD` per run, `AERO_PUBLIC_DAILY_BUDGET_USD` per UTC day |
| admin | `Authorization: Bearer <AERO_ADMIN_TOKEN>` | `AERO_ADMIN_*` ceilings; not counted in the public daily budget |

Requests above a ceiling get 422 (never silently lowered); a wrong token 401;
too many runs, or the public daily budget used up, 429; no provider key 503.
Public and admin runs have separate slots of `AERO_MAX_CONCURRENT_RUNS` each,
so public traffic cannot lock the admin out.

States: `queued`, `running`, then `finished` (its evaluation or spend budget
ran out), `cancelled`, `failed` (provider error, the model kept producing no
valid design, or an exception: read `stop_reason`), `interrupted` (the
service stopped or restarted mid-run), or `unknown` (unfinished and not owned
by this process, e.g. a CLI run still going).

Cancelling: the admin may cancel any run this process owns. A public run's
202 response includes a `cancel_token`, shown only then; send it as
`X-Cancel-Token` to cancel that run. Others get 403. A run already inside a
model call stops when that call returns (bounded by its timeout).

The daily public budget is computed from the run logs, so it survives
restarts: a finished run counts at its billed cost, an unfinished or
unreadable one at its full run budget, and a public run still open from
yesterday counts today too.

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

On the real checkpoint the uncertainty cutoff (0.013, from the contract; the
checkpoint carries no calibrated one) is dormant. Avneet measured the largest
sigma over 4,000 designs drawn uniformly from the whole legal box at 0.01057, and
over the 625 held-out designs at 0.01033: none reaches 0.013. Over the same
4,000 the ood_score gate (threshold 4.9348, the checkpoint's 0.99 validation
quantile) rejects 6.12%. Every real-model rejection therefore comes from
ood_score. The cutoff stays because it is frozen into the comparison protocol and
a dormant gate is harmless; describe the gate as an out-of-distribution check,
not as uncertainty filtering it is not doing. Field predictions have no OOD
detector at all (the field checkpoint has no `ood` key): label them "predicted,
not trusted", never "trusted with caveats".

## Configuration

The agent loads `.env` from the repository root (variables already set in the
environment win). The service does not need it. To override per process:

```powershell
$env:AERO_SERVICE_URL = 'http://127.0.0.1:8000'
$env:AERO_RUNLOG_DIR = 'runs/logs'
```

`AERO_RUNLOG_DIR` must identify the same directory for the service and agent,
and must be writable: the service creates it and fails startup otherwise.
`AERO_CHECKPOINT` must be empty for the stub. A nonempty value loads that
checkpoint with `TorchPredictor`; anything wrong with it fails startup. There
is never a fallback to the stub.

Hosting settings, read once at startup by `src/service/settings.py` (invalid
values fail startup): `AERO_CORS_ORIGINS`, `AERO_ADMIN_TOKEN`,
`AERO_PUBLIC_RUNS`, `AERO_MAX_CONCURRENT_RUNS`, `AERO_PUBLIC_MAX_ITERATIONS`,
`AERO_ADMIN_MAX_ITERATIONS`, `AERO_PUBLIC_RUN_BUDGET_USD`,
`AERO_ADMIN_RUN_BUDGET_USD`, `AERO_PUBLIC_DAILY_BUDGET_USD`,
`AERO_RATE_LIMIT_PER_MINUTE`, `AERO_MAX_BODY_BYTES`, `AERO_TRUST_PROXY`,
`AERO_RUNLOG_DIR`. `POST /evaluate`,
`/compare` and everything under `/runs` are rate limited per client IP (429 with
`Retry-After`); bodies over the size limit get 413. `GET /ready` is the
container health check (503 when the predictor is missing or the log
directory is not writable); `/health` is unchanged. On startup the service
prints one configuration line that never includes the token.

Docker, a Nebius CPU VM behind Caddy, the full variable reference, token
rotation and cost safety: **[DEPLOY.md](DEPLOY.md)**.

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

Category-held-out checkpoints are refused for ordinary serving. The global
body-style list alone does not establish a model's training coverage. Startup
also validates finite model weights/probes and the OOD detector's method,
statistics, calibration metadata and distance representation. Full SHA-256 and
source fingerprints are additive under `/health.provenance`; the existing
top-level `checkpoint_sha256` remains its short display value.

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

A checkpoint without an OOD detector loads and reports `ood_score: null`; its
gate is then uncertainty-only, so jointly extreme in-bounds designs are NOT
screened in that mode and nothing should claim otherwise. A detector that
measures point-cloud features (`space: "features"`) is refused for a params-v1
checkpoint. `/evaluate` returns the predictor's own `in_distribution`; the
shared `rejection_reason` gate is applied by `/compare` and the agent, which
for the built-in predictors gives the same decision.

Run IDs contain 1-128 letters, digits, hyphens or underscores; Windows device
names are rejected. Logs live at `runs/logs/<run_id>.jsonl` and are Git-ignored.
Malformed completed records produce a visible error. Incomplete trailing
records are excluded from history, and appending to such a log is refused.

The run-control API and hosting guards are implemented. Real-model qualification
and live evidence still require Avneet's compatible checkpoints/reference outputs.
The 4be514d pull fixes both trainers' schema stamp and adds independent reference
exporters. `verify_checkpoint` now accepts Avneet's native scalar JSON directly;
`qualify` additionally checks readiness, service routes and agent-gate parity,
and `verify_fields` checks exact fixed-query parity using the original NPZ. The UI and
optimiser integrations remain Paramveer's work; Track B supplies the tested APIs
and comparison evaluator described in INTEGRATION.md.
