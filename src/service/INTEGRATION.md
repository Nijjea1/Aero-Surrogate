# Track B integration and real-model qualification

Owner: Pranoy. Implemented against contracts 2.0.0. Updated for Avneet's
6319c7a handoff bundle (aero-handover-2026-10-15). Avneet's native scalar/field
JSON formats are supported without reshaping them.

## Track B status: October 16, 2026, real artifacts present

Bundle checksums verified (all 12 match SHA256SUMS). Offline suite with the
real files present: **956 passed, 3 skipped** (after merging Paramveer's PRs #8-#17). The 3 skips need Avneet's full
parameter table, extracted fields or geometry checkpoint, which Track B does
not serve.

Keep the field source NPZ outside `data/drivaerml/fields/` unless the whole
extracted dataset is there: `tests/test_handover.py` treats a folder holding
only `run_3.npz` as a dataset and fails to split one run. Track B keeps it at
`runs/handover/source/run_3.npz`; the pack authenticates it by hash, not path.

Qualified on this machine, offline, against the real bytes
(`runs/qualify-pranoy-20261016/`):

| Check | Result |
|---|---|
| `verify_checkpoint` | passed, 5/5 designs including the genuine rejection |
| `qualify` | passed, in-process: /ready, /health, /schema, 5 /evaluate, 10 /compare, shared agent gate |
| `verify_fields` | passed and source authenticated by the pack's hash; Cp within 3.6e-7, sigma within 1.8e-7 (tolerance 1e-4) |

Live agent evidence on the real checkpoint (fastback, no constraints, each run
qualified before any provider call; `runs/evidence-20261016/`):

| Run | Model | Scored | Baseline Cd | Best trusted Cd | Change | Rejections | Cost | Median step |
|---|---|---|---|---|---|---|---|---|
| real-nano-20261016-001 | Nano | 10 | 0.22456 | 0.22380 | -0.34% | 0 | $0.0022 | 1.07 s |
| real-super-20261016-001 | Super | 25 | 0.22456 | 0.22157 | -1.33% | 0 | $0.0218 | 1.66 s |
| real-super-20261016-002 | Super | 25 | 0.22456 | 0.21797 | -2.93% | 0 | $0.0250 | 1.77 s |

Run 002 used the exploration guidance added after run 001 fine-tuned two
parameters for twenty steps. Single runs at temperature 0.6 are anecdotes, not
a method comparison. The best design's predicted sd is about 0.005, so run
001's gain is inside one sd and run 002's is just over it. These are surrogate
predictions, not CFD-confirmed improvements. About 98% of each step is waiting
for the model; scoring takes about 1.5 ms. No live run hit the gate:
on the real checkpoint every rejection comes from ood_score (see the service
README), and the demo shows it with the pack's genuine-rejection design.

### Matched comparison: agent against the optimisers

One frozen protocol (`runs/matched-20261016/protocol.json`): real checkpoint,
fastback, all-zero start, no constraints, the shared gate, 25 predictor calls
per run. Five runs per method; Nemotron Super with the effects memory
(commit "Give the agent a memory..."), random search and CMA-ES through
Paramveer's `src.optim.baseline --protocol`. All ten agent/optimiser pairs pass
`compare_run pair`. Frontal area is computed afterwards with
`src.geometry.params.frontal_area` on each run's best design; the objective
itself is still Cd.

| Method | Median Cd change | Runs | Median frontal-area change | Median Cd × A change | Gated (per run) | Cost |
|---|---|---|---|---|---|---|
| Random search | -2.79% | -3.42, -2.79, 0.00, -4.13, -0.64 | +4.55% | **+0.98%** | 1, 2, 1, 1, 1 | free |
| CMA-ES | -3.84% | -3.84, -3.78, -1.71, -4.32, -4.53 | +3.26% | **-0.45%** | 0 each | free |
| Nemotron Super | **-3.88%** | -4.49, -1.16, -4.33, -3.88, -2.35 | 0.00% | **-3.37%** | 0 each | $0.134 total |

On Cd alone the agent is level with CMA-ES and ahead of random search. Both
optimisers find much of their Cd gain by widening or raising the car, which
grows frontal area, so their drag area (what the drag force depends on) barely
moves. The agent kept area flat in four of five runs (one run widened the car
to the limit). Five runs per method are descriptive, not a significance test,
and every number is a surrogate prediction. This is the data for the team's
objective decision; under a Cd × A objective all three methods would need
rerunning.

### Local hosting rehearsal (no VM yet)

The real service under uvicorn with both real checkpoints, an admin token,
public runs on and rate limiting at 120/min (`runs/rehearsal-20261016/`):
/ready, /health identity, the five pack designs through /evaluate (the genuine
rejection returns ood_score 7.07 against 4.93), /fields gzipped from 991 KB to
412 KB in 0.8 s, CORS preflight, a public over-limit run refused (422), a wrong
cancel token refused (403), cancel with the right token, and live runs over
HTTP on all three body styles. `python -m src.service.rehearse` then ran 3 of 3
Super runs to completion with the cancel check (fastback -3.87%, notchback
-2.21%, estateback -5.16%, $0.086). Nano found no improvement in one 10-step
public run: use Super for the demo.

| Workstream | Status | Remaining |
|---|---|---|
| Real scalar serving | Qualified on the real bytes, rehearsed over HTTP | `qualify --service-url` against the VM |
| Agent evidence | Nano10, Super25 runs, 5 matched Super runs | Rerun if the objective changes |
| Surface fields | Replayed, source authenticated, gzipped, 0.8 s locally | Latency/RSS on the VM; orientation check in the viewer |
| UI integration | PRs #8-#17 merged; start/cancel and /fields binding in the page | Browser check against the hosted service |
| Fair comparison | Matched recordings for all three methods | Team objective decision; Paramveer adds the agent rows to the results page |
| Deployment/reliability | Image, Caddy guide, `rehearse` command, local rehearsal | Nebius VM, remote qualification, `rehearse --runs 20` |

Spend ledger after all of this: about $0.39 of the $5 cap.

## Stage 1: checkpoint and reference handoff

Serve a single params-v1 checkpoint trained with all three body styles. A
category-held-out checkpoint fails startup; do not bypass this by editing its
metadata. Ensemble accuracy cannot be attributed to this single-model path.
Both trainers now import the frozen schema version. Avneet reports retrained
artifacts with training revision/config, split/seed, hashes and independent
references. Those real files must be transferred separately from a Git pull:
the repository ignores runs/ and *.pt.

Artifact identities (verified locally against the bundle on Oct 16):

- Single all-style scalar: `49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`,
  expected at `runs/drivaernet-random/best.pt`.
- Fields: `6a210c09b09205459f59d17b5ff913648587c9af086fc972e9090e3d69aa688c`,
  expected at `runs/fields/best.pt`.
- Independent packs: `runs/handover/scalar_reference.json` and
  `runs/handover/field_reference.json`.

Use the native scalar pack directly:

```powershell
.\.venv\Scripts\python.exe -m src.service.verify_checkpoint --checkpoint runs/drivaernet-random/best.pt --references runs/handover/scalar_reference.json --out runs/scalar-parity.json
.\.venv\Scripts\python.exe -m src.service.qualify --checkpoint runs/drivaernet-random/best.pt --references runs/handover/scalar_reference.json --out runs/service-qualification.json
```

The second command verifies the real FastAPI routes in-process: /ready,
/health identity/provenance, the frozen /schema, each reference through
/evaluate, all reference pairs through /compare, and the exact gate imported
by the agent. Add `--service-url http://127.0.0.1:8000` to check an already
running server over HTTP. Neither mode calls Nemotron or proves adaptation.
Reports distinguish in-process ASGI from remote HTTP. A synthetic/different
or unready server fails before scoring. Qualification output is never overwritten.

`python -m src.service.verify_checkpoint --checkpoint checkpoints/params.pt
--references checkpoints/references.json --out runs/qualification.json` loads
on CPU and returns nonzero on missing/incompatible artifacts or mismatched
predictions. It never contacts Token Factory. Use one physical command line.

The legacy Track B manifest remains supported with these exact keys:

| Key | Required value |
|---|---|
| `manifest_version` | integer `1` |
| `schema_version` | `"2.0.0"` |
| `preprocess_version` | `"params-v1"` |
| `checkpoint_sha256` | full lowercase 64-character artifact SHA-256 |
| `reference_source` | independent generation method/revision; distinguish model predictions from CFD truth |
| `absolute_tolerance` | positive number at most `0.0001` |
| `thresholds` | `{ "uncertainty": <Cd sigma cutoff>, "ood_score": <calibrated cutoff> }` |
| `references` | 3-5 distinct cases, as below |

Each case contains exactly `id`, complete `params`, `body_style`, and `expected`.
`expected` contains `cd`, `uncertainty`, `ood_score`, `in_distribution`, and
`reject_reason` (null when trusted). Numeric outputs use the absolute tolerance;
trust booleans and rejection text match exactly. Use the shared service gate
to format rejection guidance, with the checkpoint thresholds. References should
cover styles, a trusted perturbation and a genuine rejection where possible.
Exact parity checks serving; it does not independently establish CFD accuracy.

Avneet's native `kind: "scalar_reference"` pack instead carries `checkpoint`,
`training`, `ood_detector`, `thresholds_used`, `split_ids`, `tolerance` and
`designs`. The verifier validates that format and cross-checks its metadata
against the full hashed artifact. It preserves independent expected numbers and
raw contract `rejection_reason`; service `reject_reason` adds adaptation guidance
and is reported separately. There is no need to append that guidance to his pack.
His native exporter explicitly implements Mahalanobis arithmetic, so a native
pack claiming kNN is refused. The pack's training.git_commit currently records
generation revision; actual training_git_revision comes from the checkpoint.

The existing top-level health hash remains short for compatibility. Full hash,
model/schema/preprocess, actual thresholds, output provenance, Git revision,
dirty status and source fingerprint are under `/health.provenance` and recorded
in new `run_started.model_provenance`. Source fingerprint hashes relative source
names/bytes plus service/model requirements, excluding secrets, data and logs.
Git metadata can be unavailable in a container; the source hash still exists.

## Real Nano and Super evidence

Run these separately, inspecting Nano before Super:

```powershell
.\.venv\Scripts\python.exe -m src.agent.evidence --checkpoint checkpoints/params.pt --references checkpoints/references.json --model nano --run-id real-nano-001 --out runs/real-nano-001.evidence.json
.\.venv\Scripts\python.exe -m src.agent.evidence --checkpoint checkpoints/params.pt --references checkpoints/references.json --model super --run-id real-super-001 --out runs/real-super-001.evidence.json
```

The commands also accept Avneet's native paths above unchanged. They qualify
references AND service routes/shared-agent-gate parity before creating the provider, use a fresh
run/output identity, validates a trusted feasible baseline before provider calls,
then runs 10 Nano or 25 Super total evaluations including iteration zero. Nano
thinking is off. Default run cap is $0.04; this command refuses caps above $0.05.
The global ledger/cap still applies. A failed, cancelled or budget-limited run is
saved as incomplete and exits nonzero; no retries search for a flattering trace.

Reports retain qualification, full artifact identity, baseline/best trusted Cd,
absolute/relative change, first trusted improvement if any, actual rejections
and their following scored proposals, usage, timing and stop reason. A lower
prediction is a surrogate result, not CFD confirmation. A rejection follow-up
is evidence to inspect, not an automatically asserted successful adaptation.

## Surface fields and viewer binding

Set `AERO_FIELD_CHECKPOINT` to enable `POST /fields`; the request is the same
strict design object as `/evaluate`. There are no uploaded meshes or client paths.
Absent capability returns 503; a configured invalid checkpoint fails startup.
Invalid designs/meshes return 422; busy returns 503/Retry-After; invalid model
results return 500 with a visible reason. Scalar predictions are independent.

Response fields:

| Field | Meaning |
|---|---|
| `field_schema_version` | `"1.0.0"`, separate from frozen Evaluation |
| `design_id`, `design_sha256`, `body_style` | design identity |
| `mesh` | full `sha256`, hash format, vertex/face counts, association=`face`, query=`face_centroid`, exact `face_indices` |
| `cp`, `sigma`, `region` | aligned per-face predicted Cp, Cp standard deviation and learned region index |
| `regions` | name ordering for the region indices |
| `pressure_summary` | area-weighted means; null for absent regions |
| `unavailable_pressure_regions` | absent region names |
| `provenance` | artifact/model, geometry approximation, coordinate convention, unknown field trust, units and resource limits |
| `timing` | geometry and inference wall time; includes separate through-summary time, not network latency |

The service decodes the exact existing `src.geometry.viewer.mesh_payload`
vertices/faces, preserves their order and predicts every face. It does not use
`params_to_mesh`, whose cap topology differs. For binding in JavaScript:

```javascript
// Hash original buffers, before rendering transforms or vertex duplication.
const prefix = new TextEncoder().encode("viewer-f32le-u32le-v1\0");
// SHA256(prefix || decodedVerticesBytes || decodedFacesBytes)
// vertices: contiguous little-endian float32 XYZ; faces: uint32 triangle indices.
```

Require matching full hash AND vertex/face counts; bind array item i to
`face_indices[i]`. Discard stale responses after design changes. The existing
viewer consumes vertex colors: Paramveer must implement face display or an
explicit face-to-vertex visualization conversion. Do not treat a smoothed sigma
as uncertainty propagated by a statistically validated rule.

Cp/sigma are dimensionless. Field sigma is not Cd sigma and is never fed to the
Cd gate. Generated geometry is approximate, and training-to-viewer orientation
and field trust remain unqualified until independent references establish them.
Pressure-only drag/lift integrals are deliberately absent from this endpoint;
the model does not supply total-force coefficients or validated Cd uncertainty.
The loader requires a valid classifier and recomputes summaries without its
fallback zeros. Hard limits bound mesh/cloud/chunk/classifier/output sizes and
one CPU inference slot; they do not establish a hard time limit.

Avneet's 4be514d handoff supplies the reported field hash, reference format,
conventions and supported claims. Local qualification still requires transfer
of the actual field artifact, generated JSON and original extracted source NPZ.
Check their correspondence using those declared conventions rather than treating
the handoff's measured geometry behavior as validation of the approximate viewer.
Measure CPU latency/RSS on the target VM; tiny fixture timing is not production
evidence. Check region and Cp agreement on real data before labeling it trusted.

The new native field pack is supported by a separate offline parity command:

```powershell
.\.venv\Scripts\python.exe -m src.service.verify_fields --checkpoint runs/fields/best.pt --references runs/handover/field_reference.json --source-npz runs/handover/source/run_3.npz --out runs/field-parity.json
```

Use the extracted source for the run actually named by the pack (the handoff
currently says held-out run 3). The JSON omits its geometry cloud and rounds
query coordinates/normals. The original NPZ is required to replay the exporter's
exact query selection and subsequent cloud selection and infer on original
unrounded tensors. It is not valid to invent the cloud from the 64 query points
or widen tolerance. The report records supplied-source identity and alignment,
not independent authentication of an upstream source hash the pack never supplied.

This command qualifies fixed-query Cp/sigma parity only. It does not qualify
design-to-viewer preprocessing, learned region accuracy, field OOD, uncertainty
calibration or aerodynamic accuracy. CFD targets and the pack's descriptive
performance claims remain distinct. The original training-side pressure-only Cd
MAE 0.0326 used 4,096 query points per car. The received 27 October bundle's
100,000 stored cells per test car gives per-car MAE 0.0218415 and mean absolute
relative error 8.57% (three under-predictions, two over-predictions). This is
stored-cell quadrature, not a complete native CFD surface. Its Cp R² 0.5080 /
MAE 0.14554 loses to the pooled nearest-training-point reference, R² 0.6905 /
MAE 0.10251. Release claims use the fresh independent report in
`web/results/metrics.json`, not fixed-query parity or historical pack prose.
The historical Cl MAE 0.0802 and one lift sign error were not recomputed by that
bundle. Cl remains unavailable for decisions; viewer field orientation and
generated-geometry calibration remain unvalidated.

## Agent-run UI packet for Paramveer

Keep the existing `/runs` shape. Start with:

```json
{"model":"nano","iterations":10,"body_style":"fastback","constraints":[],"run_budget_usd":0.04,"thinking":"off"}
```

POST returns 202 with status and, for public runs, a one-time `cancel_token`.
Poll `/runs/{id}` for state, progress, trusted best, latest logbook/rejection,
cost and timing; poll `/history/{id}` for full scored rows. This is JSON polling,
not streaming. History can initially be 404 before the first score; trusted best
can be null. `accepted` means trusted, not necessarily improved.

Cancel with `POST /runs/{id}/cancel` and `X-Cancel-Token` for the initiating public
session. Admin bearer tokens must not be embedded in browser source. Preserve
queued/running/finished/cancelled/failed/interrupted/unknown states and display
stop_reason: finished can mean spend budget exhausted. Existing service README
documents 401/403/409/422/429/503 behavior and exact access limits.

Honor `/health.output_provenance`: show placeholder Cl as n/a and suppress derived
lift loads/effects; missing pressure is unavailable. Keep synthetic runs labeled.
Field capability/provenance is under `/health.fields`, separate from scalar output.

## Fair comparison packet for Paramveer

First scenario: all-zero baseline, one style/constraint set, same real checkpoint,
gate thresholds and 25 predictor calls per method. Agreed with Paramveer on
Oct 16: matched seeds use 25 evaluations for every method. The 10-call Nano
request in the run-control packet is a cheap smoke run, not a comparison budget. Start with seeded random search
versus one agent run; CMA-ES and repeated statistical comparisons are later work.

```powershell
.\.venv\Scripts\python.exe -m src.agent.compare_run protocol --checkpoint checkpoints/params.pt --evaluations 25 --out runs/comparison-protocol.json
.\.venv\Scripts\python.exe -m src.agent.compare_run agent --checkpoint checkpoints/params.pt --references checkpoints/references.json --protocol runs/comparison-protocol.json --model super --run-id comparison-agent-001 --out runs/comparison-agent-001.json
.\.venv\Scripts\python.exe -m src.agent.compare_run pair runs/comparison-agent-001.json runs/comparison-random-001.json --out runs/comparison-pair.json
```

The protocol freezes checkpoint/source identity, schema/preprocess policy, zero
baseline, style, constraints, actual gate cutoffs, evaluation and admission caps.
Treat it as immutable. The agent command checks qualification and protocol before
spending. Root source changes produce a different fingerprint, even at the same
Git revision. The separate evidence command's trace is reusable only if all
comparison settings and accounting were frozen identically.

The optimiser imports the Track B seam; this is an interface example, not its
implementation:

```python
from src.agent.comparison import ComparisonProtocol
from src.agent.compare_run import evaluator_for
protocol = ComparisonProtocol.from_dict(protocol_json)
score = evaluator_for(predictor, protocol)  # verifies actual artifact identity
baseline = score(protocol.baseline_params, protocol.body_style)
# Paramveer proposes designs and calls score(params, protocol.body_style).
# Persist score.summary("random_search", run_id, stop_reason,
#                       metadata={"seed": seed, "method_settings": settings}).
```

Every predictor invocation consumes a slot, including baseline, gate rejections
and exceptions. Invalid/infeasible/duplicate proposals refused before scoring
consume no predictor slot; evaluator admissions are separately bounded. Agent
and optimiser scoring both stop after a rejected baseline; that run is incomplete.
Nemotron corrections that never reach scoring remain in its event log/summary metadata.
Two-change proposals are an agent method property, not a hidden common design
constraint. Compare trusted-best curves at equal scored-call counts, preserving
cost, timing, refusals and failures. Budget/cancel/failure outcomes stay incomplete;
pair validation refuses mismatched identities and inconsistent accounting.
One matched pair is descriptive evidence, not proof of statistical superiority.

No new optimiser API/algorithm or web code was added. Paramveer owns wiring,
browser validation and independent reporting. Team agreement is still required
for external integration; these examples make that agreement concrete.
