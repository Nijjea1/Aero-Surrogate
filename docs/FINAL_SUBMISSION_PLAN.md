# Aero Surrogate — final submission plan

Prepared 27 October 2026 from the [repository audit](REPO_AUDIT_2026-10-27.md).
This replaces the original build schedule for release decisions. It is a plan, not a statement that the remaining gates have passed.

**Ship a constrained aerodynamic design assistant: real Cd surrogate, explicit domain screening, Nemotron tool loop, inspectable engineering logbook, and an honest equal-evaluation comparison.** Keep pressure visualisation as a clearly limited secondary capability. Freeze the model architecture now.

## 1. Submission story and boundaries

Suggested title: **Aero Surrogate — a trust-gated aerodynamic design agent**.

One-line pitch: “Nemotron explores vehicle design parameters with a fast neural drag surrogate, while a domain gate rejects unsupported combinations and a logbook checks each hypothesis against the measured prediction.”

Lead with three findings:

1. The actual served scalar checkpoint achieves independently reproduced **R² 0.8750 / MAE 0.00616 on 625 held-out designs**, versus linear-reference R² 0.8352.
2. A separate distance check matters because sigma alone barely ranks errors. On the audited test, parameter-domain false alarms are **0.32%**, and all sampled joint-extreme corners are flagged. Show one corner whose individual values are all legal.
3. On five 25-evaluation fastback runs, Super's median **predicted Cd reduction is 3.875%**, versus **3.841% CMA-ES** and **2.793% random search**. Present as comparable to CMA-ES on this small experiment, with additional inspectable reasoning and enforced numeric constraints.

Avoid “first-ever”, “CFD replacement”, “knows when it is wrong”, “safe autonomous engineering”, “beats CMA-ES”, “50,000 autonomous designs”, and “96% of the achievable limit”. Neither a density gate nor an uncertainty head proves physical correctness. Do not claim general natural-language constraint parsing: the implemented grammar is an explicit numeric comparison.

Use “predicted Cd reduction” throughout. A real drag-force reduction depends on Cd times reference area, and neither a viewer-estimated area nor a lower scalar Cd proves a new physical vehicle improves. CFD remains the verification step after surrogate screening.

## 2. Release priorities and owners

No contract changes are required. Owners implement fixes in their existing lanes; shared changes follow the team's agreement rule. The estimate column is planning guidance, not a guarantee.

| Priority / task | Owner | Deliverable and acceptance gate | Estimate |
|---|---|---|---|
| P0: remove hard noise-ceiling interpretation | Paramveer; Avneet reviews label semantics | R²/MAE retained; no “maximum achievable” or “irreducible noise proven” in UI, README graphics, handoff or submission | 1–2 h |
| P0: freeze artifacts | Avneet | Scalar + field identities match audited hashes; category model and required full field arrays handed over if being reported; seeds/splits/configs and provenance included | 1 h |
| P0: fix final evidence chain | Paramveer | Grade release scalar independently; grade category/field only if correct artifacts/data arrive; replace provisional website metrics; common release manifest | 2–4 h |
| P0: publish matched benchmark | Paramveer + Pranoy | Validated 15-run summaries/traces included in public evidence package; website and README show all methods and limitations | 1–2 h |
| P0: deploy real application | Pranoy; Avneet handles account if needed | Public HTTPS web + API, exact checkpoint hash in health, fields configured only if enabled, mesh endpoint works, persistence survives restart | 2–4 h |
| P0: test judge path | Paramveer leads, all three execute | 20 fresh logged-out browser attempts recorded, including successful live Super runs; public demo works without owner token | 2 h |
| P0: record and upload video | Paramveer | Public YouTube video **under** 3 min, working app footage, final numbers and technology names audible/visible | 2–3 h |
| P0: complete submission | Paramveer; all verify | Actual submitted Devpost entry, working links, track, public repo/licence, description and tooling feedback | 1 h |
| P1: clean-install reproduction | Pranoy + owners | Tested requirements/environment record, documented artifact download/checksum/load, `pytest tests -q`; unwrap handover splits for harness | 1–2 h |
| P1: correct licence/docs | Paramveer + Avneet | Separate code/data notices, DrivAerML attribution, actual stack, team, no stale lift/stack claims | 1 h |

Do not spend remaining time on Serverless Jobs, H100 retraining, framework migration, a new upload pipeline, ensemble serving, wall shear, lift, physics loss, new datasets or pressure-to-agent integration. Pressure context would change the claim and demand new field-validity evidence; it is not a release dependency.

## 3. Schedule from today to submission

All operational times below are Toronto time. Use actual dates rather than the weekday names in the original plan.

| Deadline | Gate | Cut or fallback if missed |
|---|---|---|
| **Oct 27, 6pm** | Model choices, hashes, final story, claim removals and evidence inventory frozen | Serve the already qualified single scalar; do not wait for the ensemble |
| **Oct 27, 10pm** | Public real-model application opens, runs, cancels, rejects a corner and preserves logs | Use another existing CPU host if needed; real Token Factory calls satisfy the platform condition |
| **Oct 28, noon** | Final scalar report + matched benchmark public; category/field report either verified or explicitly absent | Omit missing experiments from headlines; do not turn manifest metrics into independent results |
| **Oct 28, 5pm** | Browser judge rehearsal completed; only release-blocking bugs remain | Prioritise stable designer, runs and domain rejection; remove optional field overlay if misleading or unreliable |
| **Oct 28, 10pm** | Video cut, screenshots, description, feedback and README agree on the same release | Record a shorter, clearer loop; label any replay or time compression |
| **Oct 29, noon** | All URLs tested logged out; final release commit/artifacts captured; Devpost submitted | Fix access or upload issues immediately; do not treat a draft as submission |
| **Oct 29, 6pm** | Second person verifies submitted state and links | Correct only defects; preserve submitted-version evidence |
| **Oct 30, before 1pm** | Emergency buffer ends | No planned first submission on deadline day |

The published cutoff is **October 30 at 10am PDT / 1pm EDT**. Rules list judging December 1–15; maintain free judge access until **December 15, noon PST / 3pm EST**, including sufficient inference and hosting budget. [Official rules](https://nebiusglobalaihackathon.devpost.com/rules).

If this plan is read after one of today's times, preserve the task order and compress optional polish; do not expand scope to compensate.

## 4. One evidence package for every public surface

Create `docs/submission/` for small public evidence and an immutable artifact release for weights and larger files. Raw datasets stay outside Git. Record the licence of each redistributed sample or derived asset.

| File / item | Required contents |
|---|---|
| `release-manifest.json` | Release Git SHA, source hash where supported, model SHA-256, schema, preprocessing, train/val/test counts and IDs or split-file hash, commands, tested environment, timestamp |
| `metrics.json` | Final independent scalar metrics, per-style metrics, reference, calibration curve, explicit absent experiments; no unsupported hard-ceiling headline |
| `benchmark.json` | Matched protocol, all 15 runs, five repetitions per method, medians and spread, gate counts, cost/time, surrogate-only limitation |
| Matched logs/summaries | Copy the checked `runs/matched-20261016` data into the release package; exclude secrets and private prompts; retain failed/incomplete evidence when relevant |
| Qualification reports | Fresh scalar service and field parity summaries; exact scope and limitations retained |
| Model download | Trusted immutable release URL, checksums and load instructions; no user-uploaded checkpoint execution |
| `judge-checks.csv` | Attempt number/time, browser, deployment/release identity, action, result, run ID and failure notes |
| `submission.md` | Final description, limitations, technology mapping, feedback, repo/demo/video URLs and selected track |

Existing historical evidence has its own Git revision and must retain it. Do not relabel October 16 runs as October 27 experiments or imply the old benchmark ran on the final deployed host. Include both experiment provenance and release packaging provenance.

For the scalar rerun, the full handover split package needs extraction first:

```python
import json
from pathlib import Path
pack = json.loads(Path("runs/handover/splits.json").read_text())
Path("runs/final-splits.json").write_text(json.dumps(pack["stage1"]["parameters"]))
```

Then, once the owner has supplied the normal dataset path:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests -q
.\.venv\Scripts\python.exe -m src.eval.harness --checkpoint runs/drivaernet-random/best.pt --data data/drivaernet/parametric.csv --splits runs/final-splits.json --out runs/final-metrics.json --label "Release scalar evaluation"
.\.venv\Scripts\python.exe -m src.service.qualify --checkpoint runs/drivaernet-random/best.pt --references runs/handover/scalar_reference.json --out runs/final-service-qualification.json
```

These are scalar-only commands. Add the category checkpoint or field dataset only when available and verified; adjust their paths explicitly. Regenerate the site from final files only after fixing the ceiling interpretation in the rendering/generation code. Never just replace headline numbers by hand.

If a new statistical calibration step is attempted, fit any scale factor on validation residuals only, freeze it and rerun test coverage. Recalibration changes sigma, gate acceptance and comparison conditions: either rerun affected benchmarks under a new protocol or preserve the old benchmark policy explicitly. The lowest-risk release is to disclose current under-coverage and retain the verified protocol.

## 5. Hosted release and judge walkthrough

Use the documented two-container architecture and one API worker. The web container is required because `/api/mesh` is dynamic; static hosting alone does not reproduce this UI. Configure actual API origin/CORS, real checkpoint mounts, persistent logs/ledger, public-run limits and HTTPS. Use Token Factory for the real Nemotron runtime. AI Cloud is an option, not a prerequisite for every component. [Official platform FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs).

Operator checks before recording:

1. `/ready` is healthy; `/health.synthetic` is false and the scalar SHA matches the release manifest. Check field SHA separately if enabled.
2. Designer loads in an incognito browser without a token or custom query string. Reset gives Cd about 0.22456 for the audited default fastback; compare by artifact reference tolerance, not screenshot rounding.
3. Change one parameter and show Cd/sigma/domain score. Lift shows unavailable. Flow smoke is labelled illustrative; field Cp has its own trust caveat.
4. Start a public Super run. Polling, history, chart and logbook work. Changing tabs or reloading does not lose the run. The model's wrong directional hypothesis is visibly marked when the observed surrogate delta disagrees.
5. Show matching baselines from the checked benchmark, with all runs selectable and the equal-evaluation restriction stated.
6. Load the real in-bounds joint-extreme reference and show rejection. The best-trusted chart must not improve from the rejected row. Return to baseline and confirm recovery.
7. Test public cancel with its own token; missing/wrong token cannot cancel another caller's run. Admin credentials stay out of browser recordings and public config.
8. Restart containers, confirm historical runs and ledger survive, and confirm interrupted runs are labelled. Check HTTPS, CORS and proxy trust from an external browser, not localhost alone.
9. Exercise two tabs, a busy field request, rate-limit exhaustion/recovery, provider failure and a visible budget-exhausted state. Errors must not silently switch to synthetic output.
10. Record real API scalar p50/p95, cold/warm field response time including transfer, and full agent-step/run time on the deployed host. Keep all three timings separate.

Twenty browser attempts should cover a complete designer → live loop → history → rejection path, split across team members and at least two browser engines where available. Supplement them with the existing API rehearsal tool; do not call API-only repetitions “fresh-browser testing”. All failures are recorded and resolved or disclosed.

Plan the full judging-period budget. Historical Super spend is about 2.7 cents per 25-evaluation run, but provider pricing and retry behaviour can change. Set a sustainable public quota, preserve enough funds for judges, monitor refusals and top up the application lifetime cap deliberately. A $1 public daily cap would allow only roughly 36 such historical runs per day before other overhead; it is a limit, not a promised capacity. Do not leave the demo exhausted for judges.

## 6. Video plan — target 2:45 to 2:55

Record the working application; this plan does not require creating an animation project. No arbitrary mesh-upload beat or claimed CFD ground truth for generated geometry.

| Time | Screen | Narration content |
|---|---|---|
| 0:00–0:15 | Baseline car and design dials | “CFD is valuable and expensive. Aero Surrogate helps explore candidate vehicle designs before the next CFD verification.” |
| 0:15–0:35 | Change a dial; show Cd, sigma, domain verdict | “Our parameter-based drag model learns from 4,165 CFD-labelled designs. An independent test on 625 held-out designs gives R² 0.875 and MAE 0.00616.” |
| 0:35–1:20 | Start real Super run, show tool steps and chart | “NVIDIA Nemotron Super runs through Nebius Token Factory. It proposes a hypothesis, calls the surrogate, reads the observed result, and updates its search. The service enforces numeric design constraints and a fixed budget.” Show one observed contradiction, then recovery. |
| 1:20–1:45 | All-method benchmark | “Across five runs per method at 25 evaluations, median predicted Cd reduction was 3.88% for Nemotron, 3.84% for CMA-ES and 2.79% for random search. This small experiment shows comparable search performance with an inspectable logbook. These are predictions, not new CFD validation.” |
| 1:45–2:15 | Joint-extreme legal corner, rejection, best curve unchanged | “Every dial here is individually legal, but their combination is far from the training data. The gate rejects it and reports why. Sigma alone would not be enough.” |
| 2:15–2:35 | Calibration chart, optional pressure overlay | “We report limitations: our nominal 90% interval covered 86.6% of held-out labels. Surface pressure is a separate experimental model; its confidence on the generated mesh is not established.” Skip the overlay if it distracts. |
| 2:35–2:50 | App + repo/demo links | “A fast design assistant with explicit domain limits, reproducible comparisons and a trace engineers can inspect. Nemotron on Nebius powers the loop; CFD remains the final check.” |

Use a clearly labelled time-lapse or recorded real run if compressing a 40–60 second sequence. Keep enough uninterrupted live interaction to establish the application functions. Say Nebius and NVIDIA aloud. Show exact technology names on screen. Avoid copyrighted soundtrack or unlicensed third-party assets. Check upload visibility and playback logged out before submitting.

## 7. Devpost draft

**What we built.** Aero Surrogate is a vehicle-design assistant that combines a neural drag surrogate with NVIDIA Nemotron on Nebius Token Factory. Designers explore sixteen morph parameters across three body styles. The agent proposes changes, states an expected effect, receives a measured surrogate result and keeps a logbook of what worked and what did not.

**Why it matters.** Fast prediction enables more candidate screening before expensive CFD verification. Fast models also invite optimisers to exploit unsupported regions. Our service therefore checks predicted uncertainty and distance from the training designs before a result can count as the agent's best design.

**What we measured.** Our served scalar model independently achieved R² 0.875 and MAE 0.00616 on 625 held-out designs. In five 25-evaluation fastback runs per method, Nemotron achieved a median predicted Cd reduction of 3.875%, CMA-ES 3.841%, and random search 2.793%. We report calibration shortfalls and distinguish domain screening from physical correctness. These optimisation results remain surrogate predictions, not CFD-verified improvements.

**How it works.** Custom PyTorch surrogates, FastAPI, a hash-bound experimental pressure endpoint, a three.js interface, typed Nemotron tools, budget guards and persistent run logs. NVIDIA DoMINO research informed the separate pressure architecture; this implementation does not claim a PhysicsNeMo-trained production model. Source code is MIT; datasets retain their own licences.

Add actual demo/repository/video links and final release identity. Include category or field numbers only after independent final grading. Attribute dataset authors and link dataset terms. Select **Best Apps and Agents**. Keep the story aligned with demonstrated functionality rather than every item in the original proposal.

## 8. Tooling feedback draft and collection

Base feedback on actual traces, not generic praise. Current evidence supports:

- **Nemotron/Token Factory:** typed tool calls worked for a multi-step loop; saved Super runs completed 25 evaluations. Include real model ID, date, call count, usage and cost. Proposal latency dominated total time, while local surrogate evaluation was much faster; distinguish provider latency from application work.
- **Model interaction:** the measured-effect memory, one/two-parameter proposal restriction, duplicate detection and observed-direction checks were important application controls. Show one failure/correction trace and explain which part was the team's orchestration rather than a platform defect.
- **Useful request:** inspect actual saved `provider_corrections`/failure traces before claiming a reproducible provider problem. If warranted, request clearer tool-call/usage diagnostics or model-capability metadata, with reproduction steps and impact.
- **NVIDIA scientific tooling:** credit the examples/research actually consulted. Report the download-path mismatch only with the old URL/version, failure and working canonical dataset URL. Do not say Curator/PhysicsNeMo-CFD was used when it was not.
- **AI Cloud/Serverless:** include operational feedback only if deployment/execution actually happened and logs exist. It is acceptable to say these services were not used.

Before finalising, add one concrete successful workflow, one reproducible friction point, and one actionable improvement. Preserve the useful scale: calls/run, median latency, cost and error frequency. Redact credentials and private account IDs.

## 9. Final go/no-go checklist

- [ ] Public repo and MIT badge/licence are visible; actual final commit is pushed and identified.
- [ ] Public HTTPS demo loads logged out and serves the qualified scalar, not the stub.
- [ ] Immutable weights/checksums and clear clean-install/run instructions are available.
- [ ] Final website metrics, README, graphics, video and description identify the same models.
- [ ] No unsupported hard-ceiling, pressure-guided agent, lift, shear, CFD-validated improvement or H100/Serverless usage claims remain.
- [ ] Matched benchmark includes all three methods, five runs each, protocol, cost/time and limitations.
- [ ] Real rejection demonstration uses legal individual coordinates and a genuine domain rejection.
- [ ] Calibration shortfall is visible; OOD style indicators are disclosed.
- [ ] Dataset attribution and separate CC BY-NC / CC BY-SA notices are explicit.
- [ ] Fresh-clone setup, intended tests and model qualification pass; skips/limitations recorded.
- [ ] Twenty browser judge attempts are recorded and failures resolved; cancellation/restart/budget recovery tested.
- [ ] Secret scan rerun on final tree/history and public artifacts; .env and keys excluded.
- [ ] Public YouTube video is under three minutes and plays logged out.
- [ ] Required description, track, technology mapping and factual tooling feedback entered.
- [ ] Team eligibility/entrant details checked by the team; external form state verified.
- [ ] Devpost is submitted, not merely saved; second teammate verifies links and confirmation.
- [ ] Hosting owner, monitoring and budget assigned through December 15 judging end.

The scalar-led release remains viable if field/category experiments are omitted honestly. A synthetic-only demo, inaccessible application, or claims unsupported by the published evidence do not pass the release gate.
