# Final local completion and deployment handoff

All nine pre-deployment steps are complete locally. Final integration is on
local `main`; GitHub freshness and push remain unverified because this runner
cannot connect to GitHub. Deployment, hosted verification, the finished video
and submission remain with the team.

## Completed work

1. Preserved original commit history and integrated the tooling, evaluation and
   product worktrees. No production model or gate threshold was replaced.
2. Verified all 43 incoming payload hashes and staged 30 field arrays, the CSV
   and the evaluation-only category checkpoint. Original bundle unchanged.
3. Fixed fresh field CLI output, existing-report merges, dataset-root handling,
   cross-platform path privacy, bounded CPU chunks and corrupt-input reporting.
4. Independently regraded the actual scalar, the category-holdout checkpoint and
   all 500,000 stored field queries. Checkpoint/split/array identities recorded.
5. Published one complete metrics file and regenerated Results, README and its
   graphics. Retained all fifteen historical matched search runs unchanged.
6. Corrected historical integration-number scope and dataset licences:
   DrivAerNet CC BY-NC 4.0; DrivAerML CC BY-SA 4.0. Native reference packs retain
   their original bytes and the earlier 4,096-point/car estimate.
7. Froze Cd as the search objective. Pressure remains experimental and separate
   from the scalar domain gate; no generated-mesh accuracy or lift claim.
8. Trained five genuine CPU scaling checkpoints with unchanged schedules,
   nested training subsets and identical 625-design validation/test sets.
9. Completed local regression, native model qualification, real HTTP, browser,
   public evidence/asset/secret checks and checksum-pinned release packaging.

## Current measured results

| Experiment | Result and scope |
|---|---|
| Served scalar | R² 0.87497, MAE 0.006161, 625 held-out designs |
| Linear scalar reference | R² 0.83523 on the same split |
| Scalar intervals | 1-sigma coverage 64.48%; nominal 90% coverage 86.56%; under-coverage |
| Held-out notchback | R² 0.47921, MAE 0.010161, 1,403 designs; evaluation-only model |
| Surface pressure | Cp R² 0.50803, MAE 0.145535, 500,000 stored queries across five cars |
| Stronger field reference | Pooled nearest-training-point R² 0.69052, MAE 0.102510; beats the network |
| Pressure-only integral | Per-car MAE 0.0218413, mean absolute relative error 8.57%; three under/two over |
| New data-scaling experiment | 146–2,915 training designs, R² 0.77835–0.87852; separate checkpoints |
| Historical matched search | Five runs/method, 25 evaluations; median predicted Cd reductions 3.875% Nemotron, 3.841% CMA-ES, 2.793% random |

The field integral is a scaled estimator over stored cells, excludes viscous drag
and does not establish native full-mesh quadrature. Style-indicator novelty
explains strong category OOD detection; shape-only detection is near chance.
Surrogate optimisation improvements have not been verified by new CFD.

## Final local verification

- **1,065 tests passed, two expected skips, three existing warnings** in 104.27 s.
  Skips: optional geometry checkpoint/detector and Windows symlink privilege.
  `pip check` passed. No unexplained test failure remains.
- Scalar/shared-gate native-reference qualification, authenticated fixed-query
  pressure parity and six real HTTP checks passed. Baseline Cd is about 0.22456.
- **20/20 final Firefox fresh contexts passed**, with desktop flows and
  Designer/Runs/Results at 390/412 px. Seven earlier Chromium desktop/mobile
  contexts passed, but two subsequent Chromium attempts had host
  `ERR_NO_BUFFER_SPACE` resource failures. They are preserved, not counted as
  passes. This is not a clean twenty-context cross-browser certification.
- All 17 three.js and nine font hashes matched. The scoped tracked-text/history
  scan found no high-confidence key/private-key or exact local credential match.
- Serving archive installation verifies every payload, all fifteen published
  summary hashes, model identities, ZIP CRC and fifteen 25-row histories.
  Its final archive hash and installed-model HTTP result are recorded in the
  installation receipt; the receipt stays outside the ZIP to avoid self-reference.

Current evidence: [tests](evidence/final-tests.json),
[browsers](evidence/final-browser-verification.json),
[judge checks](evidence/final-judge-checks.csv),
[model/HTTP qualification](evidence/final-qualification.json),
[input receipt](evidence/evaluation-inputs.json),
[secret-scan scope](evidence/final-secret-scan.json),
[release manifest](release-manifest.json) and
[installation receipt](evidence/artifact-installation.json).
See [independent evaluation](INDEPENDENT_EVALUATION.md) for commands and hashes.
Earlier reports remain dated historical evidence, including the separate
[clean-environment installation](evidence/clean-environment.json).

Current real application proofs: [Designer](proof/final-designer.png),
[rejection](proof/final-rejected-corner.png), [Runs](proof/final-runs.png),
[Results](proof/final-results.png), [390 px](proof/final-results-mobile-390.png)
and [412 px](proof/final-results-mobile-412.png).

## External limits and next actions

No deployment, video rendering/upload or Devpost submission was performed.
GitHub fetch/push and new Token Factory calls require a network-enabled terminal.
Historical successful Nemotron runs remain labelled historical; local scripted
HTTP checks do not certify live provider availability. Docker daemon access was
denied, so container build/runtime certification must occur on the deployment host.
Hosted HTTPS/CORS/proxy/persistence and external Chromium/mobile checks remain
part of deployment verification.

Use [NEXT_STEPS.md](NEXT_STEPS.md) for the exact remaining sequence. No extra
training run, H100, missing evaluation file or objective change is required.
