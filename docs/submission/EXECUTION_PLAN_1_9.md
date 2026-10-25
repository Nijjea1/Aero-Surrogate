# Execution plan: complete pre-deployment steps 1–9

Prepared after read-only bundle verification on 27 October 2026. Current source
baseline: `main` at `ebc3dde`. Deployment, final video production/upload and
Devpost submission follow this plan's completion.

## Input receipt: verified and sufficient

The actual bundle is doubly nested under the workspace root:

```text
C:/Pranoy/NBIS/aero-eval-artifacts-2026-10-27/aero-eval-artifacts-2026-10-27/
```

Coordinator verification: **43/43 SHA256SUMS entries passed**, zero missing or
mismatched payloads and zero unexpected files after excluding `.DS_Store` and
the outer `__MACOSX` extraction metadata. Payload bytes: **146,948,587**.

Two independent read-only agents verified the following:

- Category SHA-256:
  `6b6033efbe90d01a10534f9cc0a93665e76227095da18fb47ca39f2d2db6848f`.
  Seed 0, schema 2.0.0, strict current model load, finite tensors, exact split
  counts **2,348 / 414 / 1,403**, notchback held out, own threshold **4.716300983351741**.
- Field SHA-256:
  `6a210c09b09205459f59d17b5ff913648587c9af086fc972e9090e3d69aa688c`.
  Strict current model load and finite tensors.
- All **thirty** field NPZs load with `allow_pickle=False`, have the expected
  arrays/shapes/types, finite values, positive areas/scales and valid regions.
  Each has **100,000 stored query cells and 8,192 cloud points**.
- Field splits are disjoint, cover runs 1–30, and match the checkpoint:
  twenty train, five validation, five test; test runs **3, 11, 12, 22, 27**.
- The supplied CSV has **4,165 unique finite labelled designs**. Exported
  category split IDs and metadata exactly match the checkpoint.

There is no remaining input blocker to regrading scalar/category/full stored-cell
fields. Scaling checkpoints were not supplied, but can be trained locally on CPU.
These checks establish transfer/input integrity, not a fresh numerical regrade.

## Corrections identified before execution

Preserve the incoming bundle unchanged, including its original inventory/reports.
Correct derived public reports and notices instead:

1. Pressure integrals show **three under-predictions and two over-predictions**,
   not four under-predictions. Runs 3, 11, 22 are below truth; 12 and 27 above.
2. The raw report gives pressure-only per-car MAE **0.021841491559060972** and
   mean **absolute** relative error **8.5677058878%**. These are over all stored
   evaluation cells, not a guarantee of complete native CFD surface quadrature.
3. Cp R² **0.5080306779** / MAE **0.1455350102** loses to the pooled
   nearest-training-point reference, R² **0.6905177845** / MAE **0.1025096801**.
   That reference pools sampled training points across twenty cars and averages
   eight neighbours in position/normal space; it does not copy one entire car.
4. Coverage worsens on held-out styles while sigma widens insufficiently. Do not
   repeat the inventory's literal “not wider” claim or compare different model
   heads as though they were one controlled distribution-shift experiment.
5. DrivAerML's author dataset card specifies **CC BY-SA 4.0**, despite the
   incoming inventory's CC BY-NC notice. Keep DrivAerNet's **CC BY-NC 4.0**
   attribution separate. Source: https://huggingface.co/datasets/neashton/drivaerml
6. Old reports use source `be76dd0`, obsolete ceiling/share fields and a command
   path fragment. Historical latency prose also differs slightly from its JSON.
   Regrade on the integrated source; do not publish the old reports verbatim.

## Parallel execution arrangement

Use three fresh worktrees/branches from the agreed integrated source. Each agent
commits only its assigned paths. Artifacts stay outside Git; agents write separate
output folders rather than concurrently overwriting the published metrics file.

| Worker | Worktree / branch | Scope |
|---|---|---|
| Tooling agent | `.audit-track-b/worktrees/final-tooling`, `finish/final-tooling` | Field CLI standalone output and root-path handling, related tests, any path-publication regression revealed by the imported reports |
| Evaluation agent | `.audit-track-b/worktrees/final-evaluation`, `finish/final-evaluation` | Independent scalar/category/field regrading, reference comparison, scaling experiments and machine-readable evidence in private staging |
| Product/evidence agent | `.audit-track-b/worktrees/final-product`, `finish/final-product` | Results rendering, README/generated graphics, accurate baseline labels, notices, historical-number scope, submission/tooling/video copy |
| Coordinator | integration checkout | Source/artifact receipt, frozen serving identities, merge sequencing, package availability/identity wiring, release generation, complete tests and browser/installer checks |

Start tooling fixes, evaluation setup/scaling, and document preparation in
parallel. Merge the tooling fix before the authoritative final field CLI run.
Publish only the reports that passed the evaluation gate. Run heavy field grading
and scaling sequentially on CPU rather than alongside each other and browser
captures. Agent concurrency does not require simultaneous heavy model workloads.

## Steps and completion gates

### 1. Lock and synchronize source

Record `ebc3dde` as the starting point; confirm local/remote identity from a
network-enabled environment when possible. Snapshot the production scalar hash
`49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`
and existing matched protocol. Create the worktrees without rewriting history.

**Gate:** explicit source revision and serving identities; remote connectivity
limitations remain explicit rather than being mistaken for a successful fetch.

### 2. Stage verified evaluation inputs

Keep the received bundle immutable. Use explicit input paths or verified copies
under ignored `data/` and `runs/` locations; ignore Mac extraction metadata.
Record input digests and split authorities. Preserve supplied raw reports as
historical evidence. Keep the large evaluation bundle separate from the small
serving ZIP and its archive-size limit.

**Gate:** all required inputs load from the execution environment and no raw
datasets/checkpoints are accidentally staged for Git.

### 3. Finish evaluator wiring

The cross-platform sanitizer fix is already implemented/tested. Fix
`src.eval.fields --metrics` for deliberate fresh-output use while preserving
merge behaviour for existing complete metric files. Make dataset-root versus
`fields/`-directory handling explicit or safely support both, with regression
tests. Preserve unrelated metric sections, meaningful failure status and public
path privacy. Do not change physical targets, splits, weights or gate thresholds.

**Gate:** fresh output, existing-file merge, supported root forms, unavailable
inputs and path-with-spaces cases all behave as documented.

### 4. Independently reproduce the complete evaluation

Regrade the production scalar on its 625 test designs; the category model on
1,403 held-out notchbacks; and fields on all 100,000 stored cells of each test
car. Include constant and pooled-neighbour references. Report per-car and pooled
field results separately and retain the pressure-only nature of integration.

Keep the category model offline: it must never replace the serving model or
overwrite its approximately 4.9348 OOD threshold. Record the category model's
approximately 4.7163 threshold separately. Record new local CPU latency with scope.

**Gate:** expected identities/splits match, results reproduce to sensible numeric
tolerance, and any discrepancies are investigated rather than copied over.

### 5. Publish one current results package

Generate current-schema metrics from the integrated source. Preserve all fifteen
matched Cd-search runs and their original experiment identities. Expose scalar
and category calibration limitations, style-indicator detection, and the field
network's loss to the nearest-point reference. Remove obsolete hard-ceiling
interpretations and keep missing scaling/lift results explicit.

**Gate:** UI, README, generated graphics and submission copy agree with the same
graded files and accurately named reference algorithms.

### 6. Correct historical references and dataset notices

Scope old **0.0326** as the earlier 4,096-point-per-car estimate and the new
**0.021841** as the all-stored-cell result. Update handoff/integration prose and
publish the new report without silently editing old checkpoint metadata or
native reference packs. If a replacement pack is needed, version/regenerate and
requalify it. Correct public licence notices while keeping the original incoming
inventory unchanged. Correct the three-versus-two integral sign count.

**Gate:** no misleading unqualified old figure, incorrect public data licence,
or broken historical reference checksum.

### 7. Freeze the search objective and product scope

Retain Cd for the release's matched optimisation experiment. Cd × approximate
viewer frontal area may appear only as a checked post-hoc analysis. Keep
pressure experimental and separate from the scalar trust gate. No H100
retraining, objective migration, lift/shear claims or pressure-driven agent
integration is needed for this release.

**Gate:** demonstrated behaviour and every public capability claim agree.

### 8. Complete scaling if feasible without replacing production

Train separate CPU evaluation checkpoints for nested **5%, 10%, 25%, 50%, 100%**
subsets using the existing scaling tool. Freeze the validation/test split and
record every subset/checkpoint identity. Grade on the same test designs. The
new 100% scaling run is not automatically the production model, even if its
training set size matches. If training fails or exceeds the agreed execution
timebox, preserve an unavailable result and omit the unverified curve.

**Gate:** an actual reproducible curve, or an explicit unavailable outcome;
never a pasted historical curve.

### 9. Integrate, package and verify before deployment

Merge tested worktree commits, then run the full suite and model qualification.
Update package availability metadata to reflect verified inputs and completed
evaluations rather than its current hard-coded missing-data flags. Build a
versioned serving/evidence ZIP; keep large raw arrays separate. Verify each
archive/published-summary checksum, fresh installation, native references,
recorded runs, real local HTTP, current Results rendering, desktop/mobile
browsers and public metadata/secret scans. Retain old release evidence as dated
snapshots. If raw reports or new metadata need publication, sanitize and generate
them rather than hand-editing historical JSON.

**Gate:** clean committed source, honest experiment/release identities, verified
installer/bundle, no unexplained test failures, and an updated completion report.
Do not claim new live Token Factory, Docker or public-host checks if the
environment still blocks them. Stop before deployment, final video and submission.

## Input requests

**No further files, accounts or keys are needed to start steps 1–9.** The only
remaining external limitation is network/runtime access for pushing code and
later verifying the hosted application/provider, not missing evaluation inputs.
