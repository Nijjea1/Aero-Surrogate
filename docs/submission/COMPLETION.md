# Local completion and deployment handoff — 27 October 2026

Integration is on branch **`finish/predeploy-complete`**. Source changes are
committed locally; remote freshness and push are unverified because outbound
GitHub access is restricted. Deployment, video upload and Devpost submission
are deliberately left to the team. The original audit and earlier verification
reports remain dated records, not certification of later source changes.

The verified incoming bundle supplies both the category checkpoint and all thirty
field arrays. Scalar/category/field regrading is complete in private staging;
publication, final scaling results, integrated tests and the refreshed package
are being completed by the coordinator. Check the final release manifest and
current machine-readable report rather than assuming an older ZIP is current.

## Completed

- Replaced provisional scalar metrics with an independent grade of the actual
  served checkpoint. Published all fifteen validated matched search summaries
  and their original scored logbooks/lifecycle events.
- Removed the unsupported hard noise-ceiling interpretation and misleading
  pressure-guided agent, lift, framework-execution and physical-validation claims.
  Corrected missing values that could render as zero/NaN and a remaining stale
  sigma/category assertion discovered during screenshot review.
- Made the evaluator accept and validate the complete nested handover splits.
- Made default pytest collection ignore old artifact/worktree directories.
- Aligned dependency groups, added CI, recorded tested direct versions, and
  supplied a 64-package hash-pinned Windows/Python 3.13 environment lock.
- Added a safe checksum-pinned ZIP installer and a model/evidence bundle builder.
  Models, metrics and benchmark are checked against the same scalar identity.
  Historical traces install under `runs/logs/` for immediate inspection.
- Removed runtime CDN dependence: pinned three.js modules and IBM Plex fonts
  are local, with upstream licence notices and file-checksum manifests.
- Fixed a reproduced provider-accounting defect: ambiguous transmitted-request
  failures retain the reserved cost estimate in the ledger and per-run usage,
  and are not automatically retried. A safe connection-timeout retry remains.
  Budget guards remain estimates rather than guarantees about provider billing.
- Added isolated real HTTP and Playwright checks, public evidence, proof images,
  final submission copy, tooling feedback, video script and teammate handoff.

No model architecture, checkpoint weights, contract fields, parameter bounds or
gate thresholds were changed. Inference remains the qualified scalar plus a
separate optional experimental pressure capability.

## Results bound to the release scalar

Scalar SHA-256:
`49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`

- **R² 0.8749717**, **MAE 0.0061606**, 625 held-out designs.
- Linear reference R² **0.8352341** on the same split.
- Nominal 90% interval coverage **86.56%**; 1σ coverage **64.48%**.
- Parameter-domain false alarms **0.32%**; all sampled joint-extreme corners
  flagged. Body-style indicators explain the strong held-out-style detector
  result; it is not evidence of learned detection of unknown physics.
- Historical five-run medians, 25 evaluations per method: predicted Cd reduction
  **3.875% Nemotron**, **3.841% CMA-ES**, **2.793% random search**. This is a small
  surrogate-only comparison, not evidence of superiority or new CFD validation.

The category checkpoint grades **1,403 held-out notchbacks**, R² **0.4792**,
MAE **0.01016**, 1σ coverage **48.75%**. It remains evaluation-only with its own
threshold; it never replaces the all-styles serving checkpoint. Style-indicator
novelty explains the strong full-input detector; aerodynamic dials alone give
AUROC about **0.536**.

Fields are graded on **500,000 stored query cells** from five held-out cars:
Cp R² **0.5080**, MAE **0.14554**. The pooled nearest-training-point reference
is stronger, R² **0.6905**, MAE **0.10251**. Field 1σ coverage **79.94%**
over-covers stored labels, unlike the scalar intervals. Pressure-only integrated
Cd per-car MAE is **0.0218415**, mean absolute relative error **8.57%**.
The old **0.0326** was a 4,096-point/car training estimate. These results use
extracted stored cells, not complete native CFD quadrature, and establish neither
generated-viewer orientation/calibration nor lift. Source arrays and raw reports
stay separate from the serving ZIP.

Scaling uses new, separately identified CPU subset checkpoints on a fixed test
split. Only the completed, independently graded curve may be published; it does
not replace the production model. Final results are pending coordinator publication.

## Earlier verification snapshots (before final evaluation integration)

The counts below describe the recorded earlier source revisions. The coordinator
must append current integrated suite, model/HTTP, archive installation and browser
results before calling steps 1–9 complete. No fresh Firefox success is claimed:
the later post-merge Firefox setup crashed before navigation and needs a retry.

| Check | Result and scope |
|---|---|
| Final main-environment full suite | **1,019 passed, 4 skipped, 3 warnings**, 106.37 seconds; default pytest collection now works |
| Expected skips | Missing scalar data at the standard training-data path, full field arrays, geometry checkpoint/detector, and Windows symlink-creation privilege |
| Last editorial UI checks | 21 presentation/module tests passed, plus one complete Chromium browser flow after the final wording change |
| Post-suite artifact regression | 42 release packaging/installer/presentation tests passed, one expected Windows symlink skip; packaging rejects stale published-summary checksums |
| Current-source scalar qualification | Passed through real FastAPI lifespan/routes/shared gate against native references |
| Field fixed-query parity | Passed with authenticated source NPZ; no generated-geometry accuracy claim |
| Isolated local HTTP | Six checks passed: identity/readiness, baseline, joint corner rejection, mesh/API configuration, real field inference and scripted HTTP agent/history |
| Fresh isolated environment | Full CPU development install from upstream-hash-verified wheels; `pip check` passed; full suite **1,017 passed / 6 skipped**, then optional Playwright installed and all nine browser-helper tests passed, resolving two optional-import skips |
| Real browser rehearsal | **20/20 fresh logged-out contexts: ten Chromium, ten Firefox**; real scalar, viewer, slider change, domain rejection, reset, recorded logbooks, final metrics and all three benchmark methods; zero page errors |
| Secret scan | No high-confidence key/private-key patterns or exact local credential matches in scanned tracked text or reachable history patches; limited scan, not an exhaustive guarantee |
| Browser assets | All 17 pinned three.js file hashes and nine font hashes matched their manifests |

Detailed scope and original identities are retained in [browser verification](evidence/browser-verification.json),
[judge checks CSV](evidence/judge-checks.csv), [qualification](evidence/local-qualification.json),
[clean environment](evidence/clean-environment.json), [scalar metrics](../../web/results/metrics.json)
and [matched benchmark](../../web/results/benchmark.json).

The main suite's three warnings are the existing Starlette/httpx deprecation and
two deliberate warnings about ignored old geometry parameter names. They are
recorded rather than hidden. New source changes after the broad runs were
comment/documentation corrections, the editorial UI wording check, and a targeted
published-summary integrity check covered by the artifact regressions above.

Proof images from the real local application:
[designer](proof/designer.png), [joint-extreme rejection](proof/rejected-corner.png),
[logbooks](proof/runs.png), [results](proof/results.png).

## External/environment limits preserved honestly

1. **New live Token Factory verification:** one real public start was attempted.
   The service scored its baseline but the outbound provider socket was denied
   with `WinError 10013`. The report is retained as a failed live attempt; it is
   not counted among successful provider runs. The application conservatively
   retained an estimate of `$0.0024582`; that is not a verified provider bill.
   Historical successful Nemotron runs remain separately published evidence.
2. **Docker execution:** access to the Docker daemon pipe was denied. Dockerfiles
   and runbooks are prepared, but this run does not claim a successful image
   build, Linux execution or container deployment.
3. **Submission video:** script, proof screenshots and raw local Chromium WebM
   captures are ready. HyperFrames setup failed fetching its npm package with
   `EACCES`, so no finished narrated/composited submission video is claimed.
   Raw recordings are under the ignored browser rehearsal directories and are
   testing assets, not a public YouTube submission.
4. **Public judging path:** local loopback tests do not establish public HTTPS,
   DNS, remote browser access, provider availability or uptime through judging.

## Local artifact handoff

Generate the ZIP with `python scripts/package_release.py --include-fields`.
The local output is `runs/release/aero-surrogate-models.zip`, with its adjacent
`.sha256` file and tracked [release manifest](release-manifest.json). The ZIP is
not committed or uploaded. The final installation check is recorded separately
in [artifact installation](evidence/artifact-installation.json); that report is
outside the ZIP to avoid a self-referential archive checksum.

See [release instructions](README.md) for the trusted installer, local startup
and verification commands. The artifact notices preserve code/data boundaries,
DrivAerNet's non-commercial attribution, DrivAerML share-alike sample attribution
and browser-asset licences.

## What the team must do next

1. Finish coordinator publication and local qualification: current metrics/graphics,
   tests, artifact hashes, installation and browser checks. Synchronize and push
   from a network-enabled terminal without rewriting history.
2. Deploy the selected source and refreshed artifacts using the API/web runbooks. Validate
   actual HTTPS/CORS/proxy/persistence and full checkpoint identity on the host.
3. In a network-enabled environment, rerun the live browser/provider rehearsal,
   public cancellation/auth checks and external judge walkthrough. Do not mark
   the blocked local live attempt as a success.
4. Publish a trusted artifact download, update pending URLs, record the final
   working deployed footage, upload a public video under three minutes, and
   submit Devpost. Submission and video/tooling copy are already prepared.
5. Fund and monitor the public demo through the verified official judging period, preserving
   the submitted version and enough inference budget for judges.
