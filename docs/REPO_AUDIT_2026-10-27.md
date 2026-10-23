# Aero Surrogate repository audit — 27 October 2026

Audited revision: `be76dd0ab1cc549d8c59576e049e9bfb5fa58010`, branch `main`.
Companion: [final submission plan](FINAL_SUBMISSION_PLAN.md).

**Verdict:** there is a credible working research prototype and agent product here. Submission readiness is conditional on publishing the correct evidence, correcting scientific claims, and proving the deployed judge path. Further model architecture work is a poor use of the remaining two days.

## Scope and limits

Reviewed the tracked repository, contracts, data and training paths, uncertainty and OOD implementation, geometry, service, agent, comparison protocol, evaluation tools, frontend, containers, documentation, and available local handover/run artifacts. There were 165 tracked files before this audit, including 106 Python files and 12 JavaScript files. No `AGENTS.md` was found in the workspace. The initial working tree was clean.

Executed the full intended test suite, fresh scalar qualification, authenticated field numerical parity, real generated-mesh field inference, independent scalar grading from the public label table, ten matched-pair validations, and a limited secret scan. Existing paid-provider evidence was inspected; no new Token Factory calls were made. No deployment, account changes, or submission was performed. No application source, frozen contract, or teammate-owned implementation was changed.

This was not a fresh interactive browser rehearsal, a full CFD validation, a complete dependency vulnerability audit, or proof of the current remote deployment. Playwright is not installed in this virtual environment. Full DrivAerML test arrays and the category checkpoint are absent locally. Those limitations prevent fresh independent field-accuracy and category-model grading here.

## What exists versus the original brief

| Area | Audited implementation | Submission consequence |
|---|---|---|
| Scalar surrogate | Custom heteroscedastic PyTorch MLP; 237,186 parameters; 16 morph deltas plus three style indicators | Describe a parameter-to-Cd surrogate, not arbitrary uploaded-mesh inference |
| Input seam | Contract 2.0.0; zero offsets are the baseline; body style is explicit | Retire the original eight invented geometry parameters and examples |
| Pressure model | Custom DoMINO-inspired point-cloud/local-neighbourhood network; 646,754 parameters | Credit NVIDIA research accurately; no evidence of a PhysicsNeMo recipe runtime or pretrained DoMINO fine-tune |
| Data | Scalar: 4,165 DrivAerNet designs; field manifest: 20 train / 5 validation / 5 test DrivAerML cars | Do not imply the full DrivAerNet++ dataset was trained on |
| Trust gate | Cd sigma plus checkpoint Mahalanobis threshold; rejection reason fed back to agent | A measured screening heuristic, not a guarantee of physical correctness |
| Agent | Nemotron Super/Nano through Token Factory; typed tools, bounded proposals, measured-effect memory, logbook, cancellation | Real multi-step model use is evidenced |
| Agent pressure context | Prompt supports regional means, but scalar `/evaluate` returns `{}`; neither CLI nor in-process agent fetches `/fields` | Current live evidence is Cd-guided, not pressure-guided optimisation |
| Lift/shear | Scalar Cl is labelled placeholder `0.0`; no operational wall-shear prediction | Do not advertise lift prediction or full pressure-plus-viscous drag integration |
| Geometry | Approximate lofted body with wheels, intended for visualisation | Viewer mesh is not the dataset's ANSA morph or independently verified CFD geometry |
| Fields | Real model output on hash-bound viewer triangles; explicit unknown trust/calibration/orientation provenance | Numerical parity and pretty pressure maps do not validate generated geometry |
| Product | Designer, three.js viewer, illustrative flow, race simulation, Runs, Results | Demonstrate the design loop first; distinguish smoke/race assumptions from CFD |
| Hosting | API and dynamic web server Dockerfiles; Caddy deployment guides | Deployment instructions exist; an operational public URL is not established by local files |
| Nebius use | Saved real Token Factory calls; AI Cloud hosting is described | Do not assert H100 training, Curator, PhysicsNeMo-CFD or Serverless Jobs execution without new evidence |

## Fresh verification

| Check | Result | Meaning |
|---|---|---|
| Intended suite | **956 passed, 3 skipped, 3 warnings**, 106.41 s | Strong functional coverage; skipped/data-dependent and browser gaps remain |
| README command `pytest -q` | Collection fails on ten inaccessible old `runs/` directories in this workspace | Use `pytest tests -q` or configure `testpaths = tests` |
| Scalar qualification | Passed using real checkpoint and native scalar reference pack | Correct serving, identity, response and shared-agent-gate behaviour |
| Field reference verification | Passed; Cp maximum difference about `3.58e-7` | Authenticated fixed-query inference parity only |
| Real generated-mesh field smoke | Passed on default fastback; about **1.08 s** total, about **1.01 s** inference | Working field inference on this CPU, not CFD accuracy or an HTTP/browser SLA |
| Current scalar independent grade | R² **0.8749717**, MAE **0.0061606**, RMSE **0.0079214**, n = **625** | Confirms current serving artifact's real held-out performance |
| Linear reference | R² **0.8352341** | Valid within-harness reference; not a comparison against published mesh-model leaderboards |
| Interval coverage | 1σ **64.48%**, nominal 68.3%; 90% interval **86.56%**; 2σ **92.96%** | Error bars under-cover; “calibrated uncertainty” needs qualification |
| Sigma/error ranking | Spearman **0.0657** | Sigma barely ranks individual-design errors |
| Parameter OOD | Test false alarms **0.32%**; sampled corners flagged **100%** | Measured on this checkpoint and these synthetic probes; not all possible OOD shapes |
| Matched comparisons | All ten Super-versus-baseline pairs validate; common protocol `322057110805…` | Same artifact, start, gate and 25-call budget; rejects count, duplicate/invalid proposals do not consume predictor calls |
| Secret checks | No matches in scanned current files or reachable history patches; local credential also not found in history | Limited patterns/exact-value scan, not a certificate that no historical secret exists |

The fresh scalar grader initially rejected `runs/handover/splits.json` with `KeyError: 'random'`. The handover package nests these fields under `stage1.parameters`, while `src/eval/data.py:load_splits` expects them at the root. Extracting that object to an audit-only file made grading work. This is a reproduced packaging incompatibility, not a model failure.

Fresh reports are under ignored `.audit-track-b/submission-*20261027*`; a portable summary is [audit evidence](submission/audit-evidence-2026-10-27.json). The public CSV was downloaded only into the ignored audit directory. It did not alter the data owner's working files.

## Available experiment evidence

### Matched search experiment

Five repetitions each, one fastback starting design, 25 scored calls including baseline, common checkpoint and gate. Values below are reductions in **surrogate-predicted Cd**, with positive percentages meaning improvement.

| Method | Individual reductions (%) | Median reduction |
|---|---|---:|
| Nemotron Super | 4.493, 1.158, 4.335, 3.875, 2.350 | **3.875%** |
| CMA-ES | 3.841, 3.784, 1.708, 4.318, 4.534 | **3.841%** |
| Random | 3.421, 2.793, 0.000, 4.131, 0.640 | **2.793%** |

Super median logged cost is **$0.02710/run**, median step latency **1.593 s**, and median sum of step times **43.59 s**. About **99.05%** of step time waits on model proposals. These are historical local measurements, not current provider prices or hosted latency guarantees. CMA-ES also uses its own proposal policy and is far faster in wall-clock; equal evaluation count does not mean equal time or cost.

The 0.034 percentage-point difference between Super and CMA-ES medians is not meaningful evidence of superiority. Five runs and one starting point cannot support broad search-performance claims. No candidate in this experiment was validated by a new CFD solve. Do not interpret a candidate's Cd sigma as a confidence interval on its improvement; prediction-error correlations between designs are not estimated.

Existing HTTP rehearsal evidence includes a final passing **3/3** Super rehearsal plus public cancellation. An earlier report contains a failed wrong-token status assertion. Preserve both; confirm deployed wrong-token handling instead of selectively citing only successes. Neither report establishes twenty fresh-browser successes.

### Artifacts and reporting identities

| Artifact | SHA-256 | Evidence status |
|---|---|---|
| Served scalar | `49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a` | Present; fresh qualification and independent held-out grade passed |
| Pressure model | `6a210c09b09205459f59d17b5ff913648587c9af086fc972e9090e3d69aa688c` | Present; fresh numerical parity and generated-mesh smoke passed |
| Category model | `6b6033efbe90d01a10534f9cc0a93665e76227095da18fb47ca39f2d2db6848f` | Manifest only here; reported R² 0.4792 requires independent regrade |
| Website scalar metrics | `fc03f081efc01d4f801555645835b9aa3e6012378e6e738cd753695eb48405ef` | Old schema 1.0.0 artifact, not the served scalar above |

The handover manifest reports Cp R² **0.5079**, Cp MAE **0.1476**, pressure-only Cd MAE **0.03261** on five test cars, and unreliable lift. Those are training-side artifact metrics, not fresh full-test independent findings from this audit. Website `field.available` remains false. The ensemble is not the served single model; its reported R² 0.9132 must not be mixed into single-model headlines.

## Findings, ordered by submission risk

### A01 — P0: the hard “noise ceiling” claim is unsupported

Locations: `src/eval/metrics.py:noise_ceiling`, `src/data/drivaernet.py:noise_ceiling`, `src/train/ensemble.py`, `docs/HANDOFF.md`, `web/results.js`, generated README graphics.

The code computes `1 - mean(Std Cd²) / var(Average Cd)` and presents it as an upper bound on attainable test R². This requires the variance of the actual regression-label error. A solver's within-run standard deviation is not automatically the variance of its averaged label, nor proof of independent run-to-run uncertainty. The dataset paper describes coefficient averaging over the last 1,000 solver iterations with mean and standard deviation. Effective sample size, autocorrelation and bias would matter when estimating uncertainty in that mean.

Therefore “96% of achievable”, “no model can go further”, and claims that the residual is proven irreducible label noise must be removed or explicitly demoted to an unvalidated reference calculation. The formula's numeric output can be reproduced; its claimed interpretation is not established. This does not invalidate the ordinary held-out R² or MAE.

Source: [dataset authors' paper, simulation-validation section](https://openreview.net/pdf/e461dee13f81957e41856f4bbf57fc0b0b7ccd1f.pdf). This statistical criticism is the audit's inference from the stated label construction, not a claim made by the authors.

### A02 — P0: public results are detached from the demonstrated artifact

`web/results/metrics.json` is explicitly provisional and identifies old checkpoint hashes. `web/results/benchmark.json` omits the newer matched Nemotron experiment. Root README repeats both. This gives judges conflicting identities and hides the strongest measured product result.

Required: independently grade the selected release artifacts, publish checked run summaries and traces, regenerate the presentation, and bind everything to one release manifest. Retain older provisional results under a clearly dated label.

### A03 — P0: no established judge-ready hosting or final video evidence

The repository has production-oriented building blocks, not proof of deployed availability. No real demo URL or public final YouTube URL was established by this audit. The external GitHub page is publicly accessible and shows MIT, but cached browsing does not prove the remote contains this exact local revision.

Official rules require free judge access through **15 December 2026, noon Pacific / 3pm Toronto**, not merely through submission day. Budget and monitoring must cover that period. [Official rules](https://nebiusglobalaihackathon.devpost.com/rules).

### A04 — P0: important evidence and weights only exist outside Git

`runs/`, model files and handover bundles are ignored. That is appropriate for large/restricted data, but there is no documented immutable public artifact release containing the selected weights, checksums, references and final graded evidence. A fresh clone starts the synthetic stub. Provide a usable artifact download path or a clearly supported retraining path, and make the default judge link use the real model.

### A05 — P1: uncertainty and OOD claims need narrower definitions

Measured under-coverage and weak sigma/error ranking contradict an unqualified “calibrated uncertainty” headline. The category detector's 100% result largely comes from explicit body-style indicator novelty: removing those indicators catches only about **2.64%** of held-out styles in the independent detector experiment. This is not discovery of unseen aerodynamic physics. Use the in-bounds jointly extreme corner as the primary gate demo, and report category OOD separately.

### A06 — P1: field and agent capabilities are conflated

`src/agent/designer.py` passes only the scalar evaluation's regional summary to the prompt; that summary is empty on the real served checkpoint. Field-serving provenance correctly says generated-mesh calibration and orientation are not independently validated. Keep these caveats visible. Remove claims of pressure-driven agent reasoning, arbitrary unseen upload validation, operational lift, wall shear, or guaranteed physical drag reduction.

### A07 — P1: reproducibility and packaging have concrete gaps

The handover splits format fails the harness without extraction. Root `pytest -q` collects ignored run directories. No tracked CI workflow, dependency lock or released tested-environment manifest was found. Requirements are mostly lower bounds; the root install list also omits some serving/test dependencies that the more specific requirements supply. Validate the documented clean install rather than assuming the populated virtual environment proves it.

### A08 — P1: licence documentation is incomplete

MIT code licensing is present. The DrivAerNet dataset's non-commercial CC BY-NC notice is present; DrivAerML's **CC BY-SA 4.0** notice and attribution should also be explicit. Document terms for distributed samples/derivatives and provenance of the model weights separately; do not imply MIT relicenses third-party datasets. [DrivAerNet repository](https://github.com/Mohamedelrefaie/DrivAerNet), [DrivAerML authors' dataset card](https://huggingface.co/datasets/neashton/drivaerml/blob/main/README.md).

### A09 — P2: deployment limits need operational rehearsal

Positive controls include non-root containers, explicit CORS, request-size and rate limits, cancellation tokens, bounded field work, startup failure for bad configured checkpoints, output provenance, reserved spend and a single-worker policy. These are material strengths.

Verify the configured proxy boundary, persistent logs/ledger, graceful restart, two browser tabs, public daily-budget exhaustion, and rate-limit recovery on the real host. Avoid scaling workers sharing one run-log directory. Spend accounting uses estimated prices; transport timeouts/retries can leave billing ambiguous, so call it an application budget guard rather than an absolute provider-billing guarantee. The lifetime spend cap also needs enough headroom for the judging period.

### A10 — P2: product and documentation polish

Root README has a dangling `st.` and team navigation with no team section; the HTML meta description still advertises lift. `docs/PLAN.md`, `data/README.md` and comments describe abandoned stack/schedule details and refer to a missing `CLAUDE.md`. CDN fonts/three.js are external availability dependencies. Smoke and race simulation are explanatory visualisations, not a learned velocity field or real vehicle validation. Fix visible misleading copy before optional visual polish.

## Rules and external checks

Verified on 27 October against the [official event page](https://nebiusglobalaihackathon.devpost.com/) and [FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs): Best Apps and Agents remains appropriate; actual NVIDIA-model calls through Token Factory satisfy the platform requirement. The rest of the application can be hosted elsewhere; Serverless is optional. Promotional credits are Token Factory credits, not a published AI Cloud grant. Do not spend the final days adding decorative services.

The [public repo](https://github.com/Nijjea1/Aero-Surrogate) is visible and has MIT detected. Remote branch protection, latest remote commit, deployed application and submission-form state were not verified.
