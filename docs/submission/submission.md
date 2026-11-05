# Aero Surrogate — submission copy

Track: **Best Apps and Agents**.

Repository: https://github.com/Nijjea1/Aero-Surrogate

Test build: https://github.com/Nijjea1/Aero-Surrogate/releases/tag/v1.0.0 (local-run
release; no hosted demo). Public YouTube video: **VIDEO_URL** (replace before submitting).

## Testing instructions

The complete application runs locally with Docker, Git and Python 3:

```bash
git clone https://github.com/Nijjea1/Aero-Surrogate.git && cd Aero-Surrogate
python scripts/install_artifacts.py --archive https://github.com/Nijjea1/Aero-Surrogate/releases/download/v1.0.0/aero-surrogate-models.zip --sha256 70b2829ae2a5e0f1a8bc82f7d1f2321cad1d8fba62ff3f191c4a53b399834a93 --destination .
docker compose up --build
```

Open http://localhost:5173. Without any key: the designer (move a dial, then choose the
"Edge of the box" preset to see a domain-gate rejection, then "Baseline"), the fifteen historical runs with logbooks,
and the complete Results page all work. Live Nemotron runs need your own Nebius Token
Factory key in `.env` (`NEBIUS_API_KEY=...`); a 25-evaluation Super run cost about
$0.03 in our tests. The video shows a genuine live run.

## Short description

An aerodynamic design assistant that lets NVIDIA Nemotron explore vehicle parameters
using a neural drag surrogate, with domain screening and a logbook that checks each
hypothesis against the observed prediction.

## Inspiration

Vehicle aerodynamics needs careful CFD verification, but evaluating every exploratory
design with a solver is expensive. Fast surrogate predictions make candidate screening
practical. They also create a problem: an optimiser can exploit predictions in parts of
the parameter space the model does not understand. We wanted the design loop to show
both its progress and its limits.

## What it does

A designer adjusts sixteen DrivAerNet morph parameters across fastback, notchback and
estateback body styles. The service predicts drag coefficient Cd and its standard
deviation, then checks uncertainty and Mahalanobis distance from the training inputs.
Unsupported combinations are rejected before they can count as an agent's best design.

NVIDIA Nemotron proposes one or two parameter changes, states an expected effect,
calls the surrogate, receives the observed result and updates its search. The logbook
marks whether the observed surrogate change agreed with its hypothesis. Numeric design
constraints, evaluation counts, cancellation and application spend guards are enforced
by code rather than left to the prompt.

The interface includes a designer, a 3D vehicle view, live run history, comparison curves
and independently graded results. A separate experimental model can colour the
approximate viewer mesh by surface pressure. That field is not used by the current
Cd-guided agent and has no established calibration on generated viewer geometry.

## How we built it

The scalar surrogate is a custom heteroscedastic PyTorch MLP trained on 4,165
CFD-labelled DrivAerNet designs. FastAPI exposes inference and run lifecycle endpoints.
The plain JavaScript/three.js interface uses a dynamic Python geometry endpoint.

**NVIDIA Nemotron Super and Nano run through Nebius Token Factory** with typed tool
calls. Saved runs record the provider model, usage, latency, cost and checkpoint identity.
NVIDIA DoMINO research informed the separate custom surface-pressure architecture.
We do not claim a PhysicsNeMo runtime, pretrained DoMINO fine-tuning, Nebius H100
training or executed Serverless Jobs in this release.

## What we measured

- Independent grading of the served scalar checkpoint: **R² 0.8750**, **MAE 0.00616**
  on **625 held-out designs**; the straight-line reference achieved R² 0.8352.
- The nominal 90% prediction interval covered **86.56%** of held-out labels, so the
  uncertainty remains under-calibrated. Sigma alone barely ranked individual errors.
- The checkpoint's parameter-distance detector falsely flagged **0.32%** of ordinary
  test designs and flagged all sampled jointly extreme corners. This is a result on
  those probes, not proof that all out-of-distribution designs are detectable.
- In five historical fastback runs per method, each with 25 scored evaluations,
  median **predicted Cd reduction** was **3.875% for Nemotron Super**, **3.841% for
  CMA-ES**, and **2.793% for random search**. Those similar medians support comparable
  search performance in this small experiment, not a claim that the agent beats CMA-ES.
- The separate category-holdout checkpoint achieved **R² 0.4792**, **MAE 0.01016**
  on **1,403 unseen notchbacks**, with 1-sigma coverage only **48.75%**. It is an
  evaluation checkpoint, not the production model. Explicit style indicators explain
  the apparently perfect style detector; shape dials alone gave AUROC about **0.536**.
- Experimental surface pressure achieved **Cp R² 0.5080**, **MAE 0.14554** over
  **500,000 stored query cells** from five test cars. A pooled nearest-training-point
  reference did better: **R² 0.6905**, **MAE 0.10251**. Both numbers must travel together.
  The network's 1-sigma coverage was **79.94%**, over-covering those stored labels.
  Pressure-only integrated drag per-car MAE was **0.0218415**, mean absolute relative
  error **8.57%**; this excludes viscous drag and complete native-surface quadrature.

The machine-readable release report is authoritative for identities, the fresh local
timing and any completed scaling curve. Do not substitute a historical scaling curve
or the earlier 4,096-point/car Cd estimate of 0.0326 for these stored-cell results.

## Challenges and limitations

The uncertainty head and domain detector answer different questions. We report
interval coverage and the detector's behaviour separately. Detecting a held-out body
style is strongly helped by explicit style indicators; it is not discovery of novel
aerodynamic physics. The primary gate demonstration is therefore a combination of
individually legal parameters that is jointly far from the training data.

Optimisation results are surrogate predictions; no new CFD solve validates the selected
designs. Equal evaluation counts do not imply equal runtime or cost. An individual Cd
sigma is not an uncertainty estimate on the difference between two designs.
The search objective remains Cd. Any Cd × approximate viewer frontal-area analysis
is post-hoc and is not an optimisation result for a different objective.

The viewer mesh approximates the original morph, smoke is illustrative, and race/energy
comparisons use stated simplifying assumptions. Lift and wall shear are unavailable.
Solver coefficient variation is not asserted as a hard ceiling on attainable R².

## Next steps

Validate selected designs with new CFD, independently validate field orientation and
calibration on generated geometry, broaden starting designs and constraints in the
optimisation study, and evaluate any recalibration on frozen held-out data.

## Data and licences

Source code: MIT. DrivAerNet, by Mohamed Elrefaie and collaborators, retains its
CC BY-NC 4.0 non-commercial terms. DrivAerML, by Neil Ashton and collaborators,
retains CC BY-SA 4.0. Dataset and artifact terms are distinct from the source-code
licence; attribution and provenance accompany distributed samples and weights.

## Final form checks

- [ ] Replace VIDEO_URL with the tested public YouTube URL.
- [ ] Confirm the v1.0.0 release download installs with the testing instructions.
- [ ] Link the public evidence package and factual [tooling feedback](tooling-feedback.md).
- [ ] Check public video playback and the release download logged out.
- [ ] Submit the entry and have a second teammate verify confirmation.
