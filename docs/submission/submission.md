# Aero Surrogate — submission copy

Track: **Best Apps and Agents**.

Repository: https://github.com/Nijjea1/Aero-Surrogate

Public demo: **pending team deployment**. Public YouTube video: **pending team upload**.
Do not submit this document with those two entries unresolved. The local application,
release package and recording script are prepared separately from publication.

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

## Challenges and limitations

The uncertainty head and domain detector answer different questions. We report
interval coverage and the detector's behaviour separately. Detecting a held-out body
style is strongly helped by explicit style indicators; it is not discovery of novel
aerodynamic physics. The primary gate demonstration is therefore a combination of
individually legal parameters that is jointly far from the training data.

Optimisation results are surrogate predictions; no new CFD solve validates the selected
designs. Equal evaluation counts do not imply equal runtime or cost. An individual Cd
sigma is not an uncertainty estimate on the difference between two designs.

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

- [ ] Replace the pending demo and video entries with tested public URLs.
- [ ] Verify the pushed release commit and artifact hashes match the application.
- [ ] Link the public evidence package and factual [tooling feedback](tooling-feedback.md).
- [ ] Check public video playback and application access logged out.
- [ ] Submit the entry and have a second teammate verify confirmation.
