# Team plan

Full living version: the team plan doc (link in the group chat). This file is
the offline copy so it is in the repo for judging and for anyone working on a
plane.

## Roles

| | Track | Owns |
|---|---|---|
| **Avneet** | data and training | dataset, preprocessing, PhysicsNeMo recipe, training runs, uncertainty head, checkpoints, splits |
| **Pranoy** | agent and backend | inference service, tool schema, Nemotron loop, uncertainty gating, Serverless Jobs batch scoring, prompt hardening |
| **Paramveer** | geometry, evaluation, product | geometry parameterisation, 3D viewer, evaluation harness, baseline optimiser, UI, README, demo video |

**Why Paramveer owns evaluation:** the person who trains a model is the worst
person to grade it. Splitting training from measurement is normal practice, it
takes a block off Avneet's critical path, and it means the numbers in our
submission were produced by someone with no stake in them looking good.

## Day 0 checklist — ninety minutes, all three together

- [x] Repo created, `contracts.py` written and frozen
- [x] Stub service returning valid `Evaluation` objects
- [x] Contract tests passing (`pytest -q`)
- [ ] Everyone confirms they can run the quick start locally
- [ ] One Nebius account (Avneet holds it), spend cap agreed out loud
- [ ] `.env` created locally by each person from `.env.example`

## The next six days

| Day | Avneet | Pranoy | Paramveer |
|---|---|---|---|
| **Mon Oct 13** | Nebius account, H100 up, PhysicsNeMo installed, drag CSV + param table downloaded | Repo, contracts, stub service pushed | Geometry module started |
| **Tue Oct 14** | Data loader: STL → cached point clouds | Service calling the stub; one Nemotron call succeeds | **Geometry module done and pushed** — Pranoy is blocked without it |
| **Wed Oct 15** | **Gate: baseline R² posted.** Checkpoint handed over | Swap stub for the real baseline model | UI shell live; design picker + confidence badge |
| **Thu Oct 16** | DrivAerML download started; preprocessing begun | Agent tool schema + first loop attempt | three.js viewer rendering meshes |
| **Fri Oct 17** | Preprocessing grind (expect pain) | **Gate: agent loop runs end to end** on bad predictions | Viewer coloured by a fake pressure field |
| **Sat Oct 18** | Preprocessing continues | Uncertainty gate with reasons | Eval harness: R², MAE, per-category splits |

## Gates — do not pass one without meeting it

| By | Gate |
|---|---|
| Oct 15 | A real R² on held-out designs. If not met, drop to AirfRANS. |
| Oct 17 | Crude agent loop running, even with bad predictions. |
| Oct 19 | DrivAerML downloaded and preprocessing started. Last call for the fallback. |
| Oct 21 | Predicted pressure fields that look right beside ground truth. |
| Oct 25 | **Full dress rehearsal** — entire demo start to finish, nothing mocked. |
| Oct 28 | Feature freeze. Paramveer drives; Avneet and Pranoy fix only what breaks. |
| Oct 29 | Submit. Oct 30 is emergency buffer only. |

## Rules that prevent the conflicts

- **Daily 15 minutes, 9pm.** One question each: what is blocking you.
- **Push every day**, even broken work, on your own branch.
- **Announce a blocker immediately**, not at the next standup.
- **Nobody waits on Avneet.** The stub stays live and the others keep building
  against it however training goes. This is the most important rule here.
- **Nobody edits another person's directory.** Ask instead.
- **Instances get shut down** by whoever started them.

## Cut list, in order

1. Physics-informed loss (stretch, never load-bearing)
2. Mixture-of-experts recipe
3. Design sensitivities
4. Volumetric field prediction — surface-only is enough for drag and lift
5. Multiple datasets — one is fine
6. Cross-category experiment — keep the uncertainty head regardless

**Never cut:** a working surrogate, the agent loop, the uncertainty gate, and a
finished three-minute video.

## Submission checklist

- [ ] Working hosted demo at a public URL
- [ ] Public repo with the MIT licence visible at the top
- [ ] README with setup instructions
- [ ] README section naming where Nemotron, PhysicsNeMo, Token Factory, AI Cloud and Serverless Jobs are used
- [ ] Demo video, 3 minutes or shorter, public on YouTube
- [ ] Track selected: Best Apps and Agents
- [ ] Written feedback on Nebius and NVIDIA tooling (ten $100 prizes)
- [ ] Dataset licensing noted (CC BY-NC 4.0)
- [ ] No API keys anywhere in the repo
- [ ] Judge flow tested in a fresh browser, logged out, 20 times

## The three minutes

| Time | Beat |
|---|---|
| 0:00–0:20 | The cost of what we replace: 3M CPU-hours, 256 cores per car |
| 0:20–0:50 | Unseen shape → pressure field and drag in under a second, next to CFD truth |
| 0:50–1:50 | The agent takes over; drag falls while Nemotron narrates its reasoning |
| 1:50–2:20 | Results: R² in and out of distribution, speedup, calibration |
| 2:20–2:50 | The uncertainty flag catches an OOD shape and refuses to answer |
| 2:50–3:00 | "Everyone builds a model that works — we built one that knows when it doesn't" |

Name Nebius and NVIDIA aloud in the narration.
