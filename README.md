# Aero Surrogate

A neural surrogate for vehicle aerodynamics, plus an LLM design agent that uses
it to search geometry space autonomously.

**Thesis:** at eight hours per CFD evaluation a human tries five designs. At
fifty milliseconds an agent can try fifty thousand.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/)
— Best Apps and Agents track. Submission closes Oct 30, 2026, 1:00pm EDT.

> **This is the skeleton.** `src/contracts.py` is complete and frozen.
> Every other module is signatures and TODOs, owned by one person.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # never commit .env
pytest -q                   # the three contract tests should pass now
```

As each person fills in their modules, more of `scripts/dev.sh` starts working.
When all of it runs, the system is integrated.

---

## Who owns what

| Path | Owner | Everyone else |
|---|---|---|
| `src/contracts.py` | **shared, frozen** | changes need all three to agree |
| `src/data/`, `src/train/`, `configs/` | Avneet | read only |
| `src/service/`, `src/agent/` | Pranoy | read only |
| `src/geometry/`, `src/eval/`, `src/optim/`, `web/` | Paramveer | read only |

If you need something changed in someone else's directory, ask them. Do not
edit it. This one rule prevents almost every merge conflict a three-person team
hits in two weeks.

---

## The three contracts

Everything joins at exactly three seams, all defined in `src/contracts.py`.

**1. Evaluate** — Pranoy serves it, Avneet's model fills it, Paramveer consumes it.

```
POST /evaluate  {"params": {"ride_height": 0.14, ...}}
->  {"cd": 0.291, "cl": -0.04, "uncertainty": 0.006,
     "in_distribution": true, "pressure_summary": {...}}
```

**2. Geometry** — Paramveer owns it, Pranoy and the viewer import it.

```python
from src.geometry.params import params_to_mesh, sample_surface
from src.contracts import PARAM_BOUNDS, clamp_params
```

**3. Run log** — Pranoy writes it, Paramveer plots it. One JSONL per run in
`runs/logs/`, append-only so a chart can stream a run that is still going.

`tests/test_contracts.py` is the tripwire. If someone changes a contract without
telling the others, these fail before the integration call does.

---

## Build order — who is blocking whom

| Due | Who | What |
|---|---|---|
| **Oct 13** | Pranoy | `StubPredictor` + the service. Unblocks everyone else. |
| **Oct 14** | Paramveer | `params_to_mesh` + `sample_surface`. Pranoy is blocked without it. |
| **Oct 15** | Avneet | **Gate:** a real R² posted in the group chat. |
| **Oct 17** | Pranoy | **Gate:** agent loop runs end to end, even on bad predictions. |

Full schedule in [`docs/PLAN.md`](docs/PLAN.md).

**Nobody waits on Avneet.** The stub stays live and the others keep building
against it however training goes. This is the most important rule here.

---

## Architecture

```
CFD dataset ──> preprocess ──> train surrogate ──> inference service
 (DrivAerML)   (Serverless    (AI Cloud H100,      (ms per design,
                 Jobs)         PhysicsNeMo)         + uncertainty)
                                                          │  ▲
                                                          ▼  │
                                              Nemotron 3 Super design agent
                                                  (Token Factory)
```

The top row runs once. The bottom row runs thousands of times. Every candidate
the agent proposes is scored and returned with a confidence value, and a
low-confidence candidate is **rejected before the agent can optimise into it**.

### Why the uncertainty gate matters

An optimiser will happily walk into whatever region of design space the model
hallucinates best in. The gate is not a disclaimer bolted on at the end — it is
load-bearing, and it is the thing almost no other submission will have.

---

## Build stages

**Stage 1 — baseline.** Point-cloud regressor predicting drag only. Trains in
minutes on one GPU. Purpose: prove the pipeline and produce a number to beat.
Reference: the DrivAerNet RegDGCNN baseline is ~3M params and reports R² above
0.84 within a car category.

**Stage 2 — the real surrogate.** PhysicsNeMo's unified external aerodynamics
recipe (DoMINO or Transolver) on DrivAerML, predicting surface fields and
integrating them for drag and lift. **Do not reimplement this** — use the
upstream recipe at `NVIDIA/physicsnemo` → `examples/cfd/external_aerodynamics/`.

**Stage 3 — the agent.** `src/agent/designer.py`.

---

## Data

**Do not download DrivAerNet++ in full — it is 39 TB.** See
[`data/README.md`](data/README.md). This is the single most likely way this
project fails.

Before training anything:

```bash
python -m src.data.dataset --check --config configs/baseline.yaml
```

DrivAerNet is **CC BY-NC 4.0 — non-commercial**. Stated here and in the
submission.

---

## What we report

Published results show R² of 0.82–0.90 within a single car category dropping to
about **0.60** across the full diverse set. Closing that gap is the
contribution, so every number is reported per category, plus:

- field-level error separate from integrated coefficients
- a reliability diagram (when the model says 90%, is it right 90%?)
- OOD detection rate on a held-out category
- speedup measured honestly, including preprocessing
- the agent against the CMA-ES baseline at equal evaluation budget

**If the agent does not beat the baseline optimiser, we report it anyway.**

---

## NVIDIA and Nebius usage

| Requirement | How it is met |
|---|---|
| NVIDIA open source model | Nemotron 3 Super + Nano (the agent); PhysicsNeMo architectures (the surrogate) |
| Runs on Nebius | AI Cloud H100 (training), Token Factory (agent inference), Serverless Jobs (preprocessing, batch scoring) |

**Compute budget:** hackathon credits are $25 + $25 and will not cover stage 2.
Budget $50–150 self-funded. An H100 left running overnight is about $70 — shut
down instances when you walk away.

---

## Team

| | Track |
|---|---|
| Avneet Nijjer | data and training |
| Pranoy | agent and backend |
| Paramveer | geometry, evaluation, product |

---

## License

MIT — see [LICENSE](LICENSE).
