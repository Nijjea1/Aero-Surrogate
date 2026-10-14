# Checkpoint handoff — training → service

**From:** Avneet (`src/data`, `src/train`)
**To:** Pranoy (`src/service`), Paramveer (`src/eval`)
**Status:** stage 1. Written Oct 13, before the Oct 15 gate, so the format is
settled before there is a real number in it.

This is not a contract change. `src/contracts.py` is untouched. It documents
the one thing `contracts.py` does not pin down and cannot be inferred from the
checkpoint alone.

---

## The problem this solves

`POST /evaluate` takes `params`. The stage-1 model takes a **point cloud**. So
the service has to do:

```
params ──> params_to_mesh ──> sample to a point cloud ──> model
          (Paramveer)         (???)                       (Avneet)
```

That middle step is not in any contract. If the service samples 1024 points
when the model trained on 4096, or centres on the centroid when training
centred on the bounding box, or samples vertices instead of sampling the
surface by area, **the model still returns a confident number and that number
is wrong**. Nothing fails. No test catches it.

So every checkpoint carries a `preprocess` key describing exactly how its
input was built. Read it at load time; do not hardcode the values.

---

## What is in the checkpoint

```python
ckpt = torch.load("runs/<run>/best.pt", map_location="cpu", weights_only=False)
```

| Key | What it is |
|---|---|
| `state_dict` | weights |
| `config` | the full resolved training config (model shape lives here) |
| `norm_stats` | `target_mean`, `target_std`, and for the tabular model `param_mean`, `param_std` |
| `version` | `<run>-<date>-<git hash>` |
| `val_metrics` | selection metrics |
| **`preprocess`** | **how to build model input — read this** |
| `test_metrics` | held-out metrics |
| `held_out_category` | set when trained on the OOD split, else `None` |
| `schema_version` | `1.0.0` |

The five keys the plan promised are all there; the rest is additive, so a
loader that only reads those five still works.

---

## Loading it

`build_model(ckpt["config"])` reconstructs the right architecture for either
model kind, so the service does not need to know which one it got.

```python
import torch
from src.train.model import build_model, split_prediction

ckpt  = torch.load(path, map_location=device, weights_only=False)
model = build_model(ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

norm = ckpt["norm_stats"]
spec = ckpt["preprocess"]
```

### Point-cloud checkpoints (`preprocess["version"] == "pointcloud-v1"`)

```python
import numpy as np, trimesh

def mesh_to_input(mesh: trimesh.Trimesh, spec: dict) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, spec["n_points"])   # area-weighted
    pts = np.asarray(pts, dtype=np.float64)
    if spec["normalize_geometry"]:
        centre = mesh.bounds.mean(axis=0)                            # bbox centre
        radius = np.linalg.norm(mesh.vertices - centre, axis=1).max()
        pts = (pts - centre) / max(radius, 1e-9)
    return pts.astype(np.float32)                                    # (n_points, 3)
```

All four details matter: area-weighted surface sampling, `n_points` from the
spec, bounding-box centre (not centroid), max **vertex** radius. The canonical
implementation is `src/data/dataset.py::sample_mesh` — if you prefer, import
that instead of copying this.

### Parameter checkpoints (`preprocess["version"] == "params-v1"`)

```python
cols = spec["param_columns"]            # ORDER IS PART OF THE CONTRACT
x = np.array([[params[c] for c in cols]], dtype=np.float32)
x = (x - np.array(norm["param_mean"])) / np.array(norm["param_std"])
```

### Predicting

```python
with torch.no_grad():
    mu, sigma = split_prediction(model(torch.from_numpy(x).to(device)))

cd    = float(mu[0])    * norm["target_std"] + norm["target_mean"]
sigma = float(sigma[0]) * norm["target_std"]      # a scale: std only, no mean
```

`split_prediction` is the only place log-variance becomes a standard
deviation. Use it rather than reimplementing `exp(0.5 * logvar)`, so the
conversion cannot drift between training and serving.

---

## Three things to decide together

**1. `UNCERTAINTY_THRESHOLD = 0.015` is a placeholder and currently a no-op.**
Measured sigma on the synthetic fixture runs ~0.009, so every proposal passes
the gate and the gate does nothing. `contracts.py` already says to calibrate it
against the reliability diagram. That needs a real number from Paramveer's
harness before the Oct 17 agent-loop gate — otherwise we demo a gate that
never fires.

**2. Stage 1 cannot fill `pressure_summary`.** It predicts one scalar. The
eight `PRESSURE_REGIONS` keys arrive in stage 2 with surface fields. Until
then the service should keep returning the stub's values and we should not
pretend otherwise in the demo.

**3. Sigma alone is NOT an OOD detector. Measured, on real data.**

Trained on real DrivAerNet with one body style held out entirely:

| Split | Test R² | mean sigma | 1-sigma coverage |
|---|---|---|---|
| random (in-distribution) | 0.873 | 0.0067 | 0.62 |
| notchback held out (OOD) | 0.277 | 0.0081 | **0.35** |

In distribution the head is well calibrated. Out of distribution R² collapses
to 0.28 while sigma widens only 20% — so coverage falls to 0.35 where it
should be 0.68. **The model is confidently wrong on a body style it has never
seen**, which is exactly the failure the gate exists to prevent.

So `in_distribution` cannot be `sigma < UNCERTAINTY_THRESHOLD`. It needs a
real distribution check, and cheap options exist:

- distance in standardised parameter space to the k nearest training designs
- a density estimate on the training parameters (a Gaussian fit is enough to
  start)
- the explicit body-style indicator, which the model already takes as input:
  if the requested style was not in training, say so without guessing

This is good news for the submission, not bad. "We measured our own OOD
detector and found sigma insufficient, so we added a density check" is a
stronger story than a gate nobody stress-tested. But it has to be built.

**4. The parameter spaces do not match, concretely.** `contracts.PARAM_BOUNDS`
has 8 absolute dimensions. Real DrivAerNet has 23 morph **deltas** from a
baseline body, in mixed units, and `ride_height`, `boat_tail_angle` and
`roof_curvature` have no counterpart at all. Partial correspondences exist
(`rear_diffuser_angle` ~ `B_Diffusor_Angle`, `track_width` ~ `A_Car_Width`),
but the semantics differ — delta vs absolute — so they are not interchangeable.

Two coherent routes, and this is a group decision:

- **Serve the point-cloud model.** Paramveer's `params_to_mesh` then only has
  to make plausible car shapes, not reproduce DrivAerNet's parameterisation.
  Needs no contract change, and the OOD check above is what keeps it honest.
- **Change the contract** to DrivAerNet's 23 parameters plus body style. More
  faithful to the data, but all three of us have to agree and Paramveer's
  geometry module changes.

One more thing the delta parameterisation explains: without the body-style
input, test R² is 0.60; with it, 0.873. The parameters are relative to a
different baseline per category, so a model that cannot tell which baseline it
is perturbing cannot resolve them. That 0.60-vs-0.87 gap is the same gap the
literature reports between the diverse set and within-category results, and
here it has a concrete mechanism. Report both numbers.

---

## What is NOT in here

No metrics suite. `evaluate()` in `train.py` computes a quantity to early-stop
on plus a coverage diagnostic, and that is all. Per-category R², reliability
diagrams, OOD detection rate, data-scaling curves and the speedup number are
Paramveer's harness, run against these checkpoints. Splits come from:

```bash
python -m src.data.dataset --dump-splits runs/splits.json --config configs/baseline.yaml
```

which writes both the random split and the held-out-category split, plus the
same `preprocess` spec, in one file.
