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

**3. `in_distribution` needs a definition, and sigma alone is not it.** A
bigger problem: the model trains on DrivAerNet meshes, and at serve time it
sees meshes from `params_to_mesh` — a *different shape generator*. Those are
arguably out of distribution by construction, which would make the uncertainty
head either reject everything or, worse, fail to notice. Worth ten minutes on a
standup. Options, cheapest first:

- Fit `params_to_mesh` to produce DrivAer-like bodies, so the two generators
  overlap.
- Serve the **parameter** model instead, which has no mesh round trip and so no
  mismatch. This is the main reason it exists.
- Define `in_distribution` on parameter-space distance to the training set
  rather than on sigma.

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
