# Checkpoint handoff — training → service

**From:** Avneet (`src/data`, `src/train`)
**To:** Pranoy (`src/service`), Paramveer (`src/eval`, `src/geometry`)
**Schema:** 2.0.0 · updated Oct 14

Read this before wiring `/evaluate` to a real checkpoint. It is short on purpose.

---

## 1. The parameter space changed. Read this first.

`src/contracts.py` moved to **schema 2.0.0**. The old eight parameters
(`ride_height`, `boat_tail_angle`, `roof_curvature`, …) were invented on day 0
and the real DrivAerNet data does not have them — three had no counterpart at
all, and the rest are morph **deltas** from a baseline body in mixed units, not
absolute dimensions. A model trained on real data could not have been served
behind a 1.0.0 `/evaluate`.

The new space is **16 morph parameters + body style = 19 model inputs.**

Not 23. All 23 were measured; the six side-mirror and door-handle dials plus
the greenhouse angle contribute under 0.003 R² between them, and dropping them
makes the model *better*:

| Contract | Random R² | Held-out-category R² |
|---|---|---|
| all 23 + body style | 0.873 | 0.277 |
| **16 aero + body style** | **0.875** | **0.479** |
| 8 aero + body style | 0.845 | 0.069 |
| 8 aero, no body style | 0.488 | 0.281 |

Three things follow:

- **`body_style` is a field on `Design` and `IterationRecord`**, not metadata.
  The parameters are deltas from a *different baseline body per style*, so a
  model that cannot tell which baseline it is perturbing loses 0.39 R².
  `BODY_STYLES = ["fastback", "notchback", "estateback"]`.
- **`default_params()` is all zeros** — the real baseline DrivAer body. Not the
  midpoint of the bounds; the bounds are asymmetric, so the midpoint is a
  different and meaningless car. Optimisation runs start at zero.
- **Units**: degrees for angles, millimetres for lengths, dimensionless for
  curvatures. All offsets from the baseline.

---

## 2. The gate is no longer a sigma threshold

Use **`contracts.is_in_distribution(uncertainty, ood_score, body_style)`**. It
returns `(ok, reason)`. Put the reason in `IterationRecord.reject_reason` and in
the agent's prompt — an agent told *why* it was rejected adapts; one told only
"no" hammers.

The old gate was `sigma < UNCERTAINTY_THRESHOLD`. It was measured and it does
not work:

| | cd R² | mean sigma | 1σ coverage |
|---|---|---|---|
| in distribution | 0.875 | 0.0072 | 0.64 |
| body style held out | 0.479 | 0.0086 | 0.49 |

Error roughly doubles; sigma widens about 20%. A sigma cutoff waves the design
straight through. The reason is structural, not tuning: during training every
example is familiar by definition, so the Gaussian NLL never sees an
unfamiliar one and the variance head has no way to represent novelty. Sigma
answers *"how noisy are the labels here"*, which is a different question from
*"have I seen anything like this"*.

So checkpoints now carry an **OOD detector** (`src/train/ood.py`): Mahalanobis
distance from the training set in standardised parameter space, with the
threshold calibrated so 99% of held-out real designs pass. Measured:

| Shown to the detector | Flagged |
|---|---|
| ordinary designs, body style seen in training | 0.3 – 1.7% |
| **in-bounds but jointly extreme parameter corners** | **100%** |
| designs 2× outside `PARAM_BOUNDS` | 100% |
| held-out body style, full 19-dim distance | 100% |
| held-out body style, aero 16 dims only | 2.6% |

Read the last two rows together — this is the honest part. An unseen body style
has entirely **ordinary** aero parameters (mean distance 4.04 against a 4.62
threshold). The 100% above it comes from the body-style indicator columns,
which `validate_body_style` already knows more cheaply.

The detector's unique contribution is row 2: **designs where every coordinate
is legal but the combination is nowhere near the training data.**
`clamp_params` accepts those — every value is in bounds — and nothing else in
the system rejects them. Sixteen parameters each at a legal extreme is a car
unlike anything ever simulated, and an optimiser pushed toward low drag will
find exactly those corners. That is the hole this closes.

---

## 3. Loading a checkpoint

```python
import torch
from src.train.model import build_model, split_prediction
from src.train.ood import OODDetector

ckpt  = torch.load(path, map_location=device, weights_only=False)
model = build_model(ckpt["config"])          # right architecture either kind
model.load_state_dict(ckpt["state_dict"])
model.eval()

norm = ckpt["norm_stats"]
spec = ckpt["preprocess"]
ood  = OODDetector.from_dict(ckpt["ood"]) if ckpt.get("ood") else None
```

| Key | What |
|---|---|
| `state_dict`, `config`, `norm_stats`, `version`, `val_metrics` | the five originally promised |
| `preprocess` | **how to build model input — read it, don't hardcode** |
| `ood` | the detector, or `None` |
| `test_metrics`, `held_out_category`, `schema_version`, `n_params` | provenance |

### Parameter checkpoints (`preprocess["version"] == "params-v1"`)

```python
cols = spec["param_columns"]        # ORDER IS PART OF THE CONTRACT
x = np.array([[design.params[c] for c in cols if not c.startswith("cat_")]
              + [1.0 if c == f"cat_{design.body_style}" else 0.0
                 for c in cols if c.startswith("cat_")]], dtype=np.float32)
x = (x - np.array(norm["param_mean"])) / np.array(norm["param_std"])
```

Simpler and less error-prone: build the row by walking `cols` in order and
filling each name from `params`, or `1.0/0.0` for a `cat_*` column matching
`body_style`. Column order is what the checkpoint promises.

### Point-cloud checkpoints (`preprocess["version"] == "pointcloud-v1"`)

Import `src.data.dataset.sample_mesh` rather than reimplementing it. If you
must inline it, all four details matter: area-weighted surface sampling,
`n_points` from the spec, **bounding-box** centre (not centroid), max **vertex**
radius for the scale.

### Predicting

```python
with torch.no_grad():
    mu, sigma = split_prediction(model(torch.from_numpy(x).to(device)))

cd    = float(mu[0])    * norm["target_std"] + norm["target_mean"]
sigma = float(sigma[0]) * norm["target_std"]       # a scale: std only, no mean
score = ood.score_one(x) if ood else None

ok, reason = is_in_distribution(sigma, score, design.body_style)
```

`split_prediction` is the only place log-variance becomes a standard deviation.
Use it, so the conversion cannot drift between training and serving.

---

## 4. What stage 1 still cannot do

**`pressure_summary` cannot be filled.** Stage 1 predicts one scalar. The eight
`PRESSURE_REGIONS` keys arrive with surface fields in stage 2. Until then the
service should return the stub's values, and the demo must not imply otherwise.

**`cl` is predicted but not validated.** The data has `Average Cl`; we train on
`Average Cd`. Treat lift as unmeasured for now.

---

## 5. One thing that makes Paramveer's job smaller

With the **parametric** model serving `/evaluate`, `params_to_mesh` is needed
only for the **viewer** — not for predictions. So the morph can be approximate:
it has to look right on screen, not be numerically faithful to ANSA. The
surrogate never sees it.

That also removes the train/serve geometry mismatch entirely: the agent
proposes parameters, the model consumes parameters, and no mesh round trip
happens in the prediction path.

If we later serve the point-cloud model instead, the mismatch comes back and
the OOD score is what keeps it honest.

---

## 6. Numbers, and what to compare them against

Real DrivAerNet, 4,165 designs, 19 inputs:

| Split | R² | MAE | 1σ coverage | OOD flagged |
|---|---|---|---|---|
| random | **0.875** | 0.0062 | 0.64 | 0.3% |
| notchback held out | 0.479 | 0.0102 | 0.49 | 100% |

Per category on the random split: fastback 0.773, notchback 0.862,
estateback 0.827.

**Report R² against 0.906, not 1.0.** The dataset's `Std Cd` column is the
solver's own run-to-run spread (mean 0.0064), which caps any model at

    R²_max = 1 - mean(Std Cd²) / var(Average Cd) = 0.906

So 0.875 is **96% of achievable**, and our predicted sigma (0.0072) sits on the
solver's own noise — the model has correctly concluded most of its residual is
label noise, not model error. A model predicting sigma well below 0.0064 would
be overconfident by construction.

---

## 7. Splits

```bash
python -m src.data.dataset --dump-splits runs/splits.json --config configs/drivaernet.yaml
```

Writes the random split, the held-out-category split, and the `preprocess` spec
in one file. No metrics suite lives in `src/train/` — `evaluate()` there
computes only what early stopping needs plus a coverage diagnostic.
Per-category R², reliability diagrams, OOD detection rate and the speedup
number are the harness's job, run against these checkpoints.
`src.train.ood.detection_rate(detector, x_ood)` is the function for that last
one.
