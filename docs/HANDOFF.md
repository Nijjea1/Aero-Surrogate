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

Agreed policy with Track B, and what `/health.output_provenance` reports:

| Output | Provenance | What the service returns |
|---|---|---|
| `cd` | `model` | the prediction |
| `uncertainty` | `model` | Cd sigma, the model's own heteroscedastic head |
| `ood_score` | `model` | parameter-space Mahalanobis distance |
| `cl` | `placeholder` | **always `0.0`, labelled** |
| `pressure_summary` | `unavailable` | **empty `{}`** |

**`pressure_summary` is empty, not stubbed.** Stage 1 predicts one scalar; the
eight `PRESSURE_REGIONS` keys cannot come out of it. Returning the stub's
invented numbers was the earlier instruction and it was wrong — a plausible
number with no model behind it is worse than a missing one, because nothing
downstream can tell the difference. The service returns `{}` and the UI shows
the regions as unavailable. Stage 2's `POST /fields` is what actually fills
them, from the learned region classifier (98.6% held-out agreement).

**`cl` is a labelled placeholder, not a prediction.** Stage 1 trains on
`Average Cd` only, so its `cl` is the constant `0.0` and is flagged
`placeholder` in provenance. Stage 2 *does* integrate lift from the predicted
pressure field, and it is measured, and it is **not good enough to act on**:

| | model, per-car MAE | trivial baseline |
|---|---|---|
| Cd, historical 4,096-point/car estimate | **0.0326** | 0.0496 (predict the test mean) |
| Cl | 0.0802 | 0.1043 (predict zero) |

In that historical subsample estimate, drag beats its mean baseline by 34%. Lift beats its by 23% **and gets the sign wrong
on one of five held-out cars** (run 12: true +0.1285, predicted −0.0189). A
wrong-signed lift coefficient is worse than no lift coefficient: it points
downforce the wrong way. So no endpoint serves Cl as a prediction, and no
derived lift load or effect is displayed anywhere.

Both stage-2 integrals are also **pressure-only**: the viscous term, roughly 8%
of total Cd, is not predicted by this model at all.

The received 27 October evaluation bundle grades all 100,000 stored query cells
for each of five test cars: pressure-only Cd per-car MAE **0.0218415**, mean
absolute relative error **8.57%**. Runs 3, 11 and 22 under-predict; 12 and 27
over-predict. This supersedes the 0.0326 headline, but does not rewrite historical
checkpoint metadata or establish complete native CFD surface quadrature.
Cp R² is **0.5080**, MAE **0.14554** over 500,000 stored cells. The pooled
nearest-training-point reference is stronger (**R² 0.6905**, **MAE 0.10251**).
It averages eight neighbours from sampled points pooled across twenty training
cars in position/normal space, rather than copying a whole training car.
Fresh integrated-source reports in `web/results/metrics.json` are the release
authority after regeneration; the incoming reports remain immutable provenance.

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

**Report ordinary held-out R-squared, MAE and interval coverage.** `Std Cd`
describes coefficient variation in solver output. It is not established as the
error variance of the averaged `Average Cd` label. The historical solver-spread
ratio is not a validated upper bound on attainable R-squared. Earlier
"96% of achievable" and irreducible-noise interpretations were withdrawn.

The served scalar's independently graded nominal 90% interval covers 86.56%
of held-out labels. Its sigma/error rank correlation is about 0.066, so sigma
alone barely ranks individual errors. Present the distance gate as parameter-domain
screening, not a guarantee of correctness. The received category artifact reproduces
R² 0.4792 / MAE 0.01016 on 1,403 held-out notchbacks; the release report records
the independent regrade. Its 1σ coverage is 48.75%. The full-input detector's
100% style detection relies on explicit style indicators; aerodynamic dials alone
give AUROC about 0.536 and detect about 2.64%. Keep this checkpoint evaluation-only.

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


---

# Stage 2 — surface field checkpoints

**New in this handoff.** A second checkpoint kind, from
`src/train/train_fields.py`, predicting **Cp at every surface point** rather
than one drag number.

```bash
python -m src.data.drivaerml --runs 30 --download
python -m src.data.preprocess --combine
python -m src.data.fields --extract
python -m src.train.train_fields --config configs/fields.yaml
```

Trains in about a minute on an M-series GPU. No Nebius instance — see the
local training setup. No Nebius training execution is claimed.

## Loading one

```python
import torch
from src.train.train_fields import build_model_from_cfg

ckpt  = torch.load(path, map_location=device, weights_only=False)
model = build_model_from_cfg(ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

cp_mean = ckpt["norm_stats"]["cp_mean"]
cp_std  = ckpt["norm_stats"]["cp_std"]
regions = ckpt["regions"]          # the PRESSURE_REGIONS present
splits  = ckpt["splits"]           # which runs trained it, by run number
```

The forward signature differs from stage 1 — it takes geometry **and** query
points:

```python
out = model(cloud, query_xyz, query_normal)   # (B, G, 3), (B, Q, 3), (B, Q, 3)
mean, sigma = model.split(out)                # each (B, Q, 1)
cp    = mean  * cp_std + cp_mean
cp_sd = sigma * cp_std
```

`cloud` and `query_xyz` must be normalised **the same way stage 1 normalises
geometry** — bounding-box centre of the combined mesh, divided by max vertex
radius. `src.data.fields.extract_one` is the reference. Get this wrong and the
model returns a confident wrong field with no error.

## This is what finally fills `pressure_summary`

```python
import numpy as np
summary = {r: float(np.average(cp[region == i], weights=area[region == i]))
           for i, r in enumerate(regions)}
```

Area-weighted mean Cp per contract region. `src/data/preprocess.py --combine`
writes the face→region index this needs, derived from the 49 semantically named
solids in each DrivAerML STL (`BodyHood`, `BodyRoof`, `BodyFasciafront1`, …),
which cover all eight `PRESSURE_REGIONS`.

That matters beyond filling a field: *"pressure is high on the rear fascia"* is
something Nemotron can act on. A single scalar is not.

## Historical training-side snapshot (superseded by §4 stored-cell grading)

20 train / 5 val / 5 test cars — split **by car**, never by point:

| | value |
|---|---|
| Cp R² | 0.5105 |
| Cp MAE | 0.1459 |
| 1σ coverage | 0.80 |
| **per-car drag error** | **0.0328** |
| drag bias | −0.0009 |

**Quote the per-car drag error, not the bias.** The bias is the average signed
error over the five test cars and it looks almost perfect at −0.0009 — partly
because cars cancel each other out. The per-car figure is ten times larger and
is the honest one.

Per car, with drag integrated from the model's own predicted pressure field,
against the **pressure-only** integral of the real CFD field (the model does
not predict the viscous term, which is ~8% of total Cd):

| held-out car | CFD, pressure only | predicted | error |
|---|---|---|---|
| run 3 | 0.2166 | 0.1972 | −0.0194 |
| run 11 | 0.2930 | 0.2286 | −0.0644 |
| run 12 | 0.2443 | 0.2415 | −0.0028 |
| run 22 | 0.2791 | 0.2445 | −0.0347 |
| run 27 | 0.1958 | 0.2347 | +0.0389 |

This table records the historical 4,096-point/car estimate (MAE about 0.032,
roughly 12% per car). The all-stored-cell evaluation in §4 supersedes this
headline; preserve the table as historical evidence of the original sampling.

### Lift: measured, and not usable

`integral_consistency_loss` had a bug (both `integrate_drag` calls omitted
`axis=axis`, so the loop constrained drag twice and lift never). Found in
review by Pranoy, fixed, and lift is now both constrained and measured:

| | model MAE | trivial baseline |
|---|---|---|
| Cd, historical 4,096-point/car estimate | **0.0326** | 0.0496 (predict the mean) |
| Cl | 0.0802 | 0.1043 (predict zero) |

On that historical subsample, drag beats its mean baseline by 34%. Lift beats its baseline by 23%
**and gets the sign wrong on 1 of 5 held-out cars** (run 12: true +0.1285,
predicted −0.0189). **Do not use Cl for decisions.** Stage 1 returning `cl` as
a placeholder `0.0` remains the honest default.

Fixing the bug did not change the historical subsample drag estimate (0.0326 vs 0.0328 before), so no
retraining was required for the drag claims — only a narrower lift claim.

**Still judge the field by its integral, not by Cp R².** A model predicting the
mean Cp everywhere gets a respectable MAE and integrates to **zero** drag,
because the normals cancel — there is a test asserting exactly that.

R² 0.30 is weak and honestly so: 20 training cars, a 647k-parameter model, and
a laptop GPU. The full-strength version is NVIDIA's upstream DoMINO recipe
fine-tuned from the pretrained `nvidia/domino_drivaerml` checkpoint, which is
what a real H100 run should do. This is the scaled-down stand-in.

---

# Four uncertainty signals, measured. Three do not work.

This is the historical training-side comparison. The release independently grades
the served scalar and separate category checkpoint in `web/results/metrics.json`.
The additional feature-distance/ensemble rows are not fresh release experiments.
The +20% sigma comparison below crosses different trained model heads; for the
category checkpoint alone, familiar-validation to held-out sigma rises about 6%.
Neither change compensates for the held-out error increase.

| signal | in distribution | on a held-out body style |
|---|---|---|
| heteroscedastic σ | coverage 0.64 ✓ | error ×2, σ +20% → coverage 0.49 ✗ |
| feature-space distance | FPR 3.3% | best of 8 methods caught 26% ✗ |
| deep-ensemble disagreement | coverage 0.79 ✓ | error ×2, epistemic +25% → 0.44 ✗ |
| **parameter-space distance** | **FPR 0.3%** | **100% caught ✓** |

Plus one external check that also failed: correlation between the field model's
per-point σ and `pPrime2MeanTrim` (the LES's own pressure-fluctuation variance,
i.e. where the flow is *physically* unsteady) is **+0.03** — nothing.

**The pattern.** OOD detection works when the novelty is **explicit in the
input** and fails whenever the model is asked to notice it from its own
behaviour. The parameter-space check succeeds because an unseen body style sits
far outside the training range of the body-style indicator. σ, ensembles and
feature distance all fail because the model's *behaviour* looks perfectly
normal to the model.

**So `contracts.is_in_distribution` is built the right way round**: it checks
body style explicitly and treats σ and `ood_score` as supporting evidence
rather than as the gate. Keep it that way.

For the writeup, this is a stronger result than "our gate works": we built four
uncertainty signals, measured each against a real distribution shift, and three
failed. Almost nobody stress-tests their own confidence claim and publishes the
failures.
