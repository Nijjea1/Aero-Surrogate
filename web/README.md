# Web UI

Owner: Paramveer. Plain HTML, CSS and JavaScript modules, with no build step.

The page lets you pick a design, see the car in 3D, see its predicted drag
and lift, and see whether the gate trusts the prediction.

## Run it

You need two terminals, both started from the repo root.

```powershell
# 1. the service (Pranoy's), on :8000
.venv\Scripts\python -m uvicorn src.service.app:app --port 8000

# 2. this page and the 3D mesh endpoint, on :5173
.venv\Scripts\python web/serve.py
```

Open **http://localhost:5173**. Use `localhost`, not `127.0.0.1`. The
service's CORS settings only allow `http://localhost:5173` and
`http://localhost:3000`.

`web/serve.py` serves the page plus `POST /api/mesh`, which builds the car
with `src/geometry`. It answers only on this machine. To show the page to
another device on purpose, use `--host 0.0.0.0`. Plain `python -m http.server` still serves the page, but
without the 3D view. three.js loads from jsDelivr (pinned to 0.160.0), so the
3D view needs a network connection. The rest of the page works without one.

To point the page at a different service, add `?api=` to the URL:
`http://localhost:5173/?api=http://my-host:8000`

## What it shows

| Piece | Comes from |
|---|---|
| Sliders, one per design parameter | `GET /schema` bounds and defaults, so they follow the contract version the service runs |
| Body style picker | `body_styles` in `/schema`, shown only when the service sends it (contracts 2.0.0) |
| Cd, Cl, σ tiles, with deltas against the baseline design | `POST /evaluate` |
| Trusted / Rejected badge | The same rules, in the same order, as `src/service/gate.py`: body style, σ against `uncertainty_threshold`, the distance score against `ood_score_threshold`, then the model's own distribution flag |
| Pressure by region | `pressure_summary`. A region the model doesn't predict shows as n/a and is never filled in |
| Real car picker and the CFD check | `web/designs.json`: real DrivAerNet cars from the test split, which the model never trained on, with their CFD Cd. Chosen by CFD drag (low, typical, high per style), not by model error. The check disappears as soon as a dial moves, because the design is then no longer the car CFD measured. Rebuild with `python -m src.eval.showcase` (add `--splits runs/splits.json` to match the served checkpoint's split). DrivAerNet is CC BY-NC 4.0 |
| Synthetic banner | `GET /health` `synthetic: true`. The stub's numbers are never presented as results |
| 3D car | `POST /api/mesh` → `src/geometry/viewer.py`. Drag to orbit, scroll to zoom, hover for region and Cp |
| Airflow and road loads | The prediction plus the car's outline and frontal area from `/api/mesh`. See [Airflow](#airflow) |

## 3D colour modes

- **Pressure:** an *illustrative* field worked out from the surface shape
  (stagnation at the nose, suction over the roof and edges). It is labelled
  as not a prediction. Stage 2's predicted surface pressure replaces it.
- **Regions:** each surface region is painted with the mean Cp from
  `pressure_summary`. This is the model's actual output, mapped onto the
  surface it describes.
- **Plain:** the body and tyres, uncoloured.

The sliders reshape the car: the service and the geometry share the
contracts 2.0.0 parameters. If the service ever sends names the geometry
doesn't know, the car keeps its baseline shape and the page says so under
the viewer.

## Airflow

The page opens on **Driving**: the car on a road running to the horizon,
with trees and posts streaming past and air streaking over and around the
body. The camera sweeps slowly from the front three-quarter to the rear
three-quarter. Dragging takes it over, and the sweep resumes a few seconds
after you let go. **Wind tunnel** streams smoke from a rake of nozzles five
lanes wide, so it wraps over the roof and round both sides. **Still** stops
the animation. **Full screen** fills the screen with the view and its
read-out, for recording the demo.

It's an illustration driven by the prediction, not a flow simulation, and the
caption says so. Every moving part comes from a predicted number:

| What you see | Set by |
|---|---|
| The path of each streamline | The car's real outline (`flow` in `/api/mesh`). The air between the body and a streamline keeps its cross-section, so a far-field offset ρ passes at r = √(1 + ρ²) body-radii. |
| Smoke colour | Predicted mean Cp of the region the air is passing, on the same scale as the paint |
| Smoke speed | That Cp, by Bernoulli: \|v\|/U = √(1 − Cp) |
| Wake size, churn and length | Predicted Cd against the selected style's baseline |
| Wake bending down or up | Predicted Cl. Lift pushes the air down behind the car. |
| Trembling | Predicted σ against the gate threshold |
| Grey smoke and a red flag | The gate rejected the prediction |

The read-out over the view (`web/physics.js`) turns the prediction into loads
at the chosen speed, at sea-level air density. Drag force is ½ρv²·Cd·A, with
± from σ. It also shows the power to push through the air, the energy per
100 km from drag alone, and lift, which is marked unvalidated until Cl is
checked against CFD. Changes are compared against the baseline body of the
same style, with its own frontal area.

With `prefers-reduced-motion`, the airflow starts paused. `?debug=1` exposes
the page state as `aero` in the console. For example,
`aero.flow.stats.frameMs` is the animation's work per frame, about 1–2 ms.

## Presets

- **Baseline:** the service's default design.
- **Random:** a uniform draw from the design box.
- **Edge of the box:** every dial near a limit at once. Each value is legal,
  but the combination is unlike any real design, so the gate should reject it.
  On contracts 2.0.0 the distance check flags every such corner design
  (`ood_detection` in `metrics.json`); the stub rejects it because its
  uncertainty rises toward the edges.

Moving sliders updates one row of the session history rather than adding a
row per pause. A preset, a body-style change or a reloaded row starts a new
row. Deltas are against the baseline design of the selected body style.
