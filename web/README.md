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
| Synthetic banner | `GET /health` `synthetic: true`. The stub's numbers are never presented as results |
| 3D car | `POST /api/mesh` → `src/geometry/viewer.py`. Drag to orbit, scroll to zoom, hover for region and Cp |

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
