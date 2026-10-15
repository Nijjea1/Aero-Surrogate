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
| 3D car | `POST /api/mesh` → `src/geometry/viewer.py`, with a per-vertex `part` (paint, glass, tyre, lights, trim) for the studio materials. Drag to orbit, scroll to zoom |
| Values on the car | One pin per pressure region, plus the drag force. Hover or focus a pin for what the number means, what that region does for drag, and where the number comes from: the model, the stub, or illustrative from the shape |
| (i) buttons | A definition of every stat (Cd, Cl, σ, Cp, the gate, drag force, power, energy) from one dictionary in `common.js` |
| Airflow, road loads and the race | The prediction plus the car's outline and frontal area from `/api/mesh`. See [Scenes](#scenes) |

## Design

Dark by default, light behind the toggle in the top bar (remembered per
browser). The 3D stage stays dark in both, because the smoke is light on
dark. Colours are role tokens in `styles.css`. The pressure scale (blue
suction, red pressure) and the chart series are kept for data and never
used for chrome; the accent is cyan.

The dials are grouped by part of the car. Each one fills from the baseline
(0) to its value, in either direction, because every dial is a delta from the
unmodified car. Type an exact value in its box, or reset it with ↺.

## Paint

- **Studio:** real materials. Clear-coated metallic paint in five colours,
  tinted glass, tyres, spoked rims, and lit head and tail lights.
- **Pressure:** an *illustrative* field worked out from the surface shape
  (stagnation at the nose, suction over the roof and edges). It is labelled
  as not a prediction. Stage 2's predicted surface pressure replaces it.
- **Regions:** each surface region is painted with the mean Cp from
  `pressure_summary`. This is the model's actual output, mapped onto the
  surface it describes.

The sliders reshape the car: the service and the geometry share the
contracts 2.0.0 parameters. If the service ever sends names the geometry
doesn't know, the car keeps its baseline shape and the page says so under
the viewer.

## Scenes

- **Road** (the opening view): the car on a road at dusk, with trees and
  posts streaming past and glowing smoke over and around the body. The camera
  sweeps from the front three-quarter to the rear three-quarter. Dragging
  takes it over, and the sweep resumes a few seconds after you let go.
- **Wind tunnel:** a rake of smoke nozzles 13 high and five lanes wide.
- **Studio:** the car on a turntable, still, for a close look.
- **Drag race:** see [Drag race](#drag-race).

The smoke is an illustration driven by the prediction, not a flow
simulation, and the caption says so. Every moving part comes from a
predicted number:

| What you see | Set by |
|---|---|
| The path of each streamline | The car's real outline (`flow` in `/api/mesh`). The air between the body and a streamline keeps its cross-section, so a far-field offset ρ passes at r = √(1 + ρ²) body-radii. |
| Smoke colour | Predicted mean Cp of the region the air is passing, on the same scale as the paint |
| Smoke speed | That Cp, by Bernoulli: \|v\|/U = √(1 − Cp) |
| Wake size, churn, length and haze | Predicted Cd against the selected style's baseline |
| Wake bending down or up | Predicted Cl, when the model predicts it |
| Trembling | Predicted σ against the gate threshold |
| Grey smoke and a red flag | The gate rejected the prediction |

The read-out over the view (`web/physics.js`) turns the prediction into loads
at the chosen speed, at sea-level air density. Drag force is ½ρv²·Cd·A, with
± from σ. It also shows the power to push through the air, the energy per
100 km from drag alone, and lift, which reads "not predicted" while the model
fills Cl with a placeholder. Changes are compared against the baseline body
of the same style, with its own frontal area.

## Drag race

The design against the original car of the same style, from a standing start
over 1 km (2 km and 5 km on the finish card), on a lit two-lane track with
start lights and a finish gantry. Both cars get the same 150 kW, 1,600 kg,
rolling resistance 0.010 and grip μ 1.0. Only drag differs: each car's
predicted Cd × its own frontal area. `standingStart` in `physics.js`
integrates

    m dv/dt = min(P / v, μ m g) − ½ ρ CdA v² − Crr m g

and the finish card reports the margin, the time and trap-speed difference,
and the energy per 100 km at 130 km/h, with the assumptions printed under it.

It races drag force, not Cd. A design with a lower Cd but a larger frontal
area can lose, and the page says so rather than hiding it.
`tests/test_web_physics.py` runs the physics in Node.

## Performance

The mesh endpoint caches recent designs and gzips them (about 200 KB). The
3D view renders only while something moves, and pauses when it is scrolled
out of sight. If frames keep costing more than 24 ms of work, it drops bloom,
then resolution. With `prefers-reduced-motion`, the airflow starts paused.
`?debug=1` exposes the page state as `aero` in the console:
`aero.viewer.perf.workMs` is the work per frame, `aero.flow.stats.frameMs`
the smoke's share of it, and `aero.race.advance(seconds)` steps a race.

## Presets

- **Baseline:** the unmodified body of the selected style.
- **Random:** a uniform draw from the design box.
- **Edge of the box:** every dial near a limit at once. Each value is legal,
  but the combination is unlike any real design, so the gate should reject it.
  On contracts 2.0.0 the distance check flags every such corner design
  (`ood_detection` in `metrics.json`); the stub rejects it because its
  uncertainty rises toward the edges.

Moving sliders updates one row of the session history rather than adding a
row per pause. A preset, a body-style change or a reloaded row starts a new
row. Deltas are against the baseline design of the selected body style.

## Results

**http://localhost:5173/results.html** is a static page. It reads
`web/results/metrics.json` (from the eval harness) and
`web/results/benchmark.json` (from `src.eval.benchmark`), and needs no
service. It shows:

- the headline numbers;
- R² against the noise ceiling for unseen designs and for an unseen body
  style, beside the predict-the-average and straight-line references;
- the reliability diagram;
- how well each check catches the unseen body style;
- the data-scaling curve;
- the agent against CMA-ES and random search, as the median and middle-half
  band over seeds.

A harness `--label` shows as a banner. The committed files are a provisional
local check until the final checkpoints are graded. See
[src/eval/README.md](../src/eval/README.md) for both commands.

## Runs

**http://localhost:5173/runs.html** compares optimisation runs. It shows the
design agent (`src/agent/designer.py`) against the CMA-ES and random-search
baselines (`src/optim/baseline.py`). It reads the service's run log through
`GET /runs`, `GET /runs/{id}` and `GET /history/{id}`, so any run the service
can see shows up: CLI runs and runs started through the API.

- **Best trusted drag so far:** one line per selected run, with up to four
  runs in a fixed colour order. A line only steps down when the gate trusts
  the lower Cd. Every evaluation of the run in focus is a dot. Gated designs
  appear as × in a lane above the plot, because their Cd never counts.
- **Comparison table:** for each run, its best Cd, the change against the
  unmodified car, which evaluation found it, and how many designs the gate
  rejected.
- **Logbook:** every step of the run in focus, with its Cd, the gate's verdict
  and the agent's reasoning or the gate's feedback. **Open** loads that step
  in the designer (`index.html?run=<id>&step=<n>`).
- **Live:** an unfinished run refreshes every 2 seconds until it ends.

By default the page selects the newest agent run, plus the newest CMA-ES and
random runs with the same body style and budget. A warning appears when the
selected runs differ in style or budget, because then they don't compare.
