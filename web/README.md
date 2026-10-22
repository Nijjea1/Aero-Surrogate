# Web UI

Owner: Paramveer. Plain HTML, CSS and JavaScript modules, with no build step.

The page lets you pick a design, see the car in 3D, see its predicted drag
and a domain verdict. Lift is unavailable in the current scalar model.

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
without the 3D view. three.js is pinned to 0.160.0 and vendored under `web/vendor/three`, with its licence and asset manifest. The 3D view does not need a CDN connection. IBM Plex font files are served from `web/vendor/fonts` with their OFL notices; system fonts are the fallback.

To point the page at a different service, add `?api=` to the URL:
`http://localhost:5173/?api=http://my-host:8000`

## What it shows

| Piece | Comes from |
|---|---|
| Sliders, one per design parameter | `GET /schema` bounds and defaults, so they follow the contract version the service runs |
| Body style picker | `body_styles` in `/schema`, shown only when the service sends it (contracts 2.0.0) |
| Cd, Cl, σ tiles, with deltas against the baseline design | `POST /evaluate` |
| Trusted / Rejected badge | The same rules, in the same order, as `src/service/gate.py`: body style, σ against `uncertainty_threshold`, the distance score against `ood_score_threshold`, then the model's own distribution flag |
| Pressure by region | The scalar model's `pressure_summary` when `/health` `output_provenance.pressure_summary` is `model`; otherwise the region means from `POST /fields`, labelled *predicted*; otherwise the stub's. A region nothing predicts shows as n/a and is never filled in |
| Surface pressure | `POST /fields` when `/health` `fields.available` is true: per-face Cp and its σ on the same mesh the viewer draws. See [Predicted surface pressure](#predicted-surface-pressure) |
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
- **Pressure:** the field model's predicted Cp, face by face, when the
  service serves `/fields`. Without it, an *illustrative* field worked out
  from the surface shape (stagnation at the nose, suction over the roof and
  edges), labelled as not a prediction.
- **σ** (only with `/fields`): the field model's Cp standard deviation, on
  one amber ramp from dark (sure) to light (unsure), scaled to the 95th
  percentile. It is not the Cd σ the gate uses.
- **Regions:** each surface region is painted with its mean Cp, from the
  source the pressure chart uses (above).

### Predicted surface pressure

The service predicts every face of the exact `mesh_payload` this page draws
(`src/service/INTEGRATION.md`, "Surface fields and viewer binding").
`fieldmap.js` binds a response only when both are true:

- the response and the car on screen are the same design;
- the response's mesh hash and vertex/face counts match the mesh drawn. The
  hash is SHA-256 over `viewer-f32le-u32le-v1\0`, the vertex bytes and the
  face bytes, computed in the browser with Web Crypto, which needs HTTPS or
  localhost.

Item *i* goes to face `face_indices[i]`. A response for an older design is
dropped. A busy service (503) is retried briefly. A mismatch is shown in the
caption instead of the paint.

The viewer colours vertices, so each vertex takes the area-weighted mean of
its faces. That conversion is for display only. Hovering the car reads the
exact face's Cp ± σ, and averaged σ is shading, not propagated uncertainty.

Field predictions are labelled **Predicted**, never Trusted. The gate checks
Cd, and nothing checks the field for distance from the training data. The
generated car also only approximates the CFD cars the field model learned
from.

`tests/test_web_fields.py` runs `fieldmap.js` in Node. It checks that the
browser's hash equals Python's, that the face and area mapping is right, and
that a response for another mesh never binds.

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
- held-out R² and MAE beside predict-the-average and straight-line references;
  solver variation is not interpreted as a hard performance bound;
- the reliability diagram;
- how well each check catches the unseen body style;
- the data-scaling curve;
- the agent against CMA-ES and random search, as the median and middle-half
  band over seeds.

A harness `--label` shows as a banner. The committed files identify the graded checkpoint and experiment provenance. Unavailable category and field experiments are explicit. See
[src/eval/README.md](../src/eval/README.md) for both commands.

## Runs

**http://localhost:5173/runs.html** compares optimisation runs. It shows the
design agent (`src/agent/designer.py`) against the CMA-ES and random-search
baselines (`src/optim/baseline.py`). It reads the service's run log through
`GET /runs`, `GET /runs/{id}` and `GET /history/{id}`, so any run the service
can see shows up: CLI runs and runs started through the API.

- **Start an agent run:** pick Nemotron Super or Nano, the number of
  evaluations (25 or 10 by default; this includes the unmodified car), the
  body style, and optional constraints, one rule per line. This sends
  `POST /runs`. The page only ever starts public runs, because an admin
  token never goes in a web page. Nano is sent with `thinking: "off"`.
  - The new run is selected and plotted against the CMA-ES and random runs
    with the same style and budget.
  - Refusals are shown in words: over a limit (422, with the service's
    reason), busy or today's public budget used (429, "try later"), public
    runs switched off (401), and the agent not configured (503).
- **Cancel:** the 202 response carries a one-time `cancel_token`. The page
  keeps it in memory and in `sessionStorage`, so a reload of that tab can
  still cancel. It sends the token as `X-Cancel-Token` to
  `POST /runs/{id}/cancel`. Other visitors see no cancel button.
- **States:** queued, running, finished, cancelled, failed, interrupted and
  unknown each get a chip, in the list and over the logbook. The stop reason
  is always shown. A failed run says why, and "finished" can mean its spend
  budget ran out. "Unknown" is a run this service process does not own: a
  CLI run still going, or one cut off.

- **Best trusted drag so far:** one line per selected run, with up to four
  runs in a fixed colour order. A line only steps down when the gate trusts
  the lower Cd. Every evaluation of the run in focus is a dot. Gated designs
  appear as × in a lane above the plot, because their Cd never counts.
- **Comparison table:** for each run, its best Cd, the change against the
  unmodified car, which evaluation found it, and how many designs the gate
  rejected.
- **Logbook:** every step of the run in focus, with its Cd, the gate's verdict
  and the agent's reasoning or the gate's feedback.
  - An agent step's reasoning is one line,
    `Hypothesis: … | Change: … | Expected: … | Observed: …`. The page splits
    it into those four parts.
  - Observed is written by the service from the score and the gate, never
    by the model. Its verdict (as expected, opposite to expected, within
    noise) shows as a chip.
  - **Open** loads that step in the designer (`index.html?run=<id>&step=<n>`).
- **Live:** an unfinished run refreshes every 2 seconds until it ends.

By default the page selects the newest agent run, plus the newest CMA-ES and
random runs with the same body style and budget. A warning appears when the
selected runs differ in style or budget, because then they don't compare.
