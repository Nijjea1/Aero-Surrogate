# Web UI

Owner: Paramveer. Plain HTML, CSS and JavaScript modules, with no build step.

The UI shell lets you pick a design, see its predicted drag and lift, and see
whether the uncertainty gate trusts the prediction.

## Run it

You need two terminals, both started from the repo root.

```powershell
# 1. the service (Pranoy's), on :8000
.venv\Scripts\python -m uvicorn src.service.app:app --port 8000

# 2. this page, on :5173
.venv\Scripts\python -m http.server 5173 -d web
```

Open **http://localhost:5173**. Use `localhost`, not `127.0.0.1`. The
service's CORS settings only allow `http://localhost:5173` and
`http://localhost:3000`.

To point the page at a different service, add `?api=` to the URL:
`http://localhost:5173/?api=http://my-host:8000`

## What it shows

| Piece | Comes from |
|---|---|
| Sliders, one per design parameter | `GET /schema` bounds and defaults, so they follow the contract version the service runs |
| Body style picker | `body_styles` in `/schema`, shown only when the service sends it (contracts 2.0.0) |
| Cd, Cl, σ tiles, with deltas against the baseline design | `POST /evaluate` |
| Trusted / Rejected badge | The same rules as `src/service/gate.py`: the distribution flag, then σ against `uncertainty_threshold` |
| Pressure by region | `pressure_summary`. A region the model doesn't predict shows as n/a and is never filled in |
| Synthetic banner | `GET /health` `synthetic: true`. The stub's numbers are never presented as results |

## Presets

- **Baseline:** the service's default design.
- **Random:** a uniform draw from the design box.
- **Edge of the box:** every dial near its limit. The stub's uncertainty
  rises toward the edges, so this shows the gate rejecting a design.
