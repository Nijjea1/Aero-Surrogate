#!/usr/bin/env bash
# One-command smoke test of the whole pipeline. Run from the repo root.
# Each line starts working as its owner fills in their module.
set -euo pipefail

echo "==> contract tests"
python -m pytest tests/ -q

echo "==> geometry module (Paramveer, Oct 14)"
python -c "from src.geometry.params import params_to_mesh; from src.contracts import default_params; m = params_to_mesh(default_params()); print(f'faces={len(m.faces)} watertight={m.is_watertight}')"

echo "==> starting service on :8000 (Pranoy, Oct 13)"
uvicorn src.service.app:app --port 8000 --log-level warning &
SERVICE_PID=$!
trap 'kill $SERVICE_PID 2>/dev/null || true' EXIT
sleep 5
curl -sf http://localhost:8000/health && echo

echo "==> agent loop, offline (Pranoy, Oct 17)"
python -m src.agent.designer --dry-run --iterations 10 --run-id smoke-agent

echo "==> baseline optimiser (Paramveer, Oct 22)"
python -m src.optim.baseline --method cmaes --iterations 10 --run-id smoke-cmaes

echo
echo "All green. Run logs in runs/logs/"
