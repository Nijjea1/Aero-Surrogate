# Data directory

Nothing here is committed (see .gitignore). Each person downloads their own.

## Stage 1 (Avneet, by Oct 15)

From https://github.com/Mohamedelrefaie/DrivAerNet — CC BY-NC 4.0, non-commercial.

    data/drag_coefficients.csv    # design_id -> Cd. Small file, linked in the repo.
    data/meshes/<design_id>.stl   # a few HUNDRED, not all 8000.

Then verify before training:

    python -m src.data.dataset --check --config configs/baseline.yaml

## Stage 2 (Avneet, from Oct 16)

DrivAerML, 500 morphed notchback variants. The download script ships inside
PhysicsNeMo's XAeroNet example. Start it in the BACKGROUND on Oct 16 — it takes
hours.

## DO NOT

Download DrivAerNet++ in full. It is 39 TB behind Globus and it will eat the
whole schedule. This is the most likely way this project fails.

## Fallback

AirfRANS (2D airfoils, small) via PhysicsNeMo's GLOBE example. Decision point
is Oct 19 — if preprocessing is still fighting you, take the fallback.
