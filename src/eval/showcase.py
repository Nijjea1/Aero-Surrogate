"""Real DrivAerNet cars for the page, with their CFD answers.  OWNER: Paramveer.

    python -m src.eval.showcase                    # writes web/designs.json
    python -m src.eval.showcase --per-style 5 --splits runs/splits.json

Lets the demo put a design the model never trained on next to its CFD ground
truth. Every car comes from the TEST set of the random split, which is the
split the served checkpoint was trained with (runs/splits.json when it exists,
otherwise rebuilt with the trainer's algorithm and seed).

The cars are chosen by their CFD drag, never by how well the model does on
them: per body style, the designs nearest evenly spaced percentiles of CFD Cd,
from the low-drag end to the high-drag end. Picking by model error would turn
a demo into an advert.

DrivAerNet is CC BY-NC 4.0 (non-commercial). The file carries the attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..contracts import BODY_STYLES, PARAM_BOUNDS, PARAM_NAMES
from . import data as D

ATTRIBUTION = ("DrivAerNet, Elrefaie et al., CFD by the dataset authors. "
               "CC BY-NC 4.0, non-commercial use only.")
LABEL_CL = "Average Cl"


def percentiles(per_style: int) -> list[float]:
    """Evenly spaced inside (0, 100), so the ends are typical rather than freak cars."""
    return [100 * (i + 0.5) / per_style for i in range(per_style)]


def pick(table: D.Table, test_ids: list[str], per_style: int = 3) -> list[dict]:
    """The showcase cars: per body style, the test design nearest each percentile of CFD Cd."""
    if per_style < 1:
        raise ValueError("per_style must be at least 1")
    test_ids = sorted(test_ids)
    cats = table.category(test_ids)
    cd = table.column(D.TARGET, test_ids)
    std = table.column(D.LABEL_STD, test_ids) if D.LABEL_STD in table.columns else None
    cl = table.column(LABEL_CL, test_ids) if LABEL_CL in table.columns else None
    params = {name: table.column(name, test_ids) for name in PARAM_NAMES}
    inside = np.ones(len(test_ids), dtype=bool)
    for name, (lo, hi) in PARAM_BOUNDS.items():
        inside &= (params[name] >= lo) & (params[name] <= hi)

    cars = []
    for style in BODY_STYLES:
        pool = np.flatnonzero((cats == style) & inside)
        if len(pool) == 0:
            continue
        chosen: list[int] = []
        for q in percentiles(per_style):
            target = np.percentile(cd[pool], q)
            order = pool[np.argsort(np.abs(cd[pool] - target), kind="stable")]
            i = next((int(j) for j in order if j not in chosen), None)
            if i is None:
                break
            chosen.append(i)
            cars.append({
                "design_id": test_ids[i],
                "body_style": style,
                "cd_percentile": round(q),
                "params": {name: round(float(params[name][i]), 4) for name in PARAM_NAMES},
                "cd_cfd": float(cd[i]),
                "cd_cfd_std": None if std is None else float(std[i]),
                "cl_cfd": None if cl is None else float(cl[i]),
            })
    return cars


def build(data_path: str | None = None, splits_path: str | None = None, seed: int = 0,
          per_style: int = 3) -> dict:
    table = D.load_table(D.find_data(data_path))
    if splits_path:
        splits = D.load_splits(splits_path)
    elif Path("runs/splits.json").exists():
        splits = D.load_splits("runs/splits.json")
    else:
        splits = D.build_splits(table, seed=seed)
    split = splits["random"]
    D.check_disjoint(split)
    return {
        "source": ATTRIBUTION,
        "split": (f"Test designs of the random split ({splits['source']}): "
                  "the model never trained or early-stopped on them."),
        "selection": (f"Per body style, the {per_style} test designs nearest evenly spaced "
                      "percentiles of CFD Cd. Chosen by drag, not by model error."),
        "cd_cfd_std": "The CFD solver's own run-to-run spread on Cd, from the dataset.",
        "designs": pick(table, split["test"], per_style),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the real test-set cars the page compares against CFD.")
    ap.add_argument("--data", help="DrivAerNet parametric CSV (default: data/drivaernet/...)")
    ap.add_argument("--splits", help="runs/splits.json (default: it, if present, else rebuilt)")
    ap.add_argument("--seed", type=int, default=0, help="split seed when rebuilding")
    ap.add_argument("--per-style", type=int, default=3)
    ap.add_argument("--out", default="web/designs.json")
    args = ap.parse_args(argv)
    payload = build(args.data, args.splits, args.seed, args.per_style)
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['designs'])} cars to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
