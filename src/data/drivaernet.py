"""Fetch and prepare the real DrivAerNet parametric table.  OWNER: Avneet.

    python -m src.data.drivaernet --fetch

One 1.4 MB file in the DrivAerNet repo -- no Globus, no Dropbox, no 39 TB:

    ParametricModels/DrivAerNet_ParametricData.csv

It carries 4,165 real designs with 23 morph parameters and CFD-measured
Average Cd, Average Cl and their standard deviations. That is enough for a
real held-out R^2 on real labels without downloading a single mesh.

Two things this module adds to the raw file:

CATEGORY. There is no category column, but the body style is the first token
of the design id -- F_S_WWC_WM_001 is a fastback, N_ a notchback, E_ an
estateback, ~1390 of each. Deriving it is what makes the held-out-category
split a genuine extrapolation test on real data.

SOLVER SPREAD. Std Cd describes coefficient variability in the CFD output.
It is not automatically the error variance of the averaged Cd target. The
historical ratio 1 - mean(Std Cd^2) / var(Average Cd) is descriptive only:
it is not a validated R2 ceiling, an irreducible-error proof, or a lower bound
on the surrogate's uncertainty. Use held-out residual coverage to assess
prediction intervals.

LICENCE: CC BY-NC 4.0, non-commercial. Must be stated in the README and the
submission. data/ is gitignored, so nothing downloaded here is committed.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/Mohamedelrefaie/DrivAerNet/main"
PARAMETRIC_CSV = f"{RAW_BASE}/ParametricModels/DrivAerNet_ParametricData.csv"
SPLIT_FILES = {
    "train": f"{RAW_BASE}/train_val_test_splits/train_design_ids.txt",
    "val": f"{RAW_BASE}/train_val_test_splits/val_design_ids.txt",
    "test": f"{RAW_BASE}/train_val_test_splits/test_design_ids.txt",
}

# Body style from the first token of the design id.
CATEGORY_PREFIX = {"F": "fastback", "N": "notchback", "E": "estateback"}

# The 23 morph parameters, in the order the CSV lists them. These are DELTAS
# from the baseline DrivAer body -- millimetres for lengths, degrees for
# angles, dimensionless for curvatures -- not absolute dimensions. That is why
# they do not line up with contracts.PARAM_BOUNDS; see the note in
# docs/HANDOFF.md.
PARAM_COLUMNS = [
    "B_Ramp_Angle",
    "B_Diffusor_Angle",
    "B_Trunklid_Angle",
    "C_Side_Mirrors_Rotation",
    "D_Rear_Window_Inclination",
    "D_Winscreen_Inclination",
    "C_Side_Mirrors_Translate_X",
    "C_Side_Mirrors_Translate_Z",
    "D_Winscreen_Length",
    "D_Rear_Window_Length",
    "E_A_B_C_Pillar_Thickness",
    "G_Trunklid_Curvature",
    "G_Trunklid_Length",
    "H_Front_Bumper_Curvature",
    "H_Front_Bumper_Length",
    "F_Door_Handles_Thickness",
    "F_Door_Handles_Z_Position",
    "E_Fenders_Arch_Offset",
    "A_Car_Length",
    "F_Door_Handles_X_Position",
    "A_Car_Width",
    "A_Car_Roof_Height",
    "A_Car_Green_House_Angle",
]

# Body style as three indicator columns, appended by fetch(). Worth 0.27 R^2;
# see the comment there.
CATEGORY_COLUMNS = ["cat_fastback", "cat_notchback", "cat_estateback"]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached   {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  fetching {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f" {dest.stat().st_size:,} bytes")
    return dest


def noise_ceiling(df) -> dict:
    """Historical solver-spread ratio; not an attainable-performance bound."""
    var = float(df["Average Cd"].var(ddof=0))
    noise = float((df["Std Cd"] ** 2).mean())
    return {
        "var_cd": var,
        "mean_cfd_variance": noise,
        "solver_spread_ratio_reference": 1.0 - noise / var,
        "is_performance_bound": False,
        "mean_std_cd": float(df["Std Cd"].mean()),
    }


def fetch(out_dir: Path) -> dict:
    """Download the parametric table and the official splits; add Category."""
    import pandas as pd

    raw = _download(PARAMETRIC_CSV, out_dir / "raw" / "DrivAerNet_ParametricData.csv")
    splits = {
        name: _download(url, out_dir / "raw" / Path(url).name).read_text().split()
        for name, url in SPLIT_FILES.items()
    }

    df = pd.read_csv(raw)
    df["Category"] = df["Experiment"].str.split("_").str[0].map(CATEGORY_PREFIX)

    # One-hot the body style as a MODEL INPUT. Measured: the 23 parameters
    # alone give test R^2 0.60, adding body style gives 0.87 -- a 0.27 jump.
    # The reason is that the parameters are deltas from a DIFFERENT baseline
    # body per category, so "A_Car_Length +50mm" describes a different shape
    # on a fastback than on an estateback, and a model that cannot tell which
    # baseline it is perturbing cannot resolve the two.
    #
    # This is legitimate, not leakage: a designer always knows which body
    # style they are working on, and so does the geometry module. But report
    # BOTH numbers -- the gap is itself the cross-category result.
    for prefix, name in CATEGORY_PREFIX.items():
        df[f"cat_{name}"] = (df["Category"] == name).astype(float)

    unknown = int(df["Category"].isna().sum())
    if unknown:
        print(f"  WARNING  {unknown} ids with an unrecognised prefix; dropping them")
        df = df.dropna(subset=["Category"])

    prepared = out_dir / "parametric.csv"
    df.to_csv(prepared, index=False)

    ceiling = noise_ceiling(df)
    return {
        "csv": prepared,
        "n": len(df),
        "categories": df["Category"].value_counts().to_dict(),
        "official_splits": {k: len(v) for k, v in splits.items()},
        **ceiling,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download and prepare")
    ap.add_argument("--out", default="data/drivaernet")
    args = ap.parse_args()

    if not args.fetch:
        ap.print_help()
        return

    print("DrivAerNet parametric table  (CC BY-NC 4.0, non-commercial)")
    info = fetch(Path(args.out))
    print(f"\nwrote {info['csv']}")
    print(f"  designs      {info['n']:,}")
    print(f"  categories   {info['categories']}")
    print(f"  official     {info['official_splits']}")
    print(f"\n  mean Std Cd  {info['mean_std_cd']:.5f}   (solver coefficient spread; not averaged-label error)")
    print(f"  spread ratio {info['solver_spread_ratio_reference']:.4f}   (not an R2 ceiling)")
    print(f"\nnext:  python -m src.data.dataset --check --config configs/drivaernet.yaml")


if __name__ == "__main__":
    main()
