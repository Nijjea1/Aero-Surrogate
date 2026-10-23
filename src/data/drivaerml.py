"""Bounded DrivAerML fetch for stage 2.  OWNER: Avneet.

    python -m src.data.drivaerml --runs 20              # measure only, downloads nothing
    python -m src.data.drivaerml --runs 20 --download   # fetch, after you have seen the size

WHY THIS EXISTS INSTEAD OF THE UPSTREAM SCRIPT

Two reasons, one of them blocking.

FIRST, THE AWS SCRIPT NO LONGER WORKS. PhysicsNeMo's
xaeronet/download_aws_dataset.sh uses `aws s3 cp --no-sign-request` against
s3://caemldatasets, and that bucket now returns 403 Forbidden for anonymous
access -- ListObjectsV2 is disabled outright and even HeadObject is refused.
Verified 2026-10-14 with aws-cli 2.37. So we use the Hugging Face mirror that
PhysicsNeMo's own domino_nim_finetuning example points at,
huggingface.co/datasets/neashton/drivaerml, over plain HTTPS. No AWS account,
no credentials, no CLI.

SECOND, THE UPSTREAM SCRIPT IS A FOOT-GUN ON A LAPTOP. Run as shipped it does
`for i in $(seq 1 500)` and pulls the volumetric field for every run. Measured
per-run sizes on the mirror:

    drivaer_N.stl          0.133 GB   geometry
    boundary_N.vtp         0.614 GB   surface fields (pressure, wall shear)
    force_mom_N.csv       ~0          the drag and lift labels
    volume_N.vtu.*.part   50+ GB      volumetric field, in two parts

    surface only   0.75 GB per run  ->  375 GB for all 500
    with volume   51   GB per run  ->   25 TB for all 500

Downloading all volumes would exceed the bounded surface-only workflow.
Remembering to edit an upstream bulk-download script is not a reliable limit.

So this module makes the safe thing the default and the dangerous thing
impossible to do by accident:

  * SURFACE ONLY by default. drivaer_N.stl (geometry) + boundary_N.vtp
    (surface fields) and nothing else. Cut-list item 4 in docs/PLAN.md says
    surface-only is sufficient for drag and lift, so the volumetric fields are
    not on our critical path. --include-volume exists but demands --force.
  * NO DEFAULT RUN COUNT. --runs is required. The DoMINO fine-tuning recipe we
    are following was validated with 16 training and 8 test samples, so start
    near 20, not 500.
  * MEASURE BEFORE FETCHING. Without --download it lists the exact byte sizes
    and stops. You decide after seeing a number.
  * DISK GUARD. Refuses if the download would exceed DISK_FRACTION of free
    space, regardless of flags.

WHAT THIS IS NOT: it does not touch DrivAerNet++ (39 TB, Harvard Dataverse,
Globus). Different dataset, different host. We never download that one, in part
or in full. DrivAerML lives in a public S3 bucket and needs no AWS account.

LICENCE: DrivAerML is CC BY-NC 4.0, non-commercial. data/ is gitignored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/neashton/drivaerml/resolve/main"

# force_mom carries the force and moment coefficients -- the labels. Tiny, and
# useless to skip.
SURFACE_FILES = ("drivaer_{i}.stl", "boundary_{i}.vtp", "force_mom_{i}.csv")
VOLUME_FILES = ("volume_{i}.vtu.00.part", "volume_{i}.vtu.01.part")

# Refuse to fill more than this share of free disk, whatever the flags say.
DISK_FRACTION = 0.25
# Above this, --force is required. The fine-tuning recipe needs ~24 samples.
SOFT_RUN_CAP = 60

GB = 1024**3


def _url(i: int, name: str) -> str:
    return f"{HF_BASE}/run_{i}/{name}"


def remote_size(url: str) -> int | None:
    """Content-Length via HEAD, following redirects. None when absent."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001 - a missing run is not an error
        return None


def plan(runs: list[int], include_volume: bool) -> tuple[list[tuple[int, str, int]], int]:
    """Resolve what would be downloaded, by HEAD request. Returns (items, bytes)."""
    wanted = SURFACE_FILES + (VOLUME_FILES if include_volume else ())
    items: list[tuple[int, str, int]] = []
    total = 0
    for i in runs:
        for template in wanted:
            name = template.format(i=i)
            size = remote_size(_url(i, name))
            if size is None:
                print(f"  run_{i}: {name} unavailable, skipping")
                continue
            items.append((i, name, size))
            total += size
    return items, total


def download(items: list[tuple[int, str, int]], out_dir: Path) -> None:
    for n, (i, name, size) in enumerate(items, 1):
        dest = out_dir / f"run_{i}" / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"  [{n}/{len(items)}] have {name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{n}/{len(items)}] {name}  {size / GB:.3f} GB ...", end="", flush=True)
        # temp-then-rename, so an interrupted download never leaves a
        # truncated file that the size check would later accept
        tmp = dest.with_suffix(dest.suffix + ".part-tmp")
        with urllib.request.urlopen(_url(i, name), timeout=120) as resp, \
                tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
        tmp.replace(dest)
        print(" done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch a BOUNDED subset of DrivAerML. Surface files only by default."
    )
    ap.add_argument("--runs", type=int, required=True,
                    help="how many runs (required on purpose; start around 20)")
    ap.add_argument("--start", type=int, default=1, help="first run number")
    ap.add_argument("--out", default="data/drivaerml")
    ap.add_argument("--download", action="store_true",
                    help="actually fetch. Without it this only measures.")
    ap.add_argument("--include-volume", action="store_true",
                    help="also fetch volume_N.vtu. Needs --force. Not on our critical path.")
    ap.add_argument("--force", action="store_true",
                    help="override the run cap and the volume-file refusal")
    args = ap.parse_args()

    if args.runs < 1:
        sys.exit("--runs must be at least 1")
    if args.runs > SOFT_RUN_CAP and not args.force:
        sys.exit(
            f"--runs {args.runs} is above the cap of {SOFT_RUN_CAP}.\n"
            f"The DoMINO fine-tuning recipe was validated on 16 train + 8 test "
            f"samples, so you probably do not need this many.\n"
            f"Pass --force if you really mean it."
        )
    if args.include_volume and not args.force:
        sys.exit(
            "--include-volume needs --force.\n"
            "Volumetric fields are the largest files by far and cut-list item 4 "
            "says surface-only is sufficient for drag and lift.\n"
            "Fetch surfaces first, confirm the pipeline, then reconsider."
        )

    runs = list(range(args.start, args.start + args.runs))
    out_dir = Path(args.out)

    print("DrivAerML  (Hugging Face mirror, no credentials, CC BY-NC 4.0 non-commercial)")
    print(f"  runs        {runs[0]}..{runs[-1]}  ({len(runs)})")
    print(f"  files       {'surface + VOLUME' if args.include_volume else 'surface only'}")
    print(f"  destination {out_dir}\n")

    items, total = plan(runs, args.include_volume)
    if not items:
        sys.exit("nothing to download")

    free = shutil.disk_usage(Path.home()).free
    budget = free * DISK_FRACTION
    per_run = total / len(runs)

    print(f"\n  {len(items)} files")
    print(f"  TOTAL       {total / GB:.2f} GB   ({per_run / GB:.2f} GB per run)")
    print(f"  disk free   {free / GB:.1f} GB")
    print(f"  all 500 runs at this rate would be {per_run * 500 / GB:,.0f} GB")

    if total > budget:
        sys.exit(
            f"\nREFUSED: {total / GB:.1f} GB is more than {DISK_FRACTION:.0%} of your "
            f"{free / GB:.0f} GB free space.\nFetch fewer runs."
        )

    if not args.download:
        print(f"\nMeasured only -- nothing downloaded.")
        print(f"To fetch: add --download")
        return

    print()
    download(items, out_dir)
    got = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\ndone. {out_dir} is {got / GB:.2f} GB")
    print("next: combine multi-solid STLs before preprocessing --")
    print("  physicsnemo/examples/cfd/external_aerodynamics/xaeronet/surface/combine_stl_solids.py")


if __name__ == "__main__":
    main()
