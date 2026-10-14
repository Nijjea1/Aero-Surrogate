"""Bounded DrivAerML fetch for stage 2.  OWNER: Avneet.

    python -m src.data.drivaerml --runs 20              # measure only, downloads nothing
    python -m src.data.drivaerml --runs 20 --download   # fetch, after you have seen the size

WHY THIS EXISTS INSTEAD OF THE UPSTREAM SCRIPT

PhysicsNeMo ships examples/cfd/external_aerodynamics/xaeronet/download_aws_dataset.sh,
and run as shipped it does:

    for i in $(seq 1 500)          # every run
    aws s3 cp .../volume_$i.vtu    # including the volumetric field

That is the runaway our own CLAUDE.md warns about. The upstream script is
correct for someone with a storage array; on a laptop with a two-week deadline
it is a foot-gun, and remembering to edit two lines under deadline pressure is
not a safeguard.

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
import json
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "caemldatasets"
PREFIX = "drivaer/dataset"

SURFACE_FILES = ("drivaer_{i}.stl", "boundary_{i}.vtp")
VOLUME_FILES = ("volume_{i}.vtu",)

# Refuse to fill more than this share of free disk, whatever the flags say.
DISK_FRACTION = 0.25
# Above this, --force is required. The fine-tuning recipe needs ~24 samples.
SOFT_RUN_CAP = 60

GB = 1024**3


def _aws(*args: str) -> str:
    if shutil.which("aws") is None:
        sys.exit(
            "aws CLI not found. Install it first:\n"
            "    brew install awscli\n"
            "No AWS account or credentials are needed -- the bucket is public."
        )
    out = subprocess.run(
        ["aws", *args, "--no-sign-request"], capture_output=True, text=True
    )
    if out.returncode != 0:
        sys.exit(f"aws failed:\n{out.stderr.strip()}")
    return out.stdout


def list_run(i: int) -> dict[str, int]:
    """Object name -> size in bytes, for one run."""
    raw = _aws("s3api", "list-objects-v2", "--bucket", BUCKET,
               "--prefix", f"{PREFIX}/run_{i}/", "--output", "json")
    payload = json.loads(raw or "{}")
    return {
        Path(o["Key"]).name: int(o["Size"]) for o in payload.get("Contents", [])
    }


def plan(runs: list[int], include_volume: bool) -> tuple[list[tuple[int, str, int]], int]:
    """Resolve what would be downloaded. Returns (items, total bytes)."""
    wanted = SURFACE_FILES + (VOLUME_FILES if include_volume else ())
    items: list[tuple[int, str, int]] = []
    total = 0
    for i in runs:
        available = list_run(i)
        if not available:
            print(f"  run_{i}: not found, skipping")
            continue
        for template in wanted:
            name = template.format(i=i)
            size = available.get(name)
            if size is None:
                print(f"  run_{i}: {name} missing, skipping")
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
        print(f"  [{n}/{len(items)}] {name}  {size / GB:.2f} GB ...", flush=True)
        _aws("s3", "cp", f"s3://{BUCKET}/{PREFIX}/run_{i}/{name}", str(dest))


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

    print(f"DrivAerML  (public S3, no credentials, CC BY-NC 4.0 non-commercial)")
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
    print(f"  all 500 runs at this rate would be {per_run * 500 / GB:.0f} GB")

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
