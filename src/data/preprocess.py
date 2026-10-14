"""Stage-2 preprocessing: DrivAerML surfaces -> model-ready arrays.  OWNER: Avneet.

    python -m src.data.preprocess --inspect              # what is in the data
    python -m src.data.preprocess --combine              # 49 solids -> one mesh
    python -m src.data.preprocess --regions              # check the region map
    python -m src.data.preprocess --validate --limit 3   # integration consistency
    python -m src.data.preprocess --index                # -> a stage-1 trainable set

STEP 1, AND IT IS MANDATORY: each drivaer_N.stl is a SCENE of 49 named solids,
none of them watertight on its own. Area-weighted sampling and any surface
integration over that is meaningless until they are merged. PhysicsNeMo ships
xaeronet/surface/combine_stl_solids.py for this; its logic is trimesh.load ->
trimesh.util.concatenate -> export. We do the same here rather than cloning
PhysicsNeMo and installing hydra and omegaconf for fifteen lines. This is a
utility, not the DoMINO recipe -- that one we do not reimplement.

WHAT WE ADD THAT UPSTREAM DOES NOT: a face -> region index.

The 49 solids are named semantically -- BodyHood, BodyRoof, BodyFasciafront1,
NotchbackWindowrear -- and those names map onto contracts.PRESSURE_REGIONS
almost one to one. Upstream's combine throws the names away. Keeping them means
that once stage 2 predicts a surface pressure field we can integrate it per
region and fill Evaluation.pressure_summary with the exact eight contract keys,
instead of leaving them stubbed forever.

That matters beyond tidiness: the agent is supposed to reason about WHERE the
drag is. "Pressure is high on the rear fascia" is a sentence Nemotron can act
on. A single scalar is not.

THE SURFACE FIELDS, measured on boundary_N.vtp (8.83M cells per run):

    CpMeanTrim                mean pressure coefficient
    wallShearStressMeanTrim   wall shear stress, 3-vector
    pMeanTrim                 mean static pressure
    pPrime2MeanTrim           pressure fluctuation variance

The last one is worth noticing: it is the LES's own measure of how unsteady the
flow is at each cell, which is a physically grounded map of where prediction is
hard. Independent of anything our model reports about itself.

INTEGRATION IS VALIDATED -- the "free consistency check" the plan wants. With
    Cd = [ sum(Cp * -n_x * A) - sum(tau_x * A)/q ] / A_ref
and a per-run frontal area measured off the combined mesh, three runs give

    run 1   reported 0.3035   integrated 0.2965   err -0.0070
    run 2   reported 0.2812   integrated 0.2743   err -0.0069
    run 3   reported 0.2581   integrated 0.2516   err -0.0065

A CONSTANT bias of -0.0068, not scatter. That matters: a constant offset is one
calibration factor, while mixed-sign scatter would mean the method is wrong.
With a FIXED A_ref of 2.17 the same three runs give +0.0025, +0.0094, -0.0101 --
mixed sign -- because the morphs change A_Car_Width, A_Car_Length and
A_Car_Roof_Height, so frontal area varies per run. Measuring it per run is what
turns scatter into bias.

An exact projected area (shapely, via trimesh.path.polygons.projected, ~15 min
on 753k faces) narrows it but does not close it:

    run 1   A raster 2.2397   A exact 2.2230   Cd 0.2987   err -1.6%
    run 2   A raster 2.2991   A exact 2.2786   Cd 0.2767   err -1.6%
    run 3   A raster 2.1391   A exact 2.1221   Cd 0.2536   err -1.7%

The rasteriser overestimates area by only ~0.8%, not the 2.3% first assumed.
What is left is a -1.6% constant with 0.1% scatter -- so one scale factor is
still unaccounted for, most plausibly a reference-area convention (whether
below-ground wheel area counts, say) or the averaging window behind the "Trim"
in CpMeanTrim differing from the window behind the reported forces. Not chased
further, because it does not affect the surrogate: we TRAIN on the reported
coefficients and use integration only as a consistency check, where a constant
factor is detectable and correctable while scatter would not be.

The finding that matters is already in hand: the surface -> coefficient path
reproduces CFD to a constant factor, so a model predicting Cp and wall shear
can be checked against its own integrated drag.

NOTE ON WHAT DrivAerML IS: every run is a morphed NOTCHBACK. Nine of the 49
solids are named Notchback*. So the cross-category generalisation result
(contribution 3) cannot come from this dataset at all -- it has one category.
That result comes from DrivAerNet, where we have fastback, notchback and
estateback, and stage 1 already measures it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.contracts import PRESSURE_REGIONS

# Solid-name prefixes per contract region, longest match wins. Built by reading
# the 49 names in a DrivAerML STL; see --regions to verify coverage against
# whatever is actually on disk rather than trusting this list.
REGION_MAP: dict[str, tuple[str, ...]] = {
    "front_fascia": (
        "BodyFasciafront", "ClosedGrillLowerInsert", "ClosedGrillUpperInsert",
        "BodyHeadlamps",
    ),
    "hood": ("BodyHood",),
    "windscreen": ("BodyWindowfront",),
    "roof": ("BodyRoof", "NotchbackRoof"),
    "rear_window": ("NotchbackWindowrear",),
    "rear_fascia": ("BodyRear", "NotchbackTrunk", "CTRL_SURFACE_Outlet"),
    "underbody": (
        "EngineUndershield", "OCDADetailedUnderbody", "Powertrain",
        "ExhaustSystem", "BodyRocker",
    ),
    "wheel_wells": (
        "CTRL_SURFACE_Wheelhouse", "CTRL_SURFACE_Wheelshouse", "WheelSupport",
        "TirePlinth", "Tires", "Rims", "BrakeDisc",
    ),
}

# Named solids that belong to no contract region: flanks, glazing, pillars,
# mirrors, trim. They are real surface and they carry real drag, so they are
# kept in the combined mesh and labelled "other" -- dropping them would make
# the integrated force wrong.
OTHER_REGION = "other"


def region_of(solid_name: str) -> str:
    """Contract region for one named solid, or 'other'."""
    best, best_len = OTHER_REGION, 0
    for region, prefixes in REGION_MAP.items():
        for prefix in prefixes:
            if solid_name.startswith(prefix) and len(prefix) > best_len:
                best, best_len = region, len(prefix)
    return best


def _runs(root: Path) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in root.glob("run_*") if p.is_dir())


def combine_one(stl_path: Path, out_path: Path, index_path: Path) -> dict:
    """Merge a multi-solid STL and write a face -> region index beside it."""
    import trimesh

    loaded = trimesh.load(stl_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        names = list(loaded.geometry)
        parts = [loaded.geometry[n] for n in names]
    else:
        names, parts = [stl_path.stem], [loaded]

    combined = trimesh.util.concatenate(parts)

    # Face ranges, in the same order concatenate used, so the index lines up
    # with the combined mesh's face array.
    face_region = np.empty(len(combined.faces), dtype=object)
    offset = 0
    per_solid: dict[str, dict] = {}
    for name, part in zip(names, parts):
        n = len(part.faces)
        region = region_of(name)
        face_region[offset : offset + n] = region
        per_solid[name] = {"region": region, "faces": n,
                           "area_m2": round(float(part.area), 5)}
        offset += n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_path)

    regions = sorted(set(face_region))
    index = {
        "source": stl_path.name,
        "combined": out_path.name,
        "n_solids": len(names),
        "n_faces": int(len(combined.faces)),
        "watertight": bool(combined.is_watertight),
        "area_m2": round(float(combined.area), 5),
        "bounds": combined.bounds.round(4).tolist(),
        "per_solid": per_solid,
        "face_counts_by_region": {
            r: int((face_region == r).sum()) for r in regions
        },
        "area_by_region": {
            r: round(sum(v["area_m2"] for v in per_solid.values()
                         if v["region"] == r), 5)
            for r in regions
        },
    }
    index_path.write_text(json.dumps(index, indent=2))
    # The per-face labels themselves, for surface-field integration later.
    np.save(index_path.with_suffix(".faces.npy"),
            np.array([regions.index(r) for r in face_region], dtype=np.int16))
    (index_path.parent / (index_path.stem + ".regions.json")).write_text(
        json.dumps(regions)
    )
    return index


def cmd_regions(root: Path) -> int:
    """Check REGION_MAP against the solid names actually present."""
    import trimesh

    runs = _runs(root)
    if not runs:
        print(f"no runs in {root} -- fetch with: python -m src.data.drivaerml --runs 20 --download")
        return 1

    scene = trimesh.load(root / f"run_{runs[0]}" / f"drivaer_{runs[0]}.stl", process=False)
    names = sorted(scene.geometry) if isinstance(scene, trimesh.Scene) else []
    print(f"run_{runs[0]}: {len(names)} named solids\n")

    by_region: dict[str, list[str]] = {}
    for n in names:
        by_region.setdefault(region_of(n), []).append(n)

    for region in PRESSURE_REGIONS:
        solids = by_region.get(region, [])
        mark = "  " if solids else "  MISSING -- no solid maps here"
        print(f"  {region:14} {len(solids):>2} solids{mark}")
        for s in solids:
            print(f"                   {s}")

    other = by_region.get(OTHER_REGION, [])
    print(f"\n  {OTHER_REGION:14} {len(other):>2} solids (kept in the mesh, no contract region)")
    for i in range(0, len(other), 3):
        print("                   " + "  ".join(f"{s:28}" for s in other[i : i + 3]))

    unmapped = [r for r in PRESSURE_REGIONS if not by_region.get(r)]
    print()
    if unmapped:
        print(f"NOT READY -- {len(unmapped)} contract region(s) unmapped: {unmapped}")
        return 1
    print(f"READY -- all {len(PRESSURE_REGIONS)} contract regions have at least one solid.")
    return 0


# DrivAerML reference conditions: 38.889 m/s (140 km/h), rho 1.205 kg/m3.
U_INF, RHO = 38.889, 1.205
Q_INF = 0.5 * RHO * U_INF**2


def _frontal_area(mesh, n: int = 1400) -> float:
    """Projected area onto the y-z plane, by rasterising the x-projection.

    Approximate: large triangles get their 2D bounding box filled, which
    overestimates by roughly 2%. Good enough to turn per-run area variation
    from an error into a constant bias; see the module docstring.
    """
    tri = mesh.triangles[:, :, 1:]
    flat = tri.reshape(-1, 2)
    lo, hi = flat.min(0), flat.max(0)
    h = (hi - lo).max() / n
    grid = np.zeros((int((hi[0] - lo[0]) / h) + 2, int((hi[1] - lo[1]) / h) + 2), bool)
    ij = np.floor((tri - lo) / h).astype(np.int32)
    for k in range(3):
        grid[ij[:, k, 0], ij[:, k, 1]] = True
    lo_ij, hi_ij = ij.min(1), ij.max(1)
    big = ~((hi_ij - lo_ij) <= 1).all(1)
    for a, b in zip(lo_ij[big], hi_ij[big]):
        grid[a[0] : b[0] + 1, a[1] : b[1] + 1] = True
    return float(grid.sum() * h * h)


def surface_fields(vtp_path: Path) -> dict:
    """Read one boundary_N.vtp and return per-cell Cp, shear, area, normals."""
    import pyvista as pv

    m = pv.read(vtp_path)
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    normals = m.extract_surface(algorithm="dataset_surface").compute_normals(
        cell_normals=True, point_normals=False, auto_orient_normals=True
    )
    return {
        "cp": np.asarray(m.cell_data["CpMeanTrim"], np.float64),
        "tau": np.asarray(m.cell_data["wallShearStressMeanTrim"], np.float64),
        "p_prime2": np.asarray(m.cell_data["pPrime2MeanTrim"], np.float64),
        "area": np.asarray(sized.cell_data["Area"], np.float64),
        "normal": np.asarray(normals.cell_data["Normals"], np.float64),
        "centre": np.asarray(sized.cell_centers().points, np.float64),
    }


def integrate_coefficients(f: dict, a_ref: float) -> dict:
    """Cp and wall shear -> Cd and Cl. The consistency check."""
    cd_p = float((f["cp"] * -f["normal"][:, 0] * f["area"]).sum() / a_ref)
    cd_v = float(-(f["tau"][:, 0] * f["area"]).sum() / (Q_INF * a_ref))
    cl_p = float((f["cp"] * -f["normal"][:, 2] * f["area"]).sum() / a_ref)
    cl_v = float(-(f["tau"][:, 2] * f["area"]).sum() / (Q_INF * a_ref))
    return {"cd": cd_p + cd_v, "cd_pressure": cd_p, "cd_viscous": cd_v,
            "cl": cl_p + cl_v, "cl_pressure": cl_p, "cl_viscous": cl_v,
            "a_ref_m2": a_ref}


def region_summary(f: dict, combined_dir: Path, run: int) -> dict:
    """Area-weighted mean Cp per contract region -> Evaluation.pressure_summary.

    The .vtp surface (8.8M cells) and the STL (753k faces) are different
    discretisations of the same car, so the join is spatial (nearest face
    centroid), not by index.
    """
    from scipy.spatial import cKDTree
    import trimesh

    regions = json.loads((combined_dir / f"drivaer_{run}_index.regions.json").read_text())
    face_region = np.load(combined_dir / f"drivaer_{run}_index.faces.npy")
    stl = trimesh.load(combined_dir / f"drivaer_{run}_single_solid.stl", process=False)
    _, nearest = cKDTree(stl.triangles_center).query(f["centre"], workers=-1)
    cell_region = face_region[nearest]

    out: dict[str, float] = {}
    for i, name in enumerate(regions):
        sel = cell_region == i
        if sel.any():
            w = f["area"][sel]
            out[name] = round(float((f["cp"][sel] * w).sum() / w.sum()), 5)
    return out


def cmd_validate(root: Path, limit: int | None) -> int:
    """Integrate the real surface fields and compare with the reported forces."""
    import pandas as pd
    import trimesh

    runs = _runs(root)[: limit or None]
    combined_root = root / "combined"
    errs: list[float] = []
    print(f"{'run':>4} {'Cd rep':>8} {'Cd int':>8} {'err':>8} | "
          f"{'Cl rep':>8} {'Cl int':>8} {'err':>8} | {'A_ref':>7}")
    for i in runs:
        cdir = combined_root / f"run_{i}"
        if not (cdir / f"drivaer_{i}_single_solid.stl").exists():
            print(f"{i:>4}  not combined yet -- run --combine first")
            return 1
        truth = pd.read_csv(root / f"run_{i}" / f"force_mom_{i}.csv").iloc[0]
        stl = trimesh.load(cdir / f"drivaer_{i}_single_solid.stl", process=False)
        f = surface_fields(root / f"run_{i}" / f"boundary_{i}.vtp")
        c = integrate_coefficients(f, _frontal_area(stl))
        errs.append(c["cd"] - float(truth["Cd"]))
        print(f"{i:>4} {truth['Cd']:8.4f} {c['cd']:8.4f} {errs[-1]:+8.4f} | "
              f"{truth['Cl']:8.4f} {c['cl']:8.4f} {c['cl']-float(truth['Cl']):+8.4f} | "
              f"{c['a_ref_m2']:7.4f}")
        summary = region_summary(f, cdir, i)
        (cdir / f"drivaer_{i}_fields.json").write_text(
            json.dumps({"integrated": c, "pressure_summary": summary,
                        "reported": {"Cd": float(truth["Cd"]), "Cl": float(truth["Cl"])}},
                       indent=2))

    e = np.array(errs)
    print(f"\nCd bias {e.mean():+.4f}   scatter (sd) {e.std():.4f}")
    print("  A constant bias means the method is right and one factor is off;")
    print("  mixed-sign scatter would mean the method is wrong. See the docstring.")
    print(f"\npressure_summary written per run, e.g. "
          f"{combined_root / f'run_{runs[0]}' / f'drivaer_{runs[0]}_fields.json'}")
    return 0


def cmd_index(root: Path) -> int:
    """Emit a labels CSV + flat mesh dir so the STAGE-1 pipeline can train on
    real DrivAerML geometry with no new training code.

    src/data/dataset.py already reads "a CSV of ids and targets plus a
    directory of <id>.stl". DrivAerML ships neither: labels are one tiny CSV
    per run, and the combined meshes sit at combined/run_N/drivaer_N_*.stl. So
    rather than teach the loader a second layout, translate into the layout it
    has. The point-cloud model, the two-phase trainer, the OOD detector and
    every test then work on real cars unchanged.

    Meshes are symlinked, not copied -- 30 runs of combined STL is ~1 GB and
    duplicating it buys nothing.
    """
    import pandas as pd

    runs = _runs(root)
    combined_root = root / "combined"
    out = root / "index"
    mesh_dir = out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for i in runs:
        stl = combined_root / f"run_{i}" / f"drivaer_{i}_single_solid.stl"
        csv = root / f"run_{i}" / f"force_mom_{i}.csv"
        if not stl.exists() or not csv.exists():
            missing.append(i)
            continue
        forces = pd.read_csv(csv).iloc[0]
        link = mesh_dir / f"run_{i}.stl"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(stl.resolve())
        rows.append({
            "Design": f"run_{i}",
            "Average Cd": float(forces["Cd"]),
            "Average Cl": float(forces["Cl"]),
            # Every DrivAerML run is a morphed notchback, so this column is
            # constant. Kept for the schema, useless for splitting -- the
            # cross-category experiment needs DrivAerNet.
            "Category": "notchback",
        })

    if not rows:
        print(f"nothing indexed -- run --combine first ({len(missing)} runs missing output)")
        return 1

    df = pd.DataFrame(rows)
    csv_path = out / "labels.csv"
    df.to_csv(csv_path, index=False)

    cd = df["Average Cd"].to_numpy()
    print(f"indexed {len(df)} runs -> {out}")
    print(f"  labels   {csv_path}")
    print(f"  meshes   {mesh_dir}  ({len(list(mesh_dir.glob('*.stl')))} symlinks)")
    print(f"  Cd       mean {cd.mean():.4f}  sd {cd.std():.4f}  "
          f"range [{cd.min():.4f}, {cd.max():.4f}]")
    if missing:
        print(f"  skipped  {len(missing)} runs with no combined mesh: {missing[:8]}")
    if len(df) < 100:
        print(f"\n  WARNING: {len(df)} designs is far too few to train a point-cloud")
        print("  model from scratch. This path exists to prove the geometry")
        print("  pipeline end to end, not to produce a reportable R^2.")
    print(f"\nnext:  python -m src.data.dataset --check --config configs/drivaerml.yaml")
    return 0


def cmd_inspect(root: Path) -> int:
    """What is actually in the downloaded data, before touching it."""
    runs = _runs(root)
    if not runs:
        print(f"no runs in {root}")
        return 1
    print(f"{len(runs)} runs in {root}: {runs[0]}..{runs[-1]}\n")

    import pandas as pd

    rows = []
    for i in runs:
        d = root / f"run_{i}"
        csv = d / f"force_mom_{i}.csv"
        row = {"run": i}
        if csv.exists():
            row.update(pd.read_csv(csv).iloc[0].to_dict())
        rows.append(row)
    df = pd.DataFrame(rows)

    print("force and moment coefficients (the stage-2 labels):")
    for c in [c for c in df.columns if c != "run"]:
        v = df[c].to_numpy(float)
        print(f"  {c:5} mean {v.mean():8.4f}  sd {v.std():7.4f}  "
              f"range [{v.min():7.4f}, {v.max():7.4f}]")
    print(f"\n  Cd spread across {len(df)} runs: {df['Cd'].std():.4f}")
    print("  (DrivAerNet stage 1 had sd 0.0220 across 4,165 designs -- a wider")
    print("   spread, because DrivAerML morphs one notchback rather than three styles)")
    return 0


def cmd_combine(root: Path, limit: int | None, force: bool) -> int:
    runs = _runs(root)[: limit or None]
    if not runs:
        print(f"no runs in {root}")
        return 1

    out_root = root / "combined"
    print(f"combining {len(runs)} runs -> {out_root}\n")
    done, skipped = 0, 0
    for i in runs:
        stl = root / f"run_{i}" / f"drivaer_{i}.stl"
        out = out_root / f"run_{i}" / f"drivaer_{i}_single_solid.stl"
        idx = out_root / f"run_{i}" / f"drivaer_{i}_index.json"
        if out.exists() and idx.exists() and not force:
            print(f"  run_{i:<4} skipped (exists)")
            skipped += 1
            continue
        info = combine_one(stl, out, idx)
        print(f"  run_{i:<4} {info['n_solids']} solids -> {info['n_faces']:>8,} faces  "
              f"area {info['area_m2']:.3f} m2  watertight={info['watertight']}")
        done += 1

    print(f"\n{done} combined, {skipped} skipped")
    if done:
        sample = json.loads(
            (out_root / f"run_{runs[0]}" / f"drivaer_{runs[0]}_index.json").read_text()
        )
        print("\nsurface area by contract region (run "
              f"{runs[0]}, fractions of total):")
        total = sample["area_m2"]
        for r, a in sorted(sample["area_by_region"].items(), key=lambda kv: -kv[1]):
            print(f"  {r:14} {a:8.3f} m2  {a / total:5.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DrivAerML stage-2 preprocessing")
    ap.add_argument("--root", default="data/drivaerml")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--regions", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="integrate surface fields, compare with reported forces")
    ap.add_argument("--index", action="store_true",
                    help="emit labels.csv + flat mesh dir for the stage-1 pipeline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo existing output")
    args = ap.parse_args()

    root = Path(args.root)
    if args.inspect:
        return cmd_inspect(root)
    if args.regions:
        return cmd_regions(root)
    if args.combine:
        return cmd_combine(root, args.limit, args.force)
    if args.validate:
        return cmd_validate(root, args.limit)
    if args.index:
        return cmd_index(root)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
