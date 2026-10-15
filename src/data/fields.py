"""Stage-2 field extraction: boundary_N.vtp -> trainable arrays.  OWNER: Avneet.

    python -m src.data.fields --extract            # all runs
    python -m src.data.fields --extract --limit 3
    python -m src.data.fields --summary

WHY FIELD PREDICTION CHANGES THE DATA SITUATION

Stage 1 had 30 DrivAerML cars and therefore 30 labels, which is why training a
point-cloud net on them gave test R^2 -0.73. Field prediction has 8.83 MILLION
labels per car -- one Cp value per surface cell -- so the same 30 cars carry
~265 million supervised targets. Sample count stops being the binding
constraint, which is exactly why NVIDIA's DoMINO fine-tuning recipe works on
16 training samples.

WHAT GETS STORED, per run

    query_xyz     (M,3)  cell centres, normalised to the unit sphere
    query_normal  (M,3)  outward cell normals
    query_area    (M,)   cell areas, m^2 -- needed to integrate
    cp            (M,)   target: mean pressure coefficient
    pprime2       (M,)   pressure fluctuation variance from the LES
    shear         (M,3)  wall shear stress, non-dimensionalised by q_inf
    region        (M,)   contract PRESSURE_REGIONS index
    cloud_xyz     (G,3)  a separate subsample for the global geometry encoder

NORMALISATION IS THE SAME CONVENTION AS STAGE 1 -- bounding-box centre of the
combined mesh, divided by max vertex radius (see dataset.preprocess_spec). The
two stages must agree or a geometry encoder trained on one cannot read the
other, and that mismatch produces confident wrong answers rather than an error.

pprime2 is kept because it is the LES's own measure of where the flow is
unsteady, and therefore a physically grounded map of where prediction is hard.
That is an uncertainty signal from the data rather than from the model's
opinion of itself, which makes it a genuinely independent check on the
calibration story.

Shear is divided by q_inf so the target is dimensionless like Cp. Integrating
the two then gives Cd directly, which is the consistency check validated in
src/data/preprocess.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# DrivAerML reference conditions, same as src/data/preprocess.py.
U_INF, RHO = 38.889, 1.205
Q_INF = 0.5 * RHO * U_INF**2

N_QUERY = 100_000      # cells sampled per run; 8.83M available, 100k is plenty
N_CLOUD = 8_192        # points for the global geometry encoder
FIELD_VERSION = "fields-v1"


def _runs(root: Path) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in root.glob("run_*") if p.is_dir())


def extract_one(root: Path, run: int, out_dir: Path,
                n_query: int = N_QUERY, n_cloud: int = N_CLOUD,
                seed: int | None = None) -> dict:
    """Read one run's surface and write a compressed .npz of trainable arrays."""
    import pyvista as pv
    import trimesh
    from scipy.spatial import cKDTree

    combined = root / "combined" / f"run_{run}"
    stl_path = combined / f"drivaer_{run}_single_solid.stl"
    if not stl_path.exists():
        raise FileNotFoundError(f"{stl_path} -- run --combine first")

    mesh = trimesh.load(stl_path, process=False)
    # Same normalisation as stage 1. See the module docstring.
    centre = mesh.bounds.mean(axis=0)
    radius = float(np.linalg.norm(mesh.vertices - centre, axis=1).max())

    surf = pv.read(root / f"run_{run}" / f"boundary_{run}.vtp")
    sized = surf.compute_cell_sizes(length=False, area=True, volume=False)
    normals = surf.extract_surface(algorithm="dataset_surface").compute_normals(
        cell_normals=True, point_normals=False, auto_orient_normals=True
    )

    centres = np.asarray(sized.cell_centers().points, np.float64)
    area = np.asarray(sized.cell_data["Area"], np.float64)
    normal = np.asarray(normals.cell_data["Normals"], np.float64)
    cp = np.asarray(surf.cell_data["CpMeanTrim"], np.float64)
    pprime2 = np.asarray(surf.cell_data["pPrime2MeanTrim"], np.float64)
    shear = np.asarray(surf.cell_data["wallShearStressMeanTrim"], np.float64)
    n_cells = len(cp)

    # Integrate over EVERY cell before subsampling -- the reference Cd belongs
    # to the full surface, and a subsample would bias it.
    a_ref = _frontal_area(mesh)
    cd = float(((cp * -normal[:, 0] * area).sum()
                - (shear[:, 0] * area).sum() / Q_INF) / a_ref)
    cl = float(((cp * -normal[:, 2] * area).sum()
                - (shear[:, 2] * area).sum() / Q_INF) / a_ref)

    # Region labels, by spatial join onto the combined mesh's face index.
    regions = json.loads((combined / f"drivaer_{run}_index.regions.json").read_text())
    face_region = np.load(combined / f"drivaer_{run}_index.faces.npy")
    _, nearest = cKDTree(mesh.triangles_center).query(centres, workers=-1)
    cell_region = face_region[nearest]

    rng = np.random.default_rng(run if seed is None else seed)
    take = rng.choice(n_cells, size=min(n_query, n_cells), replace=False)
    cloud_take = rng.choice(n_cells, size=min(n_cloud, n_cells), replace=False)

    payload = {
        "query_xyz": ((centres[take] - centre) / radius).astype(np.float32),
        "query_normal": normal[take].astype(np.float32),
        "query_area": area[take].astype(np.float32),
        "cp": cp[take].astype(np.float32),
        "pprime2": pprime2[take].astype(np.float32),
        "shear": (shear[take] / Q_INF).astype(np.float32),
        "region": cell_region[take].astype(np.int16),
        "cloud_xyz": ((centres[cloud_take] - centre) / radius).astype(np.float32),
    }
    meta = {
        "run": run, "version": FIELD_VERSION,
        "n_cells_total": int(n_cells), "n_query": int(len(take)),
        "centre": centre.tolist(), "radius": radius,
        "a_ref_m2": a_ref, "total_area_m2": float(area.sum()),
        "cd_integrated": cd, "cl_integrated": cl,
        "regions": regions,
        "cp_mean": float(cp.mean()), "cp_std": float(cp.std()),
    }
    forces = root / f"run_{run}" / f"force_mom_{run}.csv"
    if forces.exists():
        import pandas as pd

        row = pd.read_csv(forces).iloc[0]
        meta["cd_reported"] = float(row["Cd"])
        meta["cl_reported"] = float(row["Cl"])

    out_dir.mkdir(parents=True, exist_ok=True)
    # Through a handle: savez_compressed(path) appends ".npz" when the path
    # does not end in it, so the rename would look for a file that is not
    # there. Same trap as np.save in src/data/dataset.py.
    tmp = out_dir / f"run_{run}.tmp"
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **payload, meta=json.dumps(meta))
    tmp.replace(out_dir / f"run_{run}.npz")
    return meta


def _frontal_area(mesh, n: int = 1400) -> float:
    """Projected area onto the y-z plane. See src/data/preprocess.py."""
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


def load_run(path: Path) -> tuple[dict, dict]:
    """Read one .npz back as (arrays, meta)."""
    z = np.load(path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != "meta"}
    return arrays, json.loads(str(z["meta"]))


def cmd_extract(root: Path, limit: int | None, force: bool) -> int:
    runs = _runs(root)[: limit or None]
    out_dir = root / "fields"
    if not runs:
        print(f"no runs in {root}")
        return 1

    print(f"extracting {len(runs)} runs -> {out_dir}")
    print(f"  {N_QUERY:,} query cells + {N_CLOUD:,} cloud points per run\n")
    done = skipped = 0
    for i in runs:
        dest = out_dir / f"run_{i}.npz"
        if dest.exists() and not force:
            print(f"  run_{i:<4} skipped (exists)")
            skipped += 1
            continue
        meta = extract_one(root, i, out_dir)
        err = ""
        if "cd_reported" in meta:
            err = f"  (reported {meta['cd_reported']:.4f}, err " \
                  f"{meta['cd_integrated'] - meta['cd_reported']:+.4f})"
        print(f"  run_{i:<4} {meta['n_cells_total']:>9,} cells  "
              f"Cp {meta['cp_mean']:+.4f}+-{meta['cp_std']:.4f}  "
              f"Cd_int {meta['cd_integrated']:.4f}{err}", flush=True)
        done += 1

    print(f"\n{done} extracted, {skipped} skipped")
    if out_dir.exists():
        size = sum(f.stat().st_size for f in out_dir.glob("*.npz"))
        print(f"{out_dir} is {size / 2**20:.0f} MB")
    return 0


def cmd_summary(root: Path) -> int:
    out_dir = root / "fields"
    paths = sorted(out_dir.glob("*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        print(f"nothing in {out_dir} -- run --extract first")
        return 1

    cps, cds, errs = [], [], []
    for p in paths:
        arrays, meta = load_run(p)
        cps.append(arrays["cp"])
        cds.append(meta["cd_integrated"])
        if "cd_reported" in meta:
            errs.append(meta["cd_integrated"] - meta["cd_reported"])

    cp = np.concatenate(cps)
    print(f"{len(paths)} runs, {len(cp):,} query cells total\n")
    print(f"  Cp        mean {cp.mean():+.4f}  sd {cp.std():.4f}  "
          f"range [{cp.min():+.3f}, {cp.max():+.3f}]")
    print(f"  Cd_int    mean {np.mean(cds):.4f}  sd {np.std(cds):.4f}")
    if errs:
        e = np.array(errs)
        print(f"  vs report bias {e.mean():+.4f}  scatter {e.std():.4f}")
    print(f"\n  a constant bias is the validated signature; see preprocess.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DrivAerML surface field extraction")
    ap.add_argument("--root", default="data/drivaerml")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if args.extract:
        return cmd_extract(root, args.limit, args.force)
    if args.summary:
        return cmd_summary(root)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment, misc]


def field_splits(runs: list[int], val: int = 5, test: int = 5,
                 seed: int = 0) -> dict[str, list[int]]:
    """Split BY RUN, never by point.

    This is the one decision that decides whether the numbers mean anything.
    Each car contributes 100,000 query points; splitting points would put the
    same car in train and test, and the model would be scored on interpolating
    within a geometry it had already seen. Generalisation here means a car it
    has never seen, so whole runs move together.
    """
    rng = np.random.default_rng(seed)
    shuffled = [int(runs[i]) for i in rng.permutation(len(runs))]
    if val + test >= len(shuffled):
        raise ValueError(f"val+test={val + test} leaves no training runs "
                         f"out of {len(shuffled)}")
    return {
        "test": sorted(shuffled[:test]),
        "val": sorted(shuffled[test:test + val]),
        "train": sorted(shuffled[test + val:]),
    }


class SurfaceFieldDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """One item per CAR: a geometry cloud plus a fresh sample of query points.

    Re-sampling the query points every epoch is deliberate. A car holds 100,000
    stored cells and an item yields a few thousand, so each epoch sees a
    different slice of the same surface -- effectively free augmentation, and
    it stops the model memorising particular cells.
    """

    def __init__(self, root: Path | str, runs: list[int], n_query: int = 4096,
                 n_cloud: int = 2048, seed: int = 0, resample: bool = True):
        self.dir = Path(root) / "fields"
        self.runs = list(runs)
        self.n_query = int(n_query)
        self.n_cloud = int(n_cloud)
        self.resample = bool(resample)
        self._epoch = 0
        self.seed = int(seed)

        self.data: list[dict] = []
        self.meta: list[dict] = []
        for run in self.runs:
            path = self.dir / f"run_{run}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} -- run `python -m src.data.fields --extract` first")
            arrays, meta = load_run(path)
            self.data.append(arrays)
            self.meta.append(meta)
        self.regions: list[str] = self.meta[0]["regions"]

    def set_epoch(self, epoch: int) -> None:
        """Change the resampling seed so each epoch sees different cells."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.runs)

    def target_stats(self) -> tuple[float, float]:
        """Mean and sd of Cp over every stored cell in this split."""
        cp = np.concatenate([d["cp"] for d in self.data])
        return float(cp.mean()), float(cp.std())

    def __getitem__(self, index: int) -> dict:
        arrays, meta = self.data[index], self.meta[index]
        n_stored = len(arrays["cp"])
        base = self.runs[index] * 7919 + self.seed
        rng = np.random.default_rng(base + (self._epoch if self.resample else 0))

        q = min(self.n_query, n_stored)
        take = rng.choice(n_stored, size=q, replace=False)
        cloud = arrays["cloud_xyz"]
        if self.n_cloud < len(cloud):
            cloud = cloud[rng.choice(len(cloud), size=self.n_cloud, replace=False)]

        return {
            "cloud": cloud.astype(np.float32),
            "query_xyz": arrays["query_xyz"][take],
            "query_normal": arrays["query_normal"][take],
            "query_area": arrays["query_area"][take],
            "cp": arrays["cp"][take][:, None],
            "pprime2": arrays["pprime2"][take],
            "region": arrays["region"][take].astype(np.int64),
            "a_ref": np.float32(meta["a_ref_m2"]),
            # Ratio of the whole wetted surface to the area actually sampled.
            # Multiplying a partial surface integral by this estimates the
            # full-surface value, which is what makes the integrated Cd
            # comparable with the reported one instead of ~1e-4 of it.
            "area_scale": np.float32(
                meta["total_area_m2"] / float(arrays["query_area"][take].sum())
            ),
            "run": np.int64(meta["run"]),
        }
