"""DrivAerNet loading for stage 1.  OWNER: Avneet.

Build the --check path FIRST and run it before any training:

    python -m src.data.dataset --check --config configs/baseline.yaml

It should print the columns actually found in the CSV, how many mesh/label
pairs matched, and a sample point-cloud shape. Published DrivAerNet headers may
differ from the config defaults -- fix that here, not in the trainer.

Cache sampled point clouds as .npy. You will iterate on the model a hundred
times and must not re-sample meshes each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # torch is not needed for --check, so do not make it a hard import
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment, misc]


# --------------------------------------------------------------------------
# The preprocessing recipe. This is the un-contracted seam: at serve time the
# service turns params -> mesh -> point cloud, and if it samples or normalises
# differently from training the model sees a distribution it never saw.
#
# train.py writes this dict into every checkpoint under "preprocess", so
# whoever loads the checkpoint can reproduce it exactly. Bump the version if
# any of it changes -- an old checkpoint then fails loudly instead of quietly
# predicting nonsense.
# --------------------------------------------------------------------------

PREPROCESS_VERSION: str = "pointcloud-v1"


def preprocess_spec(cfg: dict) -> dict:
    """Everything needed to turn a design into model input.

    For the tabular path there is no mesh, so the spec is the column order --
    which matters just as much: a model fed its parameters in a different
    order than it was trained on fails silently.
    """
    d = cfg["data"]
    if str(d.get("kind", "pointcloud")).lower() == "table":
        from src.contracts import PARAM_NAMES

        cols = list(d.get("param_columns") or PARAM_NAMES)
        return {
            "version": "params-v1",
            "input": "design parameter vector",
            "param_columns": cols,
            "shape": f"({len(cols)},)",
            "dtype": "float32",
            "standardise": "subtract norm_stats.param_mean, divide by param_std",
            "note": "column ORDER is part of the contract -- use this list",
        }
    return {
        "version": PREPROCESS_VERSION,
        "n_points": int(d["n_points"]),
        "normalize_geometry": bool(d["normalize_geometry"]),
        "sampling": "trimesh.sample.sample_surface (area-weighted)",
        "center": "bounding-box centre of the mesh",
        "scale": "max vertex distance from that centre (unit sphere)",
        "dtype": "float32",
        "shape": f"({int(d['n_points'])}, 3)",
        "seed": "blake2b(mesh filename) -- per-mesh, so sampling is reproducible",
    }


# Headers vary between DrivAerNet releases. check() resolves against these and
# tells you what to put in the config; it never silently rewrites the config.
ID_CANDIDATES = ["Design", "design_id", "Design_ID", "DesignID", "id", "Name", "name"]
TARGET_CANDIDATES = [
    "Average Cd",
    "Cd",
    "cd",
    "Cd_mean",
    "Average_Cd",
    "drag_coefficient",
    "Cd_avg",
]
CATEGORY_CANDIDATES = ["Category", "category", "body_style", "Body Style", "class"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: str | Path | None, p: str | Path) -> Path:
    """Config paths are relative to the repo root, not the shell's cwd."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (Path(root) if root else _repo_root()) / path


def load_config(path: str | Path) -> dict:
    """Read a YAML config and resolve its data paths to absolute ones."""
    import yaml

    cfg_path = _resolve(None, path)
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping")

    root = _repo_root()
    d = cfg.setdefault("data", {})
    for key in ("labels_csv", "mesh_dir", "cache_dir"):
        if d.get(key):
            d[key] = str(_resolve(root, d[key]))
    t = cfg.setdefault("train", {})
    if t.get("out_dir"):
        t["out_dir"] = str(_resolve(root, t["out_dir"]))

    cfg["_config_path"] = str(cfg_path)
    return cfg


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


@dataclass
class LabelTable:
    """The labels CSV after column resolution."""

    targets: dict[str, float]
    categories: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    resolved: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    n_rows: int = 0

    @property
    def ids(self) -> list[str]:
        return list(self.targets)


def _pick_column(
    columns: list[str], configured: str | None, candidates: list[str], role: str
) -> tuple[str | None, list[str]]:
    """Resolve one column name. Returns (name, notes)."""
    notes: list[str] = []
    if configured and configured in columns:
        return configured, notes

    lowered = {c.strip().lower(): c for c in columns}
    if configured:
        hit = lowered.get(configured.strip().lower())
        if hit:
            notes.append(
                f"{role}: config says '{configured}' but the CSV has '{hit}' "
                f"-- set {role}_column to '{hit}'"
            )
            return hit, notes

    for cand in candidates:
        hit = lowered.get(cand.strip().lower())
        if hit:
            if configured:
                # A DIFFERENT column than the config asked for. Suggest it, but
                # do not adopt it: training on a silently substituted target is
                # the failure --check exists to catch. SUBSTITUTED marks it so
                # load_labels can refuse in strict mode.
                notes.append(
                    f"SUBSTITUTED {role}: config says {configured!r}, which is not "
                    f"in the CSV. Did you mean '{hit}'? Set {role}_column to it."
                )
                return None, notes
            notes.append(f"{role}: not configured; using '{hit}'")
            return hit, notes

    if configured:
        notes.append(f"{role}: no column matching {configured!r} or any known alias")
    return None, notes


def load_labels(cfg: dict, strict: bool = True) -> LabelTable:
    """Read labels_csv and resolve the id / target / category columns."""
    import pandas as pd

    d = cfg["data"]
    csv_path = Path(d["labels_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"labels_csv not found: {csv_path}\n"
            f"See data/README.md. For a no-download dev loop:\n"
            f"  python -m src.data.synthetic --n 300 --out data/synthetic"
        )

    df = pd.read_csv(csv_path)
    columns = [str(c) for c in df.columns]

    id_col, n1 = _pick_column(columns, d.get("id_column"), ID_CANDIDATES, "id")
    tgt_col, n2 = _pick_column(columns, d.get("target_column"), TARGET_CANDIDATES, "target")
    cat_col, n3 = _pick_column(
        columns, d.get("category_column"), CATEGORY_CANDIDATES, "category"
    )
    notes = n1 + n2 + n3

    table = LabelTable(
        targets={},
        columns=columns,
        resolved={"id": id_col, "target": tgt_col, "category": cat_col},
        notes=notes,
        n_rows=len(df),
    )

    if id_col is None or tgt_col is None:
        if strict:
            raise KeyError(
                "could not resolve the id and target columns in "
                f"{csv_path}\n  found: {columns}\n  " + "\n  ".join(notes)
            )
        return table

    sub = df[[id_col, tgt_col]].dropna()
    table.targets = {
        str(r[id_col]).strip(): float(r[tgt_col]) for _, r in sub.iterrows()
    }
    if cat_col is not None:
        cats = df[[id_col, cat_col]].dropna()
        table.categories = {
            str(r[id_col]).strip(): str(r[cat_col]).strip() for _, r in cats.iterrows()
        }
    return table


def available_meshes(cfg: dict) -> dict[str, Path]:
    """design_id -> mesh path, for every mesh actually on disk."""
    d = cfg["data"]
    mesh_dir = Path(d["mesh_dir"])
    if not mesh_dir.exists():
        return {}
    pattern = d.get("mesh_glob", "*.stl")
    return {p.stem.strip(): p for p in sorted(mesh_dir.glob(pattern))}


def matched_ids(cfg: dict, strict: bool = True) -> tuple[list[str], LabelTable, dict[str, Path]]:
    """Ids usable for training. Sorted, so deterministic.

    For the point-cloud path that means a label AND a mesh on disk. For the
    tabular path the CSV is all there is, so no mesh is required -- which is
    why both paths can share make_splits, and why Paramveer gets one splits
    file that is valid for either model.
    """
    table = load_labels(cfg, strict=strict)
    if str(cfg.get("data", {}).get("kind", "pointcloud")).lower() == "table":
        return sorted(table.targets), table, {}
    meshes = available_meshes(cfg)
    ids = sorted(set(table.targets) & set(meshes))
    return ids, table, meshes


# --------------------------------------------------------------------------
# Geometry -> point cloud
# --------------------------------------------------------------------------


def _mesh_seed(mesh_path: Path) -> int:
    """Per-mesh sampling seed, so a cache rebuild reproduces the same cloud."""
    digest = hashlib.blake2b(mesh_path.name.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def sample_mesh(mesh_path: Path, n_points: int, normalize: bool) -> np.ndarray:
    """Area-weighted surface sample of one mesh. Returns (n_points, 3) float32.

    Normalisation is computed from the MESH (bounding-box centre, max vertex
    radius), not from the sampled points, so it does not wobble with the
    sample. preprocess_spec() is the authoritative description of what this
    does -- keep the two in step.
    """
    import trimesh

    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):  # multi-solid STL
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh: trimesh.Trimesh = loaded

    if mesh.faces.shape[0] == 0:
        raise ValueError(f"{mesh_path} has no faces")

    seed = _mesh_seed(mesh_path)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    except TypeError:  # older trimesh has no seed kwarg
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, n_points)

    points = np.asarray(points, dtype=np.float64)
    if normalize:
        centre = mesh.bounds.mean(axis=0)
        radius = float(np.linalg.norm(mesh.vertices - centre, axis=1).max())
        points = (points - centre) / max(radius, 1e-9)

    return np.ascontiguousarray(points, dtype=np.float32)


def _cache_path(cfg: dict, design_id: str) -> Path:
    d = cfg["data"]
    tag = f"p{int(d['n_points'])}_{'unit' if d['normalize_geometry'] else 'raw'}"
    return Path(d["cache_dir"]) / PREPROCESS_VERSION / tag / f"{design_id}.npy"


def load_or_sample(cfg: dict, design_id: str, mesh_path: Path) -> np.ndarray:
    """Cached sample_mesh. The cache key covers n_points and normalisation."""
    d = cfg["data"]
    cache = _cache_path(cfg, design_id)
    if cache.exists():
        return np.load(cache)

    points = sample_mesh(mesh_path, int(d["n_points"]), bool(d["normalize_geometry"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Write through a handle: np.save(path) silently appends ".npy" when the
    # path does not already end in it, which would put the temp file somewhere
    # other than where the rename looks for it.
    tmp = cache.parent / (cache.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, points, allow_pickle=False)
    tmp.replace(cache)  # atomic, so a killed run never leaves a half file
    return points


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class DrivAerNetDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """design_id -> (point cloud, Cd). Caches sampled clouds to disk."""

    def __init__(self, cfg: dict, split: str = "train", ids: list[str] | None = None):
        self.cfg = cfg
        self.split = split

        all_ids, table, meshes = matched_ids(cfg)
        if ids is None:
            ids = make_splits(cfg, seed=int(cfg.get("seed", 0)))[split]

        missing = [i for i in ids if i not in meshes or i not in table.targets]
        if missing:
            raise KeyError(
                f"{len(missing)} ids in split '{split}' have no mesh or no label, "
                f"e.g. {missing[:5]}"
            )

        self.ids: list[str] = list(ids)
        self.meshes = {i: meshes[i] for i in self.ids}
        self.targets = np.array([table.targets[i] for i in self.ids], dtype=np.float32)
        self.categories = [table.categories.get(i, "") for i in self.ids]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        design_id = self.ids[index]
        points = load_or_sample(self.cfg, design_id, self.meshes[design_id])
        return points, np.float32(self.targets[index])

    def warm_cache(self, verbose: bool = True) -> float:
        """Sample every mesh once, up front. Returns seconds taken.

        Worth calling before the first epoch: with num_workers>0 the cache
        would otherwise be built concurrently on the first pass, which makes
        epoch-one timings meaningless and hammers the disk.
        """
        t0 = time.time()
        for n, design_id in enumerate(self.ids, 1):
            load_or_sample(self.cfg, design_id, self.meshes[design_id])
            if verbose and (n % 50 == 0 or n == len(self.ids)):
                print(f"    cached {n}/{len(self.ids)}", flush=True)
        return time.time() - t0


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def make_splits(cfg: dict, seed: int = 0) -> dict[str, list[str]]:
    """Random split by design id. Paramveer's harness also needs a split BY
    CATEGORY for the cross-category experiment -- make_category_splits below."""
    ids, _, _ = matched_ids(cfg)
    if not ids:
        raise RuntimeError(
            "no design has both a label and a mesh -- run --check to see why"
        )

    d = cfg["data"]
    val_frac = float(d.get("val_fraction", 0.15))
    test_frac = float(d.get("test_fraction", 0.15))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    shuffled = [ids[i] for i in order]

    n = len(shuffled)
    n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(
            f"val+test fractions leave no training data ({n} designs total)"
        )

    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test : n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val :]),
    }


def make_category_splits(
    cfg: dict, held_out: str | None = None, seed: int = 0
) -> dict[str, list[str]]:
    """Hold one entire car category out of training. This is the OOD split.

    train/val come from the remaining categories; test is the held-out
    category in full. Paramveer's harness needs this for the cross-category
    number and the OOD detection rate -- it is the split that shows whether
    the uncertainty head earns its keep.
    """
    ids, table, _ = matched_ids(cfg)
    if not table.categories:
        raise RuntimeError(
            "no category column resolved -- set data.category_column in the "
            "config (run --check to see the available columns)"
        )

    present = sorted({table.categories[i] for i in ids if i in table.categories})
    if held_out is None:
        if not present:
            raise RuntimeError("no categories found on the matched designs")
        held_out = present[-1]
    if held_out not in present:
        raise ValueError(f"category {held_out!r} not in {present}")

    test = sorted(i for i in ids if table.categories.get(i) == held_out)
    rest = [i for i in ids if table.categories.get(i) != held_out]
    if not rest:
        raise RuntimeError(f"holding out {held_out!r} leaves no training data")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rest))
    shuffled = [rest[i] for i in order]
    n_val = max(1, int(round(len(shuffled) * float(cfg["data"].get("val_fraction", 0.15)))))

    return {
        "test": test,
        "val": sorted(shuffled[:n_val]),
        "train": sorted(shuffled[n_val:]),
        "_held_out_category": [held_out],
    }


def write_splits(cfg: dict, out_path: Path, seed: int = 0, held_out: str | None = None) -> dict:
    """Dump both splits to one JSON for Paramveer's harness."""
    ids, table, _ = matched_ids(cfg)
    payload = {
        "config": cfg.get("_config_path"),
        "seed": seed,
        "n_designs": len(ids),
        "preprocess": preprocess_spec(cfg),
        "random": make_splits(cfg, seed=seed),
    }
    if table.categories:
        payload["held_out_category"] = make_category_splits(cfg, held_out, seed)
        payload["categories"] = sorted({table.categories[i] for i in ids if i in table.categories})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------


def check(config_path: str) -> int:
    """Sanity check before training. Returns a shell exit code."""
    print(f"config  {config_path}")
    cfg = load_config(config_path)
    d = cfg["data"]
    problems: list[str] = []

    print(f"  labels_csv  {d.get('labels_csv')}")
    print(f"  mesh_dir    {d.get('mesh_dir')}")
    print(f"  cache_dir   {d.get('cache_dir')}")

    # --- labels -----------------------------------------------------------
    print("\nlabels")
    try:
        table = load_labels(cfg, strict=False)
    except FileNotFoundError as exc:
        print(f"  MISSING: {exc}")
        return 1

    print(f"  rows        {table.n_rows}")
    print(f"  columns     {table.columns}")
    print(
        f"  resolved    id={table.resolved['id']!r}  "
        f"target={table.resolved['target']!r}  category={table.resolved['category']!r}"
    )
    for note in table.notes:
        print(f"  NOTE        {note}")
        if "no column matching" in note:
            problems.append(note)

    if table.resolved["id"] is None or table.resolved["target"] is None:
        print("\n  FIX the id/target column names in the config, then re-run --check.")
        return 1

    targets = np.array(list(table.targets.values()), dtype=np.float64)
    print(f"  labelled    {len(table.targets)} designs")
    print(
        f"  target      mean {targets.mean():.4f}  sd {targets.std():.4f}  "
        f"range [{targets.min():.4f}, {targets.max():.4f}]"
    )
    if targets.std() < 1e-6:
        problems.append("target column has no variance -- wrong column?")

    if table.categories:
        counts: dict[str, int] = {}
        for cat in table.categories.values():
            counts[cat] = counts.get(cat, 0) + 1
        print(f"  categories  {counts}")
    else:
        print("  categories  none (set data.category_column for per-category eval)")

    kind = str(d.get("kind", "pointcloud")).lower()
    if kind == "table":
        return _check_table(cfg, table, problems)

    # --- meshes -----------------------------------------------------------
    print("\nmeshes")
    meshes = available_meshes(cfg)
    print(f"  pattern     {d.get('mesh_glob')}")
    print(f"  on disk     {len(meshes)}")
    if not meshes:
        print(f"  MISSING: no files matching {d.get('mesh_glob')} in {d.get('mesh_dir')}")
        return 1

    ids = sorted(set(table.targets) & set(meshes))
    labels_only = sorted(set(table.targets) - set(meshes))
    meshes_only = sorted(set(meshes) - set(table.targets))
    print(f"  MATCHED     {len(ids)} designs with both a label and a mesh")
    if labels_only:
        print(f"  label only  {len(labels_only)}  e.g. {labels_only[:3]}")
    if meshes_only:
        print(f"  mesh only   {len(meshes_only)}  e.g. {meshes_only[:3]}")
    if not ids:
        problems.append(
            "zero overlap between CSV ids and mesh filenames -- the id column "
            "and the mesh stems do not agree (prefix? zero padding? extension?)"
        )
        print("\n  " + problems[-1])
        return 1
    if len(ids) < 50:
        print(f"  WARNING     only {len(ids)} designs; expect a weak R^2")

    # --- one real sample --------------------------------------------------
    print("\nsample")
    probe = ids[0]
    try:
        # Sample twice and time the second: the first call pays for `import
        # trimesh`, which would inflate the cache estimate by two orders of
        # magnitude on small meshes.
        points = sample_mesh(meshes[probe], int(d["n_points"]), bool(d["normalize_geometry"]))
        warm = ids[1] if len(ids) > 1 else probe
        t0 = time.time()
        sample_mesh(meshes[warm], int(d["n_points"]), bool(d["normalize_geometry"]))
        dt = time.time() - t0
    except Exception as exc:  # noqa: BLE001 - surface whatever trimesh says
        print(f"  FAILED on {probe}: {type(exc).__name__}: {exc}")
        return 1

    print(f"  design      {probe}  ({meshes[probe].name})")
    print(f"  shape       {points.shape}  {points.dtype}")
    print(f"  bounds      min {points.min(axis=0)}  max {points.max(axis=0)}")
    print(f"  centroid    {points.mean(axis=0)}")
    print(f"  radius max  {np.linalg.norm(points, axis=1).max():.4f}")
    print(f"  time        {dt * 1000:.0f} ms per mesh  ->  {dt * len(ids):.0f} s to cache all")

    if d["normalize_geometry"] and np.linalg.norm(points, axis=1).max() > 1.001:
        problems.append("normalised cloud exceeds the unit sphere -- check sample_mesh")

    cached = len(list(Path(d["cache_dir"]).rglob("*.npy"))) if Path(d["cache_dir"]).exists() else 0
    print(f"  cache       {cached} .npy already on disk")

    # --- splits -----------------------------------------------------------
    print("\nsplits")
    try:
        splits = make_splits(cfg, seed=int(cfg.get("seed", 0)))
        print(f"  random      " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"make_splits failed: {exc}")
        print(f"  FAILED      {exc}")
    if table.categories:
        try:
            cat_splits = make_category_splits(cfg, seed=int(cfg.get("seed", 0)))
            held = cat_splits["_held_out_category"][0]
            print(
                f"  by category  held out {held!r}  "
                + "  ".join(
                    f"{k}={len(v)}" for k, v in cat_splits.items() if not k.startswith("_")
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  by category  FAILED  {exc}")

    print("\npreprocess spec written into every checkpoint:")
    for k, v in preprocess_spec(cfg).items():
        print(f"  {k:18} {v}")

    print()
    if problems:
        print(f"NOT READY -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"READY -- {len(ids)} designs, {int(d['n_points'])} points each.")
    return 0


def _check_table(cfg: dict, table: LabelTable, problems: list[str]) -> int:
    """Delegate the tabular --check. Imported late to avoid a cycle."""
    from src.data.tabular import check_table

    return check_table(cfg, table, problems)


def main() -> int:
    ap = argparse.ArgumentParser(description="DrivAerNet stage-1 data loader")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--check", action="store_true", help="sanity check and exit")
    ap.add_argument("--warm-cache", action="store_true", help="sample every mesh now")
    ap.add_argument("--dump-splits", metavar="PATH", help="write splits JSON for the harness")
    ap.add_argument("--held-out-category", default=None)
    args = ap.parse_args()

    if args.check:
        return check(args.config)

    cfg = load_config(args.config)
    if args.warm_cache:
        ids, _, _ = matched_ids(cfg)
        ds = DrivAerNetDataset(cfg, ids=ids)
        print(f"caching {len(ds)} designs...")
        print(f"done in {ds.warm_cache():.1f} s")
        return 0

    if args.dump_splits:
        payload = write_splits(
            cfg,
            Path(args.dump_splits),
            seed=int(cfg.get("seed", 0)),
            held_out=args.held_out_category,
        )
        print(f"wrote {args.dump_splits}  ({payload['n_designs']} designs)")
        for name, split in payload["random"].items():
            print(f"  random/{name:6} {len(split)}")
        if "held_out_category" in payload:
            held = payload["held_out_category"]["_held_out_category"][0]
            print(f"  held out category: {held}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
