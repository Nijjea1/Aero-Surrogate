"""Design parameters -> Cd, straight from the labels CSV.  OWNER: Avneet.

The gate-insurance path. DrivAerNet publishes the parametric design variables
its geometries were morphed from, and src.contracts.PARAM_BOUNDS is drawn from
that same vocabulary. If those columns are in the CSV we can train a real
model, on real labels, with a real held-out R^2, without downloading a single
mesh.

Three reasons it is worth the ~100 lines:

  1. It de-risks Oct 15. The point-cloud path depends on a multi-GB download
     landing AND the loader working first time; this path depends on a KB CSV.
  2. It is the input signature the agent actually proposes in. The service gets
     `params`, and this model consumes `params` -- no mesh round trip, so no
     train/serve geometry mismatch.
  3. It gives the point-cloud model something honest to be measured against.
     A point cloud should beat raw parameters. If it does not, the geometry
     encoder is not earning its place and we should know that early.

It is NOT a replacement for the real surrogate. It cannot predict a pressure
field, it cannot see a shape it has no parameter for, and it only works inside
the parameterisation. Stage 2 is still PhysicsNeMo.

    python -m src.data.dataset --check --config configs/tabular.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.dataset import LabelTable, load_labels, make_splits, matched_ids

try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment, misc]


@dataclass
class ParamTable:
    """Resolved design-parameter matrix."""

    ids: list[str]
    x: np.ndarray                      # (n, p) float32
    y: np.ndarray                      # (n,)   float32
    columns: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def param_columns(cfg: dict) -> list[str]:
    """The parameter names we want, in order. Order is part of the contract."""
    from src.contracts import PARAM_NAMES

    return list(cfg.get("data", {}).get("param_columns") or PARAM_NAMES)


def load_param_table(cfg: dict, strict: bool = True) -> ParamTable:
    """Read the design parameters and the target out of labels_csv."""
    import pandas as pd

    d = cfg["data"]
    df = pd.read_csv(d["labels_csv"])
    columns = [str(c) for c in df.columns]
    lowered = {c.strip().lower(): c for c in columns}

    table: LabelTable = load_labels(cfg, strict=strict)
    id_col = table.resolved.get("id")
    tgt_col = table.resolved.get("target")

    wanted = param_columns(cfg)
    resolved: list[str] = []
    missing: list[str] = []
    notes: list[str] = []
    for name in wanted:
        hit = lowered.get(name.strip().lower())
        if hit is None:
            missing.append(name)
        else:
            resolved.append(hit)
            if hit != name:
                notes.append(f"param '{name}' found as '{hit}'")

    if missing and strict:
        raise KeyError(
            f"{len(missing)} design parameter(s) not in {d['labels_csv']}: {missing}\n"
            f"  available columns: {columns}\n"
            f"  Set data.param_columns in the config to the real names, or use "
            f"the point-cloud path instead."
        )
    if not resolved:
        return ParamTable(ids=[], x=np.zeros((0, 0), np.float32),
                          y=np.zeros((0,), np.float32), columns=[],
                          missing=missing, notes=notes)

    if id_col is None or tgt_col is None:
        raise KeyError("id/target columns unresolved -- run --check first")

    sub = df[[id_col, tgt_col, *resolved]].dropna()
    ids = [str(v).strip() for v in sub[id_col].tolist()]
    x = sub[resolved].to_numpy(dtype=np.float32)
    y = sub[tgt_col].to_numpy(dtype=np.float32)

    return ParamTable(ids=ids, x=x, y=y, columns=resolved, missing=missing, notes=notes)


class DesignParamsDataset(_TorchDataset):  # type: ignore[misc, valid-type]
    """design_id -> (parameter vector, Cd). No meshes, no cache, no disk."""

    def __init__(self, cfg: dict, split: str = "train", ids: list[str] | None = None):
        self.cfg = cfg
        self.split = split

        table = load_param_table(cfg)
        by_id = {design_id: i for i, design_id in enumerate(table.ids)}
        if ids is None:
            ids = make_splits(cfg, seed=int(cfg.get("seed", 0)))[split]

        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(
                f"{len(missing)} ids in split '{split}' have no parameter row, "
                f"e.g. {missing[:5]}"
            )

        rows = [by_id[i] for i in ids]
        self.ids = list(ids)
        self.columns = table.columns
        self.x = table.x[rows]
        self.targets = table.y[rows]

        _, label_table, _ = matched_ids(cfg)
        self.categories = [label_table.categories.get(i, "") for i in self.ids]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.x[index], np.float32(self.targets[index])

    def warm_cache(self, verbose: bool = False) -> float:
        """Nothing to cache -- the whole dataset is already in memory."""
        return 0.0


def check_table(cfg: dict, label_table: LabelTable, problems: list[str]) -> int:
    """The --check body for the tabular path. Called from dataset.check()."""
    print("\nparameters")
    wanted = param_columns(cfg)
    print(f"  wanted      {wanted}")

    table = load_param_table(cfg, strict=False)
    print(f"  resolved    {table.columns}")
    for note in table.notes:
        print(f"  NOTE        {note}")

    if table.missing:
        print(f"  MISSING     {table.missing}")
        problems.append(
            f"{len(table.missing)} design parameter(s) absent from the CSV: "
            f"{table.missing} -- set data.param_columns, or use the "
            f"point-cloud path"
        )
        print("\nNOT READY -- " + problems[-1])
        return 1

    print(f"  rows        {len(table.ids)} with every parameter and a label")
    if len(table.ids) < 50:
        print(f"  WARNING     only {len(table.ids)} rows; expect a weak R^2")

    from src.contracts import PARAM_BOUNDS

    print("\n  column                 min        max    bounds (contracts.py)")
    for i, name in enumerate(wanted):
        lo, hi = PARAM_BOUNDS.get(name, (float("nan"), float("nan")))
        col = table.x[:, i]
        flag = ""
        if col.min() < lo - 1e-6 or col.max() > hi + 1e-6:
            flag = "  <-- OUTSIDE BOUNDS"
            problems.append(
                f"'{name}' data range [{col.min():.4f}, {col.max():.4f}] falls "
                f"outside PARAM_BOUNDS [{lo}, {hi}] -- the agent can never "
                f"propose there, so that training data is unreachable"
            )
        print(
            f"  {name:20} {col.min():9.4f}  {col.max():9.4f}    "
            f"[{lo}, {hi}]{flag}"
        )

    print("\nsplits")
    try:
        splits = make_splits(cfg, seed=int(cfg.get("seed", 0)))
        print("  random      " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"make_splits failed: {exc}")
        print(f"  FAILED      {exc}")

    if label_table.categories:
        from src.data.dataset import make_category_splits

        try:
            cat = make_category_splits(cfg, seed=int(cfg.get("seed", 0)))
            held = cat["_held_out_category"][0]
            print(
                f"  by category  held out {held!r}  "
                + "  ".join(f"{k}={len(v)}" for k, v in cat.items() if not k.startswith("_"))
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  by category  FAILED  {exc}")

    from src.data.dataset import preprocess_spec

    print("\npreprocess spec written into every checkpoint:")
    for k, v in preprocess_spec(cfg).items():
        print(f"  {k:18} {v}")

    print()
    if problems:
        print(f"NOT READY -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"READY -- {len(table.ids)} designs, {len(table.columns)} parameters each.")
    return 0
