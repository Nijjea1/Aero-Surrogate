"""Labels, splits and model inputs for the harness.  OWNER: Paramveer.

Reads the DrivAerNet parametric table itself, with the csv module, so grading
does not depend on the training code's data loader. It accepts either the raw
download (ParametricModels/DrivAerNet_ParametricData.csv) or the prepared
data/drivaernet/parametric.csv that src.data.drivaernet --fetch writes.

The splits are the ones the model was trained with: runs/splits.json from
`python -m src.data.dataset --dump-splits` when it exists. Without it,
make_splits / make_category_splits rebuild them with the same algorithm, and
the tests check the two agree.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ID_COLUMNS = ("Experiment", "Design", "design_id")
TARGET = "Average Cd"
LABEL_STD = "Std Cd"
CATEGORY_PREFIX = {"F": "fastback", "N": "notchback", "E": "estateback"}

DEFAULT_DATA = ("data/drivaernet/parametric.csv", "data/drivaernet/raw/DrivAerNet_ParametricData.csv")


@dataclass
class Table:
    """One row per design, columns as float arrays, aligned with `ids`."""

    ids: list[str]
    columns: dict[str, np.ndarray]
    categories: list[str]
    path: str = ""
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {d: i for i, d in enumerate(self.ids)}

    def rows(self, ids) -> np.ndarray:
        missing = [d for d in ids if d not in self._index]
        if missing:
            raise KeyError(f"{len(missing)} split ids are not in {self.path}: {missing[:3]}")
        return np.array([self._index[d] for d in ids], dtype=np.int64)

    def column(self, name: str, ids=None) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(f"column {name!r} not in {self.path}")
        col = self.columns[name]
        return col if ids is None else col[self.rows(ids)]

    def category(self, ids) -> np.ndarray:
        cats = np.asarray(self.categories)
        return cats[self.rows(ids)]


def find_data(explicit: str | None = None) -> Path:
    for candidate in ([explicit] if explicit else DEFAULT_DATA):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "no DrivAerNet table found. Either run `python -m src.data.drivaernet --fetch` "
        "or pass --data PATH. Looked for: " + ", ".join([explicit] if explicit else DEFAULT_DATA)
    )


def load_table(path) -> Table:
    """Read the table. Rows without a Cd label are dropped, ids are unique."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        id_col = next((c for c in ID_COLUMNS if c in header), None)
        if id_col is None or TARGET not in header:
            raise KeyError(f"{path} needs an id column {ID_COLUMNS} and {TARGET!r}; has {header}")
        raw = [r for r in reader if (r.get(TARGET) or "").strip()]

    ids = [r[id_col].strip() for r in raw]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate design ids")

    columns = {}
    for name in header:
        if name in (id_col, "Category"):
            continue
        try:
            columns[name] = np.array([float(r[name]) for r in raw], dtype=np.float64)
        except (TypeError, ValueError):
            continue  # a text column; nothing here needs it

    if "Category" in header:
        categories = [r["Category"].strip() for r in raw]
    else:
        categories = [CATEGORY_PREFIX.get(d.split("_")[0], "") for d in ids]
    if any(not c for c in categories):
        raise ValueError(f"{path}: some ids have no recognisable body style")
    return Table(ids=ids, columns=columns, categories=categories, path=str(path))


def design_matrix(table: Table, ids, param_columns) -> np.ndarray:
    """Model input in the checkpoint's column order. `cat_<style>` columns are
    1.0 for designs of that body style, so the raw table works as well as the
    prepared one."""
    rows = table.rows(ids)
    cats = np.asarray(table.categories)[rows]
    out = np.empty((len(rows), len(param_columns)), dtype=np.float64)
    for j, name in enumerate(param_columns):
        if name.startswith("cat_"):
            out[:, j] = (cats == name[len("cat_"):]).astype(np.float64)
        else:
            out[:, j] = table.column(name)[rows]
    return out


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def make_splits(ids, seed: int = 0, val_fraction: float = 0.15, test_fraction: float = 0.15) -> dict:
    """The random split, by the same algorithm as src.data.dataset.make_splits."""
    ids = sorted(ids)
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n = len(shuffled)
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(f"val+test fractions leave no training data ({n} designs)")
    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val:]),
    }


def make_category_splits(ids, categories: dict, held_out: str | None = None,
                         seed: int = 0, val_fraction: float = 0.15) -> dict:
    """One body style held out entirely, by the same algorithm as
    src.data.dataset.make_category_splits. Default held out: the last style
    alphabetically, which is notchback."""
    ids = sorted(ids)
    present = sorted({categories[i] for i in ids})
    held_out = held_out or present[-1]
    if held_out not in present:
        raise ValueError(f"category {held_out!r} not in {present}")
    rest = [i for i in ids if categories[i] != held_out]
    order = np.random.default_rng(seed).permutation(len(rest))
    shuffled = [rest[i] for i in order]
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return {
        "test": sorted(i for i in ids if categories[i] == held_out),
        "val": sorted(shuffled[:n_val]),
        "train": sorted(shuffled[n_val:]),
        "held_out_category": held_out,
    }


def load_splits(path) -> dict:
    """Load a direct split export or the complete training handover package.

    Both formats normalise to random/category groups. Reject malformed,
    duplicated or overlapping IDs before they can inflate evaluation scores.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split file must contain a JSON object")
    if "stage1" in payload:
        try:
            payload = payload["stage1"]["parameters"]
        except (KeyError, TypeError):
            raise ValueError("handover split file has no stage1.parameters split object") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("random"), dict):
        raise ValueError("split file needs a random split object (or stage1.parameters.random)")

    def checked(group, name):
        if not isinstance(group, dict):
            raise ValueError(f"{name} split must be an object")
        out = {}
        for key in ("train", "val", "test"):
            ids = group.get(key)
            if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError(f"{name}.{key} must be a nonempty list of design IDs")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name}.{key} contains duplicate design IDs")
            out[key] = list(ids)
        check_disjoint(out)
        return out

    out = {"random": checked(payload["random"], "random"), "source": str(path), "seed": payload.get("seed")}
    category = payload.get("held_out_category", payload.get("category"))
    if category is not None:
        cat = checked(category, "category")
        held = category.get("held_out_category", category.get("_held_out_category"))
        if isinstance(held, list) and len(held) == 1:
            held = held[0]
        if not isinstance(held, str) or not held:
            raise ValueError("category split needs a held_out_category name")
        cat["held_out_category"] = held
        out["category"] = cat
    return out


def build_splits(table: Table, seed: int = 0, val_fraction: float = 0.15,
                 test_fraction: float = 0.15, held_out: str | None = None) -> dict:
    categories = dict(zip(table.ids, table.categories))
    return {
        "random": make_splits(table.ids, seed, val_fraction, test_fraction),
        "category": make_category_splits(table.ids, categories, held_out, seed, val_fraction),
        "source": f"rebuilt with seed {seed}",
        "seed": seed,
    }


def check_disjoint(split: dict) -> None:
    """A design in both train and test would inflate every number we report."""
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    for a, b, name in ((train, test, "train/test"), (train, val, "train/val"), (val, test, "val/test")):
        if a & b:
            raise ValueError(f"{len(a & b)} designs are in both {name}: {sorted(a & b)[:3]}")
