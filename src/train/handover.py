"""Package everything the evaluation harness needs.  OWNER: Avneet.

    python -m src.train.handover                 # write the package
    python -m src.train.handover --verify        # check it loads

This is the Oct 22-25 deliverable and the last item in track A. The plan is
explicit that I do not grade my own model: Paramveer's harness produces every
number in the submission, and my job ends at handing over checkpoints and
splits. The person who trains a model is the worst person to grade it.

So this writes ONE directory containing:

    splits.json     every split, for both stages, by design id and by run
    manifest.json   every checkpoint: version, metrics, preprocessing spec,
                    whether it carries an OOD detector and in what space
    README.md       how to load each kind and what not to trust

WHY A MANIFEST RATHER THAN "the checkpoints are in runs/"

There are now four kinds of checkpoint in this repo and they are NOT
interchangeable:

    params-v1      design parameters -> Cd      the gate result, R^2 0.9132
    pointcloud-v1  point cloud -> Cd            proven plumbing, R^2 -0.73
    fields-v1      geometry -> Cp per point     stage 2
    ensemble       5x params-v1                 the stage-1 result to report

They take different inputs, normalise differently, and three of the four carry
an OOD detector that measures a different space from the others. Handing over a
directory and hoping would guarantee somebody evaluates the -0.73 model and
reports it, or scores a parameter vector with a feature-space detector. The
manifest makes each one's contract explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import torch

from src.contracts import SCHEMA_VERSION
from src.data.dataset import load_config, make_category_splits, make_splits, preprocess_spec
from src.train.train import pick_device

# Config -> what the checkpoint is for. Order is the order they are reported.
KNOWN_RUNS: list[dict] = [
    {
        "name": "stage1-parameters",
        "config": "configs/drivaernet.yaml",
        "run": "drivaernet-random",
        "role": "THE STAGE-1 RESULT. Design parameters -> Cd on real "
                "DrivAerNet, 4,165 designs.",
        "report": True,
    },
    {
        "name": "stage1-parameters-ood-split",
        "config": "configs/drivaernet.yaml",
        "run": "drivaernet-category",
        "role": "Same model, trained with one body style held out. The "
                "cross-category generalisation experiment.",
        "report": True,
    },
    {
        "name": "stage1-ensemble",
        "config": "configs/drivaernet.yaml",
        "run": "drivaernet-ensemble-random",
        "role": "Five-member deep ensemble. Historical test R^2 0.9132; "
                "not the served single model. Report only with independent "
                "grading and a distinct artifact identity.",
        "report": True,
        "ensemble": True,
    },
    {
        "name": "stage2-fields",
        "config": "configs/fields.yaml",
        "run": "fields",
        "role": "STAGE 2. Geometry -> Cp at every surface point, on "
                "DrivAerML. Integrate for drag and lift.",
        "report": True,
    },
    {
        "name": "stage1-geometry",
        "config": "configs/drivaerml.yaml",
        "run": "drivaerml-random",
        "role": "Point cloud -> Cd on 30 DrivAerML cars. DO NOT REPORT: 18 "
                "training samples, test R^2 is negative. It exists to prove "
                "the geometry pipeline runs end to end.",
        "report": False,
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    """Write UTF-8 explicitly.

    Path.write_text() uses the platform's locale encoding, which is cp1252 on
    a default Windows install, and the README and manifest contain the
    characters 'σ', '²', '—' and '→'. Without this the handover crashes on
    Windows with a UnicodeEncodeError. Newlines are normalised too, so a
    checked-in manifest does not churn between platforms.
    """
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_ckpt(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: {path} unreadable: {exc}")
        return None


def _describe(entry: dict, repo: Path) -> dict | None:
    run_dir = repo / "runs" / entry["run"]
    out: dict = {
        "name": entry["name"],
        "role": entry["role"],
        "safe_to_report": entry["report"],
        "config": entry["config"],
        "run_dir": str(run_dir.relative_to(repo)),
    }

    if entry.get("ensemble"):
        report = run_dir / "ensemble.json"
        if not report.exists():
            return None
        payload = json.loads(report.read_text())
        members = payload.get("members", [])
        out.update({
            "kind": "ensemble",
            "members": members,
            # One hash per member: an ensemble is five files, so a single
            # sha256 would be meaningless. Pin all of them.
            "member_sha256": {
                str(Path(m).relative_to(repo) if Path(m).is_absolute()
                    and Path(m).is_relative_to(repo) else m): _sha256(Path(m))
                for m in members if Path(m).exists()
            },
            "n_members": payload.get("n_members"),
            "metrics_scope": "ensemble of members (NOT the single model)",
            "val_metrics": payload.get("val"),
            "test_metrics": payload.get("test"),
            "load_with": "src.train.ensemble.Ensemble.load([...member paths])",
            "note": "Predict with Ensemble.predict(loader); it returns mean, "
                    "aleatoric and epistemic separately. Report coverage under "
                    "the TOTAL sigma.",
        })
        return out

    ckpt = _load_ckpt(run_dir / "best.pt")
    if ckpt is None:
        return None

    pre = ckpt.get("preprocess", {})
    # Only the contract's own version is current. This used to accept
    # "1.0.0" as well, which marked pre-contract-change checkpoints as fine.
    stamped = ckpt.get("schema_version")
    out["schema_version"] = stamped
    out["schema_current"] = stamped == SCHEMA_VERSION
    if not out["schema_current"]:
        out["schema_problem"] = (
            f"stamped {stamped!r} but the contract is {SCHEMA_VERSION!r}; "
            f"retrain, do not restamp")
    ood = ckpt.get("ood")
    ckpt_file = run_dir / "best.pt"
    out.update({
        "kind": pre.get("version", "unknown"),
        "checkpoint": str(ckpt_file.relative_to(repo)),
        # Full SHA-256, not a prefix: the service pins it at /health.
        "sha256": _sha256(ckpt_file),
        "bytes": ckpt_file.stat().st_size,
        "version": ckpt.get("version"),
        "git_commit": ckpt.get("git_commit"),
        "seed": (ckpt.get("config") or {}).get("seed"),
        "trained_body_styles": ckpt.get("trained_body_styles"),
        "schema_version": ckpt.get("schema_version"),
        "n_params": ckpt.get("n_params"),
        "preprocess": pre,
        # Single-model metrics. The ensemble's are a separate manifest entry
        # with its own val/test block -- never merged with these.
        "metrics_scope": "single model",
        "val_metrics": ckpt.get("val_metrics"),
        "test_metrics": ckpt.get("test_metrics"),
        "held_out_category": ckpt.get("held_out_category"),
        "splits": ckpt.get("splits"),
        # A detector from before the `space` field existed cannot be
        # attributed to a space, so it is reported as unknown and NOT
        # trustworthy. Defaulting it to "parameters" would be the one wrong
        # answer: it would claim the 100%-detection result for a detector that
        # might be the 26% one.
        "ood_detector": None if ood is None else {
            "space": ood.get("space") or "unknown",
            "method": ood.get("method") or "unknown",
            "diagonal": ood.get("diagonal"),
            "threshold": ood.get("threshold"),
            "quantile": ood.get("quantile"),
            "n_train": ood.get("n_train"),
            "trustworthy": ood.get("space") == "parameters",
            "stale": ood.get("space") is None or ood.get("method") is None,
        },
    })
    if pre.get("version") == "fields-v1":
        out["load_with"] = "src.train.train_fields.build_model_from_cfg(ckpt['config'])"
        out["regions"] = ckpt.get("regions")
    else:
        out["load_with"] = "src.train.model.build_model(ckpt['config'])"
    return out


def build_splits(repo: Path) -> dict:
    """Every split the harness needs, for both stages."""
    payload: dict = {"generated": date.today().isoformat(), "stage1": {}, "stage2": {}}

    for name, config in (("parameters", "configs/drivaernet.yaml"),
                         ("geometry", "configs/drivaerml.yaml")):
        path = repo / config
        if not path.exists():
            continue
        try:
            cfg = load_config(path)
            seed = int(cfg.get("seed", 0))
            entry: dict = {
                "config": config,
                "seed": seed,
                "preprocess": preprocess_spec(cfg),
                "random": make_splits(cfg, seed=seed),
            }
            try:
                cat = make_category_splits(cfg, seed=seed)
                entry["held_out_category"] = cat
            except Exception as exc:  # noqa: BLE001
                entry["held_out_category"] = None
                entry["held_out_category_note"] = str(exc)
            payload["stage1"][name] = entry
        except Exception as exc:  # noqa: BLE001
            payload["stage1"][name] = {"error": str(exc), "config": config}

    # Stage 2 splits are by RUN, not by design id.
    fields_cfg = repo / "configs/fields.yaml"
    if fields_cfg.exists():
        from src.data.fields import field_splits

        cfg = load_config(fields_cfg)
        root = Path(cfg["data"]["root"])
        runs = sorted(int(p.stem.split("_")[1])
                      for p in (root / "fields").glob("*.npz"))
        if runs:
            payload["stage2"]["fields"] = {
                "config": "configs/fields.yaml",
                "seed": int(cfg.get("seed", 0)),
                "split_by": "run, never by point -- each car holds 100k query "
                            "cells, so splitting points would put the same "
                            "geometry in train and test",
                "runs": field_splits(runs,
                                     val=int(cfg["data"].get("val_runs", 5)),
                                     test=int(cfg["data"].get("test_runs", 5)),
                                     seed=int(cfg.get("seed", 0))),
            }
    return payload


README = """# Handover — checkpoints and splits

Generated by `python -m src.train.handover`. Avneet's track-A output, for the
evaluation harness. **I do not produce the submission numbers** — this is the
input to the harness that does.

## What to report, and what not to

`manifest.json` marks every checkpoint with `safe_to_report`. One is **false**:
`stage1-geometry` trained on 18 DrivAerML cars and has a negative test R². It
exists to prove the geometry pipeline runs, not to be a result.

The release headline must describe the artifact actually served. The current
single model reports historical test R² 0.8750; a separate five-member ensemble
reports 0.9132. Do not mix these identities.

The dataset's `Std Cd` describes solver coefficient variability, not a proven
error variance of the averaged regression label. The historical ratio
`1 - mean(Std Cd²)/var(Cd)` is not a validated R² ceiling or evidence that
remaining prediction error is irreducible. Grade held-out R², MAE and interval
coverage directly.

## Four kinds of checkpoint, not interchangeable

| kind | input | load with |
|---|---|---|
| `params-v1` | 16 morph params + body style (19) | `src.train.model.build_model` |
| `pointcloud-v1` | (N, 3) sampled surface points | `src.train.model.build_model` |
| `fields-v1` | geometry cloud + query points | `src.train.train_fields.build_model_from_cfg` |
| `ensemble` | as `params-v1`, five of them | `src.train.ensemble.Ensemble.load` |

Each carries its own `preprocess` dict. **Read it; do not hardcode.** A model
fed input built to the wrong recipe returns a confident wrong number with no
error and no failing test.

## OOD detectors: check `space` before using one

`manifest.json` reports each detector's `space` and a `trustworthy` flag.

Only `space == "parameters"` is trustworthy. Measured:

| signal | in distribution | held-out body style |
|---|---|---|
| parameter-space distance | FPR 0.3% | **100% caught** |
| feature-space distance | FPR 3.3% | 26% (best of 8 methods) |
| heteroscedastic σ | coverage 0.64 | 0.49 |
| ensemble disagreement | coverage 0.79 | 0.44 |

Three of four fail. The pattern: OOD detection works when the novelty is
explicit in the input, and fails when the model must notice it from its own
behaviour. Do not score a parameter vector with a feature-space detector —
`src/service/predictor.py` already refuses this.

## Stage 2

`stage2-fields` predicts Cp at every surface point. **Judge it by integrated
Cd, not by Cp R².** A model predicting the mean Cp everywhere gets a
respectable MAE and integrates to *zero* drag, because the normals cancel.
There is a test asserting that.

Measured on 5 held-out cars: Cp R² 0.2984, integrated Cd 0.2536 vs true 0.2078.
It also fills `Evaluation.pressure_summary` — area-weighted mean Cp per
contract region, using the face→region index from
`src/data/preprocess.py --combine`.

## Splits

`splits.json` has all of them. Stage 1 is split by **design id**, both randomly
and with one body style held out. Stage 2 is split by **run** — each car holds
100k query cells, so splitting points would put the same geometry in train and
test and score interpolation within a shape already seen.

Reproduce any of them with the `seed` and `config` recorded alongside.
"""


def cmd_build(repo: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(repo)
    _write(out_dir / "splits.json", json.dumps(splits, indent=2))

    entries, missing = [], []
    for entry in KNOWN_RUNS:
        described = _describe(entry, repo)
        if described is None:
            missing.append(entry["name"])
        else:
            entries.append(described)

    manifest = {
        "generated": date.today().isoformat(),
        "owner": "Avneet (track A: data and training)",
        "note": "Avneet does not grade his own model. These are inputs to the "
                "evaluation harness, which produces the submission numbers.",
        "checkpoints": entries,
        "missing": missing,
    }
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2))
    _write(out_dir / "README.md", README)

    print(f"handover -> {out_dir}\n")
    print(f"{'name':32} {'kind':14} {'report':>7}  metric")
    for e in entries:
        t = e.get("test_metrics") or {}
        metric = (f"R2 {t['r2']:.4f}" if "r2" in t else
                  f"Cp R2 {t['cp_r2']:.4f}" if "cp_r2" in t else "-")
        print(f"  {e['name']:30} {e['kind']:14} "
              f"{'YES' if e['safe_to_report'] else 'no':>7}  {metric}")
    if missing:
        print(f"\nnot found (train them first): {missing}")

    print(f"\nsplits:")
    for stage, group in (("stage1", splits["stage1"]), ("stage2", splits["stage2"])):
        for name, entry in group.items():
            if "random" in entry:
                counts = "  ".join(f"{k}={len(v)}" for k, v in entry["random"].items())
                held = entry.get("held_out_category")
                extra = (f"  held-out-category: {held['_held_out_category'][0]}"
                         if held else "  (no category split: " +
                         entry.get("held_out_category_note", "n/a")[:40] + ")")
                print(f"  {stage}/{name:12} {counts}{extra}")
            elif "runs" in entry:
                counts = "  ".join(f"{k}={len(v)}" for k, v in entry["runs"].items())
                print(f"  {stage}/{name:12} {counts}  (by run)")
    return 0


def cmd_verify(repo: Path, out_dir: Path) -> int:
    """Load every reportable checkpoint the way the README says to."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"{manifest_path} missing -- build the handover first")
        return 1
    manifest = json.loads(manifest_path.read_text())

    problems: list[str] = []
    print("verifying every checkpoint loads as documented\n")
    for e in manifest["checkpoints"]:
        name = e["name"]
        try:
            if e["kind"] == "ensemble":
                from src.train.ensemble import Ensemble

                ens = Ensemble.load([Path(p) for p in e["members"]],
                                    device=torch.device("cpu"))
                print(f"  {name:30} OK  {len(ens.models)} members")
                continue

            ckpt = torch.load(repo / e["checkpoint"], map_location="cpu",
                              weights_only=False)
            if e["kind"] == "fields-v1":
                from src.train.train_fields import build_model_from_cfg

                model = build_model_from_cfg(ckpt["config"])
                model.freeze_mean_path()
            else:
                from src.train.model import build_model

                model = build_model(ckpt["config"])
                try:
                    model.freeze_mean_path()
                except Exception:  # noqa: BLE001
                    pass
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            n = sum(p.numel() for p in model.parameters())
            ood = e.get("ood_detector")
            tag = (f"ood={ood['space']}/{ood['method']}" if ood else "no ood")
            print(f"  {name:30} OK  {n:>9,} params  {tag}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  {name:30} FAILED  {type(exc).__name__}: {exc}")

    # An empty package is NOT ready. It used to print
    # "READY -- all 0 checkpoints load", which is true and useless.
    entries = manifest["checkpoints"]
    if not entries:
        print("NOT READY -- the manifest describes no checkpoints at all.")
        print("  Train them first, then rebuild: python -m src.train.handover")
        return 1
    if manifest.get("missing"):
        print(f"NOT READY -- {len(manifest['missing'])} expected run(s) absent: "
              f"{manifest['missing']}")
        return 1

    stale = [e["name"] for e in entries
             if e.get("schema_current") is False]
    if stale:
        print(f"NOT READY -- {len(stale)} checkpoint(s) carry a schema stamp that "
              f"disagrees with contracts.SCHEMA_VERSION {SCHEMA_VERSION!r}:")
        for name in stale:
            e = next(x for x in entries if x["name"] == name)
            print(f"  - {name}: {e.get('schema_problem')}")
        problems.extend(stale)

    print()
    if problems:
        print(f"NOT READY -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"READY -- all {len(entries)} checkpoints load as documented, "
          f"schema {SCHEMA_VERSION}.")
    return 0


def cmd_benchmark(repo: Path, out_dir: Path, reps: int = 50) -> int:
    """Measure inference latency honestly, including preprocessing.

    Include preprocessing when reporting end-to-end model latency, and keep
    comparisons against assumed CFD hours separate. The three model kinds are not
    comparable on a forward pass alone:

      params-v1      no geometry step at all. The agent proposes 19 numbers
                     and the model consumes them.
      pointcloud-v1  needs a mesh sampled first, and on a 753k-face DrivAerML
                     surface that sampling dominates the forward pass by orders
                     of magnitude.
      fields-v1      same mesh step, then a much larger forward pass.

    Quoting the forward pass alone would be the dishonest version of this
    number, so each row below reports preprocessing and inference separately
    and the total is what the speedup is computed from.

    The CFD reference is DrivAerML's own cost: the plan cites roughly 8 hours
    per case on 256 cores.
    """
    import numpy as np

    CFD_SECONDS = 8 * 3600.0
    device = pick_device()
    rows: list[dict] = []
    print(f"inference latency, device {device}, {reps} reps\n")

    # --- params-v1 --------------------------------------------------------
    params_run = repo / "runs" / "drivaernet-random" / "best.pt"
    if params_run.exists():
        from src.contracts import PARAM_NAMES, default_params
        from src.train.model import build_model, split_prediction

        ckpt = torch.load(params_run, map_location=device, weights_only=False)
        model = build_model(ckpt["config"])
        try:
            model.freeze_mean_path()
        except Exception:  # noqa: BLE001
            pass
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        cols = ckpt["preprocess"]["param_columns"]
        mean = np.asarray(ckpt["norm_stats"]["param_mean"])
        std = np.asarray(ckpt["norm_stats"]["param_std"])

        design = default_params()
        import time

        def once() -> None:
            row = [[(1.0 if c == "cat_fastback" else 0.0) if c.startswith("cat_")
                    else design[c] for c in cols]]
            x = (np.asarray(row, dtype=np.float64) - mean) / std
            with torch.no_grad():
                split_prediction(model(torch.tensor(x, dtype=torch.float32,
                                                    device=device)))

        once()
        t0 = time.perf_counter()
        for _ in range(reps):
            once()
        total = (time.perf_counter() - t0) / reps
        rows.append({"model": "params-v1", "preprocess_ms": 0.0,
                     "inference_ms": total * 1000, "total_ms": total * 1000,
                     "note": "no geometry step: the agent's own action space"})

    # --- pointcloud-v1 and fields-v1, which both need a mesh --------------
    mesh_dir = repo / "data" / "drivaerml" / "index" / "meshes"
    meshes = sorted(mesh_dir.glob("*.stl")) if mesh_dir.exists() else []
    if meshes:
        import time

        from src.data.dataset import sample_mesh

        for name, run, n_points in (("pointcloud-v1", "drivaerml-random", None),
                                    ("fields-v1", "fields", None)):
            ckpt_path = repo / "runs" / run / "best.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            pre = ckpt.get("preprocess", {})

            if name == "pointcloud-v1":
                from src.train.model import build_model

                model = build_model(ckpt["config"])
                try:
                    model.freeze_mean_path()
                except Exception:  # noqa: BLE001
                    pass
                pts = int(pre["n_points"])
            else:
                from src.train.train_fields import build_model_from_cfg

                model = build_model_from_cfg(ckpt["config"])
                model.freeze_mean_path()
                pts = int(pre.get("n_cloud", 2048))
            model.load_state_dict(ckpt["state_dict"])
            model.eval().to(device)

            # preprocessing: sample the mesh, as the service would
            t0 = time.perf_counter()
            cloud = sample_mesh(meshes[0], pts, True)
            pre_s = time.perf_counter() - t0

            x = torch.tensor(cloud, device=device).unsqueeze(0)
            if name == "pointcloud-v1":
                def once() -> None:
                    with torch.no_grad():
                        model(x)
            else:
                q = x[:, : min(4096, x.shape[1])]
                def once() -> None:
                    with torch.no_grad():
                        model(x, q, q)

            once()
            t0 = time.perf_counter()
            for _ in range(max(5, reps // 5)):
                once()
            inf_s = (time.perf_counter() - t0) / max(5, reps // 5)
            rows.append({
                "model": name, "preprocess_ms": pre_s * 1000,
                "inference_ms": inf_s * 1000,
                "total_ms": (pre_s + inf_s) * 1000,
                "note": f"includes sampling {pts:,} points from a "
                        f"{meshes[0].name} surface",
            })

    if not rows:
        print("nothing to benchmark -- train a checkpoint first")
        return 1

    print(f"{'model':16} {'preprocess':>11} {'inference':>10} {'total':>9} "
          f"{'speedup vs CFD':>15}")
    for r in rows:
        r["speedup_vs_cfd"] = CFD_SECONDS / (r["total_ms"] / 1000.0)
        print(f"  {r['model']:14} {r['preprocess_ms']:9.2f}ms "
              f"{r['inference_ms']:8.2f}ms {r['total_ms']:7.2f}ms "
              f"{r['speedup_vs_cfd']:14,.0f}x")
    for r in rows:
        print(f"\n  {r['model']}: {r['note']}")

    payload = {"device": str(device), "reps": reps,
               "cfd_reference_seconds": CFD_SECONDS,
               "cfd_reference_note": "DrivAerML, ~8 h per case on 256 cores",
               "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "latency.json", json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'latency.json'}")
    print("\nThe thesis number is the params-v1 total: that is what the agent")
    print("actually calls. The mesh-based rows are slower because sampling a")
    print("753k-face surface dominates, which is the honest caveat.")
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Package checkpoints and splits")
    ap.add_argument("--out", default=str(repo / "runs" / "handover"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="measure inference latency including preprocessing")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.verify:
        return cmd_verify(repo, out_dir)
    if args.benchmark:
        return cmd_benchmark(repo, out_dir, reps=args.reps)
    return cmd_build(repo, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
