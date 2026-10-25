"""Build a local, checksum-pinned model bundle. Never uploads or deploys it.

    python scripts/package_release.py --include-fields

Run after grading web/results and qualifying the selected trusted checkpoints.
The archive can be installed with scripts/install_artifacts.py. Dataset samples
retain their own licences; the manifest does not relicense model weights/data.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAT = "aero-surrogate-release/1"


@dataclass(frozen=True)
class Artifact:
    source: Path
    path: str
    role: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_path(value: str) -> str:
    from scripts.install_artifacts import _safe_name

    try:
        name = _safe_name(value)
        if not name.startswith("runs/"):
            raise ValueError("payload must be under runs/")
        return name
    except ValueError as exc:
        raise ValueError(f"invalid release artifact path: {value!r}") from exc


def build_bundle(artifacts: list[Artifact], output: Path, *, models: dict,
                 source: dict, created_at: str, metadata: dict | None = None) -> dict:
    """Snapshot bytes once and create a deterministic ZIP for a fixed manifest.

    Refuse symlinks and duplicates. File bytes in the manifest are precisely the
    bytes archived, even if an operator changes the source after this snapshot.
    """
    if set(metadata or {}) & {"format", "created_at", "source", "models", "files", "claims"}:
        raise ValueError("metadata cannot override release identity or payload declarations")
    entries: dict[str, bytes] = {}
    folded = set()
    records = []
    for item in sorted(artifacts, key=lambda item: item.path):
        name = _archive_path(item.path)
        if name.casefold() in folded:
            raise ValueError(f"duplicate artifact: {name}")
        if item.source.is_symlink() or not item.source.is_file():
            raise ValueError(f"artifact missing or symlink: {item.source}")
        data = item.source.read_bytes()
        if not data:
            raise ValueError(f"empty artifact: {name}")
        entries[name] = data
        folded.add(name.casefold())
        records.append({"path": name, "sha256": sha256(data), "bytes": len(data), "role": item.role})
    if not records:
        raise ValueError("a release must contain artifacts")
    for name, model in models.items():
        if model["path"] not in entries or sha256(entries[model["path"]]) != model["sha256"]:
            raise ValueError(f"model identity differs from payload: {name}")
    manifest = {"format": FORMAT, "created_at": created_at, "source": source,
                "models": models, "files": records,
                "claims": {"optimisation": "surrogate-predicted Cd changes, not CFD-validated improvements",
                           "fields": "experimental; generated viewer geometry is not CFD geometry",
                           "licensing": "MIT code does not relicense third-party datasets or model weights"},
                **(metadata or {})}
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".release-", suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in [("release-manifest.json", payload), *sorted(entries.items())]:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def validate_evidence_binding(checkpoint_sha256: str, metrics: dict, benchmark: dict) -> None:
    if metrics.get("models", {}).get("in_distribution", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("website metrics do not grade the selected scalar checkpoint; regenerate them first")
    if benchmark.get("protocol", {}).get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("matched benchmark uses a different scalar checkpoint")


def validate_published_summaries(root: Path, benchmark: dict) -> None:
    """Refuse stale published evidence, including checkout newline conversions."""
    records = benchmark.get("published_runs")
    if not isinstance(records, list) or not records:
        raise ValueError("matched benchmark must declare its published summary files")
    seen = set()
    for record in records:
        name = record.get("file") if isinstance(record, dict) else None
        if (not isinstance(name, str) or not name.endswith(".json") or "/" in name
                or "\\" in name or ":" in name or name.casefold() in seen):
            raise ValueError("invalid or duplicate published matched summary name")
        path = root / "docs/submission/evidence/matched" / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()) != record.get("published_summary_sha256"):
            raise ValueError(f"published matched summary checksum differs: {name}; restore canonical LF evidence before packaging")
        seen.add(name.casefold())


def evaluation_availability(metrics: dict, field_sha256: str | None = None) -> dict:
    """Describe completed grades separately from serving-archive contents.

    Evaluation data and the held-out-style checkpoint travel separately. An
    available grade must identify the checkpoint and the field arrays it used.
    """
    category = metrics.get("models", {}).get("out_of_distribution") or {}
    category_score = metrics.get("out_of_distribution") or {}
    category_graded = bool(category.get("sha256") and category_score.get("model") == "checkpoint"
                           and category_score.get("n_test", 0) > 0
                           and category_score.get("overall", {}).get("n", 0) > 0)
    field = metrics.get("field") or {}
    arrays = field.get("evaluation", {}).get("arrays") or []
    field_graded = bool(field.get("available") and field.get("checkpoint", {}).get("sha256")
                        and arrays and all(row.get("sha256") for row in arrays))
    if field_graded and field_sha256 and field["checkpoint"]["sha256"] != field_sha256:
        raise ValueError("field evaluation uses a different checkpoint from the serving bundle")
    return {"category_checkpoint": category_graded,
            "full_field_test_arrays": field_graded,
            "meaning": "verified inputs used by published evaluations; raw evaluation inputs are not included in this serving ZIP",
            "category_checkpoint_in_serving_archive": False,
            "full_field_test_arrays_in_serving_archive": False,
            "deployment": "pending; bundle generation does not publish anything"}


def collect_release(root: Path, *, include_fields: bool) -> tuple[list[Artifact], dict, dict, dict]:
    """Select explicit trusted local files, and verify the scalar metric binding."""
    from src.service.predictor import TorchPredictor
    from src.service.provenance import describe_predictor

    scalar = root / "runs/drivaernet-random/best.pt"
    identity = describe_predictor(TorchPredictor(str(scalar)))
    metrics = json.loads((root / "web/results/metrics.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "web/results/benchmark.json").read_text(encoding="utf-8"))
    validate_evidence_binding(identity["checkpoint_sha256"], metrics, benchmark)
    validate_published_summaries(root, benchmark)
    models = {"scalar": {"path": "runs/drivaernet-random/best.pt", "sha256": identity["checkpoint_sha256"],
                          "version": identity["model_version"], "schema_version": identity["schema_version"],
                          "preprocess": identity["preprocess_version"], "trained_body_styles": identity["trained_body_styles"],
                          "split_counts": identity["training_split_counts"], "seed": identity["training_seed"]}}
    mapping = [
        ("runs/drivaernet-random/best.pt", "runs/drivaernet-random/best.pt", "scalar checkpoint"),
        ("runs/handover/scalar_reference.json", "runs/handover/scalar_reference.json", "native scalar reference"),
        ("runs/handover/splits.json", "runs/handover/splits.json", "exact handover split package"),
        ("web/results/metrics.json", "runs/release/metrics.json", "independent scalar evaluation"),
        ("web/results/benchmark.json", "runs/release/benchmark.json", "historical matched experiment summary"),
        ("docs/submission/DATA_AND_MODEL_NOTICES.md", "runs/handover/DATA_AND_MODEL_NOTICES.md", "dataset and model provenance notices"),
    ]
    if include_fields:
        from src.service.fields import FieldService
        fields = FieldService.load(root / "runs/fields/best.pt").provenance()
        models["fields"] = {"path": "runs/fields/best.pt", "sha256": fields["checkpoint_sha256"],
                            "version": fields["model_version"], "preprocess": "fields-v1",
                            "status": "experimental; no generated-geometry accuracy or calibration qualification"}
        mapping.extend([
            ("runs/fields/best.pt", "runs/fields/best.pt", "experimental field checkpoint"),
            ("runs/handover/field_reference.json", "runs/handover/field_reference.json", "native field reference"),
            ("runs/handover/source/run_3.npz", "runs/handover/source/run_3.npz", "CC BY-SA 4.0 fixed-query reference sample"),
        ])
    artifacts = [Artifact(root / name, archive, role) for name, archive, role in mapping]
    evidence = root / "docs/submission/evidence"
    if evidence.is_dir():
        for path in sorted(evidence.rglob("*")):
            # This post-install report identifies the ZIP itself. Keep it
            # outside future bundles instead of archiving a stale ZIP digest.
            if path.name == "artifact-installation.json":
                continue
            if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".md"):
                relative = path.relative_to(evidence).as_posix()
                destination = "runs/" + relative if relative.startswith("logs/") else "runs/release/evidence/" + relative
                artifacts.append(Artifact(path, destination, "historical trace" if relative.startswith("logs/") else "public evaluation evidence"))
    source = {key: identity[key] for key in ("git_revision", "source_dirty", "source_sha256")}
    source["source_sha256_scope"] = "serving Python sources under src/ and requirements-service.txt/requirements-model.txt; not the full repository"
    metadata = {"dataset_notices": [
        {"name": "DrivAerNet", "licence": "CC BY-NC 4.0", "source": "https://github.com/Mohamedelrefaie/DrivAerNet"},
        {"name": "DrivAerML", "licence": "CC BY-SA 4.0", "source": "https://huggingface.co/datasets/neashton/drivaerml"}],
        "experiment_identities": {"scalar_evaluation_git_revision": metrics.get("git_commit"),
                                  "matched_protocol": benchmark.get("protocol_id"),
                                  "matched_experiment_git_revision": benchmark.get("protocol", {}).get("git_revision")},
        "availability": evaluation_availability(metrics, models.get("fields", {}).get("sha256"))}
    return artifacts, models, source, metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--include-fields", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/release/aero-surrogate-models.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/submission/release-manifest.json")
    parser.add_argument("--created-at", default=None, help="Fixed ISO timestamp makes repeat builds byte-identical")
    args = parser.parse_args(argv)
    try:
        artifacts, models, source, metadata = collect_release(args.root.resolve(), include_fields=args.include_fields)
        manifest = build_bundle(artifacts, args.output, models=models, source=source,
                                created_at=args.created_at or datetime.now(timezone.utc).isoformat(), metadata=metadata)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        digest = sha256(args.output.read_bytes())
        args.output.with_suffix(".zip.sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
        print(f"Local bundle: {args.output}\nSHA-256: {digest}\nManifest: {args.manifest}\nNothing uploaded or deployed.")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"release packaging failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
