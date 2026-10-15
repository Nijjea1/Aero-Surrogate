"""Structured predictor identity for health snapshots and reproducible runs.

No local checkpoint paths are exposed: a digest identifies the artifact without
leaking a machine's directory layout. Missing model metadata stays unavailable.
"""

import hashlib
import subprocess
from pathlib import Path

from ..contracts import SCHEMA_VERSION


def _source_sha256() -> str | None:
    """Fingerprint shipped sources without reading secrets, logs or datasets."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    try:
        files = sorted((root / "src").rglob("*.py"))
        files += [root / name for name in ("requirements-service.txt", "requirements-model.txt")
                  if (root / name).is_file()]
        for path in files:
            name, data = path.relative_to(root).as_posix().encode("utf-8"), path.read_bytes()
            digest.update(len(name).to_bytes(8, "big") + name + len(data).to_bytes(8, "big") + data)
        return digest.hexdigest()
    except OSError:
        return None


def _source_identity() -> dict:
    """Source identity is useful evidence; installed packages may lack git."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=2, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True,
                                text=True, timeout=2, check=True)
        return {"git_revision": result.stdout.strip() or None, "source_dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_revision": None, "source_dirty": None}


def describe_predictor(predictor) -> dict:
    """Return JSON-safe provenance without changing the Evaluation contract."""
    return {
        "predictor": predictor.name,
        "synthetic": predictor.name == "stub",
        "schema_version": SCHEMA_VERSION,
        **_source_identity(),
        "source_sha256": _source_sha256(),
        "model_version": getattr(predictor, "model_version", None),
        "checkpoint_sha256": getattr(predictor, "checkpoint_sha256_full", None),
        "preprocess_version": getattr(predictor, "preprocess_version", None),
        "training_git_revision": getattr(predictor, "training_git_revision", None),
        "training_seed": getattr(predictor, "training_seed", None),
        "trained_body_styles": getattr(predictor, "trained_body_styles", None),
        "training_split_counts": getattr(predictor, "training_split_counts", None),
        "output_provenance": dict(getattr(predictor, "output_provenance", {})),
        "uncertainty_threshold": getattr(predictor, "uncertainty_threshold", None),
        "ood_score_threshold": getattr(predictor, "ood_threshold", None),
        "serving_policy": "synthetic" if predictor.name == "stub" else "all-styles-only",
    }
