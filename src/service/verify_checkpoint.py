"""Qualify a real params-v1 checkpoint against independent reference outputs.

Run explicitly with --checkpoint PATH --references MANIFEST [--out REPORT].
There are no implicit artifact locations and no provider/API calls. References
must be produced independently by the checkpoint owner; agreement verifies
serving parity, not aerodynamic accuracy or uncertainty calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from ..contracts import BODY_STYLES, Design, SCHEMA_VERSION, is_in_distribution, validate_body_style
from .gate import rejection_reason
from .predictor import PARAMS_PREPROCESS_VERSION, TorchPredictor, validate_design_params
from .provenance import describe_predictor


def _keys(value, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def _number(value, label: str, *, nonnegative: bool = False) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError(f"{label} must be a finite {'nonnegative ' if nonnegative else ''}number")
    return float(value)


def validate_manifest(manifest: object) -> dict:
    """Reject ambiguous, incomplete or permissive qualification manifests."""
    _keys(manifest, {"manifest_version", "schema_version", "preprocess_version",
                     "checkpoint_sha256", "reference_source", "absolute_tolerance",
                     "thresholds", "references"}, "manifest")
    if type(manifest["manifest_version"]) is not int or manifest["manifest_version"] != 1:
        raise ValueError("manifest_version must be 1")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema_version does not match the service")
    if manifest["preprocess_version"] != PARAMS_PREPROCESS_VERSION:
        raise ValueError("manifest preprocess_version must be params-v1")
    digest = manifest["checkpoint_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("checkpoint_sha256 must be a full lowercase SHA-256 digest")
    if not isinstance(manifest["reference_source"], str) or not manifest["reference_source"].strip():
        raise ValueError("reference_source must identify the independent reference computation")
    tolerance = _number(manifest["absolute_tolerance"], "absolute_tolerance", nonnegative=True)
    if not 0 < tolerance <= 1e-4:
        raise ValueError("absolute_tolerance must be positive and at most 1e-4")
    _keys(manifest["thresholds"], {"uncertainty", "ood_score"}, "thresholds")
    for key, value in manifest["thresholds"].items():
        if _number(value, f"thresholds.{key}", nonnegative=True) <= 0:
            raise ValueError(f"thresholds.{key} must be positive")
    refs = manifest["references"]
    if not isinstance(refs, list) or not 3 <= len(refs) <= 5:
        raise ValueError("references must contain 3 to 5 designs")
    names, designs = set(), set()
    for reference in refs:
        _keys(reference, {"id", "params", "body_style", "expected"}, "reference")
        name = reference["id"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("reference ids must be nonempty and unique")
        names.add(name)
        problems = validate_design_params(reference["params"]) + validate_body_style(reference["body_style"])
        if problems:
            raise ValueError(f"reference {name}: {'; '.join(problems)}")
        identity = (reference["body_style"], tuple(sorted(reference["params"].items())))
        if identity in designs:
            raise ValueError("reference designs must be distinct")
        designs.add(identity)
        expected = reference["expected"]
        _keys(expected, {"cd", "uncertainty", "ood_score", "in_distribution", "reject_reason"}, "expected")
        _number(expected["cd"], "expected.cd")
        for key in ("uncertainty", "ood_score"):
            _number(expected[key], f"expected.{key}", nonnegative=True)
        if type(expected["in_distribution"]) is not bool:
            raise ValueError("expected.in_distribution must be boolean")
        reason = expected["reject_reason"]
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("expected.reject_reason must be null or nonempty text")
        if expected["in_distribution"] != (reason is None):
            raise ValueError("expected trust verdict and reject_reason disagree")
    return manifest


def normalize_reference_pack(pack: object) -> dict:
    """Validate either known format and return the common numeric manifest.

    Native rejection text is the contract's raw reason, whereas the legacy
    manifest describes service guidance. Normalization does not rewrite that
    text: the verifier selects the appropriate comparison explicitly.
    """
    if not isinstance(pack, dict) or "kind" not in pack:
        return validate_manifest(pack)
    if pack["kind"] != "scalar_reference":
        raise ValueError("reference kind must be scalar_reference")
    _keys(pack, {"kind", "generated_utc", "generated_by", "independent_of", "tolerance",
                 "tolerance_note", "checkpoint", "training", "ood_detector", "thresholds_used",
                 "split_ids", "designs"}, "scalar reference pack")
    for key in ("generated_utc", "generated_by", "independent_of", "tolerance_note"):
        if not isinstance(pack[key], str) or not pack[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    checkpoint = pack["checkpoint"]
    _keys(checkpoint, {"path", "sha256", "bytes", "version", "schema_version",
                       "contract_schema_version", "n_params", "preprocess", "norm_stats",
                       "trained_body_styles", "trained_body_styles_note"}, "checkpoint")
    if checkpoint["contract_schema_version"] != SCHEMA_VERSION:
        raise ValueError("reference contract schema differs from the service")
    if checkpoint["trained_body_styles"] != sorted(BODY_STYLES):
        raise ValueError("native reference requires all three trained body styles")
    if type(checkpoint["bytes"]) is not int or checkpoint["bytes"] <= 0:
        raise ValueError("checkpoint.bytes must be a positive integer")
    if not isinstance(checkpoint["preprocess"], dict):
        raise ValueError("checkpoint.preprocess must be an object")
    _keys(pack["training"], {"git_commit", "seed", "config"}, "training")
    _keys(pack["split_ids"], {"note", "counts", "ids"}, "split_ids")
    detector = pack["ood_detector"]
    _keys(detector, {"space", "method", "diagonal", "threshold", "quantile", "n_train",
                     "calibration", "formula"}, "ood_detector")
    # The independent native forward implements Mahalanobis arithmetic only;
    # allowing a kNN claim here would validate a different algorithm by accident.
    if detector["space"] != "parameters" or detector["method"] != "mahalanobis":
        raise ValueError("native references require a parameter-space Mahalanobis detector")
    thresholds = pack["thresholds_used"]
    _keys(thresholds, {"uncertainty_threshold", "uncertainty_threshold_source", "ood_threshold",
                       "ood_threshold_source"}, "thresholds_used")
    if detector["threshold"] != thresholds["ood_threshold"]:
        raise ValueError("native detector and gate thresholds disagree")
    references = []
    if not isinstance(pack["designs"], list):
        raise ValueError("designs must be a list")
    for row in pack["designs"]:
        _keys(row, {"id", "why", "params", "body_style", "expected"}, "native design")
        _keys(row["expected"], {"cd", "uncertainty", "ood_score", "in_distribution",
                                "rejection_reason"}, "native expected")
        expected = dict(row["expected"])
        expected["reject_reason"] = expected.pop("rejection_reason")
        references.append({key: row[key] for key in ("id", "params", "body_style")} | {"expected": expected})
    return validate_manifest({
        "manifest_version": 1, "schema_version": checkpoint["schema_version"],
        "preprocess_version": checkpoint["preprocess"].get("version"),
        "checkpoint_sha256": checkpoint["sha256"], "reference_source": pack["generated_by"],
        "absolute_tolerance": pack["tolerance"],
        "thresholds": {"uncertainty": thresholds["uncertainty_threshold"], "ood_score": thresholds["ood_threshold"]},
        "references": references})


def _check_native_metadata(path: str, pack: dict) -> None:
    """Bind descriptive native metadata to the already trusted loaded artifact."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    declared = pack["checkpoint"]
    if Path(path).stat().st_size != declared["bytes"]:
        raise ValueError("checkpoint byte size differs from native reference")
    for name in ("version", "schema_version", "n_params", "preprocess", "norm_stats", "trained_body_styles"):
        if checkpoint.get(name) != declared[name]:
            raise ValueError(f"checkpoint {name} differs from native reference")
    for name in ("space", "method", "diagonal", "threshold", "quantile", "n_train"):
        if checkpoint["ood"].get(name) != pack["ood_detector"][name]:
            raise ValueError(f"OOD {name} differs from native reference")
    if checkpoint["ood"].get("train_scores") != pack["ood_detector"]["calibration"]:
        raise ValueError("OOD calibration differs from native reference")
    if checkpoint.get("config") != pack["training"]["config"]:
        raise ValueError("training config differs from native reference")
    if checkpoint["config"].get("seed") != pack["training"]["seed"]:
        raise ValueError("training seed differs from native reference")
    splits = checkpoint.get("splits") or None
    counts = {key: len(value) for key, value in splits.items()} if splits else None
    if pack["split_ids"]["ids"] != splits or pack["split_ids"]["counts"] != counts:
        raise ValueError("training splits differ from native reference")


def verify_checkpoint(checkpoint: str, manifest: object) -> dict:
    """Return an auditable report; loading/validation failures become errors."""
    report = {"passed": False, "errors": [], "provenance": None, "results": []}
    try:
        native = isinstance(manifest, dict) and manifest.get("kind") == "scalar_reference"
        original = manifest
        manifest = normalize_reference_pack(manifest)
        report["reference_format"] = "scalar_reference" if native else "legacy_manifest"
        report["absolute_tolerance"] = manifest["absolute_tolerance"]
        report["thresholds"] = dict(manifest["thresholds"])
        predictor = TorchPredictor(checkpoint)
        report["provenance"] = describe_predictor(predictor)
        report["reference_source"] = manifest["reference_source"]
        if predictor.checkpoint_sha256_full != manifest["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA-256 differs from reference manifest")
        if predictor.ood_threshold is None:
            raise ValueError("qualification requires a parameter-space OOD detector")
        if native:
            _check_native_metadata(checkpoint, original)
        for key, actual in (("uncertainty", predictor.uncertainty_threshold),
                            ("ood_score", predictor.ood_threshold)):
            if actual != manifest["thresholds"][key]:
                raise ValueError(f"{key} threshold differs from reference manifest")
        for index, reference in enumerate(manifest["references"]):
            actual = predictor.evaluate(Design(reference["params"], reference["body_style"])).to_dict()
            actual["reject_reason"] = rejection_reason(
                actual, threshold=predictor.uncertainty_threshold,
                body_style=reference["body_style"], ood_threshold=predictor.ood_threshold)
            _, actual["contract_rejection_reason"] = is_in_distribution(
                actual["uncertainty"], actual["ood_score"], reference["body_style"],
                uncertainty_threshold=predictor.uncertainty_threshold, ood_threshold=predictor.ood_threshold)
            expected_values = original["designs"][index]["expected"] if native else reference["expected"]
            errors = []
            for key, expected in expected_values.items():
                observed = actual["contract_rejection_reason" if key == "rejection_reason" else key]
                agrees = (abs(observed - expected) <= manifest["absolute_tolerance"]
                          if key in ("cd", "uncertainty", "ood_score") else observed == expected)
                if not agrees:
                    errors.append(f"{reference['id']}: {key} differs from reference")
            report["results"].append({"id": reference["id"], "passed": not errors,
                                      "params": reference["params"], "body_style": reference["body_style"],
                                      "expected": expected_values, "actual": actual})
            report["errors"].extend(errors)
    except Exception as exc:  # an explicit local qualification must always produce a report
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["passed"] = not report["errors"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(Path(args.references).read_text(encoding="utf-8"))
        report = verify_checkpoint(args.checkpoint, manifest)
    except (OSError, ValueError) as exc:
        report = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"], "results": []}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        try:
            Path(args.out).write_text(encoded + "\n", encoding="utf-8")
        except OSError as exc:
            report["passed"] = False
            report["errors"].append(f"cannot write report: {exc}")
            encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
