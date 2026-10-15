"""Offline serving qualification, using the trainer-produced synthetic fixture.

These references test parity plumbing, not real aerodynamics. Real checkpoint
qualification remains an explicit command with Avneet's independent manifest.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_torch_predictor import trained, _reference, _save  # noqa: F401
from src.contracts import BODY_STYLES, Design, default_params, is_in_distribution
from src.service.gate import rejection_reason
from src.service.predictor import CheckpointError, StubPredictor, TorchPredictor
from src.service.provenance import describe_predictor
from src.service.verify_checkpoint import main, validate_manifest, verify_checkpoint


@pytest.fixture
def manifest(trained):
    path, checkpoint = trained
    thresholds = {"uncertainty": 0.013, "ood_score": checkpoint["ood"]["threshold"]}
    refs = []
    for style in BODY_STYLES:
        design = Design(default_params(), style)
        cd, sigma, score = _reference(checkpoint, design)
        trusted, _ = is_in_distribution(sigma, score, style,
                                       uncertainty_threshold=thresholds["uncertainty"],
                                       ood_threshold=thresholds["ood_score"])
        expected = {"cd": cd, "uncertainty": sigma, "ood_score": score,
                    "in_distribution": trusted}
        expected["reject_reason"] = rejection_reason(expected | {"cl": 0.0},
                                                     body_style=style,
                                                     ood_threshold=thresholds["ood_score"])
        refs.append({"id": style, "params": design.params, "body_style": style, "expected": expected})
    return {"manifest_version": 1, "schema_version": "2.0.0", "preprocess_version": "params-v1",
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "reference_source": "Synthetic fixture; independent HANDOFF reference computation",
            "absolute_tolerance": 1e-4, "thresholds": thresholds, "references": refs}


def test_reference_parity_and_full_provenance(trained, manifest):
    report = verify_checkpoint(trained[0], manifest)
    assert report["passed"], report
    assert len(report["results"]) == 3
    assert report["provenance"]["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    predictor = TorchPredictor(trained[0])
    assert predictor.checkpoint_sha256 == manifest["checkpoint_sha256"][:12]
    assert report["provenance"]["synthetic"] is False  # checkpoint-backed, fixture-trained model
    assert describe_predictor(StubPredictor())["synthetic"] is True


@pytest.mark.parametrize("change", ["digest", "threshold", "cd", "gate", "reason"])
def test_reference_disagreement_fails(trained, manifest, change):
    if change == "digest":
        manifest["checkpoint_sha256"] = "0" * 64
    elif change == "threshold":
        manifest["thresholds"]["ood_score"] += 0.1
    elif change == "cd":
        manifest["references"][0]["expected"]["cd"] += 0.001
    else:
        expected = manifest["references"][0]["expected"]
        expected["in_distribution"] = False
        expected["reject_reason"] = "deliberate mismatched rejection"
    assert not verify_checkpoint(trained[0], manifest)["passed"]


@pytest.mark.parametrize("change", ["unknown", "tolerance", "bool", "count", "duplicate", "nonfinite"])
def test_malformed_manifests_rejected(manifest, change):
    if change == "unknown":
        manifest["extra"] = True
    elif change == "tolerance":
        manifest["absolute_tolerance"] = 0.1
    elif change == "bool":
        manifest["references"][0]["expected"]["cd"] = True
    elif change == "count":
        manifest["references"] = manifest["references"][:2]
    elif change == "duplicate":
        manifest["references"][1] = copy.deepcopy(manifest["references"][0])
    else:
        manifest["references"][0]["expected"]["ood_score"] = float("nan")
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_cli_success_and_failure_reports(trained, manifest, tmp_path, capsys):
    refs, output = tmp_path / "refs.json", tmp_path / "report.json"
    refs.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["--checkpoint", trained[0], "--references", str(refs), "--out", str(output)]) == 0
    assert json.loads(output.read_text())["passed"]
    assert json.loads(capsys.readouterr().out)["passed"]
    refs.write_text("{invalid", encoding="utf-8")
    assert main(["--checkpoint", trained[0], "--references", str(refs)]) == 1
    assert not json.loads(capsys.readouterr().out)["passed"]


@pytest.mark.parametrize("case", ["held_out", "weights", "mean", "covariance", "indefinite", "zero",
                                 "asymmetric", "method", "quantile", "n_train", "knn", "stats", "threshold"])
def test_bad_checkpoint_refused_at_startup(trained, tmp_path, case):
    checkpoint = copy.deepcopy(trained[1])
    if case == "held_out":
        checkpoint["held_out_category"] = "notchback"
    elif case == "weights":
        next(iter(checkpoint["state_dict"].values())).flatten()[0] = float("nan")
    elif case == "mean":
        checkpoint["ood"]["mean"][0] = float("inf")
    elif case == "covariance":
        checkpoint["ood"]["cov_inv"][0][0] = float("nan")
    elif case in ("indefinite", "zero", "asymmetric"):
        size = len(checkpoint["ood"]["mean"])
        checkpoint["ood"]["cov_inv"] = [[float(i == j) for j in range(size)] for i in range(size)]
        if case == "indefinite":
            checkpoint["ood"]["cov_inv"][0][0] = -1.0
        elif case == "asymmetric":
            checkpoint["ood"]["cov_inv"][0][1] = 1.0
        else:
            checkpoint["ood"]["cov_inv"] = [[0.0] * size for _ in range(size)]
    elif case == "method":
        checkpoint["ood"]["method"] = "unsupported"
    elif case == "quantile":
        checkpoint["ood"]["quantile"] = True
    elif case == "n_train":
        checkpoint["ood"]["n_train"] = 0
    elif case == "stats":
        checkpoint["ood"]["train_scores"]["mean"] = float("nan")
    elif case == "threshold":
        checkpoint["ood"]["threshold"] = True
    else:
        checkpoint["ood"]["method"] = "knn"
        checkpoint["ood"]["reference"] = [[float("nan")]]
    with pytest.raises(CheckpointError):
        TorchPredictor(_save(tmp_path, checkpoint))


def test_legacy_parameter_space_default_remains_servable(trained, tmp_path):
    checkpoint = copy.deepcopy(trained[1])
    checkpoint["ood"].pop("space", None)
    assert TorchPredictor(_save(tmp_path, checkpoint)).ood_threshold > 0


def test_qualification_requires_ood(trained, manifest, tmp_path):
    checkpoint = copy.deepcopy(trained[1])
    checkpoint["ood"] = None
    path = _save(tmp_path, checkpoint)
    manifest["checkpoint_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert "requires a parameter-space" in verify_checkpoint(path, manifest)["errors"][0]


def test_finite_weights_with_nonfinite_probe_are_refused(trained, monkeypatch):
    import torch
    from src.train import model as training_model

    original = training_model.build_model

    def build(config):
        model = original(config)
        model.forward = lambda x: torch.full((len(x), 2), float("inf"), device=x.device)
        return model

    monkeypatch.setattr(training_model, "build_model", build)
    with pytest.raises(CheckpointError, match="probe must produce finite"):
        TorchPredictor(trained[0])


def test_provenance_without_git_is_nonfatal(monkeypatch):
    from src.service import provenance

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(provenance.subprocess, "run", unavailable)
    result = describe_predictor(StubPredictor())
    assert result["git_revision"] is None and result["source_dirty"] is None
