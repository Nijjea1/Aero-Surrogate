"""Avneet's actual scalar exporter and serving verifier must interoperate offline.

The artifact here is trained on synthetic data by the existing trainer fixture;
these tests establish format and arithmetic parity, never real-model evidence.
"""

import copy
import json
import shutil

import pytest

from tests.test_torch_predictor import trained  # noqa: F401
from src.service.verify_checkpoint import main, verify_checkpoint
from src.train.reference import build_scalar


@pytest.fixture(scope="module")
def native_pack(trained, tmp_path_factory):
    repo = tmp_path_factory.mktemp("native-reference")
    checkpoint = repo / "runs" / "drivaernet-random" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(trained[0], checkpoint)
    return str(checkpoint), build_scalar(repo, repo / "handover")


def test_native_exporter_parity_and_reason_semantics(native_pack):
    path, pack = native_pack
    report = verify_checkpoint(path, pack)
    assert report["passed"], report["errors"]
    assert report["reference_format"] == "scalar_reference"
    assert report["absolute_tolerance"] == 1e-4
    assert report["thresholds"]["ood_score"] == pack["thresholds_used"]["ood_threshold"]
    assert len(report["results"]) == 5
    for result, design in zip(report["results"], pack["designs"], strict=True):
        assert result["params"] == design["params"]
        assert result["body_style"] == design["body_style"]
        assert result["expected"] == design["expected"]
        assert result["actual"]["contract_rejection_reason"] == design["expected"]["rejection_reason"]
    rejected = report["results"][-1]
    assert rejected["expected"]["in_distribution"] is False
    assert rejected["actual"]["reject_reason"].endswith("return toward the last trusted geometry")
    assert not rejected["actual"]["contract_rejection_reason"].endswith("return toward the last trusted geometry")
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("case", ["kind", "digest", "schema", "contract", "styles", "bytes", "version",
                                 "columns", "norm", "config", "seed", "split", "calibration", "method",
                                 "threshold", "cd", "reason", "unknown", "tolerance", "nonfinite", "bool"])
def test_native_tampering_fails(native_pack, case):
    path, original = native_pack
    pack = copy.deepcopy(original)
    if case == "kind":
        pack["kind"] = "field_reference"
    elif case == "digest":
        pack["checkpoint"]["sha256"] = "0" * 64
    elif case == "schema":
        pack["checkpoint"]["schema_version"] = "1.0.0"
    elif case == "contract":
        pack["checkpoint"]["contract_schema_version"] = "1.0.0"
    elif case == "styles":
        pack["checkpoint"]["trained_body_styles"] = ["fastback"]
    elif case == "bytes":
        pack["checkpoint"]["bytes"] += 1
    elif case == "version":
        pack["checkpoint"]["version"] = "wrong-version"
    elif case == "columns":
        pack["checkpoint"]["preprocess"]["param_columns"].reverse()
    elif case == "norm":
        pack["checkpoint"]["norm_stats"]["target_mean"] += 0.1
    elif case == "config":
        pack["training"]["config"]["seed"] += 1
    elif case == "seed":
        pack["training"]["seed"] += 1
    elif case == "split":
        pack["split_ids"]["counts"]["train"] += 1
    elif case == "calibration":
        pack["ood_detector"]["calibration"]["mean"] += 0.1
    elif case == "method":
        pack["ood_detector"]["method"] = "knn"
    elif case == "threshold":
        pack["thresholds_used"]["ood_threshold"] += 0.1
    elif case == "cd":
        pack["designs"][0]["expected"]["cd"] += 0.001
    elif case == "reason":
        pack["designs"][-1]["expected"]["rejection_reason"] += "; return toward the last trusted geometry"
    elif case == "unknown":
        pack["manifest_version"] = 1
    elif case == "tolerance":
        pack["tolerance"] = 0.1
    elif case == "bool":
        pack["designs"][0]["expected"]["uncertainty"] = True
    else:
        pack["designs"][0]["expected"]["ood_score"] = float("nan")
    report = verify_checkpoint(path, pack)
    assert not report["passed"], case
    assert report["errors"]


def test_native_cli_saves_qualification(native_pack, tmp_path, capsys):
    path, pack = native_pack
    references, output = tmp_path / "scalar_reference.json", tmp_path / "qualification.json"
    references.write_text(json.dumps(pack), encoding="utf-8")
    assert main(["--checkpoint", path, "--references", str(references), "--out", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] and report["reference_format"] == "scalar_reference"
    assert json.loads(capsys.readouterr().out)["passed"]
