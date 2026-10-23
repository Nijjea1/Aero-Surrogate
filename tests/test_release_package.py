"""The installer must receive an exact, reproducible snapshot of the declared files."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.package_release import Artifact, build_bundle, validate_evidence_binding, validate_published_summaries


def test_bundle_snapshots_hashes_and_builds_identically(tmp_path):
    model = tmp_path / "trusted.pt"
    model.write_bytes(b"a locally selected model fixture")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    kwargs = dict(models={"scalar": {"path": "runs/scalar/best.pt", "sha256": expected}},
                  source={"git_revision": "fixture"}, created_at="2026-10-27T00:00:00Z")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    manifest = build_bundle([Artifact(model, "runs/scalar/best.pt", "scalar checkpoint")], a, **kwargs)
    build_bundle([Artifact(model, "runs/scalar/best.pt", "scalar checkpoint")], b, **kwargs)
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as archive:
        assert set(archive.namelist()) == {"release-manifest.json", "runs/scalar/best.pt"}
        declared = json.loads(archive.read("release-manifest.json"))
        assert declared == manifest
        record = declared["files"][0]
        data = archive.read(record["path"])
        assert record["sha256"] == hashlib.sha256(data).hexdigest() == expected
        assert record["bytes"] == len(data)


@pytest.mark.parametrize("name", ["../outside", "/runs/a", "runs/../outside", "runs\\evil", "runs/a:b", "runs//a", "other/model.pt", "runs/CON", "runs/file.", "runs/a ", "runs/line\nname"])
def test_bundle_refuses_unsafe_names(tmp_path, name):
    f = tmp_path / "fixture"
    f.write_bytes(b"payload")
    with pytest.raises(ValueError, match="invalid release"):
        build_bundle([Artifact(f, name, "fixture")], tmp_path / "bad.zip", models={}, source={}, created_at="fixed")
    assert not (tmp_path / "bad.zip").exists()


def test_bundle_refuses_duplicate_or_mismatched_model(tmp_path):
    f = tmp_path / "fixture"
    f.write_bytes(b"payload")
    item = Artifact(f, "runs/a", "fixture")
    with pytest.raises(ValueError, match="duplicate"):
        build_bundle([item, item], tmp_path / "bad.zip", models={}, source={}, created_at="fixed")
    with pytest.raises(ValueError, match="duplicate"):
        build_bundle([item, Artifact(f, "runs/A", "fixture")], tmp_path / "bad.zip", models={}, source={}, created_at="fixed")
    with pytest.raises(ValueError, match="model identity"):
        build_bundle([item], tmp_path / "bad.zip", models={"scalar": {"path": "runs/a", "sha256": "0" * 64}}, source={}, created_at="fixed")


def test_release_refuses_stale_benchmark_binding():
    metric = {"models": {"in_distribution": {"sha256": "current"}}}
    with pytest.raises(ValueError, match="different scalar"):
        validate_evidence_binding("current", metric, {"protocol": {"checkpoint_sha256": "older"}})
    validate_evidence_binding("current", metric, {"protocol": {"checkpoint_sha256": "current"}})


def test_metadata_cannot_replace_payload_declarations(tmp_path):
    with pytest.raises(ValueError, match="cannot override"):
        build_bundle([], tmp_path / "bad.zip", models={}, source={}, created_at="fixed", metadata={"files": []})


def test_packaging_refuses_checkout_newline_changes_to_published_evidence(tmp_path):
    source = tmp_path / "docs/submission/evidence/matched/super-1.json"
    source.parent.mkdir(parents=True)
    canonical = b'{"run_id":"matched-super-1"}\n'
    record = {"file": source.name, "published_summary_sha256": hashlib.sha256(canonical).hexdigest()}
    source.write_bytes(canonical.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="checksum differs"):
        validate_published_summaries(tmp_path, {"published_runs": [record]})
    source.write_bytes(canonical)
    validate_published_summaries(tmp_path, {"published_runs": [record]})


def test_packaging_refuses_unsafe_or_duplicate_summary_references(tmp_path):
    for records in [[{"file": "../outside.json"}], [{"file": "model\\a.json"}]]:
        with pytest.raises(ValueError, match="summary name"):
            validate_published_summaries(tmp_path, {"published_runs": records})
