"""A verified release must not install partial, undeclared or escaping payloads."""

import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.install_artifacts import ArtifactError, FORMAT, install


def bundle(tmp_path, payload=None, *, records=None, extra=None, manifest=None):
    payload = payload or {"runs/drivaernet-random/best.pt": b"trusted-weights"}
    if manifest is None:
        manifest = {"format": FORMAT, "files": records if records is not None else [
            {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for name, content in payload.items()]}
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("release-manifest.json", json.dumps(manifest))
        for name, content in payload.items():
            zipped.writestr(name, content)
        if extra:
            for name, content in extra:
                zipped.writestr(name, content)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_release_installs_verified_payload_and_manifest(tmp_path):
    archive, checksum = bundle(tmp_path)
    destination = tmp_path / "fresh-clone"
    names = install(archive, checksum.upper(), destination)
    assert set(names) == {"runs/drivaernet-random/best.pt", "release-manifest.json"}
    assert (destination / names[0]).read_bytes() == b"trusted-weights"
    assert json.loads((destination / "release-manifest.json").read_text())["format"] == FORMAT


def test_archive_hash_mismatch_does_not_create_destination(tmp_path):
    archive, _ = bundle(tmp_path)
    destination = tmp_path / "fresh-clone"
    with pytest.raises(ArtifactError, match="archive SHA-256"):
        install(archive, "0" * 64, destination)
    assert not destination.exists()


def test_each_payload_is_verified_before_destination_changes(tmp_path):
    payload = {"runs/first.json": b"good", "runs/second.pt": b"bad"}
    records = [{"path": name, "bytes": len(value), "sha256": "0" * 64}
               for name, value in payload.items()]
    archive, checksum = bundle(tmp_path, payload, records=records)
    destination = tmp_path / "fresh-clone"
    with pytest.raises(ArtifactError, match="payload SHA-256"):
        install(archive, checksum, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../outside.pt", "/absolute.pt", "runs/../outside.pt",
                                 "runs\\outside.pt", "runs/C:outside.pt", "runs/NUL.pt",
                                 "runs/file. ", "scripts/surprise.py", "runs//model.pt"])
def test_unsafe_paths_are_refused_without_writing(tmp_path, name):
    archive, checksum = bundle(tmp_path, {name: b"no"})
    destination = tmp_path / "fresh-clone"
    with pytest.raises(ArtifactError):
        install(archive, checksum, destination)
    assert not destination.exists()


def test_undeclared_payload_is_refused(tmp_path):
    archive, checksum = bundle(tmp_path, extra=[("runs/extra.pt", b"extra")])
    with pytest.raises(ArtifactError, match="differs from the manifest"):
        install(archive, checksum, tmp_path / "destination")


def test_missing_declared_payload_is_refused(tmp_path):
    archive, checksum = bundle(tmp_path, records=[
        {"path": "runs/missing.pt", "bytes": 1, "sha256": "0" * 64}])
    with pytest.raises(ArtifactError, match="differs from the manifest"):
        install(archive, checksum, tmp_path / "destination")


def test_symlink_archive_member_is_refused(tmp_path):
    link = zipfile.ZipInfo("runs/link.pt")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive, checksum = bundle(tmp_path, extra=[(link, b"/outside")])
    with pytest.raises(ArtifactError, match="regular files"):
        install(archive, checksum, tmp_path / "destination")


def test_case_conflicting_members_are_refused(tmp_path):
    archive, checksum = bundle(tmp_path, {"runs/model.pt": b"a", "runs/MODEL.pt": b"b"})
    with pytest.raises(ArtifactError, match="case-conflicting"):
        install(archive, checksum, tmp_path / "destination")


def test_existing_install_requires_explicit_replacement(tmp_path):
    archive, checksum = bundle(tmp_path)
    destination = tmp_path / "destination"
    existing = destination / "runs/drivaernet-random/best.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    with pytest.raises(ArtifactError, match="destination exists"):
        install(archive, checksum, destination)
    assert existing.read_bytes() == b"old"
    assert not (destination / "release-manifest.json").exists()
    install(archive, checksum, destination, replace=True)
    assert existing.read_bytes() == b"trusted-weights"


def test_oversized_declared_payload_is_refused(tmp_path):
    archive, checksum = bundle(tmp_path, records=[
        {"path": "runs/drivaernet-random/best.pt", "bytes": 10 ** 12, "sha256": "0" * 64}])
    with pytest.raises(ArtifactError, match="payload size"):
        install(archive, checksum, tmp_path / "destination")


@pytest.mark.parametrize("checksum", ["bad", "", "z" * 64])
def test_invalid_checksum_is_refused(tmp_path, checksum):
    with pytest.raises(ArtifactError, match="SHA-256"):
        install(tmp_path / "missing.zip", checksum, tmp_path / "destination")


def test_plain_http_download_is_refused_before_network(tmp_path):
    with pytest.raises(ArtifactError, match="HTTPS"):
        install("http://example.invalid/release.zip", "0" * 64, tmp_path / "destination")


def test_destination_symlink_is_refused(tmp_path):
    archive, checksum = bundle(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks requires privileges on this platform")
    with pytest.raises(ArtifactError, match="symbolic link"):
        install(archive, checksum, linked)
    assert not list(real.iterdir())
