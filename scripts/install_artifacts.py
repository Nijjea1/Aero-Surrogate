"""Install an operator-trusted, checksum-pinned Aero Surrogate release ZIP.

This script uses only the standard library and never imports or unpickles a
checkpoint. Obtain the archive SHA-256 through a trusted release channel. The
archive checksum authenticates the manifest, which in turn checks every file.
All validation happens in a temporary directory before destination files change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

FORMAT = "aero-surrogate-release/1"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_FILES = 256
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEVICE = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)


class ArtifactError(ValueError):
    """An archive or destination fails the release installation contract."""


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise ArtifactError("SHA-256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise ArtifactError("invalid archive member path")
    if "\\" in value or ":" in value or any(ord(c) < 32 for c in value):
        raise ArtifactError("archive member path contains unsafe characters")
    parts = value.split("/")
    if any(p in {"", ".", ".."} or p.endswith((" ", ".")) or _DEVICE.match(p) for p in parts):
        raise ArtifactError("archive member path is not a portable relative path")
    if value != "release-manifest.json" and not (len(parts) > 1 and parts[0] == "runs"):
        raise ArtifactError("payload files must be underneath runs/")
    return str(PurePosixPath(value))


def _copy_bounded(source, output, limit: int) -> str:
    count, digest = 0, hashlib.sha256()
    while block := source.read(1024 * 1024):
        count += len(block)
        if count > limit:
            raise ArtifactError("archive or payload exceeds the size limit")
        digest.update(block)
        output.write(block)
    return digest.hexdigest()


def _fetch(archive: str | Path, output: Path, checksum: str) -> None:
    location = str(archive)
    parsed = urlsplit(location)
    remote = parsed.scheme.lower() in {"http", "https"}
    if remote and (parsed.scheme != "https" or parsed.username or parsed.password):
        raise ArtifactError("remote archives require an HTTPS URL without embedded credentials")
    if "://" in location and not remote:
        raise ArtifactError("only local paths or HTTPS release URLs are supported")
    source = urlopen(location, timeout=30) if remote else Path(archive).open("rb")
    with source, output.open("wb") as stream:
        if remote and urlsplit(source.geturl()).scheme != "https":
            raise ArtifactError("archive download redirected outside HTTPS")
        actual = _copy_bounded(source, stream, MAX_ARCHIVE_BYTES)
    if actual != checksum:
        raise ArtifactError("archive SHA-256 does not match; nothing was installed")


def _validate_manifest(manifest: object) -> dict[str, dict]:
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ArtifactError("unsupported release manifest format")
    records = manifest.get("files")
    if not isinstance(records, list) or not records or len(records) >= MAX_FILES:
        raise ArtifactError("release manifest needs a bounded, nonempty files list")
    declared, folded, total = {}, set(), 0
    for record in records:
        if not isinstance(record, dict):
            raise ArtifactError("invalid release manifest file record")
        name = _safe_name(record.get("path"))
        if name == "release-manifest.json" or name.casefold() in folded:
            raise ArtifactError("duplicate or reserved manifest payload path")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_ARCHIVE_BYTES:
            raise ArtifactError("invalid declared payload size")
        total += size
        if total > MAX_PAYLOAD_BYTES:
            raise ArtifactError("declared payload exceeds size limit")
        declared[name] = {**record, "sha256": _digest(record.get("sha256"))}
        folded.add(name.casefold())
    return declared


def _stage(archive: Path, staging: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if not 1 < len(members) <= MAX_FILES:
            raise ArtifactError("archive member count exceeds release limits")
        indexed, folded = {}, set()
        for member in members:
            name = _safe_name(member.filename)
            mode = member.external_attr >> 16
            if (member.is_dir() or stat.S_ISLNK(mode) or member.flag_bits & 1
                    or (stat.S_IFMT(mode) and not stat.S_ISREG(mode))):
                raise ArtifactError("archive must contain unencrypted regular files only")
            if name.casefold() in folded:
                raise ArtifactError("archive contains duplicate or case-conflicting paths")
            indexed[name] = member
            folded.add(name.casefold())
        info = indexed.get("release-manifest.json")
        if info is None or info.file_size > MAX_MANIFEST_BYTES:
            raise ArtifactError("release manifest is missing or too large")
        manifest_bytes = zipped.read(info)
        try:
            declared = _validate_manifest(json.loads(manifest_bytes.decode("utf-8")))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("release manifest is not valid UTF-8 JSON") from exc
        if set(indexed) != {"release-manifest.json", *declared}:
            raise ArtifactError("archive payload differs from the manifest files list")
        for name, record in declared.items():
            member = indexed[name]
            if member.file_size != record["bytes"]:
                raise ArtifactError(f"declared size differs for {name}")
            target = staging.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                digest = _copy_bounded(source, output, record["bytes"])
            if digest != record["sha256"]:
                raise ArtifactError(f"payload SHA-256 differs for {name}")
        (staging / "release-manifest.json").write_bytes(manifest_bytes)
    return [*declared, "release-manifest.json"]


def _target(root: Path, name: str) -> Path:
    target = root.joinpath(*PurePosixPath(name).parts)
    for part in (target, *target.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ArtifactError("destination contains a symbolic link")
    if not target.resolve().is_relative_to(root):
        raise ArtifactError("destination escapes the installation root")
    return target


def install(archive: str | Path, checksum: str, destination: str | Path = ".", *, replace: bool = False) -> list[str]:
    """Verify and install a release without executing its model files.

    Existing files are refused unless the operator explicitly requests replace.
    Validation failures never change the destination. Filesystem failures during
    the final copy can still leave a partial installation and should be retried.
    """
    checksum = _digest(checksum)
    raw_root = Path(destination).absolute()
    if raw_root.is_symlink() or any(p.is_symlink() for p in raw_root.parents):
        raise ArtifactError("installation root must not contain symbolic links")
    root = raw_root.resolve()
    with tempfile.TemporaryDirectory(prefix="aero-artifacts-") as temporary:
        work = Path(temporary)
        archive_path, staging = work / "release.zip", work / "payload"
        staging.mkdir()
        _fetch(archive, archive_path, checksum)
        names = _stage(archive_path, staging)
        targets = [_target(root, name) for name in names]
        for target in targets:
            if target.exists() and (not replace or not target.is_file()):
                raise ArtifactError(f"destination exists: {target.name}; use --replace only for a trusted update")
        for name, target in zip(names, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recheck after mkdir, then write using a temporary sibling and atomic replace.
            _target(root, name)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".aero-", delete=False) as output:
                pending = Path(output.name)
                with staging.joinpath(*PurePosixPath(name).parts).open("rb") as source:
                    shutil.copyfileobj(source, output)
            try:
                if target.exists() and not replace:
                    raise ArtifactError("destination appeared during installation; refusing overwrite")
                os.replace(pending, target)
            finally:
                pending.unlink(missing_ok=True)
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="local ZIP or HTTPS release URL")
    parser.add_argument("--sha256", required=True, help="archive SHA-256 from a trusted release channel")
    parser.add_argument("--destination", default=".", help="repository root (default: current directory)")
    parser.add_argument("--replace", action="store_true", help="replace an existing trusted installation")
    args = parser.parse_args(argv)
    try:
        names = install(args.archive, args.sha256, args.destination, replace=args.replace)
    except (ArtifactError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Artifacts not installed: {exc}\n")
    print(f"Verified and installed {len(names) - 1} payload files plus release-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
