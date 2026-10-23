"""Refresh the pinned three.js module dependency closure used by this app.

No npm or runtime CDN is required. Fetches only three@0.160.0 from jsDelivr,
preserves the upstream MIT licence, and records SHA-256 for every local file.
Run deliberately when updating the vendored snapshot; normal startup is offline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.160.0"
BASE = f"https://cdn.jsdelivr.net/npm/three@{VERSION}/"
SEEDS = ["build/three.module.js", "examples/jsm/controls/OrbitControls.js",
         "examples/jsm/environments/RoomEnvironment.js", "examples/jsm/postprocessing/EffectComposer.js",
         "examples/jsm/postprocessing/RenderPass.js", "examples/jsm/postprocessing/UnrealBloomPass.js",
         "examples/jsm/postprocessing/OutputPass.js", "examples/jsm/lines/LineSegments2.js",
         "examples/jsm/lines/LineSegmentsGeometry.js", "examples/jsm/lines/LineMaterial.js", "LICENSE"]


def normalise(parent: str, reference: str) -> str:
    parts: list[str] = list(PurePosixPath(parent).parent.parts)
    for part in reference.split("/"):
        if part == "..":
            if not parts:
                raise ValueError("dependency escaped package")
            parts.pop()
        elif part not in (".", ""):
            parts.append(part)
    return "/".join(parts)


def main() -> None:
    destination = ROOT / "web/vendor/three"
    queue, seen, records = list(SEEDS), set(), []
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        if len(seen) >= 80 or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError("unexpected dependency closure")
        request = urllib.request.Request(BASE + path, headers={"User-Agent": "Aero-Surrogate-vendor/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(4_000_001)
        if len(data) > 4_000_000:
            raise ValueError("dependency exceeds expected size")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        seen.add(path)
        records.append({"path": path, "source": BASE + path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        if path.endswith(".js"):
            # Upstream modules use relative imports plus the bare 'three' specifier.
            refs = re.findall(r"(?:from\s*|import\s*)['\"]([^'\"]+)['\"]", data.decode())
            for ref in refs:
                if ref.startswith("."):
                    queue.append(normalise(path, ref))
                elif ref != "three":
                    raise ValueError(f"unexpected bare dependency: {ref}")
    (destination / "manifest.json").write_text(json.dumps({"package": "three", "version": VERSION,
        "licence": "MIT", "files": sorted(records, key=lambda item: item["path"])}, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(records)} pinned files under {destination}; runtime requires no CDN.")


if __name__ == "__main__":
    main()
