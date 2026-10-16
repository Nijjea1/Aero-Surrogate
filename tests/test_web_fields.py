"""Binding /fields predictions to the viewer mesh (web/fieldmap.js), run in Node.

The browser must hash the mesh exactly as the service does
(src/service/INTEGRATION.md: SHA-256 over "viewer-f32le-u32le-v1\\0", the vertex
bytes and the face bytes), bind only on a matching hash and counts, and turn
per-face values into per-vertex display values by area weighting.
Skipped without Node.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.contracts import PRESSURE_REGIONS, default_params
from src.geometry.viewer import mesh_payload

FIELDMAP = (Path(__file__).resolve().parents[1] / "web" / "fieldmap.js").as_uri()
NODE = shutil.which("node")

SCRIPT = """
import { readFileSync } from "node:fs";
import { bindField, meshIdentity } from "%s";
const { payload, response } = JSON.parse(readFileSync(process.argv[1], "utf8"));
const id = await meshIdentity(payload);
const field = bindField(response, id, payload);
const refuse = (patch) => {
  try { bindField({ ...response, ...patch(response) }, id, payload); return null; } catch (e) { return e.constructor.name; }
};
console.log(JSON.stringify({
  id,
  vertexCp: Array.from(field.vertexCp),
  faceCp: Array.from(field.faceCp.slice(0, 50)),
  summary: field.summary,
  otherHash: refuse((r) => ({ mesh: { ...r.mesh, sha256: "0".repeat(64) } })),
  otherCount: refuse((r) => ({ mesh: { ...r.mesh, n_faces: r.mesh.n_faces - 1 } })),
  perVertex: refuse((r) => ({ mesh: { ...r.mesh, association: "vertex" } })),
  short: refuse((r) => ({ cp: r.cp.slice(1) })),
}));
""" % FIELDMAP


@pytest.fixture(scope="module")
def bound(tmp_path_factory):
    if NODE is None:
        pytest.skip("node is not installed")
    params = default_params()
    params["B_Ramp_Angle"] = 2.5
    payload = mesh_payload(params, "notchback")
    vertices = np.frombuffer(base64.b64decode(payload["vertices"]), "<f4").reshape(-1, 3)
    faces = np.frombuffer(base64.b64decode(payload["faces"]), "<u4").reshape(-1, 3)
    digest = hashlib.sha256(b"viewer-f32le-u32le-v1\0" + base64.b64decode(payload["vertices"])
                            + base64.b64decode(payload["faces"])).hexdigest()
    n = len(faces)
    # Reverse the item order, so binding has to go through face_indices.
    order = np.arange(n)[::-1]
    rng = np.random.default_rng(0)
    cp_face = rng.uniform(-1, 1, n)
    response = {
        "mesh": {"sha256": digest, "hash_format": "viewer-f32le-u32le-v1", "n_vertices": len(vertices),
                 "n_faces": n, "association": "face", "query_location": "face_centroid",
                 "face_indices": order.tolist()},
        "cp": cp_face[order].tolist(), "sigma": np.full(n, 0.05).tolist(), "region": (order % 8).tolist(),
        "regions": list(PRESSURE_REGIONS),
        "pressure_summary": {"roof": -0.4, "wheel_wells": None},
    }
    path = tmp_path_factory.mktemp("fields") / "case.json"
    path.write_text(json.dumps({"payload": payload, "response": response}), encoding="utf-8")
    out = subprocess.run([NODE, "--input-type=module", "-e", SCRIPT, str(path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout), digest, vertices, faces, cp_face


def test_the_browser_hashes_the_mesh_like_the_service(bound):
    result, digest, vertices, faces, _ = bound
    assert result["id"] == {"sha256": digest, "nVertices": len(vertices), "nFaces": len(faces)}


def test_values_land_on_their_faces_and_average_by_area(bound):
    result, _, vertices, faces, cp_face = bound
    assert result["faceCp"] == pytest.approx(cp_face[:50].astype(np.float32).tolist(), abs=1e-6)
    tri = vertices[faces].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    num = np.zeros(len(vertices))
    den = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(num, faces[:, k], area * cp_face)
        np.add.at(den, faces[:, k], area)
    used = den > 0
    assert np.allclose(np.array(result["vertexCp"])[used], (num / np.where(used, den, 1))[used], atol=1e-5)


def test_missing_regions_stay_missing(bound):
    assert bound[0]["summary"] == {"roof": -0.4}


def test_a_field_for_another_mesh_never_binds(bound):
    result = bound[0]
    assert result["otherHash"] == result["otherCount"] == result["perVertex"] == result["short"] == "FieldMismatch"
