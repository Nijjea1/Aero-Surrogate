// Binds POST /fields predictions to the car on screen.  OWNER: Paramveer.
//
//   const id = await meshIdentity(payload)     // payload from POST /api/mesh
//   const field = bindField(response, id, payload)   // throws FieldMismatch
//
// The service predicts every face of the same mesh_payload the viewer draws
// (src/service/INTEGRATION.md, "Surface fields and viewer binding"). A field
// is bound only when its mesh hash and vertex/face counts match the mesh on
// screen. The hash is SHA-256 over "viewer-f32le-u32le-v1\0", the decoded
// vertex bytes (float32 XYZ) and the decoded face bytes (uint32 triangles).
//
// The viewer colours vertices, so each vertex takes the area-weighted mean of
// the faces around it. That is a display conversion: the per-face values stay
// available for the hover card, and averaged sigma is only shading, not
// uncertainty propagated by any validated rule.

export const HASH_FORMAT = "viewer-f32le-u32le-v1";

export class FieldMismatch extends Error {}

export function base64Bytes(b64) {
  const text = atob(b64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

export async function meshIdentity(payload) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new FieldMismatch("this page is not served over HTTPS or localhost, so it cannot check the mesh hash");
  const vertices = base64Bytes(payload.vertices);
  const faces = base64Bytes(payload.faces);
  const prefix = new TextEncoder().encode(`${HASH_FORMAT}\0`);
  const all = new Uint8Array(prefix.length + vertices.length + faces.length);
  all.set(prefix, 0);
  all.set(vertices, prefix.length);
  all.set(faces, prefix.length + vertices.length);
  const digest = new Uint8Array(await subtle.digest("SHA-256", all));
  return {
    sha256: Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join(""),
    nVertices: vertices.length / 12,
    nFaces: faces.length / 12,
  };
}

export function bindField(response, identity, payload) {
  const mesh = response?.mesh;
  if (!mesh || mesh.hash_format !== HASH_FORMAT) throw new FieldMismatch(`unknown mesh hash format ${mesh?.hash_format}`);
  if (mesh.association !== "face") throw new FieldMismatch(`values are per ${mesh.association}, not per face`);
  if (mesh.sha256 !== identity.sha256 || mesh.n_vertices !== identity.nVertices || mesh.n_faces !== identity.nFaces) {
    throw new FieldMismatch("the prediction is for a different mesh than the one on screen");
  }
  const { cp, sigma, region } = response;
  const index = mesh.face_indices;
  const n = identity.nFaces;
  if (![cp, sigma, region, index].every((a) => Array.isArray(a) && a.length === index.length)) {
    throw new FieldMismatch("cp, sigma, region and face_indices do not line up");
  }

  const faceCp = new Float32Array(n).fill(NaN);
  const faceSigma = new Float32Array(n).fill(NaN);
  const faceRegion = new Int16Array(n).fill(-1);
  for (let i = 0; i < index.length; i++) {
    const f = index[i];
    if (!Number.isInteger(f) || f < 0 || f >= n) throw new FieldMismatch(`face index ${f} is out of range`);
    faceCp[f] = cp[i];
    faceSigma[f] = sigma[i];
    faceRegion[f] = Number.isInteger(region[i]) ? region[i] : -1;
  }

  const vertices = new Float32Array(base64Bytes(payload.vertices).buffer);
  const faces = new Uint32Array(base64Bytes(payload.faces).buffer);
  const { vertexCp, vertexSigma } = faceToVertex(vertices, faces, faceCp, faceSigma);
  const summary = {};
  for (const [name, value] of Object.entries(response.pressure_summary || {})) {
    if (typeof value === "number" && Number.isFinite(value)) summary[name] = value;
  }
  return {
    faceCp, faceSigma, faceRegion, vertexCp, vertexSigma, summary,
    regions: response.regions || [],
    sigmaScale: percentile(faceSigma, 0.95),
    provenance: response.provenance || {},
    timing: response.timing || null,
    sha256: mesh.sha256,
  };
}

// Area-weighted mean of the faces around each vertex; NaN where no face has a value.
export function faceToVertex(vertices, faces, faceCp, faceSigma) {
  const nv = vertices.length / 3;
  const cpSum = new Float64Array(nv);
  const sigmaSum = new Float64Array(nv);
  const weight = new Float64Array(nv);
  for (let f = 0; f < faces.length / 3; f++) {
    if (!Number.isFinite(faceCp[f])) continue;
    const a = faces[f * 3] * 3, b = faces[f * 3 + 1] * 3, c = faces[f * 3 + 2] * 3;
    const ux = vertices[b] - vertices[a], uy = vertices[b + 1] - vertices[a + 1], uz = vertices[b + 2] - vertices[a + 2];
    const vx = vertices[c] - vertices[a], vy = vertices[c + 1] - vertices[a + 1], vz = vertices[c + 2] - vertices[a + 2];
    const area = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    const cp = area * faceCp[f], sigma = area * faceSigma[f];
    for (let k = 0; k < 3; k++) {
      const v = faces[f * 3 + k];
      cpSum[v] += cp;
      sigmaSum[v] += sigma;
      weight[v] += area;
    }
  }
  const vertexCp = new Float32Array(nv);
  const vertexSigma = new Float32Array(nv);
  for (let v = 0; v < nv; v++) {
    vertexCp[v] = weight[v] > 0 ? cpSum[v] / weight[v] : NaN;
    vertexSigma[v] = weight[v] > 0 ? sigmaSum[v] / weight[v] : NaN;
  }
  return { vertexCp, vertexSigma };
}

function percentile(values, q) {
  const finite = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return 0;
  return finite[Math.min(finite.length - 1, Math.floor(q * finite.length))];
}
