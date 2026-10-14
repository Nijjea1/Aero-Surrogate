// 3D car viewer.  OWNER: Paramveer.
//
//   const viewer = createViewer(container, { onHover })
//   viewer.setMesh(payload)          // from POST /api/mesh (src/geometry/viewer.py)
//   viewer.setRegionValues(summary)  // Evaluation.pressure_summary
//   viewer.setMode("field" | "regions" | "plain")
//
// Geometry arrives in the car's frame (x nose to tail, y lateral, z up) and is
// turned into three.js's y-up frame here, centred on the car.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CP_RANGE = 0.8;   // saturate a little early so typical suction reads
const COLORS = {
  neg: new THREE.Color("#2a78d6"),      // suction
  pos: new THREE.Color("#e34948"),      // pressure
  mid: new THREE.Color("#c8c7c2"),      // Cp = 0
  none: new THREE.Color("#b9b8b2"),     // surface no region covers
  body: new THREE.Color("#c9ccd3"),
  tyre: new THREE.Color("#3a3a38"),
};

export function createViewer(container, { onHover } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6a6a66, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(-4, 6, 3);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(5, 2, -4);
  scene.add(fill);
  scene.add(groundShadow());

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(-4.4, 1.9, 3.9);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.6, 0);
  controls.enablePan = false;
  controls.minDistance = 3.2;
  controls.maxDistance = 16;
  controls.maxPolarAngle = Math.PI * 0.49;   // stay above the ground
  controls.update();
  controls.addEventListener("change", render);

  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.05 });
  let mesh = null;
  let data = null;          // decoded per-vertex fields
  let mode = "field";
  let regionValues = {};

  function setMesh(payload) {
    const vertices = decode(payload.vertices, Float32Array);
    const faces = decode(payload.faces, Uint32Array);
    const region = decode(payload.region, Int8Array);
    const cp = decode(payload.cp, Float32Array);

    const [lo, hi] = payload.bounds;
    const cx = (lo[0] + hi[0]) / 2;
    const cy = (lo[1] + hi[1]) / 2;
    const positions = new Float32Array(vertices.length);
    for (let i = 0; i < vertices.length; i += 3) {
      positions[i] = vertices[i] - cx;           // x: along the car
      positions[i + 1] = vertices[i + 2];        // y: up
      positions[i + 2] = -(vertices[i + 1] - cy); // z: lateral
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(positions.length), 3));
    geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    geometry.computeVertexNormals();

    if (mesh) {
      mesh.geometry.dispose();
      mesh.geometry = geometry;
    } else {
      mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);
    }
    data = { region, cp, regions: payload.regions };
    paint();
  }

  function setRegionValues(summary) {
    regionValues = summary || {};
    if (mode === "regions") paint();
  }

  function setMode(next) {
    mode = next;
    paint();
  }

  function valueAt(vertex) {
    const r = data.region[vertex];
    const name = r >= 0 ? data.regions[r] : null;
    if (mode === "field") return { name, value: data.cp[vertex] };
    if (mode === "regions") return { name, value: name && typeof regionValues[name] === "number" ? regionValues[name] : null };
    return { name, value: null };
  }

  function paint() {
    if (!mesh) return;
    const colors = mesh.geometry.getAttribute("color");
    const c = new THREE.Color();
    for (let v = 0; v < data.region.length; v++) {
      const { name, value } = valueAt(v);
      if (mode === "plain") c.copy(name === "wheel_wells" ? COLORS.tyre : COLORS.body);
      else if (value === null) c.copy(COLORS.none);
      else diverging(value, c);
      colors.setXYZ(v, c.r, c.g, c.b);
    }
    colors.needsUpdate = true;
    render();
  }

  // ---- hover: which region, what value, under the pointer
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hoverQueued = false;
  renderer.domElement.addEventListener("pointermove", (event) => {
    if (!onHover || hoverQueued || event.buttons) return;
    hoverQueued = true;
    requestAnimationFrame(() => {
      hoverQueued = false;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = mesh && raycaster.intersectObject(mesh)[0];
      onHover(hit ? { ...valueAt(hit.face.a), mode } : null, event);
    });
  });
  renderer.domElement.addEventListener("pointerleave", () => onHover?.(null));

  // ---- sizing & drawing
  function resize() {
    const { clientWidth: w, clientHeight: h } = container;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  }
  new ResizeObserver(resize).observe(container);

  function render() {
    renderer.render(scene, camera);
  }

  function resetView() {
    camera.position.set(-4.4, 1.9, 3.9);
    controls.target.set(0, 0.6, 0);
    controls.update();
  }

  resize();
  return { setMesh, setRegionValues, setMode, resetView };
}

// Blue below zero, red above, a neutral gray midpoint.
function diverging(value, out) {
  const t = Math.max(-1, Math.min(1, value / CP_RANGE));
  return t < 0 ? out.copy(COLORS.mid).lerp(COLORS.neg, -t) : out.copy(COLORS.mid).lerp(COLORS.pos, t);
}

function decode(b64, Type) {
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  return new Type(bytes.buffer);
}

// A soft contact shadow under the car, so it sits on something.
function groundShadow() {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(0,0,0,0.35)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(6.4, 3.2),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = 0.002;
  return plane;
}
