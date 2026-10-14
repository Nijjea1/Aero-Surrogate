// 3D car viewer.  OWNER: Paramveer.
//
//   const viewer = createViewer(container, { onHover })
//   viewer.setMesh(payload)          // from POST /api/mesh (src/geometry/viewer.py)
//   viewer.setRegionValues(summary)  // Evaluation.pressure_summary
//   viewer.setMode("field" | "regions" | "plain")
//   viewer.setAnimator(fn | null)    // fn(dt, t) every frame; null renders on demand
//   viewer.setView("orbit" | "side" | "chase")
//
// Geometry arrives in the car's frame (x nose to tail, y lateral, z up) and is
// turned into three.js's y-up frame here, centred on the car: X = x - cx,
// Y = z, Z = -(y - cy). viewer.frame says what cx and cy were.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

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
  // Soft studio reflections, so the body reads as a solid in three dimensions
  // rather than a flat-shaded shape.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
  pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6a6a66, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(-4, 6, 3);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(5, 2, -4);
  scene.add(fill);
  scene.add(groundShadow());

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
  camera.position.set(-4.4, 1.9, 3.9);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.6, 0);
  controls.enablePan = false;
  controls.minDistance = 3.2;
  controls.maxDistance = 16;
  controls.maxPolarAngle = Math.PI * 0.49;   // stay above the ground
  controls.update();
  controls.addEventListener("change", render);

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.42, metalness: 0.1, envMapIntensity: 0.55,
  });
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
    api.frame = { cx, cy, length: hi[0] - lo[0] };
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

  // ---- animation: on demand by default, every frame while an animator runs
  let animator = null;
  let looping = false;   // one frame loop at most, however often the animator changes
  let last = 0;
  function frame(now) {
    if (!animator) { looping = false; return; }
    const dt = Math.min((now - last) / 1000, 0.05);  // a hidden tab must not jump
    last = now;
    animator(dt, now / 1000);
    render();
    requestAnimationFrame(frame);
  }

  function setAnimator(fn) {
    animator = fn;
    if (fn && !looping) {
      looping = true;
      last = performance.now();
      requestAnimationFrame(frame);
    }
    if (!fn) render();
  }

  const VIEWS = {
    orbit: [[-4.4, 1.9, 3.9], [0, 0.6, 0]],
    side: [[0.4, 1.0, 8.6], [0.4, 0.75, 0]],
    chase: [[-6.3, 2.2, 4.6], [0.35, 0.62, 0]],
    tunnel: [[-5.0, 2.9, 6.6], [0.8, 0.6, 0]],
  };
  function setView(name = "orbit") {
    const [position, target] = VIEWS[name] || VIEWS.orbit;
    camera.position.set(...position);
    controls.target.set(...target);
    controls.update();
    render();
  }

  resize();
  const api = {
    setMesh, setRegionValues, setMode, setAnimator, setView, render,
    resetView: () => setView("orbit"),
    scene, camera, controls, frame: null,
  };
  return api;
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
