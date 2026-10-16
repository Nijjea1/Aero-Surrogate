// 3D car viewer.  OWNER: Paramveer.
//
//   const viewer = createViewer(container, { onHover })
//   viewer.setMesh(payload)          // the design, from POST /api/mesh (src/geometry/viewer.py)
//   viewer.setRegionValues(summary)  // region name -> mean Cp
//   viewer.setPressureSource("illustrative" | "predicted")
//   viewer.setField(field | null)    // a bound /fields prediction (fieldmap.js) for the car on screen
//   viewer.setMode("plain" | "field" | "sigma" | "regions")
//   viewer.setPaint("#c8ccd2")       // studio paint colour
//   viewer.setStage("studio" | "none")
//   viewer.setAnimator(fn | null)    // fn(dt, t) every frame; null renders on demand
//   viewer.setView("orbit" | "side" | "chase" | "tunnel", { animate })
//   viewer.addCar(payload, opts) / viewer.removeCar(car)   // a second car, for the race
//   viewer.onRender(fn), viewer.onResize(fn)
//
// Geometry arrives in the car's frame (x nose to tail, y lateral, z up) and is
// turned into three.js's y-up frame here, centred on the car: X = x - cx,
// Y = z, Z = -(y - cy). car.frame says what cx and cy were.
//
// "plain" is the studio look: real materials per part (paint, glass, tyres,
// lights, trim). "field" paints the whole car with the pressure coefficient:
// predicted face by face when the service serves /fields, else illustrative
// from the shape. "sigma" paints the field model's Cp uncertainty, and
// "regions" each region's mean Cp.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const CP_RANGE = 0.6;   // saturate early so typical suction and pressure both read
const COLORS = {
  neg: new THREE.Color("#4f86e0"),      // suction
  pos: new THREE.Color("#e0553f"),      // pressure
  mid: new THREE.Color("#a9adb4"),      // Cp = 0
  none: new THREE.Color("#5d6672"),     // surface no region covers, or no value yet
  sigmaLo: new THREE.Color("#3a2a0c"),  // Cp uncertainty: one amber hue, dark to light
  sigmaHi: new THREE.Color("#ffd27a"),
};
export const PAINTS = {
  "Liquid silver": "#c9ced6",
  Graphite: "#2b3036",
  "Racing red": "#c4141b",
  "Arctic white": "#eef1f4",
  "Signal orange": "#e8551c",
};
const PART = { paint: 0, glass: 1, tyre: 2, headlight: 3, taillight: 4, trim: 5 };
// Draw groups in the studio look. Paint, glass and trim share one clear-coated
// material and differ by vertex colour, so their edges blend across a
// triangle instead of stepping triangle by triangle.
const SLOT = { [PART.paint]: 0, [PART.glass]: 0, [PART.trim]: 0, [PART.tyre]: 1, [PART.headlight]: 2, [PART.taillight]: 3 };
const SURFACE = { [PART.glass]: new THREE.Color("#04070b"), [PART.trim]: new THREE.Color("#0a0c0f"), [PART.tyre]: new THREE.Color("#0f1012") };

export function createViewer(container, { onHover } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  let pixelRatio = Math.min(window.devicePixelRatio, 2);
  renderer.setPixelRatio(pixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.92;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  // Studio reflections for the clear coat and the glass.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
  pmrem.dispose();

  const hemi = new THREE.HemisphereLight(0x9fb7d0, 0x07090c, 0.45);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(-4, 7, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = key.shadow.camera.bottom = -4.5;
  key.shadow.camera.right = key.shadow.camera.top = 4.5;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 20;
  key.shadow.bias = -0.0004;
  key.shadow.radius = 4;
  scene.add(key);
  const rimCool = new THREE.DirectionalLight(0xc8d6ff, 1.0);   // cool edge light from behind
  rimCool.position.set(6, 3, -5);
  scene.add(rimCool);
  const rimWarm = new THREE.DirectionalLight(0xffd2a8, 0.5);
  rimWarm.position.set(-6, 2, -4);
  scene.add(rimWarm);

  const studio = makeStudio();
  scene.add(studio);

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 400);
  camera.position.set(-4.4, 1.9, 3.9);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.6, 0);
  controls.enablePan = false;
  controls.enableDamping = false;
  controls.minDistance = 3.2;
  controls.maxDistance = 18;
  controls.maxPolarAngle = Math.PI * 0.49;   // stay above the ground
  controls.update();
  controls.addEventListener("change", () => render());

  // Bloom only lifts what is already bright: lights, the gantry, the smoke.
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.38, 0.5, 0.9);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  let useComposer = true;

  let car = null;              // the design
  const cars = new Set();
  let mode = "plain";
  let paint = PAINTS["Liquid silver"];
  let regionValues = {};
  let pressureSource = "illustrative";

  function setMesh(payload) {
    const next = createCar(payload, { paint });
    next.group.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    if (car) { scene.remove(car.group); cars.delete(car); car.dispose(); }
    car = next;
    car.pressureSource = pressureSource;
    car.setEnvScale(envScale);
    cars.add(car);
    scene.add(car.group);
    car.apply(mode, regionValues);
    api.car = car;
    api.frame = car.frame;
    render();
  }

  function addCar(payload, opts = {}) {
    const extra = createCar(payload, { paint: opts.paint ?? PAINTS.Graphite });
    extra.group.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    extra.apply("plain", {});
    extra.setEnvScale(envScale);
    cars.add(extra);
    scene.add(extra.group);
    render();
    return extra;
  }

  function removeCar(extra) {
    if (!extra || extra === car) return;
    scene.remove(extra.group);
    cars.delete(extra);
    extra.dispose();
    render();
  }

  function setRegionValues(summary) {
    regionValues = summary || {};
    if (car && mode === "regions") car.apply(mode, regionValues);
    render();
  }

  function setMode(next) {
    mode = next;
    car?.apply(mode, regionValues);
    render();
  }

  // "predicted": the Pressure paint waits (grey) for setField rather than
  // falling back to the illustrative shape pressure.
  function setPressureSource(source) {
    pressureSource = source;
    for (const c of cars) c.pressureSource = source;
    car?.apply(mode, regionValues);
    render();
  }

  function setField(field) {
    if (!car) return;
    car.setField(field);
    car.apply(mode, regionValues);
    render();
  }

  function setPaint(hex) {
    paint = hex;
    car?.setPaint(hex);
    render();
  }

  // The studio is lit like a showroom; outdoors (road, tunnel) the key light
  // turns to a cool moon and reflections dim, so the paint does not glow white
  // against a night sky. The race is floodlit, so both paints read at a glance.
  let envScale = 0.55;
  function setStage(name) {
    const indoor = name === "studio";
    const floodlit = name === "race";
    studio.visible = indoor;
    envScale = indoor ? 0.55 : floodlit ? 0.75 : 0.4;
    key.intensity = indoor ? 1.5 : floodlit ? 1.9 : 1.15;
    key.color.set(indoor || floodlit ? "#ffffff" : "#cfdcff");
    hemi.intensity = floodlit ? 0.55 : 0.3;
    rimCool.intensity = indoor ? 1.1 : floodlit ? 0.9 : 0.7;
    for (const c of cars) c.setEnvScale(envScale);
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
      if (!car) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(car.body, false)[0];
      onHover(hit ? { ...car.valueAt(hit.face.a, mode, regionValues, hit.faceIndex), mode, point: hit.point } : null, event);
    });
  });
  renderer.domElement.addEventListener("pointerleave", () => onHover?.(null));

  // ---- sizing & drawing
  const resizeListeners = new Set();
  const renderListeners = new Set();
  const size = new THREE.Vector2();
  function resize() {
    const { clientWidth: w, clientHeight: h } = container;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.fov = camera.aspect < 1 ? 50 : camera.aspect < 1.4 ? 40 : 32;   // portrait screens see the whole car
    camera.updateProjectionMatrix();
    renderer.getSize(size);
    for (const fn of resizeListeners) fn(w * pixelRatio, h * pixelRatio, w, h);
    render();
  }
  new ResizeObserver(resize).observe(container);

  function render() {
    if (useComposer) composer.render();
    else renderer.render(scene, camera);
    for (const fn of renderListeners) fn(camera);
  }

  // ---- animation: on demand by default, every frame while an animator runs
  let animator = null;
  let looping = false;   // one frame loop at most, however often the animator changes
  let visible = true;
  let last = 0;
  let slowFrames = 0;
  // workMs is the time a frame costs us (simulation plus drawing), not the
  // gap between frames: a window in the background is throttled by the
  // browser, and that must never be mistaken for a slow machine.
  const perf = { workMs: 8, quality: "high" };
  function frame(now) {
    if (!animator || !visible) { looping = false; return; }
    // The true frame time, so anything keyed to the clock (the race) keeps
    // real time however slow the frames. A gap of a second or more is a
    // hidden or stalled tab, not a frame: it counts as nothing.
    const gap = (now - last) / 1000;
    const dt = gap >= 1 ? 0 : gap;
    last = now;
    const started = performance.now();
    animator(dt, now / 1000);
    tween?.(now);
    render();
    perf.workMs = 0.92 * perf.workMs + 0.08 * (performance.now() - started);
    adaptQuality();
    requestAnimationFrame(frame);
  }
  function startLoop() {
    if (animator && visible && !looping) {
      looping = true;
      last = performance.now();
      requestAnimationFrame(frame);
    }
  }
  function setAnimator(fn) {
    animator = fn;
    startLoop();
    if (!fn) render();
  }
  // Nothing is drawn while the stage is scrolled out of sight.
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    startLoop();
  }).observe(container);

  // Frames consistently slow: drop bloom, then resolution. Never back up
  // mid-session, so the picture does not flicker between qualities.
  function adaptQuality() {
    slowFrames = perf.workMs > 24 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
    if (slowFrames < 90) return;
    slowFrames = 0;
    if (useComposer) {
      useComposer = false;
      perf.quality = "no bloom";
    } else if (pixelRatio > 1) {
      pixelRatio = 1;
      renderer.setPixelRatio(1);
      perf.quality = "1x";
      resize();
    }
  }

  const VIEWS = {
    orbit: [[-5.6, 2.0, 5.3], [0.1, 0.55, 0]],
    side: [[0.4, 1.0, 8.6], [0.4, 0.75, 0]],
    chase: [[-6.3, 2.2, 4.6], [0.35, 0.62, 0]],
    tunnel: [[-5.0, 2.9, 6.6], [0.8, 0.6, 0]],
  };
  let tween = null;
  function setView(name = "orbit", { animate = false } = {}) {
    const [position, target] = VIEWS[name] || VIEWS.orbit;
    if (!animate) {
      tween = null;
      camera.position.set(...position);
      controls.target.set(...target);
      controls.update();
      render();
      return;
    }
    const p0 = camera.position.clone(), t0 = controls.target.clone();
    const p1 = new THREE.Vector3(...position), t1 = new THREE.Vector3(...target);
    const start = performance.now();
    tween = (now) => {
      const k = Math.min(1, (now - start) / 700);
      const e = 1 - (1 - k) ** 3;
      camera.position.lerpVectors(p0, p1, e);
      controls.target.lerpVectors(t0, t1, e);
      controls.update();
      if (k >= 1) tween = null;
    };
    if (!looping) {
      const step = (now) => { tween?.(now); render(); if (tween) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    }
  }

  resize();
  const api = {
    setMesh, addCar, removeCar, setRegionValues, setPressureSource, setField, setMode, setPaint, setStage, setAnimator, setView, render,
    resetView: () => setView("orbit", { animate: true }),
    onRender: (fn) => renderListeners.add(fn),
    onResize: (fn) => { resizeListeners.add(fn); fn(size.x * pixelRatio, size.y * pixelRatio, size.x, size.y); },
    scene, camera, controls, renderer, perf, car: null, frame: null,
    get mode() { return mode; },
    get paint() { return paint; },
  };
  return api;
}

// ---------------------------------------------------------------- one car

export function createCar(payload, { paint = PAINTS["Liquid silver"] } = {}) {
  const vertices = decode(payload.vertices, Float32Array);
  const faces = decode(payload.faces, Uint32Array);
  const region = decode(payload.region, Int8Array);
  const cp = decode(payload.cp, Float32Array);
  const part = payload.part ? decode(payload.part, Int8Array) : new Int8Array(region.length);

  const [lo, hi] = payload.bounds;
  const cx = (lo[0] + hi[0]) / 2;
  const cy = (lo[1] + hi[1]) / 2;
  const positions = new Float32Array(vertices.length);
  for (let i = 0; i < vertices.length; i += 3) {
    positions[i] = vertices[i] - cx;            // x: along the car
    positions[i + 1] = vertices[i + 2];         // y: up
    positions[i + 2] = -(vertices[i + 1] - cy); // z: lateral
  }

  // Faces sorted by draw group, one material each in the studio look.
  const nFaces = faces.length / 3;
  const order = Array.from({ length: nFaces }, (_, f) => f);
  const facePart = (f) => SLOT[Math.max(0, part[faces[f * 3]])] ?? 0;
  order.sort((a, b) => facePart(a) - facePart(b));
  const index = new Uint32Array(faces.length);
  order.forEach((f, k) => index.set(faces.subarray(f * 3, f * 3 + 3), k * 3));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(positions.length), 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  let start = 0;
  for (let p = 0; p <= 3; p++) {
    let count = 0;
    while (start + count < nFaces && facePart(order[start + count]) === p) count++;
    if (count) geometry.addGroup(start * 3, count * 3, p);
    start += count;
  }

  const materials = studioMaterials();
  const field = new THREE.MeshPhysicalMaterial({
    vertexColors: true, roughness: 0.5, metalness: 0.0, clearcoat: 0.35, clearcoatRoughness: 0.25, envMapIntensity: 0.35,
  });
  const body = new THREE.Mesh(geometry, materials);
  const group = new THREE.Group();
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), CONTACT_MATERIAL);   // travels with the car
  shadow.scale.set((hi[0] - lo[0]) * 1.35, (hi[1] - lo[1]) * 1.6, 1);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.004;
  group.add(shadow, body);

  // Spoked rims on both faces of every wheel, turned by spin().
  const flow = payload.flow;
  const rims = [];
  const rimGeometry = new THREE.CircleGeometry(flow.wheel_radius * 0.7, 48);
  for (const x of flow.axles) {
    for (const side of [1, -1]) {
      const rim = new THREE.Mesh(rimGeometry, RIM_MATERIAL);
      rim.position.set(x - cx, flow.wheel_radius, side * (flow.wheel_y + flow.tyre_width / 2 + 0.004) + cy);
      if (side < 0) rim.rotateY(Math.PI);
      rim.userData.side = side;
      rims.push(rim);
      group.add(rim);
    }
  }

  // Where each region's stats are pinned: its mean surface point, nudged out
  // along its mean normal so the pin floats just off the panel.
  const normals = geometry.getAttribute("normal");
  const anchors = {};
  const sums = payload.regions.map(() => ({ p: new THREE.Vector3(), n: new THREE.Vector3(), k: 0 }));
  for (let v = 0; v < region.length; v++) {
    const r = region[v];
    if (r < 0) continue;
    const s = sums[r];
    s.p.x += positions[v * 3]; s.p.y += positions[v * 3 + 1]; s.p.z += positions[v * 3 + 2];
    s.n.x += normals.getX(v); s.n.y += normals.getY(v); s.n.z += normals.getZ(v);
    s.k++;
  }
  payload.regions.forEach((name, r) => {
    const s = sums[r];
    if (!s.k) return;
    const n = s.n.normalize();
    const p = s.p.divideScalar(s.k);
    if (name === "wheel_wells") {
      // The front wheel's outer face, where the eye looks for it.
      p.set(flow.axles[0] - cx, flow.wheel_radius * 1.15, flow.wheel_y + flow.tyre_width + 0.05 + cy);
      n.set(0, 0, 1);
    }
    if (name === "underbody") {
      // Under the near-side sill, so it can be seen from above.
      p.set(p.x, 0.1, (hi[1] - lo[1]) / 2 - 0.05);
      n.set(0, 0, 1);
    }
    anchors[name] = { position: p.add(n.clone().multiplyScalar(0.12)), normal: n };
  });

  // Mean illustrative Cp per region, for pins when the model predicts none.
  const fieldMeans = {};
  payload.regions.forEach((name, r) => {
    let sum = 0, k = 0;
    for (let v = 0; v < region.length; v++) if (region[v] === r) { sum += cp[v]; k++; }
    if (k) fieldMeans[name] = sum / k;
  });

  // A bound /fields prediction (fieldmap.js), or null.
  let bound = null;
  const self = { pressureSource: "illustrative" };

  // What the paint shows at a vertex; with sortedFace (a raycast hit's
  // faceIndex) the predicted field answers for that exact face.
  function valueAt(vertex, mode, regionValues, sortedFace = null) {
    const r = region[vertex];
    const name = r >= 0 ? payload.regions[r] : null;
    if (mode === "regions") {
      return { name, value: name && typeof regionValues[name] === "number" ? regionValues[name] : null, source: "regions" };
    }
    if (mode === "field" && self.pressureSource !== "predicted") return { name, value: cp[vertex], source: "illustrative" };
    if (!bound) return { name, value: null, sigma: null, source: "pending" };
    if (sortedFace !== null && sortedFace !== undefined) {
      const f = order[sortedFace];
      const learned = bound.regions[bound.faceRegion[f]] ?? null;
      return { name: learned, value: finite(bound.faceCp[f]), sigma: finite(bound.faceSigma[f]), source: "predicted", face: f };
    }
    return { name, value: finite(bound.vertexCp[vertex]), sigma: finite(bound.vertexSigma[vertex]), source: "predicted" };
  }

  // The mesh grid is a few centimetres coarse, so a paint/glass edge drawn
  // vertex by vertex steps like a staircase. Two passes of neighbour
  // averaging over the shared clear-coat surface round it into a clean line.
  const surface = Uint8Array.from(part, (p) => (SLOT[Math.max(0, p)] ?? 0) === 0 ? 1 : 0);
  function smoothSurface(c, passes = 2) {
    const sum = new Float32Array(c.length);
    const count = new Float32Array(part.length);
    for (let pass = 0; pass < passes; pass++) {
      sum.set(c);
      count.fill(1);
      for (let f = 0; f < faces.length; f += 3) {
        for (let k = 0; k < 3; k++) {
          const a = faces[f + k], b = faces[f + ((k + 1) % 3)];
          if (!surface[a] || !surface[b]) continue;
          sum[a * 3] += c[b * 3]; sum[a * 3 + 1] += c[b * 3 + 1]; sum[a * 3 + 2] += c[b * 3 + 2]; count[a]++;
          sum[b * 3] += c[a * 3]; sum[b * 3 + 1] += c[a * 3 + 1]; sum[b * 3 + 2] += c[a * 3 + 2]; count[b]++;
        }
      }
      for (let v = 0; v < part.length; v++) {
        if (!surface[v]) continue;
        c[v * 3] = sum[v * 3] / count[v]; c[v * 3 + 1] = sum[v * 3 + 1] / count[v]; c[v * 3 + 2] = sum[v * 3 + 2] / count[v];
      }
    }
  }

  let current = "plain";
  const paintColor = new THREE.Color(paint);
  function apply(mode, regionValues) {
    current = mode;
    const colors = geometry.getAttribute("color");
    if (mode === "plain") {
      for (let v = 0; v < part.length; v++) {
        const c = SURFACE[part[v]] || paintColor;
        colors.setXYZ(v, c.r, c.g, c.b);
      }
      smoothSurface(colors.array);
      colors.needsUpdate = true;
      body.material = materials;
      return;
    }
    const c = new THREE.Color();
    const scale = bound?.sigmaScale > 0 ? bound.sigmaScale : 1;
    for (let v = 0; v < region.length; v++) {
      const { value, sigma } = valueAt(v, mode, regionValues);
      if (mode === "sigma") {
        if (sigma === null || sigma === undefined) c.copy(COLORS.none);
        else c.copy(COLORS.sigmaLo).lerp(COLORS.sigmaHi, Math.min(1, sigma / scale));
      } else if (value === null) c.copy(COLORS.none);
      else diverging(value, c);
      colors.setXYZ(v, c.r, c.g, c.b);
    }
    colors.needsUpdate = true;
    body.material = field;
  }

  function setPaint(hex) {
    paintColor.set(hex);
    if (current === "plain") apply("plain", {});
  }

  const baseEnv = [...materials, field].map((m) => m.envMapIntensity);
  function setEnvScale(k) {
    [...materials, field].forEach((m, i) => { m.envMapIntensity = baseEnv[i] * k; });
  }

  function spin(angle) {
    for (const rim of rims) rim.rotation.z = rim.userData.side * angle;
  }

  function dispose() {
    geometry.dispose();
    rimGeometry.dispose();
    field.dispose();
    for (const m of materials) m.dispose();
  }

  return Object.assign(self, {
    group, body, rims, anchors, fieldMeans, apply, setPaint, setEnvScale, spin, valueAt, dispose,
    setField(next) { bound = next; },
    get field() { return bound; },
    frame: { cx, cy, length: hi[0] - lo[0], height: hi[2] - lo[2], width: hi[1] - lo[1] },
    payload,
  });
}

const finite = (x) => (Number.isFinite(x) ? x : null);

function studioMaterials() {
  return [
    new THREE.MeshPhysicalMaterial({   // paint, glass, trim: metallic base under a clear coat
      vertexColors: true, metalness: 0.65, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 1.0,
    }),
    new THREE.MeshStandardMaterial({ color: "#111214", roughness: 0.88, metalness: 0 }),   // tyre
    new THREE.MeshStandardMaterial({   // headlight
      color: "#dfeeff", emissive: "#cfe6ff", emissiveIntensity: 1.25, roughness: 0.15,
    }),
    new THREE.MeshStandardMaterial({   // taillight
      color: "#3a0000", emissive: "#ff1e1e", emissiveIntensity: 1.6, roughness: 0.3,
    }),
  ];
}

// Blue below zero, red above, a neutral light midpoint.
function diverging(value, out) {
  const t = Math.max(-1, Math.min(1, value / CP_RANGE));
  return t < 0 ? out.copy(COLORS.mid).lerp(COLORS.neg, -t) : out.copy(COLORS.mid).lerp(COLORS.pos, t);
}

export function cpColor(value) {
  return "#" + diverging(value, new THREE.Color()).getHexString();
}

function decode(b64, Type) {
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  return new Type(bytes.buffer);
}

// ---------------------------------------------------------------- the studio

// A glossy dark floor that fades into the backdrop, and a faint turntable
// ring. Each car brings its own contact shadow.
function makeStudio() {
  const group = new THREE.Group();
  const fade = radialTexture([[0, "rgba(255,255,255,1)"], [0.55, "rgba(255,255,255,0.9)"], [1, "rgba(255,255,255,0)"]]);
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(14, 96),
    new THREE.MeshStandardMaterial({ color: "#07090c", roughness: 0.85, metalness: 0, envMapIntensity: 0.2, alphaMap: fade, transparent: true }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(3.15, 3.19, 160),
    new THREE.MeshBasicMaterial({ color: "#8a8d93", transparent: true, opacity: 0.2 }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.003;
  group.add(floor, ring);
  return group;
}

const CONTACT_MATERIAL = new THREE.MeshBasicMaterial({
  map: radialTexture([[0, "rgba(0,0,0,0.6)"], [0.6, "rgba(0,0,0,0.3)"], [1, "rgba(0,0,0,0)"]]), transparent: true, depthWrite: false,
});

function radialTexture(stops) {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [at, colour] of stops) g.addColorStop(at, colour);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

const RIM_MATERIAL = new THREE.MeshStandardMaterial({ map: rimTexture(), metalness: 0.75, roughness: 0.28, envMapIntensity: 1.4 });

function rimTexture() {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const c = size / 2;
  const metal = ctx.createRadialGradient(c, c, 10, c, c, c);
  metal.addColorStop(0, "#e4e7ec");
  metal.addColorStop(0.8, "#a9aeb6");
  metal.addColorStop(1, "#6d727a");
  ctx.fillStyle = metal;
  ctx.beginPath(); ctx.arc(c, c, c - 1, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#121315";
  for (let k = 0; k < 5; k++) {          // five gaps between five spokes
    const a = (k / 5) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(c + Math.cos(a + 0.2) * 36, c + Math.sin(a + 0.2) * 36);
    ctx.arc(c, c, c - 16, a + 0.3, a + (Math.PI * 2) / 5 - 0.3);
    ctx.lineTo(c + Math.cos(a + (Math.PI * 2) / 5 - 0.2) * 36, c + Math.sin(a + (Math.PI * 2) / 5 - 0.2) * 36);
    ctx.fill();
  }
  ctx.fillStyle = "#3a3d42";
  ctx.beginPath(); ctx.arc(c, c, 24, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#2a2c30";
  ctx.lineWidth = 6;
  ctx.beginPath(); ctx.arc(c, c, c - 4, 0, Math.PI * 2); ctx.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}
