// Airflow animation, driven by the prediction.  OWNER: Paramveer.
//
//   const flow = createFlow(viewer)
//   flow.setShape(payload.flow, viewer.frame)    // the outline from /api/mesh
//   flow.setPrediction({ cd, cl, sigma, threshold, trusted, pressure, baselineCd })
//   flow.setSpeed(kmh); flow.setMode("off" | "tunnel" | "drive"); flow.setPaused(bool)
//
// This is NOT a flow simulation, and the page says so. It is an illustration
// whose every moving part is set by a number the surrogate predicted:
//
//   path     each streamline bends around the car's real outline (side profile
//            and plan width at that station). Continuity sets how far: the air
//            between the body and a streamline keeps its cross-section, so a
//            far-field offset rho sits at r = sqrt(1 + rho^2) body-radii. Close
//            in it hugs the surface, far out it runs straight
//   colour   the predicted mean Cp of the region the air is passing, fading with
//            distance from the surface: blue suction, red pressure, the same
//            scale as the car's paint
//   speed    from that Cp by Bernoulli, |v| / U = sqrt(1 - Cp): air speeds up
//            where the model predicts suction and slows where it predicts pressure
//   wake     its size, churn and length grow with predicted Cd against the
//            baseline body's; it is pushed down by predicted lift (the air
//            carries the reaction) and up by downforce
//   fuzz     the smoke trembles in proportion to predicted sigma, and turns grey
//            when the gate rejects the prediction

import * as THREE from "three";

const SLOW_MOTION = 0.15;          // screen speed / road speed
const STEP = 0.02;                 // m, outline grid
const LOOKAHEAD = 0.6;             // m, streamlines start bending this far ahead
const UPSTREAM = 2.8;              // m ahead of the nose where smoke is released
const DOWNSTREAM = 7.0;            // m behind the tail where it is retired
const EXP = 3.0;                   // superellipse sections, as in src/geometry
const CP_RANGE = 0.5;              // smoke saturates sooner than paint, so it reads at a glance
const RAKE_HEIGHTS = [0.14, 0.3, 0.48, 0.66, 0.84, 1.02, 1.22, 1.45, 1.7];
const RAKE_LANES = [0, 0.45, -0.45, 0.9, -0.9];   // across the car, so the smoke wraps the sides too
const RAKE_SPACING = 0.08;         // m of travel between puffs from one nozzle
const RAKE_SLOTS = 220;            // puffs alive per nozzle
const VOLUME_PARTICLES = 1100;     // drive mode: free particles drawn as streaks
const STREAK_SECONDS = 0.12;       // drive mode: a streak is where the particle was this long ago
const CAPACITY = Math.max(RAKE_HEIGHTS.length * RAKE_LANES.length * RAKE_SLOTS, VOLUME_PARTICLES);

const COLOR = {
  neg: new THREE.Color("#2a78d6"),
  pos: new THREE.Color("#e34948"),
  mid: new THREE.Color("#a8a7a1"),
  grey: new THREE.Color("#9c9b96"),
};

export function createFlow(viewer) {
  const group = new THREE.Group();
  group.visible = false;
  viewer.scene.add(group);

  // ---- particle state, struct of arrays
  const px = new Float32Array(CAPACITY);
  const y0 = new Float32Array(CAPACITY);
  const z0 = new Float32Array(CAPACITY);
  const seed = new Float32Array(CAPACITY);
  const under = new Uint8Array(CAPACITY);
  const alive = new Uint8Array(CAPACITY);
  const seq = new Int32Array(CAPACITY);
  const cur = new Float32Array(CAPACITY * 3);
  const prev = new Float32Array(CAPACITY * 3);
  const rgba = new Float32Array(CAPACITY * 4);

  // ---- drawing: soft points plus line segments (streaklines or streaks)
  const pointGeo = new THREE.BufferGeometry();
  const pointPos = new THREE.BufferAttribute(new Float32Array(CAPACITY * 3), 3).setUsage(THREE.DynamicDrawUsage);
  const pointCol = new THREE.BufferAttribute(new Float32Array(CAPACITY * 4), 4).setUsage(THREE.DynamicDrawUsage);
  pointGeo.setAttribute("position", pointPos);
  pointGeo.setAttribute("color", pointCol);
  const points = new THREE.Points(pointGeo, new THREE.PointsMaterial({
    size: 0.05, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false,
  }));
  const lineGeo = new THREE.BufferGeometry();
  const linePos = new THREE.BufferAttribute(new Float32Array(CAPACITY * 6), 3).setUsage(THREE.DynamicDrawUsage);
  const lineCol = new THREE.BufferAttribute(new Float32Array(CAPACITY * 8), 4).setUsage(THREE.DynamicDrawUsage);
  lineGeo.setAttribute("position", linePos);
  lineGeo.setAttribute("color", lineCol);
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, depthWrite: false,
  }));
  points.renderOrder = lines.renderOrder = 2;
  points.frustumCulled = lines.frustumCulled = false;
  group.add(lines, points);

  // ---- road and wheel rims, drive mode only
  const road = makeRoad();
  const rims = [];
  const scenery = new THREE.Group();        // drive mode: the road world
  const roadside = makeRoadside();
  scenery.add(road.mesh, roadside.group);
  scenery.visible = false;
  viewer.scene.add(scenery);
  const floor = new THREE.GridHelper(18, 36, 0x8a8984, 0x8a8984);   // tunnel mode: depth cues
  floor.material.transparent = true;
  floor.material.opacity = 0.25;
  floor.position.set(1.5, 0.001, 0);
  floor.visible = false;
  viewer.scene.add(floor);
  let travel = 0;                           // m the road has moved, for the roadside
  let warmed = false;
  let clock = 0;     // simulation time: every wobble and swirl runs on this, never the wall clock

  // Drive mode: a slow camera sweep from front three-quarter to rear
  // three-quarter, looking down a little. Any drag hands the camera to the
  // viewer; the sweep picks up again a few seconds after they let go, from
  // where they left it.
  let autoCamera = true;
  let resumeAt = 0;
  let sweepClock = 0;
  viewer.controls.addEventListener("start", () => { autoCamera = false; resumeAt = 0; });
  viewer.controls.addEventListener("end", () => { resumeAt = performance.now() + 6000; });

  let shape = null;     // outline on a grid, set by setShape
  let field = null;     // prediction-driven grids and wake numbers, set by setPrediction
  let prediction = null;
  let mode = "off";
  let paused = false;
  let speedKmh = 110;
  let wheelAngle = 0;
  const nozzleCarry = new Float32Array(RAKE_HEIGHTS.length * RAKE_LANES.length);
  const nozzleSeq = new Int32Array(RAKE_HEIGHTS.length * RAKE_LANES.length);
  let rng = mulberry32(7);

  // ------------------------------------------------------------ the outline

  function setShape(outline, frame) {
    if (!outline || !frame) return;
    const xs = outline.x.map((x) => x - frame.cx);
    const nose = xs[0];
    const tail = xs[xs.length - 1];
    const x0 = nose - UPSTREAM - 0.2;
    const n = Math.ceil((tail + DOWNSTREAM + 0.2 - x0) / STEP) + 1;
    const at = (arr, X) => interp(xs, arr, X);

    const rawA = new Float32Array(n);
    const rawB = new Float32Array(n);
    const gap = new Float32Array(n);
    const hSplit = (outline.z_top[0] + outline.z_bot[0]) / 2;   // where the oncoming air divides
    const botNose = outline.z_bot[0];
    const botTail = outline.z_bot[outline.z_bot.length - 1];
    for (let i = 0; i < n; i++) {
      const X = x0 + i * STEP;
      if (X >= nose && X <= tail) {
        rawA[i] = at(outline.half_width, X) + 0.04;
        rawB[i] = at(outline.z_top, X) + 0.04;
        gap[i] = at(outline.z_bot, X) - 0.02;
      } else if (X < nose) {
        gap[i] = lerp(hSplit, botNose - 0.02, smoothstep((X - (nose - 0.5)) / 0.5));
      } else {
        gap[i] = lerp(botTail, hSplit + 0.15, smoothstep((X - tail) / 1.2));
      }
    }
    // Look ahead, so a streamline starts to bend before it reaches the nose.
    const reach = Math.round(LOOKAHEAD / STEP);
    const lookA = new Float32Array(n);
    const lookB = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let a = 0, b = 0;
      for (let j = i; j < Math.min(n, i + reach); j++) {
        const w = Math.sqrt(1 - (j - i) / reach);
        a = Math.max(a, rawA[j] * w);
        b = Math.max(b, rawB[j] * w);
      }
      lookA[i] = a;
      lookB[i] = b;
    }
    const marks = Object.fromEntries(Object.entries(outline.landmarks).map(([k, v]) => [k, v - frame.cx]));
    shape = {
      x0, n, nose, tail, lookA, lookB, gap, hSplit, marks,
      tailA: at(outline.half_width, tail - 0.15) + 0.04,
      tailB: at(outline.z_top, tail - 0.05) + 0.04,
      underHalf: 0.85 * at(outline.half_width, nose + 0.4),
      axles: outline.axles.map((a) => a - frame.cx),
      wheelZ: outline.wheel_y + outline.tyre_width / 2 + 0.004,
      wheelR: outline.wheel_radius,
    };
    buildRims();
    rebuildField();
  }

  // ------------------------------------------------------------ the prediction

  function setPrediction(p) {
    prediction = p;
    rebuildField();
  }

  function rebuildField() {
    if (!shape || !prediction) return;
    const { n, x0, nose, tail, marks, axles } = shape;
    const cp = (name) => {
      const v = prediction.pressure?.[name];
      return typeof v === "number" && Number.isFinite(v) ? v : 0;
    };
    const top = new Float32Array(n);
    const bottom = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const X = x0 + i * STEP;
      let t, u;
      if (X < nose) {
        // Stagnation builds in front of the nose.
        const k = Math.exp(-(nose - X) / 0.35);
        t = cp("front_fascia") * k;
        u = cp("front_fascia") * 0.5 * k;
      } else if (X > tail) {
        const k = Math.exp(-(X - tail) / 1.0);
        t = cp("rear_fascia") * k;
        u = cp("rear_fascia") * k;
      } else {
        t = X < marks.hood_front + 0.1 ? cp("front_fascia")
          : X < marks.cowl ? cp("hood")
            : X < marks.roof_front ? cp("windscreen")
              : X < marks.roof_end ? cp("roof")
                : cp("rear_window");                  // rear glass and trunk lid
        const nearWheel = Math.max(...axles.map((a) => Math.exp(-(((X - a) / 0.3) ** 2))));
        u = lerp(cp("underbody"), cp("wheel_wells"), nearWheel);
      }
      top[i] = t;
      bottom[i] = u;
    }
    blur(top, 6);
    blur(bottom, 6);

    // Wake: grows with Cd against the baseline body's, so a slipperier design
    // visibly tidies it.
    const { cd, cl, sigma, threshold, baselineCd } = prediction;
    const strength = clamp(
      Number.isFinite(baselineCd) ? 0.55 + ((cd - baselineCd) / 0.03) * 0.45 : (cd - 0.2) / 0.12,
      0.15, 1,
    );
    field = {
      top, bottom,
      wake: strength,
      wakeLength: 1.0 + 2.5 * strength,
      closeLength: 0.5 + 1.2 * strength,
      swirl: 0.05 + 0.22 * strength,
      downwash: clamp(cl, -0.3, 0.3) * 0.35,
      rearCp: cp("rear_fascia"),
      jitter: clamp(threshold > 0 ? sigma / threshold : 0, 0, 1.5) * 0.015,
      trusted: prediction.trusted !== false,
    };
  }

  // ------------------------------------------------------------ simulation

  const xStart = () => shape.nose - UPSTREAM;
  const xEnd = () => shape.tail + DOWNSTREAM;
  const roadSpeed = () => (speedKmh / 3.6) * SLOW_MOTION;   // m/s on screen

  function spawn(i, X, Y0, Z0, phase = rng()) {
    px[i] = X;
    y0[i] = Y0;
    z0[i] = Z0;
    seed[i] = phase;
    under[i] = Y0 < shape.hSplit && Math.abs(Z0) < shape.underHalf ? 1 : 0;
    alive[i] = 1;
    place(i, 0);
    prev.set(cur.subarray(i * 3, i * 3 + 3), i * 3);
  }

  function emit(dt) {
    const U = roadSpeed();
    if (mode === "tunnel") {
      let k = 0;
      for (const lane of RAKE_LANES) {
        for (const h of RAKE_HEIGHTS) {
          nozzleCarry[k] += U * dt;
          while (nozzleCarry[k] >= RAKE_SPACING) {
            nozzleCarry[k] -= RAKE_SPACING;
            const s = nozzleSeq[k]++;
            const i = k * RAKE_SLOTS + (s % RAKE_SLOTS);
            seq[i] = s;
            spawn(i, xStart() + nozzleCarry[k], h, lane, (k * 0.618) % 1);
          }
          k++;
        }
      }
    } else {
      for (let i = 0; i < VOLUME_PARTICLES; i++) {
        if (!alive[i]) spawn(i, xStart() + rng() * 0.3, 0.04 + rng() * 2.0, (rng() * 2 - 1) * 1.6);
      }
    }
  }

  // Where particle i is drawn at time t, and its local Cp and speed factor.
  let lastCp = 0;
  let lastNear = 1;   // 1 at the body, falling to 0 in the free stream
  function place(i, t) {
    const X = px[i];
    const g = Math.min(shape.n - 1, Math.max(0, Math.round((X - shape.x0) / STEP)));
    let Y, Z, cp;
    lastNear = 1;
    if (under[i]) {
      Y = (y0[i] / shape.hSplit) * shape.gap[g];
      Z = z0[i];
      cp = field.bottom[g];
    } else {
      let a = shape.lookA[g], b = shape.lookB[g];
      if (X > shape.tail) {
        const k = Math.exp(-(X - shape.tail) / field.closeLength);
        a = Math.max(a, shape.tailA * k);
        b = Math.max(b, shape.tailB * k);
      }
      a = Math.max(a, 1e-3);
      b = Math.max(b, 1e-3);
      const u = z0[i] / a, v = y0[i] / b;
      const rho = Math.max(Math.pow(Math.abs(u) ** EXP + Math.abs(v) ** EXP, 1 / EXP), 1e-4);
      const k = Math.sqrt(1 + 1 / (rho * rho));             // r / rho, from r^2 = 1 + rho^2
      Y = y0[i] * k;
      Z = z0[i] * k;
      const near = Math.exp(-2 * (rho * k - 1));
      lastNear = near;
      cp = near * field.top[g] * (0.55 + 0.45 * Math.min(1, Math.abs(v) / rho));
    }
    let s = Math.sqrt(Math.max(0.06, 1 - cp));
    if (X > shape.tail) {
      const dx = X - shape.tail;
      const core = Math.exp(-((Z / (shape.tailA * 1.15)) ** 2 + ((Y - shape.tailB * 0.45) / (shape.tailB * 0.7)) ** 2));
      const w = field.wake * Math.exp(-dx / field.wakeLength) * core;
      s *= 1 - 0.6 * w;
      const phase = seed[i] * 6.283 + dx * 2.6 - t * 3.0;
      Y += w * field.swirl * Math.sin(phase);
      Z += w * field.swirl * Math.cos(phase * 1.3);
      Y -= field.downwash * Math.min(dx, 3) * Math.exp(-((Z / (shape.tailA * 1.6)) ** 2));
      cp = lerp(cp, field.rearCp, w);
    }
    if (field.jitter) {
      // Smooth along the line and in time: a tremble, not static.
      Y += field.jitter * Math.sin(t * 6 + X * 1.7 + seed[i] * 50);
      Z += field.jitter * Math.cos(t * 7 + X * 1.3 + seed[i] * 31);
    }
    const o = i * 3;
    cur[o] = X;
    cur[o + 1] = Math.max(Y, 0.01);
    cur[o + 2] = Z;
    lastCp = cp;
    return s;
  }

  // A streak stretches the motion between the last two drawn positions, which
  // the PREVIOUS step's dt produced; dividing by this step's dt instead made
  // streaks tens of metres long whenever two frames came close together.
  let lastDt = 1 / 60;
  let advancedBy = 1 / 60;
  function step(dt, t) {
    lastDt = Math.max(advancedBy, 1e-3);
    advancedBy = dt;
    const U = roadSpeed();
    const end = xEnd();
    for (let i = 0; i < CAPACITY; i++) {
      if (!alive[i]) continue;
      const o = i * 3;
      prev[o] = cur[o]; prev[o + 1] = cur[o + 1]; prev[o + 2] = cur[o + 2];
      const s = place(i, t);
      px[i] += U * s * dt;
      if (px[i] > end) { alive[i] = 0; continue; }
      colour(i, lastCp, lastNear);
    }
    emit(dt);
  }

  const c = new THREE.Color();
  function colour(i, cp, near) {
    if (!field.trusted) c.copy(COLOR.grey);
    else {
      const t = clamp(cp / CP_RANGE, -1, 1);
      c.copy(COLOR.mid).lerp(t < 0 ? COLOR.neg : COLOR.pos, Math.abs(t));
    }
    const X = px[i];
    const fade = Math.min(1, (X - xStart()) / 0.6, (xEnd() - X) / 1.5);
    const o = i * 4;
    rgba[o] = c.r; rgba[o + 1] = c.g; rgba[o + 2] = c.b;
    // On the road, air the car is not disturbing fades back, so the eye goes
    // to the flow around the body rather than a screen full of streaks.
    const focus = mode === "drive" ? 0.05 + 0.8 * near : 1;
    rgba[o + 3] = Math.max(0, fade) * (field.trusted ? 0.9 : 0.55) * focus;
  }

  function draw() {
    let p = 0, l = 0;
    const P = pointPos.array, PC = pointCol.array, L = linePos.array, LC = lineCol.array;
    // From (ax, ay, az) to particle i; the far end fades to `tailAlpha` of i's.
    const segment = (i, ax, ay, az, tailAlpha) => {
      const v = l * 6, o = i * 3, q = i * 4, w = l * 8;
      L[v] = ax; L[v + 1] = ay; L[v + 2] = az;
      L[v + 3] = cur[o]; L[v + 4] = cur[o + 1]; L[v + 5] = cur[o + 2];
      LC[w] = LC[w + 4] = rgba[q];
      LC[w + 1] = LC[w + 5] = rgba[q + 1];
      LC[w + 2] = LC[w + 6] = rgba[q + 2];
      LC[w + 3] = rgba[q + 3] * tailAlpha;
      LC[w + 7] = rgba[q + 3];
      l++;
    };
    const pointAlpha = mode === "tunnel" ? 0.45 : 1;
    for (let i = 0; i < CAPACITY; i++) {
      if (!alive[i]) continue;
      const o = i * 3, q = i * 4;
      P[p * 3] = cur[o]; P[p * 3 + 1] = cur[o + 1]; P[p * 3 + 2] = cur[o + 2];
      PC[p * 4] = rgba[q]; PC[p * 4 + 1] = rgba[q + 1]; PC[p * 4 + 2] = rgba[q + 2];
      PC[p * 4 + 3] = rgba[q + 3] * pointAlpha;
      p++;
      if (mode === "tunnel") {
        // Streakline: join each puff to the one its nozzle released before it.
        const nozzle = Math.floor(i / RAKE_SLOTS);
        const j = nozzle * RAKE_SLOTS + ((seq[i] - 1 + RAKE_SLOTS * 4) % RAKE_SLOTS);
        if (alive[j] && seq[j] === seq[i] - 1) {
          const jo = j * 3;
          const d = Math.hypot(cur[jo] - cur[o], cur[jo + 1] - cur[o + 1], cur[jo + 2] - cur[o + 2]);
          if (d < 0.5) segment(i, cur[jo], cur[jo + 1], cur[jo + 2], 1);   // continuous, no taper
        }
      } else {
        // Streak: a tail behind the particle, along its motion, a fixed time long.
        const k = STREAK_SECONDS / lastDt;
        segment(i, cur[o] - (cur[o] - prev[o]) * k, cur[o + 1] - (cur[o + 1] - prev[o + 1]) * k,
          cur[o + 2] - (cur[o + 2] - prev[o + 2]) * k, 0.15);
      }
    }
    pointGeo.setDrawRange(0, p);
    lineGeo.setDrawRange(0, l * 2);
    pointPos.needsUpdate = pointCol.needsUpdate = linePos.needsUpdate = lineCol.needsUpdate = true;
    points.material.size = mode === "tunnel" ? 0.035 : 0.03;
  }

  const stats = { frameMs: 0 };   // exponential mean of the work per frame
  function tick(dt) {
    if (!shape || !field) return;
    if (!warmed) { warmUp(); warmed = true; }
    const started = performance.now();
    clock += dt;
    step(dt, clock);
    draw();
    stats.frameMs = 0.9 * stats.frameMs + 0.1 * (performance.now() - started);
    if (mode === "drive") {
      const U = roadSpeed();
      road.texture.offset.x -= (U * dt) / road.tile;
      travel += U * dt;
      roadside.place(travel);
      wheelAngle += (U / shape.wheelR) * dt;
      for (const rim of rims) rim.rotation.z = rim.userData.side * wheelAngle;
      sweepCamera(dt);
    }
  }

  const SWEEP_PERIOD = 26;   // s for one front-to-rear-and-back pass
  function sweepCamera(dt) {
    if (!autoCamera) {
      if (!resumeAt || performance.now() < resumeAt) return;
      // Pick the sweep up from wherever the viewer left the camera.
      const at = viewer.camera.position;
      const s = clamp((Math.atan2(at.z, -at.x) - Math.PI / 2) / 0.95, -1, 1);
      sweepClock = (Math.asin(s) * SWEEP_PERIOD) / (2 * Math.PI);
      autoCamera = true;
      resumeAt = 0;
    }
    sweepClock += dt;
    const phase = (2 * Math.PI * sweepClock) / SWEEP_PERIOD;
    const angle = Math.PI / 2 + 0.95 * Math.sin(phase);   // about 36 to 144 degrees round the car
    const radius = 7.8;
    viewer.camera.position.set(-radius * Math.cos(angle), 2.05 + 0.3 * Math.sin(phase * 1.5), radius * Math.sin(angle));
    viewer.controls.target.set(0.35, 0.62, 0);
    viewer.controls.update();
  }

  // Fill the scene before the first frame, so the smoke is already flowing.
  // Free particles start spread along the whole path (released together they
  // would travel as one slab); the rake has to grow its streaklines.
  function warmUp() {
    if (!shape || !field) return;
    const dt = 1 / 15;
    let steps = 4;
    if (mode === "drive") {
      for (let i = 0; i < VOLUME_PARTICLES; i++) {
        spawn(i, xStart() + rng() * (xEnd() - xStart()), 0.04 + rng() * 2.0, (rng() * 2 - 1) * 1.6);
      }
    } else {
      const path = UPSTREAM + (shape.tail - shape.nose) + DOWNSTREAM;
      steps = Math.ceil(path / Math.max(roadSpeed(), 0.5) / dt) + 4;
    }
    for (let k = 0; k < steps; k++) { clock += dt; step(dt, clock); }
    draw();
  }

  function reset() {
    warmed = false;
    alive.fill(0);
    nozzleCarry.fill(0);
    nozzleSeq.fill(0);
    rng = mulberry32(7);
  }

  // ------------------------------------------------------------ scenery

  const rimMaterial = new THREE.MeshStandardMaterial({ map: rimTexture(), metalness: 0.6, roughness: 0.35 });
  let rimGeometry = null;
  function buildRims() {
    for (const rim of rims) scenery.remove(rim);
    rims.length = 0;
    rimGeometry?.dispose();
    rimGeometry = new THREE.CircleGeometry(shape.wheelR * 0.68, 40);
    for (const x of shape.axles) {
      for (const side of [1, -1]) {
        const rim = new THREE.Mesh(rimGeometry, rimMaterial);
        rim.position.set(x, shape.wheelR, side * shape.wheelZ);
        if (side < 0) rim.rotateY(Math.PI);
        rim.userData.side = side;
        rims.push(rim);
        scenery.add(rim);
      }
    }
  }

  // ------------------------------------------------------------ controls

  function run() {
    viewer.setAnimator(mode !== "off" && !paused ? tick : null);
  }

  function setMode(next) {
    if (next === mode) return;
    mode = next;
    group.visible = mode !== "off";
    scenery.visible = mode === "drive";
    floor.visible = mode === "tunnel";
    setAtmosphere(mode === "drive");
    reset();
    autoCamera = true;
    resumeAt = 0;
    sweepClock = -SWEEP_PERIOD / 4;   // open on the front three-quarter view
    viewer.setView(mode === "drive" ? "chase" : mode === "tunnel" ? "tunnel" : "orbit");
    run();
  }

  // Drive mode gets a sky and distance fog in the page's own colours, so the
  // road runs to a horizon in either theme.
  function setAtmosphere(on) {
    const scene = viewer.scene;
    if (!on) {
      scene.background = null;
      scene.fog = null;
      return;
    }
    const page = getComputedStyle(document.documentElement).getPropertyValue("--page").trim() || "#f9f9f7";
    const horizon = new THREE.Color(page);
    const dark = horizon.getHSL({}).l < 0.5;
    const zenith = horizon.clone().lerp(new THREE.Color(dark ? "#1d2a3a" : "#a9c4de"), 0.75);
    scene.background = skyTexture(zenith, horizon);
    scene.fog = new THREE.Fog(horizon, 18, 70);
    roadside.tint(dark);
  }

  function setPaused(value) {
    paused = value;
    run();
  }

  function setSpeed(kmh) {
    speedKmh = kmh;
  }

  return { setShape, setPrediction, setMode, setPaused, setSpeed, stats, get mode() { return mode; } };
}

// ---------------------------------------------------------------- helpers

// Posts along both verges and trees beyond them, instanced, wrapped round a
// 120 m loop as the road moves.
function makeRoadside() {
  const group = new THREE.Group();
  const LOOP = 120;
  const BEHIND = -45;
  const N_POSTS = 16;   // far verge only: near-side posts swept across the car
  const N_TREES = 46;
  const postGeo = new THREE.BoxGeometry(0.08, 0.9, 0.08);
  postGeo.translate(0, 0.45, 0);
  const posts = new THREE.InstancedMesh(postGeo, new THREE.MeshStandardMaterial({ color: "#d9d8d2", roughness: 0.6 }), N_POSTS);
  const trunkGeo = new THREE.CylinderGeometry(0.09, 0.12, 1.2, 6);
  trunkGeo.translate(0, 0.6, 0);
  const crownGeo = new THREE.ConeGeometry(1.0, 3.2, 8);
  crownGeo.translate(0, 2.6, 0);
  const crownMat = new THREE.MeshStandardMaterial({ color: "#56724f", roughness: 0.85, flatShading: true });
  const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: "#5b4a3a", roughness: 0.9 }), N_TREES);
  const crowns = new THREE.InstancedMesh(crownGeo, crownMat, N_TREES);
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: "#7d876f", roughness: 1 }));
  grass.rotation.x = -Math.PI / 2;
  grass.position.y = -0.01;
  group.add(grass, posts, trunks, crowns);

  const rand = mulberry32(11);
  const postBase = Array.from({ length: N_POSTS }, (_, k) => ({ x: k * (LOOP / N_POSTS), z: -5.8 }));
  const treeBase = Array.from({ length: N_TREES }, () => {
    const near = rand() < 0.5;
    // Near-side trees stay beyond the camera's path (7.8 m out), so none cuts across the shot.
    return { x: rand() * LOOP, z: near ? 11 + rand() * 16 : -9 - rand() * 18, s: 0.7 + rand() * 0.8 };
  });
  const m = new THREE.Matrix4();
  const wrap = (x, travel) => BEHIND + ((((x + travel) % LOOP) + LOOP) % LOOP);
  function place(travel) {
    postBase.forEach((p, k) => posts.setMatrixAt(k, m.makeTranslation(wrap(p.x, travel), 0, p.z)));
    treeBase.forEach((p, k) => {
      m.makeScale(p.s, p.s, p.s).setPosition(wrap(p.x, travel), 0, p.z);
      trunks.setMatrixAt(k, m);
      crowns.setMatrixAt(k, m);
    });
    posts.instanceMatrix.needsUpdate = true;
    trunks.instanceMatrix.needsUpdate = true;
    crowns.instanceMatrix.needsUpdate = true;
  }
  function tint(dark) {
    grass.material.color.set(dark ? "#2b3128" : "#7d876f");
    crownMat.color.set(dark ? "#33452f" : "#56724f");
  }
  place(0);
  return { group, place, tint };
}

function skyTexture(zenith, horizon) {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, "#" + zenith.getHexString());
  g.addColorStop(1, "#" + horizon.getHexString());
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeRoad() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#3a3a38";
  ctx.fillRect(0, 0, 512, 256);
  for (let k = 0; k < 900; k++) {   // a little asphalt grain
    ctx.fillStyle = `rgba(255,255,255,${0.02 + Math.random() * 0.04})`;
    ctx.fillRect(Math.random() * 512, Math.random() * 256, 2, 2);
  }
  ctx.fillStyle = "#e9e8e3";
  ctx.fillRect(0, 18, 512, 6);          // edge line
  ctx.fillRect(0, 232, 512, 6);         // edge line
  ctx.fillRect(0, 125, 256, 6);         // centre dash
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const length = 160;
  const tile = 6;                        // m of road per texture repeat
  texture.repeat.set(length / tile, 1);
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(length, 7.5),
    new THREE.MeshStandardMaterial({ map: texture, roughness: 0.95 }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(length / 2 - 45, 0.0005, -1.7);   // the car in the near lane
  return { mesh, texture, tile };
}

function rimTexture() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const c = size / 2;
  ctx.fillStyle = "#b9bcc2";
  ctx.beginPath(); ctx.arc(c, c, c - 1, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#2b2b2a";
  for (let k = 0; k < 5; k++) {          // five gaps between five spokes
    const a = (k / 5) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(c + Math.cos(a + 0.22) * 18, c + Math.sin(a + 0.22) * 18);
    ctx.arc(c, c, c - 9, a + 0.32, a + (Math.PI * 2) / 5 - 0.32);
    ctx.lineTo(c + Math.cos(a + (Math.PI * 2) / 5 - 0.22) * 18, c + Math.sin(a + (Math.PI * 2) / 5 - 0.22) * 18);
    ctx.fill();
  }
  ctx.fillStyle = "#77797d";
  ctx.beginPath(); ctx.arc(c, c, 12, 0, Math.PI * 2); ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function dotTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.4, "rgba(255,255,255,0.5)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

function blur(a, radius) {
  for (let pass = 0; pass < 2; pass++) {
    const src = Float32Array.from(a);
    for (let i = 0; i < a.length; i++) {
      let sum = 0, count = 0;
      for (let j = Math.max(0, i - radius); j <= Math.min(a.length - 1, i + radius); j++) { sum += src[j]; count++; }
      a[i] = sum / count;
    }
  }
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (t) => { const c = clamp(t, 0, 1); return c * c * (3 - 2 * c); };
