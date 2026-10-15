// The drag race: the original car against the design, same engine.  OWNER: Paramveer.
//
//   const race = createRace(viewer, hudElement)
//   race.start({ design, original, distance })   // each { cd, area, label, trusted? }; original also { payload, paint }
//   race.stop()
//   race.advance(seconds)   // step race time forward, for scripted demos and tests
//
// Both cars get the same power, mass, tyres and grip (physics.standingStart);
// only their predicted Cd and their frontal area differ. Everything on screen
// comes from that integration, and the assumptions are printed with the result.
// The design is the viewer's own car (viewer.car), so it is the shape on the
// sliders; the original is a second car built from the style's default dials.

import * as THREE from "three";
import { escapeHtml, fmt, infoButton, signed } from "./common.js";
import { aeroLoads, positionAt, standingStart } from "./physics.js";
import { PAINTS } from "./viewer.js";

export const RACE = { power: 150e3, mass: 1600, crr: 0.010, mu: 1.0 };
const LANE = 2.1;                 // m from the centre line to each car
const COUNTDOWN = 3.6;            // s of lights before GO
const COLORS = { design: "#22d3ee", original: "#c9ced6" };

export function createRace(viewer, hud) {
  const world = new THREE.Group();
  world.visible = false;
  viewer.scene.add(world);
  let built = 0;                  // distance the scenery was built for

  let original = null;            // the second car
  let plan = null;                // both integrations and labels
  let clock = 0;                  // s since the lights started, scaled
  let timeScale = 1;
  let finished = false;
  let camPos = new THREE.Vector3(), camLook = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const lookAt = new THREE.Vector3();

  // ------------------------------------------------------------ start / stop

  function start({ design, original: base, distance = 1000 }) {
    stop({ keepWorld: built === distance });
    if (built !== distance) buildWorld(distance);
    world.visible = true;
    setSky(viewer.scene);
    original = viewer.addCar(base.payload, { paint: base.paint || PAINTS["Liquid silver"] });
    const settings = { ...RACE, distance };
    const runs = {
      design: standingStart({ ...settings, cdA: design.cd * design.area }),
      original: standingStart({ ...settings, cdA: base.cd * base.area }),
    };
    plan = { design, original: base, distance, runs };
    clock = 0;
    finished = false;
    timeScale = distance > 2500 ? 3 : distance > 1500 ? 2 : 1;
    viewer.controls.enabled = false;
    camPos.set(15, 1.7, 1.2);
    camLook.set(-20, 0.7, 0);
    renderHud();
    place(0, 0);
    viewer.setAnimator(tick);
  }

  function stop({ keepWorld = false } = {}) {
    if (plan) viewer.setAnimator(null);
    if (original) { viewer.removeCar(original); original = null; }
    if (viewer.car) { viewer.car.group.position.set(0, 0, 0); viewer.car.spin(0); }
    if (!keepWorld) {
      world.visible = false;
      viewer.scene.background = null;
      viewer.scene.fog = null;
    }
    plan = null;
    hud.hidden = true;
    hud.replaceChildren();
    viewer.controls.enabled = true;
  }

  // ------------------------------------------------------------ the frame loop

  function tick(dt) {
    if (!plan) return;
    clock += dt * timeScale;
    const t = Math.max(0, clock - COUNTDOWN);
    const d = positionAt(plan.runs.design.samples, t);
    const o = positionAt(plan.runs.original.samples, t);
    place(d.x, o.x);
    camera(d.x, o.x, dt);
    updateHud(t, d, o);
    const done = t > Math.max(plan.runs.design.time, plan.runs.original.time) + 1.2;
    if (done && !finished) { finished = true; showFinish(); }
  }

  // Cars face -X in the viewer's frame, so the race runs toward -X. A car's
  // nose sits half its length ahead of its group's origin.
  function place(xDesign, xOriginal) {
    const car = viewer.car;
    if (car) {
      car.group.position.set(car.frame.length / 2 - xDesign, 0, LANE);
      car.spin(xDesign / car.payload.flow.wheel_radius);
    }
    if (original) {
      original.group.position.set(original.frame.length / 2 - xOriginal, 0, -LANE);
      original.spin(xOriginal / original.payload.flow.wheel_radius);
    }
  }

  // Low behind the grid during the lights, then a chase cam on the pair, then
  // a side shot at the line as the cars cross it.
  function camera(xd, xo, dt) {
    const lead = Math.max(xd, xo);
    const mid = -(xd + xo) / 2;
    const toGo = plan.distance - lead;
    let pos, look;
    if (clock < COUNTDOWN) {
      pos = tmp.set(15 - 0.8 * clock, 1.7 + 0.12 * clock, 1.2);
      look = lookAt.set(-20, 0.7, 0);
    } else if (toGo > 70) {
      pos = tmp.set(mid + 11.5, 3.2, 3.6);
      look = lookAt.set(mid - 9, 0.6, 0);
    } else {
      pos = tmp.set(-plan.distance + 6, 1.5, 13);
      look = lookAt.set(Math.max(mid, -plan.distance - 4), 0.8, 0);
    }
    const k = 1 - Math.exp(-dt * (clock < COUNTDOWN ? 6 : 3.2));
    camPos.lerp(pos, k);
    camLook.lerp(look, k);
    viewer.camera.position.copy(camPos);
    viewer.camera.lookAt(camLook);
  }

  // ------------------------------------------------------------ HUD

  function renderHud() {
    hud.hidden = false;
    hud.innerHTML = `
      <div class="race-top">
        <div class="race-card glass" style="--car:${COLORS.design}">
          <h4><i></i>${escapeHtml(plan.design.label)}</h4>
          <div class="big" data-r="d-speed">0</div><small>km/h · Cd ${fmt(plan.design.cd, 4)}</small>
        </div>
        <div class="race-card race-gap glass">
          <h4>Gap ${infoButton("race")}</h4>
          <div class="big" data-r="gap">0.0 m</div><small data-r="dist">0 / ${plan.distance} m</small>
        </div>
        <div class="race-card glass" style="--car:${COLORS.original}">
          <h4><i></i>${escapeHtml(plan.original.label)}</h4>
          <div class="big" data-r="o-speed">0</div><small>km/h · Cd ${fmt(plan.original.cd, 4)}</small>
        </div>
      </div>
      <div class="race-lights glass" data-r="lights"><i></i><i></i><i></i><i></i><i></i></div>
      <div class="race-progress"><span data-r="p-o" style="--car:${COLORS.original}"></span><span data-r="p-d" style="--car:${COLORS.design}"></span></div>
      <span class="car-tag" data-r="tag-d" style="--car:${COLORS.design}"></span>
      <span class="car-tag" data-r="tag-o" style="--car:${COLORS.original}"></span>
      ${plan.design.trusted === false ? `<div class="hud-flag" style="position:absolute;left:14px;top:120px">The gate rejected this design's prediction: race it, but don't trust it.</div>` : ""}`;
  }

  const r = (name) => hud.querySelector(`[data-r="${name}"]`);
  const projected = new THREE.Vector3();
  function tag(el, group, frame, text, label) {
    projected.set(group.position.x, frame.height + 0.35, group.position.z).project(viewer.camera);
    const box = viewer.renderer.domElement.getBoundingClientRect();
    const hostBox = hud.getBoundingClientRect();
    const visible = projected.z < 1 && Math.abs(projected.x) < 1.1 && Math.abs(projected.y) < 1.1;
    el.style.display = visible ? "" : "none";
    el.style.left = `${box.left - hostBox.left + ((projected.x + 1) / 2) * box.width}px`;
    el.style.top = `${box.top - hostBox.top + ((1 - projected.y) / 2) * box.height}px`;
    el.innerHTML = `${text}<small>${label}</small>`;
  }

  function updateHud(t, d, o) {
    const lights = r("lights");
    if (lights) {
      const lit = Math.min(5, Math.floor(clock / 0.6));
      [...lights.children].forEach((light, k) => {
        light.dataset.on = clock >= COUNTDOWN ? "go" : k < lit ? "red" : "";
      });
      lights.style.opacity = clock > COUNTDOWN + 0.8 ? "0" : "1";
      lights.style.transition = "opacity 400ms";
    }
    r("d-speed").textContent = fmt(d.v * 3.6, 0);
    r("o-speed").textContent = fmt(o.v * 3.6, 0);
    const gap = d.x - o.x;
    r("gap").textContent = `${signed(gap, 1)} m`;
    r("dist").textContent = `${fmt(Math.min(plan.distance, Math.max(d.x, o.x)), 0)} / ${plan.distance} m`;
    r("p-d").style.left = `${Math.min(100, (d.x / plan.distance) * 100)}%`;
    r("p-o").style.left = `${Math.min(100, (o.x / plan.distance) * 100)}%`;
    if (viewer.car) tag(r("tag-d"), viewer.car.group, viewer.car.frame, `${fmt(d.v * 3.6, 0)} km/h`, "design");
    if (original) tag(r("tag-o"), original.group, original.frame, `${fmt(o.v * 3.6, 0)} km/h`, "original");
  }

  function showFinish() {
    const { design: d, original: o } = plan.runs;
    const gap = plan.distance - positionAt(o.samples, d.time).x;     // where the original is as the design crosses
    const ahead = gap >= 0;
    const margin = Math.abs(ahead ? gap : plan.distance - positionAt(d.samples, o.time).x);
    const quicker = o.time - d.time;
    const at130 = (car) => aeroLoads({ cd: car.cd, area: car.area, speedKmh: 130 }).energyPer100km;
    const saved = at130(plan.original) - at130(plan.design);
    const headline = margin < 0.05 ? "Dead heat"
      : ahead ? `The design wins by ${fmt(margin, 1)} m` : `The original wins by ${fmt(margin, 1)} m`;
    const card = document.createElement("div");
    card.className = "race-finish glass";
    card.innerHTML = `
      <h3>${fmt(plan.distance / 1000, plan.distance % 1000 ? 1 : 0)} km from a standing start · same ${fmt(RACE.power / 1000, 0)} kW engine</h3>
      <div class="headline">${headline}</div>
      <div class="stats">
        <div><b>${signed(quicker, 2)} s</b><small>design ${quicker >= 0 ? "quicker" : "slower"}</small></div>
        <div><b>${fmt(d.trapSpeed * 3.6, 0)} vs ${fmt(o.trapSpeed * 3.6, 0)}</b><small>trap speed, km/h</small></div>
        <div><b>${signed(-saved, 2)} kWh</b><small>per 100 km at 130 km/h, drag only</small></div>
      </div>
      <div class="actions">
        <button type="button" class="button primary" data-a="replay">Replay</button>
        ${[1000, 2000, 5000].map((m) => `<button type="button" class="button" data-a="dist" data-m="${m}" ${m === plan.distance ? 'aria-pressed="true"' : ""}>${m / 1000} km</button>`).join("")}
        <button type="button" class="button" data-a="clean">Clean view</button>
      </div>
      <div class="race-foot">Same power, ${fmt(RACE.mass, 0)} kg, rolling resistance ${RACE.crr}, grip μ ${RACE.mu}; only drag differs,
        from each car's predicted Cd × frontal area (${fmt(plan.design.cd * plan.design.area, 3)} vs ${fmt(plan.original.cd * plan.original.area, 3)} m²).
        An illustration of what the prediction means, not a vehicle simulation.</div>`;
    card.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-a]");
      if (!button) return;
      if (button.dataset.a === "replay") start({ design: plan.design, original: plan.original, distance: plan.distance });
      if (button.dataset.a === "dist") start({ design: plan.design, original: plan.original, distance: Number(button.dataset.m) });
      if (button.dataset.a === "clean") {
        const stage = hud.closest(".stage");
        stage.dataset.clean = "true";
        card.remove();
        const restore = () => { delete stage.dataset.clean; stage.removeEventListener("click", restore); };
        setTimeout(() => stage.addEventListener("click", restore), 0);
      }
    });
    hud.append(card);
  }

  // ------------------------------------------------------------ the track

  function buildWorld(distance) {
    world.clear();
    built = distance;
    const length = distance + 260;
    const startX = 120;                     // road begins behind the grid

    const road = new THREE.Mesh(
      new THREE.PlaneGeometry(length, 11),
      new THREE.MeshStandardMaterial({ map: trackTexture(length), roughness: 0.7, metalness: 0.05, envMapIntensity: 0.35 }),
    );
    road.rotation.x = -Math.PI / 2;
    road.position.set(startX - length / 2, 0.001, 0);
    road.receiveShadow = true;
    const grass = new THREE.Mesh(new THREE.PlaneGeometry(length + 600, 600), new THREE.MeshStandardMaterial({ color: "#0b100c", roughness: 1 }));
    grass.rotation.x = -Math.PI / 2;
    grass.position.set(startX - length / 2, -0.01, 0);
    world.add(grass, road);

    world.add(line(0, "#9ea4ab"));                     // start
    world.add(line(-distance, null));                  // chequered finish
    world.add(gantry(0, "START", "#e9edf2", 0.4));
    world.add(gantry(-distance, "FINISH", "#22d3ee", 1.6));

    // Lamp posts on both verges every 40 m: they stream past and glow.
    const count = Math.ceil(length / 40) * 2;
    const poleGeo = new THREE.CylinderGeometry(0.06, 0.08, 6, 6);
    poleGeo.translate(0, 3, 0);
    const lampGeo = new THREE.BoxGeometry(0.9, 0.12, 0.3);
    lampGeo.translate(0, 6, 0);
    const poles = new THREE.InstancedMesh(poleGeo, new THREE.MeshStandardMaterial({ color: "#3a4048", roughness: 0.6, metalness: 0.5 }), count);
    const lamps = new THREE.InstancedMesh(lampGeo, new THREE.MeshStandardMaterial({ color: "#fff4d6", emissive: "#ffd9a0", emissiveIntensity: 3 }), count);
    const m = new THREE.Matrix4();
    let k = 0;
    for (let x = startX; x > startX - length && k < count; x -= 40) {
      for (const z of [-7.2, 7.2]) {
        m.makeTranslation(x, 0, z);
        poles.setMatrixAt(k, m);
        lamps.setMatrixAt(k, m);
        k++;
      }
    }
    poles.count = lamps.count = k;
    world.add(poles, lamps);

    // Distance boards every 100 m on the far verge.
    for (let x = 100; x < distance; x += 100) {
      const board = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.8), new THREE.MeshBasicMaterial({ map: labelTexture(`${x} m`, "#0b1118", "#e9edf2"), toneMapped: false }));
      board.position.set(-x, 1.4, -6.4);
      world.add(board);
    }
  }

  function advance(seconds) {
    for (let t = 0; t < seconds && plan; t += 1 / 60) tick(1 / 60 / timeScale);
    viewer.render();
  }

  return { start, stop, advance, get running() { return !!plan; } };
}

function setSky(scene) {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, "#02040a");
  g.addColorStop(0.6, "#101b2c");
  g.addColorStop(1, "#33405a");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  scene.background = texture;
  scene.fog = new THREE.Fog("#0e1624", 40, 260);
}

function trackTexture(length) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#16181b";
  ctx.fillRect(0, 0, 512, 256);
  for (let k = 0; k < 1600; k++) {
    ctx.fillStyle = `rgba(255,255,255,${0.012 + Math.random() * 0.03})`;
    ctx.fillRect(Math.random() * 512, Math.random() * 256, 2, 2);
  }
  ctx.fillStyle = "#d9dde2";
  ctx.fillRect(0, 10, 512, 5);              // edge lines
  ctx.fillRect(0, 241, 512, 5);
  ctx.fillRect(0, 126, 300, 4);             // centre dash
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.repeat.set(length / 9, 1);
  return texture;
}

function line(x, colour) {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 16; j++) {
      ctx.fillStyle = colour || ((i + j) % 2 ? "#0b0b0b" : "#f2f4f6");
      ctx.fillRect(i * 8, j * 8, 8, 8);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.NearestFilter;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 11), new THREE.MeshStandardMaterial({ map: texture, roughness: 0.85, envMapIntensity: 0.2 }));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(x, 0.004, 0);
  return mesh;
}

function gantry(x, text, colour, glow) {
  const group = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: "#2b3138", roughness: 0.45, metalness: 0.7 });
  for (const z of [-6.2, 6.2]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.35, 6.2, 0.35), metal);
    post.position.set(x, 3.1, z);
    group.add(post);
  }
  const beam = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.2, 12.8), metal);
  beam.position.set(x, 6.2, 0);
  const sign = new THREE.Mesh(
    new THREE.PlaneGeometry(9, 1.0),
    new THREE.MeshStandardMaterial({ map: labelTexture(text, "#05080c", colour), emissive: colour, emissiveIntensity: glow, emissiveMap: labelTexture(text, "#000000", "#ffffff") }),
  );
  sign.position.set(x + 0.27, 6.2, 0);
  sign.rotation.y = Math.PI / 2;
  group.add(beam, sign);
  return group;
}

function labelTexture(text, background, colour) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, 512, 64);
  ctx.fillStyle = colour;
  ctx.font = "700 40px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 34);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}
