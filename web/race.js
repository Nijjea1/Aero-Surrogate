// The race: the original car against the design, same engine.  OWNER: Paramveer.
//
//   const race = createRace(viewer, { hud, panel })
//   race.configure({ design, original })   // each { cd, area, label, trusted? }; original also { payload }
//   race.start(); race.stop()
//   race.advance(seconds)                  // step race time forward, for scripted demos and tests
//
// Three modes, all from physics.simulateRace: Distance (standing start over a
// set distance, photo finish at the line), Top speed (flat out until both stop
// gaining) and Coast-down (both released at a speed with the engine off). The
// two cars share power, mass, tyres and grip; only their predicted Cd and
// frontal area differ, and the assumptions are printed with the result.
//
// The race keeps its own clock from the true frame time, so it plays in real
// time however slow the frames are. Long races are compressed (Auto): real
// time through the lights and the launch, then fast, then real time again for
// the finish. The track is a tile that recentres on the cars every 200 m, so
// a 100 km race costs no more to draw than a 1 km one.

import * as THREE from "three";
import { escapeHtml, fmt, infoButton, signed } from "./common.js";
import { aeroLoads, positionAt, simulateRace, timeAt } from "./physics.js";
import { PAINTS } from "./viewer.js";

export const RACE = { power: 150e3, mass: 1600, crr: 0.010, mu: 1.0 };
export const LIMITS = {
  distance: [0.4, 100],    // km
  power: [50, 500],        // kW
  mass: [900, 2500],       // kg
  v0: [60, 200],           // km/h
};
const LANE = 2.1;                     // m from the centre line to each car
const LIGHTS = 2;                     // s: red, then yellow, then green
const TILE = 200;                     // m between recentrings of the track
const DESIGN_PAINT = "#e8551c";
const ORIGINAL_PAINT = PAINTS["Liquid silver"];

export function createRace(viewer, { hud, panel }) {
  const world = new THREE.Group();
  world.visible = false;
  viewer.scene.add(world);
  const scenery = buildScenery();
  world.add(scenery.group);

  let cars = null;                    // { design, original } from configure
  let other = null;                   // the second car's 3D model
  let savedPaint = null;
  let plan = null;                    // both integrations, the mode and its inputs
  let phase = "off";                  // off | idle | lights | run | photo | done
  let wall = 0, sim = 0, goWall = 0, photoWall = 0, rateNow = 1;
  let mode = "distance", rate = "auto";
  let origin = 0;                     // race metres at scene x = 0
  let snap = false;                   // a jump (skip): the camera cuts instead of gliding
  const camLook = new THREE.Vector3();
  const tmp = new THREE.Vector3(), look = new THREE.Vector3();

  // ------------------------------------------------------------ controls

  const inputs = panel.inputs;
  panel.mode.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    mode = button.dataset.mode;
    for (const b of panel.mode.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
    for (const el of panel.root.querySelectorAll("[data-for]")) el.hidden = !el.dataset.for.split(" ").includes(mode);
    if (cars) prepare();
  });
  panel.rate.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-rate]");
    if (!button) return;
    if (button.dataset.rate === "skip") { skip(); return; }
    rate = button.dataset.rate;
    for (const b of panel.rate.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
  });
  panel.go.addEventListener("click", () => start());
  for (const el of Object.values(inputs)) el.addEventListener("change", () => { if (cars) prepare(); });

  function readInputs() {
    const read = (el, [lo, hi], fallback) => {
      const n = parseFloat(String(el.value).replace(",", "."));
      const v = Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
      el.value = String(v);
      return v;
    };
    return {
      distance: read(inputs.distance, LIMITS.distance, 5) * 1000,
      power: read(inputs.power, LIMITS.power, RACE.power / 1000) * 1000,
      mass: read(inputs.mass, LIMITS.mass, RACE.mass),
      v0: read(inputs.v0, LIMITS.v0, 130) / 3.6,
    };
  }

  // ------------------------------------------------------------ set up

  function configure(next) {
    cars = next;
    if (phase === "off") {
      world.visible = true;
      setSky(viewer.scene);
      savedPaint = savedPaint ?? viewer.paint;
      viewer.setPaint(DESIGN_PAINT);
      viewer.controls.enabled = false;
      viewer.setAnimator(tick);
    }
    if (other) viewer.removeCar(other);
    other = viewer.addCar(cars.original.payload, { paint: ORIGINAL_PAINT });
    prepare();
  }

  function prepare() {
    const settings = { ...RACE, ...readInputs() };
    const design = simulateRace(mode, { ...settings, cdA: cars.design.cd * cars.design.area });
    const original = simulateRace(mode, { ...settings, cdA: cars.original.cd * cars.original.area });
    const last = Math.max(design.end.t, original.end.t);
    const end = mode === "top" ? last + 2 : last + 1;
    const key = mode === "distance" ? Math.min(design.end.t, original.end.t) : end;
    const total = mode === "distance" ? settings.distance
      : Math.max(positionAt(design.samples, end).x, positionAt(original.samples, end).x);
    const vTop = Math.max(...[design, original].flatMap((r) => r.samples.map((s) => s.v)));
    plan = {
      mode, settings, design, original, end, key, total,
      fast: Math.max(1, (end - 6) / 16),     // the Auto rate: about 25 s of playback
      vScale: Math.max(50, Math.ceil((vTop * 3.6) / 50) * 50),
      identical: Math.abs(cars.design.cd * cars.design.area - cars.original.cd * cars.original.area) < 1e-9,
    };
    phase = "idle";
    sim = 0; wall = 0;
    origin = 0;
    scenery.relabel(0, plan);
    setLights(-1);
    renderHud();
    idle(plan.identical
      ? "Identical shapes tie.<br>Move a dial to give your design something to race with."
      : `Press <b>Start race</b> to race your design against the original.`);
    resetCamera();
    place();
    drawPanel();
    viewer.render();
  }

  function start() {
    if (!cars) return;
    prepare();
    if (plan.identical) return;
    idle(null);
    phase = "lights";
    hud.querySelector("[data-r=lights]").hidden = false;
    hud.querySelector("[data-r=lights]").style.opacity = "1";
  }

  function skip() {
    if (phase === "lights" || phase === "run" || phase === "photo") {
      sim = plan.end;
      phase = "run";
      goWall = wall;
      setLights(2);
    }
  }

  function stop() {
    viewer.setAnimator(null);
    if (other) { viewer.removeCar(other); other = null; }
    if (viewer.car) { viewer.car.group.position.set(0, 0, 0); viewer.car.spin(0); }
    if (savedPaint) viewer.setPaint(savedPaint);
    savedPaint = null;
    world.visible = false;
    viewer.scene.background = null;
    viewer.scene.fog = null;
    phase = "off";
    plan = null;
    cars = null;
    hud.hidden = true;
    hud.replaceChildren();
    viewer.controls.enabled = true;
    viewer.render();
  }

  // ------------------------------------------------------------ the clock

  function autoRate() {
    const { fast, key, end } = plan;
    const launch = THREE.MathUtils.smoothstep(sim, 1.2, 4);
    if (plan.mode === "distance" && sim < key) {
      if (key - sim < 0.6) return 0.3;                              // slow motion at the line
      return 1 + (fast - 1) * launch * (1 - THREE.MathUtils.smoothstep(sim, key - fast * 1.4, key - 1.2));
    }
    if (plan.mode === "distance") return fast;                     // the loser's run-in after the photo
    return 1 + (fast - 1) * launch * (1 - THREE.MathUtils.smoothstep(sim, end - fast * 1.5, end - 0.5));
  }

  function tick(dt) {
    if (!plan) return;
    if (phase === "lights" || phase === "run" || phase === "photo") wall += dt;
    if (phase === "lights") {
      setLights(wall < 1 ? 0 : wall < LIGHTS ? 1 : 2);
      if (wall >= LIGHTS) { phase = "run"; goWall = wall; }
    } else if (phase === "run" || phase === "photo") {
      const lights = hud.querySelector("[data-r=lights]");
      if (lights && wall - goWall > 0.8) lights.style.opacity = "0";
      if (lights && wall - goWall > 1.2) lights.hidden = true;
    }
    if (phase === "photo" && wall - photoWall > 1.8) { phase = "run"; hud.querySelector("[data-r=photo]").hidden = true; }
    if (phase === "run") {
      rateNow = rate === "auto" ? autoRate() : Number(rate);
      const before = sim;
      sim = Math.min(plan.end, sim + dt * rateNow);
      if (plan.mode === "distance" && rate === "auto" && before < plan.key && sim >= plan.key) {
        sim = plan.key;
        phase = "photo";
        photoWall = wall;
        hud.querySelector("[data-r=photo]").hidden = false;
      }
      if (sim >= plan.end) { phase = "done"; showFinish(); }
    }
    place();
    camera(dt);
    updateHud();
    drawPanel();
  }

  function advance(seconds) {
    for (let t = 0; t < seconds && plan; t += 1 / 60) tick(1 / 60);
    viewer.render();
  }

  // ------------------------------------------------------------ the scene

  const at = (t) => ({ d: positionAt(plan.design.samples, t), o: positionAt(plan.original.samples, t) });
  const sceneX = (xm) => -(xm - origin);

  // Cars face -X in the viewer's frame, so the race runs toward -X. A car's
  // nose sits half its length ahead of its group's origin.
  function place() {
    const { d, o } = at(sim);
    const focus = phase === "photo" ? plan.settings.distance : Math.max(d.x, o.x);
    const next = Math.round(focus / TILE) * TILE;
    if (next !== origin) {
      const shift = next - origin;
      origin = next;
      if (Math.abs(shift) > TILE) snap = true;
      scenery.relabel(origin, plan);
    }
    const car = viewer.car;
    if (car) {
      car.group.position.set(car.frame.length / 2 + sceneX(d.x), 0, LANE);
      car.spin(d.x / car.payload.flow.wheel_radius);
    }
    if (other) {
      other.group.position.set(other.frame.length / 2 + sceneX(o.x), 0, -LANE);
      other.spin(o.x / other.payload.flow.wheel_radius);
    }
    scenery.placeLines(sceneX(0), plan.mode === "distance" ? sceneX(plan.settings.distance) : null);
  }

  function resetCamera() {
    snap = true;
    camera(0);
  }

  // Low behind the grid through the lights, a chase camera on the pair (on
  // the leader once they are far apart), and a side shot at the finish. The
  // camera rides with its anchor and only its offset from it eases, so it
  // never falls behind fast cars.
  const camOff = new THREE.Vector3(), lookOff = new THREE.Vector3();
  function camera(dt) {
    const { d, o } = at(sim);
    const lead = Math.max(d.x, o.x);
    const focus = Math.abs(d.x - o.x) < 18 ? (d.x + o.x) / 2 : lead - 3;
    const line = plan.settings.distance;
    let anchor, speed = 3.2;
    if (phase === "idle" || phase === "lights") {
      anchor = 0;
      tmp.set(15 - 0.6 * wall, 1.6 + 0.1 * wall, 1.4);
      look.set(-25, 0.7, 0);
      speed = 6;
    } else if (plan.mode === "distance" && (phase === "photo" || (line - lead < 60 && phase !== "done"))) {
      anchor = line;
      tmp.set(-1.5, 1.4, 6.3);
      look.set(1.5, 0.8, 0);
      speed = phase === "photo" ? 8 : 4;
    } else if (phase === "done") {
      anchor = focus;
      tmp.set(-4, 1.5, 6.3);
      look.set(2, 0.8, 0);
    } else {
      anchor = focus;
      tmp.set(11, 3.1, 4.2);
      look.set(-9, 0.6, 0);
    }
    const k = snap ? 1 : 1 - Math.exp(-dt * speed);
    snap = false;
    camOff.lerp(tmp, k);
    lookOff.lerp(look, k);
    const x = sceneX(anchor);
    viewer.camera.position.set(x + camOff.x, camOff.y, camOff.z);
    camLook.set(x + lookOff.x, lookOff.y, lookOff.z);
    viewer.camera.lookAt(camLook);
  }

  // ------------------------------------------------------------ the broadcast graphics

  const r = (name) => hud.querySelector(`[data-r="${name}"]`);
  const title = () => ({ distance: "Distance", top: "Top speed", coast: "Coast-down" })[plan.mode];

  function renderHud() {
    hud.hidden = false;
    const s = plan.settings;
    const label = plan.mode === "distance" ? `Distance race · ${km(s.distance)} · same ${fmt(s.power / 1000, 0)} kW`
      : plan.mode === "top" ? `Top speed · same ${fmt(s.power / 1000, 0)} kW`
        : `Coast-down from ${fmt(s.v0 * 3.6, 0)} km/h · engine off`;
    hud.innerHTML = `
      <div class="tower" data-r="tower"></div>
      <div class="badge"><span data-r="clock">T+ 0:00.0</span><b data-r="rate">1×</b></div>
      <div class="race-lights" data-r="lights" hidden><i class="r"></i><i class="y"></i><i class="g"></i></div>
      <div class="photo" data-r="photo" hidden>PHOTO FINISH</div>
      <div class="third">${escapeHtml(label)} · ${fmt(s.mass, 0)} kg ${infoButton("race")}</div>
      ${cars.design.trusted === false ? '<div class="untrusted">The gate rejected this prediction: race it, but don\'t trust it</div>' : ""}
      <div class="idle" data-r="idle" hidden></div>`;
    updateHud();
  }

  function idle(html) {
    const el = r("idle");
    if (!el) return;
    el.hidden = !html;
    if (html) el.innerHTML = `<div>${html}</div>`;
  }

  function setLights(state) {
    const lamps = r("lights")?.children;
    if (lamps) [...lamps].forEach((lamp, k) => { lamp.dataset.on = String(k === state); });
    scenery.setLamps(state);
  }

  function updateHud() {
    if (!plan || !r("tower")) return;
    const { d, o } = at(sim);
    const designLeads = d.x >= o.x;
    const gap = Math.abs(d.x - o.x);
    const leader = designLeads ? plan.design : plan.original;
    const interval = sim - timeAt(leader.samples, designLeads ? o.x : d.x);
    const level = gap < 0.005;
    const behind = plan.mode === "coast" || !(interval > 0) ? `+${metres(gap)}` : `+${fmt(interval, interval < 10 ? 2 : 1)} s`;
    const rows = [
      { name: cars.design.label, colour: "var(--accent)", x: d.x },
      { name: cars.original.label, colour: "var(--silver)", x: o.x },
    ].sort((a, b) => b.x - a.x);
    const head = plan.mode === "distance" ? `${km(Math.max(0, plan.settings.distance - Math.max(d.x, o.x)), 2)} to go`
      : plan.mode === "top" ? "flat out" : "engine off";
    r("tower").innerHTML = `<div class="hd"><span>${title()}</span><span>${head}</span></div>` +
      rows.map((row, i) => `<div class="row"><span class="p">${i + 1}</span><i style="background:${row.colour}"></i>` +
        `<span>${escapeHtml(row.name.toUpperCase())}</span><span class="iv">${i === 0 ? (level ? "level" : "leader") : level ? "level" : behind}</span></div>`).join("") +
      `<div class="ft"><span>${fmt(d.v * 3.6, 0)} km/h</span><span>${fmt(o.v * 3.6, 0)} km/h</span></div>`;
    const minutes = Math.floor(sim / 60), seconds = sim - minutes * 60;
    r("clock").textContent = `T+ ${minutes}:${seconds < 9.95 ? "0" : ""}${fmt(seconds, 1)}`;
    r("rate").textContent = phase === "photo" ? "FREEZE" : phase !== "run" ? "1×" : rateNow < 0.9 ? "SLOW-MO" : `×${fmt(rateNow, 0)}`;
  }

  function showFinish() {
    const { design: d, original: o, settings: s } = plan;
    const lights = r("lights");
    if (lights) lights.hidden = true;
    let headline, stats;
    if (plan.mode === "distance") {
      const wins = d.end.t <= o.end.t;
      const margin = wins ? s.distance - positionAt(o.samples, d.end.t).x : s.distance - positionAt(d.samples, o.end.t).x;
      const dt = Math.abs(o.end.t - d.end.t);
      headline = margin < 0.005 ? "Dead heat."
        : `${wins ? "Your design" : "The original"} finishes ${metres(margin)} ahead over ${km(s.distance)}, ${fmt(dt, dt < 1 ? 3 : 1)} s quicker on the same ${fmt(s.power / 1000, 0)} kW.`;
      stats = [[`${fmt(d.end.t, 1)} vs ${fmt(o.end.t, 1)} s`, "finish time"],
        [`${fmt(d.end.v * 3.6, 0)} vs ${fmt(o.end.v * 3.6, 0)}`, "trap speed, km/h"],
        [`${wins ? "+" : "−"}${metres(margin)}`, "your design's margin"]];
    } else if (plan.mode === "top") {
      const dv = (d.vmax - o.vmax) * 3.6;
      headline = `Your design tops out at ${fmt(d.vmax * 3.6, 1)} km/h: ${signed(dv, 1)} km/h against the original on the same ${fmt(s.power / 1000, 0)} kW.`;
      stats = [[`${fmt(d.vmax * 3.6, 1)} vs ${fmt(o.vmax * 3.6, 1)}`, "top speed, km/h"],
        [`${fmt(d.end.t, 0)} vs ${fmt(o.end.t, 0)} s`, "to within 0.3 km/h of it"],
        [`${signed(dv, 1)} km/h`, "your design's advantage"]];
    } else {
      const dx = d.end.x - o.end.x;
      headline = `From ${fmt(s.v0 * 3.6, 0)} km/h with the engine off, your design rolls ${metres(Math.abs(dx))} ${dx >= 0 ? "further" : "less far"} before it stops.`;
      stats = [[`${fmt(d.end.x, 0)} vs ${fmt(o.end.x, 0)} m`, "rolling distance"],
        [`${fmt(d.end.t, 0)} vs ${fmt(o.end.t, 0)} s`, "time to stop"],
        [`${signed(dx, 1)} m`, "your design's advantage"]];
    }
    const at130 = (car) => aeroLoads({ cd: car.cd, area: car.area, speedKmh: 130 }).energyPer100km;
    const saved = at130(cars.original) - at130(cars.design);
    const card = document.createElement("div");
    card.className = "race-finish";
    card.innerHTML = `
      <div class="headline">${headline}</div>
      <div class="stats">${stats.map(([b, small]) => `<div><b>${b}</b><small>${small}</small></div>`).join("")}</div>
      <div class="race-foot">Same ${fmt(s.mass, 0)} kg, rolling resistance ${RACE.crr} and grip μ ${RACE.mu}${plan.mode === "coast" ? "" : `, ${fmt(s.power / 1000, 0)} kW`}.
        Only drag differs, from each car's predicted Cd × frontal area (${fmt(cars.design.cd * cars.design.area, 3)} vs ${fmt(cars.original.cd * cars.original.area, 3)} m²);
        at 130 km/h that is ${signed(-saved, 2)} kWh per 100 km. An illustration of what the prediction means, not a vehicle simulation.</div>
      <div class="actions">
        <button type="button" class="button primary" data-a="again">Race again</button>
        <button type="button" class="button" data-a="clean">Clean view</button>
        <button type="button" class="button" data-a="close">Close</button>
      </div>`;
    card.addEventListener("click", (event) => {
      const action = event.target.closest("button[data-a]")?.dataset.a;
      if (action === "again") start();
      if (action === "close") card.remove();
      if (action === "clean") {
        const stage = hud.closest(".stage");
        stage.dataset.clean = "true";
        hud.style.visibility = "hidden";
        card.remove();
        const restore = () => { delete stage.dataset.clean; hud.style.visibility = ""; stage.removeEventListener("click", restore); };
        setTimeout(() => stage.addEventListener("click", restore), 0);
      }
    });
    hud.append(card);
  }

  // ------------------------------------------------------------ timing under the stage

  function drawPanel() {
    if (!plan) return;
    const { d, o } = at(sim);
    const total = plan.total || 1;
    panel.mkD.style.left = `${Math.min(100, (d.x / total) * 100)}%`;
    panel.mkO.style.left = `${Math.min(100, (o.x / total) * 100)}%`;
    const long = total >= 2000;
    const ticks = Array.from({ length: 6 }, (_, i) => {
      const x = (total * i) / 5;
      return `<span>${long ? fmt(x / 1000, x / 1000 < 10 ? 1 : 0) : fmt(x, 0)}${i === 5 ? (long ? " km" : " m") : ""}</span>`;
    }).join("");
    if (panel.ticks.dataset.key !== ticks) { panel.ticks.innerHTML = ticks; panel.ticks.dataset.key = ticks; }
    drawTrace(panel.trace);
  }

  function drawTrace(canvas) {
    const box = canvas.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(box.width * ratio), h = Math.round(box.height * ratio);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const css = getComputedStyle(canvas);
    const token = (name) => css.getPropertyValue(name).trim();
    const W = box.width, H = box.height, left = 46, right = 120, top = 14, bottom = 22;
    const X = (t) => left + (t / plan.end) * (W - left - right);
    const Y = (v) => H - bottom - ((v * 3.6) / plan.vScale) * (H - top - bottom);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10.5px 'IBM Plex Mono', monospace";
    ctx.lineWidth = 1;
    ctx.strokeStyle = token("--grid");
    ctx.fillStyle = token("--text-muted");
    for (let v = 0; v <= plan.vScale; v += 50) {
      const y = Y(v / 3.6);
      ctx.beginPath(); ctx.moveTo(left, y + 0.5); ctx.lineTo(W - right, y + 0.5); ctx.stroke();
      ctx.fillText(String(v), 10, y + 4);
    }
    ctx.fillText("km/h", 10, top - 3);
    const step = plan.end > 600 ? 120 : plan.end > 150 ? 30 : plan.end > 40 ? 10 : 5;
    for (let t = 0; t <= plan.end; t += step) {
      const text = t >= 60 ? `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}` : `${t} s`;
      ctx.fillText(text, X(t) - 8, H - 6);
    }
    // End labels, held at least a line apart so two near-equal speeds stay legible.
    const yD = Y(positionAt(plan.design.samples, sim).v), yO = Y(positionAt(plan.original.samples, sim).v);
    const apart = Math.max(0, 13 - Math.abs(yD - yO)) / 2;
    const labelY = { design: yD - (yD <= yO ? apart : -apart), original: yO + (yD <= yO ? apart : -apart) };
    for (const [run, colour, name] of [[plan.original, token("--silver"), cars.original.label], [plan.design, token("--accent"), cars.design.label]]) {
      const samples = run.samples;
      const stride = Math.max(1, Math.floor(samples.length / 700));
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let drawn = false;
      for (let i = 0; i < samples.length && samples[i].t <= sim; i += stride) {
        const p = samples[i];
        if (drawn) ctx.lineTo(X(p.t), Y(p.v)); else { ctx.moveTo(X(p.t), Y(p.v)); drawn = true; }
      }
      const now = positionAt(samples, sim);
      if (drawn) { ctx.lineTo(X(sim), Y(now.v)); ctx.stroke(); }
      ctx.fillStyle = colour;
      ctx.beginPath(); ctx.arc(X(sim), Y(now.v), 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = token("--text-primary");
      ctx.fillText(`${fmt(now.v * 3.6, 1)} ${name.toLowerCase()}`, W - right + 8, labelY[run === plan.design ? "design" : "original"] + 4);
    }
  }

  return { configure, start, stop, advance, get running() { return phase === "lights" || phase === "run" || phase === "photo"; }, get phase() { return phase; } };
}

const km = (m, d) => `${fmt(m / 1000, d ?? (m % 1000 ? 1 : 0))} km`;
const metres = (m) => (m >= 1000 ? `${fmt(m / 1000, 2)} km` : m >= 100 ? `${fmt(m, 0)} m` : m >= 1 ? `${fmt(m, 1)} m` : `${fmt(m * 100, 0)} cm`);

// ---------------------------------------------------------------- the track

// One tile of track from 440 m ahead to 440 m behind the scene origin: road,
// verges, lamp posts every 40 m and distance boards every 100 m, all
// periodic in the 200 m recentring step. The start and finish gantries move
// with the race.
function buildScenery() {
  const group = new THREE.Group();
  const half = 440;
  const road = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * half, 11),
    new THREE.MeshStandardMaterial({ map: trackTexture(2 * half), roughness: 0.7, metalness: 0.05, envMapIntensity: 0.35 }),
  );
  road.rotation.x = -Math.PI / 2;
  road.position.y = 0.001;
  road.receiveShadow = true;
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(2 * half + 400, 600), new THREE.MeshStandardMaterial({ color: "#0b0e0c", roughness: 1 }));
  grass.rotation.x = -Math.PI / 2;
  grass.position.y = -0.01;
  group.add(grass, road);

  const count = Math.ceil((2 * half) / 40) * 2 + 2;
  const poleGeo = new THREE.CylinderGeometry(0.06, 0.08, 6, 6);
  poleGeo.translate(0, 3, 0);
  const lampGeo = new THREE.BoxGeometry(0.9, 0.12, 0.3);
  lampGeo.translate(0, 6, 0);
  const poles = new THREE.InstancedMesh(poleGeo, new THREE.MeshStandardMaterial({ color: "#3a3d42", roughness: 0.6, metalness: 0.5 }), count);
  const lamps = new THREE.InstancedMesh(lampGeo, new THREE.MeshStandardMaterial({ color: "#fff4d6", emissive: "#ffd9a0", emissiveIntensity: 3 }), count);
  const m = new THREE.Matrix4();
  let k = 0;
  for (let x = -half; x <= half && k < count; x += 40) {
    for (const z of [-7.2, 7.2]) { m.makeTranslation(x, 0, z); poles.setMatrixAt(k, m); lamps.setMatrixAt(k, m); k++; }
  }
  poles.count = lamps.count = k;
  group.add(poles, lamps);

  const boards = [];
  for (let i = 0; i <= 8; i++) {
    const canvas = document.createElement("canvas");
    canvas.width = 256; canvas.height = 128;
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const board = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.8), new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }));
    board.position.set(400 - i * 100, 1.4, -6.4);
    board.userData = { canvas, texture, offset: 400 - i * 100 };
    boards.push(board);
    group.add(board);
  }

  const start = gantry("START", "#e9edf2", 0.4, true);
  const finish = gantry("FINISH", "#ff6a2b", 1.4, false);
  const startLine = line("#9ea4ab");
  const finishLine = line(null);
  group.add(start.group, finish.group, startLine, finishLine);

  function relabel(origin, plan) {
    for (const board of boards) {
      // Scene x = -(race m - origin), so a board at scene offset u marks race metre origin - u.
      const xm = origin - board.userData.offset;
      const show = xm > 0 && !(plan?.mode === "distance" && xm >= plan.settings.distance);
      board.visible = show;
      if (!show) continue;
      const ctx = board.userData.canvas.getContext("2d");
      ctx.fillStyle = "#0b0c0e"; ctx.fillRect(0, 0, 256, 128);
      ctx.fillStyle = "#e9edf2"; ctx.font = "600 56px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(xm >= 1000 ? `${fmt(xm / 1000, xm % 1000 ? 1 : 0)} km` : `${xm} m`, 128, 68);
      board.userData.texture.needsUpdate = true;
    }
  }
  relabel(0, null);

  function placeLines(startX, finishX) {
    const near = (x) => x !== null && Math.abs(x) < half;
    start.group.visible = startLine.visible = near(startX);
    start.group.position.x = startLine.position.x = startX;
    finish.group.visible = finishLine.visible = near(finishX);
    if (finishX !== null) finish.group.position.x = finishLine.position.x = finishX;
  }

  return { group, relabel, placeLines, setLamps: start.setLamps };
}

function setSky(scene) {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, "#03050a");
  g.addColorStop(0.6, "#111a28");
  g.addColorStop(1, "#2c3648");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  scene.background = texture;
  scene.fog = new THREE.Fog("#0f1622", 70, 330);
}

// 10 m of road per repeat, so the 200 m recentring lands on a whole repeat.
function trackTexture(length) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#17191c";
  ctx.fillRect(0, 0, 512, 256);
  for (let k = 0; k < 1600; k++) {
    ctx.fillStyle = `rgba(255,255,255,${0.012 + Math.random() * 0.03})`;
    ctx.fillRect(Math.random() * 512, Math.random() * 256, 2, 2);
  }
  ctx.fillStyle = "#d9dde2";
  ctx.fillRect(0, 10, 512, 5);
  ctx.fillRect(0, 241, 512, 5);
  ctx.fillRect(0, 126, 256, 4);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.repeat.set(length / 10, 1);
  return texture;
}

function line(colour) {
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
  mesh.position.y = 0.004;
  return mesh;
}

// An overhead gantry with a sign; the start one carries the red, yellow and
// green lamps that the broadcast lights mirror.
function gantry(text, colour, glow, withLamps) {
  const group = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: "#2b2e33", roughness: 0.45, metalness: 0.7 });
  for (const z of [-6.2, 6.2]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.35, 6.2, 0.35), metal);
    post.position.set(0, 3.1, z);
    group.add(post);
  }
  const beam = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.2, 12.8), metal);
  beam.position.set(0, 6.2, 0);
  const sign = new THREE.Mesh(
    new THREE.PlaneGeometry(withLamps ? 5 : 9, 1.0),
    new THREE.MeshStandardMaterial({ map: labelTexture(text, "#05080c", colour), emissive: colour, emissiveIntensity: glow, emissiveMap: labelTexture(text, "#000000", "#ffffff") }),
  );
  sign.position.set(0.27, 6.2, withLamps ? -3.2 : 0);
  sign.rotation.y = Math.PI / 2;
  group.add(beam, sign);
  const lamps = [];
  if (withLamps) {
    ["#ff3b30", "#ffcc00", "#34c759"].forEach((hex, i) => {
      const lamp = new THREE.Mesh(new THREE.CircleGeometry(0.34, 32), new THREE.MeshStandardMaterial({ color: "#222326", emissive: hex, emissiveIntensity: 0 }));
      lamp.position.set(0.27, 6.2, 0.6 + i * 1.0);
      lamp.rotation.y = Math.PI / 2;
      lamps.push(lamp);
      group.add(lamp);
    });
  }
  const setLamps = (state) => lamps.forEach((lamp, i) => { lamp.material.emissiveIntensity = i === state ? 4 : 0; });
  return { group, setLamps };
}

function labelTexture(text, background, colour) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, 512, 64);
  ctx.fillStyle = colour;
  ctx.font = "700 40px 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 34);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}
