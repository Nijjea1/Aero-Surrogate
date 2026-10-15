// Aero Surrogate designer.  OWNER: Paramveer.
//
// Talks to Pranoy's service (src/service/app.py):
//   GET /health, GET /schema, POST /evaluate
//   POST /fields, when /health.fields says available: per-face predicted Cp and
//   its uncertainty on the same mesh the viewer draws (bound in fieldmap.js)
// and to web/serve.py for the 3D car (POST /api/mesh). Reads designs.json,
// real test-set cars with their CFD Cd, written by `python -m src.eval.showcase`.
//
// The controls are built from /schema, so the page follows whatever parameter
// space the service runs, without code changes here.
//
// Point it at another service with ?api=http://host:port
// ?run=<run id>&step=<n> opens that step of an optimisation run (runs.html links here).

import {
  api, animateNumber, bindInfo, escapeHtml, fmt, hideTooltip, infoButton, infoHtml, initTheme, LOGO, pageUrl,
  REDUCED_MOTION, ServiceError, showTooltip, signed, toast,
} from "./common.js";
import { aeroLoads } from "./physics.js";
import { bindField, meshIdentity } from "./fieldmap.js";

// Readable names, units and one line on what each dial does to the car.
const PARAM_INFO = {
  A_Car_Length: ["Car length", "mm", "Stretches the whole body; a longer tail lets the air close in more gently."],
  A_Car_Width: ["Car width", "mm", "Widens the body, and with it the frontal area that every newton of drag scales with."],
  A_Car_Roof_Height: ["Roof height", "mm", "Raises or lowers the roof line: more frontal area up, a slimmer profile down."],
  H_Front_Bumper_Length: ["Front bumper length", "mm", "How far the nose reaches ahead of the front wheels."],
  H_Front_Bumper_Curvature: ["Front bumper curvature", "", "How rounded the nose is in plan view. Rounder noses ease the air around the corners."],
  D_Winscreen_Inclination: ["Windscreen rake", "", "Tilts the windscreen back. A faster rake turns the air more gently over the roof."],
  D_Winscreen_Length: ["Windscreen length", "mm", "Moves where the glass meets the roof."],
  E_A_B_C_Pillar_Thickness: ["Pillar thickness", "mm", "Thickens the roof pillars, pinching or widening the glasshouse."],
  D_Rear_Window_Inclination: ["Rear window angle", "", "How steeply the rear glass drops. Too steep and the flow lets go early, growing the wake."],
  D_Rear_Window_Length: ["Rear window length", "mm", "Lengthens the rear glass toward the tail."],
  B_Trunklid_Angle: ["Trunk lid angle", "°", "Kicks the trunk lid up or down: a small lip can tidy the wake."],
  G_Trunklid_Length: ["Trunk lid length", "mm", "Extends the deck behind the rear glass."],
  G_Trunklid_Curvature: ["Trunk lid curvature", "", "How much the trunk lid crowns."],
  B_Ramp_Angle: ["Underbody ramp", "°", "Angle of the rear underbody ramp, which starts recovering pressure under the car."],
  B_Diffusor_Angle: ["Diffuser angle", "°", "Opens the diffuser under the tail. Moderate angles slow the underbody air gently."],
  E_Fenders_Arch_Offset: ["Fender arch offset", "mm", "Pushes the wheel arches out or in around the tyres."],
};
const GROUPS = [
  ["Proportions", ["A_Car_Length", "A_Car_Width", "A_Car_Roof_Height"]],
  ["Front", ["H_Front_Bumper_Length", "H_Front_Bumper_Curvature", "D_Winscreen_Inclination", "D_Winscreen_Length"]],
  ["Cabin", ["E_A_B_C_Pillar_Thickness", "D_Rear_Window_Inclination", "D_Rear_Window_Length"]],
  ["Rear", ["B_Trunklid_Angle", "G_Trunklid_Length", "G_Trunklid_Curvature"]],
  ["Underbody & wheels", ["B_Ramp_Angle", "B_Diffusor_Angle", "E_Fenders_Arch_Offset"]],
];

// What each region does for drag, for the hover cards.
const REGION_NOTES = {
  front_fascia: "Where the oncoming air stops against the car. A rounder, lower nose lowers this pressure peak.",
  hood: "Air accelerating up from the nose toward the windscreen.",
  windscreen: "A steep windscreen holds pressure here; a raked one lets the air turn gently onto the roof.",
  roof: "The fastest air on the car. Suction here mostly lifts; it only costs drag if the flow lets go.",
  rear_window: "Where the flow decides to stay attached or separate. That decision sets the size of the wake.",
  rear_fascia: "The base of the car. Low pressure here pulls the car backward: most of a car's drag lives here.",
  underbody: "Air squeezed between car and road. A diffuser slows it gently and recovers pressure.",
  wheel_wells: "Rotating wheels churn the air: a large, stubborn share of a car's drag.",
};

const PRESETS = {
  baseline: { label: "Baseline", hint: "The unmodified body of this style.", params: (schema) => ({ ...schema.defaults }) },
  random: {
    label: "Random", hint: "A uniform draw from the design box, like the dataset's own designs.",
    params: (schema) => mapBounds(schema, (lo, hi) => lo + Math.random() * (hi - lo)),
  },
  edge: {
    label: "Edge of the box",
    hint: "Every dial near a limit at once: each value is legal, but the combination is unlike any design the model learned from, so the gate should reject it.",
    params: (schema) => {
      let i = 0;
      return mapBounds(schema, (lo, hi) => (i++ % 2 ? lo + 0.03 * (hi - lo) : hi - 0.03 * (hi - lo)));
    },
  },
};

const SCENES = {
  studio: { label: "Studio", caption: "Drag to orbit, scroll to zoom. Hover a pin or the paint for what each number means." },
  tunnel: { label: "Wind tunnel", caption: "" },
  drive: { label: "Road", caption: "" },
  race: { label: "Drag race", caption: "Same engine, same mass, same tyres: only the predicted drag differs. An illustration of what the prediction means, not a vehicle simulation." },
};
const FLOW_CAPTION = "<strong>Airflow is an illustration driven by the prediction, not a simulation.</strong> " +
  "The smoke takes the pressure of each region it passes (colour, and speed by Bernoulli); the wake grows with " +
  "predicted Cd, bends with predicted lift when the model predicts it, and trembles with σ. Shown in slow motion.";
function paintCaption(mode) {
  if (mode === "regions") {
    const { source } = regionSummary();
    return source === "illustrative"
      ? "<strong>Regions</strong> are painted with the illustrative mean Cp of each: this model predicts no surface pressure. Grey surfaces belong to no region."
      : `<strong>Regions</strong> are painted with each region's mean Cp, ${source === "predicted" ? "predicted by the surface-pressure model" : source === "stub" ? "from the synthetic stub" : "predicted by the model"}. Grey surfaces belong to no region.`;
  }
  if (mode === "field" && !state.fieldsAvailable) {
    return "<strong>Pressure paint is illustrative:</strong> worked out from each surface's angle to the air, not predicted. This service does not serve surface pressure.";
  }
  if (mode === "field" || mode === "sigma") {
    if (state.fieldProblem) return `<strong>Predicted pressure is not shown:</strong> ${escapeHtml(state.fieldProblem)}.`;
    if (!state.field) return "Predicting surface pressure…";
    return mode === "field"
      ? "<strong>Predicted surface pressure</strong> from the field model, face by face, blended to the corners for display. " +
        "<strong>Predicted, not trusted:</strong> no distribution check covers field predictions, and this generated car only approximates the CFD cars it learned from."
      : "<strong>Pressure uncertainty:</strong> the field model's Cp standard deviation, face by face. It is not the Cd σ the gate uses, and it is not calibrated on this geometry.";
  }
  return "";
}

const state = {
  schema: null,
  health: null,
  params: {},
  bodyStyle: null,
  label: "Baseline",
  baselines: {},      // body style -> evaluation of the default design, for deltas
  baselineMesh: {},   // body style -> mesh payload of the default design, for the race
  current: null,
  history: [],
  session: 0,         // bumps on each preset or history load; one slider session = one history row
  requestSeq: 0,
  viewer: null,
  viewerMode: "plain",
  meshSeq: 0,
  flow: null,         // airflow animation (web/flow.js), once three.js has loaded
  scene: "drive",
  paused: false,
  pins: null,
  race: null,
  paint: null,
  labels: true,
  speedKmh: 110,
  area: null,         // frontal area of the car on screen, m^2, from /api/mesh
  baselineArea: null, // and of the selected style's baseline body
  realCars: null,     // designs.json, when present
  reference: null,    // the real car on screen, until any dial or the style changes
  fieldsAvailable: false,  // /health.fields.available: the service predicts surface pressure
  fieldSeq: 0,
  fieldResponse: null,     // { key, response | error } for the latest design asked
  meshIdentity: null,      // { key, identity } of the car on screen
  field: null,             // the bound field (fieldmap.js) on the car on screen, or null
  fieldProblem: null,      // why a field could not be shown
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- service

function designRequest(params) {
  const body = { params };
  // body_style only exists from contracts 2.0.0; the 1.0.0 service forbids extra keys.
  if (state.bodyStyle) body.body_style = state.bodyStyle;
  if (state.reference && state.bodyStyle) body.design_id = state.reference.design_id;
  return body;
}

async function evaluate(params) {
  const seq = ++state.requestSeq;
  const started = performance.now();
  const evaluation = await api("/evaluate", designRequest(params));
  const roundTrip = performance.now() - started;
  if (seq !== state.requestSeq) return null;  // a newer request superseded this one
  return { evaluation, roundTrip };
}

// ---------------------------------------------------------------- gate

// Mirrors src/service/gate.py so the badge says exactly what the agent is told.
// Same checks in the same order: a malformed prediction, then
// contracts.is_in_distribution (body style, sigma, distance from the training
// set), then the model's own flag.
function rejectionReason(e, bodyStyle) {
  const { uncertainty_threshold: threshold, ood_score_threshold: oodThreshold, body_styles: styles } = state.schema;
  for (const name of ["cd", "cl", "uncertainty"]) {
    if (typeof e[name] !== "number" || !Number.isFinite(e[name])) return `Invalid prediction: ${name} is not a finite number.`;
  }
  if (e.uncertainty < 0) return "Invalid prediction: negative uncertainty.";
  const score = e.ood_score ?? null;
  if (score !== null && (typeof score !== "number" || !Number.isFinite(score) || score < 0)) {
    return "Invalid prediction: the distance score is not a finite, nonnegative number.";
  }
  if (typeof e.in_distribution !== "boolean") return "Invalid prediction: no distribution flag.";
  if (bodyStyle && Array.isArray(styles) && !styles.includes(bodyStyle)) return `Unknown body style '${bodyStyle}'.`;
  if (e.uncertainty > threshold) {
    return `Uncertainty σ = ${fmt(e.uncertainty, 4)} is above the ${fmt(threshold, 4)} threshold.`;
  }
  if (score !== null && typeof oodThreshold === "number" && score > oodThreshold) {
    return `Distance from the training data is ${fmt(score, 2)}, above the ${fmt(oodThreshold, 2)} limit. ` +
      "Every dial may be legal, but no real design looks like this combination.";
  }
  if (!e.in_distribution) return "Outside the training distribution. The model would be guessing here.";
  return null;
}

// False when /health says Cl is a placeholder or missing, not a model output.
function liftPredicted() {
  const source = state.health?.output_provenance?.cl;
  return source !== "placeholder" && source !== "unavailable";
}

// ---------------------------------------------------------------- controls

function buildControls() {
  const { schema } = state;

  $("presets").replaceChildren(...Object.entries(PRESETS).map(([key, preset]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset";
    button.dataset.preset = key;
    button.textContent = preset.label;
    button.addEventListener("click", () => applyPreset(key));
    return button;
  }));

  const styles = schema.body_styles;
  if (Array.isArray(styles) && styles.length) {
    state.bodyStyle = schema.default_body_style || styles[0];
    $("body-style").replaceChildren(...styles.map((style) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.style = style;
      button.textContent = style[0].toUpperCase() + style.slice(1);
      button.addEventListener("click", () => {
        if (state.bodyStyle === style) return;
        setStyleUi(style);
        state.session++;  // a different car: its own history row
        markCustom();
        scheduleEvaluate();
      });
      return button;
    }));
    setStyleUi(state.bodyStyle);
    $("body-style-field").hidden = false;
  }

  // Dials grouped by part of the car; anything the groups do not name goes last.
  const names = Object.keys(schema.param_bounds);
  const grouped = new Set(GROUPS.flatMap(([, list]) => list));
  const groups = [...GROUPS.map(([title, list]) => [title, list.filter((n) => n in schema.param_bounds)]),
    ["Other", names.filter((n) => !grouped.has(n))]].filter(([, list]) => list.length);
  $("sliders").replaceChildren(...groups.map(([title, list], k) => {
    const details = document.createElement("details");
    details.className = "group";
    details.open = k < 3;
    details.innerHTML = `<summary>${escapeHtml(title)}<span class="group-count"></span></summary>`;
    details.append(...list.map(buildDial));
    return details;
  }));

  $("reset-button").addEventListener("click", () => applyPreset("baseline"));
}

function buildDial(name) {
  const [lo, hi] = state.schema.param_bounds[name];
  const [label, unit, about] = PARAM_INFO[name] || [name, "", ""];
  const step = (hi - lo) / 200;
  const row = document.createElement("div");
  row.className = "dial";
  row.dataset.name = name;
  row.innerHTML = `
    <div class="dial-top">
      <label class="dial-name" for="p-${name}" title="${escapeHtml(about || name)}">${escapeHtml(label)}</label>
      <input class="dial-value" id="v-${name}" type="number" step="${decimalsStep(lo, hi)}" min="${lo}" max="${hi}"
             aria-label="${escapeHtml(label)} value">
      <span class="dial-unit">${escapeHtml(unit)}</span>
      <button type="button" class="dial-reset" aria-label="Reset ${escapeHtml(label)}" title="Reset to the baseline">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>
      </button>
    </div>
    <div class="range">
      <span class="range-track"></span><span class="range-fill"></span><span class="range-zero"></span>
      <input type="range" id="p-${name}" min="${lo}" max="${hi}" step="${step}">
    </div>`;
  const slider = row.querySelector("input[type=range]");
  const box = row.querySelector(".dial-value");
  const set = (value) => {
    state.params[name] = Math.min(hi, Math.max(lo, value));
    updateDial(name);
    markCustom();
    scheduleEvaluate();
  };
  slider.addEventListener("input", () => set(Number(slider.value)));
  box.addEventListener("change", () => { if (box.value !== "" && Number.isFinite(Number(box.value))) set(Number(box.value)); else updateDial(name); });
  box.addEventListener("keydown", (event) => { if (event.key === "Enter") box.blur(); });
  row.querySelector(".dial-reset").addEventListener("click", () => set(state.schema.defaults[name]));
  slider.addEventListener("dblclick", () => set(state.schema.defaults[name]));
  row.querySelector(".dial-name").addEventListener("pointerenter", (event) => {
    if (about) showTooltip(event, `<strong>${escapeHtml(label)}</strong><div class="tip-meaning">${escapeHtml(about)}</div>` +
      `<div class="tip-source">${escapeHtml(name)} · ${lo} to ${hi}${unit ? ` ${unit}` : ""}, 0 is the unmodified car</div>`);
  });
  row.querySelector(".dial-name").addEventListener("pointerleave", hideTooltip);
  return row;
}

function decimalsStep(lo, hi) {
  return 10 ** -decimalsFor(lo, hi);
}

function setStyleUi(style) {
  state.bodyStyle = style;
  for (const button of $("body-style").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.style === style));
  }
}

function syncSliders() {
  for (const name of Object.keys(state.schema.param_bounds)) updateDial(name);
}

// The fill runs from the baseline (0) to the value, either way, since every
// dial is a delta from the unmodified car.
function updateDial(name) {
  const [lo, hi] = state.schema.param_bounds[name];
  const [label, unit] = PARAM_INFO[name] || [name, ""];
  const value = state.params[name];
  const base = state.schema.defaults[name];
  const row = $(`p-${name}`).closest(".dial");
  $(`p-${name}`).value = value;
  $(`p-${name}`).setAttribute("aria-valuetext", `${fmt(value, decimalsFor(lo, hi))}${unit ? ` ${unit}` : ""}`);
  if (document.activeElement !== $(`v-${name}`)) $(`v-${name}`).value = fmt(value, decimalsFor(lo, hi));
  const at = (v) => ((v - lo) / (hi - lo)) * 100;
  const fill = row.querySelector(".range-fill");
  fill.style.left = `${Math.min(at(value), at(base))}%`;
  fill.style.width = `${Math.abs(at(value) - at(base))}%`;
  row.querySelector(".range-zero").style.left = `${at(base)}%`;
  const changed = Math.abs(value - base) > (hi - lo) * 1e-6;
  row.dataset.changed = String(changed);
  row.title = changed ? "" : `${label}: at the baseline`;
  const group = row.closest(".group");
  if (group) {
    const count = group.querySelectorAll('.dial[data-changed="true"]').length;
    group.querySelector(".group-count").textContent = count ? `${count} changed` : "";
  }
}

function applyPreset(key) {
  const preset = PRESETS[key];
  state.params = clampToBounds(preset.params(state.schema));
  state.label = preset.label;
  setReference(null);
  state.session++;
  for (const button of document.querySelectorAll(".preset")) {
    button.setAttribute("aria-pressed", String(button.dataset.preset === key));
  }
  $("preset-hint").textContent = preset.hint;
  syncSliders();
  runEvaluate();
}

function markCustom() {
  state.label = "Custom";
  setReference(null);  // edited, so no longer the car CFD measured
  for (const button of document.querySelectorAll(".preset")) button.setAttribute("aria-pressed", "false");
  $("preset-hint").textContent = "Your own design. Type a value, drag a dial, or reset it with ↺.";
}

// ---------------------------------------------------------------- real cars

async function loadRealCars() {
  try {
    const response = await fetch("designs.json", { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    const styles = state.schema.body_styles;
    // Only cars this service can score: its body styles and every parameter in bounds.
    const cars = (data.designs || []).filter((car) => Array.isArray(styles) && styles.includes(car.body_style) &&
      Object.entries(state.schema.param_bounds).every(([k, [lo, hi]]) => car.params?.[k] >= lo && car.params[k] <= hi));
    if (!cars.length) return;
    state.realCars = { ...data, designs: cars };
  } catch {
    return;  // no file, no picker
  }
  const select = $("real-car");
  for (const style of state.schema.body_styles) {
    const group = document.createElement("optgroup");
    group.label = style[0].toUpperCase() + style.slice(1);
    for (const car of state.realCars.designs.filter((c) => c.body_style === style)) {
      group.append(new Option(`${dragBand(car.cd_percentile)} · ${car.design_id} · CFD Cd ${fmt(car.cd_cfd, 4)}`, car.design_id));
    }
    if (group.children.length) select.append(group);
  }
  select.addEventListener("change", () => {
    const car = state.realCars.designs.find((c) => c.design_id === select.value);
    if (car) applyRealCar(car);
  });
  $("real-car-field").hidden = false;
}

function dragBand(percentile) {
  return percentile < 34 ? "Low drag" : percentile > 66 ? "High drag" : "Typical";
}

function applyRealCar(car) {
  state.params = clampToBounds(car.params);
  setStyleUi(car.body_style);
  state.label = car.design_id;
  state.session++;
  for (const button of document.querySelectorAll(".preset")) button.setAttribute("aria-pressed", "false");
  setReference(car);
  $("preset-hint").textContent = "A real DrivAerNet car from the test set. The model never trained on it; " +
    "CFD measured its drag, shown under the prediction.";
  syncSliders();
  runEvaluate();
}

function setReference(car) {
  state.reference = car;
  if (state.realCars) $("real-car").value = car ? car.design_id : "";
}

function renderCfd() {
  const box = $("cfd-check");
  const car = state.current.reference;
  box.hidden = !car;
  if (!car) return;
  const e = state.current.evaluation;
  const error = e.cd - car.cd_cfd;
  const sigma = e.uncertainty;
  const z = sigma > 0 ? Math.abs(error) / sigma : Infinity;

  $("cfd-id").textContent = `${car.design_id} · ${car.body_style}`;
  $("cfd-truth").textContent = fmt(car.cd_cfd, 4);
  $("cfd-noise").textContent = typeof car.cd_cfd_std === "number" ? `solver spread ± ${fmt(car.cd_cfd_std, 4)}` : "";
  $("cfd-pred").textContent = fmt(e.cd, 4);
  $("cfd-sigma").textContent = `± ${fmt(sigma, 4)}`;
  $("cfd-error").textContent = `${signed(error, 4)} (${signed((100 * error) / car.cd_cfd, 1)}%)`;
  $("cfd-within").textContent = z <= 1 ? "within 1σ" : z <= 2 ? "within 2σ" : `${fmt(z, 1)}σ away`;
  $("cfd-within").dataset.good = String(z <= 2);

  // One Cd axis, wide enough for the 2σ interval and the CFD value with a margin.
  const half = Math.max(2.6 * sigma, Math.abs(error) * 1.3, 0.004);
  const mid = (e.cd + car.cd_cfd) / 2;
  const lo = mid - half, hi = mid + half;
  const at = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const span = (el, a, b) => { el.style.left = at(a); el.style.width = `calc(${at(b)} - ${at(a)})`; };
  span($("cfd-band2"), e.cd - 2 * sigma, e.cd + 2 * sigma);
  span($("cfd-band1"), e.cd - sigma, e.cd + sigma);
  $("cfd-dot").style.left = at(e.cd);
  $("cfd-mark").style.left = at(car.cd_cfd);
  $("cfd-scale").textContent = `Cd ${fmt(lo, 3)} to ${fmt(hi, 3)}`;

  const reason = rejectionReason(e, state.current.bodyStyle);
  $("cfd-note").textContent = state.health?.synthetic
    ? "The service is running the synthetic stub, so this comparison means nothing yet."
    : reason
      ? "The gate rejected this prediction, so the agent would not have trusted it either way."
      : "A test-set car, chosen by its CFD drag, not by how well the model does on it.";
}

// ---------------------------------------------------------------- evaluation

let pending = null;
function scheduleEvaluate() {
  clearTimeout(pending);
  pending = setTimeout(runEvaluate, 120);
}

async function runEvaluate({ record = true } = {}) {
  const params = { ...state.params };
  refreshMesh(params);
  requestField(params);
  try {
    const result = await evaluate(params);
    if (!result) return;
    showError(null);
    state.current = { ...result, params, label: state.label, bodyStyle: state.bodyStyle, session: state.session,
      reference: state.reference };
    if (record) pushHistory(state.current);
    renderResult();
  } catch (error) {
    showError(error);
  }
}

// The baseline body differs per style, so each style's deltas are against its
// own default design. Fetched once per style, outside the request sequence so
// it never cancels the design being shown.
const baselineKey = () => state.bodyStyle || "default";

async function ensureBaseline() {
  const key = baselineKey();
  if (key in state.baselines) return;
  state.baselines[key] = null;  // in flight
  try {
    const body = { params: clampToBounds(state.schema.defaults) };
    if (state.bodyStyle) body.body_style = state.bodyStyle;
    state.baselines[key] = await api("/evaluate", body);
    if (state.current && baselineKey() === key) renderResult();
  } catch {
    delete state.baselines[key];  // try again next time
  }
}

// ---------------------------------------------------------------- rendering

function renderResult() {
  const { evaluation: e, roundTrip } = state.current;
  const threshold = state.schema.uncertainty_threshold;
  const reason = rejectionReason(e, state.current.bodyStyle);
  const distance = typeof e.ood_score === "number" && typeof state.schema.ood_score_threshold === "number"
    ? `distance from the training data ${fmt(e.ood_score, 2)} is within ${fmt(state.schema.ood_score_threshold, 2)}, and `
    : "";

  const verdict = $("verdict");
  verdict.dataset.state = reason ? "rejected" : "trusted";
  verdict.querySelector(".verdict-icon").textContent = reason ? "✕" : "✓";
  $("verdict-label").innerHTML = `${reason ? "Rejected by the gate" : "Trusted by the gate"} ${infoButton("gate")}`;
  $("verdict-reason").textContent = reason ||
    `Inside the training distribution: ${distance}σ is below the ${fmt(threshold, 4)} threshold.`;

  animateNumber($("cd-value"), e.cd, (v) => fmt(v, 4));
  $("cl-value").textContent = liftPredicted() ? signed(e.cl, 4) : "–";
  animateNumber($("sigma-value"), e.uncertainty, (v) => `± ${fmt(v, 4)}`);
  const baseline = state.baselines[baselineKey()];
  if (baseline === undefined) ensureBaseline();
  renderDelta($("cd-delta"), e.cd, baseline?.cd, true);
  if (liftPredicted()) renderDelta($("cl-delta"), e.cl, baseline?.cl, null);
  else $("cl-delta").textContent = "Not predicted by this model";

  // Meter spans 0 to 1.5x the threshold; the mark is the threshold itself.
  const span = threshold * 1.5;
  for (const [fill, mark, meter] of [["sigma-fill", "sigma-mark", "sigma-meter"], ["hud-meter-fill", "hud-meter-mark", "hud-meter"]]) {
    $(fill).style.width = `${Math.min(e.uncertainty / span, 1) * 100}%`;
    $(mark).style.left = `calc(${(threshold / span) * 100}% - 1px)`;
    $(meter).dataset.over = String(e.uncertainty > threshold);
  }
  $("sigma-note").textContent = `Gate threshold ${fmt(threshold, 4)}`;

  // The HUD on the stage.
  $("hud-chip").dataset.state = reason ? "rejected" : "trusted";
  $("hud-chip-text").textContent = reason ? "Gate rejected" : "Trusted";
  $("hud-chip").title = reason || "The gate trusts this prediction.";
  animateNumber($("hud-cd"), e.cd, (v) => fmt(v, 4));
  renderDelta($("hud-cd-delta"), e.cd, baseline?.cd, true);
  $("hud-sigma").textContent = fmt(e.uncertainty, 4);
  $("hud-threshold").textContent = fmt(threshold, 4);

  const latency = e.latency_ms > 0 ? `model ${fmt(e.latency_ms, 1)} ms · ` : "";
  $("timing").textContent = `${latency}round trip ${fmt(roundTrip, 0)} ms · ${e.model_version}`;

  renderCfd();
  renderPressure(e.pressure_summary || {});
  renderHistory();
  state.viewer?.setRegionValues(regionSummary().values);
  updatePins();
  updateFlow();
  renderHud();
}

// ---------------------------------------------------------------- surface fields

const designKey = (params, style) => JSON.stringify([style || null, params]);

// One field per design. Busy (503) is retried briefly; any newer design
// supersedes the request, and its answer is dropped.
async function requestField(params) {
  if (!state.fieldsAvailable) return;
  const seq = ++state.fieldSeq;
  const key = designKey(params, state.bodyStyle);
  const body = designRequest(params);
  let result;
  for (let attempt = 0; ; attempt++) {
    try {
      result = { key, response: await api("/fields", body) };
      break;
    } catch (error) {
      if (seq !== state.fieldSeq) return;
      if (error.status === 503 && attempt < 4) {
        await new Promise((resolve) => setTimeout(resolve, 400 * (attempt + 1)));
        if (seq !== state.fieldSeq) return;
        continue;
      }
      result = { key, error };
      break;
    }
  }
  if (seq !== state.fieldSeq) return;
  state.fieldResponse = result;
  bindFieldIfReady();
}

// Binds only when the field and the car on screen are the same design, and
// then only when the service's mesh hash and counts match the mesh drawn.
function bindFieldIfReady() {
  const car = state.viewer?.car;
  const answer = state.fieldResponse;
  const mesh = state.meshIdentity;
  if (!car || !answer || !mesh || answer.key !== mesh.key) return;   // waiting for the other half
  let field = null;
  let problem = null;
  if (answer.error) {
    problem = answer.error.status === 503 ? "the field model is busy; it will try again with the next change"
      : `the service could not predict it (${answer.error.status || "unreachable"}: ${answer.error.detail || "no detail"})`;
  } else {
    try {
      field = bindField(answer.response, mesh.identity, car.payload);
    } catch (error) {
      problem = error.message;
    }
  }
  state.field = field;
  state.fieldProblem = problem;
  state.viewer.setField(field);
  state.viewer.setRegionValues(regionSummary().values);
  if (state.current) {
    renderPressure(regionSummary().values);
    updatePins();
    updateFlow();
  }
  setViewerMode(state.viewerMode);
}

function renderDelta(el, value, base, lowerIsBetter) {
  if (base === undefined || base === null || state.current.label === "Baseline") {
    el.textContent = "";
    el.removeAttribute("data-good");
    return;
  }
  const delta = value - base;
  const pct = base ? ` (${signed((100 * delta) / base, 1)}%)` : "";
  el.textContent = `${signed(delta, 4)}${pct} vs baseline`;
  if (lowerIsBetter === null || Math.abs(delta) < 1e-9) el.removeAttribute("data-good");
  else el.dataset.good = String(lowerIsBetter ? delta < 0 : delta > 0);
}

function renderPressure(summary) {
  const { source } = regionSummary();
  const regions = state.schema.pressure_regions;
  const values = regions.map((r) => summary[r]).filter((v) => typeof v === "number");
  // Fixed +/-1 domain so bars stay comparable between designs; widen only if needed.
  const extent = Math.max(1, Math.ceil(Math.max(0, ...values.map(Math.abs)) * 2) / 2);

  const rows = regions.map((region) => {
    const value = summary[region];
    const row = document.createElement("div");
    row.className = "p-row";
    row.setAttribute("role", "row");
    const name = regionName(region);
    if (typeof value !== "number") {
      row.classList.add("missing");
      row.innerHTML = `<span class="p-label" role="rowheader">${escapeHtml(name)}</span>
        <span class="p-track"></span><span class="p-value" role="cell">n/a</span>`;
      return row;
    }
    const half = (Math.abs(value) / extent) * 50;
    const left = value < 0 ? 50 - half : 50;
    row.innerHTML = `<span class="p-label" role="rowheader">${escapeHtml(name)}</span>
      <span class="p-track"><span class="p-bar ${value < 0 ? "neg" : "pos"}"
        style="left:${left}%;width:${half}%"></span></span>
      <span class="p-value" role="cell">${signed(value, 2)}</span>`;
    row.addEventListener("mousemove", (event) => showTooltip(event, regionCard(region, value, source)));
    row.addEventListener("mouseleave", hideTooltip);
    return row;
  });

  const axis = document.createElement("div");
  axis.className = "p-axis";
  axis.innerHTML = `<span></span><span class="p-ticks"><span>${signed(-extent, 1)}</span><span>0</span>
    <span>${signed(extent, 1)}</span></span><span></span>`;
  $("pressure").replaceChildren(...rows, axis);

  const missing = regions.filter((r) => typeof summary[r] !== "number").length;
  const gaps = missing ? ` ${missing} region${missing > 1 ? "s" : ""} not covered.` : "";
  $("pressure-note").textContent = {
    model: gaps.trim(),
    predicted: "Area-weighted means of the surface-pressure model's per-face Cp, over the regions it recognises. " +
      `Predicted, not trusted: no distribution check covers them.${gaps}`,
    stub: "Region values come from the synthetic stub, not a trained model.",
    illustrative: "This model predicts drag only. The pins on the car show illustrative pressure worked out from the shape.",
  }[source];
}

// Where the region numbers come from, best first: the scalar model's own
// regional prediction, the surface-pressure model's region means, the
// synthetic stub, or nothing (the pins then fall back to illustrative).
function regionSummary() {
  const e = state.current?.evaluation;
  const p = state.health?.output_provenance?.pressure_summary;
  if (e && p === "model") return { values: e.pressure_summary || {}, source: "model" };
  if (state.field) return { values: state.field.summary, source: "predicted" };
  if (e && (state.health?.synthetic || p === "synthetic")) return { values: e.pressure_summary || {}, source: "stub" };
  return { values: {}, source: "illustrative" };
}

function regionName(region) {
  return {
    front_fascia: "Front fascia", hood: "Hood", windscreen: "Windscreen", roof: "Roof",
    rear_window: "Rear window", rear_fascia: "Rear fascia", underbody: "Underbody", wheel_wells: "Wheels",
  }[region] || region;
}

function cpMeaning(v) {
  if (v > 0.3) return `The air is stopping here and pushes on the panel with ${fmt(100 * v, 0)}% of the oncoming dynamic pressure.`;
  if (v > 0.05) return "Slightly above free-stream pressure: the air is slowing as it meets the surface.";
  if (v > -0.05) return "About free-stream pressure: the air passes without speeding up or slowing down.";
  if (v > -0.3) return "Mild suction: the air speeds up over this surface.";
  return `Strong suction: the air accelerates hard here, pulling on the panel with ${fmt(-100 * v, 0)}% of the dynamic pressure.`;
}

const SOURCE_CHIPS = {
  model: '<span class="chip model"><i></i>Model</span> predicted by the surrogate',
  predicted: '<span class="chip predicted"><i></i>Predicted</span> by the surface-pressure model; no distribution check covers it',
  stub: '<span class="chip stub"><i></i>Stub</span> a synthetic test value',
  illustrative: '<span class="chip illustrative"><i></i>Illustrative</span> from the surface shape, not predicted',
};

function regionCard(region, value, source, sigma = null) {
  const has = typeof value === "number";
  const spread = typeof sigma === "number" ? ` ± ${fmt(sigma, 2)}` : "";
  return `<strong>${escapeHtml(region ? regionName(region) : "Surface")}</strong>` +
    `<div class="tip-value">${has ? `Cp ${signed(value, 2)}${spread}` : "no value"}</div>` +
    (has ? `<div class="tip-meaning">${escapeHtml(cpMeaning(value))}</div>` : "") +
    (region ? `<div class="tip-meaning" style="margin-top:6px">${escapeHtml(REGION_NOTES[region] || "")}</div>` : "") +
    `<div class="tip-source">${SOURCE_CHIPS[source] || ""}</div>`;
}

// Pins show predicted region values when there are any (a region they do not
// cover reads "not available"), else the illustrative pressure of each region.
function pinValues() {
  const { values: summary, source } = regionSummary();
  const car = state.viewer?.car;
  const out = {};
  for (const region of state.schema.pressure_regions) {
    if (typeof summary[region] === "number" && source !== "illustrative") out[region] = { value: summary[region], source };
    else if (source === "illustrative" && car && typeof car.fieldMeans[region] === "number") out[region] = { value: car.fieldMeans[region], source: "illustrative" };
  }
  return out;
}

function updatePins() {
  if (!state.pins) return;
  state.pinValues = pinValues();
  state.pins.setValues(state.pinValues);
  if (state.current && state.area) {
    const e = state.current.evaluation;
    const loads = aeroLoads({ cd: e.cd, cl: e.cl, sigma: e.uncertainty, area: state.area, speedKmh: state.speedKmh });
    state.pins.setDrag({ newtons: loads.drag, sigma: loads.dragSigma });
  }
}

function describePin(key) {
  if (key === "drag") {
    const e = state.current?.evaluation;
    if (!e || !state.area) return infoHtml("drag");
    const loads = aeroLoads({ cd: e.cd, sigma: e.uncertainty, area: state.area, speedKmh: state.speedKmh });
    return infoHtml("drag", `<div class="tip-value">${fmt(loads.drag, 0)} N ± ${fmt(loads.dragSigma, 0)}</div>`) +
      `<div class="tip-source">at ${state.speedKmh} km/h · Cd ${fmt(e.cd, 4)} × ${fmt(state.area, 2)} m² frontal area</div>`;
  }
  const v = state.pinValues?.[key];
  return regionCard(key, v?.value, v?.source || regionSummary().source);
}

// Dragging sliders is one design being refined, so it updates one row rather
// than adding a row per pause; a preset, a style change or a reloaded row
// starts a new one.
function pushHistory(entry) {
  const top = state.history[0];
  if (top && entry.label === "Custom" && top.label === "Custom" && top.session === entry.session) {
    state.history[0] = entry;
    return;
  }
  state.history.unshift(entry);
  state.history.length = Math.min(state.history.length, 12);
}

function renderHistory() {
  $("history").replaceChildren(...state.history.map((entry) => {
    const e = entry.evaluation;
    const ok = !rejectionReason(e, entry.bodyStyle);
    const tr = document.createElement("tr");
    const style = entry.bodyStyle ? ` · ${entry.bodyStyle}` : "";
    tr.innerHTML = `<td>${escapeHtml(entry.label + style)}</td><td class="num">${fmt(e.cd, 4)}</td>
      <td class="num">${fmt(e.uncertainty, 4)}</td>
      <td><span class="tag ${ok ? "ok" : "no"}"><i></i>${ok ? "Trusted" : "Rejected"}</span></td>`;
    tr.addEventListener("click", () => {
      state.params = { ...entry.params };
      state.label = entry.label;
      if (entry.bodyStyle) setStyleUi(entry.bodyStyle);
      setReference(entry.reference || null);
      state.session++;
      syncSliders();
      runEvaluate({ record: false });  // already in the history
    });
    return tr;
  }));
}

// ---------------------------------------------------------------- 3D stage

async function initViewer() {
  const message = $("viewer-message");
  try {
    const viewerModule = await import("./viewer.js");
    state.viewer = viewerModule.createViewer($("viewer"), { onHover: hoverCar });
    state.viewer.setPressureSource(state.fieldsAvailable ? "predicted" : "illustrative");
    state.paints = viewerModule.PAINTS;
    state.paint = viewerModule.PAINTS["Liquid silver"];
    const { createAnnotations } = await import("./annotations.js");
    state.pins = createAnnotations(state.viewer, $("pins"), { describe: describePin });
  } catch (error) {
    message.innerHTML = "<div>The 3D view could not load. three.js comes from a CDN, so it needs a connection.</div>";
    console.error(error);
    return;
  }
  buildSwatches();
  for (const button of $("viewer-modes").querySelectorAll("button")) {
    button.addEventListener("click", () => setViewerMode(button.dataset.mode));
  }
  for (const button of $("scene-modes").querySelectorAll("button")) {
    button.addEventListener("click", () => setScene(button.dataset.scene));
  }
  $("labels-toggle").addEventListener("click", () => {
    state.labels = !state.labels;
    $("labels-toggle").setAttribute("aria-pressed", String(state.labels));
    state.pins.setVisible(state.labels && state.scene !== "race");
  });
  $("speed").addEventListener("input", () => {
    state.speedKmh = Number($("speed").value);
    state.flow?.setSpeed(state.speedKmh);
    renderHud();
    updatePins();
  });
  $("flow-pause").addEventListener("click", () => {
    state.paused = !state.paused;
    state.flow?.setPaused(state.paused);
    syncPause();
  });
  $("fullscreen").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else $("stage").requestFullscreen?.().catch(() => {});
  });
  // ?debug=1 exposes the page state in the console, e.g. aero.viewer.perf, aero.flow.stats
  if (new URLSearchParams(location.search).has("debug")) window.aero = state;
  setViewerMode(state.viewerMode);
  try {
    const [{ createFlow }, { createRace }] = await Promise.all([import("./flow.js"), import("./race.js")]);
    state.flow = createFlow(state.viewer);
    state.race = createRace(state.viewer, $("race-hud"));
  } catch (error) {
    console.error(error);
  }
  state.flow?.setSpeed(state.speedKmh);
  // Respect reduced motion: the airflow starts paused, one click to play.
  state.paused = REDUCED_MOTION;
  updateFlow();
  setScene(state.scene);
}

function buildSwatches() {
  $("swatches").replaceChildren(...Object.entries(state.paints).map(([name, hex]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "swatch-dot";
    button.style.background = hex;
    button.title = name;
    button.setAttribute("aria-label", `${name} paint`);
    button.setAttribute("aria-pressed", String(hex === state.paint));
    button.addEventListener("click", () => {
      state.paint = hex;
      state.viewer.setPaint(hex);
      for (const b of $("swatches").children) b.setAttribute("aria-pressed", String(b === button));
      if (state.viewerMode !== "plain") setViewerMode("plain");
    });
    return button;
  }));
}

function syncPause() {
  const button = $("flow-pause");
  const playing = !state.paused;
  button.setAttribute("aria-pressed", String(state.paused));
  button.setAttribute("aria-label", playing ? "Pause the airflow" : "Play the airflow");
  button.innerHTML = playing
    ? '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  button.hidden = state.scene === "studio" || state.scene === "race";
}

function setScene(scene) {
  if (!state.viewer) return;
  const previous = state.scene;
  state.scene = scene;
  for (const button of $("scene-modes").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.scene === scene));
  }
  $("hud-scene").textContent = SCENES[scene].label;
  if (previous === "race" && scene !== "race") state.race?.stop();
  const flowMode = scene === "tunnel" ? "tunnel" : scene === "drive" ? "drive" : "off";
  state.flow?.setMode(flowMode);
  if (flowMode !== "off") state.flow?.setPaused(state.paused);
  state.viewer.setStage(scene === "studio" ? "studio" : "none");
  if (scene === "studio") {
    state.viewer.scene.background = null;
    state.viewer.scene.fog = null;
    state.viewer.setView("orbit", { animate: previous !== scene });
  }
  state.pins?.setVisible(state.labels && scene !== "race");
  $("hud").hidden = scene === "race";
  $("hud-verdict").hidden = scene === "race";
  if (scene === "race") startRace();
  syncPause();
  setViewerMode(state.viewerMode);
  renderHud();
}

// The race needs the original car of this style: its mesh and its prediction.
async function startRace() {
  if (!state.race || !state.current || !state.area) {
    toast("The race needs a prediction and the 3D car first.");
    return;
  }
  const style = state.bodyStyle;
  try {
    await ensureBaseline();
    if (!state.baselineMesh[style]) state.baselineMesh[style] = await fetchMesh(clampToBounds(state.schema.defaults), style);
  } catch (error) {
    toast(`No race: ${escapeHtml(error.message)}`);
    return;
  }
  const base = state.baselines[baselineKey()];
  if (!base || state.scene !== "race") return;
  const e = state.current.evaluation;
  state.race.start({
    design: { cd: e.cd, area: state.area, label: state.current.label === "Baseline" ? "Design (baseline)" : "Your design",
      trusted: !rejectionReason(e, state.current.bodyStyle) },
    original: { cd: base.cd, area: state.baselineMesh[style].frontal_area_m2, label: "Original", payload: state.baselineMesh[style],
      paint: state.paint === state.paints.Graphite ? state.paints["Liquid silver"] : state.paints.Graphite },
  });
}

// Hand the current prediction to the airflow. Called whenever the prediction,
// the baseline or the shape changes.
function updateFlow() {
  if (!state.flow || !state.current) return;
  const e = state.current.evaluation;
  const baseline = state.baselines[baselineKey()];
  state.flow.setPrediction({
    cd: e.cd, cl: liftPredicted() ? e.cl : 0, sigma: e.uncertainty,
    threshold: state.schema.uncertainty_threshold,
    trusted: !rejectionReason(e, state.current.bodyStyle),
    pressure: regionSummary().source === "illustrative" ? e.pressure_summary || {} : regionSummary().values,
    baselineCd: baseline?.cd,
  });
}

function renderHud() {
  const speedKmh = state.speedKmh;
  $("speed-value").textContent = `${speedKmh} km/h`;
  $("hud-speed").textContent = String(speedKmh);
  const [lo, hi] = [Number($("speed").min), Number($("speed").max)];
  $("speed-fill").style.left = "0%";
  $("speed-fill").style.width = `${((speedKmh - lo) / (hi - lo)) * 100}%`;
  if (!state.current || !state.area) return;
  const e = state.current.evaluation;
  const now = aeroLoads({ cd: e.cd, cl: e.cl, sigma: e.uncertainty, area: state.area, speedKmh });
  const baseline = state.baselines[baselineKey()];
  const base = baseline && state.baselineArea
    ? aeroLoads({ cd: baseline.cd, cl: baseline.cl, area: state.baselineArea, speedKmh })
    : null;
  $("hud-drag").textContent = `${fmt(now.drag, 0)} N ± ${fmt(now.dragSigma, 0)}`;
  $("hud-power").textContent = `${fmt(now.power, 1)} kW`;
  $("hud-energy").textContent = `${fmt(now.energyPer100km, 2)} kWh`;
  $("hud-lift").textContent = !liftPredicted() ? "not predicted"
    : now.lift < 0 ? `${fmt(-now.lift, 0)} N down` : `${fmt(now.lift, 0)} N up`;
  hudDelta($("hud-drag-delta"), base && now.drag - base.drag, (d) => `${signed(d, 0)} N vs baseline`);
  hudDelta($("hud-energy-delta"), base && now.energyPer100km - base.energyPer100km,
    (d) => `${signed(d, 2)} kWh vs baseline`);
  $("hud-flag").hidden = !rejectionReason(e, state.current.bodyStyle) || state.scene === "studio";
}

function hudDelta(el, delta, text) {
  const show = typeof delta === "number" && state.current.label !== "Baseline";
  el.textContent = show ? text(delta) : "";
  if (show && Math.abs(delta) > 1e-9) el.dataset.good = String(delta < 0);
  else el.removeAttribute("data-good");
}

function setViewerMode(mode) {
  state.viewerMode = mode;
  for (const button of $("viewer-modes").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  $("cp-legend").hidden = !(mode === "field" || mode === "regions") || state.scene === "race";
  $("sigma-legend").hidden = mode !== "sigma" || state.scene === "race";
  $("sigma-max").textContent = state.field ? `${fmt(state.field.sigmaScale, 2)}+` : "–";
  const flowing = state.scene === "tunnel" || state.scene === "drive";
  $("viewer-caption").innerHTML = [SCENES[state.scene]?.caption, paintCaption(mode),
    flowing ? FLOW_CAPTION : ""].filter(Boolean).join(" ");
  state.viewer?.setMode(mode);
}

async function fetchMesh(params, style) {
  const body = { params };
  if (style) body.body_style = style;
  const response = await fetch("/api/mesh", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (response.status === 404 || response.status === 501) {
    throw new Error("Start the page with <code>python web/serve.py</code> to see the car in 3D.");
  }
  const payload = await response.json();
  if (!response.ok) throw new Error(escapeHtml(payload.detail || response.statusText));
  return payload;
}

async function refreshMesh(params) {
  if (!state.viewer) return;
  const seq = ++state.meshSeq;
  let payload;
  try {
    payload = await fetchMesh(params, state.bodyStyle);
  } catch (error) {
    if (seq === state.meshSeq) {
      const message = $("viewer-message");
      message.hidden = false;
      message.innerHTML = `<div>${error.message.includes("<code>") ? error.message : `No 3D view: ${escapeHtml(error.message)}`}</div>`;
    }
    return;
  }
  if (seq !== state.meshSeq) return;   // a newer shape is on its way
  $("viewer-message").hidden = true;
  state.viewer.setMesh(payload);
  if (state.paint) state.viewer.setPaint(state.paint);
  // A new car carries no field: the matching one binds once both halves are in.
  state.field = null;
  state.meshIdentity = null;
  if (state.fieldsAvailable) {
    const key = designKey(params, state.bodyStyle);
    meshIdentity(payload).then((identity) => {
      if (seq !== state.meshSeq) return;
      state.meshIdentity = { key, identity };
      bindFieldIfReady();
    }, (error) => {
      if (seq !== state.meshSeq) return;
      state.fieldProblem = error.message;
      setViewerMode(state.viewerMode);
    });
  }
  // The prediction may have landed before the viewer finished loading.
  state.viewer.setRegionValues(regionSummary().values);
  state.area = payload.frontal_area_m2;
  state.baselineArea = payload.baseline_frontal_area_m2 ?? null;
  state.flow?.setShape(payload.flow, state.viewer.frame);
  updateFlow();
  updatePins();
  renderHud();
  const ignored = payload.ignored_params || [];
  $("viewer-ignored").hidden = !ignored.length;
  $("viewer-ignored").textContent = ignored.length
    ? `The service runs contracts ${state.schema.schema_version}; the 3D car follows the 2.0.0 parameters, ` +
      "so these sliders don't reshape it yet. It shows the baseline body until 2.0.0 lands."
    : "";
}

function hoverCar(info, event) {
  if (!info) { hideTooltip(); return; }
  if (info.source === "pending") {
    showTooltip(event, `<strong>Surface pressure</strong><div class="tip-meaning">${escapeHtml(
      state.fieldProblem ? `Not shown: ${state.fieldProblem}.` : "Waiting for the prediction for this shape.")}</div>`);
    return;
  }
  if (info.mode === "sigma") {
    showTooltip(event, `<strong>${escapeHtml(info.name ? regionName(info.name) : "Surface")}</strong>` +
      `<div class="tip-value">Cp σ ${info.sigma === null ? "–" : fmt(info.sigma, 3)}</div>` +
      `<div class="tip-meaning">How unsure the surface-pressure model is about the pressure here. Not the Cd σ the gate uses.</div>` +
      `<div class="tip-source">${SOURCE_CHIPS.predicted}</div>`);
    return;
  }
  if (!info.name && info.source !== "predicted") {
    showTooltip(event, "<strong>Side panel</strong><div class=\"tip-meaning\">No pressure region covers the sides.</div>");
    return;
  }
  const source = info.mode === "regions" ? regionSummary().source : info.source;
  showTooltip(event, regionCard(info.name, info.value, source, info.mode === "field" ? info.sigma : null));
}

// ---------------------------------------------------------------- service status

function renderService() {
  const el = $("service-status");
  if (!state.health) {
    el.dataset.state = "down";
    el.querySelector(".service-text").textContent = "Service unreachable";
    return;
  }
  el.dataset.state = "ok";
  const h = state.health;
  el.querySelector(".service-text").textContent =
    `${h.synthetic ? "Synthetic stub" : `${h.predictor} · ${h.model_version || "model"}`} · schema ${h.schema_version}`;
  el.title = h.checkpoint_sha256 ? `checkpoint ${h.checkpoint_sha256}` : "";
  $("synthetic-banner").hidden = !h.synthetic;
  state.fieldsAvailable = h.fields?.available === true;
  $("sigma-mode").hidden = !state.fieldsAvailable;
  state.viewer?.setPressureSource(state.fieldsAvailable ? "predicted" : "illustrative");
}

function showError(error) {
  $("error-banner").hidden = !error;
  if (error) $("error-text").innerHTML = error instanceof ServiceError ? error.message : escapeHtml(String(error));
}

// ---------------------------------------------------------------- helpers

function mapBounds(schema, fn) {
  return Object.fromEntries(Object.entries(schema.param_bounds).map(([k, [lo, hi]]) => [k, fn(lo, hi)]));
}

// The service never clamps (it returns 422), so the page keeps every value in bounds.
function clampToBounds(params) {
  return Object.fromEntries(Object.entries(state.schema.param_bounds).map(([k, [lo, hi]]) =>
    [k, Math.min(Math.max(Number(params[k] ?? state.schema.defaults[k]), lo), hi)]));
}

function decimalsFor(lo, hi) {
  const span = hi - lo;
  return span >= 50 ? 0 : span >= 5 ? 1 : span >= 0.5 ? 2 : 3;
}

// ---------------------------------------------------------------- start

async function start() {
  initTheme();
  bindInfo();
  $("logo").innerHTML = LOGO;
  $("runs-link").href = pageUrl("runs.html");
  $("results-link").href = pageUrl("results.html");
  document.addEventListener("fullscreenchange", () => state.viewer?.render());
  try {
    [state.health, state.schema] = await Promise.all([api("/health"), api("/schema")]);
  } catch (error) {
    renderService();
    showError(error);
    $("viewer-message").innerHTML = "<div>Waiting for the service.</div>";
    return;
  }
  renderService();
  buildControls();
  loadRealCars();
  // three.js comes from a CDN; never hold the first prediction for it.
  initViewer().then(() => refreshMesh({ ...state.params }));

  // Baseline first, so every later design can show its delta against it.
  state.params = clampToBounds(state.schema.defaults);
  await ensureBaseline();
  applyPreset("baseline");
  const query = new URLSearchParams(location.search);
  if (query.has("run")) openRunStep(query.get("run"), Number(query.get("step")));
}

async function openRunStep(runId, step) {
  try {
    const record = (await api(`/history/${encodeURIComponent(runId)}`)).find((r) => r.iteration === step);
    if (!record) throw new ServiceError(`Run ${escapeHtml(runId)} has no step ${escapeHtml(step)}.`);
    state.params = clampToBounds(record.params);
    if (record.body_style && state.schema.body_styles?.includes(record.body_style)) setStyleUi(record.body_style);
    state.label = `${runId} #${step}`;
    setReference(null);
    state.session++;
    for (const button of document.querySelectorAll(".preset")) button.setAttribute("aria-pressed", "false");
    $("preset-hint").innerHTML = `Step ${escapeHtml(step)} of run ${escapeHtml(runId)}. ` +
      `<a href="${pageUrl("runs.html", { run: runId })}">Back to the run</a>`;
    syncSliders();
    runEvaluate();
  } catch (error) {
    showError(error);
  }
}

start();
