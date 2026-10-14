// Aero Surrogate UI shell.  OWNER: Paramveer.
//
// Talks to Pranoy's service (src/service/app.py) and nothing else:
//   GET /health, GET /schema, POST /evaluate
//
// The controls are built from /schema, so the page follows whatever parameter
// space the service runs -- contracts 1.0.0 today, 2.0.0 once it merges --
// without code changes here.
//
// Point it at another service with ?api=http://host:port

const API = (new URLSearchParams(location.search).get("api") || "http://127.0.0.1:8000").replace(/\/$/, "");

// Readable names and units. Anything not listed falls back to the raw name.
const PARAM_INFO = {
  // contracts 2.0.0 -- deltas from the baseline body
  B_Ramp_Angle: ["Underbody ramp angle", "°"],
  B_Diffusor_Angle: ["Diffuser angle", "°"],
  B_Trunklid_Angle: ["Trunk lid angle", "°"],
  D_Rear_Window_Inclination: ["Rear window inclination", ""],
  D_Winscreen_Inclination: ["Windscreen inclination", ""],
  D_Winscreen_Length: ["Windscreen length", "mm"],
  D_Rear_Window_Length: ["Rear window length", "mm"],
  E_A_B_C_Pillar_Thickness: ["Pillar thickness", "mm"],
  E_Fenders_Arch_Offset: ["Fender arch offset", "mm"],
  G_Trunklid_Curvature: ["Trunk lid curvature", ""],
  G_Trunklid_Length: ["Trunk lid length", "mm"],
  H_Front_Bumper_Curvature: ["Front bumper curvature", ""],
  H_Front_Bumper_Length: ["Front bumper length", "mm"],
  A_Car_Length: ["Car length", "mm"],
  A_Car_Width: ["Car width", "mm"],
  A_Car_Roof_Height: ["Roof height", "mm"],
  // contracts 1.0.0 -- absolute values
  ride_height: ["Ride height", "m"],
  rear_deck_angle: ["Rear deck angle", "°"],
  windscreen_rake: ["Windscreen rake", "°"],
  boat_tail_angle: ["Boat-tail angle", "°"],
  front_overhang: ["Front overhang", "m"],
  roof_curvature: ["Roof curvature", ""],
  rear_diffuser_angle: ["Rear diffuser angle", "°"],
  track_width: ["Track width", "m"],
};

const REGION_NAMES = {
  front_fascia: "Front fascia",
  hood: "Hood",
  windscreen: "Windscreen",
  roof: "Roof",
  rear_window: "Rear window",
  rear_fascia: "Rear fascia",
  underbody: "Underbody",
  wheel_wells: "Wheel wells",
};

const PRESETS = {
  baseline: {
    label: "Baseline",
    hint: "The service's default design.",
    params: (schema) => ({ ...schema.defaults }),
  },
  random: {
    label: "Random",
    hint: "A uniform draw from the design box.",
    params: (schema) => mapBounds(schema, (lo, hi) => lo + Math.random() * (hi - lo)),
  },
  edge: {
    label: "Edge of the box",
    hint: "Every dial near its limit, where the model has least data. Expect the gate to reject it.",
    params: (schema) => {
      let i = 0;
      return mapBounds(schema, (lo, hi) => (i++ % 2 ? lo + 0.03 * (hi - lo) : hi - 0.03 * (hi - lo)));
    },
  },
};

const state = {
  schema: null,
  health: null,
  params: {},
  bodyStyle: null,
  label: "Baseline",
  baseline: null,     // evaluation of the default design, for deltas
  current: null,
  history: [],
  requestSeq: 0,
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- service

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let response;
  try {
    response = await fetch(API + path, options);
  } catch {
    throw new ServiceError(
      `Can't reach the service at ${API}. Start it with ` +
      "<code>uvicorn src.service.app:app --port 8000</code> and open this page at " +
      "<code>http://localhost:5173</code>.",
    );
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = Array.isArray(data?.detail)
      ? data.detail.map((d) => d.msg).join("; ")
      : data?.detail || response.statusText;
    throw new ServiceError(`The service rejected the request (${response.status}): ${escapeHtml(detail)}`);
  }
  return data;
}

class ServiceError extends Error {}

function designRequest(params) {
  const body = { params };
  // body_style only exists from contracts 2.0.0; the 1.0.0 service forbids extra keys.
  if (state.bodyStyle) body.body_style = state.bodyStyle;
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
function rejectionReason(e, threshold) {
  for (const name of ["cd", "cl", "uncertainty"]) {
    if (typeof e[name] !== "number" || !Number.isFinite(e[name])) return `Invalid prediction: ${name} is not a finite number.`;
  }
  if (e.uncertainty < 0) return "Invalid prediction: negative uncertainty.";
  if (typeof e.in_distribution !== "boolean") return "Invalid prediction: no distribution flag.";
  if (!e.in_distribution) return "Outside the training distribution. The model would be guessing here.";
  if (e.uncertainty > threshold) {
    return `Uncertainty σ = ${fmt(e.uncertainty, 4)} is above the ${fmt(threshold, 4)} threshold.`;
  }
  return null;
}

// ---------------------------------------------------------------- controls

function buildControls() {
  const { schema } = state;

  const presets = $("presets");
  presets.replaceChildren(...Object.entries(PRESETS).map(([key, preset]) => {
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
    const select = $("body-style");
    select.replaceChildren(...styles.map((s) => new Option(s[0].toUpperCase() + s.slice(1), s)));
    state.bodyStyle = schema.default_body_style || styles[0];
    select.value = state.bodyStyle;
    select.addEventListener("change", () => { state.bodyStyle = select.value; markCustom(); scheduleEvaluate(); });
    $("body-style-field").hidden = false;
  }

  const sliders = $("sliders");
  sliders.replaceChildren(...Object.entries(schema.param_bounds).map(([name, [lo, hi]]) => {
    const [label, unit] = PARAM_INFO[name] || [name, ""];
    const row = document.createElement("div");
    row.className = "slider-row";
    row.dataset.name = name;
    row.innerHTML = `
      <div class="slider-top">
        <label class="slider-name" for="p-${name}">${escapeHtml(label)}</label>
        <span class="slider-value" id="v-${name}"></span>
      </div>
      <input type="range" id="p-${name}" min="${lo}" max="${hi}" step="${(hi - lo) / 200}"
             title="${escapeHtml(name)} · ${lo} to ${hi}${unit ? " " + unit : ""} · double-click to reset">`;
    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      state.params[name] = Number(input.value);
      updateSliderLabel(name);
      markCustom();
      scheduleEvaluate();
    });
    input.addEventListener("dblclick", () => {
      state.params[name] = schema.defaults[name];
      syncSliders();
      scheduleEvaluate();
    });
    return row;
  }));

  $("reset-button").addEventListener("click", () => applyPreset("baseline"));
}

function syncSliders() {
  for (const name of Object.keys(state.schema.param_bounds)) {
    $(`p-${name}`).value = state.params[name];
    updateSliderLabel(name);
  }
}

function updateSliderLabel(name) {
  const unit = (PARAM_INFO[name] || [])[1] || "";
  const value = state.params[name];
  const [lo, hi] = state.schema.param_bounds[name];
  $(`v-${name}`).textContent = `${fmt(value, decimalsFor(lo, hi))}${unit ? " " + unit : ""}`;
  const changed = Math.abs(value - state.schema.defaults[name]) > (hi - lo) * 1e-6;
  $(`v-${name}`).closest(".slider-row").dataset.changed = String(changed);
}

function applyPreset(key) {
  const preset = PRESETS[key];
  state.params = clampToBounds(preset.params(state.schema));
  state.label = preset.label;
  for (const button of document.querySelectorAll(".preset")) {
    button.setAttribute("aria-pressed", String(button.dataset.preset === key));
  }
  $("preset-hint").textContent = preset.hint;
  syncSliders();
  runEvaluate();
}

function markCustom() {
  state.label = "Custom";
  for (const button of document.querySelectorAll(".preset")) button.setAttribute("aria-pressed", "false");
  $("preset-hint").textContent = "Your own design. Double-click a slider to reset it.";
}

let pending = null;
function scheduleEvaluate() {
  clearTimeout(pending);
  pending = setTimeout(runEvaluate, 120);
}

async function runEvaluate() {
  const params = { ...state.params };
  try {
    const result = await evaluate(params);
    if (!result) return;
    showError(null);
    state.current = { ...result, params, label: state.label, bodyStyle: state.bodyStyle };
    pushHistory(state.current);
    renderResult();
  } catch (error) {
    showError(error);
  }
}

// ---------------------------------------------------------------- rendering

function renderResult() {
  const { evaluation: e, roundTrip } = state.current;
  const threshold = state.schema.uncertainty_threshold;
  const reason = rejectionReason(e, threshold);

  const verdict = $("verdict");
  verdict.dataset.state = reason ? "rejected" : "trusted";
  verdict.querySelector(".verdict-icon").textContent = reason ? "✕" : "✓";
  $("verdict-label").textContent = reason ? "Rejected by the uncertainty gate" : "Trusted";
  $("verdict-reason").textContent = reason ||
    `Inside the training distribution, and σ is below the ${fmt(threshold, 4)} threshold.`;

  $("cd-value").textContent = fmt(e.cd, 4);
  $("cl-value").textContent = signed(e.cl, 4);
  $("sigma-value").textContent = `± ${fmt(e.uncertainty, 4)}`;
  renderDelta($("cd-delta"), e.cd, state.baseline?.cd, true);
  renderDelta($("cl-delta"), e.cl, state.baseline?.cl, null);

  // Meter spans 0 to 1.5x the threshold; the mark is the threshold itself.
  const span = threshold * 1.5;
  $("sigma-fill").style.width = `${Math.min(e.uncertainty / span, 1) * 100}%`;
  $("sigma-mark").style.left = `calc(${(threshold / span) * 100}% - 1px)`;
  $("sigma-meter").dataset.over = String(e.uncertainty > threshold);
  $("sigma-note").textContent = `Gate threshold ${fmt(threshold, 4)}`;

  const latency = e.latency_ms > 0 ? `model ${fmt(e.latency_ms, 1)} ms · ` : "";
  $("timing").textContent = `${latency}round trip ${fmt(roundTrip, 0)} ms · ${e.model_version}`;

  renderPressure(e.pressure_summary || {});
  renderHistory();
}

function renderDelta(el, value, base, lowerIsBetter) {
  if (base === undefined || base === null || state.current.label === "Baseline") {
    el.textContent = "";
    el.removeAttribute("data-good");
    return;
  }
  const delta = value - base;
  const sign = delta > 0 ? "+" : delta < 0 ? "−" : "±";
  el.textContent = `${sign}${fmt(Math.abs(delta), 4)} vs baseline`;
  if (lowerIsBetter === null || Math.abs(delta) < 1e-9) el.removeAttribute("data-good");
  else el.dataset.good = String(lowerIsBetter ? delta < 0 : delta > 0);
}

function renderPressure(summary) {
  const regions = state.schema.pressure_regions;
  const values = regions.map((r) => summary[r]).filter((v) => typeof v === "number");
  // Fixed +/-1 domain so bars stay comparable between designs; widen only if needed.
  const extent = Math.max(1, Math.ceil(Math.max(0, ...values.map(Math.abs)) * 2) / 2);

  const rows = regions.map((region) => {
    const value = summary[region];
    const row = document.createElement("div");
    row.className = "p-row";
    row.setAttribute("role", "row");
    const name = REGION_NAMES[region] || region;
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
    const base = state.baseline?.pressure_summary?.[region];
    row.addEventListener("mousemove", (event) => showTooltip(event,
      `<strong>${escapeHtml(name)}</strong>Cp ${signed(value, 3)}` +
      (typeof base === "number" && state.current.label !== "Baseline"
        ? `<br>${signed(value - base, 3)} vs baseline` : "")));
    row.addEventListener("mouseleave", hideTooltip);
    return row;
  });

  const axis = document.createElement("div");
  axis.className = "p-axis";
  axis.innerHTML = `<span></span><span class="p-ticks"><span>${signed(-extent, 1)}</span><span>0</span>
    <span>${signed(extent, 1)}</span></span><span></span>`;
  $("pressure").replaceChildren(...rows, axis);

  const missing = regions.filter((r) => typeof summary[r] !== "number").length;
  $("pressure-note").textContent = state.health?.synthetic
    ? "Region values come from the synthetic stub. Real surface pressure arrives with the stage 2 model."
    : missing ? `${missing} region${missing > 1 ? "s" : ""} not predicted by this model.` : "";
}

function pushHistory(entry) {
  state.history.unshift(entry);
  state.history.length = Math.min(state.history.length, 12);
}

function renderHistory() {
  const threshold = state.schema.uncertainty_threshold;
  $("history").replaceChildren(...state.history.map((entry) => {
    const e = entry.evaluation;
    const ok = !rejectionReason(e, threshold);
    const tr = document.createElement("tr");
    const style = entry.bodyStyle ? ` · ${entry.bodyStyle}` : "";
    tr.innerHTML = `<td>${escapeHtml(entry.label + style)}</td><td class="num">${fmt(e.cd, 4)}</td>
      <td class="num">${fmt(e.uncertainty, 4)}</td>
      <td><span class="tag ${ok ? "ok" : "no"}"><i></i>${ok ? "Trusted" : "Rejected"}</span></td>`;
    tr.addEventListener("click", () => {
      state.params = { ...entry.params };
      state.label = entry.label;
      if (entry.bodyStyle) { state.bodyStyle = entry.bodyStyle; $("body-style").value = entry.bodyStyle; }
      syncSliders();
      runEvaluate();
    });
    return tr;
  }));
}

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
    `Connected · ${h.predictor}${h.synthetic ? " (synthetic)" : ""} · schema ${h.schema_version}`;
  $("synthetic-banner").hidden = !h.synthetic;
}

function showError(error) {
  $("error-banner").hidden = !error;
  if (error) $("error-text").innerHTML = error instanceof ServiceError ? error.message : escapeHtml(String(error));
}

function showTooltip(event, html) {
  const tip = $("tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: w } = window;
  const rect = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(event.clientX + pad, w - rect.width - 8)}px`;
  tip.style.top = `${event.clientY + pad}px`;
}

function hideTooltip() { $("tooltip").hidden = true; }

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

function fmt(value, decimals) {
  return Number(value).toFixed(decimals);
}

function signed(value, decimals) {
  const text = Math.abs(value).toFixed(decimals);
  return value < 0 && Number(text) !== 0 ? `−${text}` : value > 0 && Number(text) !== 0 ? `+${text}` : text;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// ---------------------------------------------------------------- start

async function start() {
  try {
    [state.health, state.schema] = await Promise.all([api("/health"), api("/schema")]);
  } catch (error) {
    renderService();
    showError(error);
    return;
  }
  renderService();
  buildControls();

  // Baseline first, so every later design can show its delta against it.
  state.params = clampToBounds(state.schema.defaults);
  try {
    const result = await evaluate(state.params);
    state.baseline = result?.evaluation ?? null;
  } catch (error) {
    showError(error);
  }
  applyPreset("baseline");
}

start();
