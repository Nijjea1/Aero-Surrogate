// Shared by the designer (app.js), the run view (runs.js) and the results page.
// OWNER: Paramveer.
//
// Point any page at another service with ?api=http://host:port

export const API = (new URLSearchParams(location.search).get("api") || "http://127.0.0.1:8000").replace(/\/$/, "");

export const REDUCED_MOTION = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

// message is display HTML; status (0 when unreachable) and detail (plain
// text) let a page react to the specific refusal.
export class ServiceError extends Error {
  constructor(message, { status = 0, detail = "" } = {}) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

// GET without a body, POST with one. headers are added to the request.
export async function api(path, body, { headers = {} } = {}) {
  const options = body === undefined
    ? { headers }
    : { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) };
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
    throw new ServiceError(`The service rejected the request (${response.status}): ${escapeHtml(detail)}`,
      { status: response.status, detail: String(detail) });
  }
  return data;
}

// Keeps ?api= when moving between pages.
export function pageUrl(page, params = {}) {
  const query = new URLSearchParams(params);
  const apiParam = new URLSearchParams(location.search).get("api");
  if (apiParam) query.set("api", apiParam);
  const text = query.toString();
  return text ? `${page}?${text}` : page;
}

export function fmt(value, decimals) {
  return Number(value).toFixed(decimals);
}

export function signed(value, decimals) {
  const text = Math.abs(value).toFixed(decimals);
  return value < 0 && Number(text) !== 0 ? `−${text}` : value > 0 && Number(text) !== 0 ? `+${text}` : text;
}

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// ---------------------------------------------------------------- tooltips

export function showTooltipAt(x, y, html) {
  const tip = document.getElementById("tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  const left = Math.min(x + pad, window.innerWidth - rect.width - 8);
  const top = y + pad + rect.height > window.innerHeight - 8 ? y - rect.height - pad : y + pad;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

export function showTooltip(event, html) {
  showTooltipAt(event.clientX, event.clientY, html);
}

export function hideTooltip() { document.getElementById("tooltip").hidden = true; }

// What every number on the pages means, for the (i) buttons and hover cards.
export const INFO = {
  cd: ["Drag coefficient, Cd",
    "How easily the shape slips through air, whatever its size or speed. Drag force = ½ × air density × speed² × Cd × frontal area. Lower is better; road cars sit around 0.22 to 0.35."],
  cl: ["Lift coefficient, Cl",
    "The vertical aerodynamic force, scaled the same way as Cd. Positive lifts the car off the road; negative is downforce that presses it down."],
  sigma: ["Uncertainty, σ",
    "The model's own standard deviation on Cd. The CFD that produced the training data scatters by about ±0.0064 run to run, so a σ near that is honest. Above the gate threshold, the prediction is rejected."],
  cp: ["Pressure coefficient, Cp",
    "Local pressure compared with the oncoming air. +1 where the air stops dead against the car, 0 at free-stream pressure, negative where the air speeds up and sucks on the surface. Pressure on front-facing panels and suction on rear-facing ones both hold the car back."],
  gate: ["The trust gate",
    "Before a prediction is allowed to steer a design, two checks: σ, the error bar, and the distance from every design the model learned from. If either is too large, the model is guessing, and the prediction is rejected with the reason."],
  ood: ["Distance from the training data",
    "How unusual this combination of dials is compared with the designs the model trained on, in standard deviations. Every dial can be within its range while the combination is new."],
  drag: ["Drag force",
    "Newtons of air resistance at the chosen speed: ½ × air density × speed² × Cd × frontal area. The ± carries the model's σ through."],
  power: ["Power to push the air",
    "Drag force × speed: the engine power spent only on air resistance at this speed. It grows with the cube of speed."],
  energy: ["Energy per 100 km, drag only",
    "The energy the air resistance costs over 100 km at this speed. Rolling resistance and drivetrain losses come on top."],
  lift: ["Lift or downforce",
    "The vertical aerodynamic force at this speed. Shown only when the model predicts Cl; the current model predicts drag alone."],
  area: ["Frontal area",
    "The car's silhouette seen from the front, in m². Drag scales with Cd × area, so a narrower, lower car pays less even at the same Cd."],
  illustrative: ["Illustrative",
    "Worked out from the angle of each surface to the oncoming air, not predicted by the model. It shows where pressure and suction usually sit on a shape like this."],
  predicted: ["Predicted surface pressure",
    "Cp on every face of the car from the surface-pressure model. Predicted, not trusted: unlike Cd, no distribution check covers these values, and this generated car only approximates the CFD cars the model learned from."],
  cpsigma: ["Pressure uncertainty, Cp σ",
    "The surface-pressure model's own standard deviation on Cp at each face. It is not the Cd σ the gate uses, and its calibration on this geometry is not established, so read it as where the model is less sure, not as an error bar."],
  cfd: ["CFD ground truth",
    "The drag a full computational fluid dynamics simulation measured for this exact car, from the DrivAerNet dataset. The model never trained on it."],
  race: ["Equal-power race",
    "Both cars get the same engine power and mass, and only their predicted drag and frontal area differ. Integrated step by step: force = min(power ÷ speed, grip) − drag − rolling resistance."],
};

export function infoHtml(key, extra = "") {
  const [title, body] = INFO[key] || [key, ""];
  return `<strong>${escapeHtml(title)}</strong>${extra}<div class="tip-meaning">${escapeHtml(body)}</div>`;
}

// Any element with data-info="cd" opens that definition on hover or focus.
export function bindInfo(root = document) {
  const open = (el, x, y) => showTooltipAt(x, y, infoHtml(el.dataset.info));
  root.addEventListener("pointerover", (event) => {
    const el = event.target.closest?.("[data-info]");
    if (el) open(el, event.clientX, event.clientY);
  });
  root.addEventListener("pointerout", (event) => {
    if (event.target.closest?.("[data-info]")) hideTooltip();
  });
  root.addEventListener("focusin", (event) => {
    const el = event.target.closest?.("[data-info]");
    if (!el) return;
    const r = el.getBoundingClientRect();
    open(el, r.left, r.bottom);
  });
  root.addEventListener("focusout", (event) => {
    if (event.target.closest?.("[data-info]")) hideTooltip();
  });
}

export function infoButton(key) {
  return `<button type="button" class="info" data-info="${key}" aria-label="What is ${escapeHtml(INFO[key]?.[0] || key)}?">i</button>`;
}

// ---------------------------------------------------------------- theme, toasts, motion

const THEME_KEY = "aero-theme";

export function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem(THEME_KEY); } catch { /* private window */ }
  if (stored === "light") document.documentElement.dataset.theme = "light";
  const button = document.getElementById("theme-toggle");
  if (!button) return;
  const sync = () => {
    const light = document.documentElement.dataset.theme === "light";
    button.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
    button.innerHTML = light
      ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
      : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  };
  sync();
  button.addEventListener("click", () => {
    const light = document.documentElement.dataset.theme !== "light";
    if (light) document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
    try { localStorage.setItem(THEME_KEY, light ? "light" : "dark"); } catch { /* not remembered */ }
    sync();
    document.dispatchEvent(new CustomEvent("themechange"));
  });
}

export function toast(html, ms = 6000) {
  let host = document.querySelector(".toasts");
  if (!host) {
    host = document.createElement("div");
    host.className = "toasts";
    host.setAttribute("role", "status");
    document.body.append(host);
  }
  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = html;
  host.append(el);
  setTimeout(() => el.remove(), ms);
}

// Count a number to its new value; instant under reduced motion.
const counting = new WeakMap();
export function animateNumber(el, value, format, ms = 420) {
  const from = counting.get(el)?.value;
  cancelAnimationFrame(counting.get(el)?.frame);
  if (REDUCED_MOTION || typeof from !== "number" || !Number.isFinite(from) || !Number.isFinite(value)) {
    el.textContent = format(value);
    counting.set(el, { value });
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / ms);
    const eased = 1 - (1 - t) ** 3;
    el.textContent = format(from + (value - from) * eased);
    const state = { value, frame: t < 1 ? requestAnimationFrame(step) : 0 };
    if (t < 1) counting.set(el, { ...state, value: from + (value - from) * eased });
    else counting.set(el, { value });
  };
  counting.set(el, { value: from, frame: requestAnimationFrame(step) });
}

export const LOGO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 15c3-5 7-7 11-7 3 0 5 1 7 3"/><path d="M3 19c4-3 8-4 12-4 2 0 4 .5 6 1.5"/><path d="M3 11c2-3 5-5 9-5"/></svg>';
