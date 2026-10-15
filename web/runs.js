// Run view: the agent against the baseline optimisers.  OWNER: Paramveer.
//
// Reads Pranoy's run API and nothing else:
//   GET /runs              every run in the service's log, newest first
//   GET /runs/{id}         one run's status (live while it runs)
//   GET /history/{id}      its IterationRecords, agent and baselines alike
//
// ?run=<id> opens with that run in focus.

import { api, escapeHtml, fmt, hideTooltip, pageUrl, ServiceError, showTooltip, signed } from "./common.js";

const SLOTS = 4;                                  // --series-1..4, assigned in fixed order
const LIVE = new Set(["queued", "running", "unknown"]);
const POLL_MS = 2000;
// "unknown" is the service's word for a run it does not own that has not
// finished: a CLI run still going, or one cut off and not yet closed.
const STATE_TEXT = { unknown: "not finished", queued: "queued", running: "running" };

const state = {
  health: null,
  runs: [],                 // /runs summaries, newest first
  records: {},              // run id -> IterationRecords
  selected: new Map(),      // run id -> slot; a run keeps its colour while selected
  focus: null,              // run id shown in the logbook
  hoverStep: null,
  openStep: null,           // logbook row shown in full
  timer: null,
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- data

async function loadRuns() {
  state.runs = await api("/runs?limit=50");
  await Promise.all(state.runs.map((run) => loadRecords(run, false)));
}

async function loadRecords(run, force) {
  if (!force && state.records[run.run_id] && !LIVE.has(run.state)) return;
  try {
    state.records[run.run_id] = await api(`/history/${encodeURIComponent(run.run_id)}`);
  } catch {
    state.records[run.run_id] = state.records[run.run_id] || [];
  }
}

function runOf(id) {
  return state.runs.find((run) => run.run_id === id);
}

function kindOf(run) {
  return state.records[run.run_id]?.[0]?.optimiser || "agent";
}

function labelOf(run) {
  const kind = kindOf(run);
  if (kind === "cmaes") return "CMA-ES";
  if (kind === "random") return "Random search";
  return run.model ? `Agent · ${run.model}` : "Scripted agent (dry run)";
}

function summarise(run) {
  const records = state.records[run.run_id] || [];
  const start = records[0];
  let best = null;
  let bestAt = null;
  for (const r of records) {
    if (r.accepted && (best === null || r.cd < best)) { best = r.cd; bestAt = r.iteration; }
  }
  return {
    records,
    start: start ? start.cd : null,
    best,
    bestAt,
    gated: records.filter((r) => !r.accepted && !isReference(r)).length,
    budget: run.iterations ?? records.length,
  };
}

const isReference = (r) => typeof r.reject_reason === "string" && r.reject_reason.startsWith("reference only");

// ---------------------------------------------------------------- selection

function select(id, on) {
  if (on && !state.selected.has(id) && state.selected.size < SLOTS) {
    const used = new Set(state.selected.values());
    let slot = 1;
    while (used.has(slot)) slot++;
    state.selected.set(id, slot);
  } else if (!on) {
    state.selected.delete(id);
    if (state.focus === id) state.focus = state.selected.keys().next().value ?? null;
  }
}

// One agent run against the newest CMA-ES and random runs of the same style
// and budget: the comparison the README makes.
function defaultSelection(wanted) {
  const agents = state.runs.filter((run) => kindOf(run) === "agent");
  const lead = runOf(wanted) || agents.find((run) => run.model) || agents[0] || state.runs[0];
  if (!lead) return;
  select(lead.run_id, true);
  state.focus = lead.run_id;
  for (const kind of ["agent", "cmaes", "random"]) {
    if (kind === kindOf(lead)) continue;
    const match = state.runs.find((run) => kindOf(run) === kind && run.body_style === lead.body_style
      && run.iterations === lead.iterations);
    if (match) select(match.run_id, true);
  }
}

// ---------------------------------------------------------------- rendering

function render() {
  renderList();
  renderLegend();
  renderChart();
  renderCompare();
  renderLog();
  schedulePoll();
}

function renderList() {
  $("run-empty").hidden = state.runs.length > 0;
  $("run-list").replaceChildren(...state.runs.map((run) => {
    const s = summarise(run);
    const slot = state.selected.get(run.run_id);
    const li = document.createElement("li");
    li.className = "run-item";
    li.dataset.focus = String(run.run_id === state.focus);
    const full = !slot && state.selected.size >= SLOTS;
    li.innerHTML = `
      <input type="checkbox" aria-label="Compare ${escapeHtml(run.run_id)}" ${slot ? "checked" : ""} ${full ? "disabled" : ""}>
      <i class="swatch-line" ${slot ? `style="background: var(--series-${slot})"` : ""}></i>
      <button type="button" class="run-name">
        <span>${escapeHtml(labelOf(run))}</span>
        <small>${escapeHtml(run.run_id)} · ${escapeHtml(run.body_style || "")}</small>
      </button>
      <span class="run-stats">
        <span>${s.best === null ? "–" : fmt(s.best, 4)}</span>
        <small>${s.records.length}/${s.budget}${LIVE.has(run.state) ? " · live" : ""}</small>
      </span>`;
    li.querySelector("input").addEventListener("change", (event) => {
      select(run.run_id, event.target.checked);
      if (event.target.checked && !state.focus) state.focus = run.run_id;
      render();
    });
    li.querySelector(".run-name").addEventListener("click", () => {
      select(run.run_id, true);
      if (state.selected.has(run.run_id)) state.focus = run.run_id;
      state.openStep = null;
      render();
    });
    return li;
  }));
}

function selectedRuns() {
  return [...state.selected.entries()]
    .map(([id, slot]) => ({ run: runOf(id), slot }))
    .filter((entry) => entry.run)
    .sort((a, b) => a.slot - b.slot);
}

function renderLegend() {
  $("legend").replaceChildren(...selectedRuns().map(({ run, slot }) => {
    const span = document.createElement("span");
    span.innerHTML = `<i class="swatch-line" style="background: var(--series-${slot})"></i>${escapeHtml(labelOf(run))}`;
    return span;
  }));
  const runs = selectedRuns().map((entry) => entry.run);
  const styles = new Set(runs.map((run) => run.body_style));
  const budgets = new Set(runs.map((run) => run.iterations));
  const warnings = [];
  if (styles.size > 1) warnings.push("These runs optimise different body styles, so their Cd values do not compare.");
  if (budgets.size > 1) warnings.push("These runs had different evaluation budgets.");
  $("mix-warning").hidden = !warnings.length;
  $("mix-warning").textContent = warnings.join(" ");
}

function renderChart() {
  const host = $("chart");
  const entries = selectedRuns();
  if (!entries.length) {
    host.innerHTML = `<p class="chart-empty">Tick a run to plot it.</p>`;
    $("chart-meta").textContent = "";
    return;
  }
  const width = Math.max(host.clientWidth, 260);
  const narrow = width < 560;
  const height = narrow ? 260 : 320;
  const m = { left: 52, right: narrow ? 16 : 170, top: 30, bottom: 34 };
  const lane = 14;                            // the gated lane's centre, above the plot
  const focus = entries.find((e) => e.run.run_id === state.focus) || entries[0];
  const focusSummary = summarise(focus.run);

  const maxStep = Math.max(1, ...entries.map(({ run }) => Math.max(summarise(run).budget, summarise(run).records.length) - 1));
  const values = [];
  for (const { run } of entries) {
    for (const r of summarise(run).records) if (r.best_cd_so_far !== null) values.push(r.best_cd_so_far);
  }
  for (const r of focusSummary.records) if (r.accepted || isReference(r)) values.push(r.cd);
  if (focusSummary.start !== null) values.push(focusSummary.start);
  if (!values.length) values.push(0.25);
  let lo = Math.min(...values), hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.08, 0.002);
  lo -= pad; hi += pad;

  const x = (i) => m.left + (i / maxStep) * (width - m.left - m.right);
  const y = (v) => m.top + (1 - (v - lo) / (hi - lo)) * (height - m.top - m.bottom);
  const ticks = niceTicks(lo, hi, 5);
  const xticks = niceTicks(0, maxStep, narrow ? 4 : 6).filter((t) => Number.isInteger(t));

  const parts = [];
  parts.push(`<g class="grid">${ticks.map((t) => `<line x1="${m.left}" x2="${width - m.right}" y1="${y(t)}" y2="${y(t)}"/>`).join("")}</g>`);
  parts.push(`<g class="axis">${ticks.map((t) => `<text x="${m.left - 8}" y="${y(t) + 4}" text-anchor="end">${fmt(t, 3)}</text>`).join("")}`
    + xticks.map((t) => `<text x="${x(t)}" y="${height - m.bottom + 18}" text-anchor="middle">${t}</text>`).join("")
    + `<text x="${(m.left + width - m.right) / 2}" y="${height - 2}" text-anchor="middle">evaluation</text>`
    + `<line class="baseline" x1="${m.left}" x2="${width - m.right}" y1="${height - m.bottom}" y2="${height - m.bottom}"/>`
    + `<text x="${m.left - 8}" y="${lane + 4}" text-anchor="end">gated</text></g>`);

  if (focusSummary.start !== null) {
    const yy = y(focusSummary.start);
    parts.push(`<line class="start-line" x1="${m.left}" x2="${width - m.right}" y1="${yy}" y2="${yy}"/>`
      + `<text class="start-label" x="${width - m.right - 4}" y="${yy - 6}" text-anchor="end">unmodified car ${fmt(focusSummary.start, 4)}</text>`);
  }

  // Lines first, the focused run last so it sits on top.
  const ordered = [...entries.filter((e) => e !== focus), focus];
  const ends = [];
  for (const { run, slot } of ordered) {
    const records = summarise(run).records.filter((r) => r.best_cd_so_far !== null);
    if (!records.length) continue;
    let d = `M${x(records[0].iteration)},${y(records[0].best_cd_so_far)}`;
    for (let k = 1; k < records.length; k++) {
      d += `H${x(records[k].iteration)}V${y(records[k].best_cd_so_far)}`;
    }
    const last = records[records.length - 1];
    d += `H${x(Math.max(last.iteration, summarise(run).records.length - 1))}`;
    parts.push(`<path class="run-line" d="${d}" style="stroke: var(--series-${slot})"/>`);
    ends.push({ slot, label: labelOf(run), y: y(last.best_cd_so_far), x: x(summarise(run).records.length - 1) });
  }

  // Every evaluation of the focused run: trusted as dots at their Cd, gated as x in the lane.
  for (const r of focusSummary.records) {
    if (r.accepted || isReference(r)) {
      parts.push(`<circle class="step-dot${r.accepted ? "" : " reference"}" cx="${x(r.iteration)}" cy="${y(r.cd)}" r="3.5"
        style="--dot: var(--series-${focus.slot})"/>`);
    } else {
      const cx = x(r.iteration);
      parts.push(`<path class="gated-x" d="M${cx - 4},${lane - 4}L${cx + 4},${lane + 4}M${cx - 4},${lane + 4}L${cx + 4},${lane - 4}"/>`);
    }
  }

  // Direct labels at the line ends, nudged apart so they never overlap.
  if (!narrow) {
    ends.sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 14);
    for (const end of ends) {
      parts.push(`<line class="end-tick" x1="${width - m.right + 6}" x2="${width - m.right + 16}" y1="${end.y}" y2="${end.y}"
        style="stroke: var(--series-${end.slot})"/><text class="end-label" x="${width - m.right + 20}" y="${end.y + 4}">${escapeHtml(end.label)}</text>`);
    }
  }

  const hover = state.hoverStep;
  if (hover !== null && hover <= maxStep) {
    parts.push(`<line class="crosshair" x1="${x(hover)}" x2="${x(hover)}" y1="${m.top - 22}" y2="${height - m.bottom}"/>`);
  }
  parts.push(`<rect class="hit" x="${m.left}" y="0" width="${width - m.left - m.right}" height="${height - m.bottom}"/>`);

  host.innerHTML = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${parts.join("")}</svg>`;
  const hit = host.querySelector(".hit");
  const stepAt = (event) => {
    const box = host.querySelector("svg").getBoundingClientRect();
    const px = event.clientX - box.left;
    return Math.max(0, Math.min(maxStep, Math.round(((px - m.left) / (width - m.left - m.right)) * maxStep)));
  };
  hit.addEventListener("mousemove", (event) => {
    const step = stepAt(event);
    if (step !== state.hoverStep) { state.hoverStep = step; renderChart(); }
    showTooltip(event, tooltipFor(step, entries, focus));
  });
  hit.addEventListener("mouseleave", () => { state.hoverStep = null; hideTooltip(); renderChart(); });
  hit.addEventListener("click", (event) => {
    state.openStep = stepAt(event);
    renderLog();
    document.querySelector(`#logbook tr[data-step="${state.openStep}"]`)?.scrollIntoView({ block: "nearest" });
  });

  const run = focus.run;
  $("chart-meta").textContent = `${run.body_style} · budget ${run.iterations ?? "?"} evaluations`;
}

function tooltipFor(step, entries, focus) {
  const lines = entries.map(({ run, slot }) => {
    const records = summarise(run).records;
    const upTo = records.filter((r) => r.iteration <= step && r.best_cd_so_far !== null);
    const best = upTo.length ? upTo[upTo.length - 1].best_cd_so_far : null;
    return `<span class="tip-row"><i class="swatch-line" style="background: var(--series-${slot})"></i>`
      + `${escapeHtml(labelOf(run))}: ${best === null ? "–" : fmt(best, 4)}</span>`;
  });
  const r = summarise(focus.run).records.find((rec) => rec.iteration === step);
  const own = r ? `<br>${escapeHtml(labelOf(focus.run))} tried Cd ${fmt(r.cd, 4)}, ${verdictOf(r).text.toLowerCase()}` : "";
  return `<strong>Evaluation ${step}</strong>Best trusted so far<br>${lines.join("<br>")}${own}`;
}

function verdictOf(r) {
  if (r.accepted) return { cls: "ok", text: "Trusted" };
  if (isReference(r)) return { cls: "ref", text: "Reference" };
  return { cls: "no", text: "Gated" };
}

function renderCompare() {
  $("compare-body").replaceChildren(...selectedRuns().map(({ run, slot }) => {
    const s = summarise(run);
    const change = s.best !== null && s.start ? (100 * (s.best - s.start)) / s.start : null;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><span class="tag"><i class="swatch-line" style="background: var(--series-${slot})"></i>${escapeHtml(labelOf(run))}</span></td>
      <td class="num">${s.records.length}/${s.budget}</td>
      <td class="num">${s.best === null ? "–" : fmt(s.best, 4)}</td>
      <td class="num">${change === null ? "–" : `${signed(change, 1)}%`}</td>
      <td class="num">${s.bestAt === null ? "–" : `#${s.bestAt}`}</td>
      <td class="num">${s.gated}</td>`;
    tr.addEventListener("click", () => { state.focus = run.run_id; state.openStep = null; render(); });
    return tr;
  }));
}

function renderLog() {
  const run = runOf(state.focus);
  const body = $("logbook");
  if (!run) {
    body.replaceChildren();
    $("log-meta").textContent = "";
    for (const id of ["t-best", "t-gated", "t-time"]) $(id).textContent = "–";
    return;
  }
  const s = summarise(run);
  $("log-meta").textContent = `${labelOf(run)} · ${run.run_id} · ${STATE_TEXT[run.state] || run.state}${run.stop_reason ? `, ${run.stop_reason}` : ""}`;
  $("t-best").textContent = s.best === null ? "–" : fmt(s.best, 4);
  $("t-best-note").textContent = s.best !== null && s.start
    ? `${signed((100 * (s.best - s.start)) / s.start, 1)}% vs the unmodified car, at step ${s.bestAt}` : "";
  $("t-gated").textContent = `${s.gated} of ${s.records.length}`;
  $("t-gated-note").textContent = s.gated ? "each one told why, and none counted" : "every design was trusted";
  const timing = run.timing;
  $("t-time").textContent = timing?.step_s_median != null ? `${fmt(timing.step_s_median, 2)} s` : "–";
  $("t-time-note").textContent = timing?.surrogate_latency_ms_median != null
    ? `surrogate ${fmt(timing.surrogate_latency_ms_median, 1)} ms of it`
      + (run.cost_usd != null ? ` · $${fmt(run.cost_usd, 3)} spent` : "")
    : "";

  body.replaceChildren(...s.records.map((r) => {
    const verdict = verdictOf(r);
    const note = [r.reasoning, r.reject_reason && !r.accepted ? `Gate: ${r.reject_reason}` : ""].filter(Boolean).join(" ");
    const tr = document.createElement("tr");
    tr.dataset.step = String(r.iteration);
    tr.dataset.open = String(state.openStep === r.iteration);
    tr.dataset.best = String(r.accepted && r.iteration === s.bestAt);
    tr.innerHTML = `<td class="num">${r.iteration}</td><td class="num">${fmt(r.cd, 4)}</td>
      <td><span class="tag ${verdict.cls}"><i></i>${verdict.text}${tr.dataset.best === "true" ? " · best" : ""}</span></td>
      <td class="note"><div>${note ? escapeHtml(note) : `<span class="muted">${kindOf(run) === "agent" ? "–" : "no reasoning: an optimiser, not an agent"}</span>`}</div></td>
      <td><a href="${pageUrl("index.html", { run: run.run_id, step: r.iteration })}">Open</a></td>`;
    tr.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      state.openStep = state.openStep === r.iteration ? null : r.iteration;
      renderLog();
    });
    return tr;
  }));
}

// ---------------------------------------------------------------- live

function schedulePoll() {
  clearTimeout(state.timer);
  const live = selectedRuns().some(({ run }) => LIVE.has(run.state));
  if (live) state.timer = setTimeout(poll, POLL_MS);
}

async function poll() {
  try {
    await Promise.all(selectedRuns().filter(({ run }) => LIVE.has(run.state)).map(async ({ run }) => {
      const fresh = await api(`/runs/${encodeURIComponent(run.run_id)}`);
      Object.assign(run, fresh);
      await loadRecords(run, true);
    }));
    showError(null);
  } catch (error) {
    showError(error);
  }
  render();
}

// ---------------------------------------------------------------- helpers

function niceTicks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * power).find((s) => s >= raw);
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(Number(t.toFixed(10)));
  return ticks;
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

// ---------------------------------------------------------------- start

async function start() {
  $("designer-link").href = pageUrl("index.html");
  $("results-link").href = pageUrl("results.html");
  try {
    state.health = await api("/health");
    renderService();
    await loadRuns();
  } catch (error) {
    renderService();
    showError(error);
    return;
  }
  defaultSelection(new URLSearchParams(location.search).get("run"));
  render();
  $("refresh").addEventListener("click", async () => {
    try {
      await loadRuns();
      for (const id of [...state.selected.keys()]) if (!runOf(id)) select(id, false);
      showError(null);
    } catch (error) {
      showError(error);
    }
    render();
  });
  let resize = null;
  window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(renderChart, 100); });
}

start();
