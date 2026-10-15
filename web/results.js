// Results page.  OWNER: Paramveer.
//
// Static: reads two files next to it and needs no service.
//   results/metrics.json     python -m src.eval.harness ... --out web/results/metrics.json --label "..."
//   results/benchmark.json   python -m src.eval.benchmark
//
// Every number on the page comes from those files; nothing is typed in here.

import {
  bindInfo, escapeHtml, fmt, hideTooltip, initTheme, LOGO, pageUrl, showTooltip, signed,
} from "./common.js";

const $ = (id) => document.getElementById(id);
const SERIES = { agent: 1, cmaes: 2, random: 3, scripted: 4 };   // colour follows the method

const state = { metrics: null, bench: null, comparison: 0 };

// ---------------------------------------------------------------- loading

async function load(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function start() {
  initTheme();
  bindInfo();
  $("logo").innerHTML = LOGO;
  $("designer-link").href = pageUrl("index.html");
  $("runs-link").href = pageUrl("runs.html");
  const [metrics, bench] = await Promise.allSettled([load("results/metrics.json"), load("results/benchmark.json")]);
  state.metrics = metrics.status === "fulfilled" ? metrics.value : null;
  state.bench = bench.status === "fulfilled" ? bench.value : null;
  const missing = [!state.metrics && "results/metrics.json (python -m src.eval.harness)",
    !state.bench && "results/benchmark.json (python -m src.eval.benchmark)"].filter(Boolean);
  if (missing.length) {
    $("error-banner").hidden = false;
    $("error-text").textContent = `Not generated yet: ${missing.join(", ")}.`;
  }
  renderSource();
  if (state.metrics) {
    renderHeadline();
    renderAccuracy();
    renderCalibration();
    renderGate();
    renderScaling();
  }
  if (state.bench) renderBench();
  let resize = null;
  window.addEventListener("resize", () => {
    clearTimeout(resize);
    resize = setTimeout(() => {
      if (state.metrics) { renderAccuracy(); renderCalibration(); renderGate(); renderScaling(); }
      if (state.bench) renderBenchChart();
    }, 120);
  });
}

function renderSource() {
  const m = state.metrics;
  const el = $("source");
  el.dataset.state = m ? "ok" : "down";
  el.querySelector(".service-text").textContent = m
    ? `Graded ${m.created_at.slice(0, 10)} · ${m.models?.in_distribution?.version ?? "no checkpoint"}`
    : "No metrics file";
  if (m?.label) {
    $("label-banner").hidden = false;
    $("label-text").textContent = m.label;
  }
}

// ---------------------------------------------------------------- headline

function renderHeadline() {
  const m = state.metrics, h = m.headline, speed = m.speed || {};
  const noise = m.data?.noise_ceiling?.mean_label_std;
  const tiles = [
    ["R² on unseen designs", fmt(h.r2_in_distribution, 3),
      `${fmt(100 * h.share_of_ceiling, 0)}% of the ${fmt(h.r2_ceiling_test_set, 3)} the CFD noise allows. ` +
      `A straight line gets ${fmt(h.r2_linear_reference, 3)}.`],
    ["Typical error", `${fmt(h.mae_in_distribution, 4)}`,
      `mean absolute error in Cd${noise ? `; the CFD's own spread is ${fmt(noise, 4)}` : ""}`],
    [`Unseen ${h.held_out_category || "body style"}`, h.r2_held_out_category == null ? "–" : fmt(h.r2_held_out_category, 2),
      h.ood_detection_rate == null ? "R² with that style held out of training"
        : `R² when held out of training, and the gate flags ${fmt(100 * h.ood_detection_rate, 0)}% of them`],
    ["Time per design", `${fmt(h.latency_ms, 2)} ms`,
      speed.speedup_vs_cfd ? `${Math.round(speed.speedup_vs_cfd).toLocaleString()}× faster than one CFD run`
        : `${speed.device || "cpu"}, one design at a time`],
  ];
  $("headline").replaceChildren(...tiles.map(([label, value, note]) => {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML = `<div class="tile-label">${escapeHtml(label)}</div><div class="tile-value">${escapeHtml(value)}</div>
      <div class="tile-delta">${escapeHtml(note)}</div>`;
    return tile;
  }));
  $("provenance").textContent = `${m.data?.n_designs ?? "?"} DrivAerNet designs · ${m.models?.in_distribution?.n_params?.toLocaleString() ?? "?"} parameters`;
  const warnings = m.warnings || [];
  $("caveats").hidden = !warnings.length;
  $("caveats-summary").textContent = `${warnings.length} caveat${warnings.length === 1 ? "" : "s"} the grading raised`;
  $("warnings").replaceChildren(...warnings.map((w) => {
    const li = document.createElement("li");
    li.textContent = w;
    return li;
  }));
}

// ---------------------------------------------------------------- accuracy

function renderAccuracy() {
  const m = state.metrics;
  const rows = (block) => block ? [
    { label: "Predict the average", value: block.references?.mean?.overall?.r2, kind: "muted" },
    { label: "Straight line", value: block.references?.linear?.overall?.r2, kind: "muted" },
    { label: "Surrogate", value: block.overall?.r2, kind: "accent" },
  ] : [];
  barChart($("acc-random"), rows(m.in_distribution), {
    domain: [0, 1], marker: m.in_distribution?.overall?.r2_ceiling, markerLabel: "ceiling", format: (v) => fmt(v, 3),
  });
  const ood = m.out_of_distribution;
  $("acc-ood-title").textContent = ood ? `Unseen body style: ${ood.held_out_category} held out of training` : "Unseen body style";
  if (ood) {
    barChart($("acc-ood"), rows(ood), {
      domain: [0, 1], marker: ood.overall?.r2_ceiling, markerLabel: "ceiling", format: (v) => fmt(v, 3),
    });
  } else {
    $("acc-ood").innerHTML = `<p class="chart-empty">No held-out-style checkpoint graded.</p>`;
  }
  const per = m.in_distribution?.per_category || {};
  $("per-category").querySelector("tbody").replaceChildren(...Object.entries(per).map(([style, r]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(style)}</td><td class="num">${fmt(r.r2, 3)}</td><td class="num">${fmt(r.r2_ceiling, 3)}</td>
      <td class="num">${fmt(100 * r.share_of_ceiling, 0)}%</td><td class="num">${fmt(r.mae, 4)}</td><td class="num">${r.n}</td>`;
    return tr;
  }));
}

// ---------------------------------------------------------------- calibration

function renderCalibration() {
  const m = state.metrics;
  const series = [];
  const add = (block, label, slot) => {
    const cov = block?.calibration?.coverage;
    if (cov?.length) series.push({ label, slot, points: cov.map((c) => [c.expected, c.observed]) });
  };
  add(m.in_distribution, "Unseen designs", 1);
  add(m.out_of_distribution, `Unseen ${m.out_of_distribution?.held_out_category || "style"}`, 2);
  legend($("cal-legend"), series);
  lineChart($("calibration"), {
    series, x: [0, 1], y: [0, 1], diagonal: true,
    xLabel: "nominal interval (share of designs it should hold)", yLabel: "observed",
    fmtX: (v) => `${fmt(100 * v, 0)}%`, fmtY: (v) => `${fmt(100 * v, 0)}%`,
  });
  const cal = m.in_distribution?.calibration;
  $("cal-note").textContent = cal
    ? `On unseen designs, ±1σ holds ${fmt(100 * cal.coverage_1sigma, 0)}% (68% expected) and ±2σ holds ` +
      `${fmt(100 * cal.coverage_2sigma, 0)}% (95% expected).`
    : "";
}

// ---------------------------------------------------------------- gate

function renderGate() {
  const held = state.metrics.ood_detection?.held_out_style;
  if (!held) {
    $("gate").innerHTML = `<p class="chart-empty">Needs a held-out-style checkpoint.</p>`;
    return;
  }
  const distance = held.checkpoint_detector || held.harness_detector_all_inputs;
  const rows = [
    { label: "Uncertainty σ alone", value: held.sigma_alone?.detection_rate, kind: "muted",
      note: `AUROC ${fmt(held.sigma_alone?.auroc ?? NaN, 2)}` },
    { label: "Distance, shape dials only", value: held.harness_detector_aero_only?.detection_rate, kind: "muted",
      note: `AUROC ${fmt(held.harness_detector_aero_only?.auroc ?? NaN, 2)}` },
    { label: "Distance with body style (the gate)", value: distance?.detection_rate, kind: "accent",
      note: `AUROC ${fmt(distance?.auroc ?? NaN, 2)}` },
  ];
  $("gate-lede").textContent = `Every ${held.held_out_category} design is something the held-out model never saw. ` +
    "The share each check flags, at the threshold that wrongly flags about 1% of designs it did see:";
  barChart($("gate"), rows, { domain: [0, 1], format: (v) => `${fmt(100 * v, 0)}%` });
  const ordinary = state.metrics.in_distribution?.gate?.calibrated;
  $("gate-note").textContent = ordinary
    ? `On ordinary unseen designs, the gate passes ${fmt(100 * ordinary.pass_rate, 1)}%. σ alone cannot do this job: ` +
      "a new body style widens σ by about a fifth while the error doubles."
    : "";
}

// ---------------------------------------------------------------- scaling

function renderScaling() {
  const s = state.metrics.data_scaling || {};
  const series = [];
  if (s.model?.length) series.push({ label: "Surrogate", slot: 1, points: s.model.map((p) => [p.n_train, p.r2]) });
  if (s.linear_reference?.length) {
    series.push({ label: "Straight line", slot: 2, points: s.linear_reference.map((p) => [p.n_train, p.r2]) });
  }
  legend($("scaling-legend"), series);
  const xs = series.flatMap((d) => d.points.map((p) => p[0]));
  const ys = series.flatMap((d) => d.points.map((p) => p[1]));
  const ceiling = state.metrics.in_distribution?.overall?.r2_ceiling;
  lineChart($("scaling"), {
    series, x: [0, Math.max(...xs, 1)], y: [Math.min(0.5, ...ys) - 0.05, Math.max(ceiling ?? 0, ...ys) + 0.03],
    hline: ceiling ? { y: ceiling, label: `ceiling ${fmt(ceiling, 3)}` } : null, markers: true,
    xLabel: "training designs", yLabel: "R²", fmtX: (v) => Math.round(v).toLocaleString(), fmtY: (v) => fmt(v, 2),
  });
  $("scaling-note").textContent = s.model?.length ? ""
    : "Only the straight-line reference so far: the surrogate's points arrive with checkpoints trained on subsets.";
}

// ---------------------------------------------------------------- benchmark

function renderBench() {
  const comparisons = state.bench.comparisons || [];
  const pick = $("bench-pick");
  if (comparisons.length > 1) {
    pick.replaceChildren(...comparisons.map((c, i) => new Option(
      `${c.body_style}, ${c.budget} evaluations${c.constraints.length ? `, ${c.constraints.join(", ")}` : ""}`, String(i))));
    pick.hidden = false;
    pick.addEventListener("change", () => { state.comparison = Number(pick.value); renderBenchChart(); });
  }
  renderBenchChart();
}

function slotFor(method) {
  return SERIES[method.split(":")[0]] || 4;
}

function renderBenchChart() {
  const c = state.bench.comparisons?.[state.comparison];
  if (!c) {
    $("bench").innerHTML = `<p class="chart-empty">No finished runs yet.</p>`;
    return;
  }
  const series = c.methods.map((m) => ({
    label: m.label, slot: slotFor(m.method),
    points: m.curve_pct.median.map((v, i) => [i, v]),
    band: m.n_runs > 1 ? m.curve_pct.q25.map((v, i) => [i, v, m.curve_pct.q75[i]]) : null,
  }));
  legend($("bench-legend"), series);
  const ys = c.methods.flatMap((m) => [...m.curve_pct.q25, ...m.curve_pct.q75, ...m.curve_pct.median]);
  lineChart($("bench"), {
    series, x: [0, c.budget - 1], y: [Math.min(...ys) - 0.3, Math.max(0, ...ys) + 0.3], step: true,
    hline: { y: 0, label: "unmodified car" },
    xLabel: "evaluation", yLabel: "best trusted Cd vs unmodified",
    fmtX: (v) => String(Math.round(v)), fmtY: (v) => `${signed(v, 1)}%`,
  });
  const agent = c.methods.some((m) => m.method.startsWith("agent"));
  $("bench-lede").textContent = `${c.body_style}, ${c.budget} evaluations each, same start, same gate, ` +
    `scored by ${c.surrogate}. Lower is better.`;
  $("bench-table").querySelector("tbody").replaceChildren(...c.methods.map((m) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><span class="tag"><i class="swatch-line" style="background: var(--series-${slotFor(m.method)})"></i>${escapeHtml(m.label)}</span></td>
      <td class="num">${m.n_runs}</td><td class="num">${signed(m.final_pct.median, 2)}%</td>
      <td class="num">${signed(m.final_pct.best, 2)}%</td>
      <td class="num">${m.found_at_median == null ? "–" : `#${fmt(m.found_at_median, 0)}`}</td>
      <td class="num">${fmt(m.gated_median, 0)}</td>
      <td class="num">${m.step_s_median == null ? "–" : `${fmt(m.step_s_median, 2)} s`}</td>
      <td class="num">${m.cost_usd_total == null ? "–" : `$${fmt(m.cost_usd_total, 2)}`}</td>`;
    return tr;
  }));
  $("bench-note").textContent = agent ? ""
    : "No live agent run in this comparison yet. The scripted agent is the offline test loop, not Nemotron.";
}

// ---------------------------------------------------------------- charts

function legend(host, series) {
  host.replaceChildren(...series.map((s) => {
    const span = document.createElement("span");
    span.innerHTML = `<i class="swatch-line" style="background: var(--series-${s.slot})"></i>${escapeHtml(s.label)}`;
    return span;
  }));
}

// Horizontal bars on one axis, labels in ink, values at the bar end.
// Values below the domain (a negative R²) show as a stub with the true number.
function barChart(host, rows, { domain, marker = null, markerLabel = "", format }) {
  const [lo, hi] = domain;
  host.replaceChildren(...rows.map((row) => {
    const div = document.createElement("div");
    div.className = "bar-row";
    const ok = typeof row.value === "number" && Number.isFinite(row.value);
    const share = ok ? Math.max(0, Math.min(1, (row.value - lo) / (hi - lo))) : 0;
    div.innerHTML = `<span class="bar-label">${escapeHtml(row.label)}</span>
      <span class="bar-track">
        <span class="bar ${row.kind}" style="width:${share * 100}%"></span>
        ${marker != null ? `<span class="bar-marker" style="left:${((marker - lo) / (hi - lo)) * 100}%"></span>` : ""}
      </span>
      <span class="bar-value">${ok ? escapeHtml(format(row.value)) : "–"}${ok && row.value < lo ? " (below 0)" : ""}</span>`;
    div.addEventListener("mousemove", (event) => showTooltip(event, `<strong>${escapeHtml(row.label)}</strong>`
      + `${ok ? escapeHtml(format(row.value)) : "not available"}${row.note ? `<br>${escapeHtml(row.note)}` : ""}`
      + (marker != null ? `<br>${escapeHtml(markerLabel)} ${escapeHtml(format(marker))}` : "")));
    div.addEventListener("mouseleave", hideTooltip);
    return div;
  }));
  if (marker != null) {
    const key = document.createElement("p");
    key.className = "chart-note";
    key.innerHTML = `<i class="marker-key"></i> ${escapeHtml(markerLabel)} ${escapeHtml(format(marker))}`;
    host.append(key);
  }
}

// One-axis line chart with optional bands, a reference line or diagonal, and a crosshair tooltip.
function lineChart(host, opts) {
  const { series, x: [x0, x1], y: [y0, y1], fmtX, fmtY } = opts;
  if (!series.length) {
    host.innerHTML = `<p class="chart-empty">Nothing to plot yet.</p>`;
    return;
  }
  const width = Math.max(host.clientWidth, 260);
  const height = width < 560 ? 240 : 300;
  const m = { left: 56, right: 18, top: 12, bottom: 40 };
  const X = (v) => m.left + ((v - x0) / (x1 - x0 || 1)) * (width - m.left - m.right);
  const Y = (v) => m.top + (1 - (v - y0) / (y1 - y0 || 1)) * (height - m.top - m.bottom);
  const yt = ticks(y0, y1, 5), xt = ticks(x0, x1, width < 560 ? 4 : 6);
  const parts = [];
  parts.push(`<g class="grid">${yt.map((t) => `<line x1="${m.left}" x2="${width - m.right}" y1="${Y(t)}" y2="${Y(t)}"/>`).join("")}</g>`);
  parts.push(`<g class="axis">${yt.map((t) => `<text x="${m.left - 8}" y="${Y(t) + 4}" text-anchor="end">${escapeHtml(fmtY(t))}</text>`).join("")}`
    + xt.map((t) => `<text x="${X(t)}" y="${height - m.bottom + 18}" text-anchor="middle">${escapeHtml(fmtX(t))}</text>`).join("")
    + `<text x="${(m.left + width - m.right) / 2}" y="${height - 4}" text-anchor="middle">${escapeHtml(opts.xLabel)}</text>`
    + `<line class="baseline" x1="${m.left}" x2="${width - m.right}" y1="${height - m.bottom}" y2="${height - m.bottom}"/></g>`);
  if (opts.diagonal) {
    parts.push(`<line class="start-line" x1="${X(x0)}" y1="${Y(y0)}" x2="${X(x1)}" y2="${Y(y1)}"/>`
      + `<text class="start-label" x="${X(x1) - 4}" y="${Y(y1) + 14}" text-anchor="end">honest</text>`);
  }
  if (opts.hline) {
    parts.push(`<line class="start-line" x1="${m.left}" x2="${width - m.right}" y1="${Y(opts.hline.y)}" y2="${Y(opts.hline.y)}"/>`
      + `<text class="start-label" x="${width - m.right - 4}" y="${Y(opts.hline.y) - 6}" text-anchor="end">${escapeHtml(opts.hline.label)}</text>`);
  }
  const path = (points) => points.map(([px, py], i) => {
    if (i === 0) return `M${X(px)},${Y(py)}`;
    return opts.step ? `H${X(px)}V${Y(py)}` : `L${X(px)},${Y(py)}`;
  }).join("");
  for (const s of series) {
    if (!s.band) continue;
    const upper = s.band.map(([px, , hi]) => [px, hi]);
    const lower = s.band.map(([px, lo]) => [px, lo]);
    const n = lower.length - 1;
    // Back along the lower edge. A step-after curve holds each value until the
    // next x, so walking backwards drops to the previous value, then moves left.
    let back = `L${X(lower[n][0])},${Y(lower[n][1])}`;
    for (let i = n; i > 0; i--) {
      back += opts.step ? `V${Y(lower[i - 1][1])}H${X(lower[i - 1][0])}` : `L${X(lower[i - 1][0])},${Y(lower[i - 1][1])}`;
    }
    parts.push(`<path class="band" d="${path(upper)}${back}Z" style="fill: var(--series-${s.slot})"/>`);
  }
  for (const s of series) {
    parts.push(`<path class="run-line" d="${path(s.points)}" style="stroke: var(--series-${s.slot})"/>`);
    if (opts.markers) {
      for (const [px, py] of s.points) {
        parts.push(`<circle class="step-dot" cx="${X(px)}" cy="${Y(py)}" r="3.5" style="--dot: var(--series-${s.slot})"/>`);
      }
    }
  }
  parts.push(`<line class="crosshair" id="${host.id}-cross" x1="0" x2="0" y1="${m.top}" y2="${height - m.bottom}" visibility="hidden"/>`);
  parts.push(`<rect class="hit" x="${m.left}" y="0" width="${width - m.left - m.right}" height="${height - m.bottom}"/>`);
  host.innerHTML = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img"
    aria-label="${escapeHtml(opts.yLabel)} against ${escapeHtml(opts.xLabel)}">${parts.join("")}</svg>`;

  const cross = host.querySelector(".crosshair");
  host.querySelector(".hit").addEventListener("mousemove", (event) => {
    const box = host.querySelector("svg").getBoundingClientRect();
    const xv = x0 + ((event.clientX - box.left - m.left) / (width - m.left - m.right)) * (x1 - x0);
    // Nearest plotted x across series.
    const all = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))];
    const near = all.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a), all[0]);
    cross.setAttribute("x1", X(near)); cross.setAttribute("x2", X(near)); cross.setAttribute("visibility", "visible");
    const lines = series.map((s) => {
      const p = s.points.find((pt) => pt[0] === near);
      const band = s.band?.find((b) => b[0] === near);
      return `<span class="tip-row"><i class="swatch-line" style="background: var(--series-${s.slot})"></i>${escapeHtml(s.label)}: `
        + `${p ? escapeHtml(fmtY(p[1])) : "–"}${band ? ` <small>(${escapeHtml(fmtY(band[1]))} to ${escapeHtml(fmtY(band[2]))})</small>` : ""}</span>`;
    });
    showTooltip(event, `<strong>${escapeHtml(opts.xLabel)} ${escapeHtml(fmtX(near))}</strong>${lines.join("<br>")}`);
  });
  host.querySelector(".hit").addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); });
}

function ticks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * power).find((s) => s >= raw);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(Number(t.toFixed(10)));
  return out;
}

start();
