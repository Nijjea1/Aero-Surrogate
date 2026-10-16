// The designer's instrument cluster.  OWNER: Paramveer.
//
//   const cd = createCdGauge(svg)
//   cd.update({ cd, baseline })            // needle on the design, a mark on the original
//
//   const meters = createMeters(host, [{ id, label, unit, info }])
//   meters.set(id, { value, max, min = 0, gate, ref, text, sub, good, off })
//
// Every number sits on a scale that shows where it falls: a needle for Cd on
// an arc that brackets the original car, a pointer on a ruled scale for the
// rest, with the gate as a red line where there is one and the original car
// as a silver mark. Scales re-rule themselves when their range changes.

import { escapeHtml, fmt, infoButton } from "./common.js";

const NS = "http://www.w3.org/2000/svg";

// ---------------------------------------------------------------- Cd arc

const G = { cx: 128, cy: 124, r: 100, sweep: 90 };   // a half circle: the end labels sit level with its base

export function createCdGauge(svg) {
  let range = null;
  const angleOf = (v) => -G.sweep + ((Math.min(Math.max(v, range[0]), range[1]) - range[0]) / (range[1] - range[0])) * 2 * G.sweep;
  const polar = (deg, r) => {
    const a = ((deg - 90) * Math.PI) / 180;
    return [G.cx + r * Math.cos(a), G.cy + r * Math.sin(a)];
  };
  const arc = (r, a0, a1) => {
    const [x0, y0] = polar(a0, r), [x1, y1] = polar(a1, r);
    return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
  };

  // A 0.12-wide window on whole hundredths, centred on the original car, so a
  // change of a few thousandths still moves the needle visibly.
  function rule(baseline) {
    const lo = Math.floor((baseline - 0.06) * 100) / 100;
    range = [lo, lo + 0.12];
    let s = `<path d="${arc(G.r, -G.sweep, G.sweep)}" fill="none" stroke="var(--border-strong)" stroke-width="2"/>`;
    s += '<path data-arc fill="none" stroke="var(--accent)" stroke-width="3"/>';
    for (let k = 0; k <= 24; k++) {
      const v = lo + k * 0.005, a = angleOf(v), major = k % 4 === 0;
      const [x0, y0] = polar(a, G.r - (major ? 12 : k % 2 === 0 ? 8 : 5)), [x1, y1] = polar(a, G.r - 2);
      s += `<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}" stroke="var(--text-muted)" stroke-width="${major ? 1.4 : 1}"/>`;
      if (major) {
        // Labels outside the arc; the two at its ends sit under them, clear of the reading.
        const end = Math.abs(a) > 80;
        const [lx, ly] = end ? [polar(a, G.r - 6)[0], G.cy + 17] : polar(a, G.r + 13);
        s += `<text x="${lx.toFixed(1)}" y="${(end ? ly : ly + 4).toFixed(1)}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--text-muted)">${fmt(v, 2)}</text>`;
      }
    }
    const [ox, oy] = polar(angleOf(baseline), G.r - 17), [ox2, oy2] = polar(angleOf(baseline), G.r + 3);
    s += `<line x1="${ox.toFixed(1)}" y1="${oy.toFixed(1)}" x2="${ox2.toFixed(1)}" y2="${oy2.toFixed(1)}" stroke="var(--silver)" stroke-width="3"><title>The original car, Cd ${fmt(baseline, 4)}</title></line>`;
    // A pointer that rides the arc, clear of the reading in the middle.
    s += `<g class="needle" data-needle><path d="M${G.cx} ${G.cy - G.r + 4} L${G.cx - 5} ${G.cy - G.r + 30} L${G.cx + 5} ${G.cy - G.r + 30} Z" fill="var(--accent)"/></g>`;
    svg.innerHTML = s;
    svg.dataset.baseline = String(baseline);
  }

  return {
    update({ cd, baseline }) {
      const ref = Number.isFinite(baseline) ? baseline : cd;
      if (!range || svg.dataset.baseline !== String(ref)) rule(ref);
      svg.querySelector("[data-needle]").style.transform = `rotate(${angleOf(cd)}deg)`;
      const a0 = angleOf(ref), a1 = angleOf(cd);
      svg.querySelector("[data-arc]").setAttribute("d", Math.abs(a1 - a0) < 0.2 ? "" : arc(G.r, Math.min(a0, a1), Math.max(a0, a1)));
    },
  };
}

// ---------------------------------------------------------------- ruled scales

// Steps of 1, 2, 2.5 and 5 per decade, so labels land on round numbers.
export function niceStep(span, count = 5) {
  const raw = span / count;
  const base = 10 ** Math.floor(Math.log10(raw));
  for (const k of [1, 2, 2.5, 5, 10]) if (k * base >= raw) return k * base;
  return 10 * base;
}

export function niceCeil(value) {
  if (!(value > 0)) return 1;
  const step = niceStep(value, 5);
  return Math.ceil(value / step - 1e-9) * step;
}

// Decimals that show a step exactly: 2.5 needs one, 0.005 needs three.
function decimalsOf(step) {
  for (let d = 0; d < 6; d++) if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d;
  return 6;
}

export function createMeters(host, defs) {
  const rows = {};
  host.replaceChildren(...defs.map((def) => {
    const el = document.createElement("div");
    el.className = "m";
    el.dataset.m = def.id;
    el.innerHTML = `<div class="m-head"><span class="k">${escapeHtml(def.label)}${def.info ? ` ${infoButton(def.info)}` : ""}</span>
      <span class="v"><span data-v>–</span>${def.unit ? `<small>${escapeHtml(def.unit)}</small>` : ""}</span></div>
      <div class="scale" data-scale></div><div class="s" data-s></div>`;
    rows[def.id] = { el, key: "" };
    return el;
  }));

  function ruleScale(row, min, max, gate) {
    const key = `${min}|${max}|${gate}`;
    if (row.key === key) return;
    row.key = key;
    const at = (v) => ((v - min) / (max - min)) * 100;
    const major = niceStep(max - min, 4), minor = major / 5;
    let html = '<div class="rail"></div><div class="fill" data-fill></div>';
    const first = Math.ceil(min / minor - 1e-9) * minor;
    for (let v = first; v <= max + 1e-9; v += minor) {
      const isMajor = Math.abs(v / major - Math.round(v / major)) < 1e-6;
      html += `<span class="tick${isMajor ? " major" : ""}" style="left:${at(v)}%"></span>`;
    }
    const decimals = decimalsOf(major);
    const labels = [];
    for (let v = Math.ceil(min / major - 1e-9) * major; v <= max + 1e-9; v += major) labels.push(v);
    labels.forEach((v, i) => {
      const edge = i === 0 && at(v) < 4 ? " first" : i === labels.length - 1 && at(v) > 96 ? " last" : "";
      html += `<span class="lbl${edge}" style="left:${at(v)}%">${fmt(v, decimals)}</span>`;
    });
    if (Number.isFinite(gate)) html += `<span class="gate" style="left:${at(gate)}%" title="The gate"></span>`;
    html += '<span class="ref" data-ref hidden></span><span class="ptr" data-ptr></span>';
    row.el.querySelector("[data-scale]").innerHTML = html;
  }

  return {
    set(id, { value, min = 0, max, gate, ref, text, sub = "", good, off = false, fill = false }) {
      const row = rows[id];
      if (!row) return;
      row.el.dataset.off = String(off);
      const top = Number.isFinite(max) && max > min ? max : niceCeil(Math.max(Math.abs(value) || 0, 1e-6));
      ruleScale(row, min, top, gate);
      const clampPct = (v) => `${Math.min(100, Math.max(0, ((v - min) / (top - min)) * 100))}%`;
      row.el.querySelector("[data-v]").textContent = text ?? (Number.isFinite(value) ? String(value) : "–");
      const ptr = row.el.querySelector("[data-ptr]");
      ptr.hidden = off || !Number.isFinite(value);
      if (!ptr.hidden) ptr.style.left = clampPct(value);
      const bar = row.el.querySelector("[data-fill]");
      bar.style.width = fill && Number.isFinite(value) && !off ? clampPct(value) : "0%";
      bar.dataset.over = String(Number.isFinite(gate) && value > gate);
      const refEl = row.el.querySelector("[data-ref]");
      refEl.hidden = !Number.isFinite(ref) || off;
      if (!refEl.hidden) { refEl.style.left = clampPct(ref); refEl.title = "The original car"; }
      const s = row.el.querySelector("[data-s]");
      s.textContent = sub;
      if (good === true || good === false) s.dataset.good = String(good); else s.removeAttribute("data-good");
    },
  };
}
