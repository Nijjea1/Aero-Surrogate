"""Build the README's graphics and numbers from the published results.  OWNER: Paramveer.

    python docs/readme/build.py

Writes, into docs/readme/:
  hero.svg       the banner: air streaming over a fastback, coloured by pressure
  pipeline.svg   how a design moves through the system, with the agent's loop
  stats.svg      the headline numbers as cards, from web/results/metrics.json
                 and web/results/benchmark.json
and rewrites the blocks between <!-- name:start --> and <!-- name:end --> in
README.md, so the prose never disagrees with the graded files. Run it again
whenever those files change (for example after the final checkpoints).

The SVGs are animated with CSS inside the file, which GitHub renders in an
<img>; prefers-reduced-motion stops every animation. Colours are the site's
own tokens (web/styles.css).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

BG, PANEL, LINE, INK, INK2, INK3 = "#0f1012", "#16171a", "#2a2c30", "#ececec", "#a3a4a9", "#727379"
ACCENT, COOL, WARM, GOOD, BAD = "#ff5f1f", "#5b8fe0", "#e8664f", "#5fc47c", "#f0605a"
FONT = "'IBM Plex Sans','Segoe UI',Helvetica,Arial,sans-serif"
MONO = "'IBM Plex Mono',ui-monospace,SFMono-Regular,Consolas,monospace"
REDUCED = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"

# The fastback side profile the site's wind-tunnel mockup uses, in a 1000 x 450 world.
CAR = ("M70 282 C55 270 52 242 70 232 L330 206 Q342 204 352 199 L470 138 Q500 128 560 128 "
       "C640 130 760 160 880 202 L905 216 C925 225 930 255 918 278 L905 290 L830 290 "
       "A62 62 0 0 0 706 290 L300 290 A62 62 0 0 0 176 290 L120 290 Q85 290 70 282 Z")
GLASS = "M368 202 L472 145 Q502 136 560 136 C622 138 702 158 792 190 Z"
TOP = [(95, 300), (110, 272), (370, 246), (392, 239), (510, 178), (600, 168), (700, 172), (800, 190),
       (920, 242), (945, 256), (958, 290), (962, 318)]
FLOOR = 380


# ---------------------------------------------------------------- streamlines

def _top(x: float) -> float:
    if x < TOP[0][0] or x > TOP[-1][0]:
        return FLOOR
    for (x0, y0), (x1, y1) in zip(TOP, TOP[1:]):
        if x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return FLOOR


def _height_field(lo: int, hi: int) -> dict[int, float]:
    """Body height above the floor, smoothed so the air starts turning before the nose."""
    raw = {}
    for x in range(lo, hi + 1):          # TOP is already in world units (the car is drawn 40 down)
        raw[x] = 92 * math.exp(-(x - 962) / 150) if x > 962 else max(0.0, FLOOR - _top(x))
    s, out = 20, {}
    for x in range(lo, hi + 1):
        acc = w = 0.0
        for k in range(-3 * s, 3 * s + 1, 3):
            wk = math.exp(-(k * k) / (2 * s * s))
            acc += raw.get(min(hi, max(lo, x + k)), 0.0) * wk
            w += wk
        out[x] = max(acc / w, raw[x] if x > 962 else 0.0)
    return out


def streamline_paths(lo=-900, hi=1150, heights=(20, 36, 64, 80, 98, 118, 140, 166, 196, 232, 274, 320, 370)):
    H = _height_field(lo, hi)
    paths = []
    for i, h0 in enumerate(heights):
        pts = []
        for x in range(lo, hi + 1, 10):
            Hx = H[x]
            h = h0 * (1 - 0.12 * (95 < x < 962)) if h0 < 50 else math.sqrt(h0 * h0 + Hx * Hx)
            if x > 940:   # the wake: a gentle, fixed wobble
                h += 9 * math.exp(-((h - 80) / 90) ** 2) * math.sin(0.045 * x + i) * min(1, (x - 940) / 80)
            pts.append((x, FLOOR - h))
        paths.append("M" + " L".join(f"{x:.0f} {y:.1f}" for x, y in pts))
    return paths


# ---------------------------------------------------------------- hero

def hero() -> str:
    W, Hh = 1280, 440
    s, tx, ty = 0.66, 590, 92          # car world -> banner
    lines = streamline_paths()
    defs = f"""
  <defs>
    <pattern id="g" width="22" height="22" patternUnits="userSpaceOnUse">
      <path d="M22 0H0V22" fill="none" stroke="rgba(120,160,210,0.07)" stroke-width="1"/>
    </pattern>
    <pattern id="G" width="110" height="110" patternUnits="userSpaceOnUse">
      <path d="M110 0H0V110" fill="none" stroke="rgba(120,160,210,0.13)" stroke-width="1"/>
    </pattern>
    <linearGradient id="cp" x1="52" x2="930" y1="0" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{WARM}"/><stop offset="0.08" stop-color="#b9a7a3"/>
      <stop offset="0.22" stop-color="#9aa6bd"/><stop offset="0.34" stop-color="#b7aaa8"/>
      <stop offset="0.46" stop-color="#7f9bd0"/><stop offset="0.6" stop-color="{COOL}"/>
      <stop offset="0.84" stop-color="#6e93d6"/><stop offset="1" stop-color="#a3a9b6"/>
    </linearGradient>
    <linearGradient id="air" x1="-900" x2="1150" y1="0" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{INK2}" stop-opacity="0.15"/>
      <stop offset="0.43" stop-color="{WARM}" stop-opacity="0.95"/>
      <stop offset="0.5" stop-color="#c9cacd" stop-opacity="0.8"/>
      <stop offset="0.66" stop-color="{COOL}" stop-opacity="1"/>
      <stop offset="0.9" stop-color="#8fb3ec" stop-opacity="0.7"/>
      <stop offset="1" stop-color="{INK2}" stop-opacity="0.2"/>
    </linearGradient>
    <linearGradient id="sheen" x1="0" x2="0" y1="120" y2="300" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#fff" stop-opacity="0.32"/><stop offset="0.45" stop-color="#fff" stop-opacity="0"/>
      <stop offset="1" stop-color="#000" stop-opacity="0.28"/>
    </linearGradient>
    <linearGradient id="fade" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0" stop-color="{BG}" stop-opacity="1"/><stop offset="0.42" stop-color="{BG}" stop-opacity="0.92"/>
      <stop offset="0.6" stop-color="{BG}" stop-opacity="0"/>
    </linearGradient>
  </defs>"""
    style = f"""
  <style>
    .flow {{ fill: none; stroke: url(#air); stroke-width: 2.4; stroke-linecap: round;
             stroke-dasharray: 44 122; animation: flow 3.2s linear infinite; }}
    .flow:nth-child(3n) {{ animation-duration: 2.6s; }}
    .flow:nth-child(3n+1) {{ animation-duration: 3.8s; }}
    .base {{ fill: none; stroke: rgba(200,215,235,0.10); stroke-width: 1.2; }}
    .wheel {{ animation: spin 1.1s linear infinite; transform-box: fill-box; transform-origin: center; }}
    .belt {{ animation: belt 0.9s linear infinite; }}
    .pulse {{ animation: pulse 2.4s ease-in-out infinite; }}
    @keyframes flow {{ to {{ stroke-dashoffset: -332; }} }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes belt {{ to {{ transform: translateX(40px); }} }}
    @keyframes pulse {{ 50% {{ opacity: 0.35; }} }}
    {REDUCED}
  </style>"""
    base = "\n".join(f'      <path class="base" d="{d}"/>' for d in lines)
    flow = "\n".join(f'      <path class="flow" d="{d}" style="animation-delay:-{(i * 0.37) % 3:.2f}s"/>'
                     for i, d in enumerate(lines))
    def wheel(cx):
        sp = "".join(f'<line x1="{cx + 8 * math.cos(a):.1f}" y1="{290 + 8 * math.sin(a):.1f}" '
                     f'x2="{cx + 28 * math.cos(a):.1f}" y2="{290 + 28 * math.sin(a):.1f}"/>'
                     for a in [k * 2 * math.pi / 5 for k in range(5)])
        return (f'<circle cx="{cx}" cy="290" r="50" fill="#0b0c0e"/><circle cx="{cx}" cy="290" r="31" fill="#3e4146"/>'
                f'<g class="wheel" stroke="#8e9197" stroke-width="4">{sp}<circle cx="{cx}" cy="290" r="31" fill="none" stroke="none"/></g>'
                f'<circle cx="{cx}" cy="290" r="6" fill="#b5b8be"/>')

    belt = "".join(f'<rect x="{x}" y="0" width="18" height="2" fill="rgba(255,255,255,0.08)"/>' for x in range(-40, 1500, 40))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" role="img"
     aria-label="Aero Surrogate: smoke streamlines flowing over a fastback car, with illustrative pressure colours">{defs}{style}
  <rect width="{W}" height="{Hh}" fill="{BG}"/>
  <rect width="{W}" height="{Hh}" fill="url(#g)"/>
  <rect width="{W}" height="{Hh}" fill="url(#G)"/>
  <g transform="translate({tx} {ty}) scale({s})">
    <g>
{base}
    </g>
    <g>
{flow}
    </g>
    <rect x="-900" y="{FLOOR}" width="2100" height="160" fill="#15171b"/>
    <g class="belt" transform="translate(0 {FLOOR + 12})">{belt}</g>
    <g transform="translate(40 40)">
      <path d="{CAR}" fill="url(#cp)"/>
      <path d="{CAR}" fill="url(#sheen)"/>
      <path d="{GLASS}" fill="#0b0d11" opacity="0.6"/>
      {wheel(238)}
      {wheel(768)}
    </g>
  </g>
  <g font-family="{MONO}" font-size="14">{pin(tx, ty, s, 560, 168, "Roof", "−0.56", COOL)}{pin(tx, ty, s, 102, 300, "Front", "+0.50", WARM)}</g>
  <rect width="{W}" height="{Hh}" fill="url(#fade)"/>
  <g font-family="{FONT}">
    <rect x="64" y="70" width="6" height="22" fill="{ACCENT}"/>
    <text x="82" y="87" fill="{INK2}" font-size="15" letter-spacing="2.4" font-family="{MONO}">NEBIUS × NVIDIA GLOBAL AI HACKATHON</text>
    <text x="62" y="170" fill="{INK}" font-size="76" font-weight="600" letter-spacing="-1.5">Aero Surrogate</text>
    <text x="64" y="222" fill="{INK2}" font-size="24">Explore parametric vehicle designs before CFD,</text>
    <text x="64" y="254" fill="{INK2}" font-size="24">with a domain gate and an inspectable agent logbook.</text>
    <g font-family="{MONO}" font-size="15">
      <circle class="pulse" cx="70" cy="321" r="5" fill="{GOOD}"/>
      <text x="84" y="326" fill="{INK3}">surrogate</text><text x="174" y="326" fill="{INK}">Cd + sigma</text>
      <text x="262" y="326" fill="{INK3}">CFD</text><text x="302" y="326" fill="{INK}">hours</text>
      <text x="64" y="356" fill="{INK3}">trust gate on every prediction · Nemotron 3 design agent</text>
    </g>
  </g>
  <rect y="{Hh - 3}" width="{W}" height="3" fill="{ACCENT}"/>
</svg>
"""


def pin(tx, ty, s, wx, wy, name, value, colour) -> str:
    """A flat label on the car like the designer's: a dot on the surface, a leader, a value."""
    x, y = tx + wx * s, ty + wy * s
    lx, ly = x + 10, y - 46
    return (f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{lx:.1f}" y2="{ly + 10:.1f}" stroke="{INK2}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{INK}"/>'
            f'<rect x="{lx - 6:.1f}" y="{ly - 12:.1f}" width="{len(name) * 8.6 + 64:.0f}" height="24" rx="2" fill="{PANEL}" stroke="#3d3f45"/>'
            f'<text x="{lx + 4:.1f}" y="{ly + 5:.1f}" fill="{INK2}">{name}</text>'
            f'<text x="{lx + len(name) * 8.6 + 10:.1f}" y="{ly + 5:.1f}" fill="{colour}">{value}</text>')


# ---------------------------------------------------------------- pipeline

def pipeline() -> str:
    W, H = 1280, 350
    nodes = [
        (40, "1", "Design", "16 shape dials and a body style", "src/geometry"),
        (350, "2", "Surrogate", "Cd, its σ and a distance score", "src/train · custom PyTorch"),
        (660, "3", "Trust gate", "σ and distance from training data", "src/service/gate.py"),
        (970, "4", "Design agent", "Nemotron 3 on Token Factory", "src/agent"),
    ]
    w, h, y = 270, 118, 128
    boxes = []
    for x, n, title, sub, path in nodes:
        boxes.append(f"""
    <g transform="translate({x} {y})">
      <rect width="{w}" height="{h}" rx="4" fill="{PANEL}" stroke="{LINE}"/>
      <rect width="4" height="{h}" fill="{ACCENT if n in '24' else LINE}"/>
      <text x="22" y="34" fill="{INK3}" font-family="{MONO}" font-size="13">{n}</text>
      <text x="44" y="34" fill="{INK}" font-size="21" font-weight="600">{title}</text>
      <text x="22" y="66" fill="{INK2}" font-size="15">{sub}</text>
      <text x="22" y="96" fill="{INK3}" font-family="{MONO}" font-size="12.5">{path}</text>
    </g>""")
    mid = y + h / 2
    arrows = "".join(f'<path d="M{x + w + 6} {mid} H{x2 - 8}" stroke="{INK3}" stroke-width="1.5" marker-end="url(#a)"/>'
                     for (x, *_), (x2, *_) in zip(nodes, nodes[1:]))
    loop = f"M{970 + w / 2} {y + h + 8} V{y + h + 70} H{40 + w / 2} V{y + h + 10}"
    reject = f"M{660 + w / 2} {y - 8} V{y - 44} H{970 + w / 2} V{y - 10}"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="Pipeline: design dials to surrogate to trust gate to Nemotron agent, which proposes the next design">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L10 5 L0 10 Z" fill="{INK3}"/>
    </marker>
    <marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L10 5 L0 10 Z" fill="{BAD}"/>
    </marker>
    <marker id="o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L10 5 L0 10 Z" fill="{ACCENT}"/>
    </marker>
  </defs>
  <style>
    .dash {{ stroke-dasharray: 6 6; animation: march 1s linear infinite; }}
    @keyframes march {{ to {{ stroke-dashoffset: -12; }} }}
    {REDUCED}
  </style>
  <rect width="{W}" height="{H}" rx="6" fill="{BG}"/>
  <g font-family="{FONT}">
    <text x="40" y="44" fill="{INK}" font-size="18" font-weight="600">One design, one loop</text>
    <text x="40" y="68" fill="{INK3}" font-size="15">thousands of times faster than a CFD run, and refused when the model would be guessing</text>
    {''.join(boxes)}
    {arrows}
    <path class="dash" d="{reject}" fill="none" stroke="{BAD}" stroke-width="1.6" marker-end="url(#r)"/>
    <text x="{660 + w / 2 + 12}" y="{y - 50}" fill="{BAD}" font-size="13.5">gated: the reason goes back to the agent, the design never counts</text>
    <path class="dash" d="{loop}" fill="none" stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#o)"/>
    <text x="{40 + w / 2 + 14}" y="{y + h + 62}" fill="{ACCENT}" font-size="13.5">the agent's next hypothesis: one change to the dials, with what it expects to happen</text>
    <circle r="6" fill="{ACCENT}">
      <animateMotion dur="5s" repeatCount="indefinite"
        path="M{40 + w / 2} {mid} H{970 + w / 2} V{y + h + 70} H{40 + w / 2} V{mid}"/>
    </circle>
  </g>
</svg>
"""


# ---------------------------------------------------------------- numbers

def load_results():
    metrics = json.loads((ROOT / "web/results/metrics.json").read_text(encoding="utf-8"))
    bench_path = ROOT / "web/results/benchmark.json"
    bench = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {"comparisons": []}
    return metrics, bench


def best_method(bench):
    """The comparison's best median result among real methods (the scripted dry run never counts)."""
    rows = []
    for comp in bench.get("comparisons", []):
        for m in comp.get("methods", []):
            if m.get("method") == "scripted" or m.get("final_pct", {}).get("median") is None:
                continue
            rows.append((m["final_pct"]["median"], m, comp))
    return min(rows, key=lambda r: r[0]) if rows else None


def stats(metrics, bench) -> str:
    h = metrics["headline"]
    agent = next(((m, c) for c in bench.get("comparisons", [])
                  for m in c.get("methods", []) if m.get("method", "").split(":")[0] in {"agent", "nemotron"}), None)
    coverage = h.get("coverage_90")
    cards = [
        (f"{h['r2_in_distribution']:.3f}", "Held-out R-squared", f"{metrics['in_distribution'].get('n_test', '?')} test designs; independent grader", None),
        (f"{h['mae_in_distribution']:.5f}", "Mean absolute Cd error", "A prediction, not CFD verification", None),
        (f"{coverage * 100:.1f}%" if coverage is not None else "pending", "Nominal 90% interval coverage",
         "Intervals under-cover held-out labels", None),
    ]
    if agent:
        m, comp = agent
        cards.append((f"{-m['final_pct']['median']:.2f}%", "Median predicted Cd reduction",
                      f"Nemotron: {comp['budget']} evaluations, {m['n_runs']} runs", None))
    W, H, cw = 1280, 210, 296
    out = []
    for i, (big, label, note, bar) in enumerate(cards):
        x = 16 + i * (cw + 16)
        bar_svg = ""
        if bar is not None:
            bar_svg = (f'<rect x="24" y="168" width="{cw - 48}" height="3" fill="{LINE}"/>'
                       f'<rect class="fill" x="24" y="168" width="{(cw - 48) * bar:.1f}" height="3" fill="{ACCENT}"'
                       f' style="animation-delay:{0.2 * i:.1f}s"/>')
        out.append(f"""
    <g transform="translate({x} 16)">
      <rect width="{cw}" height="{H - 32}" rx="4" fill="{PANEL}" stroke="{LINE}"/>
      <text class="rise" x="24" y="72" fill="{INK}" font-family="{MONO}" font-size="44" font-weight="500" style="animation-delay:{0.15 * i:.2f}s">{big}</text>
      <text x="24" y="108" fill="{INK2}" font-size="17">{label}</text>
      <text x="24" y="136" fill="{INK3}" font-size="13.5">{note}</text>
      {bar_svg}
    </g>""")
    label = metrics.get("label")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + (24 if label else 0)}" width="{W}" height="{H + (24 if label else 0)}" role="img"
     aria-label="Headline results: accuracy, speed, out-of-distribution detection and optimisation gain">
  <style>
    .fill {{ transform-box: fill-box; transform-origin: left; animation: grow 1.4s cubic-bezier(.2,.8,.2,1) both; }}
    .rise {{ animation: rise 0.9s cubic-bezier(.2,.8,.2,1) both; }}
    @keyframes grow {{ from {{ transform: scaleX(0); }} }}
    @keyframes rise {{ from {{ opacity: 0; transform: translateY(10px); }} }}
    {REDUCED}
  </style>
  <rect width="{W}" height="{H + (24 if label else 0)}" rx="6" fill="{BG}"/>
  <g font-family="{FONT}">{''.join(out)}
    {f'<text x="20" y="{H + 12}" fill="{INK3}" font-size="13">{label}</text>' if label else ''}
  </g>
</svg>
"""


def results_block(metrics, bench) -> str:
    h = metrics["headline"]
    ind, ood = metrics["in_distribution"], metrics.get("out_of_distribution") or {}
    rows = [
        "| Metric | Surrogate | Reference |",
        "|---|---:|---:|",
        f"| Held-out R-squared | **{h['r2_in_distribution']:.4f}** | Linear {h['r2_linear_reference']:.4f} |",
        f"| Mean absolute Cd error | **{h['mae_in_distribution']:.5f}** | |",
        f"| Test designs | **{ind.get('n_test', ind.get('overall', {}).get('n', '?'))}** | |",
    ]
    if h.get("r2_held_out_category") is not None:
        lin = ood.get("references", {}).get("linear", {}).get("overall", {}).get("r2")
        rows.append(f"| R-squared with {h['held_out_category']} held out | **{h['r2_held_out_category']:.3f}** | "
                    f"{'' if lin is None else f'Linear {lin:.3f}'} |")
    lines = ["\n".join(rows), ""]
    per = ind.get("per_category") or {}
    if per:
        lines.append("By body style, held-out R-squared: " + ", ".join(
            f"{k} {v['r2']:.3f}" for k, v in sorted(per.items()) if isinstance(v, dict) and "r2" in v) + ".")
        lines.append("")
    coverage = h.get("coverage_90")
    if coverage is not None:
        lines.append(f"- **Intervals:** nominal 90% coverage is {coverage * 100:.2f}% on held-out labels; error bars under-cover.")
    false_alarm = h.get("ood_false_alarm_rate")
    corners = metrics.get("ood_detection", {}).get("checkpoint_detector", {}).get("corners_flagged")
    if false_alarm is not None:
        lines.append(f"- **Domain screening:** distance false alarms {false_alarm * 100:.2f}% on ordinary test designs" +
                     (f"; {corners * 100:.0f}% of sampled jointly extreme corners flagged." if corners is not None else "."))
    if h.get("r2_held_out_category") is None:
        lines.append("- **Category generalisation:** no independently graded release category checkpoint is included.")
    else:
        lines.append("- **Category generalisation:** the separate evaluation-only checkpoint holds out an entire body style. "
                     "Style-indicator novelty helps the detector; it does not demonstrate aerodynamic novelty detection.")
    f = metrics.get("field") or {}
    if f.get("available"):
        overall = f['overall']
        lines.append(f"- **Surface pressure:** Cp R-squared {overall['r2']:.3f}, MAE {overall['mae']:.5f} "
                     f"on {len(f['test_runs'])} held-out DrivAerML cars ({f['n_points']:,} stored query cells). "
                     "Generated viewer geometry remains unvalidated.")
        near = f.get('references', {}).get('nearest_training_car')
        if near:
            lines.append(f"- **Field reference:** pooled nearest-training-point R-squared {near['r2']:.3f}, "
                         f"MAE {near['mae']:.5f}" + ("; this reference beats the field network." if near['r2'] > overall['r2'] else ".") +
                         " It averages eight neighbours from pooled training-car points in position/normal space.")
        if overall.get('coverage_1sigma') is not None:
            coverage1 = overall['coverage_1sigma']
            lines.append(f"- **Field intervals:** 1-sigma coverage {coverage1 * 100:.2f}% versus nominal 68.3%; "
                         + ("over-coverage" if coverage1 > .683 else "under-coverage") + " on stored CFD arrays, not generated geometry.")
        integral = f.get('integrated_cd')
        if integral:
            lines.append(f"- **Pressure-only drag:** per-car MAE {integral['mae']:.5f}, mean absolute relative error "
                         f"{integral['mean_relative_error'] * 100:.2f}% over stored query cells. "
                         "Viscous drag is excluded; this is not full native CFD surface quadrature.")
    else:
        lines.append("- **Surface pressure:** optional experimental inference is available with its checkpoint; full release test-set grading is unavailable locally.")
    scaling = metrics.get('data_scaling', {}).get('model') or []
    if scaling:
        first, last = scaling[0], scaling[-1]
        lines.append(f"- **Data scaling:** {len(scaling)} separately trained subset checkpoints, "
                     f"{first['n_train']:,}–{last['n_train']:,} training designs, R-squared "
                     f"{first['r2']:.3f}–{last['r2']:.3f} on a fixed test split. "
                     "The scaling checkpoints do not replace the serving checkpoint.")
    label = metrics.get("label")
    if label:
        lines.extend(["", f"> {label}. Generated by `src/eval/harness.py`; artifact identities are recorded in the metric file."])
    return "\n".join(lines)


def bench_block(bench) -> str:
    comps = bench.get("comparisons") or []
    if not comps:
        return "_No benchmark yet: `python -m src.eval.benchmark`._"
    comp = comps[0]
    rows = ["| Method | Runs | Median predicted Cd change | Best run | Best found at (median) | Gated (median) |",
            "|---|---:|---:|---:|---:|---:|"]
    pct = lambda v: f"{v:+.2f}%".replace("-", "−")  # noqa: E731
    live_agent = False
    for m in comp["methods"]:
        if m["method"] == "scripted":
            continue
        live_agent |= m["method"].split(":")[0] in {"agent", "nemotron"}
        fp = m["final_pct"]
        found, gated = m.get("found_at_median"), m.get("gated_median")
        rows.append(f"| {m['label']} | {m['n_runs']} | {pct(fp['median'])} | {pct(fp['best'])} | "
                    f"{'' if found is None else f'evaluation {found:.0f}'} | {'' if gated is None else f'{gated:.0f}'} |")
    note = (f"{comp['body_style'].capitalize()}, {comp['budget']} evaluations each, the same start, the same surrogate "
            "and the same gate. Lower is better. These historical experiments measure predicted Cd, not CFD-validated improvements. "
            "Nemotron and CMA-ES medians are comparable in this small experiment; equal calls do not mean equal time or cost.")
    if not live_agent:
        note += " The live Nemotron runs join this table once they are recorded."
    return "\n".join(rows) + "\n\n" + note


def fill(readme: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:start -->)(.*?)(<!-- {name}:end -->)", re.S)
    if not pattern.search(readme):
        raise SystemExit(f"README.md has no <!-- {name}:start --> block")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", readme)


def main() -> int:
    metrics, bench = load_results()
    (OUT / "hero.svg").write_text("\n".join(line.rstrip() for line in hero().splitlines()) + "\n", encoding="utf-8")
    (OUT / "pipeline.svg").write_text("\n".join(line.rstrip() for line in pipeline().splitlines()) + "\n", encoding="utf-8")
    (OUT / "stats.svg").write_text("\n".join(line.rstrip() for line in stats(metrics, bench).splitlines()) + "\n", encoding="utf-8")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if "<!-- results:start -->" in readme:
        readme = fill(readme, "results", results_block(metrics, bench))
        readme = fill(readme, "bench", bench_block(bench))
        readme_path.write_text(readme, encoding="utf-8")
    print("wrote docs/readme/hero.svg, pipeline.svg, stats.svg" + (" and README.md numbers" if "results:start" in readme else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
