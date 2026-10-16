// Stats pinned on the car.  OWNER: Paramveer.
//
//   const pins = createAnnotations(viewer, host, { describe })
//   pins.setValues({ region: { value, source } })   // what each region pin shows
//   pins.setDrag({ newtons, sigma } | null)          // the drag arrow behind the car
//   pins.setVisible(bool)
//
// One pin per pressure region, at the region's mean surface point (car.anchors)
// projected to the screen after every render. A pin whose panel faces away
// from the camera dims rather than vanishing, so the set stays readable while
// the camera sweeps. Hover or focus a pin for describe(region)'s card.

import * as THREE from "three";
import { escapeHtml, fmt, hideTooltip, showTooltipAt, signed } from "./common.js";
import { cpColor } from "./viewer.js";

export const REGION_NAMES = {
  front_fascia: "Front fascia",
  hood: "Hood",
  windscreen: "Windscreen",
  roof: "Roof",
  rear_window: "Rear window",
  rear_fascia: "Rear fascia",
  underbody: "Underbody",
  wheel_wells: "Wheels",
};

export function createAnnotations(viewer, host, { describe }) {
  let car = null;
  let pins = {};
  let values = {};
  let visible = true;
  const at = new THREE.Vector3();
  const toCamera = new THREE.Vector3();
  const dragAt = new THREE.Vector3();

  // ---- the drag arrow: a shaft and head behind the tail, pointing backwards
  // (cars face -X), as long as the drag force is large.
  const arrow = new THREE.Group();
  const glow = new THREE.MeshStandardMaterial({ color: "#ff6a2b", emissive: "#ff5f1f", emissiveIntensity: 0.9, roughness: 0.4 });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1, 16), glow);
  shaft.rotation.z = Math.PI / 2;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.26, 24), glow);
  head.rotation.z = -Math.PI / 2;
  arrow.add(shaft, head);
  arrow.visible = false;
  viewer.scene.add(arrow);
  const dragPin = makePin("force");
  dragPin.addEventListener("pointerenter", () => openCard(dragPin, "drag"));
  dragPin.addEventListener("focus", () => openCard(dragPin, "drag"));
  dragPin.addEventListener("pointerleave", hideTooltip);
  dragPin.addEventListener("blur", hideTooltip);
  let drag = null;

  function makePin(kind) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = `pin${kind ? ` ${kind}` : ""}`;
    host.append(el);
    return el;
  }

  function openCard(el, key) {
    const r = el.getBoundingClientRect();
    showTooltipAt(r.left + r.width / 2, r.bottom - 6, describe(key));
  }

  function rebuild() {
    for (const el of Object.values(pins)) el.remove();
    pins = {};
    if (!car) return;
    for (const name of Object.keys(car.anchors)) {
      const el = makePin("");
      el.dataset.region = name;
      el.addEventListener("pointerenter", () => openCard(el, name));
      el.addEventListener("focus", () => openCard(el, name));
      el.addEventListener("pointerleave", hideTooltip);
      el.addEventListener("blur", hideTooltip);
      pins[name] = el;
    }
    paint();
  }

  function paint() {
    for (const [name, el] of Object.entries(pins)) {
      const v = values[name];
      const has = v && typeof v.value === "number";
      el.style.setProperty("--pin", has ? cpColor(v.value) : "#6f7075");
      el.innerHTML = `${escapeHtml(REGION_NAMES[name] || name)} <b>${has ? signed(v.value, 2) : "–"}</b>`;
      el.setAttribute("aria-label", `${REGION_NAMES[name] || name}: Cp ${has ? signed(v.value, 2) : "not available"}`);
    }
    if (drag) {
      dragPin.innerHTML = `Drag <b>${fmt(drag.newtons, 0)} N</b>${drag.sigma ? ` <span>± ${fmt(drag.sigma, 0)}</span>` : ""}`;
      dragPin.setAttribute("aria-label", `Drag force ${fmt(drag.newtons, 0)} newtons`);
    }
  }

  function place(camera) {
    if (viewer.car !== car) { car = viewer.car; rebuild(); }
    const show = visible && car;
    host.dataset.off = String(!show);
    if (!show) { arrow.visible = false; dragPin.style.display = "none"; return; }
    const box = viewer.renderer.domElement.getBoundingClientRect();
    const hostBox = host.getBoundingClientRect();
    const offset = car.group.position;
    const put = (el, position, normal) => {
      at.copy(position).add(offset);
      toCamera.copy(camera.position).sub(at);
      at.project(camera);
      const onScreen = at.z < 1 && Math.abs(at.x) < 1.05 && Math.abs(at.y) < 1.05;
      el.style.display = onScreen ? "" : "none";
      if (!onScreen) return;
      el.style.left = `${box.left - hostBox.left + ((at.x + 1) / 2) * box.width}px`;
      el.style.top = `${box.top - hostBox.top + ((1 - at.y) / 2) * box.height}px`;
      el.dataset.away = String(normal ? normal.dot(toCamera) < 0 : false);
    };
    for (const [name, el] of Object.entries(pins)) put(el, car.anchors[name].position, car.anchors[name].normal);

    if (drag) {
      const length = THREE.MathUtils.clamp(0.35 + drag.newtons / 450, 0.45, 2.6);
      const tail = car.frame.length / 2 + 0.2;
      arrow.visible = true;
      arrow.position.set(offset.x + tail + length / 2, 0.72, offset.z);
      shaft.scale.set(1, length - 0.24, 1);
      shaft.position.x = -0.12;
      head.position.x = length / 2 - 0.13;
      put(dragPin, dragAt.set(tail + length + 0.25, 0.95, 0), null);
    } else {
      arrow.visible = false;
      dragPin.style.display = "none";
    }
    declutter([...Object.values(pins), dragPin]);
  }

  // Labels that would overlap are stacked: each later one moves down (or up,
  // if there is more room above) just clear of those already placed.
  function declutter(els) {
    const placed = [];
    const W = host.clientWidth;
    const shown = els.filter((el) => el.style.display !== "none" && el.dataset.away !== "true");
    // Keep every label inside the stage: on a narrow screen a pin near the
    // edge would otherwise be cut off.
    for (const el of shown) {
      const half = el.offsetWidth / 2 + 6;
      const x = parseFloat(el.style.left);
      if (W > 2 * half) el.style.left = `${Math.min(W - half, Math.max(half, x))}px`;
    }
    shown.sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
    for (const el of shown) {
      const w = el.offsetWidth, h = el.offsetHeight;
      const x = parseFloat(el.style.left), base = parseFloat(el.style.top);
      let y = base;
      for (let tries = 0; tries < 8; tries++) {
        const hit = placed.find((p) => Math.abs(p.x - x) < (p.w + w) / 2 + 4 && Math.abs(p.y - y) < (p.h + h) / 2 + 3);
        if (!hit) break;
        y = hit.y + (hit.h + h) / 2 + 3;
      }
      if (y !== base) el.style.top = `${y}px`;
      placed.push({ x, y, w, h });
    }
  }
  viewer.onRender(place);

  return {
    setValues(next) { values = next || {}; paint(); viewer.render(); },
    setDrag(next) { drag = next; paint(); viewer.render(); },
    setVisible(on) { visible = on; viewer.render(); },
  };
}
