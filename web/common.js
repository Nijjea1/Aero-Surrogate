// Shared by the designer (app.js) and the run view (runs.js).  OWNER: Paramveer.
//
// Point either page at another service with ?api=http://host:port

export const API = (new URLSearchParams(location.search).get("api") || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ServiceError extends Error {}

export async function api(path, body) {
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

export function showTooltip(event, html) {
  const tip = document.getElementById("tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(event.clientX + pad, window.innerWidth - rect.width - 8)}px`;
  tip.style.top = `${event.clientY + pad}px`;
}

export function hideTooltip() { document.getElementById("tooltip").hidden = true; }
