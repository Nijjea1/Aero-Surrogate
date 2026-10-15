// What a drag and lift prediction means on the road.  OWNER: Paramveer.
//
// Standard aerodynamic loads at sea-level air density: dynamic pressure
// q = 1/2 rho v^2, force = q * coefficient * frontal area. Nothing here is
// modelled; every number follows from the prediction and the car's size.

export const AIR_DENSITY = 1.225;   // kg/m^3, sea level, 15 C

export function aeroLoads({ cd, cl, sigma = 0, area, speedKmh }) {
  const v = speedKmh / 3.6;                         // m/s
  const q = 0.5 * AIR_DENSITY * v * v;              // Pa
  const drag = q * cd * area;                       // N
  return {
    speed: v,
    dynamicPressure: q,
    cdA: cd * area,                                 // m^2
    drag,
    dragSigma: q * sigma * area,                    // N, one sigma of the Cd prediction
    lift: q * cl * area,                            // N, negative is downforce
    power: (drag * v) / 1000,                       // kW to push through the air
    energyPer100km: (drag * 100_000) / 3.6e6,       // kWh, drag alone
  };
}

export const GRAVITY = 9.81;   // m/s^2

// A standing-start sprint at constant engine power. Both cars in the race get
// the same power, mass, tyres and grip; only drag (Cd x frontal area) differs.
//
//   m dv/dt = min(P / v, mu m g) - 1/2 rho CdA v^2 - Crr m g
//
// Traction is grip-limited until power takes over. Fixed-step integration,
// sampled every 0.05 s; the finish is interpolated inside the last step.
export function standingStart({
  cdA, power = 150e3, mass = 1600, crr = 0.010, mu = 1.0, distance = 1000, dt = 0.002,
}) {
  let t = 0, x = 0, v = 0, sinceSample = 0;
  const samples = [{ t: 0, x: 0, v: 0 }];
  const grip = mu * mass * GRAVITY;
  const rolling = crr * mass * GRAVITY;
  while (t < 300) {
    const traction = v > 0 ? Math.min(power / v, grip) : grip;
    const a = (traction - 0.5 * AIR_DENSITY * cdA * v * v - rolling) / mass;
    const vNext = Math.max(0, v + a * dt);
    const xNext = x + 0.5 * (v + vNext) * dt;
    if (xNext >= distance) {
      const f = (distance - x) / (xNext - x);
      t += f * dt;
      v += f * (vNext - v);
      x = distance;
      samples.push({ t, x, v });
      return { time: t, trapSpeed: v, samples, finished: true };
    }
    t += dt; x = xNext; v = vNext; sinceSample += dt;
    if (sinceSample >= 0.05 - 1e-9) { samples.push({ t, x, v }); sinceSample = 0; }
  }
  return { time: t, trapSpeed: v, samples, finished: false };
}

// Where a car is at time t, from its samples; past the finish it coasts on at
// its trap speed so the gap keeps reading true.
export function positionAt(samples, t) {
  const last = samples[samples.length - 1];
  if (t >= last.t) return { x: last.x + last.v * (t - last.t), v: last.v };
  let lo = 0, hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) lo = mid; else hi = mid;
  }
  const a = samples[lo], b = samples[hi];
  const k = (t - a.t) / (b.t - a.t || 1);
  return { x: a.x + k * (b.x - a.x), v: a.v + k * (b.v - a.v) };
}
