// What a drag and lift prediction means on the road.  OWNER: Paramveer.
//
// Standard aerodynamic loads at sea-level air density: dynamic pressure
// q = 1/2 rho v^2, force = q * coefficient * frontal area. Nothing here is
// modelled; every number follows from the prediction and the car's size.

export const AIR_DENSITY = 1.225;   // kg/m^3, sea level, 15 C
export const AIR_VISCOSITY = 1.81e-5;   // Pa s, for the Reynolds number

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

export function reynolds(speedKmh, length) {
  return (AIR_DENSITY * (speedKmh / 3.6) * length) / AIR_VISCOSITY;
}

export const GRAVITY = 9.81;   // m/s^2

// Both cars in a race get the same power, mass, tyres and grip; only drag
// (Cd x frontal area) differs.
//
//   m dv/dt = traction - 1/2 rho CdA v^2 - Crr m g,   traction = min(P / v, mu m g)
//
// Traction is grip-limited until power takes over; it is zero in a coast-down.
// Fixed-step integration, sampled every 0.05 s; an end inside a step is
// interpolated. Every mode returns { samples, end, vmax }: end is { t, x, v }
// at the finish line, at top speed, or where the car stops.
const SAMPLE = 0.05;

export function simulateRace(mode, {
  cdA, power = 150e3, mass = 1600, crr = 0.010, mu = 1.0, distance = 1000, v0 = 130 / 3.6, dt = 0.004,
}) {
  const grip = mu * mass * GRAVITY;
  const rolling = crr * mass * GRAVITY;
  const vmax = topSpeed({ cdA, power, mass, crr });
  const coast = mode === "coast";
  let t = 0, x = 0, v = coast ? v0 : 0, sinceSample = 0, end = null;
  const samples = [{ t: 0, x: 0, v }];
  // A ceiling no real race reaches: 100 km at walking pace would still finish.
  const limit = 6 * 3600;
  while (t < limit) {
    const traction = coast ? 0 : v > 0 ? Math.min(power / v, grip) : grip;
    const a = (traction - 0.5 * AIR_DENSITY * cdA * v * v - rolling) / mass;
    const vNext = v + a * dt;
    if (coast && vNext <= 0) {
      const f = v / (v - vNext);
      t += f * dt;
      x += 0.5 * v * f * dt;
      v = 0;
      end = { t, x, v };
      samples.push({ t, x, v });
      break;
    }
    const xNext = x + 0.5 * (v + vNext) * dt;
    if (mode === "distance" && xNext >= distance) {
      const f = (distance - x) / (xNext - x);
      t += f * dt;
      v += f * (vNext - v);
      x = distance;
      end = { t, x, v };
      samples.push({ t, x, v });
      break;
    }
    t += dt; x = xNext; v = vNext; sinceSample += dt;
    if (sinceSample >= SAMPLE - 1e-9) { samples.push({ t, x, v }); sinceSample = 0; }
    if (mode === "top" && v >= vmax - 0.3 / 3.6) { end = { t, x, v }; samples.push({ t, x, v }); break; }
  }
  if (!end) end = { t, x, v };
  return { samples, end, vmax, finished: t < limit };
}

// A standing-start sprint at constant engine power over a fixed distance.
export function standingStart({ distance = 1000, ...rest }) {
  const run = simulateRace("distance", { distance, dt: 0.002, ...rest });
  return { time: run.end.t, trapSpeed: run.end.v, samples: run.samples, finished: run.finished };
}

// The speed where engine power exactly meets drag plus rolling resistance:
// P = (1/2 rho CdA v^2 + Crr m g) v, solved by bisection.
export function topSpeed({ cdA, power = 150e3, mass = 1600, crr = 0.010 }) {
  const rolling = crr * mass * GRAVITY;
  let lo = 0, hi = 300;
  for (let i = 0; i < 80; i++) {
    const v = (lo + hi) / 2;
    if ((0.5 * AIR_DENSITY * cdA * v * v + rolling) * v < power) lo = v; else hi = v;
  }
  return lo;
}

// Closed form of a coast-down from v0 to rest, for checking the integration:
// x = m / (rho CdA) * ln(1 + rho CdA v0^2 / (2 Crr m g)).
export function coastDistance({ cdA, mass = 1600, crr = 0.010, v0 }) {
  const k = AIR_DENSITY * cdA;
  return (mass / k) * Math.log(1 + (k * v0 * v0) / (2 * crr * mass * GRAVITY));
}

// Where a car is at time t, from its samples; past the last sample it carries
// on at its last speed (zero after a coast-down), so the gap keeps reading true.
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

// When a car reached distance x: the inverse of positionAt, for intervals.
export function timeAt(samples, x) {
  if (x <= samples[0].x) return samples[0].t;
  const last = samples[samples.length - 1];
  if (x >= last.x) return last.v > 0 ? last.t + (x - last.x) / last.v : last.t;
  let lo = 0, hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].x <= x) lo = mid; else hi = mid;
  }
  const a = samples[lo], b = samples[hi];
  const k = (x - a.x) / (b.x - a.x || 1);
  return a.t + k * (b.t - a.t);
}
